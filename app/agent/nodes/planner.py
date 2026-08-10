"""
app/agent/nodes/planner.py
───────────────────────────
Planner Node — turns the conversation into structured trip entities and decides
whether the agent has enough to answer.

Split of responsibility:
  • the LLM EXTRACTS — it reads the conversation and reports what the user said
  • this node DECIDES — it merges, classifies, and applies the slot policy

The LLM is never asked whether to ask a question or what to ask. It was, once, and
it produced questions in the wrong language, questions for details the user had
already given, and questions for details the request never needed. The policy now
lives in code (app/agent/slots.py) where it is deterministic and testable, and the
model is left to do the one thing it is good at: reading the conversation.

Three guarantees this node makes, each covering a reported failure:

  1. A message is only ever read as an answer to a question we actually asked, and
     only when it has the right shape (`slots.read_answer`). Nothing is inferred
     from an unrelated message, so "plan a trip" can never become a destination.
  2. The agent asks for any one detail at most `max_clarification_attempts` times,
     and stops asking altogether the moment the user says to get on with it. It
     cannot loop.
  3. Whatever it proceeds without is returned in `assumed_fields`, so the writer
     states the assumption instead of quietly inventing a fact.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent import slots
from app.agent.language import detect_reply_language
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import get_logger
from app.models.itinerary import BudgetRange, GroupType
from app.models.session import ExtractedEntities
from app.services import geo
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

LOCAL = "local"
OUTSTATION = "outstation"

# Bare greetings, recognised so "hi" costs no LLM call. Matched on the whole
# message after punctuation is stripped — never on a substring, so "hi, plan
# Manali" still goes to the extractor.
_GREETINGS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "yo", "hola", "namaste",
    "namaskar", "namaste ji", "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "start", "hey there",
}

# Backstop for "stop asking and just do it" when the extraction LLM is unavailable.
# The model is the primary detector (`user_wants_to_proceed_now`), which is what
# makes this work in languages and phrasings no list can enumerate.
_PROCEED_PHRASES = (
    "just make", "just plan", "just give", "just show", "just do", "go ahead",
    "anything is fine", "anything works", "you decide", "you choose", "up to you",
    "whatever you", "dont know", "don't know", "no idea", "not sure", "skip",
    "bas bana", "bana do", "bana dijiye", "kuch bhi", "aap batao", "aap hi",
    "tum batao", "pata nahi", "nahi pata", "jo bhi", "chhodo", "chalega",
)


async def classify_visitor_type(
    origin: Optional[str],
    origin_in_himachal: Optional[bool] = None,
) -> Optional[str]:
    """Classify the traveller from their starting city.

    `origin_in_himachal` is the extraction model's geography call and wins when it is
    available, because it costs nothing — it came back with the extraction. Otherwise
    the city is geocoded and the provider's administrative state decides, which is the
    same source that prices the route, so a town is never known to one and unknown to
    the other.

    Returns None when there is no origin, or when neither source can place it, so the
    caller keeps whatever classification it already had rather than guessing.
    """
    if not origin:
        return None
    if origin_in_himachal is not None:
        return LOCAL if origin_in_himachal else OUTSTATION
    in_himachal = await geo.place_in_himachal(origin)
    if in_himachal is None:
        return None
    return LOCAL if in_himachal else OUTSTATION


class PlannerExtraction(BaseModel):
    """What the model reports about the conversation. Facts only — no flow control.

    Note there is no `needs_clarification`/`clarification_question` here any more:
    those were asked for, then discarded by the node, and the model spent tokens
    producing questions nobody read.
    """

    destination: Optional[str] = None
    origin: Optional[str] = None
    duration_days: Optional[int] = None
    travel_dates: Optional[str] = None
    group_type: Optional[GroupType] = None
    group_size: Optional[int] = None
    budget_range: Optional[BudgetRange] = None
    interests: list[str] = Field(default_factory=list)
    special_requirements: list[str] = Field(default_factory=list)
    return_journey_preference: Optional[str] = None
    home_city: Optional[str] = None
    origin_in_himachal: Optional[bool] = Field(
        default=None,
        description="True if `origin` is a town inside Himachal Pradesh, False if outside, null if unknown.",
    )
    user_wants_to_proceed_now: bool = Field(
        default=False,
        description="True if the user asked to stop being asked questions and just get the plan.",
    )
    confidence: float = 0.5


PLANNER_SYSTEM_PROMPT = """You are an information extractor for a Himachal Pradesh travel service.

You are NOT talking to the user. You read the conversation and report what the user
has actually said. Another system decides what to ask next — never write a question.

Extract these fields (null when the user has not stated them):
- destination: where in Himachal they want to go
- origin: the city they are starting from
- origin_in_himachal: true if `origin` is a town/city inside Himachal Pradesh, false if
  it is outside (Delhi, Chandigarh, Mumbai, Amritsar, Dehradun...), null if no origin yet.
  Use your own geography knowledge — this drives whether they are treated as a resident.
- duration_days: integer, ONLY if the user gave a number of days (or a length like "a week")
- travel_dates: their rough travel time, verbatim ("next week", "kal", "15 Jan")
- group_type: one of [solo, couple, family, friends, corporate]
- group_size: integer, ONLY if the user stated how many people
- budget_range: one of [budget, moderate, luxury]
- interests: things they said they want to do or see
- special_requirements: constraints they stated (accessibility, dietary, pets...)
- return_journey_preference: "booked" | "not_booked" | "own_arrangement" | null
- home_city: their home city, if it is inside Himachal Pradesh
- user_wants_to_proceed_now: true when the user asks you to stop asking and just produce
  the plan, says the choice is up to you, or says they do not know / it does not matter.
  Recognise this in ANY language or phrasing ("just make it", "bas bana do", "kuch bhi",
  "aap decide karo", "I don't mind", "skip that"). Otherwise false.
- confidence: 0-1

MERGING (the conversation is multi-turn):
- The latest user message is often a SHORT answer to the assistant's previous question.
  Read that question to understand it: "How many days?" + "5" means duration_days = 5;
  "Which city are you travelling from?" + "delhi" means origin = "Delhi".
- Previously extracted values are given to you. Repeat them in your output. Never replace
  a known value with null, and never overwrite one unless the user has just changed it.

NEVER INVENT:
- Every field the user has not stated is null. Not a plausible guess, not a common
  default. No days, no group size, no dates, no budget unless they said so.
- A greeting ("hi", "namaste") carries no information — everything is null.
- Words describing the request are not places: "plan a trip", "somewhere in the hills",
  and "I want a holiday" all mean destination = null.
- Interpret relative dates against TODAY'S DATE from the request, and never produce a
  date in the past. A rough answer like "next month" is a perfectly good travel_dates.
"""


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            return m.content.strip()
    return ""


def _looks_like_proceed_request(text: str) -> bool:
    """Backstop for the LLM's `user_wants_to_proceed_now` — see `_PROCEED_PHRASES`."""
    low = text.lower()
    return any(phrase in low for phrase in _PROCEED_PHRASES)


def _is_bare_greeting(text: str) -> bool:
    cleaned = text.strip().strip("!.,?").lower()
    return cleaned in _GREETINGS


def _has_trip_facts(entities: ExtractedEntities) -> bool:
    """True once the user has told us anything at all about the trip."""
    return any(
        slots.SLOTS[name].read(entities) for name in slots.SLOTS
    )


def _build_entities(merged: dict) -> ExtractedEntities:
    """Rebuild the entity bag from the merged dict, preserving conversation control.

    `last_asked_field` is deliberately NOT carried over — it is set by `_resolve`
    from this turn's decision, so a stale question can never be used to reinterpret a
    later message.
    """
    return ExtractedEntities(
        destination=merged.get("destination"),
        origin=merged.get("origin"),
        duration_days=merged.get("duration_days"),
        travel_dates=merged.get("travel_dates"),
        group_type=merged.get("group_type"),
        group_size=merged.get("group_size"),
        budget_range=merged.get("budget_range"),
        interests=merged.get("interests") or [],
        special_requirements=merged.get("special_requirements") or [],
        visitor_type=merged.get("visitor_type"),
        return_journey_preference=merged.get("return_journey_preference"),
        home_city=merged.get("home_city"),
        asked_counts=dict(merged.get("asked_counts") or {}),
        proceed_requested=bool(merged.get("proceed_requested")),
        no_progress_turns=int(merged.get("no_progress_turns") or 0),
    )


def _known_slots(entities: ExtractedEntities) -> set[str]:
    """Which slots currently hold a value — used to tell progress from repetition."""
    return {name for name, slot in slots.SLOTS.items() if slot.read(entities)}


def _resolve(
    state: AgentState,
    entities: ExtractedEntities,
    visitor_type: Optional[str],
    lang_key: str,
) -> dict:
    """Apply the slot policy and package the node's return value.

    Shared by the LLM path and the LLM-failure path so a Vertex outage changes what
    the agent KNOWS, never how it BEHAVES.
    """
    decision = slots.decide(
        entities,
        state.intent,
        visitor_type,
        lang_key,
        max_attempts=settings.max_clarification_attempts,
    )

    # Count the ask now, so both caps hold even when the user ignores the question.
    if decision.ask_field:
        entities.asked_counts[decision.ask_field] = (
            entities.asked_counts.get(decision.ask_field, 0) + 1
        )
    entities.last_asked_field = decision.ask_field

    return {
        "entities": entities,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.question,
        "visitor_type": visitor_type,
        "last_asked_field": decision.ask_field,
        "assumed_fields": list(decision.assumed),
        "node_trace": ["planner"],
    }


async def planner_node(state: AgentState) -> dict:
    """LangGraph Planner Node — extract, merge, classify, then decide."""
    log.info("planner_node_entered", session_id=state.session_id, intent=state.intent)

    lang_key, reply_language = detect_reply_language(state)
    latest_text = _latest_user_text(state)
    # The single detail our last question asked about. The reply is only ever read
    # as an answer to THIS field.
    previously_asked = state.entities.last_asked_field
    known_before = _known_slots(state.entities)
    existing_entities = state.entities.model_dump(exclude_none=True)

    # ── Greeting fast path ────────────────────────────────────────────────────
    # A bare "hi" with nothing known yet has nothing to extract. Skipping the LLM
    # here saves a call and removes its best opportunity to hallucinate a trip.
    if _is_bare_greeting(latest_text) and not _has_trip_facts(state.entities):
        log.info("greeting_shortcircuit", session_id=state.session_id)
        greeting_entities = state.entities.model_copy(deep=True)
        # A greeting in reply to a question is a turn that told us nothing — it counts
        # towards the patience limit exactly like any other non-answer.
        if previously_asked:
            greeting_entities.no_progress_turns = state.entities.no_progress_turns + 1
        return _resolve(state, greeting_entities, state.visitor_type, lang_key)

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.messages[-10:]
    )
    today = datetime.now(timezone.utc).strftime("%A, %d %B %Y")
    asked_context = (
        f"\nThe assistant's last question asked about: {', '.join(previously_asked)}. The "
        f"user's latest message is most likely the answer to it."
        if previously_asked else ""
    )

    prompt = f"""Today's date: {today}
Conversation language: {reply_language}
Visitor classification so far: {state.visitor_type or 'unknown'}{asked_context}

Conversation so far:
{conversation_text}

Already extracted (carry these forward; do not null them out):
{json.dumps(existing_entities, indent=2, default=str)}

Report what the user has stated. Use null for anything they have not."""

    try:
        llm = (
            get_vertex_ai_service()
            .get_flash_model(streaming=False)
            .with_structured_output(PlannerExtraction)
        )
        raw_result = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=PLANNER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]),
            timeout=settings.llm_call_timeout_seconds,
        )
        result_dict = (
            raw_result.model_dump()
            if isinstance(raw_result, PlannerExtraction)
            else dict(raw_result)
        )
        llm_ok = True

    except Exception as e:
        # Degrade, don't derail: keep what we know, read the reply in code, and let
        # the same policy decide. The user sees a slightly less clever turn, not an error.
        log.error(
            "planner_extraction_failed",
            error=str(e),
            error_type=type(e).__name__,
            session_id=state.session_id,
        )
        result_dict = {}
        llm_ok = False

    # ── Merge: new facts win, known facts are never nulled out ────────────────
    merged = {**existing_entities}
    control_keys = ("confidence", "user_wants_to_proceed_now", "origin_in_himachal")
    for key, val in result_dict.items():
        if key not in control_keys and val not in (None, []):
            merged[key] = val

    # ── Visitor classification (code, from the origin city) ───────────────────
    visitor_type = (
        await classify_visitor_type(merged.get("origin"), result_dict.get("origin_in_himachal"))
        or state.visitor_type
    )
    if visitor_type:
        merged["visitor_type"] = visitor_type
        if visitor_type == LOCAL and not merged.get("home_city"):
            merged["home_city"] = merged.get("origin")

    # ── Read the reply as an answer to the question we asked ──────────────────
    # Strictly scoped: only the fields that question covered, only if the reply has
    # the right shape. A safety net for a missed one-word answer, not an interpreter.
    recovered = slots.read_answer(previously_asked, latest_text, merged, visitor_type) if latest_text else None
    if recovered:
        log.info("short_answer_recovered", field=recovered, session_id=state.session_id)
        if recovered == "origin":
            # An origin recovered in code still decides the persona.
            visitor_type = await classify_visitor_type(merged.get("origin")) or visitor_type
            if visitor_type:
                merged["visitor_type"] = visitor_type
                if visitor_type == LOCAL and not merged.get("home_city"):
                    merged["home_city"] = merged.get("origin")

    # ── "Stop asking and just do it" ──────────────────────────────────────────
    if result_dict.get("user_wants_to_proceed_now") or (
        not llm_ok and _looks_like_proceed_request(latest_text)
    ):
        merged["proceed_requested"] = True
        log.info("proceed_requested", session_id=state.session_id)

    entities = _build_entities(merged)

    # ── Progress tracking ─────────────────────────────────────────────────────
    # "They replied but told us nothing new." Counting these is what lets the agent
    # notice it is getting nowhere and deliver instead of asking again.
    #
    # The condition is deliberately NOT "…and we asked a question last turn". It was,
    # and that made giving up last exactly one turn: the agent stopped asking, which
    # cleared `last_asked_field`, which reset the streak, which let it start asking
    # again on the very next message. The user got interrogated, then a plan, then
    # interrogated again. Only real progress — a slot that gained a value — resets it.
    if _known_slots(entities) - known_before:
        entities.no_progress_turns = 0
    else:
        entities.no_progress_turns = state.entities.no_progress_turns + 1

    result = _resolve(state, entities, visitor_type, lang_key)

    log.info(
        "entities_extracted",
        destination=entities.destination,
        origin=entities.origin,
        duration=entities.duration_days,
        group_size=entities.group_size,
        travel_dates=entities.travel_dates,
        visitor_type=visitor_type,
        intent=state.intent,
        needs_clarification=result["needs_clarification"],
        asking=result["last_asked_field"],
        previously_asked=previously_asked,
        assumed=result["assumed_fields"],
        proceed_requested=entities.proceed_requested,
        no_progress_turns=entities.no_progress_turns,
        llm_ok=llm_ok,
        session_id=state.session_id,
    )
    if not llm_ok:
        result["error"] = "planner extraction unavailable"
    return result


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "PlannerExtraction",
    "classify_visitor_type",
    "planner_node",
]
