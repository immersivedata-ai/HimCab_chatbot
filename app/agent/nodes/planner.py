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
from datetime import UTC, datetime
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent import sanitize, slots
from app.agent.language import detect_reply_language
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import SlotName
from app.core.logging import get_logger
from app.domain.conversation import ExtractedEntities
from app.domain.itinerary import BudgetRange, GroupType
from app.integrations.himcab import available_packages, offered_hours
from app.services import geo
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

# Aliases onto app/core/enums.Persona, via slots — one set of objects for the whole
# system, so a persona compare here cannot disagree with one in chat_session.
LOCAL = slots.LOCAL
OUTSTATION = slots.OUTSTATION
#: Hiring a car by the hour. Not a kind of person, so `classify_visitor_type` never
#: reclassifies it — see the note there.
RENTAL = slots.RENTAL

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
    *,
    home_city: Optional[str] = None,
    current: Optional[str] = None,
) -> Optional[str]:
    """Decide whether this rider is a Himachal resident or a visitor.

    The evidence is asymmetric, and treating it symmetrically was a bug:

      • an origin OUTSIDE Himachal is strong evidence of a visitor. Nobody drives in
        from Delhi and needs to be told about their own state.
      • an origin INSIDE Himachal proves nothing on its own. It fits a resident, and
        it equally fits a tourist who is already here — someone in Manali asking to
        go on to Kasol. Reading it as residence flipped English-speaking visitors to
        the local persona the moment they named their starting city, which switched
        them to Hindi and replaced their itinerary with a cab quote.

    So residence has to be stated, not inferred: `home_city` is where they LIVE (the
    extractor only fills it for a Himachal town), while `origin` is merely where this
    journey starts. An in-state origin leaves `current` untouched.

    Returns None when there is no evidence either way, so the caller keeps whatever
    classification it already had rather than guessing.
    """
    # A rental is a SERVICE the rider chose, not a guess about who they are, so no
    # amount of geography cancels it. Rentals are picked up from anywhere — Delhi,
    # Chandigarh, Ambala — and reading that pickup as "a visitor" turned the hire into
    # a holiday: the rider was asked where they wanted to go and for how many days,
    # and got an itinerary instead of a car. They leave this flow by opening a
    # different tab, never by naming a city.
    if current == RENTAL:
        return RENTAL

    # Residence beats everything, including an origin somewhere else — a Shimla
    # resident flying out of Delhi is still a local.
    if home_city and await geo.place_in_himachal(home_city):
        return LOCAL

    if not origin:
        return None

    in_himachal = (
        origin_in_himachal
        if origin_in_himachal is not None
        else await geo.place_in_himachal(origin)
    )
    if in_himachal is None:
        return None
    if not in_himachal:
        return OUTSTATION

    # In-state origin: consistent with a resident AND with a visitor already here.
    # Not enough on its own — keep what we had (the GPS fix or the front end they
    # opened), and let them say "I live in Manali" if they want to be treated as one.
    return current


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
    rental_hours: Optional[int] = Field(
        default=None,
        description="Hours the rider wants to hire a car for, if they said (rental flow).",
    )
    pickup_time: Optional[str] = Field(
        default=None,
        description=(
            'The clock time the cab should arrive, as "HH:mm" on the 24-hour clock. '
            'Only from an actual time the user gave. Null for a vague "morning".'
        ),
    )
    cab_category: Optional[str] = Field(
        default=None,
        description=(
            "The cab tier the rider chose, lower-cased and nothing else "
            '("luxury", "economy", "mini bus"). Only when they picked one.'
        ),
    )
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
  Use your own geography knowledge. This says where the JOURNEY starts, not who they are.
- duration_days: integer, ONLY if the user gave a number of days (or a length like "a week")
- travel_dates: their rough travel time, verbatim ("next week", "kal", "15 Jan")
- group_type: one of [solo, couple, family, friends, corporate]
- group_size: integer, ONLY if the user stated how many people
- budget_range: one of [budget, moderate, luxury]
- interests: things they said they want to do or see
- special_requirements: constraints they stated (accessibility, dietary, pets...)
- return_journey_preference: "booked" | "not_booked" | "own_arrangement" | null
- rental_hours: integer — hours they want to HIRE a car for, only in a rental
  conversation and only if they gave a number ("8 hours", "aadha din" = 4, "full day"
  = 8). This is a duration, never a distance and never a trip length in days.
- cab_category: the cab tier they picked from a list you offered them ("standard",
  "economy", "luxury", "mini bus", "premium"), lower-cased. Null unless they chose one.
- pickup_time: the clock time the cab should ARRIVE, as "HH:mm" on the 24-hour clock.
  "6 am" -> "06:00", "shaam 5 baje" -> "17:00", "18:30" -> "18:30", "subah 6" -> "06:00".
  A DAY is not a time: "kal" and "tomorrow" are travel_dates, not this. Neither is a
  vague part of the day — "morning", "subah" and "evening" with no hour are null here,
  because the fare is calculated from this figure and a guessed hour changes the price.
  A phrase can fill both: "kal subah 6 baje" is travel_dates="kal" and pickup_time="06:00".
- home_city: where they LIVE, if that is inside Himachal Pradesh, and ONLY when they
  have actually said so ("I'm from Kullu", "main Shimla mein rehta hoon", "we live in
  Mandi"). This is the one field that marks somebody as a resident rather than a
  visitor, so do NOT fill it from where their trip happens to start: a tourist saying
  "I'm in Manali right now" is origin=Manali with home_city=null.
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

A RENTAL CONVERSATION (the request says so above) IS NOT A TRIP:
- The rider is hiring a car and a driver BY THE HOUR. They have no destination, no trip
  length in days and no itinerary, and they are not expected to have one.
- destination and duration_days are ALWAYS null here, however the message is phrased.
  "A car for 8 hours around Manali" is rental_hours=8 with destination null — Manali is
  where the car will be driven about, not a trip they asked you to plan.
- A place they name is the PICKUP (`origin`) only when they said so: "from Shimla",
  "pick me up in Kullu", "Mandi se". A place mentioned any other way fills nothing.
- What matters is exactly what their booking screen asks for: the duration, the pickup
  address, the date, the clock time and which cab. Nothing else.

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


#: Trip-shaped fields that CANNOT exist in a rental conversation. The extraction model
#: still reports them — "a car for 8 hours around Manali" reliably comes back with
#: destination="Manali" — and every one of them then leaks somewhere it does not belong:
#: into the writer's TRIP block, into the itinerary shortcut in `route_after_router`,
#: into the profile's past-trip list in `persist`. Hiring a car by the hour means the
#: rider has not chosen a destination and need not, so these are dropped at the door
#: rather than guarded against one consumer at a time.
_NOT_IN_A_RENTAL = ("destination", "duration_days", "return_journey_preference")


def _strip_trip_fields(merged: dict, session_id: str) -> None:
    """Remove trip-only facts from a rental conversation's entities (mutates)."""
    dropped = [f for f in _NOT_IN_A_RENTAL if merged.get(f) is not None]
    for field in dropped:
        merged.pop(field, None)
    if dropped:
        log.info("rental_trip_fields_dropped", fields=dropped, session_id=session_id)


async def _reject_unsold_duration(
    merged: dict, rider_jwt: Optional[str], session_id: str
) -> Optional[int]:
    """Clear a rental duration Him Cab does not sell, returning what it was.

    Asked and answered in the SAME turn, which is the whole point. The duration used to
    be validated only where the fare is requested — after the pickup, the date, the time
    and the headcount had all been collected — so a rider who opened with "6 hours" was
    put through four more questions before being told the first answer had never been
    usable. Clearing it here puts the duration question back in flight immediately, and
    `clarify_node` names the packages in the same breath.

    The offered list is TTL-cached in the rental service, so this costs a network call
    once an hour rather than once a turn. An empty list means we cannot tell, and
    nothing is rejected on a guess — the fare gate is still there as the backstop.
    """
    hours = merged.get("rental_hours")
    if not hours:
        return None
    offered = await available_packages(rider_jwt)
    if not offered or hours in offered_hours(offered):
        return None

    merged["rental_hours"] = None
    log.info(
        "rental_duration_rejected_early",
        requested_hours=hours,
        offered_hours=offered_hours(offered),
        session_id=session_id,
    )
    return hours


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
        rental_hours=merged.get("rental_hours"),
        pickup_time=merged.get("pickup_time"),
        cab_category=merged.get("cab_category"),
        asked_counts=dict(merged.get("asked_counts") or {}),
        proceed_requested=bool(merged.get("proceed_requested")),
        no_progress_turns=int(merged.get("no_progress_turns") or 0),
    )


def _known_slots(entities: ExtractedEntities) -> set[str]:
    """Which slots currently hold a value — used to tell progress from repetition."""
    return {name for name, slot in slots.SLOTS.items() if slot.read(entities)}


def _same_value(a: Any, b: Any) -> bool:
    """True when two slot values are the same, ignoring case for the string ones.

    Place names arrive title-cased from the code readers ("Dharamshala") and however
    the extraction model happened to capitalise them; "Kangra" and "kangra" are one
    slot value, and reading them as different would flag a change that never happened.
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return bool(a == b)


def _resolve(
    state: AgentState,
    entities: ExtractedEntities,
    visitor_type: Optional[str],
    lang_key: str,
    unbookable_hours: Optional[int] = None,
) -> dict:
    """Apply the slot policy and package the node's return value.

    Shared by the LLM path and the LLM-failure path so a Vertex outage changes what
    the agent KNOWS, never how it BEHAVES.

    `unbookable_hours` is a duration the rider asked for that Him Cab does not sell. It
    travels on state so the question that goes back to them can NAME it — see
    `graph.clarify_node`.
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
        "unbookable_rental_hours": unbookable_hours,
        "node_trace": ["planner"],
    }


async def planner_node(state: AgentState) -> dict:
    """LangGraph Planner Node — extract, merge, classify, then decide."""
    log.info("planner_node_entered", session_id=state.session_id, intent=state.intent)

    lang_key, reply_language = detect_reply_language(state)
    latest_text = _latest_user_text(state)
    # Hiring a car by the hour, decided by the SERVICE the rider opened rather than by
    # this turn's intent — the router re-reads every message and calls half of a rental
    # chat `plan_trip`. Two things hang off it: the trip-shaped fields are dropped, and
    # the message is read for rental details in code as well as by the model.
    is_rental = RENTAL in (state.visitor_type, state.entities.visitor_type)
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
    today = datetime.now(UTC).strftime("%A, %d %B %Y")
    asked_context = (
        f"\nThe assistant's last question asked about: {previously_asked}. The "
        f"user's latest message is most likely the answer to it (it may be a short "
        f"reply or contain a typo)."
        if previously_asked else ""
    )

    # Named in the prompt rather than left to be inferred from the transcript: the
    # extractor's rules for a rental are different, and "rental" in the persona line was
    # being read as just another kind of traveller.
    service_line = (
        "\nRequest type: RENTAL — a car and driver BY THE HOUR. There is no destination "
        "and no trip length. Read the RENTAL CONVERSATION rules below."
        if is_rental else ""
    )

    prompt = f"""Today's date: {today}
Conversation language: {reply_language}
Visitor classification so far: {state.visitor_type or 'unknown'}{service_line}{asked_context}

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

    if is_rental:
        _strip_trip_fields(merged, state.session_id)

    # ── Visitor classification (code, from what they told us) ─────────────────
    # `current` is passed so an in-state ORIGIN cannot promote a visitor to resident;
    # only a stated home city does that. See `classify_visitor_type`.
    # A rental never uses this: `classify_visitor_type` returns RENTAL before it reads
    # `origin_in_himachal` (the pickup says nothing about who the rider is), and the
    # `home_city` promotion below is LOCAL-only. Geocoding the pickup here would be a
    # wasted Nominatim round-trip on the planner's critical path.
    origin_in_hp = result_dict.get("origin_in_himachal")
    if origin_in_hp is None and merged.get("origin") and not is_rental:
        origin_in_hp = await geo.place_in_himachal(merged["origin"])

    visitor_type = (
        await classify_visitor_type(
            merged.get("origin"),
            origin_in_hp,
            home_city=merged.get("home_city"),
            current=state.visitor_type,
        )
        or state.visitor_type
    )
    if visitor_type:
        merged["visitor_type"] = visitor_type
        # The local flow asks "which city in Himachal are you based in?", so when a rider
        # we ALREADY treat as a resident names an in-state origin, that answer is their
        # home city — recorded so the classification survives later turns. Deliberately
        # not done for a visitor: their in-state origin is where the trip starts, not
        # where they live, and writing it here is what used to turn them into a local.
        if visitor_type == LOCAL and origin_in_hp and not merged.get("home_city"):
            merged["home_city"] = merged.get("origin")

    # ── "Here" — the pickup answered by pointing rather than naming ───────────
    # Only for the rental pickup, and only as a reply to that exact question. Now that
    # a rental always asks where the car should collect them, "here" / "yahin" / "my
    # location" is a reply riders genuinely give — and it used to parse as a PLACE
    # called "Here", which the geocoder resolved to something and the car was sent to.
    #
    # The fix is the honest reading of it, and a better one than the pickup this flow
    # used to assume: exact coordinates the rider has just confirmed, rather than a
    # town centre nobody was asked about. It is accepted only when the fix is precise
    # — `is_precise` is what makes `place_name` a name worth saying out loud, and the
    # reply announces the pickup. Without one we would be guessing at a name and
    # sending a car on it, so the question is simply asked again.
    if (
        is_rental
        and latest_text
        and previously_asked == SlotName.ORIGIN
        and not merged.get("origin")
        and slots.means_current_location(latest_text)
    ):
        fix = state.rider_location
        if fix is not None and fix.is_precise and fix.place_name:
            merged["origin"] = fix.place_name
            log.info(
                "rental_pickup_from_device_fix",
                place=fix.place_name,
                distance_km=fix.place_distance_km,
                session_id=state.session_id,
            )
        else:
            # Deliberately fills nothing: `decide` then re-asks, and the second wording
            # ("just the name of the city or town the car should come to") is exactly
            # the right thing to say to somebody whose "here" we could not place.
            log.info(
                "rental_pickup_here_without_usable_fix",
                has_fix=fix is not None,
                is_precise=bool(fix and fix.is_precise),
                session_id=state.session_id,
            )

    # ── Read the reply as an answer to the question we asked ──────────────────
    # Strictly scoped: only the fields that question covered, only if the reply has
    # the right shape. A safety net for a missed one-word answer, not an interpreter.
    recovered = slots.read_answer(previously_asked, latest_text, merged, visitor_type) if latest_text else None
    if recovered:
        log.info("short_answer_recovered", field=recovered, session_id=state.session_id)
        # Skipped for a rental for the same reason as the classification above: the
        # pickup does not decide the persona (`classify_visitor_type` returns RENTAL),
        # so geocoding it and re-classifying would change nothing but the latency.
        if recovered == "origin" and not is_rental:
            # An origin recovered in code carries the same weight as an extracted one.
            recovered_in_hp = await geo.place_in_himachal(merged["origin"])
            visitor_type = await classify_visitor_type(
                merged.get("origin"),
                recovered_in_hp,
                home_city=merged.get("home_city"),
                current=visitor_type,
            ) or visitor_type
            if visitor_type:
                merged["visitor_type"] = visitor_type
                if visitor_type == LOCAL and recovered_in_hp and not merged.get("home_city"):
                    merged["home_city"] = merged.get("origin")

    # ── Read the whole sentence, in code ──────────────────────────────────────
    # `read_answer` above handles a BARE reply to the question in flight ("manali").
    # This handles the other thing riders actually type — a sentence — which used to
    # depend entirely on the extraction model. A miss there was silent: the detail
    # vanished, and the assistant asked for something it had just been told, which is
    # what a rider experiences as not being listened to.
    #
    # Reported twice, once per flow. For rentals it was "I need a car tomorrow at 9 am
    # from Shimla for 8 hours". For trips it was the very first turn: the greeting asks
    # "आप कहाँ जाना चाहते हैं?", the rider answers "mei shimla jana chahta hu", and the
    # same question comes back — because `parse_place` caps an answer at three words on
    # purpose, so a bare "shimla" was recovered and a sentence containing it was not.
    #
    # Every value here is anchored on an unmistakable marker — see the two readers in
    # `slots`. A marker-anchored value fills an EMPTY slot, and it also CORRECTS a slot
    # that already holds one when the rider changes their mind — see `corrected` below.
    corrected: dict[str, Any] = {}
    if latest_text:
        reader = slots.read_rental_details if is_rental else slots.read_trip_details
        filled: dict[str, Any] = {}
        for field, value in reader(latest_text).items():
            current = merged.get(field)
            if current in (None, "", [], {}):
                # Nothing there yet — fill it, exactly as before.
                filled[field] = value
            elif not _same_value(value, current) and _same_value(
                current, existing_entities.get(field)
            ):
                # The rider has just RESTATED this slot as something different, and the
                # model left the previous turn's value standing (`merged[field]` still
                # equals what was already confirmed). This is the reported bug: every
                # detail is in, the booking is on screen, and "mujhe dharamshala jaana
                # hai" changes the destination — but the old "Kangra" survived and the
                # booking never moved. The whole cascade (route, fare, weather, handoff)
                # is rebuilt from the entities each turn, so correcting the slot here is
                # all it takes for the updated booking to ship.
                #
                # Guarded to the case where the model did NOT change it: reading a whole
                # sentence is the model's strength, so a value IT already corrected wins
                # and these regexes never fight it. Only a value the model kept is
                # overwritten, on the same marker that would have filled an empty slot.
                corrected[field] = value

        changes = {**filled, **corrected}
        if changes:
            merged.update(changes)
            log.info(
                "sentence_read_in_code",
                flow="rental" if is_rental else "trip",
                filled=sorted(filled),
                corrected=sorted(corrected),
                llm_ok=llm_ok,
                session_id=state.session_id,
            )

        # A changed pickup re-decides who the rider is, exactly as a recovered or
        # extracted origin does above: "actually, start from Delhi" turns a resident's
        # ride into a visitor's transfer. Skipped for a rental, where the pickup never
        # sets the persona (`classify_visitor_type` returns RENTAL regardless).
        if "origin" in corrected and not is_rental:
            corrected_in_hp = await geo.place_in_himachal(merged["origin"])
            visitor_type = await classify_visitor_type(
                merged.get("origin"),
                corrected_in_hp,
                home_city=merged.get("home_city"),
                current=visitor_type,
            ) or visitor_type
            if visitor_type:
                merged["visitor_type"] = visitor_type
                if visitor_type == LOCAL and corrected_in_hp and not merged.get("home_city"):
                    merged["home_city"] = merged.get("origin")

    # ── "Stop asking and just do it" ──────────────────────────────────────────
    if result_dict.get("user_wants_to_proceed_now") or (
        not llm_ok and _looks_like_proceed_request(latest_text)
    ):
        merged["proceed_requested"] = True
        log.info("proceed_requested", session_id=state.session_id)

    # The booking screen wants a DATE in its date field, not the sentence the rider
    # happened to answer with. Rentals only: a trip's "sometime after Diwali" is worth
    # keeping word for word, and nothing renders it into a form.
    if is_rental and merged.get("travel_dates"):
        merged["travel_dates"] = slots.tidy_when(merged["travel_dates"])

    # ── The validation gate ───────────────────────────────────────────────────
    # Everything above WRITES into `merged` — the extraction model, the answer parsers
    # and the sentence reader — and until this existed only the last two checked what
    # they wrote. The model's output went straight through, so a group of 500, a
    # negative trip length, a pickup of "Here" or a travel date of "yesterday" reached
    # the fare request and the booking screen unopposed.
    #
    # Placed here, after every writer and before `_build_entities`, so there is exactly
    # one gate and no route around it. A value that fails is dropped rather than
    # corrected: the slot is empty again, so `slots.decide` asks for it on this very
    # turn instead of the rider discovering the problem at the booking screen.
    #
    # For a rental this is also where "tomorrow" becomes a real date — see
    # `sanitize.travel_date`.
    merged, rejected = sanitize.sanitize_entities(merged, is_rental=is_rental)
    if rejected:
        log.info(
            "entity_values_rejected",
            fields=rejected,
            session_id=state.session_id,
        )

    # ── A duration we do not sell, refused on the turn it arrives ─────────────
    unbookable_hours = (
        await _reject_unsold_duration(merged, state.rider_jwt, state.session_id)
        if is_rental else None
    )

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
    # A correction is progress too, even though it fills no NEW slot: the rider handed
    # us a real, different value. Without `or corrected`, a turn that only changes an
    # already-known detail would count as no progress and inch the agent towards giving
    # up on a rider who is actively engaged.
    if (_known_slots(entities) - known_before) or corrected:
        entities.no_progress_turns = 0
    else:
        entities.no_progress_turns = state.entities.no_progress_turns + 1

    result = _resolve(state, entities, visitor_type, lang_key, unbookable_hours)

    log.info(
        "entities_extracted",
        destination=entities.destination,
        origin=entities.origin,
        duration=entities.duration_days,
        group_size=entities.group_size,
        travel_dates=entities.travel_dates,
        rental_hours=entities.rental_hours,
        pickup_time=entities.pickup_time,
        cab_category=entities.cab_category,
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
