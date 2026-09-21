"""
app/agent/nodes/router.py
──────────────────────────
Router Node — the first node in the LangGraph.
Responsibilities:
  1. Detect user language (Hindi/Hinglish/English)
  2. Classify user intent using Gemini Flash (fast, cheap)
  3. Route to the correct next node

Intent categories:
  - plan_trip:      Full trip planning request
  - weather_query:  User asking about weather/snowfall
  - hotel_query:    User asking specifically about hotels
  - places_query:   User asking about attractions/food/places
  - cost_query:     User asking about trip cost
  - show_itinerary: User wants to see their existing itinerary
  - general_query:  General Himachal travel question
  - clarification:  User answering a clarification question

Prompt template: app/prompts/templates/router_system.yaml
"""

import asyncio
import json

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.cancellation import looks_like_cancellation
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import Intent, Persona
from app.core.logging import get_logger
from app.prompts.builder import build_router_prompt
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

ROUTER_SYSTEM_PROMPT = build_router_prompt()


def _answers_the_question(asked_field: str, text: str, *, is_rental: bool = False) -> bool:
    """True when `text` reads as an answer to the question we just asked.

    Asked of the SAME strict parser the planner will use on the next node, so the two
    can never disagree about whether a reply was an answer. Those parsers are
    deliberately biased towards rejecting — see app/agent/slots.py — which is what
    keeps a genuine topic switch ("actually, what's the weather there?") from being
    swallowed as one: it does not parse as a city, so it keeps its own intent.
    """
    from app.agent import slots

    slot = slots.SLOTS.get(asked_field)
    if slot is None or not text or not text.strip():
        return False
    try:
        # The same strip the planner applies before it reads a correction. Without it
        # the two DO disagree: "nahi, Kasol" is a correction the planner will act on,
        # but the bare parser reads four words and rejects it, so the turn is routed
        # as something other than an answer to the question still on screen.
        if slot.parse(slots.strip_correction_prefix(text)) is not None:
            return True

        # A SENTENCE answers the question too, and asking only the bare parser meant it
        # did not count. `parse_place` caps an answer at three words on purpose, so
        # "mei himachal mei shimla mei rehta hu" — a rider answering "which city do you
        # live in?" — parsed as nothing, kept the classifier's `general_query`, and took
        # the whole turn down a path with no fare and no booking screen. The rider had
        # answered the question exactly.
        #
        # The planner reads that sentence perfectly well (`read_trip_details`), so the
        # two nodes disagreed about whether the rider had answered — the precise thing
        # this function exists to prevent. Same readers on both sides now.
        read = slots.read_rental_details if is_rental else slots.read_trip_details
        return read(text).get(asked_field) is not None
    except Exception:  # a parser must never take down the turn
        return False


async def router_node(state: AgentState) -> dict:
    """
    LangGraph Router Node.
    Classifies the user's intent (for routing) and detected language.

    NOTE: the router does NOT set `visitor_type`. That is owned by the persona seed
    (which front-end page the user is on, set in app/api/v1/routes/chat.py) and, once an
    origin city is known, by the planner's code-based classification. The router's LLM
    guess at visitor_type was unreliable (e.g. it would flip an outstation visitor to
    "local" on a bare HP place name), so we deliberately leave the field untouched here —
    LangGraph keeps the seeded value for any channel a node doesn't write.
    """
    log.info("router_node_entered", session_id=state.session_id)

    # Get the latest human message
    latest_message = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            latest_message = msg.content
            break

    if not latest_message:
        return {
            "intent": "general_query",
            "language": state.language,
            "node_trace": ["router"],
        }

    # ── Calling it off, decided in code ───────────────────────────────────────
    # Before the model, and not overridable by it. A rider who says "cancel" has said
    # the one thing this system must never get creatively wrong, and everything
    # downstream keys off this intent: the graph skips straight to `cancel_node`, so
    # the extraction LLM never runs, the slot parsers never read "cancel" as an answer
    # to the question in flight, and no tool is called for a request that is over.
    # The classifier still gets its say for phrasings this list does not cover — see
    # the CANCEL rules in the router prompt.
    if looks_like_cancellation(latest_message):
        log.info(
            "cancellation_detected",
            source="phrases",
            asked_about=state.entities.last_asked_field,
            session_id=state.session_id,
        )
        return {
            "intent": Intent.CANCEL_REQUEST,
            "language": state.language,
            "node_trace": ["router"],
        }

    try:
        llm = get_vertex_ai_service().get_flash_model(streaming=False)
        response = await asyncio.wait_for(
            llm.ainvoke([
                SystemMessage(content=ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=f"User message: {latest_message}"),
            ]),
            timeout=settings.llm_call_timeout_seconds,
        )

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        intent = result.get("intent", "general_query")
        detected_language = result.get("language", "en")
        confidence = result.get("confidence", 0.5)

        # A cancellation the phrase list missed. Logged separately from the code path
        # so the two detectors can be compared: a rise here is a phrase worth adding,
        # and a rise there is the classifier drifting.
        if intent == Intent.CANCEL_REQUEST:
            log.info(
                "cancellation_detected",
                source="classifier",
                asked_about=state.entities.last_asked_field,
                session_id=state.session_id,
            )

        # A bare reply to our own question ("kasol", "delhi", "5") carries no signal of
        # its own, so the classifier files it under whatever the WORD looks like. That
        # used to be handled for `general_query` alone, and the gap was the reported
        # bug: asked "which city in Himachal do you live in?", the rider answered
        # "kasol" and the router returned `places_query` — a question about a place.
        #
        # Everything downstream then did the right thing for the wrong question.
        # `places_query` is not a travel intent, so `response_mode` returned ANSWER, the
        # fare was never requested (`pricing_intent=False`), no booking handoff shipped,
        # and the rider who wanted a cab from Kasol to Shimla got a write-up of Shimla's
        # attractions instead.
        #
        # So the test is no longer which bucket the classifier chose, but whether the
        # message actually ANSWERS the question in flight — asked of the same parser the
        # planner will use, so the two cannot disagree. A real topic switch ("actually,
        # what's the weather there?") does not parse as an answer and keeps its intent.
        asked = state.entities.last_asked_field
        if (
            asked
            and intent != Intent.CLARIFICATION
            and _answers_the_question(
                asked, latest_message, is_rental=state.visitor_type == Persona.RENTAL
            )
        ):
            log.info(
                "intent_reclassified_as_answer",
                asked_about=asked,
                classifier_said=intent,
                session_id=state.session_id,
            )
            intent = Intent.CLARIFICATION

        log.info(
            "intent_classified",
            intent=intent,
            language=detected_language,
            confidence=confidence,
            # informational only — NOT written to state (see docstring)
            router_visitor_guess=result.get("visitor_type", "unknown"),
            session_id=state.session_id,
        )

        return {
            "intent": intent,
            "language": detected_language,
            "node_trace": ["router"],
        }

    except Exception as e:
        log.warning("router_fallback", error=str(e), error_type=type(e).__name__)
        return {
            "intent": "general_query",
            "language": state.language,
            "node_trace": ["router"],
        }
