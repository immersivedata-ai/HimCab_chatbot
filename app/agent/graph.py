"""
app/agent/graph.py
───────────────────
LangGraph assembly — wires all nodes together and defines routing logic.

Graph topology:
  START → router → context → planner → [clarify | tool_executor]
        │            ↑ (skipped                │
        │             for rentals)        tool_executor → writer → persist → END
        │                                                  │
        │                                           [clarify] → persist → END
        └→ cancel → persist → END

Conditional edges:
  - After `router`:  a cancellation stops the flow dead (see `cancel_node`); a request
                     to see an itinerary we already have skips straight to the writer;
                     a rental goes straight to the planner, skipping `context` because
                     nothing before the writer reads the user profile on that path; and
                     everything else goes through the full flow.
  - After `planner`: ask the user (clarify) or answer them (tool_executor → writer).

Both branch decisions are pure functions of state and are unit-tested directly —
routing is where "the agent asked instead of answering" is decided, so it is kept
free of I/O and of the LLM.

The graph is compiled once at startup and reused across all requests.
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph

from app.agent.nodes.context import context_node
from app.agent.nodes.persist import persist_node
from app.agent.nodes.planner import planner_node
from app.agent.nodes.router import router_node
from app.agent.nodes.tool_executor import tool_executor_node
from app.agent.nodes.writer import writer_node
from app.agent.state import AgentState
from app.core.logging import get_logger

log = get_logger(__name__)


# ── Inline nodes ──────────────────────────────────────────────────────────────

async def clarify_node(state: AgentState) -> dict:
    """Put the planner's question to the user, with no tool calls and no writer LLM.

    The planner always supplies the question (it is the only thing that decides to
    ask), so this node is deliberately thin. The fallback below re-runs the same slot
    policy rather than inventing wording of its own, so there is exactly one place
    where a question's text is chosen.

    The single exception is the rental duration, which is rewritten to name the
    packages the rate card actually sells. The wording still comes from `slots`; only
    the list of options comes from the backend.
    """
    from langchain_core.messages import AIMessage

    from app.agent import slots
    from app.agent.language import detect_reply_language
    from app.core.config import settings
    from app.integrations.himcab import available_packages

    lang_key, _ = detect_reply_language(state)
    question = state.clarification_question
    asked_field = state.last_asked_field

    if not question:
        decision = slots.decide(
            state.entities,
            state.intent,
            state.visitor_type,
            lang_key,
            max_attempts=settings.max_clarification_attempts,
        )
        question, asked_field = decision.question, decision.ask_field

    if not question:
        # Reached only if the graph routed here with nothing left to ask — treat it
        # as "go ahead" rather than showing the user an empty bubble.
        log.warning("clarify_without_question", session_id=state.session_id)
        question = (
            "ठीक है, मैं आपकी यात्रा की योजना बना रहा हूँ..."
            if lang_key == "hi"
            else "Alright — putting your trip plan together..."
        )

    # A day that has already gone. The slot policy cannot word this itself — it reads
    # the entities, and the value was dropped by the validator precisely because it was
    # unusable — so the re-wording happens here, where the rejected phrase is still on
    # state. Without it the rider who typed "yesterday" got "when are you planning to
    # travel?" straight back, which is how this was reported: not as a refusal they
    # disagreed with, but as the assistant failing to read a date at all.
    if asked_field == slots.SlotName.TRAVEL_DATES and state.past_travel_date:
        question = slots.when_question(
            lang_key,
            state.entities.asked_counts.get(slots.SlotName.TRAVEL_DATES, 0) - 1,
            past_day=state.past_travel_date,
        )

    # The packages travel with the question as a structured event as well as in the
    # text, so the app can offer them as taps instead of making the rider type a
    # duration back — the one field where a free-text answer is likely to be unbookable.
    # Read here rather than in the planner because the backend matches a duration
    # EXACTLY: an open "how many hours?" invites an answer no chart matches, and the
    # rider only finds that out after also giving a pickup and a date.
    packages: list[dict] = []
    if asked_field == slots.SlotName.RENTAL_HOURS:
        packages = await available_packages(state.rider_jwt)
        if packages:
            # The attempt count decides the wording: asking a second time with the
            # identical sentence is what made the chat look stuck after a reply we
            # could not read ("8 hourse"). A duration we do not sell outranks both
            # wordings — the rider asked for something specific and is owed a straight
            # answer about it, in the same message that offers the alternatives.
            question = slots.rental_hours_question(
                [p["label"] for p in packages],
                lang_key,
                state.entities.asked_counts.get(slots.SlotName.RENTAL_HOURS, 0) - 1,
                rejected=state.unbookable_rental_hours,
            )

    events: list[dict] = []
    if packages:
        events.append({"type": "rental_packages", "payload": {"packages": packages}})
    # A date or a headcount is one tap in the app (a calendar, a 1-20 number row) rather
    # than something to type — and a tapped answer always parses, where "25/9" or "four
    # of us" might not.
    input_request = slots.input_request_for(asked_field)
    if input_request:
        events.append({"type": "input_request", "payload": input_request})

    return {
        "final_response_text": question,
        "messages": [AIMessage(content=question)],
        "last_asked_field": asked_field,
        "rental_packages": packages,
        "structured_events": events,
        "node_trace": ["clarify"],
    }


async def cancel_node(state: AgentState) -> dict:
    """The rider called it off. Say so, drop the request, and stop.

    Deliberately the shortest path through the graph: no extraction model, no tools, no
    writer. Everything this turn has to say is a fact — what was dropped, that nothing
    was booked or charged, and that only the app can cancel a confirmed booking — and
    each of those is a claim about somebody's money or somebody's driver. The writer LLM
    is the component that was getting them wrong, so it is not consulted; the text comes
    from `cancellation.cancellation_message`.

    It also emits NO fare, NO cab list and NO booking handoff. That is the other half of
    the reported bug: the assistant answered a cancellation with a price and a pre-filled
    booking screen, which is worse than answering nothing at all.
    """
    from langchain_core.messages import AIMessage

    from app.agent import cancellation
    from app.agent.language import detect_reply_language

    lang_key, _ = detect_reply_language(state)
    latest = ""
    for message in reversed(state.messages):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            latest = message.content
            break

    scope = cancellation.cancel_scope(latest, state.entities)
    entities, cleared = cancellation.clear_request(state.entities)
    text = cancellation.cancellation_message(scope, lang_key)

    log.info(
        "request_cancelled",
        scope=scope,
        cleared_fields=cleared,
        was_asking=state.entities.last_asked_field,
        visitor_type=state.visitor_type,
        session_id=state.session_id,
    )

    return {
        "final_response_text": text,
        "messages": [AIMessage(content=text)],
        # The cleared entities are what `persist` writes, so the NEXT message starts a
        # genuinely new request rather than one built half out of the abandoned one.
        "entities": entities,
        "last_asked_field": None,
        "needs_clarification": False,
        # Everything an earlier turn may have put on screen is void. Blanking it here,
        # rather than trusting every later reader to check the intent, is what stops a
        # stale fare card or package chooser outliving the request it belonged to.
        "cab_fare": None,
        "offers": [],
        "route_estimate": None,
        "pickup": None,
        "rental_packages": [],
        "awaiting_cab_choice": False,
        "unavailable_cab_category": None,
        "unbookable_rental_hours": None,
        "past_travel_date": None,
        "structured_events": [cancellation.cancellation_event(scope, cleared)],
        "node_trace": ["cancel"],
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Decide what happens after the intent is known.

    A cancellation leaves immediately, before the planner and before any tool. This is
    the branch that fixes the reported failure: on the old path a rider who typed
    "cancel" reached the extraction model, which reported nothing, and then the slot
    parsers, which read "cancel" as the ANSWER to whatever question was in flight — so
    it became the pickup city, was geocoded, was priced, and came back as a booking
    screen for the request they had just abandoned.

    Only a `show_itinerary` request for a trip we already have a destination for can
    skip the planner. Without one there is nothing to show, and jumping to the writer
    would have it invent a trip out of thin air — so it goes through the normal flow
    and gets asked where they want to go.

    Never taken in a rental chat, whatever the router made of the message. There is no
    itinerary to show — and a `destination` the extractor picked up in passing was
    never a trip the rider asked for, so acting on it would hand somebody hiring a car
    for four hours a multi-day plan.
    """
    from app.agent import slots

    if state.intent == slots.CANCEL_INTENT:
        return "cancel"
    if state.visitor_type == slots.RENTAL:
        # Straight to the planner, skipping `context`. A rental reply never reads the
        # user profile — the writer omits it in RENTAL_MODE (see writer._build_system_
        # prompt) and `persist` fetches its own for `add_past_trip` — so the context
        # node's Firestore round-trip is pure latency on this path. This skips CONTEXT,
        # NOT the planner: a `destination` the extractor caught in passing must still be
        # stripped rather than acted on (see `_strip_trip_fields` and the itinerary
        # shortcut below), which only the planner does.
        return "planner"
    if state.intent == "show_itinerary" and state.entities.destination:
        return "writer"
    return "context"


def route_after_planner(state: AgentState) -> str:
    """Ask the user, or answer them.

    The destination guard is a backstop, not policy — the slot policy already makes
    `destination` mandatory wherever it matters. It is here because an itinerary
    without a destination is a fabricated itinerary, which is worth catching as a
    question — as is a cab quote with nowhere to drive to. It deliberately does NOT
    apply to a general question ("when's the best time to visit Himachal?"), which can
    be answered without one.
    """
    from app.agent import slots

    if state.needs_clarification:
        return "clarify"
    mode = slots.response_mode(state.intent, state.visitor_type)
    # RENTAL_MODE is deliberately absent: hiring a car by the hour HAS no destination,
    # so guarding on one would ask a rider to invent an answer they do not have.
    if mode in (slots.ITINERARY_MODE, slots.RIDE_MODE) and not state.entities.destination:
        log.warning(
            "planner_cleared_without_destination",
            session_id=state.session_id,
            intent=state.intent,
        )
        return "clarify"
    return "tool_executor"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    """
    Constructs and returns the compiled LangGraph StateGraph.
    Called once at startup via the `get_agent_graph()` singleton.
    """
    builder = StateGraph(AgentState)

    # ── Add nodes ─────────────────────────────────────────────────────────────
    builder.add_node("router", router_node)
    builder.add_node("cancel", cancel_node)
    builder.add_node("context", context_node)
    builder.add_node("planner", planner_node)
    builder.add_node("clarify", clarify_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("writer", writer_node)
    builder.add_node("persist", persist_node)

    # ── Define edges ──────────────────────────────────────────────────────────

    # Entry point
    builder.add_edge(START, "router")

    # After router: conditional branch
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "context": "context",
            # A rental turn joins the graph one node later: nothing before the writer
            # reads the user profile, so context is skipped (see `route_after_router`).
            "planner": "planner",
            "writer": "writer",
            "cancel": "cancel",
        },
    )

    # Cancel → persist (which saves the reply and the cleared entities) → END.
    builder.add_edge("cancel", "persist")

    # After context: always go to planner
    builder.add_edge("context", "planner")

    # After planner: conditional branch
    builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "clarify": "clarify",
            "tool_executor": "tool_executor",
        },
    )

    # Clarify → persist (save clarification message) → END
    builder.add_edge("clarify", "persist")
    builder.add_edge("persist", END)

    # Tool executor → writer → persist → END
    builder.add_edge("tool_executor", "writer")
    builder.add_edge("writer", "persist")

    log.info("langgraph_built")
    return builder


@lru_cache(maxsize=1)
def get_agent_graph():
    """
    Return the compiled LangGraph graph.
    The graph is compiled once and cached — thread safe.
    """
    builder = _build_graph()
    graph = builder.compile()
    log.info("langgraph_compiled")
    return graph
