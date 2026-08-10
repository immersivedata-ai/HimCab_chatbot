"""
app/agent/graph.py
───────────────────
LangGraph assembly — wires all nodes together and defines routing logic.

Graph topology:
  START → router → context → planner → [clarify | tool_executor]
                                            │
                                       tool_executor → writer → persist → END
                                                            │
                                                     [clarify] → persist → END

Conditional edges:
  - After `router`:  a request to see an itinerary we already have skips straight to
                     the writer; everything else goes through the full flow.
  - After `planner`: ask the user (clarify) or answer them (tool_executor → writer).

Both branch decisions are pure functions of state and are unit-tested directly —
routing is where "the agent asked instead of answering" is decided, so it is kept
free of I/O and of the LLM.

The graph is compiled once at startup and reused across all requests.
"""

from functools import lru_cache

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
    """
    from langchain_core.messages import AIMessage

    from app.agent import slots
    from app.agent.language import detect_reply_language
    from app.core.config import settings

    question = state.clarification_question
    asked_field = state.last_asked_field

    if not question:
        lang_key, _ = detect_reply_language(state)
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
        lang_key, _ = detect_reply_language(state)
        question = (
            "ठीक है, मैं आपकी यात्रा की योजना बना रहा हूँ..."
            if lang_key == "hi"
            else "Alright — putting your trip plan together..."
        )

    return {
        "final_response_text": question,
        "messages": [AIMessage(content=question)],
        "last_asked_field": asked_field,
        "node_trace": ["clarify"],
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """Decide what happens after the intent is known.

    Only a `show_itinerary` request for a trip we already have a destination for can
    skip the planner. Without one there is nothing to show, and jumping to the writer
    would have it invent a trip out of thin air — so it goes through the normal flow
    and gets asked where they want to go.
    """
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
            "writer": "writer",
        },
    )

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
