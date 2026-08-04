"""
app/agent/graph.py
───────────────────
LangGraph assembly — wires all nodes together and defines routing logic.

Graph topology:
  START → router → context → planner → [clarify | tool_executor]
                                            │
                                       tool_executor → writer → persist → END
                                                            │
                                                     [clarify] → END

Conditional edges:
  - After `planner`: if needs_clarification → go to `clarify_node` (skip tools)
                     else → go to `tool_executor`
  - After `router`:  intent == "show_itinerary" → go directly to `writer` (no tools needed)
                     otherwise → full planning flow

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
    """
    Lightweight node that surfaces a clarification question to the user
    without invoking any tools or the writer LLM.
    """
    from langchain_core.messages import AIMessage
    question = state.clarification_question or (
        "आपकी यात्रा के बारे में थोड़ी और जानकारी दे सकते हैं? आप कहाँ जाना चाहते हैं और कितने दिनों के लिए?"
    )
    return {
        "final_response_text": question,
        "messages": [AIMessage(content=question)],
        "node_trace": ["clarify"],
    }


# ── Conditional routing functions ─────────────────────────────────────────────

def route_after_router(state: AgentState) -> str:
    """
    After the router classifies intent, decide which node comes next.
    - show_itinerary → skip to writer (no tool calls needed)
    - everything else → context → planner → tools → writer
    """
    if state.intent == "show_itinerary":
        return "writer"
    return "context"


def route_after_planner(state: AgentState) -> str:
    """
    After the planner extracts entities, decide:
    - If we need more info → ask user (clarify)
    - If we have enough → call tools
    """
    if state.needs_clarification:
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
