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

import json
from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.prompts.builder import build_router_prompt
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)


async def router_node(state: AgentState) -> dict:
    """
    LangGraph Router Node.
    Takes the latest user message and classifies intent + language + visitor_type.
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
            "visitor_type": state.visitor_type,
            "node_trace": ["router"],
        }

    try:
        llm = get_vertex_ai_service().get_flash_model(streaming=False)
        response = await llm.ainvoke([
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"User message: {latest_message}"),
        ])

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())
        intent = result.get("intent", "general_query")
        detected_language = result.get("language", "en")
        confidence = result.get("confidence", 0.5)
        visitor_type = result.get("visitor_type", "unknown")

        # Don't downgrade a previously classified visitor_type
        if visitor_type == "unknown" and state.visitor_type:
            visitor_type = state.visitor_type

        log.info(
            "intent_classified",
            intent=intent,
            language=detected_language,
            confidence=confidence,
            visitor_type=visitor_type,
            session_id=state.session_id,
        )

        return {
            "intent": intent,
            "language": detected_language,
            "visitor_type": visitor_type,
            "node_trace": ["router"],
        }

    except (json.JSONDecodeError, Exception) as e:
        log.warning("router_fallback", error=str(e))
        return {
            "intent": "general_query",
            "language": state.language,
            "visitor_type": state.visitor_type,
            "node_trace": ["router"],
        }
