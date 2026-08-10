"""
app/agent/nodes/context.py
───────────────────────────
Context Node — loads long-term user memory and injects it into agent state.
Responsibilities:
  1. Fetch UserProfile from Firestore
  2. Convert past trips and preferences into a context summary string
  3. Hydrate state.user_profile and state.user_context_summary
  4. Detect frequent travellers and set loyalty tier info
"""

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.services.firestore import get_user_profile_service

log = get_logger(__name__)


async def context_node(state: AgentState) -> dict:
    """
    LangGraph Context Node.
    Loads user profile and prepares personalization context for downstream nodes.
    Detects frequent travellers and sets loyalty tier info.
    """
    log.info("context_node_entered", session_id=state.session_id, user_id=state.user_id)

    if not state.user_id:
        log.warning("context_node_no_user_id")
        return {"node_trace": ["context"], "user_context_summary": ""}

    try:
        profile_service = get_user_profile_service()
        profile = await profile_service.get_or_create_profile(
            user_id=state.user_id,
            firebase_uid=state.user_id,
        )

        context_summary = profile.to_context_string()

        # Detect frequent traveller status
        is_frequent = profile.is_frequent_traveller
        loyalty_tier = profile.get_loyalty_tier()

        log.info(
            "user_profile_loaded",
            user_id=state.user_id,
            past_trips=len(profile.past_trips),
            is_premium=profile.is_premium,
            is_frequent_traveller=is_frequent,
            loyalty_tier=loyalty_tier.value,
            home_city=profile.home_city,
        )

        return {
            "user_profile": profile,
            "user_context_summary": context_summary,
            "is_frequent_traveller": is_frequent,
            "node_trace": ["context"],
        }

    except Exception as e:
        # Non-fatal — proceed without user context
        log.warning("context_node_failed", user_id=state.user_id, error=str(e))
        return {
            "user_context_summary": "",
            "node_trace": ["context"],
        }
