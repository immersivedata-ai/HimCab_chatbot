"""
app/agent/nodes/persist.py
───────────────────────────
Persist Node — the final node in the graph.
Responsibilities:
  1. Append the AI message to Firestore atomically
  2. Save the itinerary and the extracted entities
  3. Update the user profile with new trip data if an itinerary was generated
"""

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.services.firestore import get_session_service, get_user_profile_service
from app.models.session import MessageRole

log = get_logger(__name__)


async def persist_node(state: AgentState) -> dict:
    """
    LangGraph Persist Node.
    Writes all session state back to Firestore.
    """
    log.info("persist_node_entered", session_id=state.session_id)

    session_svc = get_session_service()

    try:
        # 1. Append the assistant's message to Firestore (atomic array append)
        if state.final_response_text:
            await session_svc.append_message(
                session_id=state.session_id,
                role=MessageRole.ASSISTANT,
                content=state.final_response_text,
            )

        # 2. If we have a new itinerary, update it in Firestore
        if state.structured_itinerary:
            await session_svc.update_itinerary(
                session_id=state.session_id,
                itinerary=state.structured_itinerary.model_dump(mode="json"),
            )

            # 3. Update user profile with new trip if destination is known
            if state.entities.destination and state.user_id:
                try:
                    profile_svc = get_user_profile_service()
                    await profile_svc.add_past_trip(
                        user_id=state.user_id,
                        trip_data={
                            "destination": state.entities.destination,
                            "duration_days": state.entities.duration_days,
                            "travel_date": state.entities.travel_dates or "upcoming",
                            "session_id": state.session_id,
                        },
                    )
                except Exception as e:
                    log.warning("profile_update_failed", error=str(e))

        # 4. Update the extracted entities in Firestore so we don't forget them
        if state.entities:
            await session_svc.update_entities(
                session_id=state.session_id,
                entities=state.entities.model_dump(mode="json"),
            )

        log.info(
            "persist_completed",
            session_id=state.session_id,
            has_itinerary=state.structured_itinerary is not None,
        )

    except Exception as e:
        # Persist failure is non-fatal — user already got their response
        log.error("persist_failed", session_id=state.session_id, error=str(e))

    return {"node_trace": ["persist"]}
