"""
app/api/v1/routes/session.py
─────────────────────────────
Session management endpoints.

POST   /v1/session           — Create new trip session
GET    /v1/session/{id}      — Get session details
GET    /v1/session/{id}/history    — Conversation history
GET    /v1/session/{id}/itinerary  — Latest itinerary
DELETE /v1/session/{id}      — Delete session

AUTH: the same `Authorization: Bearer <Him Cab rider JWT>` the /v1/chat/* endpoints
take. These routes previously required a Firebase ID token instead, so a token that
worked for chat got a 401 here — which broke conversation restore in the app, since
/history is how a client rehydrates a chat it already owns. One credential now works
across the whole v1 API.
"""

from fastapi import APIRouter, Depends

from app.api.middleware.rider_auth import RiderIdentity, get_rider
from app.core.exceptions import AuthorizationError
from app.core.logging import get_logger
from app.domain.conversation import TripSession
from app.repositories.firestore import get_session_service
from app.schemas.session import CreateSessionRequest, CreateSessionResponse

log = get_logger(__name__)
router = APIRouter(prefix="/session", tags=["Session"])


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    rider: RiderIdentity = Depends(get_rider),
):
    """
    Create a new trip planning session.
    Call this before opening the chat UI.
    Returns session_id which must be passed to /v1/chat/stream.
    """
    session_svc = get_session_service()
    session = TripSession(
        user_id=rider.user_id,
        language=body.language,
    )

    # Add system greeting message if initial_message provided
    if body.initial_message:
        from app.domain.conversation import MessageRole
        session.add_message(MessageRole.USER, body.initial_message)

    await session_svc.create_session(session)

    log.info("session_created_api", session_id=session.session_id, user_id=rider.user_id)

    return CreateSessionResponse(
        session_id=session.session_id,
        created_at=session.created_at,
        status=session.status,
    )


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    rider: RiderIdentity = Depends(get_rider),
):
    """Get session metadata and current status."""
    session_svc = get_session_service()
    session = await session_svc.get_session(session_id)

    if session.user_id != rider.user_id:
        raise AuthorizationError("You do not own this session")

    return {
        "session_id": session.session_id,
        "status": session.status,
        "entities": session.entities.model_dump(exclude_none=True),
        "language": session.language,
        "message_count": len(session.messages),
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
    }


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str,
    rider: RiderIdentity = Depends(get_rider),
    limit: int = 50,
):
    """
    Get full conversation history for a session.
    Used to restore chat UI when user re-opens the app.
    """
    session_svc = get_session_service()
    session = await session_svc.get_session(session_id)

    if session.user_id != rider.user_id:
        raise AuthorizationError("You do not own this session")

    messages = session.messages[-limit:]
    return {
        "session_id": session_id,
        "messages": [m.model_dump() for m in messages],
        "total_messages": len(session.messages),
    }


@router.get("/{session_id}/itinerary")
async def get_session_itinerary(
    session_id: str,
    rider: RiderIdentity = Depends(get_rider),
):
    """
    Get the latest generated itinerary for a session.
    Returns null if no itinerary has been generated yet.
    """
    session_svc = get_session_service()
    session = await session_svc.get_session(session_id)

    if session.user_id != rider.user_id:
        raise AuthorizationError("You do not own this session")

    if not session.current_itinerary:
        return {"session_id": session_id, "itinerary": None}

    return {
        "session_id": session_id,
        "itinerary": session.current_itinerary.model_dump(),
    }


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    rider: RiderIdentity = Depends(get_rider),
):
    """Hard-delete a session and all its data."""
    session_svc = get_session_service()
    session = await session_svc.get_session(session_id)

    if session.user_id != rider.user_id:
        raise AuthorizationError("You do not own this session")

    await session_svc.delete_session(session_id)
    return {"message": "Session deleted", "session_id": session_id}
