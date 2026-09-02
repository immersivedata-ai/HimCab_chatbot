"""
app/api/v1/routes/user.py
──────────────────────────
User profile and trip history endpoints.

GET  /v1/user/profile   — Get user's profile and preferences
PUT  /v1/user/profile   — Update preferences
GET  /v1/user/trips     — Get past trip sessions

AUTH: the same `Authorization: Bearer <Him Cab rider JWT>` as /v1/chat/* and
/v1/session/*. These routes used to require a Firebase ID token, which is why a token
accepted by chat was rejected here with 401. See app/api/v1/routes/session.py.
"""

from fastapi import APIRouter, Depends

from app.api.middleware.rider_auth import RiderIdentity, get_rider
from app.core.logging import get_logger
from app.domain.user import TravelPreferences
from app.repositories.firestore import get_session_service, get_user_profile_service

log = get_logger(__name__)
router = APIRouter(prefix="/user", tags=["User"])


@router.get("/profile")
async def get_profile(rider: RiderIdentity = Depends(get_rider)):
    """Get authenticated user's profile and travel preferences."""
    profile_svc = get_user_profile_service()
    profile = await profile_svc.get_or_create_profile(
        user_id=rider.user_id,
        firebase_uid=rider.user_id,
    )
    return profile.model_dump()


@router.put("/profile")
async def update_profile(
    preferences: TravelPreferences,
    rider: RiderIdentity = Depends(get_rider),
):
    """Update user's travel preferences."""
    profile_svc = get_user_profile_service()
    profile = await profile_svc.get_or_create_profile(
        user_id=rider.user_id,
        firebase_uid=rider.user_id,
    )
    profile.preferences = preferences
    await profile_svc.update_profile(profile)
    return {"message": "Profile updated", "preferences": preferences.model_dump()}


@router.get("/trips")
async def get_past_trips(
    rider: RiderIdentity = Depends(get_rider),
    limit: int = 10,
):
    """
    Get user's past trip planning sessions.
    Each session contains the full itinerary and conversation history.
    """
    session_svc = get_session_service()
    sessions = await session_svc.list_user_sessions(user_id=rider.user_id, limit=limit)

    return {
        "user_id": rider.user_id,
        "trips": [
            {
                "session_id": s.session_id,
                "destination": s.entities.destination,
                "duration_days": s.entities.duration_days,
                "status": s.status,
                "has_itinerary": s.current_itinerary is not None,
                "created_at": s.created_at.isoformat(),
                "updated_at": s.updated_at.isoformat(),
            }
            for s in sessions
        ],
        "total": len(sessions),
    }
