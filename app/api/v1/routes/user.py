"""
app/api/v1/routes/user.py
──────────────────────────
User profile and trip history endpoints.

GET  /v1/user/profile   — Get user's profile and preferences
PUT  /v1/user/profile   — Update preferences
GET  /v1/user/trips     — Get past trip sessions
"""

from fastapi import APIRouter, Depends

from app.api.middleware.auth import FirebaseUser, get_current_user
from app.core.logging import get_logger
from app.models.user import TravelPreferences, UserProfile
from app.services.firestore import get_session_service, get_user_profile_service

log = get_logger(__name__)
router = APIRouter(prefix="/user", tags=["User"])


@router.get("/profile")
async def get_profile(user: FirebaseUser = Depends(get_current_user)):
    """Get authenticated user's profile and travel preferences."""
    profile_svc = get_user_profile_service()
    profile = await profile_svc.get_or_create_profile(
        user_id=user.uid,
        firebase_uid=user.uid,
    )
    return profile.model_dump()


@router.put("/profile")
async def update_profile(
    preferences: TravelPreferences,
    user: FirebaseUser = Depends(get_current_user),
):
    """Update user's travel preferences."""
    profile_svc = get_user_profile_service()
    profile = await profile_svc.get_or_create_profile(
        user_id=user.uid,
        firebase_uid=user.uid,
    )
    profile.preferences = preferences
    await profile_svc.update_profile(profile)
    return {"message": "Profile updated", "preferences": preferences.model_dump()}


@router.get("/trips")
async def get_past_trips(
    user: FirebaseUser = Depends(get_current_user),
    limit: int = 10,
):
    """
    Get user's past trip planning sessions.
    Each session contains the full itinerary and conversation history.
    """
    session_svc = get_session_service()
    sessions = await session_svc.list_user_sessions(user_id=user.uid, limit=limit)

    return {
        "user_id": user.uid,
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
