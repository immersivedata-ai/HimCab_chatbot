"""
app/services/firestore.py
──────────────────────────
Async Firestore service for session and user profile persistence.
This is the long-term memory layer — data persists across app sessions.
"""

from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.core.logging import get_logger
from app.models.session import TripSession, SessionStatus
from app.models.user import UserProfile

log = get_logger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────

_firestore_client: Optional[AsyncClient] = None


def get_firestore_client() -> AsyncClient:
    """Return the shared async Firestore client."""
    global _firestore_client
    if _firestore_client is None:
        _firestore_client = firestore.AsyncClient(
            project=settings.firebase_project_id,
            database=settings.firestore_database_id,
        )
    return _firestore_client


# ── Session Service ───────────────────────────────────────────────────────────

class SessionService:
    """
    Manages TripSession documents in Firestore.
    Collection: `trip_sessions/{session_id}`
    """

    def __init__(self) -> None:
        self._db = get_firestore_client()
        self._col = self._db.collection(settings.firestore_sessions_collection)

    async def create_session(self, session: TripSession) -> TripSession:
        """Persist a new session to Firestore."""
        doc_ref = self._col.document(session.session_id)
        await doc_ref.set(session.to_firestore_dict())
        log.info("session_created", session_id=session.session_id, user_id=session.user_id)
        return session

    async def get_session(self, session_id: str) -> TripSession:
        """
        Fetch session from Firestore.
        Raises SessionNotFoundError if not found.
        """
        doc = await self._col.document(session_id).get()
        if not doc.exists:
            raise SessionNotFoundError(f"Session {session_id} not found")
        data = doc.to_dict()
        return TripSession.from_firestore_dict(data)

    async def update_session(self, session: TripSession) -> None:
        """Persist updated session data."""
        session.updated_at = datetime.now(timezone.utc)
        doc_ref = self._col.document(session.session_id)
        await doc_ref.set(session.to_firestore_dict())
        log.debug("session_updated", session_id=session.session_id)

    async def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **kwargs,
    ) -> None:
        """
        Atomically append a message to the session's messages array.
        More efficient than a full document rewrite.
        """
        from google.cloud.firestore_v1 import ArrayUnion
        from app.models.session import Message, MessageRole

        msg = Message(role=MessageRole(role), content=content, **kwargs)
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "messages": ArrayUnion([msg.model_dump(mode="json")]),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    async def update_itinerary(self, session_id: str, itinerary: dict) -> None:
        """Update only the itinerary field in a session."""
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "current_itinerary": itinerary,
            "status": SessionStatus.ITINERARY_READY.value,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    async def update_entities(self, session_id: str, entities: dict) -> None:
        """Update only the extracted entities in a session."""
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "entities": entities,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })

    async def list_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[TripSession]:
        """Fetch all sessions for a user, ordered by most recent."""
        query = (
            self._col
            .where("user_id", "==", user_id)
            .order_by("created_at", direction=firestore.Query.DESCENDING)
            .limit(limit)
        )
        sessions = []
        async for doc in query.stream():
            try:
                sessions.append(TripSession.from_firestore_dict(doc.to_dict()))
            except Exception as e:
                log.warning("session_parse_failed", doc_id=doc.id, error=str(e))
                continue
        return sessions

    async def delete_session(self, session_id: str) -> None:
        """Hard-delete a session."""
        await self._col.document(session_id).delete()
        log.info("session_deleted", session_id=session_id)


# ── User Profile Service ──────────────────────────────────────────────────────

class UserProfileService:
    """
    Manages UserProfile documents in Firestore.
    Collection: `user_profiles/{user_id}`
    """

    def __init__(self) -> None:
        self._db = get_firestore_client()
        self._col = self._db.collection(settings.firestore_users_collection)

    async def get_or_create_profile(self, user_id: str, firebase_uid: str) -> UserProfile:
        """Fetch user profile, creating a default one if it doesn't exist."""
        doc = await self._col.document(user_id).get()
        if doc.exists:
            return UserProfile.from_firestore_dict(doc.to_dict())

        # Create default profile on first login
        profile = UserProfile(user_id=user_id, firebase_uid=firebase_uid)
        await self._col.document(user_id).set(profile.to_firestore_dict())
        log.info("user_profile_created", user_id=user_id)
        return profile

    async def update_profile(self, profile: UserProfile) -> None:
        """Persist updated user profile."""
        profile.updated_at = datetime.now(timezone.utc)
        await self._col.document(profile.user_id).set(profile.to_firestore_dict())

    async def add_past_trip(self, user_id: str, trip_data: dict) -> None:
        """Append a completed trip to the user's history."""
        from google.cloud.firestore_v1 import ArrayUnion
        await self._col.document(user_id).update({
            "past_trips": ArrayUnion([trip_data]),
            "total_trips": firestore.Increment(1),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })


# ── Singletons ────────────────────────────────────────────────────────────────

_session_service: Optional[SessionService] = None
_user_profile_service: Optional[UserProfileService] = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service


def get_user_profile_service() -> UserProfileService:
    global _user_profile_service
    if _user_profile_service is None:
        _user_profile_service = UserProfileService()
    return _user_profile_service
