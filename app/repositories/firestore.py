"""
app/repositories/firestore.py
──────────────────────────
Async Firestore service for session and user profile persistence.
This is the long-term memory layer — data persists across app sessions.
"""

from datetime import UTC, datetime
from typing import Optional

from google.cloud import firestore
from google.cloud.firestore_v1.async_client import AsyncClient

from app.core.config import settings
from app.core.exceptions import SessionNotFoundError
from app.core.logging import get_logger
from app.domain.conversation import SessionStatus, TripSession
from app.domain.user import UserProfile

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pointer_doc_id(user_id: str, persona: str) -> str:
    """Build a Firestore-legal document id for the active-conversation pointer.

    `user_id` looks like `rider:7428186933`. Firestore forbids '/' in document ids and
    reserves ids of the form `__x__`, so both are neutralised here.
    """
    raw = f"{persona}__{user_id}"
    cleaned = raw.replace("/", "_").replace("\\", "_")
    return cleaned.strip("_") or "unknown"


def _sort_key_updated_at(session: TripSession) -> datetime:
    """Timezone-safe sort key — Firestore round-trips can yield naive datetimes."""
    value = session.updated_at or session.created_at
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value


# ── Session Service ───────────────────────────────────────────────────────────

class SessionService:
    """
    Manages TripSession documents in Firestore.
    Collection: `trip_sessions/{session_id}`
    """

    def __init__(self) -> None:
        self._db = get_firestore_client()
        self._col = self._db.collection(settings.firestore_sessions_collection)
        self._pointers = self._db.collection(settings.firestore_active_chats_collection)

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

        from app.domain.conversation import Message, MessageRole

        msg = Message(role=MessageRole(role), content=content, **kwargs)
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "messages": ArrayUnion([msg.model_dump(mode="json")]),
            "updated_at": datetime.now(UTC).isoformat(),
        })

    async def update_itinerary(self, session_id: str, itinerary: dict) -> None:
        """Update only the itinerary field in a session."""
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "current_itinerary": itinerary,
            "status": SessionStatus.ITINERARY_READY.value,
            "updated_at": datetime.now(UTC).isoformat(),
        })

    async def update_entities(self, session_id: str, entities: dict) -> None:
        """Update only the extracted entities in a session."""
        doc_ref = self._col.document(session_id)
        await doc_ref.update({
            "entities": entities,
            "updated_at": datetime.now(UTC).isoformat(),
        })

    async def update_persona(self, session_id: str, persona: str, language: str) -> None:
        """Re-file a conversation under a different persona.

        Written because the stored `persona` is not decoration: `list_user_sessions`
        consumers filter on it, so a document left saying "outstation" after the rider
        was moved to the local assistant becomes unresumable the moment the active-chat
        pointer is lost.
        """
        await self._col.document(session_id).update({
            "persona": persona,
            "language": language,
            "updated_at": datetime.now(UTC).isoformat(),
        })

    async def update_location(
        self, session_id: str, location: dict, entities: Optional[dict] = None
    ) -> None:
        """Store the rider's latest GPS fix on a session.

        `entities` goes in the same write because a fix can fill in the pickup city
        (see `chat_session.apply_location`), and splitting that across two round trips
        risks a session whose stored location and stored origin disagree.
        """
        payload: dict = {
            "rider_location": location,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if entities is not None:
            payload["entities"] = entities
        await self._col.document(session_id).update(payload)

    async def list_user_sessions(
        self,
        user_id: str,
        limit: int = 10,
    ) -> list[TripSession]:
        """Fetch a user's sessions, most recent first.

        The ordered query needs a composite index on (user_id, created_at). If that
        index is missing Firestore raises FAILED_PRECONDITION, which used to take the
        whole conversation-resume path down with it. Fall back to an unordered scan
        sorted in memory so a missing index degrades result quality, not correctness.
        """
        try:
            query = (
                self._col
                .where("user_id", "==", user_id)
                .order_by("created_at", direction=firestore.Query.DESCENDING)
                .limit(limit)
            )
            sessions = await self._collect(query)
            return sessions
        except Exception as e:
            log.warning(
                "session_list_ordered_query_failed",
                user_id=user_id,
                error=str(e),
                hint="create a composite index on (user_id ASC, created_at DESC)",
            )

        # Unordered fallback: over-fetch, then sort client-side.
        query = self._col.where("user_id", "==", user_id).limit(max(limit * 5, 25))
        sessions = await self._collect(query)
        sessions.sort(key=_sort_key_updated_at, reverse=True)
        return sessions[:limit]

    async def _collect(self, query) -> list[TripSession]:
        """Stream a query into TripSession models, skipping undeserialisable docs."""
        sessions: list[TripSession] = []
        async for doc in query.stream():
            try:
                sessions.append(TripSession.from_firestore_dict(doc.to_dict()))
            except Exception as e:
                log.warning("session_parse_failed", doc_id=doc.id, error=str(e))
                continue
        return sessions

    # ── Active-conversation pointer (durable) ─────────────────────────────────
    #
    # "Which conversation is this rider currently in?" is the single fact that makes
    # the chat multi-turn. It used to live only in Redis, so with Redis down every
    # message started a brand-new session and the agent had no memory of the previous
    # turn. Mirroring it into its own Firestore document keeps multi-turn working when
    # the cache is unavailable. A direct document get by id — no index required.

    async def set_active_session(self, user_id: str, persona: str, session_id: str) -> None:
        """Record `session_id` as this rider's current conversation for `persona`."""
        try:
            await self._pointer_ref(user_id, persona).set({
                "user_id": user_id,
                "persona": persona,
                "session_id": session_id,
                "updated_at": datetime.now(UTC).isoformat(),
            })
        except Exception as e:
            log.warning(
                "active_session_pointer_write_failed",
                user_id=user_id, persona=persona, error=str(e),
            )

    async def get_active_session_id(self, user_id: str, persona: str) -> Optional[str]:
        """Read back this rider's current conversation id (None if there isn't one)."""
        try:
            doc = await self._pointer_ref(user_id, persona).get()
            if not doc.exists:
                return None
            value = (doc.to_dict() or {}).get("session_id")
            return value if isinstance(value, str) and value else None
        except Exception as e:
            log.warning(
                "active_session_pointer_read_failed",
                user_id=user_id, persona=persona, error=str(e),
            )
            return None

    def _pointer_ref(self, user_id: str, persona: str):
        return self._pointers.document(_pointer_doc_id(user_id, persona))

    async def ping(self) -> bool:
        """True if Firestore is reachable. Used by /v1/health.

        Reads one document that is never expected to exist — a missing document is a
        successful round-trip, so this costs nothing and needs no index. Unlike the
        pointer accessors, this deliberately does NOT swallow the error, since the
        whole point is to detect it.

        The id avoids the `__x__` form, which Firestore reserves and rejects.
        """
        try:
            await self._pointers.document("health-probe").get()
            return True
        except Exception as e:
            log.warning("firestore_ping_failed", error=str(e))
            return False

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
        profile.updated_at = datetime.now(UTC)
        await self._col.document(profile.user_id).set(profile.to_firestore_dict())

    async def add_past_trip(self, user_id: str, trip_data: dict) -> None:
        """Append a completed trip to the user's history."""
        from google.cloud.firestore_v1 import ArrayUnion
        await self._col.document(user_id).update({
            "past_trips": ArrayUnion([trip_data]),
            "total_trips": firestore.Increment(1),
            "updated_at": datetime.now(UTC).isoformat(),
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
