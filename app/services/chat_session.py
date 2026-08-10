"""
app/services/chat_session.py
─────────────────────────────
Server-side session ownership for the chat API.

The front end need not send (or store) a session id. It sends its Bearer token, and
this module maps that caller to the right conversation:

    POST /chat/{persona}/start   → start_chat()    — always a FRESH conversation
    GET  /chat/{persona}/stream  → resolve_chat()  — the caller's CURRENT conversation
    POST /chat/{persona}/message → resolve_chat()  — ditto

A client that DOES want to pin a conversation (two devices on one account, or
reopening a specific chat) may pass `session_id` explicitly — see resolve_chat().

Where the pointer lives, in order:
  1. Explicit `session_id` from the caller, if it belongs to them  (exact pin)
  2. Firestore `active_chats/{persona}__{user_id}` document        (a direct document
     read, so it needs no composite index)
  3. Firestore recent-session scan                                 (last-ditch resume)
  4. Brand-new session                                             (last resort)

Tier 2 used to sit behind a Redis cache. The cache is gone: it was one more thing that
could be down, and when it was, tier 2 carried the whole load anyway. A single document
read per turn is well within Firestore's budget.

Tier 4 means these endpoints never 404 on a missing/expired session — the front end
cannot get into a broken state that requires it to "reset" anything.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.logging import get_logger
from app.models.session import Language, MessageRole, TripSession
from app.services.firestore import get_session_service

log = get_logger(__name__)

# Personas = which front-end app opened the chat. Fixed by the route, never sent by
# the client. Durably stored on the session (TripSession.persona).
LOCAL = "local"
OUTSTATION = "outstation"
VALID_PERSONAS = (LOCAL, OUTSTATION)

# How far back the recent-session scan will look to resume a conversation when the
# active-chat pointer is missing. Beyond this we'd rather start clean than resurrect a
# chat the user has mentally moved on from.
_RESUME_WINDOW = timedelta(hours=24)

# Opening line shown the moment the chat dialog opens, before the user types anything.
# Persisted as a real assistant message so the planner sees it as conversation context
# — that is what lets a one-word reply ("manali") be understood as the answer to it.
WELCOME_MESSAGES: dict[str, str] = {
    LOCAL: (
        "नमस्ते! Him Cab में आपका स्वागत है 🙏 "
        "मैं आपकी हिमाचल यात्रा की योजना बनाने में मदद कर सकता हूँ। "
        "आप कहाँ जाना चाहते हैं?"
    ),
    OUTSTATION: (
        "Welcome to Him Cab! 🙏 "
        "I can help you plan your trip to Himachal Pradesh. "
        "Where would you like to go?"
    ),
}

# Each front end has a default reply language; the agent still adapts to whatever the
# user actually writes (see app/agent/language.py).
DEFAULT_LANGUAGE: dict[str, Language] = {
    LOCAL: Language.HINDI,
    OUTSTATION: Language.ENGLISH,
}


def welcome_message(persona: str) -> str:
    """The opening line for a persona (falls back to the outstation/English copy)."""
    return WELCOME_MESSAGES.get(persona, WELCOME_MESSAGES[OUTSTATION])


async def _remember_active(persona: str, user_id: str, session_id: str) -> None:
    """Point this caller at `session_id` as their current conversation."""
    await get_session_service().set_active_session(user_id, persona, session_id)


async def _recall_active(persona: str, user_id: str) -> Optional[str]:
    """Read back the caller's current session id. None when there isn't one."""
    session_id = await get_session_service().get_active_session_id(user_id, persona)
    if session_id:
        log.debug(
            "chat_session_pointer_from_firestore",
            user_id=user_id,
            persona=persona,
            session_id=session_id,
        )
    return session_id


async def _load_owned_session(session_id: str, user_id: str) -> Optional[TripSession]:
    """Load `session_id` only if it belongs to `user_id`. None on miss or mismatch."""
    try:
        session = await get_session_service().get_session(session_id)
    except Exception as e:
        log.info("chat_session_lookup_failed", session_id=session_id, error=str(e))
        return None

    if session.user_id != user_id:
        log.warning(
            "chat_session_owner_mismatch",
            session_id=session_id,
            expected_user=user_id,
        )
        return None
    return session


async def _create_session(persona: str, user_id: str) -> TripSession:
    """Create + persist a fresh conversation, pre-seeded with the welcome message."""
    language = DEFAULT_LANGUAGE.get(persona, Language.ENGLISH)
    session = TripSession(
        user_id=user_id,
        language=language,
        persona=persona,
    )
    # Seed the agent's starting assumption about who this is. The planner may still
    # correct it later from a real origin city (a "local" page user travelling from
    # Delhi is reclassified as outstation).
    session.entities.visitor_type = persona
    # The welcome message below asks where they want to go, so record that as the
    # question in flight. Without it the planner treats the first reply as unprompted
    # and its answer-parsing safety net can't fire on a one-word "manali".
    session.entities.last_asked_field = "destination"
    session.add_message(
        MessageRole.ASSISTANT,
        welcome_message(persona),
        language=language,
    )

    await get_session_service().create_session(session)
    await _remember_active(persona, user_id, session.session_id)

    log.info(
        "chat_session_started",
        session_id=session.session_id,
        user_id=user_id,
        persona=persona,
    )
    return session


async def _find_resumable(persona: str, user_id: str) -> Optional[TripSession]:
    """Look for a recent conversation to resume when the pointer document is missing.

    Best-effort only: this query needs a Firestore composite index, so a failure here
    is logged and treated as "nothing to resume" rather than surfaced to the caller.
    """
    try:
        recent = await get_session_service().list_user_sessions(user_id, limit=5)
    except Exception as e:
        log.warning("chat_session_resume_lookup_failed", user_id=user_id, error=str(e))
        return None

    cutoff = datetime.now(timezone.utc) - _RESUME_WINDOW
    for session in recent:
        if session.persona != persona:
            continue
        updated = session.updated_at
        if updated.tzinfo is None:
            updated = updated.replace(tzinfo=timezone.utc)
        if updated >= cutoff:
            return session
    return None


async def start_chat(user_id: str, persona: str) -> TripSession:
    """Open the chat dialog: ALWAYS begins a fresh conversation.

    Called when the user opens the chat UI, so every open starts from a clean greeting
    rather than dropping them into a half-finished plan from days ago.
    """
    return await _create_session(persona, user_id)


async def resolve_chat(
    user_id: str,
    persona: str,
    session_id: Optional[str] = None,
) -> TripSession:
    """Return the caller's current conversation, creating one if there isn't a live one.

    Args:
        session_id: optional explicit pin. When the caller supplies one they own, that
            conversation is used and becomes their active one. An unknown id, or one
            owned by someone else, is ignored and resolution falls through to the
            normal tiers — never a 4xx, so a stale id on the client can't wedge the UI.

    Never raises for a missing session — a front end that sends a message without
    having called /start still gets a working conversation.
    """
    if session_id:
        pinned = await _load_owned_session(session_id, user_id)
        if pinned is not None:
            await _remember_active(persona, user_id, pinned.session_id)
            return pinned
        log.info(
            "chat_session_pin_ignored",
            session_id=session_id,
            user_id=user_id,
            reason="unknown or not owned by caller",
        )

    active_id = await _recall_active(persona, user_id)
    if active_id:
        session = await _load_owned_session(active_id, user_id)
        if session is not None:
            return session

    resumable = await _find_resumable(persona, user_id)
    if resumable is not None:
        await _remember_active(persona, user_id, resumable.session_id)
        log.info(
            "chat_session_resumed",
            session_id=resumable.session_id,
            user_id=user_id,
            persona=persona,
        )
        return resumable

    return await _create_session(persona, user_id)
