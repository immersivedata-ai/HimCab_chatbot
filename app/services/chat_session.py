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

from app.core.enums import Persona
from app.core.logging import get_logger
from app.models.location import RiderLocation
from app.models.session import Language, MessageRole, TripSession
from app.services.firestore import get_session_service

log = get_logger(__name__)

# Personas = which front-end app opened the chat. Fixed by the route, never sent by
# the client. Durably stored on the session (TripSession.persona).
#
# Aliases onto app/core/enums.Persona — the same objects the agent compares against.
# These used to be three independent string literals here, three more in
# app/agent/slots.py and two in the planner, with nothing checking they matched.
LOCAL = Persona.LOCAL
OUTSTATION = Persona.OUTSTATION
#: Hiring a car by the hour. A SERVICE the rider chose, not a guess about who they
#: are — which is why a GPS fix never overrides it (see `resolve_persona`).
RENTAL = Persona.RENTAL
VALID_PERSONAS = tuple(Persona)

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
    RENTAL: (
        "Welcome to Him Cab! 🙏 "
        "I can arrange a car and driver by the hour. "
        "How long do you need the car for?"
    ),
}

#: The rental greeting, minus its closing question. The question is swapped for one
#: that names the real packages whenever the rate card is readable — see
#: `welcome_message`. Split here so the greeting is written once.
_RENTAL_GREETING = (
    "Welcome to Him Cab! 🙏 I can arrange a car and driver by the hour. "
)

# Each front end has a default reply language; the agent still adapts to whatever the
# user actually writes (see app/agent/language.py).
DEFAULT_LANGUAGE: dict[str, Language] = {
    LOCAL: Language.HINDI,
    OUTSTATION: Language.ENGLISH,
    # Rentals are taken by residents and visitors alike, so there is no safe default
    # from the service itself — English, then `app/agent/language.py` follows whatever
    # the rider actually writes from their first message onward.
    RENTAL: Language.ENGLISH,
}


def welcome_message(
    persona: str, rental_packages: Optional[list[dict]] = None
) -> str:
    """The opening line for a persona (falls back to the outstation/English copy).

    A rental greeting names the packages the rate card sells when they are known. It is
    the very first thing the rider reads AND it asks for the one field the backend
    matches exactly, so leaving it open invites an unbookable answer before the
    conversation has started. With no packages it asks the open question, unchanged.
    """
    if persona == RENTAL and rental_packages:
        from app.agent import slots

        # English to match the greeting it is appended to; the agent switches to
        # whatever the rider actually writes from their first message onward.
        return _RENTAL_GREETING + slots.rental_hours_question(
            [p["label"] for p in rental_packages], "en"
        )
    return WELCOME_MESSAGES.get(persona, WELCOME_MESSAGES[OUTSTATION])


# ── Which assistant is this rider talking to ──────────────────────────────────

def resolve_persona(
    requested: str, location: Optional[RiderLocation]
) -> tuple[str, str]:
    """Pick the persona to serve, returning (persona, why).

    The route the app called is a REQUEST, not the answer. It is chosen by which screen
    the chat bubble was tapped on — a Delhi visitor browsing the home screen lands on
    /local and gets greeted in Hindi about a state they have never been to, and a Shimla
    resident opening the outstation screen gets an itinerary for the town they live in.

    A GPS fix answers the question directly, so when there is one it wins:

        inside Himachal  → local       (Hindi, cab-first: they know the place)
        outside          → outstation  (English, itinerary-first: they are visiting)

    With no usable fix the requested persona stands, which is exactly the old behaviour
    — this never makes a client that sends no coordinates worse off.

    Note what this does NOT do: override the planner. Once the rider names a starting
    city, `planner.classify_visitor_type` reclassifies from their own words, and a
    person who says "I'm coming from Delhi" is a visitor whatever their phone says. The
    fix is the opening assumption; the conversation is the correction.
    """
    # A rental is a service the rider picked, not an inference about them. Their
    # coordinates say whether they live here; they do not say they stopped wanting a
    # car by the hour, so the fix never overrides this one.
    if requested == RENTAL:
        return RENTAL, "requested"
    if location is None:
        return (requested if requested in VALID_PERSONAS else OUTSTATION), "default"
    return location.visitor_type, "coordinates"


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


def apply_location(session: TripSession, location: Optional[RiderLocation]) -> bool:
    """Attach a fresh GPS fix to a conversation. True when anything changed.

    Beyond storing the fix, this fills in the pickup city — but ONLY when the fix
    actually sits on a town we know (`is_precise`). A fix 40 km from the nearest named
    place is perfectly good for measuring a distance and useless as a place name: it
    would have the reply announce "your trip from Mandi" to somebody who has never been
    there. In that case the coordinates are still used for the route, and the assistant
    asks which city they are starting from, as it always did.

    A rider who has already NAMED their starting city keeps it. Their words outrank
    their phone: someone booking a Delhi pickup from their Shimla sofa means Delhi.

    What this must NEVER do is re-persona a rental. See the visitor_type block below.
    """
    if location is None:
        return False

    # A rider sitting still resends the same fix every turn. Reporting that as a change
    # would mean a Firestore write per message for nothing.
    previous = session.rider_location
    changed = previous is None or (previous.lat, previous.lon) != (location.lat, location.lon)

    session.rider_location = location

    if location.place_name and location.is_precise and not session.entities.origin:
        session.entities.origin = location.place_name
        if location.in_himachal:
            session.entities.home_city = location.place_name
        # The origin question is answered, so it must not be the question in flight —
        # otherwise the next reply gets read as an answer to a question nobody asked.
        if session.entities.last_asked_field == "origin":
            session.entities.last_asked_field = None
        changed = True

    # A fix says "local" or "outstation" — a guess about WHO the rider is, which is
    # exactly what coordinates are good for. It says nothing about a RENTAL, which is a
    # service they chose, and overwriting it here was the single worst bug this flow has
    # had: the rental screen sends a GPS fix, so from the very first turn the rider was
    # served as a local or a visitor. `needs_for` fell back to the trip checklist and
    # asked where they wanted to GO and for how many DAYS; `response_mode` produced a
    # cab quote or an itinerary; the reply had nothing to do with hiring a car. Every
    # other guard against this (`resolve_persona`, `classify_visitor_type`,
    # `needs_for`, `response_mode`) reads a value this line had already destroyed.
    is_rental = RENTAL in (session.persona, session.entities.visitor_type)
    if not is_rental and session.entities.visitor_type != location.visitor_type:
        session.entities.visitor_type = location.visitor_type
        changed = True

    return changed


async def _create_session(
    persona: str,
    user_id: str,
    location: Optional[RiderLocation] = None,
    rental_packages: Optional[list[dict]] = None,
) -> TripSession:
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
    # The welcome asks the first question, so record which one — that is what lets a
    # bare "8 hours" or "manali" be read as the answer to it.
    session.entities.last_asked_field = (
        "rental_hours" if persona == RENTAL else "destination"
    )
    # Applied after the seed so a GPS fix wins over the persona for visitor_type, and
    # so the pickup city is known before the rider's first message.
    apply_location(session, location)
    session.add_message(
        MessageRole.ASSISTANT,
        welcome_message(persona, rental_packages),
        language=language,
    )

    await get_session_service().create_session(session)
    await _remember_active(persona, user_id, session.session_id)

    log.info(
        "chat_session_started",
        session_id=session.session_id,
        user_id=user_id,
        persona=persona,
        located=location is not None,
        origin=session.entities.origin,
    )
    return session


async def _adopt_conversation(
    persona: str, adopt_from: str, user_id: str
) -> Optional[TripSession]:
    """Carry a live conversation across when coordinates change which persona serves it.

    Without this, a rider whose fix moves them from outstation to local (because they
    have ARRIVED in Himachal — the single most likely way for this to happen) is dropped
    into an empty chat, having already told the assistant where they are going, for how
    long and with how many people. The persona is a routing decision; it should not cost
    them the conversation.

    Only ever adopts when the target persona has nothing of its own, so a rider who
    genuinely uses both screens keeps their two separate conversations.
    """
    if adopt_from == persona:
        return None

    previous_id = await _recall_active(adopt_from, user_id)
    if not previous_id:
        return None
    session = await _load_owned_session(previous_id, user_id)
    if session is None:
        return None

    session.persona = persona
    # The persona's default reply language moves with it. What the agent actually
    # writes in is decided per turn from the rider's own words (app/agent/language.py),
    # but this is what /history reports and what a resumed conversation starts from.
    session.language = DEFAULT_LANGUAGE.get(persona, Language.ENGLISH)
    await _remember_active(persona, user_id, session.session_id)
    try:
        await get_session_service().update_persona(
            session_id=session.session_id,
            persona=persona,
            language=str(getattr(session.language, "value", session.language)),
        )
    except Exception as e:
        # In-memory adoption still serves this turn; the stored document catches up on
        # the next write. Logged because a document left under the old persona is
        # unresumable if the pointer is ever lost.
        log.warning(
            "chat_session_persona_persist_failed",
            session_id=session.session_id,
            error=str(e),
        )
    log.info(
        "chat_session_adopted_across_personas",
        session_id=session.session_id,
        user_id=user_id,
        from_persona=adopt_from,
        to_persona=persona,
        messages=len(session.messages),
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


async def _refresh_location(
    session: TripSession, location: Optional[RiderLocation]
) -> TripSession:
    """Update an existing conversation with the fix that arrived on this request.

    The rider moves between turns — they open the chat at home and send the next
    message from the taxi rank — so the newest fix always wins. Persisting it is
    best-effort: a Firestore hiccup costs the next turn its head start, never this
    turn's reply.
    """
    if not apply_location(session, location):
        return session
    try:
        await get_session_service().update_location(
            session_id=session.session_id,
            location=session.rider_location.model_dump(mode="json"),
            entities=session.entities.model_dump(mode="json"),
        )
    except Exception as e:
        log.warning(
            "chat_session_location_persist_failed",
            session_id=session.session_id,
            error=str(e),
        )
    return session


async def start_chat(
    user_id: str,
    persona: str,
    location: Optional[RiderLocation] = None,
    rental_packages: Optional[list[dict]] = None,
) -> TripSession:
    """Open the chat dialog: ALWAYS begins a fresh conversation.

    Called when the user opens the chat UI, so every open starts from a clean greeting
    rather than dropping them into a half-finished plan from days ago.

    `rental_packages` is passed in rather than fetched here so the caller can send the
    same list to the client as a `rental_packages` event — the greeting and the chips
    beneath it are then guaranteed to offer the same durations.
    """
    return await _create_session(persona, user_id, location, rental_packages)


async def resolve_chat(
    user_id: str,
    persona: str,
    session_id: Optional[str] = None,
    location: Optional[RiderLocation] = None,
    adopt_from: Optional[str] = None,
) -> TripSession:
    """Return the caller's current conversation, creating one if there isn't a live one.

    Args:
        session_id: optional explicit pin. When the caller supplies one they own, that
            conversation is used and becomes their active one. An unknown id, or one
            owned by someone else, is ignored and resolution falls through to the
            normal tiers — never a 4xx, so a stale id on the client can't wedge the UI.
        location: the device's current fix, if it sent one. Written onto whichever
            conversation is resolved, so the pickup point follows the rider.
        adopt_from: the persona the caller ASKED for, when coordinates chose a different
            one. A live conversation there is carried over rather than abandoned —
            see `_adopt_conversation`.

    Never raises for a missing session — a front end that sends a message without
    having called /start still gets a working conversation.
    """
    if session_id:
        pinned = await _load_owned_session(session_id, user_id)
        if pinned is not None:
            await _remember_active(persona, user_id, pinned.session_id)
            return await _refresh_location(pinned, location)
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
            return await _refresh_location(session, location)

    resumable = await _find_resumable(persona, user_id)
    if resumable is not None:
        await _remember_active(persona, user_id, resumable.session_id)
        log.info(
            "chat_session_resumed",
            session_id=resumable.session_id,
            user_id=user_id,
            persona=persona,
        )
        return await _refresh_location(resumable, location)

    if adopt_from:
        adopted = await _adopt_conversation(persona, adopt_from, user_id)
        if adopted is not None:
            return await _refresh_location(adopted, location)

    return await _create_session(persona, user_id, location)
