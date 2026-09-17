"""
app/api/v1/routes/chat.py
──────────────────────────
The chat API consumed by the mobile app.

╔══════════════════════════════════════════════════════════════════════════════════╗
║  FRONT-END CONTRACT — this is everything the app team needs to know.              ║
║                                                                                  ║
║  Send ONE header on every call:                                                  ║
║      Authorization: Bearer <Him Cab rider JWT>                                   ║
║                                                                                  ║
║  There is no persona or rider id to pass. The backend owns the conversation and   ║
║  resolves it from the token, so multi-turn context works with no client state.    ║
║                                                                                  ║
║  Pick the set that matches what the rider wants:                                 ║
║      Himachal local →  /v1/chat/local/*        cab-first replies, Hindi default  ║
║      visiting       →  /v1/chat/outstation/*   day-by-day itineraries, English   ║
║      car by the hour → /v1/chat/rental/*       hourly packages and a fare        ║
║                                                                                  ║
║  /rental is chosen by the RIDER (your Rental tab), never by their location: a    ║
║  GPS fix reroutes /local and /outstation, but never a rental.                    ║
║                                                                                  ║
║  0. Which pair? Ask, with the device's GPS fix:                                  ║
║       POST /v1/chat/route   body: {"lat": 31.10, "lon": 77.17}                   ║
║       → {"persona": "local", "base_path": "/v1/chat/local", "location": {...}}   ║
║       A fix inside Himachal is a local; anywhere else is a visitor. Skip this    ║
║       and the screen the chat was opened from decides, as it used to.            ║
║                                                                                  ║
║  1. User opens the chat dialog:                                                  ║
║       POST /v1/chat/{local|outstation}/start?lat=&lon=                           ║
║       → {"session_id", "message", "language", "persona", "location"}             ║
║       Send lat/lon here too and the reply's `persona` is authoritative — it may  ║
║       differ from the path you called, and if it does, use it from then on.      ║
║                                                                                  ║
║     DO NOT WAIT FOR THIS TO DRAW YOUR FIRST BUBBLE. /start reverse-geocodes the  ║
║     fix, reads the rental rate card and writes a session before it can answer,   ║
║     and none of that changes the greeting TEXT. Ship a copy of the greeting in   ║
║     the app, render it the moment the screen opens, and let /start finish in the ║
║     background — then replace the text only if what comes back differs. For      ║
║     local and outstation it will not, so nothing flickers; for rental it will,   ║
║     because /start names the real hourly packages.                               ║
║                                                                                  ║
║       GET /v1/chat/{local|outstation|rental}/greeting   ← no auth, no I/O        ║
║       → {"persona", "message", "language", "is_final"}                           ║
║       The same strings, so your bundled copy can be checked against them. Only   ║
║       your own copy survives a cold start, when no endpoint answers at all.      ║
║                                                                                  ║
║  2. User sends a message (streaming — preferred):                                ║
║       GET /v1/chat/{local|outstation}/stream?message=<text>&lat=&lon=            ║
║       → text/event-stream: token | structured | done | error                     ║
║       Render each `token` as it lands. Waiting for the whole reply is what makes ║
║       a fast backend feel slow.                                                  ║
║                                                                                  ║
║     …or non-streaming, if SSE is inconvenient:                                   ║
║       POST /v1/chat/{local|outstation}/message                                   ║
║         body: {"message": "<text>", "lat": 31.10, "lon": 77.17}                  ║
║       → full JSON reply                                                          ║
║                                                                                  ║
║  That's the whole integration. Step 2 works even if steps 0-1 were skipped —     ║
║  the backend creates the conversation on demand, so these endpoints never 404    ║
║  with a "session expired" the app has to recover from.                           ║
║                                                                                  ║
║  COORDINATES — optional everywhere, worth sending everywhere:                    ║
║    lat/lon are what let the assistant skip two questions it otherwise has to     ║
║    ask ("local or visiting?", "which city are you starting from?") and price the ║
║    trip from where the rider actually is rather than a city centroid. Resend     ║
║    them on every turn; the newest fix wins, because riders move between turns.   ║
║    Send nothing and everything still works — the assistant just asks.            ║
║                                                                                  ║
║  OPTIONAL — pinning a conversation:                                              ║
║    Both /stream and /message accept a `session_id` (query param and body field    ║
║    respectively). Omit it and the token decides, which is right for a single      ║
║    device. Pass the id from /start when one account may be signed in on several   ║
║    devices and you don't want them sharing one conversation. An unknown or        ║
║    unowned id is ignored — it never 4xxs, so a stale client id can't wedge the UI.║
╚══════════════════════════════════════════════════════════════════════════════════╝

SSE events emitted by /stream:
  1. `token`      → raw AI text chunks (for chat bubble animation)
  2. `structured` → a renderable card. Types: itinerary | weather_alert |
                    snowfall_alert | cab_fare | offers | booking_handoff |
                    rental_packages | request_cancelled | flow_options. See
                    `StructuredEvent` in app/schemas/chat.py for each payload shape.
                    Ignore unknown types.

BEFORE A FLOW IS CHOSEN — the in-chat service picker (replaces the persona dropdown):
  GET  /v1/chat/flows          → the neutral greeting + selectable flow boxes (no auth)
  POST /v1/chat/select {flow}  → start the tapped flow; returns a StartChatResponse
  The boxes carry the persona each one starts; /select is the same as calling that
  persona's /start, so the rider chooses the flow instead of the screen deciding it.
  3. `done`       → completion; carries the `persona` this turn was served as
  4. `error`      → failure

CANCELLATION — the one event you must ACT on, not just render:
  When the rider backs out ("cancel", "never mind", "rehne do"), the reply carries a
  `request_cancelled` event. Dismiss any booking sheet a previous `booking_handoff`
  opened and clear this conversation's fare/package cards: what they describe no longer
  exists, and a live "Book" button above a cancellation is how somebody books the thing
  they just cancelled.

  It never cancels anything on the backend. A CONFIRMED booking is cancelled only from
  your Trips screen (`DELETE /api/rider/trip/cancel-ride/{id}`), which is what notifies
  the driver — the assistant has no trip id and says so. `payload.open_screen` is
  "trips" when the rider needs to be sent there.

Rate limit: 30 requests/minute per rider. Over that, both endpoints return HTTP 429
with `{"error": {"code": "RATE_LIMIT_EXCEEDED", ...}}` and a `Retry-After` header in
seconds. Retry after that delay; do not treat 429 as a fatal error.
"""

import asyncio
import json
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Query
from langchain_core.messages import AIMessage, HumanMessage
from sse_starlette.sse import EventSourceResponse

from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.api.middleware.rider_auth import RiderIdentity, get_rider
from app.core.config import settings
from app.core.exceptions import RateLimitExceededError
from app.core.logging import get_logger
from app.domain.conversation import MessageRole, TripSession
from app.domain.location import RiderLocation
from app.integrations.himcab import available_packages
from app.repositories.firestore import get_session_service
from app.schemas.chat import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatResponse,
    FlowChooserResponse,
    FlowOption,
    FlowSelectionRequest,
    GreetingResponse,
    PersonaRouteResponse,
    ResolvedLocation,
    RiderCoordinates,
    StartChatResponse,
)
from app.services import geo
from app.services.chat_session import (
    DEFAULT_LANGUAGE,
    LOCAL,
    OUTSTATION,
    RENTAL,
    chooser_message,
    flow_options,
    resolve_chat,
    resolve_persona,
    start_chat,
    welcome_message,
)
from app.services.local_cache import RateLimiter

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

#: The rental endpoints live on their own router purely so Swagger UI gives them their
#: own section — the handlers are the same three helpers the other two personas use.
#: The paths are unchanged (/v1/chat/rental/*); only the OpenAPI tag differs, which is
#: what makes /docs#/Rental a link you can hand to another team. Mounted alongside
#: `router` in app/api/v1/router.py, NOT inside it — a nested include would stack the
#: parent's "Chat" tag on top and list every rental route twice.
rental_router = APIRouter(prefix="/chat/rental", tags=["Rental"])

#: Per-rider request budget. In-process — see app/services/local_cache.py for what
#: that means on a multi-instance deployment.
_rate_limiter = RateLimiter(limit=settings.rate_limit_per_minute, window_seconds=60)

_SESSION_ID_DESCRIPTION = (
    "Optional. Omit to let the backend resolve the conversation from your Bearer "
    "token. Pass the session_id from /start to pin a specific conversation — needed "
    "only when one account is signed in on several devices. Unknown or unowned ids "
    "are ignored, not rejected."
)

_LAT_DESCRIPTION = (
    "Optional device latitude. Sent with `lon`, it becomes the pickup point: the fare "
    "is priced from here, and a fix inside Himachal serves the LOCAL assistant "
    "whichever path you called. Resend it every turn — the newest fix wins."
)
_LON_DESCRIPTION = "Optional device longitude. See `lat`."


def _resolved_location_dto(location: RiderLocation | None) -> ResolvedLocation | None:
    """The client-facing view of a fix: what we made of it, and how sure we are."""
    if location is None:
        return None
    return ResolvedLocation(
        lat=location.lat,
        lon=location.lon,
        in_himachal=location.in_himachal,
        place_name=location.place_name,
        place_distance_km=location.place_distance_km,
        state=location.state,
    )


def _effective_persona(
    requested: str, location: RiderLocation | None, user_id: str
) -> str:
    """The persona to actually serve, given what the device says about where they are.

    Coordinates outrank the route the app called, because the route is chosen by which
    screen the chat bubble was tapped on and the coordinates are chosen by physics. A
    visitor browsing the home screen would otherwise be greeted in Hindi and handed a
    resident's cab-first reply about a state they have never driven in.
    """
    persona, reason = resolve_persona(requested, location)
    if persona != requested:
        log.info(
            "chat_persona_overridden",
            requested=requested,
            served=persona,
            reason=reason,
            state=location.state if location else None,
            user_id=user_id,
        )
    return persona


def _build_history_messages(messages: list, limit: int = 20) -> list:
    """
    Convert Firestore message models to LangChain message objects.
    Keeps only the last `limit` messages for context window management.
    """
    history = []
    for m in messages[-limit:]:
        if m.role == "user":
            history.append(HumanMessage(content=m.content))
        elif m.role == "assistant":
            history.append(AIMessage(content=m.content))
    return history


def _build_state(
    session: TripSession,
    rider: RiderIdentity,
    message: str,
    persona: str,
    token_queue: asyncio.Queue | None = None,
) -> AgentState:
    """Assemble the agent's starting state for one turn."""
    history_messages = _build_history_messages(session.messages)
    history_messages.append(HumanMessage(content=message))

    return AgentState(
        messages=history_messages,
        session_id=session.session_id,
        user_id=rider.user_id,
        language=session.language,
        entities=session.entities,
        # The session's own classification wins once the planner has confirmed one
        # from a real origin city; otherwise fall back to the front end's persona.
        visitor_type=session.entities.visitor_type or persona,
        # Carry forward last_asked_field so the planner knows what was asked last
        # turn and can skip it if the answer was unparseable (prevents re-asking).
        last_asked_field=session.entities.last_asked_field,
        rider_jwt=rider.jwt,
        rider_id=rider.rider_id,
        # Where they are, if the app has told us — used for the route, the fare and
        # the pickup pin. Already merged with any fix sent on THIS request.
        rider_location=session.rider_location,
        token_stream=token_queue,
    )


def _enforce_rate_limit(user_id: str) -> None:
    """30 requests/minute per rider, fixed 60s window.

    Raises 429 with a `Retry-After` header. The window is a whole minute, so the
    worst-case wait is 60s — we send that rather than tracking per-key expiry.
    """
    allowed, count = _rate_limiter.check(user_id)
    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit exceeded: {count}/{_rate_limiter.limit} requests per minute. "
            "Please wait a moment before sending another message.",
            retry_after=60,
            limit=_rate_limiter.limit,
        )


async def _stream_agent_response(
    session: TripSession,
    rider: RiderIdentity,
    message: str,
    persona: str,
):
    """
    Core generator — runs the LangGraph agent and yields SSE events.

    Yielded format (sse-starlette compatible):
      {"event": "token", "data": '{"token": "..."}'}
      {"event": "structured", "data": '{"type": "itinerary", "payload": {...}}'}
      {"event": "done", "data": '{"session_id": "..."}'}
    """
    graph = get_agent_graph()
    session_svc = get_session_service()
    session_id = session.session_id

    # Build the agent state from the pre-message snapshot FIRST, then persist. Doing it
    # in this order keeps the turn correct whether or not the store mutates `session`
    # in place — otherwise the new message can end up in the history twice.
    token_queue: asyncio.Queue = asyncio.Queue()
    initial_state = _build_state(session, rider, message, persona, token_queue)

    await session_svc.append_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=message,
    )

    try:
        structlog.contextvars.bind_contextvars(
            session_id=session_id,
            user_id=rider.user_id,
            trace_id=str(uuid4()),
        )

        # Run the graph concurrently; the Writer pushes tokens onto token_queue as it
        # generates. We drain them here and forward each as a `token` SSE event.
        graph_task = asyncio.create_task(graph.ainvoke(initial_state))

        streamed_any = False
        while True:
            getter = asyncio.ensure_future(token_queue.get())
            done, _ = await asyncio.wait(
                {getter, graph_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if getter in done:
                token = getter.result()
                if token is None:      # end-of-stream sentinel from the Writer
                    break
                streamed_any = True
                yield {"event": "token", "data": json.dumps({"token": token})}
            else:
                # Graph finished without streaming (clarify path / no Writer node).
                getter.cancel()
                break

        final_state: dict = await graph_task  # re-raises if the graph errored

        # Fallback: paths that don't stream (e.g. clarification) — chunk the final text
        # so the client still gets a token-by-token feel.
        if not streamed_any:
            response_text = final_state.get("final_response_text", "")
            words = response_text.split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                yield {"event": "token", "data": json.dumps({"token": chunk})}
                await asyncio.sleep(0.02)

        # ── Emit structured events ────────────────────────────────────────────
        for event in final_state.get("structured_events", []):
            yield {
                "event": "structured",
                "data": json.dumps(event),
            }
            await asyncio.sleep(0.1)

        # ── Done ──────────────────────────────────────────────────────────────
        yield {
            "event": "done",
            "data": json.dumps({
                "session_id": session_id,
                # The persona this turn actually ran as. Differs from the path only
                # when coordinates said otherwise — switch the client to it if so.
                "persona": persona,
                "node_trace": final_state.get("node_trace", []),
            }),
        }

        log.info(
            "chat_stream_completed",
            session_id=session_id,
            persona=persona,
            nodes=final_state.get("node_trace", []),
            has_itinerary=final_state.get("structured_itinerary") is not None,
        )

    except Exception as e:
        log.error(
            "chat_stream_error",
            error=str(e),
            error_type=type(e).__name__,
            session_id=session_id,
        )
        yield {
            "event": "error",
            "data": json.dumps({
                "code": "AGENT_ERROR",
                "message": "The AI assistant encountered an error. Please try again.",
            }),
        }
    finally:
        # If the client disconnected or an error aborted the drain loop, don't leave
        # the graph running in the background.
        task = locals().get("graph_task")
        if task is not None and not task.done():
            task.cancel()
        structlog.contextvars.clear_contextvars()


# ── Shared handlers (persona-agnostic; the routes below just pin the persona) ─

async def _start_persona(
    rider: RiderIdentity,
    served: str,
    location: RiderLocation | None,
) -> StartChatResponse:
    """Open a fresh conversation for an ALREADY-RESOLVED persona and build the response.

    Shared by `/start` (where the persona comes from the path, refined by GPS) and
    `/select` (where it comes from the box the rider tapped). Both arrive here with the
    persona already decided, so the greeting, the session and the rental packages are
    assembled one way.
    """
    # The rental greeting asks for a duration, and a duration is matched exactly by the
    # backend — so it must name what is on the rate card, and the client needs the same
    # list to render as taps. One fetch feeds both.
    packages = await available_packages(rider.jwt) if served == RENTAL else []

    session = await start_chat(rider.user_id, served, location, packages)
    # The greeting is already stored on the session by start_chat().
    greeting = session.messages[-1].content if session.messages else ""
    return StartChatResponse(
        session_id=session.session_id,
        message=greeting,
        language=str(session.language),
        persona=served,
        location=_resolved_location_dto(session.rider_location),
        structured_events=(
            [{"type": "rental_packages", "payload": {"packages": packages}}]
            if packages
            else []
        ),
    )


async def _handle_start(
    rider: RiderIdentity,
    persona: str,
    lat: float | None = None,
    lon: float | None = None,
    accuracy_m: float | None = None,
) -> StartChatResponse:
    location = await geo.resolve_rider_location(lat, lon, accuracy_m)
    served = _effective_persona(persona, location, rider.user_id)
    return await _start_persona(rider, served, location)


async def _handle_stream(
    rider: RiderIdentity,
    persona: str,
    message: str,
    session_id: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    accuracy_m: float | None = None,
) -> EventSourceResponse:
    _enforce_rate_limit(rider.user_id)
    location = await geo.resolve_rider_location(lat, lon, accuracy_m)
    served = _effective_persona(persona, location, rider.user_id)
    session = await resolve_chat(
        rider.user_id,
        served,
        session_id=session_id,
        location=location,
        adopt_from=persona,
    )
    return EventSourceResponse(
        _stream_agent_response(session, rider, message, served),
        media_type="text/event-stream",
    )


def _itinerary_payload(final_state: dict) -> dict | None:
    """The itinerary for this turn, or None if it didn't produce one.

    Sourced from the same `structured_events` entry the app renders, so the top-level
    `itinerary` field and the event payload can never disagree — they are one object.
    """
    for event in final_state.get("structured_events", []):
        if event.get("type") == "itinerary":
            payload = event.get("payload")
            return payload if isinstance(payload, dict) else None
    return None


async def _handle_message(
    rider: RiderIdentity, persona: str, body: ChatMessageRequest
) -> ChatResponse:
    _enforce_rate_limit(rider.user_id)
    location = await geo.resolve_rider_location(body.lat, body.lon, body.accuracy_m)
    served = _effective_persona(persona, location, rider.user_id)
    session = await resolve_chat(
        rider.user_id,
        served,
        session_id=body.session_id,
        location=location,
        adopt_from=persona,
    )

    # Snapshot-then-persist, for the same reason as the streaming path above.
    initial_state = _build_state(session, rider, body.message, served)
    await get_session_service().append_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content=body.message,
    )

    final_state: dict = await get_agent_graph().ainvoke(initial_state)

    return ChatResponse(
        session_id=session.session_id,
        response=final_state.get("final_response_text", ""),
        persona=served,
        structured_events=final_state.get("structured_events", []),
        itinerary=_itinerary_payload(final_state),
        node_trace=final_state.get("node_trace", []),
    )


async def _handle_history(
    rider: RiderIdentity, persona: str, limit: int, session_id: str | None
) -> ChatHistoryResponse:
    """Replay the caller's current conversation so the app can rehydrate its UI.

    Resolves the same conversation /message would, so the client can restore a chat
    without ever having stored a session id. Returns an empty list rather than 404
    when there's nothing yet — "no history" is a normal state, not an error.
    """
    session = await resolve_chat(rider.user_id, persona, session_id=session_id)
    messages = session.messages[-limit:] if limit > 0 else session.messages
    return ChatHistoryResponse(
        session_id=session.session_id,
        persona=persona,
        language=str(session.language),
        messages=[
            ChatHistoryMessage(
                role=str(getattr(m.role, "value", m.role)),
                content=m.content,
                timestamp=m.timestamp,
            )
            for m in messages
        ],
        total_messages=len(session.messages),
    )


# ── Persona routing (call this first) ─────────────────────────────────────────

# ── The opening bubble, with nothing in front of it ───────────────────────────

def _greeting(persona: str) -> GreetingResponse:
    """Build the greeting from constants. No Firestore, no geocoder, no rate card.

    `/start` cannot be fast: it reverse-geocodes the fix, reads the rental rate card
    (with retries, if the Him Cab backend is slow) and writes a session to Firestore
    before it can answer. None of that work changes the TEXT, so an app that waits for
    `/start` to render its first bubble is waiting on three round trips for a string
    that was already known.

    Rental is the one persona whose greeting improves later: `/start` names the real
    hourly packages once the rate card has been read, which is a better question than
    the open one here. `is_final` says which case the caller is in, so the app knows
    whether to expect a replacement or leave the bubble alone.
    """
    return GreetingResponse(
        persona=persona,
        message=welcome_message(persona),
        # `.value`, not `str()`: DEFAULT_LANGUAGE holds real enum members, and
        # `str(Language.HINDI)` is "Language.HINDI". Elsewhere the value comes off a
        # TripSession, which stores enums by value already, so `str()` is right there
        # and wrong here.
        language=DEFAULT_LANGUAGE.get(persona, DEFAULT_LANGUAGE[OUTSTATION]).value,
        is_final=persona != RENTAL,
    )


#: Deliberately unauthenticated, unlike everything else under /v1/chat. The response is
#: one constant string, identical for every rider and carrying nothing personal — and
#: the entire point is that it answers before the app has done anything at all, which
#: includes before it has a token to send. Requiring one would put the thing the app
#: needs first behind the thing it gets last.
_GREETING_DESCRIPTION = """\
The opening assistant bubble for this persona, with no work behind it — no session is
created, nothing is written, and no backend is called. Needs no `Authorization` header.

**Render your own bundled copy of this string at launch rather than calling this.** A
cold start blocks every endpoint including this one, so only text the app already has
can appear while the container is still booting. Call this in the background to check
the copy is current, and to pick up wording changes without shipping a release.
"""


@router.get(
    "/local/greeting",
    response_model=GreetingResponse,
    description=_GREETING_DESCRIPTION,
)
async def greeting_local() -> GreetingResponse:
    """The LOCAL opening line. Hindi, and final — `/start` returns the same text."""
    return _greeting(LOCAL)


@router.get(
    "/outstation/greeting",
    response_model=GreetingResponse,
    description=_GREETING_DESCRIPTION,
)
async def greeting_outstation() -> GreetingResponse:
    """The OUTSTATION opening line. English, and final — `/start` returns the same text."""
    return _greeting(OUTSTATION)


@router.post("/route", response_model=PersonaRouteResponse)
async def route_by_location(
    coordinates: RiderCoordinates,
    rider: RiderIdentity = Depends(get_rider),
):
    """Which assistant does this rider belong to? Ask before opening the chat.

    Send the device's GPS fix and use the `persona` that comes back for every
    /v1/chat/{persona}/* call afterwards:

      • inside Himachal Pradesh → **local** — Hindi, cab-first. They know the place;
        what they want is a ride, a price and a drive time.
      • anywhere else          → **outstation** — English, itinerary-first. They are
        planning several days in a state they do not know.

    Nothing is created here: no session, no message, no state. It is a pure lookup, so
    it is safe to call on app start, on a location change, or not at all — the chat
    endpoints accept the same coordinates directly and will route themselves.

    `location.state` carries the administrative state the fix resolved to, which is the
    evidence behind the call. The rider's own words still win over it: someone who says
    "I'm coming from Delhi" is treated as a visitor whatever their phone reports.
    """
    location = await geo.resolve_rider_location(
        coordinates.lat, coordinates.lon, coordinates.accuracy_m
    )
    persona, reason = resolve_persona(OUTSTATION, location)
    log.info(
        "chat_persona_routed",
        persona=persona,
        reason=reason,
        in_himachal=bool(location and location.in_himachal),
        state=location.state if location else None,
        user_id=rider.user_id,
    )
    return PersonaRouteResponse(
        persona=persona,
        base_path=f"/v1/chat/{persona}",
        reason=reason,
        location=_resolved_location_dto(location),
    )


# ── The flow chooser (the in-chat replacement for the persona dropdown) ───────

_FLOWS_DESCRIPTION = """\
The selectable services to show before a flow is chosen — the replacement for the old
persona dropdown. Render `message` as the opening assistant bubble and `flows` as
tappable boxes; `structured_events` carries the same list as a `flow_options` card for
card-based rendering.

Needs no `Authorization` header and does no I/O — it is a constant, like `/greeting`, so
it can appear the instant the chat opens. When the rider taps a box, call
`POST /v1/chat/select` with `flow` = that box's `persona`; the backend starts that flow
and returns its greeting, exactly as `/start` does.

Pass `?lang=hi` for the Hindi copy; anything else is served in English.
"""

_SELECT_DESCRIPTION = """\
The rider tapped a flow box — start that flow and return its greeting.

Body: `{"flow": "local" | "outstation" | "rental", "lat"?, "lon"?, "accuracy_m"?}`. The
response is a `StartChatResponse`, identical to `POST /v1/chat/{flow}/start` — this is
the single "the rider chose a service" call, so the app does not have to build a
per-persona path. Coordinates behave exactly as on `/start`: they set the pickup point,
and for `local`/`outstation` a fix decides local-vs-visiting (a Delhi fix on a `local`
selection is served as `outstation`, and the response's `persona` says so). `rental` is
never re-routed by a fix.
"""


def _flow_chooser(lang_key: str) -> FlowChooserResponse:
    """Build the neutral chooser from constants — no auth, no I/O (see `/greeting`)."""
    flows = flow_options(lang_key)
    return FlowChooserResponse(
        message=chooser_message(lang_key),
        language=lang_key,
        flows=[FlowOption(**flow) for flow in flows],
        # The same list as a card, so a client can render it through whatever already
        # handles structured events rather than special-casing this response.
        structured_events=[{"type": "flow_options", "payload": {"flows": flows}}],
    )


@router.get("/flows", response_model=FlowChooserResponse, description=_FLOWS_DESCRIPTION)
async def chat_flows(
    lang: str = Query(
        default="en",
        description="Language for the copy: 'en' (default) or 'hi'. Others fall back to English.",
    ),
) -> FlowChooserResponse:
    """The selectable services — the in-chat replacement for the persona dropdown."""
    return _flow_chooser("hi" if lang == "hi" else "en")


@router.post("/select", response_model=StartChatResponse, description=_SELECT_DESCRIPTION)
async def select_flow(
    body: FlowSelectionRequest,
    rider: RiderIdentity = Depends(get_rider),
) -> StartChatResponse:
    """Start the flow the rider tapped. Same resolution as `/start`, persona from the box.

    Delegates to the same handler `/start` uses, so an explicit selection is resolved
    identically: coordinates still refine local-vs-visiting and set the pickup, and the
    conversation's classification is corrected from the rider's own words thereafter.
    """
    return await _handle_start(rider, body.flow, body.lat, body.lon, body.accuracy_m)


# ── LOCAL routes (Him Cab in-state riders) ────────────────────────────────────

@router.post("/local/start", response_model=StartChatResponse)
async def start_chat_local(
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Open the LOCAL chat dialog. Starts a fresh conversation and returns the
    welcome greeting to render as the first assistant bubble.

    Send lat/lon and the response's `persona` is authoritative: a fix outside Himachal
    is served as **outstation** even on this path, because the rider's location is a
    better answer than the screen they tapped from.
    """
    return await _handle_start(rider, LOCAL, lat, lon, accuracy_m)


@router.get("/local/stream")
async def stream_chat_local(
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Send a message on the LOCAL chat and stream the reply back as SSE."""
    return await _handle_stream(rider, LOCAL, message, session_id, lat, lon, accuracy_m)


@router.post("/local/message", response_model=ChatResponse)
async def send_message_local(
    body: ChatMessageRequest,
    rider: RiderIdentity = Depends(get_rider),
):
    """Non-streaming version of the LOCAL chat, for clients without SSE support."""
    return await _handle_message(rider, LOCAL, body)


@router.get("/local/history", response_model=ChatHistoryResponse)
async def history_local(
    limit: int = Query(default=50, ge=1, le=200, description="Most recent N messages"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    rider: RiderIdentity = Depends(get_rider),
):
    """Replay the LOCAL conversation so the app can restore its chat panel.

    Call this instead of /start when reopening a panel the user was already using —
    /start deliberately begins a fresh conversation.
    """
    return await _handle_history(rider, LOCAL, limit, session_id)


# ── OUTSTATION routes (visitors travelling into Himachal) ─────────────────────

@router.post("/outstation/start", response_model=StartChatResponse)
async def start_chat_outstation(
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Open the OUTSTATION chat dialog. Starts a fresh conversation and returns the
    welcome greeting to render as the first assistant bubble.

    Send lat/lon and the response's `persona` is authoritative: a fix inside Himachal
    is served as **local** even on this path — a Shimla resident does not need an
    itinerary for Shimla.
    """
    return await _handle_start(rider, OUTSTATION, lat, lon, accuracy_m)


@router.get("/outstation/stream")
async def stream_chat_outstation(
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Send a message on the OUTSTATION chat and stream the reply back as SSE."""
    return await _handle_stream(
        rider, OUTSTATION, message, session_id, lat, lon, accuracy_m
    )


@router.post("/outstation/message", response_model=ChatResponse)
async def send_message_outstation(
    body: ChatMessageRequest,
    rider: RiderIdentity = Depends(get_rider),
):
    """Non-streaming version of the OUTSTATION chat, for clients without SSE support."""
    return await _handle_message(rider, OUTSTATION, body)


@router.get("/outstation/history", response_model=ChatHistoryResponse)
async def history_outstation(
    limit: int = Query(default=50, ge=1, le=200, description="Most recent N messages"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    rider: RiderIdentity = Depends(get_rider),
):
    """Replay the OUTSTATION conversation so the app can restore its chat panel.

    Call this instead of /start when reopening a panel the user was already using —
    /start deliberately begins a fresh conversation.
    """
    return await _handle_history(rider, OUTSTATION, limit, session_id)


# ── RENTAL routes (a car and driver by the hour) ──────────────────────────────
# A rental is a SERVICE the rider chose, not an inference about who they are, so —
# unlike /local and /outstation — coordinates never reroute these. Their fix decides
# the pickup point and the reply language, not whether they still want a car.

@rental_router.get(
    "/greeting",
    response_model=GreetingResponse,
    description=_GREETING_DESCRIPTION,
)
async def greeting_rental() -> GreetingResponse:
    """The RENTAL opening line, in its generic form.

    `is_final` is false here: this asks an open "how long do you need the car for?",
    and `/start` replaces it with one that names the packages the rate card actually
    sells. Both are answerable, so showing this one first costs the rider nothing —
    but the duration is matched EXACTLY by the backend, so the named version is what
    stops an unbookable answer, and it should replace this as soon as it arrives.
    """
    return _greeting(RENTAL)


@rental_router.post("/start", response_model=StartChatResponse)
async def start_chat_rental(
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Open the RENTAL chat dialog — hiring a car and driver by the hour.

    The assistant asks for the duration, the pickup and when they need it, then quotes
    the cab options for that package. Every price comes from the Him Cab fare service;
    nothing is calculated here.

    Unlike the other two paths, the returned `persona` is always **rental**: a fix
    tells us where to collect them, not that they have stopped wanting a rental.
    """
    return await _handle_start(rider, RENTAL, lat, lon, accuracy_m)


@rental_router.get("/stream")
async def stream_chat_rental(
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    lat: float | None = Query(default=None, ge=-90, le=90, description=_LAT_DESCRIPTION),
    lon: float | None = Query(default=None, ge=-180, le=180, description=_LON_DESCRIPTION),
    accuracy_m: float | None = Query(default=None, ge=0, description="Device accuracy in metres."),
    rider: RiderIdentity = Depends(get_rider),
):
    """Send a message on the RENTAL chat and stream the reply back as SSE."""
    return await _handle_stream(rider, RENTAL, message, session_id, lat, lon, accuracy_m)


@rental_router.post("/message", response_model=ChatResponse)
async def send_message_rental(
    body: ChatMessageRequest,
    rider: RiderIdentity = Depends(get_rider),
):
    """Non-streaming version of the RENTAL chat, for clients without SSE support."""
    return await _handle_message(rider, RENTAL, body)


@rental_router.get("/history", response_model=ChatHistoryResponse)
async def history_rental(
    limit: int = Query(default=50, ge=1, le=200, description="Most recent N messages"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    rider: RiderIdentity = Depends(get_rider),
):
    """Replay the RENTAL conversation so the app can restore its chat panel."""
    return await _handle_history(rider, RENTAL, limit, session_id)
