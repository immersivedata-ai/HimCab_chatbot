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
║  Pick the pair that matches your app:                                            ║
║      local app  →  /v1/chat/local/*                                              ║
║      outstation →  /v1/chat/outstation/*                                         ║
║                                                                                  ║
║  1. User opens the chat dialog:                                                  ║
║       POST /v1/chat/{local|outstation}/start                                     ║
║       → {"session_id", "message", "language", "persona"}                         ║
║       Render `message` as the first assistant bubble (the welcome greeting).     ║
║                                                                                  ║
║  2. User sends a message (streaming — preferred):                                ║
║       GET /v1/chat/{local|outstation}/stream?message=<text>                      ║
║       → text/event-stream: token | structured | done | error                     ║
║                                                                                  ║
║     …or non-streaming, if SSE is inconvenient:                                   ║
║       POST /v1/chat/{local|outstation}/message   body: {"message": "<text>"}     ║
║       → full JSON reply                                                          ║
║                                                                                  ║
║  That's the whole integration. Step 2 works even if step 1 was skipped — the     ║
║  backend creates the conversation on demand, so these endpoints never 404 with   ║
║  a "session expired" the app has to recover from.                                ║
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
                    snowfall_alert | cab_fare. See `StructuredEvent` in
                    app/models/session.py for each payload shape. Ignore unknown types.
  3. `done`       → completion
  4. `error`      → failure

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
from app.models.session import (
    ChatHistoryMessage,
    ChatHistoryResponse,
    ChatMessageRequest,
    ChatResponse,
    MessageRole,
    StartChatResponse,
    TripSession,
)
from app.services.chat_session import LOCAL, OUTSTATION, resolve_chat, start_chat
from app.services.firestore import get_session_service
from app.services.local_cache import RateLimiter

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])

#: Per-rider request budget. In-process — see app/services/local_cache.py for what
#: that means on a multi-instance deployment.
_rate_limiter = RateLimiter(limit=settings.rate_limit_per_minute, window_seconds=60)

_SESSION_ID_DESCRIPTION = (
    "Optional. Omit to let the backend resolve the conversation from your Bearer "
    "token. Pass the session_id from /start to pin a specific conversation — needed "
    "only when one account is signed in on several devices. Unknown or unowned ids "
    "are ignored, not rejected."
)


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

async def _handle_start(rider: RiderIdentity, persona: str) -> StartChatResponse:
    session = await start_chat(rider.user_id, persona)
    # The greeting is already stored on the session by start_chat().
    greeting = session.messages[-1].content if session.messages else ""
    return StartChatResponse(
        session_id=session.session_id,
        message=greeting,
        language=str(session.language),
        persona=persona,
    )


async def _handle_stream(
    rider: RiderIdentity, persona: str, message: str, session_id: str | None = None
) -> EventSourceResponse:
    _enforce_rate_limit(rider.user_id)
    session = await resolve_chat(rider.user_id, persona, session_id=session_id)
    return EventSourceResponse(
        _stream_agent_response(session, rider, message, persona),
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
    session = await resolve_chat(rider.user_id, persona, session_id=body.session_id)

    # Snapshot-then-persist, for the same reason as the streaming path above.
    initial_state = _build_state(session, rider, body.message, persona)
    await get_session_service().append_message(
        session_id=session.session_id,
        role=MessageRole.USER,
        content=body.message,
    )

    final_state: dict = await get_agent_graph().ainvoke(initial_state)

    return ChatResponse(
        session_id=session.session_id,
        response=final_state.get("final_response_text", ""),
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


# ── LOCAL routes (Him Cab in-state riders) ────────────────────────────────────

@router.post("/local/start", response_model=StartChatResponse)
async def start_chat_local(rider: RiderIdentity = Depends(get_rider)):
    """Open the LOCAL chat dialog. Starts a fresh conversation and returns the
    welcome greeting to render as the first assistant bubble."""
    return await _handle_start(rider, LOCAL)


@router.get("/local/stream")
async def stream_chat_local(
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    rider: RiderIdentity = Depends(get_rider),
):
    """Send a message on the LOCAL chat and stream the reply back as SSE."""
    return await _handle_stream(rider, LOCAL, message, session_id)


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
async def start_chat_outstation(rider: RiderIdentity = Depends(get_rider)):
    """Open the OUTSTATION chat dialog. Starts a fresh conversation and returns the
    welcome greeting to render as the first assistant bubble."""
    return await _handle_start(rider, OUTSTATION)


@router.get("/outstation/stream")
async def stream_chat_outstation(
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    session_id: str | None = Query(default=None, description=_SESSION_ID_DESCRIPTION),
    rider: RiderIdentity = Depends(get_rider),
):
    """Send a message on the OUTSTATION chat and stream the reply back as SSE."""
    return await _handle_stream(rider, OUTSTATION, message, session_id)


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
