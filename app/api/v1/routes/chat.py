"""
app/api/v1/routes/chat.py
──────────────────────────
SSE Chat endpoint — the primary API consumed by the mobile app.

GET /v1/chat/stream
  - Accepts: session_id + message as query params
  - Returns: text/event-stream (SSE)
  - Events: token | structured | done | error

The SSE stream delivers:
  1. `token` events → raw AI text chunks (for chat bubble animation)
  2. `structured` events → JSON cards (itinerary, weather, hotels)
  3. `done` event → signals completion with token usage
  4. `error` event → signals failure

POST /v1/chat/message
  - Non-streaming version (for simpler mobile implementations)
  - Returns full JSON response after AI completes
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
from app.api.middleware.auth import FirebaseUser, get_current_user
from app.core.exceptions import RateLimitExceededError, SessionNotFoundError
from app.core.logging import get_logger
from app.models.session import ChatRequest, ChatResponse, MessageRole
from app.services.firestore import get_session_service
from app.services.redis_cache import get_redis_cache

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


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


async def _stream_agent_response(
    session_id: str,
    user_id: str,
    message: str,
    language: str = "en",
):
    """
    Core generator function — runs the LangGraph agent and yields SSE events.

    Yielded format (sse-starlette compatible):
      {"event": "token", "data": '{"token": "..."}'}
      {"event": "structured", "data": '{"type": "itinerary", "payload": {...}}'}
      {"event": "done", "data": '{"session_id": "..."}'}
    """
    graph = get_agent_graph()
    session_svc = get_session_service()

    # Load session from Firestore
    try:
        session = await session_svc.get_session(session_id)
    except SessionNotFoundError:
        yield {
            "event": "error",
            "data": json.dumps({"code": "SESSION_NOT_FOUND", "message": "Session not found"}),
        }
        return

    # Persist user message to Firestore
    await session_svc.append_message(
        session_id=session_id,
        role=MessageRole.USER,
        content=message,
    )

    # Build initial agent state from session
    history_messages = _build_history_messages(session.messages)
    # Append the new message
    history_messages.append(HumanMessage(content=message))

    initial_state = AgentState(
        messages=history_messages,
        session_id=session_id,
        user_id=user_id,
        language=language or session.language,
        entities=session.entities,
    )

    # Run the LangGraph agent
    try:
        structlog.contextvars.bind_contextvars(
            session_id=session_id,
            user_id=user_id,
            trace_id=str(uuid4()),
        )

        final_state: dict = await graph.ainvoke(initial_state)

        # ── Simulate token streaming ──────────────────────────────────────────
        # Since we use non-streaming Gemini calls for quality,
        # we chunk the final text response for the streaming UX effect.
        response_text = final_state.get("final_response_text", "")
        words = response_text.split(" ")

        for i, word in enumerate(words):
            chunk = word + (" " if i < len(words) - 1 else "")
            yield {
                "event": "token",
                "data": json.dumps({"token": chunk}),
            }
            await asyncio.sleep(0.02)  # ~50 tokens/sec pace

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
                "tokens_used": final_state.get("tokens_used", 0),
                "node_trace": final_state.get("node_trace", []),
            }),
        }

        log.info(
            "chat_stream_completed",
            session_id=session_id,
            nodes=final_state.get("node_trace", []),
            has_itinerary=final_state.get("structured_itinerary") is not None,
        )

    except Exception as e:
        log.error("chat_stream_error", error=str(e), error_type=type(e).__name__, session_id=session_id)
        yield {
            "event": "error",
            "data": json.dumps({
                "code": "AGENT_ERROR",
                "message": "The AI assistant encountered an error. Please try again.",
            }),
        }
    finally:
        structlog.contextvars.clear_contextvars()


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stream")
async def stream_chat(
    session_id: str = Query(..., description="Session ID from POST /v1/session"),
    message: str = Query(..., min_length=1, max_length=2000, description="User's message text"),
    language: str = Query(default="en", description="Language: en | hi | hinglish"),
    user: FirebaseUser = Depends(get_current_user),
):
    """
    SSE endpoint — streams AI response token by token.

    Connect with EventSource in the mobile app:
    ```
    const source = new EventSource(`/v1/chat/stream?session_id=xxx&message=yyy`)
    source.addEventListener('token', e => appendToken(JSON.parse(e.data).token))
    source.addEventListener('structured', e => renderCard(JSON.parse(e.data)))
    source.addEventListener('done', () => source.close())
    ```
    """
    # Rate limit check
    cache = get_redis_cache()
    allowed, count = await cache.check_rate_limit(user.uid)
    if not allowed:
        raise RateLimitExceededError(
            f"Rate limit exceeded: {count}/{30} requests per minute"
        )

    return EventSourceResponse(
        _stream_agent_response(
            session_id=session_id,
            user_id=user.uid,
            message=message,
            language=language,
        ),
        media_type="text/event-stream",
    )


@router.post("/message", response_model=ChatResponse)
async def send_message_sync(
    body: ChatRequest,
    user: FirebaseUser = Depends(get_current_user),
):
    """
    Non-streaming endpoint — returns full JSON response.
    Use when SSE is not supported by the client.
    """
    graph = get_agent_graph()
    session_svc = get_session_service()

    session = await session_svc.get_session(body.session_id)

    history_messages = _build_history_messages(session.messages)
    history_messages.append(HumanMessage(content=body.message))

    initial_state = AgentState(
        messages=history_messages,
        session_id=body.session_id,
        user_id=user.uid,
        language=body.language,
        entities=session.entities,
    )

    final_state: dict = await graph.ainvoke(initial_state)

    return ChatResponse(
        session_id=body.session_id,
        response=final_state.get("final_response_text", ""),
        structured_events=final_state.get("structured_events", []),
        itinerary=final_state.get("structured_itinerary").model_dump() if final_state.get("structured_itinerary") else None,
        node_trace=final_state.get("node_trace", []),
    )
