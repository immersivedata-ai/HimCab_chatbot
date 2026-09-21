"""
app/services/vertex_ai.py
──────────────────────────
Vertex AI / Gemini client service.
Provides async wrappers for:
  - Gemini Pro (complex planning, structured output)
  - Gemini Flash (fast streaming responses)
  - Embeddings (for semantic search if needed)
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Literal, Optional

from google.api_core.exceptions import ResourceExhausted

from app.core.config import settings
from app.core.logging import get_logger

if TYPE_CHECKING:
    from langchain_google_vertexai import ChatVertexAI

log = get_logger(__name__)


def _init_vertex_ai() -> None:
    """Initialize Vertex AI SDK once."""
    import vertexai

    vertexai.init(
        project=settings.gcp_project_id,
        location=settings.vertex_ai_location,
    )


class VertexAIService:
    """
    Singleton service providing LangChain-compatible Gemini models.
    Use `get_vertex_ai_service()` to obtain the shared instance.
    """

    def __init__(self) -> None:
        # Imported HERE, not at module scope. `vertexai` and `langchain_google_vertexai`
        # are about three seconds of Python between them, and at module scope every
        # import of this app paid it before the server could bind a port — including on
        # a Cloud Run cold start, where the browser cannot fetch the test page until it
        # finishes. Nothing above this line needs the SDK, so nothing above it waits.
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_google_vertexai import ChatVertexAI

        _init_vertex_ai()

        # One bucket for BOTH clients, deliberately. The quota Vertex enforces is per
        # project and model, not per Python object, so two independently-paced clients
        # would simply take turns exceeding it.
        limiter = InMemoryRateLimiter(
            requests_per_second=settings.vertex_requests_per_second,
            check_every_n_seconds=0.05,
            max_bucket_size=settings.vertex_burst_size,
        )

        # Flash only. The Pro model was instantiated on every startup and never called:
        # the router and planner do classification and extraction, and the writer is
        # latency-critical — Flash is the right tier for all three.
        # WITH a standby configured, the primary does not retry at all.
        #
        # Retrying and failing over are two answers to the same question, and paying for
        # both is the worst of each. Measured against a genuinely exhausted primary:
        # `max_retries=2` spent 24 SECONDS climbing the backoff ladder before handing
        # over — three calls a turn, so over a minute of a rider watching a spinner, and
        # past the 45s per-call timeout for the slower ones. The standby is the retry,
        # and it is a retry against a pool that actually has room.
        #
        # The standby itself keeps `vertex_max_retries`: by then there is nowhere else
        # to go, so a short ladder is worth walking.
        retries = 0 if (settings.gemini_fallback_model or "").strip() else settings.vertex_max_retries
        shared: dict[str, Any] = {
            "model_name": settings.gemini_flash_model,
            "project": settings.gcp_project_id,
            "location": settings.vertex_ai_location,
            "max_output_tokens": 4096,
            "rate_limiter": limiter,
            "max_retries": retries,
        }

        # Gemini Flash — used in Writer node (speed + streaming)
        self.gemini_flash = ChatVertexAI(temperature=0.8, streaming=True, **shared)

        # Gemini Flash with streaming disabled (for structured JSON output)
        self.gemini_flash_structured = ChatVertexAI(
            temperature=0.4, streaming=False, **shared
        )

        # The standby. Quota on Vertex is PER MODEL, and the pools are not the same
        # size: measured on this project while production was down, flash-lite returned
        # 429 on 8 of 8 calls while flash answered 8 of 8, same region, same minute. The
        # cheap model stays primary and this is used only when its pool is empty, so the
        # dearer one is billed for the calls that would otherwise have been an outage.
        self.fallback_name: str = (settings.gemini_fallback_model or "").strip()
        self.fallback_flash: Optional[ChatVertexAI] = None
        self.fallback_structured: Optional[ChatVertexAI] = None
        if self.fallback_name and self.fallback_name != settings.gemini_flash_model:
            standby: dict[str, Any] = {
                **shared,
                "model_name": self.fallback_name,
                "max_retries": settings.vertex_max_retries,
            }
            self.fallback_flash = ChatVertexAI(
                temperature=0.8, streaming=True, **standby
            )
            self.fallback_structured = ChatVertexAI(
                temperature=0.4, streaming=False, **standby
            )
        else:
            self.fallback_name = ""

        log.info(
            "vertex_ai_initialized",
            flash_model=settings.gemini_flash_model,
            fallback_model=self.fallback_name or None,
            location=settings.vertex_ai_location,
            primary_max_retries=retries,
            standby_max_retries=settings.vertex_max_retries,
            requests_per_second=settings.vertex_requests_per_second,
            max_in_flight=settings.vertex_max_in_flight,
        )

    def get_flash_model(self, streaming: bool = True):
        """The model to generate with — the standby already attached.

        Returns a Runnable, not a bare `ChatVertexAI`: `with_fallbacks` wraps it. That
        is invisible to `ainvoke`/`astream`, which is all the router and writer use, but
        it does mean `.with_structured_output(...)` is NOT available on the result —
        that method belongs to `ChatVertexAI`. The planner needs it, so it calls
        `structured()` instead, which composes the two in the order that works.
        """
        return self._standby(
            self.gemini_flash if streaming else self.gemini_flash_structured,
            self.fallback_flash if streaming else self.fallback_structured,
        )

    def primary_model(self, streaming: bool = True) -> ChatVertexAI:
        """The bare client, no standby. For callers that need `ChatVertexAI` itself."""
        return self.gemini_flash if streaming else self.gemini_flash_structured

    def _standby(self, primary, fallback):
        """`primary`, backed by `fallback` when the pool is empty.

        ONLY `ResourceExhausted` is handled. A fallback that catches everything hides
        real faults — a bad schema would be retried against a second model and fail
        there too, at twice the cost and with the error reported from the wrong place.
        """
        if fallback is None:
            return primary
        return primary.with_fallbacks(
            [fallback], exceptions_to_handle=(ResourceExhausted,)
        )

    def structured(self, schema, *, method: Literal["json_mode"] = "json_mode"):
        """A structured-output runnable that survives the primary model being full."""
        return self._standby(
            self.gemini_flash_structured.with_structured_output(schema, method=method),
            None if self.fallback_structured is None
            else self.fallback_structured.with_structured_output(schema, method=method),
        )


    async def count_tokens(self, text: str) -> int:
        """Estimate token count for a given text string."""
        try:
            from vertexai.generative_models import GenerativeModel

            _model = GenerativeModel(settings.gemini_flash_model)
            response = await asyncio.to_thread(
                _model.count_tokens, [text]
            )
            return response.total_tokens
        except Exception as e:
            log.warning("token_count_failed", error=str(e))
            return len(text.split()) * 2  # rough fallback


# ── The concurrency ceiling ───────────────────────────────────────────────────
#
# The rate limiter paces how often a call STARTS. This bounds how many are outstanding
# at once, and the two do different jobs: when Vertex is slow — which is exactly what it
# is when the pool is under pressure — calls stop retiring, a paced trickle still piles
# up, and the instance ends up holding hundreds of open requests against a service that
# is already struggling. That is a thundering herd built out of politeness.
#
# Per process, like everything else here. See `settings.vertex_max_in_flight`.
_in_flight: Optional[asyncio.Semaphore] = None


def _gate() -> asyncio.Semaphore:
    """The semaphore, created on the running loop rather than at import time.

    Built lazily because an `asyncio.Semaphore` binds to the loop that first uses it,
    and this module is imported long before uvicorn has one.
    """
    global _in_flight
    if _in_flight is None:
        _in_flight = asyncio.Semaphore(settings.vertex_max_in_flight)
    return _in_flight


async def call_model(coro):
    """Await a Vertex call under the process-wide concurrency ceiling.

    Every model call in the agent goes through here. It is a plain wrapper rather than
    anything cleverer so that the one thing it guarantees — that we never have more than
    `vertex_max_in_flight` requests open — cannot be bypassed by forgetting a decorator.

    It does NOT retry and does NOT catch: the callers each have their own timeout and
    their own idea of what a failure means, and burying either here would hide the
    throttling this exists to relieve.
    """
    async with _gate():
        return await coro


# ── Singleton ─────────────────────────────────────────────────────────────────

_vertex_ai_service: Optional[VertexAIService] = None


def get_vertex_ai_service() -> VertexAIService:
    """Return the shared VertexAIService singleton."""
    global _vertex_ai_service
    if _vertex_ai_service is None:
        _vertex_ai_service = VertexAIService()
    return _vertex_ai_service
