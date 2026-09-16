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
from typing import TYPE_CHECKING, Optional

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
        from langchain_google_vertexai import ChatVertexAI

        _init_vertex_ai()

        # Flash only. The Pro model was instantiated on every startup and never called:
        # the router and planner do classification and extraction, and the writer is
        # latency-critical — Flash is the right tier for all three.

        # Gemini Flash — used in Writer node (speed + streaming)
        self.gemini_flash = ChatVertexAI(
            model_name=settings.gemini_flash_model,
            project=settings.gcp_project_id,
            location=settings.vertex_ai_location,
            temperature=0.8,
            max_output_tokens=4096,
            streaming=True,
        )

        # Gemini Flash with streaming disabled (for structured JSON output)
        self.gemini_flash_structured = ChatVertexAI(
            model_name=settings.gemini_flash_model,
            project=settings.gcp_project_id,
            location=settings.vertex_ai_location,
            temperature=0.4,
            max_output_tokens=4096,
            streaming=False,
        )

        log.info("vertex_ai_initialized", flash_model=settings.gemini_flash_model)

    def get_flash_model(self, streaming: bool = True) -> ChatVertexAI:
        """Return Gemini Flash for fast streaming responses."""
        return self.gemini_flash if streaming else self.gemini_flash_structured

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


# ── Singleton ─────────────────────────────────────────────────────────────────

_vertex_ai_service: Optional[VertexAIService] = None


def get_vertex_ai_service() -> VertexAIService:
    """Return the shared VertexAIService singleton."""
    global _vertex_ai_service
    if _vertex_ai_service is None:
        _vertex_ai_service = VertexAIService()
    return _vertex_ai_service
