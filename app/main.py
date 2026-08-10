"""
app/main.py
────────────
FastAPI application entrypoint.
Configures middleware, mounts routers, sets up lifespan events.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger

log = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup and shutdown lifecycle.
    - Startup: compile the LangGraph, warm the Vertex AI client
    - Shutdown: close the HTTP clients
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    log.info("app_starting", env=settings.app_env, debug=settings.app_debug)

    # Pre-compile the LangGraph graph (avoids cold-start latency on first request)
    from app.agent.graph import get_agent_graph
    get_agent_graph()
    log.info("langgraph_warmed_up")

    # Warm up Vertex AI client
    from app.services.vertex_ai import get_vertex_ai_service
    get_vertex_ai_service()
    log.info("vertex_ai_warmed_up")

    log.info("app_started", version="1.0.0")
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("app_shutting_down")

    # Close HTTP clients used by external services
    from app.services.external.geocoding import get_geocoding_service
    from app.services.external.routing import get_routing_service
    from app.services.external.weather import get_weather_service
    from app.services.himcab import get_fare_service, get_offers_service
    try:
        await get_weather_service().close()
        await get_fare_service().close()
        await get_offers_service().close()
        # Geography is resolved over the wire now, so these hold pooled connections too.
        await get_geocoding_service().close()
        await get_routing_service().close()
    except Exception as e:
        log.warning("http_client_close_failed", error=str(e))

    log.info("app_shutdown_complete")


# ── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    configure_logging()

    app = FastAPI(
        title="Him Cab — AI Tourist Travel Assistant",
        description=(
            "Enterprise-grade AI travel concierge backend powering the Him Cab rider app. "
            "Plan Himachal Pradesh trips, get itineraries, weather alerts, and cab packages via AI."
        ),
        version="1.0.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────────

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Request ID middleware for distributed tracing
    @app.middleware("http")
    async def add_trace_id(request, call_next):
        import uuid
        import structlog
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        structlog.contextvars.clear_contextvars()
        return response

    # ── Exception Handlers ────────────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(api_router)

    # ── Frontend (test UI) ────────────────────────────────────────────────────
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    if frontend_dir.is_dir():
        app.mount("/frontend", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

        @app.get("/", include_in_schema=False)
        async def root_redirect():
            return FileResponse(str(frontend_dir / "index.html"))

        @app.get("/local", include_in_schema=False)
        async def local_ui():
            """Local HP resident travel assistant UI."""
            return FileResponse(str(frontend_dir / "local.html"))

    return app


app = create_app()


# ── Dev entrypoint ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        workers=1 if settings.app_debug else settings.app_workers,
        log_level=settings.log_level.lower(),
    )
