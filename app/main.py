"""
app/main.py
────────────
FastAPI application entrypoint.
Configures middleware, mounts routers, sets up lifespan events.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

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


# ── OpenAPI tag metadata ──────────────────────────────────────────────────────
# Each tag is a section in Swagger UI, and its `name` is the anchor an integrator can
# be sent straight to: /docs#/Rental. Keep the names stable — a renamed tag silently
# breaks every deep link already shared with another team.
_RENTAL_TAG_DESCRIPTION = """**Car and driver by the hour.**

The rider picks a pickup point and one of the hourly packages configured in the rate
card (`GET /api/rider/fare/get-all`), and the assistant quotes it from
`POST /fare/estimate`. No price is ever computed in this service.

Send `Authorization: Bearer <Him Cab rider JWT>` on every call. There is no rider id
or session to pass — the token resolves the conversation, so multi-turn context works
with no client state. `/start` is optional: `/stream` and `/message` create the
conversation on demand, so they never 404 with a session the client has to recover
from.

Unlike `/local` and `/outstation`, a rental is a service the RIDER chose. A GPS fix
here sets the pickup point and the reply language; it never reroutes the persona.

`/stream` is Server-Sent Events — `token`, `structured`, `done`, `error`. The
`structured` types a rental turn can emit are `rental_packages`, `cab_fare`, `offers`,
`booking_handoff`, `weather_alert` and `request_cancelled`. Ignore any type you do not
recognise. `request_cancelled` is the one to ACT on: dismiss the booking sheet and
clear the fare cards, because what they describe no longer exists.

Rate limit: 30 requests/minute per rider, then HTTP 429 with a `Retry-After` header.
"""

OPENAPI_TAGS = [
    {"name": "Rental", "description": _RENTAL_TAG_DESCRIPTION},
    {
        "name": "Chat",
        "description": (
            "The Himachal-local and visiting-tourist assistants, plus "
            "`POST /v1/chat/route`, which reads a GPS fix and says which of the two "
            "to use."
        ),
    },
    {"name": "Session", "description": "Trip session records behind a conversation."},
    {"name": "Alerts", "description": "Weather and snowfall advisories."},
    {"name": "User", "description": "Rider profile and preferences."},
    {"name": "Health", "description": "Readiness/liveness probe for Cloud Run."},
]


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
        openapi_tags=OPENAPI_TAGS,
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

    # ── Root ──────────────────────────────────────────────────────────────────
    # This service is an API and nothing else. It used to also serve the browser test
    # clients (/, /local, /rental) out of frontend/; those are gone from the image so
    # that the deployed URL is exactly the contract an integrator reads — no page that
    # looks like the product, no second copy of the flow to keep in step with it.
    # Test locally against the API itself, or against /docs.
    @app.get("/", include_in_schema=False)
    async def root():
        """Signpost, so the base URL is not a bare 404 for whoever opens it."""
        return {
            "service": "Him Cab — AI Tourist Travel Assistant",
            "version": "1.0.0",
            "docs": "/docs" if not settings.is_production else None,
            "health": "/v1/health",
        }

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
