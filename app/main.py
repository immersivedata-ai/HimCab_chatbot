"""
app/main.py
────────────
FastAPI application entrypoint.
Configures middleware, mounts routers, sets up lifespan events.
"""

import asyncio
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

    # Warming runs in the BACKGROUND, so the server starts answering while it happens.
    #
    # It used to run here, inline, and uvicorn does not accept a connection until
    # startup returns — so every request queued behind it, including the ones that need
    # none of it. Constructing the Vertex client pulls in the SDK, about three seconds
    # of Python, and this container also serves the browser test pages: nobody could
    # even fetch the HTML until a language-model client they were not using had been
    # built.
    #
    # The two things being warmed are still worth warming — a compiled graph and a live
    # client mean the first rider does not pay for them — they simply do not need to
    # block the port. The health check, the greeting endpoint and the pages depend on
    # neither, and the first chat turn calls the same cached singletons, so it either
    # finds them ready or builds them exactly as it would have.
    async def _warm_up() -> None:
        try:
            from app.agent.graph import get_agent_graph

            await asyncio.to_thread(get_agent_graph)
            log.info("langgraph_warmed_up")

            from app.services.vertex_ai import get_vertex_ai_service

            await asyncio.to_thread(get_vertex_ai_service)
            log.info("vertex_ai_warmed_up")
        except Exception as e:
            # A failed warm-up must never stop the service: the first request that
            # needs one of these will construct it and surface the real error in
            # context, which is far more useful than a container that will not start.
            log.warning(
                "warm_up_failed", error=str(e), error_type=type(e).__name__
            )

    warm_up_task = asyncio.create_task(_warm_up())

    log.info("app_started", version="1.0.0")
    yield

    # Let a warm-up still in flight finish or be cancelled, rather than leaving a task
    # dangling into shutdown.
    warm_up_task.cancel()

    # ── Shutdown ──────────────────────────────────────────────────────────────
    log.info("app_shutting_down")

    # Close HTTP clients used by external services
    from app.integrations.external.geocoding import get_geocoding_service
    from app.integrations.external.routing import get_routing_service
    from app.integrations.external.weather import get_weather_service
    from app.integrations.himcab import get_fare_service, get_offers_service
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

    # ── Root and the browser test clients ─────────────────────────────────────
    # The three single-file pages in frontend/ drive the three workflows end to end —
    # outstation trips, local rides and hourly rentals — against the same public
    # endpoints any client calls, with a rider JWT pasted in by whoever is testing.
    # They are the only way to exercise a whole conversation by hand, so they ship.
    #
    # Gated on the environment rather than always-on. A page that looks like the
    # product sitting beside the API is fine while a team is testing and wrong once
    # this URL is the contract somebody integrates against — so production serves the
    # API and nothing else, and no code change is needed to make that true.
    #
    # NOTE: deploy.ps1 sets APP_ENV=development on Cloud Run, so today the deployed
    # URL DOES serve these. That is deliberate while the flows are being tested; set
    # APP_ENV=production to turn them off without touching this file.
    frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
    serve_test_ui = not settings.is_production and frontend_dir.is_dir()

    if serve_test_ui:
        # Mounted as well as routed, so a page can pull a sibling asset if one is ever
        # added next to it.
        app.mount(
            "/frontend",
            StaticFiles(directory=str(frontend_dir), html=True),
            name="frontend",
        )

        def _page(filename: str):
            return FileResponse(str(frontend_dir / filename))

        @app.get("/outstation", include_in_schema=False)
        async def outstation_ui():
            """Visitor planning a trip — day-by-day itinerary, English by default."""
            return _page("index.html")

        @app.get("/local", include_in_schema=False)
        async def local_ui():
            """Himachal resident booking a ride — cab, fare and drive time, Hindi first."""
            return _page("local.html")

        @app.get("/rental", include_in_schema=False)
        async def rental_ui():
            """A car and driver by the hour — packages, pickup and a price."""
            return _page("rental.html")

    @app.get("/", include_in_schema=False)
    async def root():
        """Signpost, so the base URL is not a bare 404 for whoever opens it.

        Names the test clients when they are being served, because the usual reason
        somebody opens the bare URL is to find them.
        """
        body = {
            "service": "Him Cab — AI Tourist Travel Assistant",
            "version": "1.0.0",
            "docs": "/docs" if not settings.is_production else None,
            "health": "/v1/health",
        }
        if serve_test_ui:
            body["test_clients"] = {
                "outstation": "/outstation",
                "local": "/local",
                "rental": "/rental",
            }
        return body

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
