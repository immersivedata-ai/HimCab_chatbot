"""
app/api/v1/router.py
─────────────────────
Central v1 API router — mounts all route modules.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from app.api.v1.routes import alerts, chat, session, user

api_router = APIRouter(prefix="/v1")

api_router.include_router(chat.router)
api_router.include_router(session.router)
api_router.include_router(alerts.router)
api_router.include_router(user.router)

# Track startup time for uptime reporting
_startup_time = datetime.now(timezone.utc)


@api_router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Cloud Run readiness/liveness probes.
    Checks Redis and reports service uptime.
    """
    from app.services.redis_cache import get_redis_cache
    cache = get_redis_cache()
    redis_ok = await cache.ping()

    now = datetime.now(timezone.utc)
    uptime_seconds = (now - _startup_time).total_seconds()

    return {
        "status": "healthy" if redis_ok else "degraded",
        "version": "1.0.0",
        "uptime_seconds": round(uptime_seconds, 1),
        "timestamp": now.isoformat(),
        "services": {
            "redis": "ok" if redis_ok else "unavailable",
            "api": "ok",
        },
    }
