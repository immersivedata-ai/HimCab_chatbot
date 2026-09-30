"""
app/api/v1/router.py
─────────────────────
Central v1 API router — mounts all route modules.
"""

from datetime import UTC, datetime

from fastapi import APIRouter

from app.api.v1.routes import alerts, chat, session, user

api_router = APIRouter(prefix="/v1")

api_router.include_router(chat.router)
# Same paths (/v1/chat/rental/*), separate router so the rental endpoints get their own
# Swagger section — see the note beside `rental_router` in routes/chat.py.
api_router.include_router(chat.rental_router)
api_router.include_router(session.router)
api_router.include_router(alerts.router)
api_router.include_router(user.router)

# Track startup time for uptime reporting
_startup_time = datetime.now(UTC)


@api_router.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for Cloud Run readiness/liveness probes.

    Firestore is the only external dependency worth probing: it is the system of
    record for conversations, so losing it is what actually breaks the product. There
    is no cache tier to report on any more — caching and rate limiting are in-process.
    """
    from app.repositories.firestore import get_session_service

    # Cheap round-trip against the pointer collection — proves we can read Firestore
    # without writing anything or depending on any composite index.
    firestore_ok = await get_session_service().ping()

    now = datetime.now(UTC)

    return {
        "status": "healthy" if firestore_ok else "unhealthy",
        "version": "1.0.0",
        "uptime_seconds": round((now - _startup_time).total_seconds(), 1),
        "timestamp": now.isoformat(),
        "services": {
            "firestore": "ok" if firestore_ok else "unavailable",
            "api": "ok",
        },
        "conversation_memory": "ok" if firestore_ok else "unavailable",
    }
