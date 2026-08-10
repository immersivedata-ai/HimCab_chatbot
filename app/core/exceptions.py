"""
app/core/exceptions.py
───────────────────────
Custom exception hierarchy and global FastAPI exception handlers.
"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


# ── Base ──────────────────────────────────────────────────────────────────────

class HimCabBaseError(Exception):
    """Root exception for all application-level errors."""
    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    message: str = "An unexpected error occurred"

    def __init__(self, message: str | None = None, **context):
        self.message = message or self.__class__.message
        self.context = context
        super().__init__(self.message)


# ── Auth ──────────────────────────────────────────────────────────────────────

class AuthenticationError(HimCabBaseError):
    status_code = 401
    error_code = "AUTHENTICATION_FAILED"
    message = "Authentication failed"


class AuthorizationError(HimCabBaseError):
    status_code = 403
    error_code = "FORBIDDEN"
    message = "You do not have permission to perform this action"


# ── Session ───────────────────────────────────────────────────────────────────

class SessionNotFoundError(HimCabBaseError):
    status_code = 404
    error_code = "SESSION_NOT_FOUND"
    message = "Trip session not found"


class ExternalServiceError(HimCabBaseError):
    status_code = 502
    error_code = "EXTERNAL_SERVICE_ERROR"
    message = "An external service call failed"


class WeatherServiceError(ExternalServiceError):
    error_code = "WEATHER_SERVICE_ERROR"
    message = "Weather data could not be fetched"


class RateLimitExceededError(HimCabBaseError):
    """429 for the caller sending too many requests.

    Pass `retry_after=<seconds>` so the client is told when to come back; it surfaces
    both as a `Retry-After` response header and in `error.details.retry_after`.
    """
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please wait before trying again."


# ── Exception Handlers ────────────────────────────────────────────────────────

def _error_response(exc: HimCabBaseError) -> JSONResponse:
    headers: dict[str, str] = {}

    # Standard back-off signal for 429/503 — clients should honour it before retrying.
    retry_after = exc.context.get("retry_after")
    if retry_after is not None:
        headers["Retry-After"] = str(int(retry_after))

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.context if exc.context else None,
            }
        },
        headers=headers or None,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all custom exception handlers on the FastAPI app."""

    @app.exception_handler(HimCabBaseError)
    async def himcab_error_handler(request: Request, exc: HimCabBaseError):
        return _error_response(exc)

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        from app.core.logging import get_logger
        log = get_logger(__name__)
        log.error("unhandled_exception", exc_info=exc, path=request.url.path)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "details": None,
                }
            },
        )
