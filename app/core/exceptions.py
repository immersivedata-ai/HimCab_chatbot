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


class SessionExpiredError(HimCabBaseError):
    status_code = 410
    error_code = "SESSION_EXPIRED"
    message = "Trip session has expired"


# ── Agent ─────────────────────────────────────────────────────────────────────

class AgentExecutionError(HimCabBaseError):
    status_code = 500
    error_code = "AGENT_EXECUTION_FAILED"
    message = "AI agent encountered an error during execution"


class ToolExecutionError(HimCabBaseError):
    status_code = 502
    error_code = "TOOL_EXECUTION_FAILED"
    message = "A tool called by the AI agent failed"


class LLMRateLimitError(HimCabBaseError):
    status_code = 429
    error_code = "LLM_RATE_LIMIT"
    message = "AI service rate limit reached. Please try again shortly."


# ── External Services ─────────────────────────────────────────────────────────

class ExternalServiceError(HimCabBaseError):
    status_code = 502
    error_code = "EXTERNAL_SERVICE_ERROR"
    message = "An external service call failed"


class WeatherServiceError(ExternalServiceError):
    error_code = "WEATHER_SERVICE_ERROR"
    message = "Weather data could not be fetched"


class PlacesServiceError(ExternalServiceError):
    error_code = "PLACES_SERVICE_ERROR"
    message = "Places/Maps data could not be fetched"


# ── Validation ────────────────────────────────────────────────────────────────

class ValidationError(HimCabBaseError):
    status_code = 422
    error_code = "VALIDATION_ERROR"
    message = "Request validation failed"


# ── Rate Limit ────────────────────────────────────────────────────────────────

class RateLimitExceededError(HimCabBaseError):
    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"
    message = "Too many requests. Please wait before trying again."


# ── Exception Handlers ────────────────────────────────────────────────────────

def _error_response(exc: HimCabBaseError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.message,
                "details": exc.context if exc.context else None,
            }
        },
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
