"""
app/core/logging.py
────────────────────
Structured JSON logging using structlog.
All log records include trace_id, user_id, and environment for Cloud Logging.
"""

import logging
import sys

import structlog
from structlog.types import EventDict, Processor

from app.core.config import settings


def add_app_context(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject app-level context into every log record."""
    event_dict["env"] = settings.app_env
    event_dict["service"] = "himcab-ai-assistant"
    return event_dict


def configure_logging() -> None:
    """
    Call once at application startup.
    In production: outputs JSON (compatible with Cloud Logging).
    In development: outputs coloured console output.
    """
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        add_app_context,
    ]

    if settings.is_production:
        # JSON for Cloud Logging
        renderer = structlog.processors.JSONRenderer()
    else:
        # Human-readable for local dev
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors
        + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(settings.log_level.upper())

    # Silence noisy third-party loggers
    for noisy in ["httpx", "httpcore", "uvicorn.access"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a bound structlog logger for the given module."""
    return structlog.get_logger(name)
