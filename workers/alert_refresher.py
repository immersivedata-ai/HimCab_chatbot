"""
workers/alert_refresher.py
───────────────────────────
Cloud Run Job — refreshes Himachal travel alerts on a schedule.
Triggered by Cloud Scheduler every 15 minutes via Pub/Sub.

This keeps the Redis cache warm with fresh snowfall/weather data
so the first user request doesn't incur an API cold start.

Deploy as: Cloud Run Job (not a service)
Schedule: */15 * * * * (every 15 minutes via Cloud Scheduler)
"""

import asyncio
import sys
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging import configure_logging, get_logger
from app.models.alerts import AlertSeverity, AlertType, HimachalAlertsSummary, TravelAlert
from app.services.external.weather import get_weather_service
from app.services.redis_cache import get_redis_cache

configure_logging()
log = get_logger(__name__)

DESTINATIONS_TO_REFRESH = [
    "shimla", "manali", "dharamshala",
    "kasol", "spiti", "dalhousie", "kullu",
]


async def refresh_weather_cache() -> None:
    """
    Fetch fresh weather data for all monitored destinations
    and write to Redis cache.
    """
    weather_svc = get_weather_service()
    cache = get_redis_cache()

    results = {"refreshed": [], "failed": []}

    for destination in DESTINATIONS_TO_REFRESH:
        try:
            # Force a fresh fetch by clearing cache first
            await cache.delete(f"weather:{destination}:today")

            # Fetch fresh data (will re-populate cache)
            report = await weather_svc.get_weather(destination)
            snowfall = await weather_svc.get_snowfall_risk(destination)

            log.info(
                "destination_refreshed",
                destination=destination,
                temp_c=report.current_temp_c,
                snowfall_risk=snowfall["risk"],
            )
            results["refreshed"].append(destination)

        except Exception as e:
            log.error("destination_refresh_failed", destination=destination, error=str(e))
            results["failed"].append(destination)

    # Also invalidate the regional alerts cache so it rebuilds on next request
    await cache.delete("alerts:himachal")

    log.info(
        "alert_refresh_completed",
        refreshed=len(results["refreshed"]),
        failed=len(results["failed"]),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    # Close connections
    await cache.close()
    await weather_svc.close()


async def main() -> None:
    log.info("alert_refresher_starting", env=settings.app_env)
    await refresh_weather_cache()
    log.info("alert_refresher_done")


if __name__ == "__main__":
    asyncio.run(main())
    sys.exit(0)
