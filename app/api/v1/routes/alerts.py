"""
app/api/v1/routes/alerts.py
────────────────────────────
Himachal Pradesh travel alerts endpoint.
Aggregates weather, snowfall, and road closure alerts for the region.

GET /v1/alerts/himachal     — Regional alert summary
GET /v1/alerts/{destination} — Alerts for a specific destination
"""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.middleware.auth import optional_auth
from app.core.logging import get_logger
from app.models.alerts import AlertSeverity, AlertType, HimachalAlertsSummary, TravelAlert
from app.services.external.weather import get_weather_service
from app.services.redis_cache import get_redis_cache

log = get_logger(__name__)
router = APIRouter(prefix="/alerts", tags=["Alerts"])

# Key Himachal destinations to monitor
MONITORED_DESTINATIONS = [
    "shimla", "manali", "dharamshala", "kasol", "spiti", "dalhousie"
]


def _compute_overall_severity(alerts: list[TravelAlert]) -> AlertSeverity:
    if not alerts:
        return AlertSeverity.INFO
    severity_order = [
        AlertSeverity.INFO, AlertSeverity.LOW,
        AlertSeverity.MEDIUM, AlertSeverity.HIGH, AlertSeverity.CRITICAL
    ]
    max_sev = max(alerts, key=lambda a: severity_order.index(AlertSeverity(a.severity)))
    return AlertSeverity(max_sev.severity)


@router.get("/himachal", response_model=HimachalAlertsSummary)
async def get_himachal_alerts():
    """
    Get aggregated travel alerts for all major Himachal destinations.
    This endpoint is public — no auth required (shown on home screen).
    Cached in Redis for 15 minutes.
    """
    cache = get_redis_cache()

    # Check cache first
    cached = await cache.get_himachal_alerts()
    if cached:
        log.debug("himachal_alerts_cache_hit")
        return HimachalAlertsSummary.model_validate(cached)

    # Fetch weather for key destinations
    weather_svc = get_weather_service()
    all_alerts: list[TravelAlert] = []
    affected_destinations: set[str] = set()

    for destination in MONITORED_DESTINATIONS[:3]:  # Limit to top 3 for speed
        try:
            report = await weather_svc.get_weather(destination)
            if report.alerts:
                all_alerts.extend(report.alerts)
                affected_destinations.add(destination.title())

            # Add snowfall alert if high risk
            snowfall = await weather_svc.get_snowfall_risk(destination)
            if snowfall["risk"] in ("high",):
                all_alerts.append(TravelAlert(
                    alert_id=str(uuid4()),
                    type=AlertType.SNOWFALL,
                    severity=AlertSeverity.HIGH,
                    title=f"Heavy Snowfall Expected — {destination.title()}",
                    description=snowfall["message"],
                    affected_areas=[destination.title()],
                    source="OpenWeatherMap",
                    issued_at=datetime.now(timezone.utc),
                    action_recommended="Check road conditions before travelling. Consider delaying trip.",
                ))
                affected_destinations.add(destination.title())
        except Exception as e:
            log.warning("alert_fetch_failed", destination=destination, error=str(e))

    overall_severity = _compute_overall_severity(all_alerts)
    safe_to_travel = overall_severity not in (AlertSeverity.HIGH, AlertSeverity.CRITICAL)

    advisory = None
    if not safe_to_travel:
        advisory = "⚠️ Severe weather conditions in some areas. Please check specific destinations before booking."

    summary = HimachalAlertsSummary(
        active_alerts=all_alerts,
        destinations_affected=list(affected_destinations),
        overall_severity=overall_severity,
        last_updated=datetime.now(timezone.utc),
        safe_to_travel=safe_to_travel,
        advisory=advisory,
    )

    # Cache for 15 minutes
    await cache.set_himachal_alerts(summary.model_dump(mode="json"), ttl=900)

    return summary


@router.get("/{destination}")
async def get_destination_alerts(destination: str):
    """Get weather and travel alerts for a specific Himachal destination."""
    weather_svc = get_weather_service()

    try:
        report = await weather_svc.get_weather(destination)
        snowfall = await weather_svc.get_snowfall_risk(destination)

        return {
            "destination": destination,
            "weather": report.model_dump(mode="json"),
            "snowfall_risk": snowfall,
            "safe_to_travel": snowfall["risk"] not in ("high", "critical"),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log.error("destination_alert_failed", destination=destination, error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "error": {
                    "code": "WEATHER_SERVICE_ERROR",
                    "message": f"Could not fetch alerts for {destination}",
                }
            },
        )
