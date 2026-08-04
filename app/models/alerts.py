"""
app/models/alerts.py
─────────────────────
Weather, snowfall, and road closure alert models.
Served to the mobile app via /v1/alerts/himachal and as SSE structured events.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AlertSeverity(str, Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(str, Enum):
    SNOWFALL = "snowfall"
    RAIN = "rain"
    LANDSLIDE = "landslide"
    ROAD_CLOSURE = "road_closure"
    FLOOD = "flood"
    TEMPERATURE_DROP = "temperature_drop"
    GENERAL = "general"


class TravelAlert(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    alert_id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    description: str
    affected_areas: list[str] = Field(default_factory=list)
    source: str = "OpenWeatherMap / IMD"
    issued_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    action_recommended: Optional[str] = None


class WeatherReport(BaseModel):
    """Full weather report for a destination."""
    destination: str
    current_temp_c: float
    feels_like_c: float
    humidity_percent: int
    wind_speed_kmh: float
    condition: str
    condition_icon: Optional[str] = None
    forecast_7day: list[dict] = Field(default_factory=list)
    alerts: list[TravelAlert] = Field(default_factory=list)
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HimachalAlertsSummary(BaseModel):
    """
    Aggregated alert summary for the Himachal Pradesh region.
    Returned by GET /v1/alerts/himachal
    """
    active_alerts: list[TravelAlert]
    destinations_affected: list[str]
    overall_severity: AlertSeverity
    last_updated: datetime
    safe_to_travel: bool
    advisory: Optional[str] = None
