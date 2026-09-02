"""
app/domain/alerts.py
─────────────────────
Weather and travel-advisory entities.

`WeatherReport` is what the weather provider's response is parsed INTO, and
`TravelAlert` is one advisory derived from it. Both are consumed by the agent (the
tool executor stores them on state, the writer stamps the report onto the itinerary),
so they are domain objects rather than response shapes. The aggregate the alerts
endpoint returns is a wire format and lives in app/schemas/alerts.py.
"""

from datetime import UTC, datetime
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
    issued_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
