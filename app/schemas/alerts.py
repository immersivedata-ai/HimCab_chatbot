"""
app/schemas/alerts.py
──────────────────────
Response body for GET /v1/alerts/himachal.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from app.domain.alerts import AlertSeverity, TravelAlert


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
