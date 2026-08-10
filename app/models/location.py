"""
app/models/location.py
───────────────────────
Where the rider is, as the rest of the system carries it.

The mobile app already knows this — it has a live GPS fix for the pickup pin — and
sending it with the chat request removes two questions the assistant used to have to
ask before it could be useful:

  1. "are you a Himachal resident or visiting?"  → answered by the fix (`in_himachal`)
  2. "which city are you travelling from?"        → answered by the fix (`place_name`)

Both used to cost a round trip each, and the second one gated the fare entirely: no
origin meant no distance, and no distance meant POST /fare/estimate could not be
called at all.

`app/services/geo.py` does the resolving; this is the serialisable result, stored on
the session so later turns keep the location without the app resending it.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.services import geo


class RiderLocation(BaseModel):
    """A resolved GPS fix for the rider, as persisted on their session."""

    lat: float
    lon: float
    #: True when the fix is inside Himachal Pradesh. Decides the assistant persona.
    in_himachal: bool
    #: Nearest town in the geography tables — the origin used for routes and fares.
    #: None when the fix is far from everything we know.
    place_name: Optional[str] = None
    #: Distance to that town. Small means the name IS where they are.
    place_distance_km: Optional[float] = None
    #: True when the fix is within a few km of the state line, where the outline in
    #: `geo.HIMACHAL_OUTLINE` cannot reliably say which side it falls on.
    near_border: bool = False
    #: Reported by the device, when it sends one. Purely informational — a poor fix is
    #: still used, because it is better than the nothing we had before.
    accuracy_m: Optional[float] = None
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def visitor_type(self) -> str:
        """"local" | "outstation" — the persona this location implies."""
        return "local" if self.in_himachal else "outstation"

    @property
    def is_precise(self) -> bool:
        """True when the fix sits ON a town we know rather than merely near one.

        The gate on naming the rider's pickup city out loud — see `geo.PRECISE_FIX_KM`.
        """
        return (
            self.place_name is not None
            and self.place_distance_km is not None
            and self.place_distance_km <= geo.PRECISE_FIX_KM
        )

    @classmethod
    def from_fix(cls, fix: geo.LocationFix, accuracy_m: Optional[float] = None) -> "RiderLocation":
        return cls(
            lat=fix.lat,
            lon=fix.lon,
            in_himachal=fix.in_himachal,
            place_name=fix.place_name,
            place_distance_km=fix.nearest_city_km,
            near_border=fix.near_border,
            accuracy_m=accuracy_m,
        )

    @classmethod
    def resolve(
        cls,
        lat: Optional[float],
        lon: Optional[float],
        accuracy_m: Optional[float] = None,
    ) -> Optional["RiderLocation"]:
        """Resolve raw device coordinates, or None when they are unusable.

        Unusable covers a missing pair and the (0, 0) a phone reports before it has a
        fix — see `geo.valid_coordinates`. Returning None rather than a wrong location
        keeps the agent on the ask-them path instead of pricing the wrong journey.
        """
        if lat is None or lon is None:
            return None
        fix = geo.locate(lat, lon)
        return cls.from_fix(fix, accuracy_m) if fix else None
