"""
app/services/himcab/distances.py
─────────────────────────────────
Surveyed road distances and drive times for the routes Him Cab actually runs.

Two things depend on this table:

  • the FARE — POST /api/rider/fare/estimate is distance-driven and REQUIRES
    `distanceInMeters`, so no distance means no price
  • the DRIVE TIME quoted to a local rider deciding whether to make the trip today

It used to be the ONLY source of a distance, which made it the gate on quoting at
all: it holds 80 routes, so a rider starting from anywhere else saw no price. It is
now the PREFERRED source rather than the only one — `resolve_route()` falls back to
`app/services/geo.py`, which derives both figures from city coordinates for any pair.
These surveyed numbers still win where they exist, because they are measured rather
than derived, and only the derived ones are labelled approximate to the rider.

The lookup is bidirectional, so each pair is listed once. Drive times are typical
door-to-door figures in normal conditions — nothing here observes live traffic, road
closures or snow, so they are always presented as estimates.

Adding a route: add the pair below (measured, better), or just add the town to
`geo.CITY_COORDS` (works immediately, approximate).
"""

from dataclasses import dataclass
from typing import Optional

from app.services import geo

#: (origin, destination) lowercase → (road distance km, typical drive time minutes)
ROAD_ROUTES: dict[tuple[str, str], tuple[float, int]] = {
    # ── From Amritsar ────────────────────────────────────
    ("amritsar", "dalhousie"): (190, 210),
    ("amritsar", "dharamshala"): (200, 240),
    ("amritsar", "kasol"): (360, 420),
    ("amritsar", "kullu"): (350, 420),
    ("amritsar", "manali"): (400, 480),
    ("amritsar", "shimla"): (370, 420),

    # ── From Bangalore ───────────────────────────────────
    ("bangalore", "dharamshala"): (2500, 1920),
    ("bangalore", "manali"): (2600, 1980),
    ("bangalore", "shimla"): (2300, 1800),

    # ── From Chandigarh ──────────────────────────────────
    ("chandigarh", "bir billing"): (260, 300),
    ("chandigarh", "chail"): (150, 240),
    ("chandigarh", "dalhousie"): (310, 330),
    ("chandigarh", "dharamshala"): (240, 270),
    ("chandigarh", "kasol"): (260, 300),
    ("chandigarh", "kullu"): (280, 360),
    ("chandigarh", "manali"): (310, 390),
    ("chandigarh", "shimla"): (115, 210),
    ("chandigarh", "spiti"): (480, 600),

    # ── From Dalhousie ───────────────────────────────────
    ("dalhousie", "khajjiar"): (24, 40),

    # ── From Dehradun ────────────────────────────────────
    ("dehradun", "dharamshala"): (390, 420),
    ("dehradun", "kasol"): (350, 420),
    ("dehradun", "manali"): (460, 540),
    ("dehradun", "shimla"): (220, 270),

    # ── From Delhi ───────────────────────────────────────
    ("delhi", "bir billing"): (480, 540),
    ("delhi", "chail"): (360, 420),
    ("delhi", "dalhousie"): (560, 600),
    ("delhi", "dharamshala"): (490, 540),
    ("delhi", "kasol"): (460, 510),
    ("delhi", "kullu"): (510, 600),
    ("delhi", "manali"): (570, 660),
    ("delhi", "shimla"): (370, 420),
    ("delhi", "spiti"): (680, 780),

    # ── From Dharamshala ─────────────────────────────────
    ("dharamshala", "bir billing"): (30, 60),
    ("dharamshala", "dalhousie"): (165, 180),
    ("dharamshala", "kasol"): (160, 180),
    ("dharamshala", "mcleod ganj"): (10, 20),
    ("dharamshala", "palampur"): (35, 45),

    # ── From Hyderabad ───────────────────────────────────
    ("hyderabad", "manali"): (2200, 1800),
    ("hyderabad", "shimla"): (1900, 1620),

    # ── From Jaipur ──────────────────────────────────────
    ("jaipur", "dharamshala"): (640, 660),
    ("jaipur", "manali"): (720, 780),
    ("jaipur", "shimla"): (520, 540),

    # ── From Kalpa ───────────────────────────────────────
    ("kalpa", "chitkul"): (60, 120),

    # ── From Kangra Airport ──────────────────────────────
    ("kangra airport", "dharamshala"): (16, 30),

    # ── From Kasol ───────────────────────────────────────
    ("kasol", "jibhi"): (80, 120),
    ("kasol", "manikaran"): (5, 15),

    # ── From Kolkata ─────────────────────────────────────
    ("kolkata", "dharamshala"): (2000, 1560),
    ("kolkata", "manali"): (2100, 1620),
    ("kolkata", "shimla"): (1850, 1440),

    # ── From Kullu ───────────────────────────────────────
    ("kullu", "kasol"): (42, 90),
    ("kullu", "manali"): (42, 60),
    ("kullu", "tirthan valley"): (55, 90),

    # ── From Ludhiana ────────────────────────────────────
    ("ludhiana", "dharamshala"): (220, 240),
    ("ludhiana", "kasol"): (310, 360),
    ("ludhiana", "manali"): (370, 420),
    ("ludhiana", "shimla"): (240, 270),

    # ── From Manali ──────────────────────────────────────
    ("manali", "dharamshala"): (280, 300),
    ("manali", "kasol"): (130, 150),
    ("manali", "kullu"): (42, 60),
    ("manali", "rohtang"): (51, 90),
    ("manali", "solang valley"): (14, 25),
    ("manali", "spiti"): (195, 210),

    # ── From Mandi ───────────────────────────────────────
    ("mandi", "dharamshala"): (130, 150),
    ("mandi", "kullu"): (60, 90),

    # ── From Mumbai ──────────────────────────────────────
    ("mumbai", "dharamshala"): (1850, 1440),
    ("mumbai", "manali"): (1950, 1500),
    ("mumbai", "shimla"): (1800, 1320),

    # ── From Pathankot ───────────────────────────────────
    ("pathankot", "chamba"): (115, 120),
    ("pathankot", "dalhousie"): (80, 90),
    ("pathankot", "dharamshala"): (90, 120),

    # ── From Shimla ──────────────────────────────────────
    ("shimla", "chail"): (45, 90),
    ("shimla", "dalhousie"): (320, 330),
    ("shimla", "dharamshala"): (315, 330),
    ("shimla", "kalpa"): (260, 360),
    ("shimla", "kasol"): (195, 210),
    ("shimla", "kufri"): (16, 30),
    ("shimla", "kullu"): (200, 210),
    ("shimla", "manali"): (270, 270),
    ("shimla", "narkanda"): (65, 90),
    ("shimla", "spiti"): (430, 480),
}


@dataclass(frozen=True)
class RouteEstimate:
    """How far and how long a route is, and how much to trust the numbers."""

    km: float
    duration_minutes: int
    #: True when derived from coordinates rather than surveyed. A fare built on it
    #: must be presented to the rider as approximate.
    is_approximate: bool

    @property
    def duration_text(self) -> str:
        """Drive time as a rider would say it: "6h 30m", "45m"."""
        hours, minutes = divmod(int(round(self.duration_minutes)), 60)
        if hours and minutes:
            return f"{hours}h {minutes}m"
        return f"{hours}h" if hours else f"{minutes}m"


def resolve_route(origin: str, destination: str) -> Optional[RouteEstimate]:
    """Distance and drive time for ANY pair of known places, or None if unrecognised.

    Surveyed first, computed second. Requiring a surveyed pair was what stopped fares
    appearing at all — falling back to a coordinate-derived estimate turns "no fare"
    into "approximate fare", which is the thing a rider can actually use.

    Still returns None for a place in no table: pricing or timing a route we cannot
    locate would be worse than saying we can't.
    """
    if not origin or not destination:
        return None

    # Canonicalise first, so "New Delhi" finds the route surveyed as "delhi".
    o, d = geo.canonical_name(origin), geo.canonical_name(destination)
    surveyed = ROAD_ROUTES.get((o, d)) or ROAD_ROUTES.get((d, o))
    if surveyed:
        km, minutes = surveyed
        return RouteEstimate(km=km, duration_minutes=minutes, is_approximate=False)

    estimated = geo.estimate_road_distance_km(origin, destination)
    if estimated:
        return RouteEstimate(
            km=estimated,
            duration_minutes=geo.estimate_drive_time_minutes(estimated),
            is_approximate=True,
        )
    return None
