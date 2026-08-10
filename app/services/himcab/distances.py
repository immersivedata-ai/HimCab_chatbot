"""
app/services/himcab/distances.py
─────────────────────────────────
Surveyed road distances for the routes Him Cab actually runs, in kilometres.

POST /api/rider/fare/estimate is distance-driven — it REQUIRES `distanceInMeters`, so
no distance means no fare. This table used to be the ONLY source of one, which made it
the gate on quoting at all: it holds 80 pairs, so a rider starting from anywhere else
saw an itinerary with no price. That was the bug.

It is now the PREFERRED source rather than the only one. `resolve_distance()` falls
back to `app/services/geo.py`, which computes a distance from city coordinates for any
pair — so an unlisted route gets an approximate fare instead of none, clearly labelled
as approximate. These surveyed numbers still win where they exist, because they are
measured rather than derived.

The lookup is bidirectional, so each pair is listed once. These are road distances, not
straight-line, and are used ONLY to price a route — never to state a travel time, since
nothing here measures traffic or road conditions.

Adding a route: add the pair below (better accuracy), or just add the town to
`geo.CITY_COORDS` (works immediately, approximate).
"""

from dataclasses import dataclass
from typing import Optional

from app.services import geo
from app.services.geo import estimate_road_distance_km

#: (origin, destination) in lowercase → approximate road distance in km.
ROAD_DISTANCES_KM: dict[tuple[str, str], float] = {
    # ── From Amritsar ────────────────────────────────────────────
    ("amritsar", "dalhousie"): 190,
    ("amritsar", "dharamshala"): 200,
    ("amritsar", "kasol"): 360,
    ("amritsar", "kullu"): 350,
    ("amritsar", "manali"): 400,
    ("amritsar", "shimla"): 370,

    # ── From Bangalore ───────────────────────────────────────────
    ("bangalore", "dharamshala"): 2500,
    ("bangalore", "manali"): 2600,
    ("bangalore", "shimla"): 2300,

    # ── From Chandigarh ──────────────────────────────────────────
    ("chandigarh", "bir billing"): 260,
    ("chandigarh", "chail"): 150,
    ("chandigarh", "dalhousie"): 310,
    ("chandigarh", "dharamshala"): 240,
    ("chandigarh", "kasol"): 260,
    ("chandigarh", "kullu"): 280,
    ("chandigarh", "manali"): 310,
    ("chandigarh", "shimla"): 115,
    ("chandigarh", "spiti"): 480,

    # ── From Dalhousie ───────────────────────────────────────────
    ("dalhousie", "khajjiar"): 24,

    # ── From Dehradun ────────────────────────────────────────────
    ("dehradun", "dharamshala"): 390,
    ("dehradun", "kasol"): 350,
    ("dehradun", "manali"): 460,
    ("dehradun", "shimla"): 220,

    # ── From Delhi ───────────────────────────────────────────────
    ("delhi", "bir billing"): 480,
    ("delhi", "chail"): 360,
    ("delhi", "dalhousie"): 560,
    ("delhi", "dharamshala"): 490,
    ("delhi", "kasol"): 460,
    ("delhi", "kullu"): 510,
    ("delhi", "manali"): 570,
    ("delhi", "shimla"): 370,
    ("delhi", "spiti"): 680,

    # ── From Dharamshala ─────────────────────────────────────────
    ("dharamshala", "bir billing"): 30,
    ("dharamshala", "dalhousie"): 165,
    ("dharamshala", "kasol"): 160,
    ("dharamshala", "mcleod ganj"): 10,
    ("dharamshala", "palampur"): 35,

    # ── From Hyderabad ───────────────────────────────────────────
    ("hyderabad", "manali"): 2200,
    ("hyderabad", "shimla"): 1900,

    # ── From Jaipur ──────────────────────────────────────────────
    ("jaipur", "dharamshala"): 640,
    ("jaipur", "manali"): 720,
    ("jaipur", "shimla"): 520,

    # ── From Kalpa ───────────────────────────────────────────────
    ("kalpa", "chitkul"): 60,

    # ── From Kangra Airport ──────────────────────────────────────
    ("kangra airport", "dharamshala"): 16,

    # ── From Kasol ───────────────────────────────────────────────
    ("kasol", "jibhi"): 80,
    ("kasol", "manikaran"): 5,

    # ── From Kolkata ─────────────────────────────────────────────
    ("kolkata", "dharamshala"): 2000,
    ("kolkata", "manali"): 2100,
    ("kolkata", "shimla"): 1850,

    # ── From Kullu ───────────────────────────────────────────────
    ("kullu", "kasol"): 42,
    ("kullu", "manali"): 42,
    ("kullu", "tirthan valley"): 55,

    # ── From Ludhiana ────────────────────────────────────────────
    ("ludhiana", "dharamshala"): 220,
    ("ludhiana", "kasol"): 310,
    ("ludhiana", "manali"): 370,
    ("ludhiana", "shimla"): 240,

    # ── From Manali ──────────────────────────────────────────────
    ("manali", "dharamshala"): 280,
    ("manali", "kasol"): 130,
    ("manali", "kullu"): 42,
    ("manali", "rohtang"): 51,
    ("manali", "solang valley"): 14,
    ("manali", "spiti"): 195,

    # ── From Mandi ───────────────────────────────────────────────
    ("mandi", "dharamshala"): 130,
    ("mandi", "kullu"): 60,

    # ── From Mumbai ──────────────────────────────────────────────
    ("mumbai", "dharamshala"): 1850,
    ("mumbai", "manali"): 1950,
    ("mumbai", "shimla"): 1800,

    # ── From Pathankot ───────────────────────────────────────────
    ("pathankot", "chamba"): 115,
    ("pathankot", "dalhousie"): 80,
    ("pathankot", "dharamshala"): 90,

    # ── From Shimla ──────────────────────────────────────────────
    ("shimla", "chail"): 45,
    ("shimla", "dalhousie"): 320,
    ("shimla", "dharamshala"): 315,
    ("shimla", "kalpa"): 260,
    ("shimla", "kasol"): 195,
    ("shimla", "kufri"): 16,
    ("shimla", "kullu"): 200,
    ("shimla", "manali"): 270,
    ("shimla", "narkanda"): 65,
    ("shimla", "spiti"): 430,
}


@dataclass(frozen=True)
class RouteDistance:
    """How far a route is, and how much to trust the number."""

    km: float
    #: True when derived from coordinates rather than surveyed. The fare built on it
    #: must be presented to the rider as an estimate.
    is_approximate: bool


def resolve_distance(origin: str, destination: str) -> Optional[RouteDistance]:
    """Distance for ANY pair of known places, or None if a place is unrecognised.

    Surveyed first, computed second. Requiring a surveyed pair was what stopped fares
    appearing at all: the table holds 80 routes, so a rider from Gurgaon, Ambala,
    Mandi or Solan hit `None` and the itinerary shipped with no price. Falling back to
    a coordinate-derived estimate turns "no fare" into "approximate fare", which is
    the thing a rider can actually use.

    Still returns None for a place that is in no table — pricing a route we cannot
    locate would be worse than saying we can't.
    """
    if not origin or not destination:
        return None

    # Canonicalise first, so "New Delhi" finds the route surveyed as "delhi".
    o, d = geo.canonical_name(origin), geo.canonical_name(destination)
    surveyed = ROAD_DISTANCES_KM.get((o, d)) or ROAD_DISTANCES_KM.get((d, o))
    if surveyed:
        return RouteDistance(km=surveyed, is_approximate=False)

    estimated = estimate_road_distance_km(origin, destination)
    if estimated:
        return RouteDistance(km=estimated, is_approximate=True)
    return None
