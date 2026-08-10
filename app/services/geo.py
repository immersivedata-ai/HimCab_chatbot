"""
app/services/geo.py
────────────────────
Where places are, and how far apart they are by road.

The Him Cab fare endpoint is distance-driven: POST /fare/estimate REQUIRES
`distanceInMeters`, so a route with no distance gets no quote at all. That used to be
decided by a hand-listed table of city PAIRS, which is O(n²) to maintain and was
missing most of the country — a rider travelling from Gurgaon, Ambala, Mandi or Solan
simply never saw a price.

This module replaces the pair table as the *general* case with a table of city
COORDINATES, which is O(n), stable, and something anyone can extend from an atlas.
Distance is then computed rather than looked up:

    great-circle distance x a road-winding factor  (see ROAD_WINDING_FACTOR)

Curated pair distances still win where they exist (`himcab/distances.py`) — they are
measured, not derived. The estimate covers everything else, and is reported as
approximate so the reply can say so.

It also answers the question the chat API asks before anything else: GIVEN A GPS FIX,
is this rider a Himachal local or a visitor travelling in? See `locate()` — the fix is
tested against `HIMACHAL_OUTLINE` and matched to the nearest town we know, which is
what turns a pair of coordinates into an origin the fare endpoint can be called with.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

# ── City coordinates (lat, lon) ───────────────────────────────────────────────
# Himachal destinations plus the origin cities riders actually travel from. Adding a
# city is one line here and it immediately works for BOTH fares and weather — no pair
# table to update.

HIMACHAL_COORDS: dict[str, tuple[float, float]] = {
    # ── Kangra / Dharamshala belt ────────────────────────────────────────────
    "dharamshala":     (32.2190, 76.3234),
    "mcleod ganj":     (32.2431, 76.3214),
    "kangra":          (32.0998, 76.2691),
    "kangra airport":  (32.1651, 76.2634),
    "palampur":        (32.1100, 76.5400),
    "bir billing":     (32.0392, 76.7125),
    "jogindernagar":   (31.9880, 76.7900),
    "baijnath":        (32.0500, 76.6500),
    "nurpur":          (32.3000, 75.8900),
    # ── Kullu / Manali belt ──────────────────────────────────────────────────
    "manali":          (32.2396, 77.1887),
    "kullu":           (31.9579, 77.1095),
    "kasol":           (32.0109, 77.3160),
    "manikaran":       (32.0050, 77.3450),
    "solang valley":   (32.3200, 77.1700),
    "rohtang":         (32.3720, 77.2470),
    "tirthan valley":  (31.6300, 77.4000),
    "jibhi":           (31.6200, 77.3400),
    "banjar":          (31.6360, 77.3400),
    "barot":           (32.0500, 76.8500),
    "bhuntar":         (31.8800, 77.1500),
    # ── Shimla belt ──────────────────────────────────────────────────────────
    "shimla":          (31.1048, 77.1734),
    "kufri":           (31.0985, 77.2675),
    "chail":           (30.9632, 77.2085),
    "narkanda":        (31.2600, 77.4500),
    "rampur":          (31.4500, 77.6300),
    "sarahan":         (31.5100, 77.7930),
    "kasauli":         (30.8990, 76.9650),
    "solan":           (30.9045, 77.0967),
    "parwanoo":        (30.8370, 76.9610),
    "baddi":           (30.9578, 76.7914),
    # ── Chamba / Dalhousie belt ──────────────────────────────────────────────
    "dalhousie":       (32.5383, 75.9738),
    "khajjiar":        (32.5447, 76.0527),
    "chamba":          (32.5560, 76.1260),
    "pangi":           (33.0000, 76.5000),
    # ── Kinnaur / Spiti / Lahaul ─────────────────────────────────────────────
    "kinnaur":         (31.5800, 78.3000),
    "kalpa":           (31.5390, 78.2570),
    "reckong peo":     (31.5390, 78.2740),
    "chitkul":         (31.3530, 78.4370),
    "sangla":          (31.4200, 78.2500),
    "spiti":           (32.2461, 78.0338),
    "kaza":            (32.2260, 78.0710),
    "keylong":         (32.5710, 77.0300),
    "lahaul":          (32.5710, 77.0300),
    # ── Mandi / Bilaspur / Hamirpur / Una ────────────────────────────────────
    "mandi":           (31.7100, 76.9300),
    "bilaspur":        (31.3300, 76.7500),
    "hamirpur":        (31.6861, 76.5213),
    "una":             (31.4685, 76.2708),
    # ── Sirmaur ──────────────────────────────────────────────────────────────
    "nahan":           (30.5590, 77.2960),
    "paonta sahib":    (30.4360, 77.6250),
    "renuka":          (30.6100, 77.4500),
    "sirmaur":         (30.5590, 77.2960),
}

#: Origin cities outside Himachal. These are where riders travel FROM.
GATEWAY_COORDS: dict[str, tuple[float, float]] = {
    "delhi":       (28.6139, 77.2090),
    "gurgaon":     (28.4595, 77.0266),
    "noida":       (28.5355, 77.3910),
    "faridabad":   (28.4089, 77.3178),
    "ghaziabad":   (28.6692, 77.4538),
    "meerut":      (28.9845, 77.7064),
    "chandigarh":  (30.7333, 76.7794),
    "mohali":      (30.7046, 76.7179),
    "panchkula":   (30.6942, 76.8606),
    "ambala":      (30.3782, 76.7767),
    "karnal":      (29.6857, 76.9905),
    "panipat":     (29.3909, 76.9635),
    "kurukshetra": (29.9695, 76.8783),
    "ludhiana":    (30.9010, 75.8573),
    "jalandhar":   (31.3260, 75.5762),
    "amritsar":    (31.6340, 74.8723),
    "pathankot":   (32.2643, 75.6421),
    "bathinda":    (30.2110, 74.9455),
    "patiala":     (30.3398, 76.3869),
    "jammu":       (32.7266, 74.8570),
    "srinagar":    (34.0837, 74.7973),
    "dehradun":    (30.3165, 78.0322),
    "haridwar":    (29.9457, 78.1642),
    "rishikesh":   (30.0869, 78.2676),
    "jaipur":      (26.9124, 75.7873),
    "agra":        (27.1767, 78.0081),
    "lucknow":     (26.8467, 80.9462),
    "kanpur":      (26.4499, 80.3319),
    "varanasi":    (25.3176, 82.9739),
    "patna":       (25.5941, 85.1376),
    "indore":      (22.7196, 75.8577),
    "bhopal":      (23.2599, 77.4126),
    "ahmedabad":   (23.0225, 72.5714),
    "surat":       (21.1702, 72.8311),
    "mumbai":      (19.0760, 72.8777),
    "pune":        (18.5204, 73.8567),
    "nagpur":      (21.1458, 79.0882),
    "hyderabad":   (17.3850, 78.4867),
    "bangalore":   (12.9716, 77.5946),
    "chennai":     (13.0827, 80.2707),
    "kochi":       (9.9312, 76.2673),
    "kolkata":     (22.5726, 88.3639),
    "guwahati":    (26.1445, 91.7362),
}

CITY_COORDS: dict[str, tuple[float, float]] = {**HIMACHAL_COORDS, **GATEWAY_COORDS}

#: Spellings riders actually type, mapped onto a table key. Every place has exactly
#: ONE canonical key — alternative names live here rather than as duplicate entries
#: above, so a surveyed route keyed on "delhi" is still found when someone types
#: "New Delhi" (see `canonical_name`).
_ALIASES: dict[str, str] = {
    "new delhi": "delhi",
    "gurugram": "gurgaon",
    "bengaluru": "bangalore",
    "mcleodganj": "mcleod ganj",
    "macleodganj": "mcleod ganj",
    "mcleod": "mcleod ganj",
    "dharmshala": "dharamshala",
    "dharamsala": "dharamshala",
    "gaggal": "kangra airport",
    "kaza spiti": "kaza",
    "spiti valley": "spiti",
    "lahaul spiti": "spiti",
    "parvati valley": "kasol",
    "peo": "reckong peo",
    "reckongpeo": "reckong peo",
    "shimla city": "shimla",
    "simla": "shimla",
    "kulu": "kullu",
    "banglore": "bangalore",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "ncr": "delhi",
    "delhi ncr": "delhi",
    "new delhi railway station": "delhi",
}

# ── Road-winding factors ──────────────────────────────────────────────────────
# Great-circle distance badly under-states a road route, especially in the hills. The
# multipliers below are CALIBRATED, not guessed: each is the median ratio of the
# measured distance to the great-circle distance across the 80 surveyed routes in
# `himcab/distances.py`. Re-derive them any time with:
#
#     python scripts/check_fare.py --calibrate
#
# A single average would be wrong in both directions, because the relationship depends
# on the terrain AND the length of the trip: a Himachal hop is nearly all switchbacks,
# while a 700 km approach from Delhi is mostly straight highway with a hill tail. So
# the factor is banded on the great-circle distance.
#
# Each entry is (great-circle distance below which this applies, multiplier).

#: Both ends inside Himachal — switchbacks the whole way, and it barely eases off.
_FACTORS_HILLS = ((120, 1.95), (float("inf"), 2.10))
#: One end outside Himachal — the longer the trip, the more of it is highway.
_FACTORS_MIXED = ((120, 2.05), (300, 1.67), (float("inf"), 1.24))
#: Neither end in Himachal — ordinary highway circuity.
_FACTORS_PLAINS = ((float("inf"), 1.25),)


def _winding_factor(bands: tuple, straight_line_km: float) -> float:
    for limit, factor in bands:
        if straight_line_km < limit:
            return factor
    return bands[-1][1]

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalize(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace."""
    return " ".join(_NON_ALNUM.sub(" ", (name or "").lower()).split())


def find_city(name: str) -> Optional[tuple[str, tuple[float, float]]]:
    """Resolve a user-typed place to (canonical_key, (lat, lon)), or None.

    Matching is deliberately conservative: exact, then alias, then a whole-word
    containment check. It never guesses a nearest city — an unknown place returns
    None so the caller can degrade honestly instead of pricing the wrong route.
    """
    key = normalize(name)
    if not key:
        return None
    if key in CITY_COORDS:
        return key, CITY_COORDS[key]
    if key in _ALIASES:
        canonical = _ALIASES[key]
        return canonical, CITY_COORDS[canonical]

    # "manali himachal", "shimla, hp" → match on a contained table key.
    words = set(key.split())
    for candidate, coords in CITY_COORDS.items():
        if set(candidate.split()) <= words:
            return candidate, coords
    for alias, canonical in _ALIASES.items():
        if set(alias.split()) <= words:
            return canonical, CITY_COORDS[canonical]
    return None


def canonical_name(name: str) -> str:
    """The one spelling a place is filed under, or the normalised input if unknown.

    Used before any table lookup keyed by name, so "New Delhi", "Gurugram" and
    "McLeodGanj" reach the same row as "delhi", "gurgaon" and "mcleod ganj".
    """
    found = find_city(name)
    return found[0] if found else normalize(name)


def is_in_himachal(name: str) -> Optional[bool]:
    """True/False if the place is known, None if it isn't in the table at all."""
    found = find_city(name)
    if found is None:
        return None
    return found[0] in HIMACHAL_COORDS


# ── Where the state actually is ───────────────────────────────────────────────
# A coarse outline of Himachal Pradesh, walked clockwise from the Una corner in the
# southwest. This is what decides "local or visitor" from a GPS fix, so it is checked
# against every town in the tables above plus the gateway cities riders travel from:
# 70 of 71 classify correctly (see tests/unit/test_geo_coordinates.py).
#
# The one it cannot get right is Kalka, which is in Haryana about 2 km from Parwanoo in
# Himachal — no outline this coarse separates them, and pretending otherwise would be
# worse than saying so. Fixes within a few km of the border are therefore a judgement
# call, which is why `locate()` reports `near_border` and the chat API lets a rider's
# own words override the classification.
HIMACHAL_OUTLINE: tuple[tuple[float, float], ...] = (
    (31.28, 76.27),   # Una district, south tip
    (30.92, 76.74),   # Nalagarh / Baddi, on the Haryana line
    (30.80, 76.92),   # Parwanoo, above Kalka
    (30.72, 77.10),   # Kasauli foothills
    (30.52, 77.28),   # Nahan, southern Sirmaur
    (30.40, 77.62),   # Paonta Sahib, on the Yamuna
    (30.78, 77.88),   # up the Yamuna, Uttarakhand line
    (31.05, 78.05),
    (31.26, 78.58),   # Baspa valley / Chitkul, Uttarakhand-Tibet corner
    (31.80, 78.78),   # Shipki La
    (32.05, 78.85),   # Kinnaur, Tibet line
    (32.35, 78.72),   # eastern Spiti
    (32.62, 78.30),   # northern Spiti
    (32.78, 77.80),   # Lahaul
    (32.98, 77.20),   # Baralacha, the Ladakh line
    (32.90, 76.75),
    (33.25, 76.50),   # Pangi, the northern tip
    (33.02, 76.08),
    (32.72, 75.85),   # western Chamba
    (32.33, 75.72),   # Nurpur, the northwest corner
    (32.02, 75.95),   # Indora / western Kangra
    (31.72, 76.02),
    (31.48, 75.98),   # western Una
)

#: A fix this far from the outline is close enough that the classification could go
#: either way. Reported, never used to flip the answer — see `LocationFix.near_border`.
_BORDER_TOLERANCE_KM = 8.0

#: Beyond this from the nearest town we know, the resolved place name is a region
#: rather than a location, and any distance built on it is labelled approximate.
_NEAREST_CITY_MAX_KM = 60.0

#: Within this of a town, the fix IS that town — close enough to call it the rider's
#: pickup city out loud. Further out the coordinates still measure the distance, but
#: the name is only the nearest thing we can point at, so the agent asks instead of
#: announcing "your trip from Mandi" to somebody who has never been there.
PRECISE_FIX_KM = 15.0


def point_in_himachal(lat: float, lon: float) -> bool:
    """True when the fix falls inside `HIMACHAL_OUTLINE`.

    Ray casting in (lon, lat) space. Accurate to the outline, which is coarse near the
    border by construction — read the note on `HIMACHAL_OUTLINE` before trusting a
    result within a few kilometres of it.
    """
    inside = False
    n = len(HIMACHAL_OUTLINE)
    for i in range(n):
        lat1, lon1 = HIMACHAL_OUTLINE[i]
        lat2, lon2 = HIMACHAL_OUTLINE[(i + 1) % n]
        if (lat1 > lat) != (lat2 > lat):
            crossing_lon = lon1 + (lat - lat1) * (lon2 - lon1) / (lat2 - lat1)
            if lon < crossing_lon:
                inside = not inside
    return inside


def _distance_to_outline_km(lat: float, lon: float) -> float:
    """Shortest great-circle distance from a fix to the state boundary."""
    # Degrees are projected onto kilometres locally: one degree of latitude is ~111 km
    # everywhere, one of longitude shrinks with the cosine of the latitude. Over the
    # few tens of kilometres this is used for, that is well inside the error of the
    # outline itself.
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(lat))
    px, py = lon * km_per_deg_lon, lat * km_per_deg_lat

    best = float("inf")
    n = len(HIMACHAL_OUTLINE)
    for i in range(n):
        lat1, lon1 = HIMACHAL_OUTLINE[i]
        lat2, lon2 = HIMACHAL_OUTLINE[(i + 1) % n]
        ax, ay = lon1 * km_per_deg_lon, lat1 * km_per_deg_lat
        bx, by = lon2 * km_per_deg_lon, lat2 * km_per_deg_lat
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        # Project the point onto the segment, clamped to its ends.
        t = 0.0 if span == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / span))
        best = min(best, math.hypot(px - (ax + dx * t), py - (ay + dy * t)))
    return best


def nearest_city(
    lat: float, lon: float, *, within_km: float = _NEAREST_CITY_MAX_KM
) -> Optional[tuple[str, float]]:
    """The closest place we know to a fix, as (canonical key, straight-line km).

    None when nothing is within `within_km`: naming a town 200 km away as the rider's
    location would put a wrong origin into the fare request, and a wrong price is worse
    than no price.
    """
    best_name, best_km = None, float("inf")
    for name, coords in CITY_COORDS.items():
        km = haversine_km((lat, lon), coords)
        if km < best_km:
            best_name, best_km = name, km
    if best_name is None or best_km > within_km:
        return None
    return best_name, round(best_km, 1)


@dataclass(frozen=True)
class LocationFix:
    """What a GPS fix tells us about the rider.

    `in_himachal` is the whole point: it is what routes the chat to the local or the
    outstation assistant, and what makes a rider's pickup city known without asking.
    """

    lat: float
    lon: float
    in_himachal: bool
    #: Closest town in the geography tables, or None when the fix is far from all of
    #: them. Used as the origin for the route and the fare.
    nearest_city: Optional[str] = None
    #: How far that town is. Small means the name IS the rider's location; large means
    #: it is only the nearest thing we can name.
    nearest_city_km: Optional[float] = None
    #: True when the fix is close enough to the state line that the classification
    #: could reasonably go either way. The reply never contradicts a rider who says
    #: otherwise — see `chat_session.resolve_persona`.
    near_border: bool = False

    @property
    def place_name(self) -> Optional[str]:
        """The town to show the rider and send to the fare service, title-cased."""
        return self.nearest_city.title() if self.nearest_city else None

    @property
    def is_precise(self) -> bool:
        """True when the fix sits on a town we know rather than merely near one."""
        return self.nearest_city_km is not None and self.nearest_city_km <= PRECISE_FIX_KM

    @property
    def visitor_type(self) -> str:
        """"local" | "outstation" — the persona this fix implies."""
        return "local" if self.in_himachal else "outstation"


def valid_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
    """True for a usable Earth coordinate pair.

    (0, 0) is rejected on purpose: it is in the Atlantic, and it is what a phone sends
    when it has no fix yet. Treating that as a real location would classify a Shimla
    rider as an outstation visitor 6,000 km away.
    """
    if lat is None or lon is None:
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    return not (abs(lat) < 0.01 and abs(lon) < 0.01)


def locate(lat: float, lon: float) -> Optional[LocationFix]:
    """Turn a GPS fix into everything the agent needs from it, or None if unusable."""
    if not valid_coordinates(lat, lon):
        return None

    inside = point_in_himachal(lat, lon)
    found = nearest_city(lat, lon)
    return LocationFix(
        lat=lat,
        lon=lon,
        in_himachal=inside,
        nearest_city=found[0] if found else None,
        nearest_city_km=found[1] if found else None,
        near_border=_distance_to_outline_km(lat, lon) <= _BORDER_TOLERANCE_KM,
    )


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    earth_radius_km = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(h))


def _road_km(
    a: tuple[float, float],
    a_in_himachal: bool,
    b: tuple[float, float],
    b_in_himachal: bool,
) -> float:
    """Great-circle distance between two points, wound up to a road distance."""
    straight_line = haversine_km(a, b)
    in_hp = (a_in_himachal, b_in_himachal)
    if all(in_hp):
        bands = _FACTORS_HILLS
    elif any(in_hp):
        bands = _FACTORS_MIXED
    else:
        bands = _FACTORS_PLAINS
    return round(straight_line * _winding_factor(bands, straight_line), 1)


def estimate_road_distance_km(origin: str, destination: str) -> Optional[float]:
    """Approximate road distance in km, or None when either place is unknown.

    Approximate is the point: it exists so a rider from a city nobody thought to list
    still gets a price. Callers must present the resulting fare as an estimate.
    """
    a, b = find_city(origin), find_city(destination)
    if a is None or b is None:
        return None
    return _road_km(a[1], a[0] in HIMACHAL_COORDS, b[1], b[0] in HIMACHAL_COORDS)


def road_distance_from_point(
    lat: float, lon: float, destination: str, *, in_himachal: Optional[bool] = None
) -> Optional[float]:
    """Approximate road distance from a GPS fix to a named place, or None if unknown.

    Measured from where the rider actually is rather than from the centroid of the
    nearest town, which is the difference between pricing their trip and pricing
    somebody else's: a fix in Solang Valley is 14 km from Manali, and quoting the
    Manali → Shimla fare for it silently drops that leg.
    """
    target = find_city(destination)
    if target is None or not valid_coordinates(lat, lon):
        return None
    origin_in_hp = point_in_himachal(lat, lon) if in_himachal is None else in_himachal
    return _road_km((lat, lon), origin_in_hp, target[1], target[0] in HIMACHAL_COORDS)


# ── Drive times ───────────────────────────────────────────────────────────────
# Average speeds in km/h, also calibrated from the surveyed routes (distance divided
# by measured drive time). Speed rises with trip length because a long run is mostly
# highway while a short one is all hill road: the surveyed median is 35 km/h under
# 120 km and 58 km/h over 400 km. A single average would badly under-state a short
# hop, which is exactly the trip a Himachal local is asking about.
#
# Each entry is (distance below which this applies, average km/h).
_AVERAGE_SPEEDS_KMH = ((120, 35.0), (400, 50.0), (float("inf"), 58.0))


def estimate_drive_time_minutes(distance_km: float) -> int:
    """Typical drive time for a road distance, in minutes.

    An estimate in ordinary conditions — nothing here sees traffic, roadworks or
    snow, so callers must present it as approximate. Rounded to 5 minutes, because
    a figure like "4h 07m" implies a precision this does not have.
    """
    for limit, speed in _AVERAGE_SPEEDS_KMH:
        if distance_km < limit:
            break
    minutes = (distance_km / speed) * 60
    return max(5, int(round(minutes / 5) * 5))
