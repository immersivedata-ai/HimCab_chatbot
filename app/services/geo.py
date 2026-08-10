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
"""

from __future__ import annotations

import math
import re
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


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    earth_radius_km = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(h))


def estimate_road_distance_km(origin: str, destination: str) -> Optional[float]:
    """Approximate road distance in km, or None when either place is unknown.

    Approximate is the point: it exists so a rider from a city nobody thought to list
    still gets a price. Callers must present the resulting fare as an estimate.
    """
    a, b = find_city(origin), find_city(destination)
    if a is None or b is None:
        return None

    straight_line = haversine_km(a[1], b[1])
    in_hp = (a[0] in HIMACHAL_COORDS, b[0] in HIMACHAL_COORDS)
    if all(in_hp):
        bands = _FACTORS_HILLS
    elif any(in_hp):
        bands = _FACTORS_MIXED
    else:
        bands = _FACTORS_PLAINS

    return round(straight_line * _winding_factor(bands, straight_line), 1)


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
