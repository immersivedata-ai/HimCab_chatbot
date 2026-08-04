"""
app/services/external/places.py
─────────────────────────────────
OpenStreetMap Overpass API wrapper — 100% free, no API key required.
Fetches hotels, restaurants, temples, cafes, and tourist attractions.
Results cached in Redis for 24h.

Drive times replaced with a hardcoded table of known Himachal routes —
Gemini already knows these, and Him Cab drivers certainly do too.
"""

from typing import Optional

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.exceptions import PlacesServiceError
from app.core.logging import get_logger
from app.services.redis_cache import get_redis_cache

log = get_logger(__name__)

# ── Overpass API tag mapping per category ─────────────────────────────────────
# Uses OpenStreetMap tags: https://wiki.openstreetmap.org/wiki/Tags
CATEGORY_OSM_TAGS: dict[str, str] = {
    "hotels":      '["tourism"~"hotel|guest_house|hostel|motel|resort"]',
    "restaurants": '["amenity"~"restaurant|food_court|fast_food"]',
    "temples":     '["amenity"~"place_of_worship"]["religion"~"hindu|buddhist|sikh"]',
    "cafes":       '["amenity"="cafe"]',
    "attractions": '["tourism"~"attraction|viewpoint|museum|monastery|camp_site|picnic_site"]',
    "adventure":   '["sport"~"hiking|climbing|paragliding|skiing"]',
    "shopping":    '["shop"~"general|supermarket|clothes|souvenir|gift"]',
}

# Himachal destination bounding boxes (south, west, north, east)
HIMACHAL_BBOX: dict[str, tuple[float, float, float, float]] = {
    "dharamshala":    (32.18, 76.28, 32.28, 76.40),
    "mcleod ganj":   (32.22, 76.30, 32.27, 76.35),
    "manali":         (32.20, 77.15, 32.28, 77.22),
    "shimla":         (31.07, 77.14, 31.14, 77.22),
    "kullu":          (31.93, 77.05, 31.99, 77.15),
    "kasol":          (32.00, 77.28, 32.03, 77.34),
    "spiti":          (32.10, 77.90, 32.40, 78.20),
    "dalhousie":      (32.50, 75.94, 32.56, 76.00),
    "chail":          (30.93, 77.16, 30.99, 77.22),
    "kufri":          (31.09, 77.23, 31.12, 77.28),
    "bir billing":    (32.02, 76.70, 32.06, 76.75),
    "khajjiar":       (32.53, 76.02, 32.57, 76.07),
    # ── Additional popular destinations ──────────────────────────────────────
    "kinnaur":        (31.40, 78.10, 31.70, 78.50),
    "chitkul":        (31.34, 78.42, 31.38, 78.46),
    "tirthan valley": (31.60, 77.35, 31.68, 77.45),
    "jibhi":          (31.60, 77.32, 31.64, 77.38),
    "sangla":         (31.40, 78.20, 31.45, 78.28),
    "narkanda":       (31.25, 77.43, 31.28, 77.48),
    "manikaran":      (31.99, 77.33, 32.02, 77.37),
    "solang valley":  (32.30, 77.15, 32.34, 77.20),
    "rohtang":        (32.35, 77.20, 32.40, 77.28),
    "kalpa":          (31.52, 78.24, 31.55, 78.28),
    "mandi":          (31.69, 76.92, 31.73, 76.96),
    "palampur":       (32.09, 76.52, 32.13, 76.56),
}

DEFAULT_BBOX = (31.07, 77.14, 31.14, 77.22)  # Shimla as fallback


def _get_bbox(destination: str) -> tuple[float, float, float, float]:
    """Find the bounding box for a destination using fuzzy matching."""
    dest_lower = destination.lower().strip()
    for key, bbox in HIMACHAL_BBOX.items():
        if key in dest_lower or dest_lower in key:
            return bbox
    # Try partial word matching as a second pass
    for key, bbox in HIMACHAL_BBOX.items():
        if any(word in dest_lower for word in key.split()):
            return bbox
    log.warning("bbox_fallback_used", destination=destination, fallback="shimla")
    return DEFAULT_BBOX


# ── Hardcoded drive-time table (minutes) ─────────────────────────────────────
# These are well-known Himachal routes — no Maps API needed.
DRIVE_TIMES: dict[tuple[str, str], dict] = {
    # ── Intra-Himachal routes ─────────────────────────────────────────────────
    ("shimla", "manali"):      {"duration_minutes": 270, "distance_km": 270, "route_notes": "Via NH-3 through Mandi and Kullu valley. Scenic drive along Beas river.", "road_condition": "good"},
    ("shimla", "dharamshala"): {"duration_minutes": 330, "distance_km": 315, "route_notes": "Via Bilaspur and Kangra. Mix of highway and hill roads.", "road_condition": "good"},
    ("shimla", "kasol"):       {"duration_minutes": 210, "distance_km": 195, "route_notes": "Via Mandi and Bhuntar. Last 30km narrow Parvati Valley road.", "road_condition": "moderate"},
    ("shimla", "dalhousie"):   {"duration_minutes": 330, "distance_km": 320, "route_notes": "Via Bilaspur and Pathankot. Long but well-maintained highway.", "road_condition": "good"},
    ("shimla", "spiti"):       {"duration_minutes": 480, "distance_km": 430, "route_notes": "Via Narkanda, Rampur, Reckong Peo. Extremely scenic but challenging mountain roads. Not recommended in winter.", "road_condition": "rough"},
    ("shimla", "kufri"):       {"duration_minutes": 30,  "distance_km": 16, "route_notes": "Short uphill drive from Shimla. Popular for snow activities in winter.", "road_condition": "good"},
    ("shimla", "narkanda"):    {"duration_minutes": 90,  "distance_km": 65, "route_notes": "Via NH-5 towards Kinnaur. Beautiful apple orchards en route.", "road_condition": "good"},
    ("shimla", "chail"):       {"duration_minutes": 90,  "distance_km": 45, "route_notes": "Winding hill road through dense forests. World's highest cricket ground.", "road_condition": "moderate"},
    ("shimla", "kullu"):       {"duration_minutes": 210, "distance_km": 200, "route_notes": "Via Mandi on NH-3. Passes through Sundernagar and Pandoh Dam.", "road_condition": "good"},
    ("manali", "kasol"):       {"duration_minutes": 150, "distance_km": 130, "route_notes": "Via Bhuntar. Scenic drive through Kullu and Parvati valleys.", "road_condition": "moderate"},
    ("manali", "spiti"):       {"duration_minutes": 210, "distance_km": 195, "route_notes": "Via Rohtang Pass and Kunzum Pass. Open June-October only. Permit required.", "road_condition": "rough"},
    ("manali", "dharamshala"): {"duration_minutes": 300, "distance_km": 280, "route_notes": "Via Mandi and Kangra. Passes through Kullu valley and Kangra plains.", "road_condition": "good"},
    ("manali", "solang valley"): {"duration_minutes": 25, "distance_km": 14, "route_notes": "Short drive north of Manali. Adventure sports hub — paragliding, skiing.", "road_condition": "good"},
    ("manali", "rohtang"):     {"duration_minutes": 90,  "distance_km": 51, "route_notes": "Via Atal Tunnel or Rohtang Pass. Permit required. Stunning snow views.", "road_condition": "moderate"},
    ("manali", "kullu"):       {"duration_minutes": 60,  "distance_km": 42, "route_notes": "Easy highway drive along the Beas river. Kullu shawl market.", "road_condition": "good"},
    ("dharamshala", "dalhousie"): {"duration_minutes": 180, "distance_km": 165, "route_notes": "Via Nurpur and Chamba road. Scenic but winding hill roads.", "road_condition": "moderate"},
    ("dharamshala", "kasol"):  {"duration_minutes": 180, "distance_km": 160, "route_notes": "Via Mandi and Bhuntar. Transition from Kangra to Parvati valley.", "road_condition": "moderate"},
    ("dharamshala", "bir billing"): {"duration_minutes": 60, "distance_km": 30, "route_notes": "Short drive. World-class paragliding destination. Tea gardens en route.", "road_condition": "good"},
    ("dharamshala", "palampur"): {"duration_minutes": 45, "distance_km": 35, "route_notes": "Easy drive through tea plantations. Palampur is the tea capital of north India.", "road_condition": "good"},
    ("dharamshala", "mcleod ganj"): {"duration_minutes": 20, "distance_km": 10, "route_notes": "Short uphill drive. Tibetan monasteries and cafes.", "road_condition": "good"},
    ("kangra airport", "dharamshala"): {"duration_minutes": 30, "distance_km": 16, "route_notes": "Gaggal Airport to Dharamshala. Quick and easy transfer.", "road_condition": "good"},
    ("kasol", "manikaran"):    {"duration_minutes": 15,  "distance_km": 5, "route_notes": "Short drive along Parvati river. Famous hot springs and Gurudwara.", "road_condition": "good"},
    ("kasol", "jibhi"):        {"duration_minutes": 120, "distance_km": 80, "route_notes": "Via Banjar. Beautiful Tirthan Valley route. Hidden gem destination.", "road_condition": "moderate"},
    ("kullu", "kasol"):        {"duration_minutes": 90,  "distance_km": 42, "route_notes": "Via Bhuntar and Parvati Valley. Narrow road after Bhuntar.", "road_condition": "moderate"},
    ("kullu", "manali"):       {"duration_minutes": 60,  "distance_km": 42, "route_notes": "Easy highway drive along Beas river towards Manali.", "road_condition": "good"},
    ("dalhousie", "khajjiar"):  {"duration_minutes": 40, "distance_km": 24, "route_notes": "Short drive to 'Mini Switzerland of India'. Stunning meadow with lake.", "road_condition": "good"},
    ("kullu", "tirthan valley"): {"duration_minutes": 90, "distance_km": 55, "route_notes": "Via Banjar. Great National Park nearby. Off-beat destination.", "road_condition": "moderate"},
    ("shimla", "kalpa"):       {"duration_minutes": 360, "distance_km": 260, "route_notes": "Via Narkanda and Rampur along NH-5. Kinner Kailash views from Kalpa.", "road_condition": "moderate"},
    ("kalpa", "chitkul"):      {"duration_minutes": 120, "distance_km": 60, "route_notes": "Last village on the Indo-Tibet border. Remote and breathtaking.", "road_condition": "rough"},
    ("mandi", "kullu"):        {"duration_minutes": 90, "distance_km": 60, "route_notes": "Via NH-3 along Beas river. Pandoh Dam is a good pitstop.", "road_condition": "good"},
    ("mandi", "dharamshala"):  {"duration_minutes": 150, "distance_km": 130, "route_notes": "Via Jogindernagar and Palampur. Tea gardens and temples.", "road_condition": "good"},

    # ── From Chandigarh (major transit hub) ───────────────────────────────────
    ("chandigarh", "shimla"):      {"duration_minutes": 210, "distance_km": 115, "route_notes": "Via NH-5 (old NH-22). Well-maintained highway through Panchkula, Kalka, Solan. Toy train from Kalka is a scenic alternative.", "road_condition": "good"},
    ("chandigarh", "manali"):      {"duration_minutes": 390, "distance_km": 310, "route_notes": "Via Bilaspur, Sundernagar, Mandi, Kullu. NH-3 mostly well-maintained. Scenic Beas valley stretch.", "road_condition": "good"},
    ("chandigarh", "dharamshala"): {"duration_minutes": 270, "distance_km": 240, "route_notes": "Via Una and Kangra. Smooth highway for most of the route.", "road_condition": "good"},
    ("chandigarh", "kasol"):       {"duration_minutes": 300, "distance_km": 260, "route_notes": "Via Mandi and Bhuntar. Last 30km narrow Parvati Valley road.", "road_condition": "moderate"},
    ("chandigarh", "dalhousie"):   {"duration_minutes": 330, "distance_km": 310, "route_notes": "Via Pathankot and Chamba road. Long but scenic drive.", "road_condition": "good"},
    ("chandigarh", "kullu"):       {"duration_minutes": 360, "distance_km": 280, "route_notes": "Via Bilaspur and Mandi. Kullu valley gateway.", "road_condition": "good"},
    ("chandigarh", "spiti"):       {"duration_minutes": 600, "distance_km": 480, "route_notes": "Via Shimla, Narkanda, Rampur, Reckong Peo. Challenging but breathtaking. 2-day drive recommended.", "road_condition": "rough"},
    ("chandigarh", "bir billing"):  {"duration_minutes": 300, "distance_km": 260, "route_notes": "Via Anandpur Sahib and Kangra. World-class paragliding.", "road_condition": "good"},
    ("chandigarh", "chail"):       {"duration_minutes": 240, "distance_km": 150, "route_notes": "Via Kalka and Kandaghat. Quiet hill station near Shimla.", "road_condition": "good"},

    # ── From Delhi / NCR ──────────────────────────────────────────────────────
    ("delhi", "shimla"):       {"duration_minutes": 420, "distance_km": 370, "route_notes": "Via NH-44 to Chandigarh, then NH-5 to Shimla. Breakfast stop at Murthal dhabas recommended. Kalka-Shimla toy train is a scenic alternative for the last stretch.", "road_condition": "good"},
    ("delhi", "manali"):       {"duration_minutes": 660, "distance_km": 570, "route_notes": "Via NH-44 to Chandigarh, then NH-3 via Mandi and Kullu. Overnight bus or early morning departure recommended. Murthal, Chandigarh, and Mandi are good stops.", "road_condition": "good"},
    ("delhi", "dharamshala"):  {"duration_minutes": 540, "distance_km": 490, "route_notes": "Via NH-44 to Chandigarh, then via Kangra. Alternatively fly to Kangra (Gaggal) airport (~1.5h flight + 30min cab).", "road_condition": "good"},
    ("delhi", "kasol"):        {"duration_minutes": 510, "distance_km": 460, "route_notes": "Via Chandigarh and Mandi to Bhuntar, then Parvati Valley road. Overnight bus popular with backpackers.", "road_condition": "moderate"},
    ("delhi", "dalhousie"):    {"duration_minutes": 600, "distance_km": 560, "route_notes": "Via Chandigarh and Pathankot. Long drive — consider overnight Volvo bus.", "road_condition": "good"},
    ("delhi", "kullu"):        {"duration_minutes": 600, "distance_km": 510, "route_notes": "Via Chandigarh and Mandi. Kullu is the gateway to Manali and Parvati Valley.", "road_condition": "good"},
    ("delhi", "spiti"):        {"duration_minutes": 780, "distance_km": 680, "route_notes": "Via Shimla and Kinnaur. Extremely long — split into 2-day drive. Only accessible June-October via Kunzum Pass.", "road_condition": "rough"},
    ("delhi", "bir billing"):  {"duration_minutes": 540, "distance_km": 480, "route_notes": "Via Chandigarh and Kangra. Popular paragliding destination.", "road_condition": "good"},
    ("delhi", "chail"):        {"duration_minutes": 420, "distance_km": 360, "route_notes": "Via Chandigarh and Kandaghat. Quiet alternative to Shimla.", "road_condition": "good"},

    # ── From Amritsar ─────────────────────────────────────────────────────────
    ("amritsar", "dharamshala"): {"duration_minutes": 240, "distance_km": 200, "route_notes": "Via Pathankot and Kangra. Fastest route to Dharamshala from Punjab.", "road_condition": "good"},
    ("amritsar", "dalhousie"):   {"duration_minutes": 210, "distance_km": 190, "route_notes": "Via Pathankot. Shortest hill station from Amritsar.", "road_condition": "good"},
    ("amritsar", "manali"):      {"duration_minutes": 480, "distance_km": 400, "route_notes": "Via Mandi and Kullu. Long drive through Punjab plains then hills.", "road_condition": "good"},
    ("amritsar", "shimla"):      {"duration_minutes": 420, "distance_km": 370, "route_notes": "Via Chandigarh. Mostly highway driving.", "road_condition": "good"},
    ("amritsar", "kasol"):       {"duration_minutes": 420, "distance_km": 360, "route_notes": "Via Mandi and Bhuntar. Long but rewarding drive.", "road_condition": "moderate"},
    ("amritsar", "kullu"):       {"duration_minutes": 420, "distance_km": 350, "route_notes": "Via Mandi. Gateway to Kullu-Manali valley.", "road_condition": "good"},

    # ── From Dehradun ─────────────────────────────────────────────────────────
    ("dehradun", "shimla"):      {"duration_minutes": 270, "distance_km": 220, "route_notes": "Via Paonta Sahib and Nahan. Scenic Shivalik hills.", "road_condition": "good"},
    ("dehradun", "manali"):      {"duration_minutes": 540, "distance_km": 460, "route_notes": "Via Chandigarh and Mandi. Long but well-connected route.", "road_condition": "good"},
    ("dehradun", "dharamshala"): {"duration_minutes": 420, "distance_km": 390, "route_notes": "Via Chandigarh. Mix of highway and hill roads.", "road_condition": "good"},
    ("dehradun", "kasol"):       {"duration_minutes": 420, "distance_km": 350, "route_notes": "Via Chandigarh and Mandi. Parvati Valley at the end.", "road_condition": "moderate"},

    # ── From Ludhiana ─────────────────────────────────────────────────────────
    ("ludhiana", "shimla"):      {"duration_minutes": 270, "distance_km": 240, "route_notes": "Via Chandigarh and Kalka. Easy highway + hill road combo.", "road_condition": "good"},
    ("ludhiana", "manali"):      {"duration_minutes": 420, "distance_km": 370, "route_notes": "Via Mandi and Kullu. Central Punjab to the mountains.", "road_condition": "good"},
    ("ludhiana", "dharamshala"): {"duration_minutes": 240, "distance_km": 220, "route_notes": "Via Jalandhar and Kangra. Smooth highway for most of the drive.", "road_condition": "good"},
    ("ludhiana", "kasol"):       {"duration_minutes": 360, "distance_km": 310, "route_notes": "Via Mandi and Bhuntar into Parvati Valley.", "road_condition": "moderate"},

    # ── From Pathankot (railway gateway to Himachal) ──────────────────────────
    ("pathankot", "dharamshala"): {"duration_minutes": 120, "distance_km": 90, "route_notes": "Via Kangra. Railway gateway to Kangra valley.", "road_condition": "good"},
    ("pathankot", "dalhousie"):   {"duration_minutes": 90,  "distance_km": 80, "route_notes": "Direct hill road. Most popular route to Dalhousie.", "road_condition": "good"},
    ("pathankot", "chamba"):      {"duration_minutes": 120, "distance_km": 115, "route_notes": "Via Dalhousie or direct Chamba road. Ancient temple town.", "road_condition": "moderate"},

    # ── From Jaipur ───────────────────────────────────────────────────────────
    ("jaipur", "shimla"):      {"duration_minutes": 540, "distance_km": 520, "route_notes": "Via Delhi and Chandigarh. ~9 hours, consider overnight stop.", "road_condition": "good"},
    ("jaipur", "manali"):      {"duration_minutes": 780, "distance_km": 720, "route_notes": "Via Delhi and Chandigarh. Very long — split into 2 days or fly to Chandigarh/Kullu.", "road_condition": "good"},
    ("jaipur", "dharamshala"): {"duration_minutes": 660, "distance_km": 640, "route_notes": "Via Delhi and Chandigarh. Long drive — consider flying to Kangra.", "road_condition": "good"},

    # ── From Mumbai (long-distance) ───────────────────────────────────────────
    ("mumbai", "shimla"):      {"duration_minutes": 1320, "distance_km": 1800, "route_notes": "Very long drive via NH-48 and NH-44. Strongly recommend flight to Delhi/Chandigarh + cab. Mumbai to Chandigarh flights ~2.5h.", "road_condition": "good"},
    ("mumbai", "manali"):      {"duration_minutes": 1500, "distance_km": 1950, "route_notes": "Fly to Delhi/Chandigarh/Kullu and take cab. Kullu-Manali airport has direct flights. Road drive NOT recommended.", "road_condition": "good"},
    ("mumbai", "dharamshala"): {"duration_minutes": 1440, "distance_km": 1850, "route_notes": "Fly to Delhi + cab, or fly to Kangra (Gaggal) airport with connection. Road drive NOT recommended.", "road_condition": "good"},

    # ── From Kolkata (long-distance) ──────────────────────────────────────────
    ("kolkata", "shimla"):     {"duration_minutes": 1440, "distance_km": 1850, "route_notes": "Fly to Delhi/Chandigarh + cab. Direct road drive NOT recommended.", "road_condition": "good"},
    ("kolkata", "manali"):     {"duration_minutes": 1620, "distance_km": 2100, "route_notes": "Fly to Delhi/Chandigarh/Kullu + cab. Kullu-Manali has direct flights.", "road_condition": "good"},
    ("kolkata", "dharamshala"): {"duration_minutes": 1560, "distance_km": 2000, "route_notes": "Fly to Delhi + cab, or fly to Kangra airport. Road NOT recommended.", "road_condition": "good"},

    # ── From Bangalore (long-distance) ────────────────────────────────────────
    ("bangalore", "shimla"):     {"duration_minutes": 1800, "distance_km": 2300, "route_notes": "Fly to Delhi/Chandigarh + cab. ~3h flight + 4-7h cab.", "road_condition": "good"},
    ("bangalore", "manali"):     {"duration_minutes": 1980, "distance_km": 2600, "route_notes": "Fly to Delhi/Chandigarh/Kullu + cab. Kullu has direct flights.", "road_condition": "good"},
    ("bangalore", "dharamshala"): {"duration_minutes": 1920, "distance_km": 2500, "route_notes": "Fly to Delhi + cab or fly to Kangra airport.", "road_condition": "good"},

    # ── From Hyderabad (long-distance) ────────────────────────────────────────
    ("hyderabad", "shimla"):     {"duration_minutes": 1620, "distance_km": 1900, "route_notes": "Fly to Delhi/Chandigarh + cab. ~2.5h flight + cab.", "road_condition": "good"},
    ("hyderabad", "manali"):     {"duration_minutes": 1800, "distance_km": 2200, "route_notes": "Fly to Delhi + cab or fly to Kullu. Road NOT recommended.", "road_condition": "good"},
}


def _recommend_travel_mode(distance_km: float | None) -> str:
    """Recommend travel mode based on distance from origin."""
    if distance_km is None:
        return "cab"
    if distance_km > 800:
        return "flight+cab"
    if distance_km > 500:
        return "train+cab"
    return "cab"


def _lookup_drive_time(origin: str, destination: str) -> dict:
    """
    Look up drive time from hardcoded table (bidirectional).
    Includes a recommended travel mode based on distance.
    """
    o, d = origin.lower().strip(), destination.lower().strip()
    result = DRIVE_TIMES.get((o, d)) or DRIVE_TIMES.get((d, o))
    if result:
        mode = _recommend_travel_mode(result.get("distance_km"))
        return {
            "origin": origin,
            "destination": destination,
            "duration_minutes": result["duration_minutes"],
            "duration_hours": round(result["duration_minutes"] / 60, 1),
            "distance_km": result["distance_km"],
            "recommended_mode": mode,
        }
    # Fallback: estimate ~50 km/h average hill speed
    return {
        "origin": origin,
        "destination": destination,
        "duration_minutes": None,
        "duration_hours": None,
        "distance_km": None,
        "recommended_mode": "cab",
        "note": "Exact estimate unavailable — contact Him Cab for route timing.",
    }


# ── Overpass Service ──────────────────────────────────────────────────────────

class PlacesService:
    """
    OpenStreetMap Overpass API — free, no key required.
    Queries OSM for POIs within a destination's bounding box.
    """

    OVERPASS_URL = "https://overpass-api.de/api/interpreter"
    _TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=5.0, pool=5.0)

    def __init__(self) -> None:
        self._cache = get_redis_cache()
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client (connection pooling)."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self._TIMEOUT,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError, httpx.ReadTimeout)),
    )
    async def _fetch_overpass(self, query: str) -> list[dict]:
        """Execute an Overpass API query and return elements."""
        client = await self._get_client()
        response = await client.post(
            self.OVERPASS_URL,
            data={"data": query},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        response.raise_for_status()
        data = response.json()

        if "remark" in data:
            log.warning("overpass_remark", remark=data["remark"])

        return data.get("elements", [])

    def _parse_element(self, elem: dict) -> dict:
        """Parse a single Overpass API element into a structured place dict."""
        tags = elem.get("tags", {})
        name = (
            tags.get("name:en")
            or tags.get("name")
            or tags.get("brand")
            or "Unnamed Place"
        )

        # Get coordinates — for ways, use the 'center' field
        lat = elem.get("lat") or (elem.get("center", {}).get("lat"))
        lon = elem.get("lon") or (elem.get("center", {}).get("lon"))

        # Build a simple OSM maps link
        elem_id = elem.get("id", "")
        elem_type = elem.get("type", "node")
        maps_url = f"https://www.openstreetmap.org/{elem_type}/{elem_id}"

        return {
            "name": name,
            "address": tags.get("addr:full") or tags.get("addr:street", ""),
            "lat": lat,
            "lon": lon,
            "rating": None,          # OSM has no ratings
            "review_count": None,
            "price_level": tags.get("stars", "unknown"),
            "phone": tags.get("phone") or tags.get("contact:phone"),
            "website": tags.get("website") or tags.get("contact:website"),
            "maps_url": maps_url,
            "types": [t for t in [
                tags.get("tourism"),
                tags.get("amenity"),
                tags.get("shop"),
                tags.get("sport"),
            ] if t],
            "opening_hours": tags.get("opening_hours"),
            "source": "OpenStreetMap",
        }

    async def get_places(
        self,
        destination: str,
        category: str,
        max_results: int = 10,
    ) -> list[dict]:
        """
        Get places of interest via Overpass API (free, no key).
        Falls back to empty list on error so the AI can still generate response.
        """
        # 1. Cache check
        cached = await self._cache.get_places(destination, category)
        if cached:
            log.debug("places_cache_hit", destination=destination, category=category)
            return cached

        osm_tag = CATEGORY_OSM_TAGS.get(category, '["tourism"="attraction"]')
        s, w, n, e = _get_bbox(destination)

        # FIXED: Use valid Overpass QL syntax.
        # `out center body qt;` returns center coordinates for ways and
        # sorts by quadtile for performance. Client-side truncation handles limits.
        query = f"""
[out:json][timeout:15];
(
  node{osm_tag}({s},{w},{n},{e});
  way{osm_tag}({s},{w},{n},{e});
);
out center body qt;
"""

        try:
            elements = await self._fetch_overpass(query)

            # Filter: only elements with a name tag are useful
            places = [
                self._parse_element(el)
                for el in elements
                if el.get("tags", {}).get("name")
            ]
            places = places[:max_results]

            # Cache even empty results to avoid hammering the API
            await self._cache.set_places(destination, category, places)

            log.info(
                "places_fetched_osm",
                destination=destination,
                category=category,
                raw_elements=len(elements),
                named_places=len(places),
            )
            return places

        except httpx.HTTPStatusError as e:
            log.error(
                "overpass_http_error",
                destination=destination,
                category=category,
                status_code=e.response.status_code,
                response_text=e.response.text[:500],
            )
            return []  # graceful degradation — AI still runs

        except Exception as e:
            log.warning(
                "overpass_failed",
                destination=destination,
                category=category,
                error=str(e),
                error_type=type(e).__name__,
            )
            return []  # graceful degradation — AI still runs

    async def get_drive_time(self, origin: str, destination: str) -> dict:
        """
        Return drive time from hardcoded Himachal route table.
        No API call — instant, no cost.
        Made async to match the await call pattern in tool_executor.
        """
        return _lookup_drive_time(origin, destination)

    async def get_optimal_route(
        self,
        origin: str,
        destination: str,
        return_to_origin: bool = True,
    ) -> dict:
        """
        Compute the full route plan for an outstation visitor.
        Builds legs: origin → destination (+ destination → origin if return requested).

        Returns a dict compatible with the RoutePlan model:
        {
            "legs": [{"origin": ..., "destination": ..., "distance_km": ..., ...}],
            "total_distance_km": ...,
            "total_duration_hours": ...,
            "optimal_order": ["Delhi", "Manali", "Delhi"],
            "summary": "Delhi → Manali (11h) → Delhi (11h)"
        }
        """
        legs = []
        total_distance = 0.0
        total_duration_mins = 0
        route_order = [origin.title()]

        # Leg 1: Origin → Destination
        outbound = _lookup_drive_time(origin, destination)
        legs.append({
            "origin": origin.title(),
            "destination": destination.title(),
            "distance_km": outbound.get("distance_km"),
            "duration_hours": outbound.get("duration_hours"),
            "duration_minutes": outbound.get("duration_minutes"),
            "recommended_mode": outbound.get("recommended_mode", "cab"),
            "route_notes": outbound.get("route_notes") or outbound.get("note"),
            "road_condition": outbound.get("road_condition"),
        })
        if outbound.get("distance_km"):
            total_distance += outbound["distance_km"]
        if outbound.get("duration_minutes"):
            total_duration_mins += outbound["duration_minutes"]
        route_order.append(destination.title())

        # Leg 2: Destination → Origin (return journey)
        if return_to_origin:
            inbound = _lookup_drive_time(destination, origin)
            legs.append({
                "origin": destination.title(),
                "destination": origin.title(),
                "distance_km": inbound.get("distance_km"),
                "duration_hours": inbound.get("duration_hours"),
                "duration_minutes": inbound.get("duration_minutes"),
                "recommended_mode": inbound.get("recommended_mode", "cab"),
                "route_notes": inbound.get("route_notes") or inbound.get("note"),
                "road_condition": inbound.get("road_condition"),
            })
            if inbound.get("distance_km"):
                total_distance += inbound["distance_km"]
            if inbound.get("duration_minutes"):
                total_duration_mins += inbound["duration_minutes"]
            route_order.append(origin.title())

        # Build summary string
        summary_parts = []
        for leg in legs:
            dur_h = leg.get("duration_hours")
            dur_str = f"({dur_h}h)" if dur_h else "(time N/A)"
            summary_parts.append(f"{leg['destination']} {dur_str}")
        summary = f"{origin.title()} → " + " → ".join(summary_parts)

        return {
            "legs": legs,
            "total_distance_km": total_distance if total_distance > 0 else None,
            "total_duration_hours": round(total_duration_mins / 60, 1) if total_duration_mins > 0 else None,
            "optimal_order": route_order,
            "summary": summary,
        }

    async def close(self) -> None:
        """Close the HTTP client and release connections."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None


# ── Singleton ─────────────────────────────────────────────────────────────────
_places_service: Optional[PlacesService] = None


def get_places_service() -> PlacesService:
    global _places_service
    if _places_service is None:
        _places_service = PlacesService()
    return _places_service
