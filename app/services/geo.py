"""
app/services/geo.py
────────────────────
Journeys: where the rider is, where they are going, and the drive between.

WHAT IS AND ISN'T HERE
──────────────────────
There are no coordinates in this file. There is no list of towns, no outline of the
Himachal border, no table of road distances and no average-speed constants. All of that
used to live here, and all of it was wrong in the same way: it only knew the places and
routes somebody had already typed in.

  • a town nobody had listed had no coordinates → no route → no fare, silently
  • a hand-drawn state outline could not tell Parwanoo from Kalka, 2 km apart
  • great-circle distance scaled by a terrain factor missed by up to 170% on the
    hill routes it was supposed to be good at

Everything is now resolved live, per request:

  `app/services/external/geocoding.py`  name ↔ coordinates, and which state
  `app/services/external/routing.py`    real road distance and drive time

What remains here is the composition — turning a rider's fix and a destination name
into one `Journey` the rest of the agent can use — plus the coordinate arithmetic
(`haversine_km`) and input validation (`valid_coordinates`) that are genuinely local
concerns and involve no stored knowledge.

EVERY LOOKUP CAN FAIL, AND FAILURE IS AN ANSWER
──────────────────────────────────────────────
A provider can be unreachable and a place can be unrecognisable. Both return None, and
None must stay None: an unresolvable pickup leaves the assistant asking the rider where
they mean, which is right, whereas a guessed one prices somebody else's journey.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Optional

from app.core.logging import get_logger
from app.services.external.geocoding import Place, get_geocoding_service
from app.services.external.routing import RoadRoute, get_routing_service

log = get_logger(__name__)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")

#: Within this of a named town, the fix IS that town — close enough to call it the
#: rider's pickup out loud. Further out the coordinates still measure the journey, but
#: the name is only the nearest thing the provider could point at, so the assistant
#: asks rather than announcing "your trip from Mandi" to somebody who has never been
#: there. Reverse geocoding returns the enclosing administrative area, so this is a
#: sanity bound on how far that area's centre may be, not a lookup radius.
PRECISE_FIX_KM = 15.0


def normalize(name: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. For comparing two names."""
    return " ".join(_NON_ALNUM.sub(" ", (name or "").lower()).split())


def same_place(a: Optional[str], b: Optional[str]) -> bool:
    """True when two place names refer to the same place, ignoring spelling noise."""
    if not a or not b:
        return False
    return normalize(a) == normalize(b)


def valid_coordinates(lat: Optional[float], lon: Optional[float]) -> bool:
    """True for a usable Earth coordinate pair.

    (0, 0) is rejected on purpose: it is in the Atlantic, and it is what a phone
    reports before it has a fix. Treating that as real would classify a Shimla rider
    as a visitor 6,000 km away.
    """
    if lat is None or lon is None:
        return False
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        return False
    return not (abs(lat) < 0.01 and abs(lon) < 0.01)


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points.

    Used to sanity-check how far a reverse-geocoded name is from the fix it came from —
    never to estimate a road distance, which is what the routing engine is for.
    """
    earth_radius_km = 6371.0088
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * earth_radius_km * math.asin(math.sqrt(h))


# ── Where the rider is ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LocationFix:
    """What a GPS fix tells us about the rider, once resolved.

    `in_himachal` is the whole point: it routes the chat to the local or the visitor
    assistant, and it makes the rider's pickup known without asking. It comes from the
    provider's administrative data, so a fix near the state line no longer needs the
    "we drew this outline ourselves" caveat.
    """

    lat: float
    lon: float
    in_himachal: bool
    #: The town the provider named for this fix, or None when it could name none.
    nearest_city: Optional[str] = None
    #: How far that town's centre is from the fix.
    nearest_city_km: Optional[float] = None
    #: The state the provider reported, for logs and for explaining a classification.
    state: Optional[str] = None

    @property
    def place_name(self) -> Optional[str]:
        return self.nearest_city

    @property
    def is_precise(self) -> bool:
        """True when the fix sits on the named town rather than merely inside its area."""
        return self.nearest_city_km is not None and self.nearest_city_km <= PRECISE_FIX_KM

    @property
    def visitor_type(self) -> str:
        """"local" | "outstation" — the persona this fix implies."""
        return "local" if self.in_himachal else "outstation"


async def locate(lat: float, lon: float) -> Optional[LocationFix]:
    """Resolve a GPS fix into everything the agent needs from it, or None if unusable.

    None covers both an invalid pair and a provider that could not place it. The caller
    then behaves exactly as it did before location was ever sent — it asks.
    """
    if not valid_coordinates(lat, lon):
        return None

    place = await get_geocoding_service().reverse(lat, lon)
    if place is None:
        log.info("locate_unresolved", lat=lat, lon=lon)
        return None

    return LocationFix(
        lat=lat,
        lon=lon,
        in_himachal=place.in_himachal,
        nearest_city=place.name,
        nearest_city_km=round(haversine_km((lat, lon), place.coords), 1),
        state=place.state,
    )


async def place_in_himachal(name: str) -> Optional[bool]:
    """Is this named place inside Himachal Pradesh? None when it can't be resolved.

    None matters: it means "we don't know", and the caller must keep whatever
    classification it already had rather than assuming the rider is a visitor.
    """
    place = await get_geocoding_service().geocode(name)
    return None if place is None else place.in_himachal


# ── The journey ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Journey:
    """A resolved journey: both endpoints, and the drive between them.

    One object because the three consumers must agree. The fare is priced on
    `route.distance_km`, the reply quotes `route.duration_text`, and the app's booking
    map drops pins on `origin`/`destination` — resolving those separately is how a
    price and a map end up describing different trips.
    """

    origin: Place
    destination: Place
    route: RoadRoute

    @property
    def distance_km(self) -> float:
        return self.route.distance_km

    @property
    def duration_minutes(self) -> int:
        return self.route.duration_minutes

    @property
    def duration_text(self) -> str:
        return self.route.duration_text

    def as_dict(self) -> dict:
        """The shape carried on agent state and handed to the app."""
        return {
            "origin": {
                "name": self.origin.name,
                "lat": self.origin.lat,
                "lon": self.origin.lon,
            },
            "destination": {
                "name": self.destination.name,
                "lat": self.destination.lat,
                "lon": self.destination.lon,
            },
            "distance_km": self.route.distance_km,
            "duration_minutes": self.route.duration_minutes,
            "duration_text": self.route.duration_text,
        }


async def resolve_pickup(
    *,
    origin_name: Optional[str] = None,
    origin_fix: Optional[LocationFix] = None,
) -> Optional[dict]:
    """Just the pickup point — `{name, lat, lon}` — for a hire that has no drop.

    A car rented by the hour has no destination, so `resolve_journey` has nothing to
    measure; the booking screen still needs somewhere to open its map. Same precedence
    as a journey's origin, for the same reason: the device's fix is where the cab
    actually goes, and the named city is the fallback.

    None when neither can be placed. The handoff then carries the pickup by name alone
    and the app geocodes it — worse than a pin, better than no handoff.
    """
    if origin_fix is not None:
        return {
            "name": origin_fix.place_name or origin_name or "Your location",
            "lat": origin_fix.lat,
            "lon": origin_fix.lon,
        }
    if not origin_name:
        return None

    place = await get_geocoding_service().geocode(origin_name)
    if place is None:
        log.info("pickup_unresolved", origin=origin_name)
        return None
    return {"name": place.name, "lat": place.lat, "lon": place.lon}


async def resolve_journey(
    destination: str,
    *,
    origin_name: Optional[str] = None,
    origin_fix: Optional[LocationFix] = None,
) -> Optional[Journey]:
    """Resolve a journey to `destination`, or None when any part cannot be resolved.

    The pickup comes from the rider's GPS fix when there is one — measured from where
    they actually are, not from the centre of the nearest town. A fix in Solang Valley
    is 14 km up the road from Manali, and pricing it as "Manali" quietly drops that leg.
    Falls back to the city they named.

    Both endpoints are geocoded concurrently: they are independent lookups, and a chat
    turn should not pay for them one after the other.
    """
    if not destination:
        return None

    geocoder = get_geocoding_service()

    # A fix is already coordinates — no lookup needed, and its own name is the best
    # label we have for it.
    if origin_fix is not None:
        origin_place: Optional[Place] = Place(
            name=origin_fix.place_name or (origin_name or "Your location"),
            lat=origin_fix.lat,
            lon=origin_fix.lon,
            state=origin_fix.state,
        )
        destination_place = await geocoder.geocode(destination)
    elif origin_name:
        origin_place, destination_place = await asyncio.gather(
            geocoder.geocode(origin_name),
            geocoder.geocode(destination),
        )
    else:
        return None

    if origin_place is None or destination_place is None:
        log.info(
            "journey_unresolved",
            origin=origin_name,
            destination=destination,
            origin_found=origin_place is not None,
            destination_found=destination_place is not None,
        )
        return None

    route = await get_routing_service().route(origin_place.coords, destination_place.coords)
    if route is None:
        return None

    return Journey(origin=origin_place, destination=destination_place, route=route)
