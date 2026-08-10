"""
app/services/himcab/quote_builder.py
─────────────────────────────────────
Assembles a `FareQuoteInput` for POST /api/rider/fare/estimate and selects a display
headline from the per-category estimate response.

The estimate endpoint is DISTANCE-DRIVEN (needs distanceInMeters) and returns a fare for
every category in one call. So there is no categoryId to send; we only choose which
category to feature as the headline, using group size + budget (the endpoint gives no
seat capacity, so selection is by category name/price, not seats).
"""

from typing import Optional

from typing import Optional

from app.core.logging import get_logger
from app.models.session import ExtractedEntities
from app.services.himcab.distances import resolve_distance
from app.services.himcab.models import FareCategoryFare, FareQuoteInput

log = get_logger(__name__)

# Category-name preferences for the headline pick (real names from /fare/get-all).
_MINIBUS_HINT = "bus"
_PREF_BY_BUDGET = {
    "luxury": ["premium", "luxury"],
    "moderate": ["standard", "economy"],
}
_DEFAULT_PREF = ["standard", "economy"]

#: Under this, the journey is a local run rather than an intercity transfer. Above it,
#: the backend's OUTSTATION rate card is the right one whoever the rider is.
_LOCAL_TRIP_MAX_KM = 40.0


def classify_trip_type(distance_km: Optional[float]) -> str:
    """LOCAL or OUTSTATION, decided by the JOURNEY — not by who is travelling.

    This used to be `"LOCAL" if visitor_type == "local"`, which is a different
    question: a Himachal resident driving Shimla → Manali is a 270 km intercity
    transfer, and pricing it off the LOCAL rate card gives the rider a number that has
    nothing to do with the trip they asked about.
    """
    if distance_km is None:
        return "OUTSTATION"
    return "LOCAL" if distance_km <= _LOCAL_TRIP_MAX_KM else "OUTSTATION"


def build_fare_quote_input(
    entities: ExtractedEntities,
    trip_type: Optional[str] = None,
    booking_time: str = "12:00",
    rent_hours: int = 0,
) -> FareQuoteInput:
    """Build the estimate input for the main origin→destination journey.

    Distance is surveyed where known and computed from coordinates otherwise (see
    `distances.resolve_distance`); it is None only when a place is unrecognised, in
    which case the caller must skip the quote — the endpoint requires distanceInMeters.

    `trip_type` defaults to whatever the distance implies; pass it explicitly to override.
    """
    # Locals often only give home_city (not origin) — fall back to it so the distance
    # lookup and the request body still get a usable origin city.
    origin = entities.origin or entities.home_city or ""
    destination = entities.destination or ""
    resolved = resolve_distance(origin, destination)

    return FareQuoteInput(
        origin=origin,
        destination=destination,
        distance_km=resolved.km if resolved else None,
        distance_is_approximate=resolved.is_approximate if resolved else False,
        trip_type=trip_type or classify_trip_type(resolved.km if resolved else None),
        booking_time=booking_time,
        rent_hours=rent_hours,
        group_size=entities.group_size,
        budget_range=entities.budget_range.value if entities.budget_range else None,
    )


def select_headline_category(
    options: list[FareCategoryFare],
    group_size: Optional[int],
    budget_range: Optional[str],
) -> Optional[FareCategoryFare]:
    """Pick which category to feature. Large groups → Mini Bus; otherwise by budget
    preference; falling back to the cheapest option."""
    if not options:
        return None

    # Large group → prefer a mini-bus-type category if present.
    if group_size and group_size >= 7:
        for o in options:
            if _MINIBUS_HINT in (o.category_name or "").lower():
                return o

    # Budget travellers → the genuinely cheapest option, regardless of tier name.
    if (budget_range or "").lower() == "budget":
        return min(options, key=lambda o: o.total_fare_inr)

    prefs = _PREF_BY_BUDGET.get((budget_range or "").lower(), _DEFAULT_PREF)
    for pref in prefs:
        for o in options:
            if pref in (o.category_name or "").lower():
                return o

    # Fallback: cheapest available fare.
    return min(options, key=lambda o: o.total_fare_inr)
