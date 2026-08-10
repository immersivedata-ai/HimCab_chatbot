"""
app/services/himcab/quote_builder.py
─────────────────────────────────────
Assembles a `FareQuoteInput` for POST /api/rider/fare/estimate and selects a display
headline from the per-category estimate response.

The estimate endpoint is DISTANCE-DRIVEN (needs distanceInMeters) and returns a fare for
every category in one call. So there is no categoryId to send; the response is narrowed
to what seats the group (`vehicles.select_cabs_for_group`) and this module only picks
which of those to feature as the headline, by budget.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from app.core.logging import get_logger
from app.models.session import ExtractedEntities
from app.services.geo import Journey
from app.services.himcab.models import FareCategoryFare, FareQuoteInput

log = get_logger(__name__)

#: Him Cab operates in India; the backend prices the night-charge window off the
#: `bookingTime` we send, so it has to be the rider's clock, not the server's.
_IST = timezone(timedelta(hours=5, minutes=30))


def current_booking_time() -> str:
    """Now, as HH:mm in IST — what the fare endpoint reads for night charges.

    This used to be the constant "12:00", which meant a rider quoting a 2 a.m. airport
    run was priced as a midday one and then charged the night rate at booking. The
    quote has to be built on the same clock the booking will be.
    """
    return datetime.now(_IST).strftime("%H:%M")

# Category-name preferences for the headline pick (real names from /fare/get-all).
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
    journey: Journey,
    trip_type: Optional[str] = None,
    booking_time: Optional[str] = None,
    rent_hours: int = 0,
) -> FareQuoteInput:
    """Build the estimate input for a journey the caller has already resolved.

    `journey` is required rather than optional: this module used to fall back to
    re-deriving the route from city names, which priced a DIFFERENT trip than the one
    the reply described — a rider standing in Solang Valley got the fare from Manali,
    14 km down the road. One resolution, one distance, one price.

    `trip_type` defaults to whatever the distance implies; pass it explicitly to override.
    """
    return FareQuoteInput(
        # The journey's own names, not the entity strings: these are what the geocoder
        # resolved and therefore what the distance actually measures.
        origin=journey.origin.name,
        destination=journey.destination.name,
        distance_km=journey.distance_km,
        trip_type=trip_type or classify_trip_type(journey.distance_km),
        booking_time=booking_time or current_booking_time(),
        rent_hours=rent_hours,
        group_size=entities.group_size,
        budget_range=entities.budget_range.value if entities.budget_range else None,
    )


def select_headline_category(
    options: list[FareCategoryFare],
    group_size: Optional[int],
    budget_range: Optional[str],
) -> Optional[FareCategoryFare]:
    """Pick which of the SUITABLE cabs to feature, by budget.

    Callers pass a list already narrowed to what seats the group
    (`vehicles.select_cabs_for_group`), so this only chooses between vehicles the
    rider could actually take. It used to special-case large groups onto a minibus
    itself, which is now handled properly upstream by seat capacity — and which,
    applied to the unnarrowed list, is what put a 20-seater in front of a family.
    """
    if not options:
        return None

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
