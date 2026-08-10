"""
app/services/himcab/models.py
──────────────────────────────
Internal, contract-STABLE DTOs for Him Cab fare/cab lookups.

These are OUR vocabulary — the rest of the agent (tool_executor, writer) depends only
on these shapes. The Spring request/response wire format is translated to/from these
inside FareService (`_build_quote_request` / `_parse_quote_response`), so a change to the
backend contract touches only those two fenced methods, never these models or their consumers.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class FareQuoteInput(BaseModel):
    """Inputs for POST /api/rider/fare/estimate.

    The estimate endpoint is distance-driven: it needs `distanceInMeters`, `tripType`,
    `bookingTime` and `rentHours` — NOT coordinates or a categoryId (it returns a fare for
    EVERY category in one call). `group_size` / `budget_range` are carried only to pick a
    display headline from the per-category response; they are not sent to the backend.
    """
    origin: str
    destination: str
    # Required by the endpoint. None only when a place is unrecognised → skip the quote.
    distance_km: Optional[float] = None
    # True when the distance was derived from coordinates rather than surveyed, which
    # makes the resulting fare an estimate the rider must be told is approximate.
    distance_is_approximate: bool = False
    trip_type: str = "OUTSTATION"                        # OUTSTATION | LOCAL | RENTAL
    booking_time: str = "12:00"                          # HH:mm — drives night-charge calc
    rent_hours: int = 0                                  # >0 only for RENTAL trips
    group_size: Optional[int] = None                     # headline category selection (display-side)
    budget_range: Optional[str] = None                   # budget | moderate | luxury (display-side)


class FareCategoryFare(BaseModel):
    """One category's estimated total fare, from the per-category estimate response."""
    category_id: Optional[int] = None
    category_name: Optional[str] = None                  # e.g. "Standard", "Economy", "Luxury"
    category_image: Optional[str] = None
    total_fare_inr: float = Field(ge=0)
    seat_capacity: Optional[int] = None                  # present per-category (often null)


class Offer(BaseModel):
    """One live Him Cab coupon, as the rider would use it in the app.

    Parsed from GET /api/rider/offer/getOffersByTripType/{riderId} (or the unpersonalised
    /offer/get-all). `coupon_value` is a flat rupee discount — the Spring side subtracts
    it from the total fare at booking (see TripService: `totalFare.subtract(offerValue)`),
    so the same subtraction here is the real saving rather than an illustration.
    """

    offer_id: Optional[int] = None
    coupon_code: Optional[str] = None
    coupon_name: Optional[str] = None
    coupon_value_inr: float = Field(default=0.0, ge=0)
    #: Minimum fare the offer applies to. An offer above the quoted fare is filtered
    #: out rather than shown — a coupon the rider cannot use on this trip is noise.
    minimum_value_inr: float = Field(default=0.0, ge=0)
    #: Trip types it is valid for, upper-cased: ALL | LOCAL | OUTSTATION | RENTAL.
    trip_types: list[str] = Field(default_factory=list)
    #: NEW_USER | EXISTING_USER | None.
    user_type: Optional[str] = None
    terms_and_conditions: Optional[str] = None
    image: Optional[str] = None
    live_date: Optional[datetime] = None
    expire_date: Optional[datetime] = None

    def applies_to(self, trip_type: str) -> bool:
        """True when this coupon covers the given trip type (empty list = all types)."""
        if not self.trip_types:
            return True
        wanted = (trip_type or "").upper()
        return "ALL" in self.trip_types or wanted in self.trip_types

    def is_live(self, now: Optional[datetime] = None) -> bool:
        """True when the coupon's window is open right now."""
        moment = now or datetime.now(timezone.utc)
        if self.live_date and _as_utc(self.live_date) > moment:
            return False
        if self.expire_date and _as_utc(self.expire_date) < moment:
            return False
        return True

    def discount_on(self, fare_inr: float) -> float:
        """What this coupon actually takes off `fare_inr` — 0 when it does not apply.

        Capped at the fare so a Rs 500 coupon on a Rs 300 ride never produces a
        negative price, which is what the app would otherwise render.
        """
        if fare_inr < self.minimum_value_inr:
            return 0.0
        return round(min(self.coupon_value_inr, fare_inr), 2)


def _as_utc(value: datetime) -> datetime:
    """Firestore and Spring both hand back naive datetimes; treat those as UTC."""
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class FareQuote(BaseModel):
    """Parsed estimate: the selected headline category plus all category options.

    The estimate is GROSS (the endpoint takes no offerId), so a local discount is always
    correct — offer_applied_server_side stays False.
    """
    base_fare_inr: float = Field(ge=0)                   # headline category's total fare
    currency: str = "INR"
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    vehicle_label: Optional[str] = None                  # category name doubles as vehicle label
    seat_capacity: Optional[int] = None                  # from response (often null per-category)
    per_km_inr: Optional[float] = None
    trip_type: str = "OUTSTATION"
    distance_km: Optional[float] = None                  # what the fare was priced on
    distance_is_approximate: bool = False                # → present the fare as an estimate
    offer_applied_server_side: bool = False              # estimate takes no offerId → always False
    alternatives: list[FareCategoryFare] = Field(default_factory=list)  # all categories returned
    raw: dict = Field(default_factory=dict)              # untouched backend payload, for debugging
