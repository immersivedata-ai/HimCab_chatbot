"""
app/domain/itinerary.py
────────────────────────
Structured output models for trip itineraries.
These models are what the mobile app receives in the `structured` SSE event.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CrowdLevel(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class GroupType(str, Enum):
    SOLO = "solo"
    COUPLE = "couple"
    FAMILY = "family"
    FRIENDS = "friends"
    CORPORATE = "corporate"


class BudgetRange(str, Enum):
    BUDGET = "budget"
    MODERATE = "moderate"
    LUXURY = "luxury"


class ActivityType(str, Enum):
    SIGHTSEEING = "sightseeing"
    PICKUP = "pickup"
    DROP = "drop"
    FOOD = "food"
    ADVENTURE = "adventure"
    SPIRITUAL = "spiritual"
    SHOPPING = "shopping"
    REST = "rest"
    TRAVEL = "travel"
    ACCOMMODATION = "accommodation"
    LEISURE = "leisure"
    TREKKING = "trekking"
    DINING = "dining"
    OTHER = "other"


# ── Enum coercion ─────────────────────────────────────────────────────────────
# These models are populated from LLM output. A single off-vocabulary word used to
# fail the whole `Itinerary.model_validate()`, which is why `ChatResponse.itinerary`
# came back null on turns that clearly produced an itinerary (the writer logged the
# failure and moved on with `structured_itinerary = None`). Coerce instead of reject:
# a slightly-wrong enum is always better than losing the entire itinerary.

_E = TypeVar("_E", bound=Enum)

_GROUP_TYPE_ALIASES: dict[str, GroupType] = {
    "single": GroupType.SOLO, "alone": GroupType.SOLO, "individual": GroupType.SOLO,
    "one": GroupType.SOLO, "self": GroupType.SOLO,
    "pair": GroupType.COUPLE, "honeymoon": GroupType.COUPLE, "partner": GroupType.COUPLE,
    "romantic": GroupType.COUPLE, "two": GroupType.COUPLE,
    "families": GroupType.FAMILY, "family trip": GroupType.FAMILY,
    "family_trip": GroupType.FAMILY, "kids": GroupType.FAMILY, "parents": GroupType.FAMILY,
    "friend": GroupType.FRIENDS, "group": GroupType.FRIENDS, "gang": GroupType.FRIENDS,
    "buddies": GroupType.FRIENDS,
    "business": GroupType.CORPORATE, "office": GroupType.CORPORATE,
    "team": GroupType.CORPORATE, "work": GroupType.CORPORATE, "company": GroupType.CORPORATE,
}

_BUDGET_ALIASES: dict[str, BudgetRange] = {
    "cheap": BudgetRange.BUDGET, "low": BudgetRange.BUDGET, "economy": BudgetRange.BUDGET,
    "affordable": BudgetRange.BUDGET, "economical": BudgetRange.BUDGET,
    "mid": BudgetRange.MODERATE, "medium": BudgetRange.MODERATE,
    "standard": BudgetRange.MODERATE, "mid-range": BudgetRange.MODERATE,
    "mid_range": BudgetRange.MODERATE, "comfortable": BudgetRange.MODERATE,
    "premium": BudgetRange.LUXURY, "high": BudgetRange.LUXURY,
    "high-end": BudgetRange.LUXURY, "deluxe": BudgetRange.LUXURY, "5 star": BudgetRange.LUXURY,
}

_CROWD_ALIASES: dict[str, CrowdLevel] = {
    "medium": CrowdLevel.MODERATE, "mid": CrowdLevel.MODERATE, "average": CrowdLevel.MODERATE,
    "very high": CrowdLevel.VERY_HIGH, "extreme": CrowdLevel.VERY_HIGH,
    "packed": CrowdLevel.VERY_HIGH, "crowded": CrowdLevel.HIGH,
    "quiet": CrowdLevel.LOW, "empty": CrowdLevel.LOW, "none": CrowdLevel.LOW,
}


def _coerce_enum(
    value: object,
    enum_cls: type[_E],
    aliases: dict[str, _E],
    default: Optional[_E],
) -> Optional[_E]:
    """Best-effort map an arbitrary LLM string onto `enum_cls`, never raising.

    Order: exact value → alias table → substring match → `default`. Returning the
    default (rather than raising) is deliberate: see the note above.
    """
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return default
    if not isinstance(value, str):
        return default

    key = value.strip().lower().replace("-", " ").replace("_", " ")
    if not key:
        return default

    try:
        return enum_cls(key.replace(" ", "_"))
    except ValueError:
        pass

    normalized_aliases = {k.replace("-", " ").replace("_", " "): v for k, v in aliases.items()}
    if key in normalized_aliases:
        return normalized_aliases[key]

    # Last resort: the model wrote a phrase ("family with young kids", "luxury stay").
    # Match on whole words only — a substring check would read "restaurant" as `rest`
    # and "moneyed" as `one`.
    words = set(key.split())
    for member in enum_cls:
        if member.value.replace("_", " ") in key.replace("  ", " ") and set(
            member.value.replace("_", " ").split()
        ) <= words:
            return member
    for alias, member in normalized_aliases.items():
        if set(alias.split()) <= words:
            return member

    return default


# ── Sub-models ────────────────────────────────────────────────────────────────

class Activity(BaseModel):
    time: str = Field(default="", description="Suggested time e.g. '10:00 AM'")
    place: str = ""
    type: ActivityType = ActivityType.OTHER
    description: Optional[str] = None
    note: Optional[str] = None
    crowd_level: Optional[CrowdLevel] = None
    maps_url: Optional[str] = None
    duration_minutes: Optional[int] = None
    entry_fee_inr: Optional[float] = None

    @field_validator("type", mode="before")
    @classmethod
    def coerce_type(cls, v: object) -> ActivityType:
        # Unknown activity words ("relax", "check-in") fall back to OTHER rather
        # than invalidating the whole itinerary.
        return _coerce_enum(v, ActivityType, {}, ActivityType.OTHER) or ActivityType.OTHER

    @field_validator("crowd_level", mode="before")
    @classmethod
    def coerce_crowd_level(cls, v: object) -> Optional[CrowdLevel]:
        return _coerce_enum(v, CrowdLevel, _CROWD_ALIASES, None)


class DayPlan(BaseModel):
    day: int = 1
    theme: str = Field(default="", description="Theme for the day e.g. 'Arrival & Local Exploration'")
    activities: list[Activity] = Field(default_factory=list)
    meal_suggestions: list[str] = Field(default_factory=list)
    pro_tip: Optional[str] = None
    local_tips: list[str] = Field(default_factory=list, description="Insider tips a local would appreciate")


class CabPackage(BaseModel):
    package_type: str = Field(description="e.g. '3-Day Himachal Hill Package'")
    vehicle: str = Field(description="e.g. 'Innova Crysta'")
    estimated_cost_inr: float
    includes: list[str] = Field(default_factory=list)
    driver_language: list[str] = Field(default=["Hindi", "English"])
    contact_number: Optional[str] = None
    # Real fare fields (populated from the Him Cab backend when available; else None)
    base_fare_inr: Optional[float] = None
    #: What the rider pays once their best live coupon is applied. None when no coupon
    #: applies — never a copy of base_fare_inr, so "is there a discount" is one check.
    discounted_fare_inr: Optional[float] = None
    #: The coupon behind `discounted_fare_inr`, for the booking screen to prefill.
    coupon_code: Optional[str] = None
    seat_capacity: Optional[int] = None
    fare_source: Optional[str] = None  # "himcab_api" when from a real quote, else None/approximate


class Hotel(BaseModel):
    name: str
    category: BudgetRange = BudgetRange.MODERATE
    price_per_night_inr: float = 0.0
    rating: Optional[float] = None
    amenities: list[str] = Field(default_factory=list)
    maps_url: Optional[str] = None
    booking_url: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, v: object) -> BudgetRange:
        return _coerce_enum(v, BudgetRange, _BUDGET_ALIASES, BudgetRange.MODERATE) or BudgetRange.MODERATE


class WeatherForecast(BaseModel):
    summary: str = ""
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    condition: Optional[str] = None
    alert: Optional[str] = None
    alert_severity: Optional[str] = None  # "low" | "medium" | "high"


class CostBreakdown(BaseModel):
    cab_inr: float
    hotels_inr: float = 0.0
    food_estimate_inr: float
    entry_fees_inr: float = 0.0
    miscellaneous_inr: float = 0.0
    discount_inr: float = 0.0
    total_inr: float = Field(ge=0, description="Total must be non-negative")

    @classmethod
    def compute(
        cls,
        cab: float,
        hotels: float = 0,
        food: float = 0,
        entry_fees: float = 0,
        misc: float = 0,
        discount: float = 0,
    ) -> "CostBreakdown":
        return cls(
            cab_inr=cab,
            hotels_inr=hotels,
            food_estimate_inr=food,
            entry_fees_inr=entry_fees,
            miscellaneous_inr=misc,
            discount_inr=discount,
            total_inr=max(0.0, cab + hotels + food + entry_fees + misc - discount),
        )


class TravelInfo(BaseModel):
    """
    Captures estimated travel information from the visitor's origin city
    to the Himachal Pradesh destination. Displayed prominently in the
    itinerary so outstation travellers know what to expect.
    """
    origin: str = Field(default="", description="City the traveller is coming from, e.g. 'Delhi'")
    destination: str = Field(default="", description="Himachal destination, e.g. 'Manali'")
    estimated_duration_hours: Optional[float] = Field(default=None, description="Estimated travel time in hours, if known")
    estimated_distance_km: Optional[float] = None
    recommended_mode: str = Field(
        default="cab",
        description="Recommended travel mode: 'cab', 'flight+cab', 'train+cab'",
    )
    route_notes: Optional[str] = Field(
        default=None,
        description="Extra route info, e.g. 'Via NH-44, scenic route through Bilaspur'",
    )


# ── Root Itinerary ────────────────────────────────────────────────────────────

class Itinerary(BaseModel):
    """
    Top-level structured itinerary returned to the mobile app.

    Delivered in TWO places, always with identical content:
      - the `structured` SSE event / `structured_events[]` entry with type='itinerary'
      - the top-level `itinerary` field of the non-streaming ChatResponse

    Every field has a default so a partial or slightly off-vocabulary LLM response
    degrades into a usable itinerary instead of being dropped entirely.
    """
    model_config = ConfigDict(use_enum_values=True)

    destination: str = ""
    duration_days: int = 1
    group_type: GroupType = GroupType.FAMILY
    budget_range: BudgetRange = BudgetRange.MODERATE
    best_time_to_visit: Optional[str] = None
    days: list[DayPlan] = Field(default_factory=list)
    cab_package: Optional[CabPackage] = None
    hotels: list[Hotel] = Field(default_factory=list)
    weather: Optional[WeatherForecast] = None
    travel_info: Optional[TravelInfo] = None
    cost_breakdown: Optional[CostBreakdown] = None
    hidden_gems: list[str] = Field(default_factory=list, description="Lesser-known spots at the destination")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    language: str = "en"

    @field_validator("group_type", mode="before")
    @classmethod
    def coerce_group_type(cls, v: object) -> GroupType:
        # Guards against the persona ("local"/"outstation") leaking into this field —
        # neither is a group type, and both used to fail validation outright.
        return _coerce_enum(v, GroupType, _GROUP_TYPE_ALIASES, GroupType.FAMILY) or GroupType.FAMILY

    @field_validator("budget_range", mode="before")
    @classmethod
    def coerce_budget_range(cls, v: object) -> BudgetRange:
        return _coerce_enum(v, BudgetRange, _BUDGET_ALIASES, BudgetRange.MODERATE) or BudgetRange.MODERATE
