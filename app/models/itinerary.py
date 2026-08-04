"""
app/models/itinerary.py
────────────────────────
Structured output models for trip itineraries.
These models are what the mobile app receives in the `structured` SSE event.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

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


# ── Sub-models ────────────────────────────────────────────────────────────────

class Activity(BaseModel):
    time: str = Field(description="Suggested time e.g. '10:00 AM'")
    place: str
    type: ActivityType
    description: Optional[str] = None
    note: Optional[str] = None
    crowd_level: Optional[CrowdLevel] = None
    maps_url: Optional[str] = None
    duration_minutes: Optional[int] = None
    entry_fee_inr: Optional[float] = None

    @field_validator("type", mode="before")
    @classmethod
    def lowercase_type(cls, v: str | ActivityType) -> str | ActivityType:
        if isinstance(v, str):
            return v.lower()
        return v


class DayPlan(BaseModel):
    day: int
    theme: str = Field(description="Theme for the day e.g. 'Arrival & Local Exploration'")
    activities: list[Activity]
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


class Hotel(BaseModel):
    name: str
    category: BudgetRange
    price_per_night_inr: float
    rating: Optional[float] = None
    amenities: list[str] = Field(default_factory=list)
    maps_url: Optional[str] = None
    booking_url: Optional[str] = None

    @field_validator("category", mode="before")
    @classmethod
    def lowercase_category(cls, v: str | BudgetRange) -> str | BudgetRange:
        if isinstance(v, str):
            return v.lower()
        return v


class WeatherForecast(BaseModel):
    summary: str
    temperature_min_c: Optional[float] = None
    temperature_max_c: Optional[float] = None
    condition: Optional[str] = None
    alert: Optional[str] = None
    alert_severity: Optional[str] = None  # "low" | "medium" | "high"


class CostBreakdown(BaseModel):
    cab_inr: float
    hotels_inr: float
    food_estimate_inr: float
    entry_fees_inr: float = 0.0
    miscellaneous_inr: float = 0.0
    total_inr: float = Field(ge=0, description="Total must be non-negative")

    @classmethod
    def compute(
        cls,
        cab: float,
        hotels: float,
        food: float,
        entry_fees: float = 0,
        misc: float = 0,
    ) -> "CostBreakdown":
        return cls(
            cab_inr=cab,
            hotels_inr=hotels,
            food_estimate_inr=food,
            entry_fees_inr=entry_fees,
            miscellaneous_inr=misc,
            total_inr=cab + hotels + food + entry_fees + misc,
        )


class TravelInfo(BaseModel):
    """
    Captures estimated travel information from the visitor's origin city
    to the Himachal Pradesh destination. Displayed prominently in the
    itinerary so outstation travellers know what to expect.
    """
    origin: str = Field(description="City the traveller is coming from, e.g. 'Delhi'")
    destination: str = Field(description="Himachal destination, e.g. 'Manali'")
    estimated_duration_hours: float = Field(description="Estimated travel time in hours")
    estimated_distance_km: Optional[float] = None
    recommended_mode: str = Field(
        default="cab",
        description="Recommended travel mode: 'cab', 'flight+cab', 'train+cab'",
    )
    route_notes: Optional[str] = Field(
        default=None,
        description="Extra route info, e.g. 'Via NH-44, scenic route through Bilaspur'",
    )


class RouteLeg(BaseModel):
    """
    A single leg of the journey, e.g. Delhi → Shimla or Shimla → Manali.
    Used to display segment-by-segment routing for outstation visitors.
    """
    origin: str = Field(description="Start point of this leg, e.g. 'Delhi'")
    destination: str = Field(description="End point of this leg, e.g. 'Shimla'")
    distance_km: Optional[float] = None
    duration_hours: Optional[float] = None
    duration_minutes: Optional[int] = None
    recommended_mode: str = Field(
        default="cab",
        description="Recommended travel mode: 'cab', 'flight+cab', 'train+cab'",
    )
    route_notes: Optional[str] = Field(
        default=None,
        description="Road conditions, scenic highlights, key stops along the way",
    )
    road_condition: Optional[str] = Field(
        default=None,
        description="Current road condition: 'good', 'moderate', 'rough', 'closed'",
    )


class RoutePlan(BaseModel):
    """
    Full route plan for outstation visitors covering the entire trip.
    Contains all legs from origin through HP destinations and back.
    """
    legs: list[RouteLeg] = Field(default_factory=list)
    total_distance_km: Optional[float] = None
    total_duration_hours: Optional[float] = None
    optimal_order: list[str] = Field(
        default_factory=list,
        description="Ordered list of places: ['Delhi', 'Shimla', 'Manali', 'Delhi']",
    )
    summary: Optional[str] = Field(
        default=None,
        description="Human-readable route summary, e.g. 'Delhi → Shimla (7h) → Manali (5h) → Delhi (11h)'",
    )


# ── Root Itinerary ────────────────────────────────────────────────────────────

class Itinerary(BaseModel):
    """
    Top-level structured itinerary returned to the mobile app
    via the `structured` SSE event (type='itinerary').
    """
    model_config = ConfigDict(use_enum_values=True)

    destination: str
    duration_days: int
    group_type: GroupType
    budget_range: BudgetRange
    best_time_to_visit: Optional[str] = None
    days: list[DayPlan]
    cab_package: Optional[CabPackage] = None
    hotels: list[Hotel] = Field(default_factory=list)
    weather: Optional[WeatherForecast] = None
    travel_info: Optional[TravelInfo] = None
    route_plan: Optional[RoutePlan] = None
    cost_breakdown: Optional[CostBreakdown] = None
    offers: list[dict] = Field(default_factory=list, description="Applicable offers for this trip")
    hidden_gems: list[str] = Field(default_factory=list, description="Lesser-known spots at the destination")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    language: str = "en"

    @field_validator("group_type", mode="before")
    @classmethod
    def lowercase_group_type(cls, v: str | GroupType) -> str | GroupType:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("budget_range", mode="before")
    @classmethod
    def lowercase_budget_range(cls, v: str | BudgetRange) -> str | BudgetRange:
        if isinstance(v, str):
            return v.lower()
        return v
