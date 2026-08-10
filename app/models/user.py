"""
app/models/user.py
───────────────────
User profile models. Stored in Firestore under users/{uid}/profile.
Used by the Context Node to personalise the AI response.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

from app.models.itinerary import BudgetRange, GroupType
from app.models.session import Language


class LoyaltyTier(str, Enum):
    """Loyalty tiers auto-computed from total trip count."""
    NEW = "new"             # 0 trips
    REGULAR = "regular"     # 1-2 trips
    FREQUENT = "frequent"   # 3-5 trips
    PREMIUM = "premium"     # 6+ trips


class TravelPreferences(BaseModel):
    preferred_budget: BudgetRange = BudgetRange.MODERATE
    preferred_group_type: GroupType = GroupType.FAMILY
    interests: list[str] = Field(default_factory=list)  # e.g. ["temples", "trekking"]
    dietary_restrictions: list[str] = Field(default_factory=list)
    preferred_language: Language = Language.ENGLISH


class PastTrip(BaseModel):
    destination: str
    duration_days: int
    travel_date: str
    rating: Optional[float] = None
    session_id: Optional[str] = None


class UserProfile(BaseModel):
    """
    Persistent user profile stored in Firestore.
    Injected into every new conversation via the Context Node.
    """
    model_config = ConfigDict(use_enum_values=True)

    user_id: str
    firebase_uid: str
    display_name: Optional[str] = None
    phone_number: Optional[str] = None
    home_city: Optional[str] = None  # For local (intra-HP) users — their base city
    preferences: TravelPreferences = Field(default_factory=TravelPreferences)
    past_trips: list[PastTrip] = Field(default_factory=list)
    total_trips: int = 0
    is_premium: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def get_loyalty_tier(self) -> LoyaltyTier:
        """Auto-compute loyalty tier from total trip count."""
        if self.total_trips >= 6:
            return LoyaltyTier.PREMIUM
        elif self.total_trips >= 3:
            return LoyaltyTier.FREQUENT
        elif self.total_trips >= 1:
            return LoyaltyTier.REGULAR
        return LoyaltyTier.NEW

    @property
    def is_frequent_traveller(self) -> bool:
        """True if user has 3+ completed trips (frequent or premium tier)."""
        return self.total_trips >= 3

    def get_frequent_destinations(self) -> list[str]:
        """Return destinations visited more than once."""
        from collections import Counter
        dest_counts = Counter(t.destination for t in self.past_trips)
        return [dest for dest, count in dest_counts.items() if count > 1]

    def to_context_string(self) -> str:
        """
        Generate a natural language summary for injection into the LLM system prompt.
        """
        parts = []
        if self.display_name:
            parts.append(f"User's name: {self.display_name}")
        if self.home_city:
            parts.append(f"Home city: {self.home_city} (Himachal local)")

        # Loyalty tier info
        tier = self.get_loyalty_tier()
        parts.append(f"Loyalty tier: {tier.value} ({self.total_trips} trips)")
        if self.is_frequent_traveller:
            parts.append("Frequent traveller — eligible for loyalty discounts")

        if self.past_trips:
            destinations = [t.destination for t in self.past_trips[-3:]]
            parts.append(f"Has visited: {', '.join(destinations)} recently")
            frequent_dests = self.get_frequent_destinations()
            if frequent_dests:
                parts.append(f"Frequent destinations: {', '.join(frequent_dests)}")

        prefs = self.preferences
        parts.append(f"Preferred budget: {prefs.preferred_budget}")
        parts.append(f"Usually travels as: {prefs.preferred_group_type}")
        if prefs.interests:
            parts.append(f"Interests: {', '.join(prefs.interests)}")
        if prefs.dietary_restrictions:
            parts.append(f"Dietary: {', '.join(prefs.dietary_restrictions)}")
        if self.is_premium:
            parts.append("Premium member — prioritise luxury options")
        return " | ".join(parts)

    def to_firestore_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_firestore_dict(cls, data: dict) -> "UserProfile":
        return cls.model_validate(data)
