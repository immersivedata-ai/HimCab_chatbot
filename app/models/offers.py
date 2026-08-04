"""
app/models/offers.py
─────────────────────
Pydantic models for the frequent traveller offers system.
Used by OffersService and the Writer node to display personalised deals.
"""

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OfferType(str, Enum):
    """Categories of offers the system can generate."""
    SEASONAL = "seasonal"           # Off-peak month discounts
    LOYALTY = "loyalty"             # Tier-based frequent traveller rewards
    LOCAL_RATE = "local_rate"       # Intra-HP special pricing for locals
    WEEKEND_GETAWAY = "weekend_getaway"  # Short 1-2 day curated packages
    REVISIT = "revisit"             # Returning to a previously visited destination
    MONSOON_SPECIAL = "monsoon_special"  # Rainy season packages


class LoyaltyTier(str, Enum):
    """Loyalty tiers auto-computed from total trip count."""
    NEW = "new"             # 0 trips
    REGULAR = "regular"     # 1-2 trips
    FREQUENT = "frequent"   # 3-5 trips
    PREMIUM = "premium"     # 6+ trips


class Offer(BaseModel):
    """A single personalised offer shown to the user."""
    title: str = Field(description="Short headline, e.g. 'Frequent Traveller 10% Off'")
    description: str = Field(description="One-line description of what the offer includes")
    offer_type: OfferType
    discount_percent: float = Field(ge=0, le=100, description="Discount percentage")
    promo_code: str = Field(description="Redeemable code, e.g. 'HIMLOCAL10'")
    valid_until: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(days=30),
    )
    applicable_to: list[str] = Field(
        default_factory=lambda: ["cab"],
        description="What the offer applies to: cab, hotel, package, food",
    )
    conditions: Optional[str] = Field(
        default=None,
        description="Fine print, e.g. 'Valid on bookings above INR 2000'",
    )
    loyalty_tier_required: Optional[LoyaltyTier] = None

    def to_display_dict(self) -> dict:
        """Lightweight dict for SSE structured events and JSON output."""
        return {
            "title": self.title,
            "description": self.description,
            "type": self.offer_type.value,
            "discount_percent": self.discount_percent,
            "promo_code": self.promo_code,
            "valid_until": self.valid_until.isoformat(),
            "applicable_to": self.applicable_to,
            "conditions": self.conditions,
        }


class OfferBundle(BaseModel):
    """Container for all offers applicable to a user+trip combination."""
    offers: list[Offer] = Field(default_factory=list)
    loyalty_tier: LoyaltyTier = LoyaltyTier.NEW
    total_trips: int = 0

    @property
    def best_discount(self) -> float:
        """Highest discount percentage among all offers."""
        if not self.offers:
            return 0.0
        return max(o.discount_percent for o in self.offers)

    @property
    def has_offers(self) -> bool:
        return len(self.offers) > 0

    def to_display_dict(self) -> dict:
        return {
            "offers": [o.to_display_dict() for o in self.offers],
            "loyalty_tier": self.loyalty_tier.value,
            "total_trips": self.total_trips,
            "best_discount": self.best_discount,
        }
