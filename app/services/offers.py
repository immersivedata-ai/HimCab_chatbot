"""
app/services/offers.py
───────────────────────
Offers Service — generates personalised offers based on user profile,
travel context, and visitor type.

Offer categories:
  1. Seasonal     — off-peak month discounts
  2. Loyalty      — tier-based frequent traveller rewards
  3. Local Rate   — intra-HP special pricing for local residents
  4. Weekend Getaway — short 1-2 day curated packages for locals
  5. Revisit      — returning to a previously visited destination
  6. Monsoon      — rainy season special packages

All offers are computed locally (no external API calls) so this
runs in ~0ms and can safely be called in the tool executor's
parallel gather.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional

from app.core.logging import get_logger
from app.models.offers import (
    LoyaltyTier,
    Offer,
    OfferBundle,
    OfferType,
)
from app.models.session import ExtractedEntities
from app.models.user import UserProfile

log = get_logger(__name__)

# Months considered off-peak (lower tourist footfall)
OFF_PEAK_MONTHS = {1, 2, 3, 9, 11}  # Jan, Feb, Mar, Sep, Nov

# Monsoon months in Himachal Pradesh
MONSOON_MONTHS = {7, 8}  # Jul, Aug

# Himachal Pradesh cities (for local detection)
HP_CITIES = {
    "shimla", "manali", "dharamshala", "mcleod ganj", "kullu", "mandi",
    "solan", "bilaspur", "hamirpur", "una", "kangra", "chamba", "kinnaur",
    "lahaul", "spiti", "kasol", "dalhousie", "palampur", "nahan", "sirmaur",
    "kufri", "bir billing", "khajjiar", "chail", "chitkul", "tirthan valley",
    "jibhi", "sangla", "narkanda", "kalpa", "keylong", "rohtang",
}


class OffersService:
    """
    Generates personalised offers based on user profile and trip context.
    Pure computation — no external API calls, no async needed.
    """

    def generate_offers(
        self,
        user_profile: Optional[UserProfile],
        entities: ExtractedEntities,
        visitor_type: Optional[str] = None,
    ) -> OfferBundle:
        """
        Generate all applicable offers for the given user + trip combination.
        Returns an OfferBundle with deduplicated, sorted offers.
        """
        offers: list[Offer] = []
        now = datetime.now(timezone.utc)

        # Determine loyalty tier
        if user_profile:
            loyalty_tier = user_profile.get_loyalty_tier()
            total_trips = user_profile.total_trips
        else:
            loyalty_tier = LoyaltyTier.NEW
            total_trips = 0

        # 1. Seasonal offers
        seasonal = self._generate_seasonal_offers(entities, now)
        if seasonal:
            offers.append(seasonal)

        # 2. Loyalty / frequent traveller offers
        loyalty = self._generate_loyalty_offers(loyalty_tier, total_trips, entities)
        if loyalty:
            offers.extend(loyalty)

        # 3. Local-specific offers
        if visitor_type == "local":
            local_offers = self._generate_local_offers(entities, loyalty_tier, now)
            offers.extend(local_offers)

        # 4. Revisit offers
        if user_profile and entities.destination:
            revisit = self._generate_revisit_offer(user_profile, entities)
            if revisit:
                offers.append(revisit)

        # 5. Monsoon specials
        monsoon = self._generate_monsoon_offer(entities, now)
        if monsoon:
            offers.append(monsoon)

        log.info(
            "offers_generated",
            count=len(offers),
            loyalty_tier=loyalty_tier.value,
            visitor_type=visitor_type,
            total_trips=total_trips,
            destination=entities.destination,
        )

        return OfferBundle(
            offers=offers,
            loyalty_tier=loyalty_tier,
            total_trips=total_trips,
        )

    # ── Private generators ────────────────────────────────────────────────────

    def _generate_seasonal_offers(
        self,
        entities: ExtractedEntities,
        now: datetime,
    ) -> Optional[Offer]:
        """Off-peak month discount to encourage travel in lean season."""
        if now.month not in OFF_PEAK_MONTHS:
            return None

        month_name = now.strftime("%B")
        dest = entities.destination or "Himachal Pradesh"

        return Offer(
            title=f"{month_name} Off-Season Special",
            description=f"Enjoy {dest} with fewer crowds! 8% off on all Him Cab bookings this {month_name}.",
            offer_type=OfferType.SEASONAL,
            discount_percent=8.0,
            promo_code=f"HIM{month_name[:3].upper()}8",
            valid_until=now + timedelta(days=30),
            applicable_to=["cab", "package"],
            conditions=f"Valid on bookings made in {month_name}. Minimum booking value INR 1500.",
        )

    def _generate_loyalty_offers(
        self,
        loyalty_tier: LoyaltyTier,
        total_trips: int,
        entities: ExtractedEntities,
    ) -> list[Offer]:
        """Tier-based offers for returning customers."""
        now = datetime.now(timezone.utc)
        offers = []

        if loyalty_tier == LoyaltyTier.REGULAR:
            offers.append(Offer(
                title="Welcome Back! 5% Off",
                description="Thanks for choosing Him Cab again! Enjoy 5% off your next cab booking.",
                offer_type=OfferType.LOYALTY,
                discount_percent=5.0,
                promo_code="HIMBACK5",
                valid_until=now + timedelta(days=60),
                applicable_to=["cab"],
                conditions="Valid on bookings above INR 2000.",
                loyalty_tier_required=LoyaltyTier.REGULAR,
            ))

        elif loyalty_tier == LoyaltyTier.FREQUENT:
            offers.append(Offer(
                title="Frequent Traveller 10% Off",
                description=f"You've explored Himachal {total_trips} times! 10% off + complimentary local sightseeing on one day.",
                offer_type=OfferType.LOYALTY,
                discount_percent=10.0,
                promo_code="HIMFREQ10",
                valid_until=now + timedelta(days=90),
                applicable_to=["cab", "package"],
                conditions="Includes complimentary 2-hour local sightseeing on any one day of your trip.",
                loyalty_tier_required=LoyaltyTier.FREQUENT,
            ))

        elif loyalty_tier == LoyaltyTier.PREMIUM:
            offers.append(Offer(
                title="Premium Member 15% Off",
                description=f"Our most loyal explorer with {total_trips} trips! 15% off + free pickup + priority booking.",
                offer_type=OfferType.LOYALTY,
                discount_percent=15.0,
                promo_code="HIMPREM15",
                valid_until=now + timedelta(days=120),
                applicable_to=["cab", "package", "hotel"],
                conditions="Includes free airport/station pickup, priority booking, and flexible cancellation.",
                loyalty_tier_required=LoyaltyTier.PREMIUM,
            ))
            # Premium bonus: companion offer
            offers.append(Offer(
                title="Premium Companion Bonus",
                description="Share the love! Your travel companion gets 5% off their own booking.",
                offer_type=OfferType.LOYALTY,
                discount_percent=5.0,
                promo_code="HIMPREMPAL5",
                valid_until=now + timedelta(days=120),
                applicable_to=["cab"],
                conditions="Companion must book within 7 days of your booking. One use per trip.",
                loyalty_tier_required=LoyaltyTier.PREMIUM,
            ))

        return offers

    def _generate_local_offers(
        self,
        entities: ExtractedEntities,
        loyalty_tier: LoyaltyTier,
        now: datetime,
    ) -> list[Offer]:
        """Special offers for Himachal Pradesh residents."""
        offers = []
        dest = entities.destination or "your destination"
        duration = entities.duration_days or 2

        # Local flat-rate discount
        offers.append(Offer(
            title="HP Resident Special",
            description=f"Intra-Himachal special rate to {dest}! Local residents get flat 7% off.",
            offer_type=OfferType.LOCAL_RATE,
            discount_percent=7.0,
            promo_code="HIMLOCAL7",
            valid_until=now + timedelta(days=45),
            applicable_to=["cab"],
            conditions="Valid for residents of Himachal Pradesh. ID proof required at pickup.",
        ))

        # Weekend getaway package (1-2 day trips)
        if duration and duration <= 2:
            offers.append(Offer(
                title="Weekend Getaway Package",
                description=f"Quick escape to {dest}! All-inclusive 2-day package with cab + suggested stays.",
                offer_type=OfferType.WEEKEND_GETAWAY,
                discount_percent=12.0,
                promo_code="HIMWEEKEND12",
                valid_until=now + timedelta(days=30),
                applicable_to=["package"],
                conditions="Includes cab for up to 2 days (max 200 km/day). Hotel not included.",
            ))

        return offers

    def _generate_revisit_offer(
        self,
        user_profile: UserProfile,
        entities: ExtractedEntities,
    ) -> Optional[Offer]:
        """Offer for users returning to a previously visited destination."""
        if not entities.destination:
            return None

        dest_lower = entities.destination.lower()
        visited_destinations = [t.destination.lower() for t in user_profile.past_trips]

        if dest_lower not in visited_destinations:
            return None

        visit_count = visited_destinations.count(dest_lower)
        now = datetime.now(timezone.utc)

        return Offer(
            title=f"Welcome Back to {entities.destination}!",
            description=(
                f"You've visited {entities.destination} {visit_count} time(s) before. "
                f"This time, unlock a hidden gems itinerary with 5% off!"
            ),
            offer_type=OfferType.REVISIT,
            discount_percent=5.0,
            promo_code=f"HIMBACK{entities.destination[:4].upper()}5",
            valid_until=now + timedelta(days=30),
            applicable_to=["cab", "package"],
            conditions="Includes curated off-beat itinerary with lesser-known spots.",
        )

    def _generate_monsoon_offer(
        self,
        entities: ExtractedEntities,
        now: datetime,
    ) -> Optional[Offer]:
        """Special monsoon season package with safety emphasis."""
        if now.month not in MONSOON_MONTHS:
            return None

        dest = entities.destination or "Himachal Pradesh"

        return Offer(
            title="Monsoon Magic Package",
            description=f"Experience the lush green {dest} in monsoon! Special rates with safety-first cab service.",
            offer_type=OfferType.MONSOON_SPECIAL,
            discount_percent=10.0,
            promo_code="HIMMONSOON10",
            valid_until=now + timedelta(days=30),
            applicable_to=["cab", "package"],
            conditions="Includes real-time road condition updates and flexible rescheduling for weather disruptions.",
        )


# ── Singleton ─────────────────────────────────────────────────────────────────

_offers_service: Optional[OffersService] = None


def get_offers_service() -> OffersService:
    global _offers_service
    if _offers_service is None:
        _offers_service = OffersService()
    return _offers_service
