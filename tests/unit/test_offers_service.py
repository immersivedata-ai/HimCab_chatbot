"""
tests/unit/test_offers_service.py
──────────────────────────────────
Unit tests for the OffersService — validates offer generation
for different loyalty tiers, visitor types, seasons, and revisits.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from app.models.offers import LoyaltyTier, OfferType
from app.models.session import ExtractedEntities
from app.models.user import UserProfile, PastTrip
from app.services.offers import OffersService


@pytest.fixture
def offers_service():
    return OffersService()


@pytest.fixture
def base_entities():
    return ExtractedEntities(
        destination="Manali",
        origin="Shimla",
        duration_days=2,
        visitor_type="local",
    )


@pytest.fixture
def outstation_entities():
    return ExtractedEntities(
        destination="Manali",
        origin="Delhi",
        duration_days=4,
        visitor_type="outstation",
    )


@pytest.fixture
def new_user():
    return UserProfile(
        user_id="u-new",
        firebase_uid="fb-new",
        total_trips=0,
    )


@pytest.fixture
def regular_user():
    return UserProfile(
        user_id="u-regular",
        firebase_uid="fb-regular",
        total_trips=2,
        past_trips=[
            PastTrip(destination="Shimla", duration_days=3, travel_date="2026-01-15"),
            PastTrip(destination="Dharamshala", duration_days=2, travel_date="2026-03-20"),
        ],
    )


@pytest.fixture
def frequent_user():
    return UserProfile(
        user_id="u-frequent",
        firebase_uid="fb-frequent",
        total_trips=4,
        home_city="Shimla",
        past_trips=[
            PastTrip(destination="Manali", duration_days=3, travel_date="2025-12-01"),
            PastTrip(destination="Dharamshala", duration_days=2, travel_date="2026-02-10"),
            PastTrip(destination="Kasol", duration_days=2, travel_date="2026-04-05"),
            PastTrip(destination="Manali", duration_days=2, travel_date="2026-06-15"),
        ],
    )


@pytest.fixture
def premium_user():
    return UserProfile(
        user_id="u-premium",
        firebase_uid="fb-premium",
        total_trips=8,
        home_city="Kullu",
        is_premium=True,
        past_trips=[
            PastTrip(destination="Manali", duration_days=3, travel_date="2025-06-01"),
            PastTrip(destination="Shimla", duration_days=2, travel_date="2025-08-10"),
            PastTrip(destination="Dharamshala", duration_days=4, travel_date="2025-10-15"),
            PastTrip(destination="Kasol", duration_days=2, travel_date="2025-12-01"),
            PastTrip(destination="Spiti", duration_days=5, travel_date="2026-01-20"),
            PastTrip(destination="Manali", duration_days=2, travel_date="2026-03-10"),
            PastTrip(destination="Dalhousie", duration_days=3, travel_date="2026-05-01"),
            PastTrip(destination="Manali", duration_days=2, travel_date="2026-06-20"),
        ],
    )


# ── Loyalty Tier Tests ────────────────────────────────────────────────────────

class TestLoyaltyTiers:
    def test_new_user_tier(self, new_user):
        assert new_user.get_loyalty_tier() == LoyaltyTier.NEW
        assert not new_user.is_frequent_traveller

    def test_regular_user_tier(self, regular_user):
        assert regular_user.get_loyalty_tier() == LoyaltyTier.REGULAR
        assert not regular_user.is_frequent_traveller

    def test_frequent_user_tier(self, frequent_user):
        assert frequent_user.get_loyalty_tier() == LoyaltyTier.FREQUENT
        assert frequent_user.is_frequent_traveller

    def test_premium_user_tier(self, premium_user):
        assert premium_user.get_loyalty_tier() == LoyaltyTier.PREMIUM
        assert premium_user.is_frequent_traveller


# ── Offer Generation Tests ───────────────────────────────────────────────────

class TestOffersGeneration:
    def test_new_user_no_loyalty_offers(self, offers_service, new_user, base_entities):
        """New users should not get loyalty offers."""
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        loyalty_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOYALTY]
        assert len(loyalty_offers) == 0

    def test_regular_user_gets_5_percent(self, offers_service, regular_user, base_entities):
        """Regular tier users should get a 5% loyalty offer."""
        bundle = offers_service.generate_offers(regular_user, base_entities, "local")
        loyalty_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOYALTY]
        assert len(loyalty_offers) == 1
        assert loyalty_offers[0].discount_percent == 5.0
        assert loyalty_offers[0].promo_code == "HIMBACK5"

    def test_frequent_user_gets_10_percent(self, offers_service, frequent_user, base_entities):
        """Frequent tier users should get a 10% loyalty offer."""
        bundle = offers_service.generate_offers(frequent_user, base_entities, "local")
        loyalty_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOYALTY]
        assert len(loyalty_offers) == 1
        assert loyalty_offers[0].discount_percent == 10.0
        assert loyalty_offers[0].promo_code == "HIMFREQ10"

    def test_premium_user_gets_15_percent_plus_companion(self, offers_service, premium_user, base_entities):
        """Premium tier users should get 15% loyalty + companion offer."""
        bundle = offers_service.generate_offers(premium_user, base_entities, "local")
        loyalty_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOYALTY]
        assert len(loyalty_offers) == 2
        discounts = {o.discount_percent for o in loyalty_offers}
        assert 15.0 in discounts
        assert 5.0 in discounts  # companion offer


class TestLocalOffers:
    def test_local_user_gets_local_rate(self, offers_service, new_user, base_entities):
        """Local visitors should get HP resident special rate."""
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        local_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOCAL_RATE]
        assert len(local_offers) == 1
        assert local_offers[0].discount_percent == 7.0
        assert local_offers[0].promo_code == "HIMLOCAL7"

    def test_outstation_user_no_local_offers(self, offers_service, new_user, outstation_entities):
        """Outstation visitors should NOT get local-specific offers."""
        bundle = offers_service.generate_offers(new_user, outstation_entities, "outstation")
        local_offers = [o for o in bundle.offers if o.offer_type == OfferType.LOCAL_RATE]
        assert len(local_offers) == 0

    def test_weekend_getaway_for_short_trips(self, offers_service, new_user, base_entities):
        """1-2 day trips should get weekend getaway offer."""
        base_entities.duration_days = 2
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        weekend_offers = [o for o in bundle.offers if o.offer_type == OfferType.WEEKEND_GETAWAY]
        assert len(weekend_offers) == 1
        assert weekend_offers[0].discount_percent == 12.0

    def test_no_weekend_getaway_for_long_trips(self, offers_service, new_user, base_entities):
        """3+ day trips should NOT get weekend getaway offer."""
        base_entities.duration_days = 5
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        weekend_offers = [o for o in bundle.offers if o.offer_type == OfferType.WEEKEND_GETAWAY]
        assert len(weekend_offers) == 0


class TestRevisitOffers:
    def test_revisit_offer_for_returning_user(self, offers_service, frequent_user, base_entities):
        """Users returning to a previously visited destination should get revisit offer."""
        base_entities.destination = "Manali"  # frequent_user has visited Manali twice
        bundle = offers_service.generate_offers(frequent_user, base_entities, "local")
        revisit_offers = [o for o in bundle.offers if o.offer_type == OfferType.REVISIT]
        assert len(revisit_offers) == 1
        assert "Manali" in revisit_offers[0].title

    def test_no_revisit_offer_for_new_destination(self, offers_service, frequent_user, base_entities):
        """Users visiting a new destination should NOT get revisit offer."""
        base_entities.destination = "Spiti"
        bundle = offers_service.generate_offers(frequent_user, base_entities, "local")
        revisit_offers = [o for o in bundle.offers if o.offer_type == OfferType.REVISIT]
        assert len(revisit_offers) == 0


class TestSeasonalOffers:
    @patch("app.services.offers.datetime")
    def test_seasonal_offer_in_offpeak(self, mock_datetime, offers_service, new_user, base_entities):
        """Off-peak months should generate seasonal offers."""
        mock_now = datetime(2026, 2, 15, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        seasonal_offers = [o for o in bundle.offers if o.offer_type == OfferType.SEASONAL]
        assert len(seasonal_offers) == 1
        assert seasonal_offers[0].discount_percent == 8.0

    @patch("app.services.offers.datetime")
    def test_no_seasonal_offer_in_peak(self, mock_datetime, offers_service, new_user, base_entities):
        """Peak months should NOT generate seasonal offers."""
        mock_now = datetime(2026, 6, 15, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        seasonal_offers = [o for o in bundle.offers if o.offer_type == OfferType.SEASONAL]
        assert len(seasonal_offers) == 0

    @patch("app.services.offers.datetime")
    def test_monsoon_offer_in_july(self, mock_datetime, offers_service, new_user, base_entities):
        """July/August should generate monsoon special offers."""
        mock_now = datetime(2026, 7, 15, tzinfo=timezone.utc)
        mock_datetime.now.return_value = mock_now
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)
        bundle = offers_service.generate_offers(new_user, base_entities, "local")
        monsoon_offers = [o for o in bundle.offers if o.offer_type == OfferType.MONSOON_SPECIAL]
        assert len(monsoon_offers) == 1
        assert monsoon_offers[0].discount_percent == 10.0


class TestOfferBundle:
    def test_best_discount(self, offers_service, premium_user, base_entities):
        """Best discount should be the highest among all offers."""
        base_entities.duration_days = 2
        bundle = offers_service.generate_offers(premium_user, base_entities, "local")
        assert bundle.has_offers
        assert bundle.best_discount == 15.0  # Premium loyalty offer

    def test_display_dict_format(self, offers_service, frequent_user, base_entities):
        """Display dict should have proper structure."""
        bundle = offers_service.generate_offers(frequent_user, base_entities, "local")
        display = bundle.to_display_dict()
        assert "offers" in display
        assert "loyalty_tier" in display
        assert "total_trips" in display
        assert "best_discount" in display
        assert display["loyalty_tier"] == "frequent"
        assert display["total_trips"] == 4

    def test_no_user_profile_defaults(self, offers_service, base_entities):
        """Should handle None user_profile gracefully."""
        bundle = offers_service.generate_offers(None, base_entities, "local")
        assert bundle.loyalty_tier == LoyaltyTier.NEW
        assert bundle.total_trips == 0
