"""
tests/unit/test_offers.py
──────────────────────────
Real coupons, and only real coupons.

Two failure modes this guards, in opposite directions:

  • quoting a coupon the rider cannot use — wrong trip type, expired, or a minimum
    spend above their fare. They tap through to the payment screen, the code is
    rejected, and the assistant has cost them trust over Rs 200.
  • quoting one that does not exist. The writer's standing instruction used to be
    "never mention a coupon" precisely because that was the only way to stop it.

So: parsing is strict, filtering is strict, the arithmetic is done in code rather than
by the model, and the prompt is only allowed to name what came back from the backend.
"""

from datetime import datetime, timedelta, timezone

import pytest

from app.services.himcab.models import Offer
from app.services.himcab.offers_service import (
    _parse_offer,
    _parse_moment,
    _rows_from,
    usable_offers,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc)


def _row(**overrides) -> dict:
    """A backend offer row, in the shape /offer/getOffersByTripType returns."""
    row = {
        "id": 7,
        "couponCode": "himcab200",
        "couponName": "Monsoon saver",
        "couponValue": 200,
        "minimumValue": 500,
        "offerTripTypes": ["LOCAL", "OUTSTATION"],
        "offerUserType": "EXISTING_USER",
        "termsAndConditions": "One use per rider.",
        "status": True,
        "liveDate": "2026-08-01T00:00:00Z",
        "expireDate": "2026-09-01T00:00:00Z",
    }
    row.update(overrides)
    return row


# ── Parsing what the backend actually sends ───────────────────────────────────

class TestParsing:
    def test_parses_a_normal_row(self):
        offer = _parse_offer(_row())
        assert offer.offer_id == 7
        assert offer.coupon_code == "HIMCAB200"   # upper-cased: the app sends it as-is
        assert offer.coupon_value_inr == 200
        assert offer.minimum_value_inr == 500
        assert offer.trip_types == ["LOCAL", "OUTSTATION"]

    def test_reads_the_singular_trip_type_field_too(self):
        """Swagger says `offerTripTypes` (a list); the Android model reads
        `offerTripType` (a string). Both appear on the wire."""
        offer = _parse_offer(
            _row(offerTripTypes=None, offerTripType="RENTAL")
        )
        assert offer.trip_types == ["RENTAL"]

    @pytest.mark.parametrize(
        "raw,expected_year",
        [
            (1754784000000, 2025),      # epoch millis — what the Android model declares
            (1754784000, 2025),         # epoch seconds
            ("2026-08-01T00:00:00Z", 2026),   # ISO — what Swagger declares
            ("2026-08-01T00:00:00", 2026),    # ISO, naive → read as UTC
        ],
    )
    def test_reads_every_timestamp_shape(self, raw, expected_year):
        parsed = _parse_moment(raw)
        assert parsed is not None
        assert parsed.year == expected_year
        assert parsed.tzinfo is not None

    def test_unparseable_timestamp_does_not_hide_the_offer(self):
        """None means "no bound on this side" — a coupon is never dropped because its
        date failed to parse."""
        assert _parse_moment("not a date") is None
        offer = _parse_offer(_row(expireDate="not a date"))
        assert offer.is_live(NOW) is True

    @pytest.mark.parametrize(
        "row",
        [
            _row(couponCode=None),   # nothing to type into the coupon box
            _row(couponCode=""),
            _row(couponValue=0),     # not an offer
            _row(couponValue=None),
            _row(couponValue="free"),
        ],
    )
    def test_unusable_rows_are_dropped(self, row):
        assert _parse_offer(row) is None

    def test_reads_both_response_envelopes(self):
        """`getOffersByTripType` returns data:[...]; the paginated `get-all` returns
        data:{content:[...]}. One reader for both."""
        assert _rows_from({"data": [_row()]}) == [_row()]
        assert _rows_from({"data": {"content": [_row()]}}) == [_row()]
        assert _rows_from({"data": None}) == []
        assert _rows_from("not a dict") == []


# ── Which coupons actually apply to THIS trip ─────────────────────────────────

class TestFiltering:
    def _offer(self, **kwargs) -> Offer:
        return _parse_offer(_row(**kwargs))

    def test_keeps_a_coupon_for_this_trip_type(self):
        offers = usable_offers([self._offer()], "LOCAL", fare_inr=900, now=NOW)
        assert len(offers) == 1

    def test_drops_a_coupon_for_another_trip_type(self):
        """A rental coupon on a local hop is rejected at checkout — do not offer it."""
        rental_only = self._offer(offerTripTypes=["RENTAL"])
        assert usable_offers([rental_only], "LOCAL", fare_inr=900, now=NOW) == []

    def test_all_covers_every_trip_type(self):
        blanket = self._offer(offerTripTypes=["ALL"])
        assert usable_offers([blanket], "OUTSTATION", fare_inr=900, now=NOW)

    def test_drops_an_expired_coupon(self):
        expired = self._offer(expireDate="2026-08-01T00:00:00Z")
        assert usable_offers([expired], "LOCAL", fare_inr=900, now=NOW) == []

    def test_drops_a_coupon_that_is_not_live_yet(self):
        future = self._offer(liveDate="2026-12-01T00:00:00Z")
        assert usable_offers([future], "LOCAL", fare_inr=900, now=NOW) == []

    def test_drops_a_coupon_the_fare_does_not_reach(self):
        """Rs 500 minimum on a Rs 300 ride. Mentioning it is a promise we cannot keep."""
        assert usable_offers([self._offer()], "LOCAL", fare_inr=300, now=NOW) == []

    def test_keeps_the_coupon_when_the_fare_is_unknown(self):
        """The quote failed, so we cannot say whether the trip clears the minimum. The
        offer is real; skip the check rather than guess at it."""
        assert usable_offers([self._offer()], "LOCAL", fare_inr=None, now=NOW)

    def test_best_saving_comes_first(self):
        small = self._offer(id=1, couponCode="SMALL", couponValue=50)
        big = self._offer(id=2, couponCode="BIG", couponValue=400)
        ordered = usable_offers([small, big], "LOCAL", fare_inr=2000, now=NOW)
        assert [o.coupon_code for o in ordered] == ["BIG", "SMALL"]


class TestDiscountMath:
    def test_discount_applies_above_the_minimum(self):
        assert _parse_offer(_row()).discount_on(900) == 200

    def test_no_discount_below_the_minimum(self):
        assert _parse_offer(_row()).discount_on(300) == 0.0

    def test_discount_never_exceeds_the_fare(self):
        """A Rs 500 coupon on a Rs 300 ride must not produce a negative price."""
        generous = _parse_offer(_row(couponValue=500, minimumValue=0))
        assert generous.discount_on(300) == 300


# ── The tool executor's offer stage ───────────────────────────────────────────

class TestOfferStageInToolExecutor:
    """`_fetch_offers` degrades to [] for every failure, because the writer may only
    name what is in that list — an empty one means the reply says nothing at all."""

    @pytest.fixture
    def state(self):
        from app.agent.state import AgentState
        from app.models.session import ExtractedEntities

        return AgentState(
            session_id="offers-1",
            user_id="rider:1",
            intent="plan_trip",
            visitor_type="outstation",
            rider_jwt="jwt-token",
            rider_id=42,
            entities=ExtractedEntities(destination="Manali", origin="Delhi"),
        )

    @pytest.mark.asyncio
    async def test_returns_priced_offers(self, state, monkeypatch):
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor

        service = AsyncMock()
        service.get_offers.return_value = [_parse_offer(_row())]
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        offers = await tool_executor._fetch_offers(state, "OUTSTATION", fare_inr=4200)

        assert offers[0]["coupon_code"] == "HIMCAB200"
        # The rider will check this against their receipt, so it is computed here
        # rather than left to the model.
        assert offers[0]["fare_after_offer_inr"] == 4000
        service.get_offers.assert_awaited_once_with(
            rider_jwt="jwt-token", trip_type="OUTSTATION", rider_id=42
        )

    @pytest.mark.asyncio
    async def test_backend_failure_is_silent(self, state, monkeypatch):
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor

        service = AsyncMock()
        service.get_offers.side_effect = RuntimeError("offers backend down")
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        assert await tool_executor._fetch_offers(state, "LOCAL", 900) == []

    @pytest.mark.asyncio
    async def test_no_jwt_means_no_offers(self, state, monkeypatch):
        from app.agent.nodes import tool_executor

        state.rider_jwt = None
        assert await tool_executor._fetch_offers(state, "LOCAL", 900) == []

    @pytest.mark.asyncio
    async def test_flag_off_means_no_offers(self, state, monkeypatch):
        from app.agent.nodes import tool_executor
        from app.core.config import settings

        monkeypatch.setattr(settings, "enable_offers", False)
        assert await tool_executor._fetch_offers(state, "LOCAL", 900) == []

    @pytest.mark.asyncio
    async def test_a_coupon_worth_the_whole_fare_promises_no_total(self, state, monkeypatch):
        """A Rs 5,000 coupon on a Rs 900 ride would read as "Rs 0" — a free ride the
        Spring booking flow will not honour. Quote the saving, promise no total."""
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor

        service = AsyncMock()
        service.get_offers.return_value = [
            _parse_offer(_row(couponValue=5000, minimumValue=0))
        ]
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        offers = await tool_executor._fetch_offers(state, "LOCAL", fare_inr=900)

        assert offers[0]["coupon_value_inr"] == 5000
        assert offers[0]["fare_after_offer_inr"] is None

    @pytest.mark.asyncio
    async def test_without_a_fare_eligibility_is_marked_unverified(self, state, monkeypatch):
        """The quote failed, so the minimum-spend filter could not run. The coupon is
        real; whether it applies to THIS trip is not something we can claim."""
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor

        service = AsyncMock()
        service.get_offers.return_value = [_parse_offer(_row())]   # Rs 500 minimum
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        offers = await tool_executor._fetch_offers(state, "LOCAL", fare_inr=None)

        assert offers[0]["applies_to_this_fare"] is False
        assert offers[0]["fare_after_offer_inr"] is None

    @pytest.mark.asyncio
    async def test_a_no_minimum_coupon_is_verified_even_without_a_fare(self, state, monkeypatch):
        """Nothing to check against a fare — it applies to any of them."""
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor

        service = AsyncMock()
        service.get_offers.return_value = [_parse_offer(_row(minimumValue=0))]
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        offers = await tool_executor._fetch_offers(state, "LOCAL", fare_inr=None)

        assert offers[0]["applies_to_this_fare"] is True

    @pytest.mark.asyncio
    async def test_reply_is_capped(self, state, monkeypatch):
        """The app's offers screen has the full list; a reply naming eight coupons
        reads as a spam email rather than a recommendation."""
        from unittest.mock import AsyncMock

        from app.agent.nodes import tool_executor
        from app.core.config import settings

        service = AsyncMock()
        service.get_offers.return_value = [
            _parse_offer(_row(id=i, couponCode=f"CODE{i}", couponValue=100 + i))
            for i in range(8)
        ]
        monkeypatch.setattr(tool_executor, "get_offers_service", lambda: service)

        offers = await tool_executor._fetch_offers(state, "LOCAL", 5000)
        assert len(offers) == settings.max_offers_in_reply


# ── What the writer is allowed to say about them ──────────────────────────────

class TestOffersPrompt:
    def _state(self, offers):
        from app.agent.state import AgentState

        return AgentState(session_id="p-1", user_id="rider:1", offers=offers)

    def test_no_offers_forbids_mentioning_discounts(self):
        from app.agent.nodes.writer import _build_offers_block

        block = _build_offers_block(self._state([]))
        assert "Do NOT mention, invent or hint at any discount" in block

    OFFER = {
        "coupon_code": "HIMCAB200",
        "coupon_name": "Monsoon saver",
        "coupon_value_inr": 200,
        "minimum_value_inr": 500,
        "fare_after_offer_inr": 4000,
        "applies_to_this_fare": True,
    }

    def test_offers_are_quoted_verbatim(self):
        from app.agent.nodes.writer import _build_offers_block

        block = _build_offers_block(self._state([self.OFFER]))

        assert "HIMCAB200" in block
        assert "Rs 200 off" in block
        assert "Rs 4,000" in block
        assert "Quote ONLY these" in block
        assert "These coupons apply to THIS trip" in block

    def test_an_unverified_coupon_is_not_claimed_to_apply(self):
        """Without a fare the minimum-spend check never ran, so the prompt must make
        the weaker claim — otherwise the reply promises a discount that gets rejected
        at the payment screen."""
        from app.agent.nodes.writer import _build_offers_block

        block = _build_offers_block(self._state([
            {**self.OFFER, "applies_to_this_fare": False, "fare_after_offer_inr": None}
        ]))

        assert "These coupons apply to THIS trip" not in block
        assert "do NOT claim they apply to this trip" in block
        assert "HIMCAB200" in block          # the coupon itself is still real

    def test_the_full_prompt_switches_between_the_two_rules(self):
        """The blanket "never mention a coupon" rule and the "quote these exactly" rule
        are mutually exclusive — the prompt must never carry both."""
        from app.agent.nodes.writer import _build_system_prompt
        from app.agent.state import AgentState
        from app.models.session import ExtractedEntities

        def prompt_for(offers):
            return _build_system_prompt(AgentState(
                session_id="s", user_id="u", intent="plan_trip", visitor_type="outstation",
                entities=ExtractedEntities(destination="Manali", origin="Delhi"),
                cab_fare={"fare_inr": 4200, "category_name": "Sedan", "options": [
                    {"category_name": "Sedan", "total_fare_inr": 4200}
                ]},
                offers=offers,
            ))

        silent = prompt_for([])
        assert "Write NOTHING about coupons" in silent
        # Silence means writing nothing, NOT announcing a nothing. Naming the situation
        # ("no offer applies to this trip") got written down verbatim as a bullet.
        assert "not a line saying none applies" in silent
        assert "HIM CAB OFFERS (real" not in silent

        speaking = prompt_for([{
            "coupon_code": "HIMCAB200", "coupon_name": "Monsoon saver",
            "coupon_value_inr": 200, "minimum_value_inr": 500,
            "fare_after_offer_inr": 4000,
        }])
        assert "Write NOTHING about coupons" not in speaking
        assert "HIMCAB200" in speaking


# ── What reaches the app ──────────────────────────────────────────────────────

class TestOffersReachTheApp:
    def _fare(self):
        return {
            "fare_inr": 4200,
            "category_name": "Sedan",
            "trip_type": "OUTSTATION",
            "options": [{"category_name": "Sedan", "total_fare_inr": 4200}],
            "best_offer": {"offer_id": 7, "coupon_code": "HIMCAB200"},
            "fare_after_offer_inr": 4000,
        }

    def _offers(self, **overrides):
        offer = {
            "offer_id": 7,
            "coupon_code": "HIMCAB200",
            "coupon_name": "Monsoon saver",
            "coupon_value_inr": 200,
            "minimum_value_inr": 500,
            "fare_after_offer_inr": 4000,
            "applies_to_this_fare": True,
        }
        offer.update(overrides)
        return [offer]

    def test_the_real_discount_lands_on_the_itinerary(self):
        """The rider will check the total against their receipt, so it is subtracted
        the same way the Spring booking flow subtracts it."""
        from app.agent.nodes.writer import _apply_real_fare
        from app.models.itinerary import Itinerary

        itinerary = Itinerary(destination="Manali", duration_days=3)
        _apply_real_fare(itinerary, self._fare())

        assert itinerary.cab_package.base_fare_inr == 4200
        assert itinerary.cab_package.discounted_fare_inr == 4000
        assert itinerary.cab_package.coupon_code == "HIMCAB200"
        assert itinerary.cost_breakdown.discount_inr == 200
        assert itinerary.cost_breakdown.total_inr == 4000

    def test_no_coupon_leaves_the_discount_fields_empty(self):
        """`discounted_fare_inr` is never a copy of the base fare — "is there a
        discount" has to stay a single check for the app."""
        from app.agent.nodes.writer import _apply_real_fare
        from app.models.itinerary import Itinerary

        itinerary = Itinerary(destination="Manali", duration_days=3)
        _apply_real_fare(itinerary, {
            "fare_inr": 4200, "category_name": "Sedan",
            "options": [{"category_name": "Sedan", "total_fare_inr": 4200}],
        })

        assert itinerary.cab_package.discounted_fare_inr is None
        assert itinerary.cab_package.coupon_code is None
        assert itinerary.cost_breakdown.discount_inr == 0

    ROUTE = {
        "origin": {"name": "Shimla", "lat": 31.1, "lon": 77.17},
        "destination": {"name": "Manali", "lat": 32.24, "lon": 77.19},
        "distance_km": 270, "duration_minutes": 270, "duration_text": "4h 30m",
        "is_approximate": False, "source": "surveyed",
    }

    def _handoff(self, offers):
        from app.agent.nodes.writer import _build_booking_handoff
        from app.agent.state import AgentState

        return _build_booking_handoff(AgentState(
            session_id="s", user_id="u",
            cab_fare=self._fare(), offers=offers, route_estimate=self.ROUTE,
        ))

    def test_the_booking_handoff_prefills_the_coupon(self):
        """`offer_id` is exactly what POST /api/rider/trip/book-ride takes, so the
        booking screen opens with the coupon the reply just promised."""
        handoff = self._handoff(self._offers())

        assert handoff["offer_id"] == 7
        assert handoff["coupon_code"] == "HIMCAB200"
        assert handoff["fare_after_offer_inr"] == 4000

    def test_an_unverified_coupon_is_not_prefilled(self):
        """No fare came back, so the minimum-spend check never ran. Preselecting the
        code would put one on the booking screen that the payment step then rejects."""
        handoff = self._handoff(self._offers(applies_to_this_fare=False))

        assert handoff["offer_id"] is None
        assert handoff["coupon_code"] is None
