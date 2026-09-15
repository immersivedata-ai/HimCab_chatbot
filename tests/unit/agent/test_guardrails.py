"""
tests/unit/agent/test_guardrails.py
────────────────────────────────────
The last gate before a rider reads a generated reply.

Every rule this checks was already stated in a prompt, and stated rules are what the
model kept breaking: a prompt that forbade sightseeing in writing still produced Mall
Road and the Ridge, and one that forbade prices when no fare was fetched is exactly
the kind of rule a model breaks while trying to be helpful.

So the finished text is read back and its CLAIMS are compared against the facts the
turn actually holds. The tests below are mostly about the two halves that matter:

  * a claim we cannot support is caught, and in a quote it is removed
  * a claim we CAN support is left alone

The second half is the one that decides whether this layer is usable. A guard that
strips real fares and real coupon codes is worse than no guard, because it breaks the
replies that were working.
"""

from __future__ import annotations

import pytest

from app.agent import slots
from app.agent.guardrails import CRITICAL, WARN, apply_guardrails, inspect_reply
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities

ROUTE = {
    "origin": {"name": "Kasol"},
    "destination": {"name": "Shimla"},
    "distance_km": 215.0,
    "duration_minutes": 270,
    "duration_text": "4h 30m",
}

FARE = {
    "fare_inr": 4100,
    "options": [
        {"category_name": "Economy", "total_fare_inr": 3200, "seat_capacity": 4},
        {"category_name": "Sedan", "total_fare_inr": 4100, "seat_capacity": 4},
    ],
}

OFFERS = [{
    "coupon_code": "HIMCAB300",
    "coupon_value_inr": 300,
    "fare_after_offer_inr": 3800,
    "applies_to_this_fare": True,
}]


def _state(**overrides) -> AgentState:
    kwargs = {
        "session_id": "s",
        "user_id": "u",
        "intent": "rental_query",
        "visitor_type": "rental",
        "entities": ExtractedEntities(visitor_type="rental", origin="Shimla", rental_hours=8),
        "route_estimate": ROUTE,
        "cab_fare": FARE,
        "offers": OFFERS,
    }
    kwargs.update(overrides)
    return AgentState(**kwargs)


def _kinds(findings) -> set[str]:
    return {f.kind for f in findings}


# ── 1. Claims we cannot support ───────────────────────────────────────────────


class TestUnsupportedClaimsAreCaught:

    def test_an_invented_coupon_code(self):
        """The rider takes this to the payment screen. A code we made up is rejected
        there, which is worse than never mentioning a discount at all."""
        findings = inspect_reply(
            _state(), "Use SUMMER50 for a discount", slots.RENTAL_MODE, "en"
        )
        assert "invented_coupon" in _kinds(findings)
        assert all(f.severity == CRITICAL for f in findings if f.kind == "invented_coupon")

    def test_a_price_when_no_fare_was_fetched(self):
        """The prompt already forbids any number here. This is what enforces it."""
        findings = inspect_reply(
            _state(cab_fare=None), "It will cost about Rs 3,500", slots.RENTAL_MODE, "en"
        )
        assert "unsupported_fare" in _kinds(findings)

    def test_a_price_that_is_not_one_of_the_quoted_ones(self):
        findings = inspect_reply(_state(), "Sedan: Rs 9,999", slots.RENTAL_MODE, "en")
        assert "contradicted_fare" in _kinds(findings)

    def test_a_distance_that_contradicts_the_measured_route(self):
        findings = inspect_reply(_state(), "About 400 km away", slots.RENTAL_MODE, "en")
        assert "contradicted_distance" in _kinds(findings)

    def test_a_day_by_day_plan_inside_a_quote(self):
        """The reported failure, in the flow where it has no business at all."""
        findings = inspect_reply(
            _state(), "Day 1: visit Mall Road", slots.RENTAL_MODE, "en"
        )
        assert "itinerary_in_a_quote" in _kinds(findings)

    def test_a_reply_in_the_wrong_language(self):
        findings = inspect_reply(_state(), "Your cab is ready", slots.RENTAL_MODE, "hi")
        assert "wrong_language" in _kinds(findings)


# ── 2. Claims we CAN support are left alone ───────────────────────────────────


class TestSupportedClaimsSurvive:
    """The half that decides whether this layer is usable at all. A guard that strips
    real fares is worse than no guard."""

    def test_a_quoted_fare_is_fine(self):
        assert inspect_reply(_state(), "Sedan: Rs 4,100", slots.RENTAL_MODE, "en") == []

    def test_every_option_price_is_fine(self):
        findings = inspect_reply(
            _state(), "Economy Rs 3,200 or Sedan Rs 4,100", slots.RENTAL_MODE, "en"
        )
        assert findings == []

    def test_a_real_coupon_and_its_value_are_fine(self):
        findings = inspect_reply(
            _state(), "Use HIMCAB300 to save Rs 300", slots.RENTAL_MODE, "en"
        )
        assert findings == []

    def test_a_rounded_distance_is_fine(self):
        """"about 215 km" for a measured 215.0 is right, and so is a sensible rounding
        of it — the check is a tolerance, not an equality."""
        for text in ("about 215 km", "215 km", "about 210 km"):
            assert inspect_reply(_state(), text, slots.RENTAL_MODE, "en") == []

    @pytest.mark.parametrize("word", ["SUV", "MANALI", "HIMACHAL", "OK", "AC"])
    def test_ordinary_upper_case_words_are_not_coupon_codes(self, word):
        """A code contains a digit. Without that rule every vehicle class and place
        name in the reply would be reported as an invented coupon."""
        assert inspect_reply(_state(), f"A {word} is available", slots.RENTAL_MODE, "en") == []

    def test_a_hindi_reply_for_a_hindi_rider_is_fine(self):
        findings = inspect_reply(_state(), "आपकी गाड़ी तैयार है", slots.RENTAL_MODE, "hi")
        assert findings == []


# ── 3. The same claim is graded by the reply it appears in ────────────────────


class TestGradingDependsOnTheFlow:
    """An itinerary is prose about a holiday, where an approximate hotel figure is part
    of the product. A quote is transactional, where every number is one the rider will
    be charged."""

    def test_an_estimate_in_an_itinerary_is_only_a_warning(self):
        findings = inspect_reply(
            _state(intent="plan_trip", visitor_type="outstation"),
            "Hotels are around Rs 2,500 a night",
            slots.ITINERARY_MODE,
            "en",
        )
        assert [f.severity for f in findings] == [WARN]

    def test_the_same_figure_in_a_quote_is_critical(self):
        findings = inspect_reply(
            _state(), "That will be Rs 2,500", slots.RENTAL_MODE, "en"
        )
        assert CRITICAL in [f.severity for f in findings]

    def test_an_invented_coupon_is_critical_everywhere(self):
        """There is no reply in which naming a code the backend never returned is
        acceptable — the rider is charged by somebody else's system."""
        for mode in (slots.RENTAL_MODE, slots.ITINERARY_MODE, slots.ANSWER_MODE):
            findings = inspect_reply(_state(), "Try FAKE99 now", mode, "en")
            assert any(
                f.kind == "invented_coupon" and f.severity == CRITICAL for f in findings
            ), mode


# ── 4. What actually ships ────────────────────────────────────────────────────


class TestRemediation:

    def test_a_critical_line_is_removed_and_the_rest_kept(self):
        """Most of a flawed reply is fine, and discarding all of it costs the rider
        more than the bad line did."""
        text = (
            "Your rental is confirmed.\n"
            "- Sedan: Rs 4,100\n"
            "- Use SUMMER50 to save Rs 900\n"
            "- Use HIMCAB300 to save Rs 300"
        )
        clean, findings = apply_guardrails(_state(), text, slots.RENTAL_MODE, "en")

        assert "SUMMER50" not in clean
        assert "Rs 4,100" in clean
        assert "HIMCAB300" in clean
        assert "Your rental is confirmed." in clean
        assert findings

    def test_a_clean_reply_is_returned_untouched(self):
        text = "Your rental is confirmed.\n- Sedan: Rs 4,100"
        clean, findings = apply_guardrails(_state(), text, slots.RENTAL_MODE, "en")
        assert clean == text
        assert findings == []

    def test_it_never_ships_an_empty_reply(self):
        """If removal would leave nothing, the original is kept and the finding is
        logged loudly. An empty message helps nobody, and something that severe needs a
        person to look at it."""
        text = "Use SUMMER50 to save Rs 900"
        clean, findings = apply_guardrails(_state(), text, slots.RENTAL_MODE, "en")
        assert clean.strip()
        assert findings

    def test_an_empty_reply_in_yields_no_findings(self):
        assert inspect_reply(_state(), "", slots.RENTAL_MODE, "en") == []
        assert inspect_reply(_state(), "   ", slots.RENTAL_MODE, "en") == []

    def test_it_never_raises_on_a_sparse_turn(self):
        """A turn where nothing was fetched is the common case for a failed backend,
        and the guard must not be the thing that turns that into a 500."""
        bare = _state(cab_fare=None, offers=[], route_estimate=None)
        clean, _ = apply_guardrails(bare, "Here is your ride.", slots.RIDE_MODE, "en")
        assert clean == "Here is your ride."
