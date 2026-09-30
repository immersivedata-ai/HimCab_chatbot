"""
tests/unit/test_fare_pricing.py
───────────────────────────────
Regression tests for the fare a rider is actually shown.

Two reported failures live here:

  1. "no fare at all" — the distance behind /fare/estimate came from a hand-listed
     table of city pairs, so any route nobody had measured produced no price. Distance
     is now resolved live (see tests/unit/test_geo_live.py); what this file covers is
     what happens to the fare once there IS one.
  2. "a Mini Bus for four travellers" — the endpoint prices EVERY category and the
     reply repeated the list verbatim, which is not the same as recommending something.

`TestTripTypeDescribesTheJourney` covers the third: trip type follows the JOURNEY, not
the rider's persona, so a resident driving 270 km is not priced off the local rate card.
"""

from typing import ClassVar

import pytest

from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.integrations.external.geocoding import Place
from app.integrations.external.routing import RoadRoute
from app.integrations.himcab.quote_builder import (
    build_fare_quote_input,
    classify_trip_type,
    trip_type_for,
)
from app.services.geo import Journey


def _journey(distance_km: float, origin="Shimla", destination="Manali") -> Journey:
    """A resolved journey, as the tool executor hands one to the quote builder."""
    return Journey(
        origin=Place(name=origin, lat=31.1, lon=77.17, state="Himachal Pradesh"),
        destination=Place(name=destination, lat=32.24, lon=77.19, state="Himachal Pradesh"),
        route=RoadRoute(distance_km=distance_km, duration_minutes=270),
    )


# ── Trip type: LOCAL for a resident (matches the app), journey for a visitor ──────

class TestTripTypeChoosesTheRateCard:
    """`trip_type_for` is the one place the rate card is decided.

    The reported issue: our OUTSTATION fare (₹4,675) disagreed with the app's LOCAL fare
    (₹231) for the same Palampur→Kasol resident ride. Him Cab prices a resident on the
    flat LOCAL card regardless of distance, so the resident flow now does too — the quote
    the rider is shown must equal the one they are charged. A VISITOR is still priced by
    the journey (`classify_trip_type`), which stays distance-based and unchanged.
    """

    @pytest.mark.parametrize(
        "distance_km,expected",
        [(8, "LOCAL"), (16, "LOCAL"), (40, "LOCAL"),
         (41, "OUTSTATION"), (270, "OUTSTATION"), (570, "OUTSTATION"), (None, "OUTSTATION")],
    )
    def test_classify_trip_type_is_still_distance_based(self, distance_km, expected):
        assert classify_trip_type(distance_km) == expected

    @pytest.mark.parametrize("distance_km", [5, 16, 177, 270, 570, None])
    def test_a_resident_is_priced_local_at_every_distance(self, distance_km):
        """A resident's ride is LOCAL whether it is 5 km or 570 km — this is the reversal
        that makes the quote match the app."""
        assert trip_type_for("local", distance_km) == "LOCAL"
        assert trip_type_for("LOCAL", distance_km) == "LOCAL"  # case-insensitive

    @pytest.mark.parametrize(
        "distance_km,expected",
        [(16, "LOCAL"), (270, "OUTSTATION"), (None, "OUTSTATION")],
    )
    def test_a_visitor_or_unknown_persona_is_priced_by_the_journey(self, distance_km, expected):
        assert trip_type_for("outstation", distance_km) == expected
        assert trip_type_for(None, distance_km) == expected

    def test_the_resident_flow_prices_palampur_kasol_local_to_match_the_app(self):
        """End to end through the builder, the way tool_executor calls it: a 177 km
        resident ride is LOCAL, so the fare equals the app's ₹231-card instead of the
        distance-scaled OUTSTATION number."""
        entities = ExtractedEntities(
            origin="Palampur", home_city="Palampur", destination="Kasol", visitor_type="local"
        )
        quote = build_fare_quote_input(
            entities, _journey(177, "Palampur", "Kasol"),
            trip_type=trip_type_for(entities.visitor_type, 177),
        )
        assert quote.trip_type == "LOCAL"
        assert quote.distance_km == 177

    def test_the_quote_uses_the_journey_that_was_resolved(self):
        """The builder used to re-derive the route from entity names when none was
        passed, which priced a different trip than the reply described — a rider in
        Solang Valley got the fare from Manali, 14 km down the road."""
        entities = ExtractedEntities(origin="anything at all", destination="whatever")
        quote = build_fare_quote_input(entities, _journey(42, "Kullu", "Kasol"))

        assert (quote.origin, quote.destination) == ("Kullu", "Kasol")
        assert quote.distance_km == 42

    def test_the_booking_time_is_the_riders_clock(self):
        """A 2 a.m. airport run priced as midday is then charged the night rate at
        booking. The quote has to be built on the clock the booking will be."""
        quote = build_fare_quote_input(
            ExtractedEntities(origin="Shimla", destination="Manali"), _journey(270)
        )
        hours, _, minutes = quote.booking_time.partition(":")
        assert 0 <= int(hours) <= 23 and 0 <= int(minutes) <= 59


# ── 4. Only cabs that fit the group are offered ───────────────────────────────

class TestCabsAreNarrowedToTheGroup:
    """The reported bug: a party of four was recommended a Mini Bus.

    `/fare/estimate` prices EVERY category in one call, and the reply repeated the
    list verbatim — which is not the same as recommending something.
    """

    CATEGORIES: ClassVar[list] = [
        ("Economy", 1800), ("Sedan", 2200), ("SUV", 3100),
        ("Innova Crysta", 3900), ("Tempo Traveller", 6500), ("Mini Bus", 9800),
    ]

    def _options(self):
        from app.integrations.himcab.models import FareCategoryFare

        return [
            FareCategoryFare(category_id=i, category_name=name, total_fare_inr=fare)
            for i, (name, fare) in enumerate(self.CATEGORIES, 1)
        ]

    @pytest.mark.parametrize(
        "group_size,expected",
        [
            (1, ["Economy", "Sedan", "SUV"]),
            (4, ["Economy", "Sedan", "SUV"]),
            (6, ["SUV", "Innova Crysta", "Tempo Traveller"]),
            (7, ["Innova Crysta", "Tempo Traveller", "Mini Bus"]),
            (12, ["Tempo Traveller", "Mini Bus"]),
        ],
    )
    def test_only_cabs_that_seat_the_group_are_offered(self, group_size, expected):
        from app.integrations.himcab.vehicles import select_cabs_for_group

        chosen = [c.category_name for c in select_cabs_for_group(self._options(), group_size)]
        assert chosen == expected

    def test_a_family_of_four_is_never_shown_a_minibus(self):
        """The literal reported symptom."""
        from app.integrations.himcab.vehicles import select_cabs_for_group

        chosen = [c.category_name for c in select_cabs_for_group(self._options(), 4)]
        assert not any("Bus" in name for name in chosen)

    def test_the_smallest_suitable_cab_leads(self):
        """A family of four wants a sedan, not an empty minibus at three times the price."""
        from app.integrations.himcab.vehicles import select_cabs_for_group

        assert select_cabs_for_group(self._options(), 5)[0].category_name == "SUV"

    def test_a_group_too_big_for_any_cab_still_sees_the_largest(self):
        """They need to know two vehicles are required, not that Him Cab has nothing."""
        from app.integrations.himcab.vehicles import select_cabs_for_group

        chosen = select_cabs_for_group(self._options(), 30)
        assert chosen and chosen[0].category_name == "Mini Bus"

    def test_with_no_group_size_nothing_is_ruled_out(self):
        """Narrowing on a guess would hide a cab the rider wanted."""
        from app.integrations.himcab.vehicles import select_cabs_for_group

        assert len(select_cabs_for_group(self._options(), None)) == 3

    def test_the_backends_own_seat_count_wins_over_the_name(self):
        from app.integrations.himcab.models import FareCategoryFare
        from app.integrations.himcab.vehicles import effective_seats

        # Named like a 4-seater, but the backend says otherwise.
        option = FareCategoryFare(category_name="Sedan", total_fare_inr=100, seat_capacity=6)
        assert effective_seats(option) == 6

    @pytest.mark.parametrize(
        "name,seats",
        [("Mini Bus", 20), ("Mini", 4), ("mini bus", 20), ("Tempo Traveller", 12),
         ("Innova Crysta", 7), ("Hatchback", 4), ("Something Odd", None)],
    )
    def test_seat_inference_from_the_category_name(self, name, seats):
        """"Mini Bus" contains "mini" — longest match must win or a 20-seater reads
        as a hatchback and gets offered to a group of twelve."""
        from app.integrations.himcab.vehicles import seat_capacity_for

        assert seat_capacity_for(name) == seats

    def test_an_unrecognised_category_is_kept_rather_than_filtered_out(self):
        """Hiding a cab the rider could have taken is worse than showing one extra."""
        from app.integrations.himcab.models import FareCategoryFare
        from app.integrations.himcab.vehicles import select_cabs_for_group

        options = [FareCategoryFare(category_name="Mystery Wagon", total_fare_inr=2000)]
        assert select_cabs_for_group(options, 6)


# ── Cab prices reach the rider intact ─────────────────────────────────────────

class TestFaresAreQuotedExactly:

    def _prompt(self, cab_fare: dict | None) -> str:
        from app.agent.nodes.writer import _build_system_prompt

        return _build_system_prompt(AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="outstation",
            entities=ExtractedEntities(destination="Manali", origin="Gurgaon", duration_days=3),
            cab_fare=cab_fare,
        ))

    OPTIONS: ClassVar[list] = [
        {"category_name": "Economy", "total_fare_inr": 8400.0, "seat_capacity": 4},
        {"category_name": "Standard", "total_fare_inr": 9600.0, "seat_capacity": 4},
    ]

    def test_every_price_reaches_the_prompt_unchanged(self):
        prompt = self._prompt({
            "fare_inr": 9600, "category_name": "Standard", "options": self.OPTIONS,
            "distance_km": 522,
        })

        for option in self.OPTIONS:
            assert f"{round(option['total_fare_inr']):,}" in prompt
        assert "522 km" in prompt
        assert "Do NOT round, adjust, average or invent a number" in prompt
        assert "recommended for this group/budget" in prompt

    def test_with_no_fare_the_model_is_told_not_to_invent_one(self):
        prompt = self._prompt(None)

        assert "do NOT state, estimate or imply any price" in prompt
        # …but it must still pitch Him Cab rather than going silent on booking.
        assert "Book Your Him Cab" in prompt

    def test_the_him_cab_section_is_never_optional(self):
        for cab_fare in (None, {"fare_inr": 9600, "category_name": "Standard",
                                "options": self.OPTIONS}):
            assert "This section is REQUIRED" in self._prompt(cab_fare)


# ── The real estimate response parses ─────────────────────────────────────────

#: A CAPTURED /api/rider/fare/estimate 200 body — Palampur→Kasol, priced by the live
#: backend on 2026-09-24. Kept verbatim (image URLs trimmed) because the reported bug
#: was invisible to every hand-written stub: the backend sends `seatCapacity` as a RANGE
#: STRING ("4 - 6"), and `FareCategoryFare.seat_capacity` is an int, so the model raised
#: a ValidationError that took down the WHOLE parse — get_quote raised, _safe_call
#: swallowed it to None, and the rider got a ride reply with no cabs at all. A valid
#: rider JWT was the first thing to ever reach a real response, which is why it hid for
#: so long behind expired tokens and stubs that used the shape the code already expected.
REAL_ESTIMATE_RESPONSE: dict = {
    "message": "SUCCESS",
    "statusCode": 200,
    "errorMessage": "",
    "data": [
        {"categoryId": 16561, "name": "Premium", "totalFare": 6940.61, "seatCapacity": "4 - 6"},
        {"categoryId": 12377, "name": "Standard", "totalFare": 4675.13, "seatCapacity": "4 - 5"},
        {"categoryId": 12538, "name": "Him Cab Mini Bus", "totalFare": 6919.61,
         "seatCapacity": "4 - 10"},
        {"categoryId": 12385, "name": "Family", "totalFare": 5988.78, "seatCapacity": "6 - 7"},
        {"categoryId": 12381, "name": "Comfort", "totalFare": 4718.45, "seatCapacity": "3 - 4"},
    ],
}


class TestTheRealEstimateResponseParses:
    """The response the live backend actually returns — not the shape the code wished for.

    Every assertion here would have failed before `_parse_seat_capacity`, because the
    range-string `seatCapacity` raised inside `_extract_categories` and zeroed the quote.
    """

    def _input(self, group_size=4):
        from app.integrations.himcab.models import FareQuoteInput

        return FareQuoteInput(
            origin="Palampur", destination="Kasol", distance_km=177.3,
            trip_type="OUTSTATION", group_size=group_size,
        )

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("4 - 6", 6), ("4-5", 5), ("4 - 10", 10), ("6 - 7", 7), ("3 - 4", 4),
            ("6", 6), (6, 6), (6.0, 6), ("6 seats", 6),
            (None, None), ("", None), ("n/a", None), (0, None), (-2, None), (True, None),
        ],
    )
    def test_seat_capacity_is_read_as_the_upper_bound(self, raw, expected):
        """"4 - 6" means a cab that seats up to six, so a party of six still fits it."""
        from app.integrations.himcab.fare_service import _parse_seat_capacity

        assert _parse_seat_capacity(raw) == expected

    def test_a_range_string_no_longer_zeroes_the_whole_quote(self):
        """The literal reported symptom: one "4 - 6" used to make base_fare_inr 0, which
        _package_fare turns into None → no cabs. It must now price the trip."""
        from app.integrations.himcab.fare_service import FareService

        quote = FareService()._parse_quote_response(REAL_ESTIMATE_RESPONSE, self._input())

        assert quote.base_fare_inr > 0
        assert quote.alternatives, "the real response must yield priced categories"

    def test_every_category_gets_an_integer_capacity(self):
        from app.integrations.himcab.fare_service import FareService

        parsed = FareService._extract_categories(REAL_ESTIMATE_RESPONSE["data"])
        seats = {c.category_name: c.seat_capacity for c in parsed}

        assert seats == {
            "Premium": 6, "Standard": 5, "Him Cab Mini Bus": 10, "Family": 7, "Comfort": 4,
        }

    def test_a_party_of_four_is_offered_the_cabs_that_seat_them(self):
        """End to end, on the real payload: seats-≥4, smallest suitable first."""
        from app.integrations.himcab.fare_service import FareService
        from app.integrations.himcab.vehicles import select_cabs_for_group

        parsed = FareService._extract_categories(REAL_ESTIMATE_RESPONSE["data"])
        chosen = [c.category_name for c in select_cabs_for_group(parsed, 4)]

        # Comfort(4), Standard(5), Premium(6) — the three smallest that still seat four.
        assert chosen == ["Comfort", "Standard", "Premium"]

    def test_one_malformed_row_does_not_lose_the_others(self):
        """A row with an unparseable fare is skipped, not raised — the other cabs survive,
        instead of one bad category costing the rider every option."""
        from app.integrations.himcab.fare_service import FareService

        rows = [
            {"categoryId": 1, "name": "Broken", "totalFare": "not a number",
             "seatCapacity": "4 - 5"},
            {"categoryId": 2, "name": "Standard", "totalFare": 4675.13, "seatCapacity": "4 - 5"},
        ]
        parsed = FareService._extract_categories(rows)

        assert [c.category_name for c in parsed] == ["Standard"]
