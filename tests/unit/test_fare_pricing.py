"""
tests/unit/test_fare_pricing.py
───────────────────────────────
Regression tests for the Him Cab fare reaching the rider.

The reported failure: itineraries arrived with no fare at all. The cause was not the
backend — it was the gate in front of it. `/fare/estimate` REQUIRES `distanceInMeters`,
and the distance came from a hand-listed table of 80 city PAIRS. Any route not in that
table produced no distance, so no call was made and no price was shown. Riders from
Gurgaon, Ambala, Mandi, Solan and most of India hit that path.

Distance is now computed from a table of city COORDINATES when a surveyed pair isn't
available, so any two recognised places can be priced. Tests here lock down:

  1. a route outside the surveyed table still yields a distance (the actual bug)
  2. surveyed distances still win, because they are measured rather than derived
  3. an estimated fare is labelled approximate all the way to the prompt
  4. trip type follows the JOURNEY, not the rider's persona
"""

import pytest

from app.agent.state import AgentState
from app.models.session import ExtractedEntities
from app.services import geo
from app.services.himcab.distances import ROAD_ROUTES, resolve_route
from app.services.himcab.quote_builder import build_fare_quote_input, classify_trip_type


# ── 1. The bug: unlisted routes could not be priced ───────────────────────────

class TestEveryRecognisedRouteCanBePriced:

    @pytest.mark.parametrize(
        "origin,destination",
        [
            ("Gurgaon", "Manali"),      # NCR suburb — never in the pair table
            ("Noida", "Shimla"),
            ("Ambala", "Shimla"),       # the obvious road approach to Shimla
            ("Jammu", "Dalhousie"),
            ("Mandi", "Manali"),        # HP district town, HP destination
            ("Solan", "Shimla"),
            ("Una", "Dharamshala"),
            ("Lucknow", "Manali"),
            ("Shimla", "Chamba"),
        ],
    )
    def test_a_route_outside_the_surveyed_table_still_gets_a_distance(self, origin, destination):
        resolved = resolve_route(origin, destination)

        assert resolved is not None, f"{origin}->{destination} would ship with no fare"
        assert resolved.km > 0
        assert resolved.is_approximate, "not surveyed, so it must be flagged as an estimate"

    def test_a_surveyed_route_uses_the_measured_distance(self):
        """Estimates fill gaps; they never override a real survey."""
        resolved = resolve_route("Delhi", "Manali")

        assert resolved.km == ROAD_ROUTES[("delhi", "manali")][0]
        assert resolved.is_approximate is False

    def test_the_surveyed_table_is_bidirectional(self):
        assert resolve_route("Manali", "Delhi").km == resolve_route("Delhi", "Manali").km

    def test_an_unrecognised_place_is_still_refused(self):
        """Better no price than a price for the wrong route."""
        assert resolve_route("Atlantis", "Manali") is None
        assert resolve_route("Manali", "") is None

    def test_an_estimate_is_never_shorter_than_the_straight_line(self):
        """A road cannot be shorter than the crow's flight — a factor below 1 would
        under-charge every unlisted route."""
        for origin, destination in [("Gurgaon", "Manali"), ("Mandi", "Kasol"),
                                    ("Mumbai", "Shimla"), ("Solan", "Kufri")]:
            a, b = geo.find_city(origin), geo.find_city(destination)
            straight_line = geo.haversine_km(a[1], b[1])
            assert resolve_route(origin, destination).km >= straight_line

    def test_the_estimator_tracks_the_surveyed_distances(self):
        """Sanity bound on the calibration: half the surveyed routes within 20%.

        This is not precision engineering — the fare is presented as approximate. It
        exists so a bad edit to the winding factors fails here rather than in a quote.
        """
        errors = []
        for (origin, destination), (surveyed_km, _minutes) in ROAD_ROUTES.items():
            estimated = geo.estimate_road_distance_km(origin, destination)
            if estimated:
                errors.append(abs(estimated - surveyed_km) / surveyed_km)
        within_20pct = sum(1 for e in errors if e <= 0.20)

        assert len(errors) > 50, "coordinates missing for surveyed routes"
        assert within_20pct / len(errors) >= 0.5


# ── 2. Trip type follows the journey ──────────────────────────────────────────

class TestTripTypeDescribesTheJourney:
    """It used to be `"LOCAL" if visitor_type == "local"`, which answers a different
    question: a Himachal resident driving Shimla → Manali is a 270 km intercity
    transfer, and the LOCAL rate card prices something else entirely."""

    @pytest.mark.parametrize(
        "distance_km,expected",
        [(8, "LOCAL"), (16, "LOCAL"), (40, "LOCAL"),
         (41, "OUTSTATION"), (270, "OUTSTATION"), (570, "OUTSTATION"), (None, "OUTSTATION")],
    )
    def test_classification(self, distance_km, expected):
        assert classify_trip_type(distance_km) == expected

    def test_a_resident_taking_a_long_trip_is_priced_as_outstation(self):
        entities = ExtractedEntities(
            origin="Shimla", home_city="Shimla", destination="Manali", visitor_type="local"
        )
        assert build_fare_quote_input(entities).trip_type == "OUTSTATION"

    def test_a_short_hop_is_priced_as_local(self):
        entities = ExtractedEntities(
            origin="Shimla", destination="Kufri", visitor_type="local"
        )
        quote_input = build_fare_quote_input(entities)

        assert quote_input.trip_type == "LOCAL"
        assert quote_input.distance_km <= 40

    def test_a_local_who_only_gave_a_home_city_is_still_priced(self):
        """Locals often answer with where they live, not an 'origin'."""
        entities = ExtractedEntities(home_city="Kullu", destination="Manali")
        quote_input = build_fare_quote_input(entities)

        assert quote_input.origin == "Kullu"
        assert quote_input.distance_km is not None


# ── 3. The rider is told when a fare is an estimate ───────────────────────────

class TestApproximateFaresAreLabelled:

    def _prompt(self, cab_fare: dict | None) -> str:
        from app.agent.nodes.writer import _build_system_prompt

        return _build_system_prompt(AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="outstation",
            entities=ExtractedEntities(destination="Manali", origin="Gurgaon", duration_days=3),
            cab_fare=cab_fare,
        ))

    OPTIONS = [
        {"category_name": "Economy", "total_fare_inr": 8400.0, "seat_capacity": 4},
        {"category_name": "Standard", "total_fare_inr": 9600.0, "seat_capacity": 4},
        {"category_name": "Premium", "total_fare_inr": 14200.0, "seat_capacity": 6},
    ]

    def test_an_estimated_fare_is_marked_approximate(self):
        prompt = self._prompt({
            "fare_inr": 9600, "category_name": "Standard", "options": self.OPTIONS,
            "is_approximate": True, "distance_km": 521.6,
        })

        assert "APPROXIMATE" in prompt
        assert "approx" in prompt
        assert "confirmed at booking" in prompt
        assert "522 km" in prompt   # ~521.6 rounded
        for option in self.OPTIONS:
            assert f"{round(option['total_fare_inr']):,}" in prompt

    def test_a_surveyed_fare_is_presented_as_a_real_price(self):
        prompt = self._prompt({
            "fare_inr": 9600, "category_name": "Standard", "options": self.OPTIONS,
            "is_approximate": False, "distance_km": 570,
        })

        assert "real prices" in prompt
        assert "APPROXIMATE" not in prompt

    def test_every_category_reaches_the_prompt_with_the_recommendation_flagged(self):
        prompt = self._prompt({
            "fare_inr": 9600, "category_name": "Standard", "options": self.OPTIONS,
            "is_approximate": True, "distance_km": 521.6,
        })

        assert "Economy" in prompt and "Standard" in prompt and "Premium" in prompt
        assert "recommended for this group/budget" in prompt
        assert "List EVERY cab category" in prompt

    def test_with_no_fare_the_model_is_told_not_to_invent_one(self):
        prompt = self._prompt(None)

        assert "do NOT state, estimate or imply any price" in prompt
        # …but it must still pitch Him Cab rather than going silent on booking.
        assert "Book Your Him Cab" in prompt

    def test_the_him_cab_section_is_never_optional(self):
        for cab_fare in (None, {"fare_inr": 9600, "category_name": "Standard",
                                "options": self.OPTIONS, "is_approximate": True}):
            assert "This section is REQUIRED" in self._prompt(cab_fare)


# ── 4. Only cabs that fit the group are offered ───────────────────────────────

class TestCabsAreNarrowedToTheGroup:
    """The reported bug: a party of four was recommended a Mini Bus.

    `/fare/estimate` prices EVERY category in one call, and the reply repeated the
    list verbatim — which is not the same as recommending something.
    """

    CATEGORIES = [
        ("Economy", 1800), ("Sedan", 2200), ("SUV", 3100),
        ("Innova Crysta", 3900), ("Tempo Traveller", 6500), ("Mini Bus", 9800),
    ]

    def _options(self):
        from app.services.himcab.models import FareCategoryFare

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
        from app.services.himcab.vehicles import select_cabs_for_group

        chosen = [c.category_name for c in select_cabs_for_group(self._options(), group_size)]
        assert chosen == expected

    def test_a_family_of_four_is_never_shown_a_minibus(self):
        """The literal reported symptom."""
        from app.services.himcab.vehicles import select_cabs_for_group

        chosen = [c.category_name for c in select_cabs_for_group(self._options(), 4)]
        assert not any("Bus" in name for name in chosen)

    def test_the_smallest_suitable_cab_leads(self):
        """A family of four wants a sedan, not an empty minibus at three times the price."""
        from app.services.himcab.vehicles import select_cabs_for_group

        assert select_cabs_for_group(self._options(), 5)[0].category_name == "SUV"

    def test_a_group_too_big_for_any_cab_still_sees_the_largest(self):
        """They need to know two vehicles are required, not that Him Cab has nothing."""
        from app.services.himcab.vehicles import select_cabs_for_group

        chosen = select_cabs_for_group(self._options(), 30)
        assert chosen and chosen[0].category_name == "Mini Bus"

    def test_with_no_group_size_nothing_is_ruled_out(self):
        """Narrowing on a guess would hide a cab the rider wanted."""
        from app.services.himcab.vehicles import select_cabs_for_group

        assert len(select_cabs_for_group(self._options(), None)) == 3

    def test_the_backends_own_seat_count_wins_over_the_name(self):
        from app.services.himcab.models import FareCategoryFare
        from app.services.himcab.vehicles import effective_seats

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
        from app.services.himcab.vehicles import seat_capacity_for

        assert seat_capacity_for(name) == seats

    def test_an_unrecognised_category_is_kept_rather_than_filtered_out(self):
        """Hiding a cab the rider could have taken is worse than showing one extra."""
        from app.services.himcab.models import FareCategoryFare
        from app.services.himcab.vehicles import select_cabs_for_group

        options = [FareCategoryFare(category_name="Mystery Wagon", total_fare_inr=2000)]
        assert select_cabs_for_group(options, 6)


# ── 5. Journey distance and drive time ────────────────────────────────────────

class TestJourneyDuration:

    def test_a_surveyed_route_uses_its_measured_drive_time(self):
        route = resolve_route("Shimla", "Manali")
        assert route.duration_minutes == ROAD_ROUTES[("shimla", "manali")][1]
        assert route.duration_text == "4h 30m"

    def test_an_unsurveyed_route_gets_an_estimated_drive_time(self):
        route = resolve_route("Solan", "Kufri")
        assert route.is_approximate
        assert route.duration_minutes > 0

    @pytest.mark.parametrize(
        "minutes,text", [(30, "30m"), (60, "1h"), (270, "4h 30m"), (90, "1h 30m")]
    )
    def test_drive_time_reads_the_way_a_rider_would_say_it(self, minutes, text):
        from app.services.himcab.distances import RouteEstimate

        assert RouteEstimate(km=1, duration_minutes=minutes, is_approximate=False).duration_text == text

    def test_estimated_drive_times_track_the_surveyed_ones(self):
        """Hill roads are slow; a single average speed would badly under-state a short
        hop, which is exactly the trip a Himachal local asks about."""
        import statistics

        errors = []
        for (origin, destination), (km, surveyed_minutes) in ROAD_ROUTES.items():
            estimated = geo.estimate_drive_time_minutes(km)
            errors.append(abs(estimated - surveyed_minutes) / surveyed_minutes)

        assert statistics.median(errors) <= 0.30


# ── 6. Geography is single-sourced ────────────────────────────────────────────

class TestGeography:

    @pytest.mark.parametrize(
        "typed,canonical",
        [("Manali", "manali"), ("manali ", "manali"), ("McLeodGanj", "mcleod ganj"),
         ("Dharamsala", "dharamshala"), ("Bombay", "mumbai"), ("Delhi NCR", "delhi"),
         ("Shimla, HP", "shimla"), ("Gurugram", "gurgaon")],
    )
    def test_places_riders_actually_type_resolve(self, typed, canonical):
        found = geo.find_city(typed)
        assert found is not None and found[0] == canonical

    def test_an_unknown_place_resolves_to_nothing_rather_than_the_nearest_guess(self):
        assert geo.find_city("Narnia") is None
        assert geo.is_in_himachal("Narnia") is None

    @pytest.mark.parametrize(
        "place,in_hp",
        [("Manali", True), ("Solan", True), ("Keylong", True),
         ("Delhi", False), ("Chandigarh", False), ("Dehradun", False)],
    )
    def test_himachal_membership(self, place, in_hp):
        assert geo.is_in_himachal(place) is in_hp

    def test_the_planner_and_the_fare_share_one_geography(self):
        """A town known to one and unknown to the other is how a resident ends up in
        the outstation flow, or a priceable route has no weather."""
        from app.agent.nodes.planner import classify_visitor_type

        for town in ("Keylong", "Solan", "Nahan", "Hamirpur", "Kaza"):
            assert classify_visitor_type(town) == "local"
            assert geo.find_city(town) is not None
        for city in ("Delhi", "Ambala", "Jammu"):
            assert classify_visitor_type(city) == "outstation"

    def test_every_surveyed_route_has_coordinates(self):
        """Otherwise that route silently loses its estimate fallback."""
        missing = {
            place
            for pair in ROAD_ROUTES
            for place in pair
            if geo.find_city(place) is None
        }
        assert not missing, f"add to geo.CITY_COORDS: {sorted(missing)}"
