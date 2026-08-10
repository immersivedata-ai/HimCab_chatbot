"""
tests/unit/test_geo_coordinates.py
───────────────────────────────────
The GPS fix → "local or visitor, and how far is the trip" pipeline.

This is the load-bearing test for coordinate routing: everything downstream — which
assistant answers, which language it answers in, what the fare is priced on — hangs off
`geo.locate()` getting the side of the state line right. So the first class below walks
every town in the geography tables and every gateway city riders travel from, rather
than spot-checking three of them.
"""

import pytest

from app.models.location import RiderLocation
from app.services import geo
from app.services.himcab.distances import resolve_route, resolve_route_from_fix


# ── Is this fix inside Himachal? ──────────────────────────────────────────────

class TestHimachalBoundary:
    """`geo.point_in_himachal` is what decides local vs outstation."""

    @pytest.mark.parametrize("name", sorted(geo.HIMACHAL_COORDS))
    def test_every_himachal_town_is_inside(self, name):
        """A town filed as Himachal must classify as Himachal from its own coordinates.

        If these two ever disagree, a rider standing in Kasol is told they are visiting.
        """
        lat, lon = geo.HIMACHAL_COORDS[name]
        assert geo.point_in_himachal(lat, lon), f"{name} classified as outside HP"

    @pytest.mark.parametrize("name", sorted(geo.GATEWAY_COORDS))
    def test_every_gateway_city_is_outside(self, name):
        """The cities riders travel FROM are, by definition, not in Himachal."""
        lat, lon = geo.GATEWAY_COORDS[name]
        assert not geo.point_in_himachal(lat, lon), f"{name} classified as inside HP"

    @pytest.mark.parametrize(
        "label,lat,lon,expected",
        [
            # Neighbours that hug the border — the cases a bounding box gets wrong.
            ("panchkula (haryana)", 30.6942, 76.8606, False),
            ("yamunanagar (haryana)", 30.1290, 77.2870, False),
            ("saharanpur (up)", 29.9640, 77.5460, False),
            ("hoshiarpur (punjab)", 31.5320, 75.9110, False),
            ("uttarkashi (uttarakhand)", 30.7300, 78.4400, False),
            ("leh (ladakh)", 34.1526, 77.5771, False),
            # Himachal's own extremities.
            ("nalagarh", 31.0450, 76.7200, True),
            ("pong dam", 31.9700, 76.0800, True),
            ("sarahan", 31.5100, 77.7930, True),
        ],
    )
    def test_border_neighbours(self, label, lat, lon, expected):
        assert geo.point_in_himachal(lat, lon) is expected, label

    def test_border_proximity_is_reported(self):
        """Parwanoo sits ~2 km from Haryana, so the call is flagged as a close one.

        The outline cannot separate Parwanoo from Kalka and does not pretend to — it
        says so instead, and the chat API treats a near-border persona as a default the
        rider may correct rather than a fact.
        """
        parwanoo = geo.locate(30.8370, 76.9610)
        assert parwanoo.near_border is True

        shimla = geo.locate(31.1048, 77.1734)
        assert shimla.near_border is False


# ── Turning a fix into a place ────────────────────────────────────────────────

class TestNearestCity:
    def test_exact_town_resolves_to_itself(self):
        assert geo.nearest_city(32.2396, 77.1887) == ("manali", 0.0)

    def test_between_towns_resolves_to_the_closer_one(self):
        """Solang Valley is 14 km up the road from Manali and is its own entry."""
        name, km = geo.nearest_city(32.3200, 77.1700)
        assert name == "solang valley"
        assert km < 1

    def test_middle_of_nowhere_resolves_to_nothing(self):
        """Naming a town 200 km away would put a wrong origin into the fare request."""
        assert geo.nearest_city(20.0, 70.0) is None


class TestLocate:
    def test_local_fix(self):
        fix = geo.locate(31.1048, 77.1734)
        assert fix.in_himachal is True
        assert fix.visitor_type == "local"
        assert fix.place_name == "Shimla"
        assert fix.is_precise is True

    def test_visitor_fix(self):
        fix = geo.locate(28.6139, 77.2090)
        assert fix.in_himachal is False
        assert fix.visitor_type == "outstation"
        assert fix.place_name == "Delhi"

    @pytest.mark.parametrize(
        "lat,lon",
        [
            (0.0, 0.0),        # what a phone sends before it has a fix
            (0.0001, -0.0001),  # ...and the same thing with float noise
            (91.0, 77.0),       # out of range
            (31.0, 181.0),
            (None, 77.0),
            (31.0, None),
        ],
    )
    def test_unusable_coordinates_resolve_to_nothing(self, lat, lon):
        """A bad fix must produce None, never a confident wrong answer.

        (0, 0) is the one that matters: it is in the Atlantic, so treating it as real
        would classify a Shimla rider as a visitor 6,000 km from Himachal.
        """
        assert geo.valid_coordinates(lat, lon) is False
        assert RiderLocation.resolve(lat, lon) is None


class TestRiderLocationModel:
    def test_resolve_populates_the_persona_inputs(self):
        location = RiderLocation.resolve(32.2396, 77.1887, accuracy_m=12)
        assert location.in_himachal is True
        assert location.visitor_type == "local"
        assert location.place_name == "Manali"
        assert location.accuracy_m == 12

    def test_round_trips_through_json(self):
        """It is persisted on the session, so it has to survive Firestore."""
        original = RiderLocation.resolve(31.1048, 77.1734)
        restored = RiderLocation.model_validate(original.model_dump(mode="json"))
        assert restored.place_name == original.place_name
        assert restored.in_himachal == original.in_himachal


# ── Distance, measured from where the rider actually is ───────────────────────

class TestRouteFromFix:
    def test_a_fix_on_a_surveyed_town_keeps_the_measured_distance(self):
        """Measured beats computed. A rider in Shimla asking about Manali gets the
        270 km somebody actually drove, not 268-point-something of trigonometry."""
        route = resolve_route_from_fix(geo.locate(31.1048, 77.1734), "Manali")
        assert route.km == resolve_route("Shimla", "Manali").km
        assert route.is_approximate is False
        assert route.source == "surveyed"

    def test_a_fix_between_towns_is_measured_from_the_fix(self):
        """Solang Valley → Kasol has no surveyed row, so the rider's own position is
        what the distance is computed from, and it is labelled approximate."""
        route = resolve_route_from_fix(geo.locate(32.3200, 77.1700), "Kasol")
        assert route is not None
        assert route.source == "gps"
        assert route.is_approximate is True
        assert route.km > 0

    def test_a_fix_up_the_valley_is_priced_from_the_town_it_is_actually_near(self):
        """A rider 11 km outside Kullu is a Kullu pickup, not a Manali one.

        Both towns have surveyed routes to Shimla and they differ by 70 km, so reading
        the fix as the wrong one overcharges the trip by roughly a third.
        """
        route = resolve_route_from_fix(geo.locate(32.05, 77.15), "Shimla")
        assert route.km == resolve_route("Kullu", "Shimla").km
        assert route.km != resolve_route("Manali", "Shimla").km

    def test_a_fix_far_from_every_town_is_measured_from_the_fix(self):
        """26 km from Mandi and further from everything else — no surveyed row applies,
        so the distance comes from the coordinates and is labelled approximate."""
        route = resolve_route_from_fix(geo.locate(31.50, 77.05), "Shimla")
        assert route.source == "gps"
        assert route.is_approximate is True
        assert route.km != resolve_route("Mandi", "Shimla").km

    def test_unknown_destination_yields_nothing(self):
        """No route rather than a guessed one — the fare endpoint needs a real distance."""
        assert resolve_route_from_fix(geo.locate(31.1048, 77.1734), "Atlantis") is None

    def test_no_fix_falls_back_to_the_named_origin(self):
        route = resolve_route_from_fix(None, "Manali", fallback_origin="Delhi")
        assert route.km == resolve_route("Delhi", "Manali").km

    def test_distance_from_point_matches_the_city_estimate_at_the_city(self):
        """Standing in the middle of Delhi, the GPS distance and the city-to-city
        distance must agree — otherwise the two paths quote different fares for the
        same journey depending on whether the app sent coordinates."""
        from_point = geo.road_distance_from_point(28.6139, 77.2090, "Kasol")
        from_names = geo.estimate_road_distance_km("Delhi", "Kasol")
        assert from_point == pytest.approx(from_names, rel=0.001)
