"""
tests/unit/test_geo_live.py
───────────────────────────
Geography, now that none of it is stored.

Coordinates, the state boundary, road distances and drive times all used to be tables
in this repo. They are now looked up per request, so these tests stub the HTTP boundary
of the two providers and assert on the composition above it:

  • a GPS fix becomes a persona and a pickup name        (`geo.locate`)
  • a name becomes "is this rider a resident?"           (`geo.place_in_himachal`)
  • a fix plus a destination becomes one priced journey  (`geo.resolve_journey`)

What is deliberately NOT tested: whether Nominatim knows where Manali is. That is the
provider's job, and asserting it here would be testing the internet. What matters is
that a provider failure returns None, because None is what keeps the assistant asking
the rider instead of pricing a guess.
"""

from unittest.mock import AsyncMock, patch

import pytest

from app.services import geo
from app.services.external.geocoding import GeocodingService, Place
from app.services.external.routing import RoadRoute, RoutingService


# ── Provider stubs ────────────────────────────────────────────────────────────
# Stubbed at the HTTP boundary (`_get`/`_fetch_route`), so URL building, response
# parsing, the cache and the rate limiter all still run.

def _nominatim_row(name, lat, lon, state="Himachal Pradesh", country="India"):
    return {
        "lat": str(lat),
        "lon": str(lon),
        "name": name,
        "display_name": f"{name}, {state}, {country}",
        "address": {"city": name, "state": state, "country": country},
    }


@pytest.fixture
def geocoder(monkeypatch):
    """A GeocodingService whose provider is a dict, installed as the singleton."""
    service = GeocodingService()
    service._last_request_at = 0.0
    responses: dict = {}

    async def fake_get(path, params):
        if path == "/search":
            return responses.get(("search", params["q"].lower()), [])
        return responses.get(("reverse", round(params["lat"], 3), round(params["lon"], 3)))

    service._get = AsyncMock(side_effect=fake_get)
    # Rate limiting is real but must not make the suite sleep.
    service._throttle = AsyncMock()
    monkeypatch.setattr(
        "app.services.external.geocoding.get_geocoding_service", lambda: service
    )
    monkeypatch.setattr("app.services.geo.get_geocoding_service", lambda: service)
    service.responses = responses          # type: ignore[attr-defined]
    return service


@pytest.fixture
def router(monkeypatch):
    """A RoutingService whose engine returns a fixed route, as the singleton."""
    service = RoutingService()
    service._fetch_route = AsyncMock(
        return_value={"code": "Ok", "routes": [{"distance": 270_000, "duration": 16_200}]}
    )
    monkeypatch.setattr("app.services.geo.get_routing_service", lambda: service)
    return service


# ── A GPS fix becomes a persona ───────────────────────────────────────────────

class TestLocate:

    @pytest.mark.asyncio
    async def test_a_fix_in_himachal_makes_a_local(self, geocoder):
        geocoder.responses[("reverse", 31.105, 77.173)] = _nominatim_row(
            "Shimla", 31.1048, 77.1734
        )

        fix = await geo.locate(31.1048, 77.1734)

        assert fix.in_himachal is True
        assert fix.visitor_type == "local"
        assert fix.place_name == "Shimla"
        assert fix.state == "Himachal Pradesh"
        assert fix.is_precise

    @pytest.mark.asyncio
    async def test_a_fix_outside_makes_a_visitor(self, geocoder):
        geocoder.responses[("reverse", 28.614, 77.209)] = _nominatim_row(
            "Delhi", 28.6139, 77.2090, state="Delhi"
        )

        fix = await geo.locate(28.6139, 77.2090)

        assert fix.in_himachal is False
        assert fix.visitor_type == "outstation"

    @pytest.mark.asyncio
    async def test_the_state_comes_from_the_provider_not_from_a_polygon(self, geocoder):
        """The old hand-drawn outline could not separate Parwanoo (HP) from Kalka
        (Haryana), 2 km apart. Published boundaries can."""
        geocoder.responses[("reverse", 30.837, 76.961)] = _nominatim_row(
            "Parwanoo", 30.8370, 76.9610, state="Himachal Pradesh"
        )
        geocoder.responses[("reverse", 30.838, 76.937)] = _nominatim_row(
            "Kalka", 30.8380, 76.9370, state="Haryana"
        )

        assert (await geo.locate(30.8370, 76.9610)).in_himachal is True
        assert (await geo.locate(30.8380, 76.9370)).in_himachal is False

    @pytest.mark.asyncio
    async def test_a_fix_far_from_the_named_area_is_not_precise(self, geocoder):
        """Reverse geocoding a hill road returns the enclosing district. Naming it as
        the rider's pickup would announce a town they have never been to."""
        # The fix is ~40 km from the centre of the area the provider named.
        geocoder.responses[("reverse", 32.32, 77.17)] = _nominatim_row(
            "Kullu", 31.9579, 77.1095
        )

        fix = await geo.locate(32.32, 77.17)

        assert fix.place_name == "Kullu"
        assert fix.is_precise is False

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (None, 5.0), (91.0, 5.0)])
    async def test_unusable_coordinates_resolve_to_nothing(self, geocoder, lat, lon):
        """(0, 0) is what a phone sends before it has a fix — treating it as real would
        classify a Shimla rider as a visitor in the Atlantic."""
        assert await geo.locate(lat, lon) is None

    @pytest.mark.asyncio
    async def test_a_provider_outage_resolves_to_nothing(self, geocoder):
        geocoder._get = AsyncMock(side_effect=RuntimeError("nominatim down"))

        assert await geo.locate(31.1048, 77.1734) is None


# ── A name becomes a classification ───────────────────────────────────────────

class TestPlaceInHimachal:

    @pytest.mark.asyncio
    async def test_a_himachal_town(self, geocoder):
        geocoder.responses[("search", "keylong")] = [_nominatim_row("Keylong", 32.571, 77.03)]
        assert await geo.place_in_himachal("Keylong") is True

    @pytest.mark.asyncio
    async def test_a_city_outside(self, geocoder):
        geocoder.responses[("search", "ambala")] = [
            _nominatim_row("Ambala", 30.378, 76.777, state="Haryana")
        ]
        assert await geo.place_in_himachal("Ambala") is False

    @pytest.mark.asyncio
    async def test_an_unplaceable_name_is_unknown_not_a_visitor(self, geocoder):
        """None means "we don't know" — the caller must keep the classification it had
        rather than deciding a resident is a tourist."""
        assert await geo.place_in_himachal("Narnia") is None


# ── A fix plus a destination becomes one priced journey ───────────────────────

class TestResolveJourney:

    @pytest.mark.asyncio
    async def test_a_named_journey_carries_both_pins_and_the_real_drive(self, geocoder, router):
        geocoder.responses[("search", "shimla")] = [_nominatim_row("Shimla", 31.1048, 77.1734)]
        geocoder.responses[("search", "manali")] = [_nominatim_row("Manali", 32.2396, 77.1887)]

        journey = await geo.resolve_journey("Manali", origin_name="Shimla")

        assert journey.origin.name == "Shimla"
        assert journey.destination.name == "Manali"
        # Straight from the engine: 270 km / 16,200 s.
        assert journey.distance_km == 270.0
        assert journey.duration_text == "4h 30m"

    @pytest.mark.asyncio
    async def test_a_gps_pickup_is_measured_from_the_fix_not_the_nearest_town(
        self, geocoder, router
    ):
        """A rider in Solang Valley is 14 km up the road from Manali. Pricing them as
        "Manali" silently drops that leg."""
        geocoder.responses[("search", "shimla")] = [_nominatim_row("Shimla", 31.1048, 77.1734)]
        fix = geo.LocationFix(
            lat=32.32, lon=77.17, in_himachal=True,
            nearest_city="Manali", nearest_city_km=9.0, state="Himachal Pradesh",
        )

        journey = await geo.resolve_journey("Shimla", origin_fix=fix)

        assert (journey.origin.lat, journey.origin.lon) == (32.32, 77.17)
        origin_arg = router._fetch_route.await_args.args[0]
        assert origin_arg == (32.32, 77.17), "the engine was given the town, not the fix"

    @pytest.mark.asyncio
    async def test_both_endpoints_are_geocoded_concurrently(self, geocoder, router):
        """Two independent lookups; a chat turn should not pay for them serially."""
        geocoder.responses[("search", "shimla")] = [_nominatim_row("Shimla", 31.1, 77.17)]
        geocoder.responses[("search", "manali")] = [_nominatim_row("Manali", 32.24, 77.19)]

        with patch("app.services.geo.asyncio.gather", wraps=geo.asyncio.gather) as gather:
            await geo.resolve_journey("Manali", origin_name="Shimla")

        assert gather.called

    @pytest.mark.asyncio
    async def test_an_unplaceable_destination_yields_no_journey(self, geocoder, router):
        geocoder.responses[("search", "shimla")] = [_nominatim_row("Shimla", 31.1, 77.17)]

        assert await geo.resolve_journey("Atlantis", origin_name="Shimla") is None

    @pytest.mark.asyncio
    async def test_no_road_between_the_points_yields_no_journey(self, geocoder, router):
        """No distance means the fare endpoint cannot be called — and the reply says the
        fare is confirmed at booking rather than inventing one."""
        geocoder.responses[("search", "shimla")] = [_nominatim_row("Shimla", 31.1, 77.17)]
        geocoder.responses[("search", "manali")] = [_nominatim_row("Manali", 32.24, 77.19)]
        router._fetch_route = AsyncMock(return_value={"code": "NoRoute", "routes": []})

        assert await geo.resolve_journey("Manali", origin_name="Shimla") is None

    @pytest.mark.asyncio
    async def test_no_origin_at_all_yields_no_journey(self, geocoder, router):
        assert await geo.resolve_journey("Manali") is None

    @pytest.mark.asyncio
    async def test_the_packaged_journey_carries_map_pins(self, geocoder, router):
        geocoder.responses[("search", "kullu")] = [_nominatim_row("Kullu", 31.9579, 77.1095)]
        geocoder.responses[("search", "kasol")] = [_nominatim_row("Kasol", 32.0109, 77.316)]

        packaged = (await geo.resolve_journey("Kasol", origin_name="Kullu")).as_dict()

        for end in ("origin", "destination"):
            assert packaged[end]["lat"] and packaged[end]["lon"], f"{end} has no pin"
        assert packaged["duration_text"] == "4h 30m"
        assert packaged["distance_km"] == 270.0


# ── Provider clients ──────────────────────────────────────────────────────────

class TestRoutingClient:

    @pytest.mark.parametrize(
        "minutes,text", [(30, "30m"), (60, "1h"), (270, "4h 30m"), (90, "1h 30m")]
    )
    def test_drive_time_reads_the_way_a_rider_would_say_it(self, minutes, text):
        assert RoadRoute(distance_km=1, duration_minutes=minutes).duration_text == text

    def test_drive_time_is_rounded_to_five_minutes(self):
        """The engine reports seconds. Quoting "4h 07m" implies it knows about today's
        traffic, and it does not."""
        parsed = RoutingService()._parse_route(
            {"code": "Ok", "routes": [{"distance": 100_000, "duration": 7_620}]}
        )
        assert parsed.duration_minutes % 5 == 0

    @pytest.mark.parametrize(
        "payload",
        [
            {"code": "NoRoute", "routes": []},
            {"code": "Ok", "routes": []},
            {"code": "Ok", "routes": [{"distance": 0, "duration": 0}]},
            {"code": "Ok", "routes": [{"duration": 100}]},
            "not json at all",
        ],
    )
    def test_an_unusable_engine_response_is_no_route(self, payload):
        assert RoutingService()._parse_route(payload) is None


class TestGeocodingClient:

    def test_a_place_name_is_taken_from_the_most_specific_field(self):
        """Reverse geocoding a hill road has no `city`, but usually a `village`."""
        place = GeocodingService()._to_place({
            "lat": "32.0", "lon": "77.0",
            "address": {"village": "Jibhi", "county": "Kullu", "state": "Himachal Pradesh"},
        })
        assert place.name == "Jibhi"

    def test_in_himachal_reads_the_administrative_state(self):
        assert Place(name="x", lat=1, lon=2, state="Himachal Pradesh").in_himachal
        assert not Place(name="x", lat=1, lon=2, state="Punjab").in_himachal
        assert not Place(name="x", lat=1, lon=2).in_himachal

    @pytest.mark.parametrize(
        "row", [{"lat": "no", "lon": "1"}, {"lon": "1"}, "junk", {"lat": "1", "lon": "2"}]
    )
    def test_an_unusable_row_is_no_place(self, row):
        # The last case has coordinates but nothing nameable.
        assert GeocodingService()._to_place(row) is None

    @pytest.mark.asyncio
    async def test_a_miss_is_cached_so_the_provider_is_not_asked_twice(self, geocoder):
        """Nominatim allows one request a second; re-asking for a name it has already
        refused is both slow and impolite."""
        assert await geocoder.geocode("Nowhereville") is None
        assert await geocoder.geocode("Nowhereville") is None

        assert geocoder._get.await_count == 1

    @pytest.mark.asyncio
    async def test_a_blank_query_never_reaches_the_provider(self, geocoder):
        assert await geocoder.geocode("   ") is None
        assert geocoder._get.await_count == 0
