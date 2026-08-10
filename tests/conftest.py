"""
tests/conftest.py
──────────────────
Shared test fixtures for the Him Cab AI Travel Assistant.
Provides mock services, fake states, and test helpers.

`no_network` below is autouse: geography is resolved live now, so without it every test
that touches a journey would call a geocoder and a routing engine over the internet.
That would make the suite slow, flaky, dependent on somebody else's uptime, and — since
the providers rate-limit — occasionally wrong for reasons that have nothing to do with
the code under test.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import AgentState
from app.models.session import ExtractedEntities, Language


# ── No test may reach the network ─────────────────────────────────────────────

#: Places the suite refers to, with real coordinates and their real state. This is test
#: FIXTURE data, not application data: production geocodes every name it is given.
_KNOWN_PLACES: dict[str, tuple[float, float, str]] = {
    "shimla": (31.1048, 77.1734, "Himachal Pradesh"),
    "manali": (32.2396, 77.1887, "Himachal Pradesh"),
    "kullu": (31.9579, 77.1095, "Himachal Pradesh"),
    "kasol": (32.0109, 77.3160, "Himachal Pradesh"),
    "kufri": (31.0985, 77.2675, "Himachal Pradesh"),
    "dharamshala": (32.2190, 76.3234, "Himachal Pradesh"),
    "solang valley": (32.3200, 77.1700, "Himachal Pradesh"),
    "solan": (30.9045, 77.0967, "Himachal Pradesh"),
    "keylong": (32.5710, 77.0300, "Himachal Pradesh"),
    "parwanoo": (30.8370, 76.9610, "Himachal Pradesh"),
    "delhi": (28.6139, 77.2090, "Delhi"),
    "new delhi": (28.6139, 77.2090, "Delhi"),
    "gurgaon": (28.4595, 77.0266, "Haryana"),
    "chandigarh": (30.7333, 76.7794, "Chandigarh"),
    "ambala": (30.3782, 76.7767, "Haryana"),
    "kalka": (30.8380, 76.9370, "Haryana"),
    "mumbai": (19.0760, 72.8777, "Maharashtra"),
}


def _row(name: str) -> dict:
    lat, lon, state = _KNOWN_PLACES[name]
    return {
        "lat": str(lat),
        "lon": str(lon),
        "name": name.title(),
        "display_name": f"{name.title()}, {state}, India",
        "address": {"city": name.title(), "state": state, "country": "India"},
    }


def _match(query: str) -> str | None:
    """The known place a query names, or None. Longest match wins ("new delhi")."""
    text = (query or "").lower()
    hits = [name for name in _KNOWN_PLACES if name in text]
    return max(hits, key=len) if hits else None


def _nearest(lat: float, lon: float) -> str | None:
    """The closest known place to a fix, standing in for reverse geocoding."""
    best, best_d = None, float("inf")
    for name, (plat, plon, _state) in _KNOWN_PLACES.items():
        d = (plat - lat) ** 2 + (plon - lon) ** 2
        if d < best_d:
            best, best_d = name, d
    return best


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Stub the geocoding and routing providers at their HTTP boundary.

    Everything above the boundary still runs — URL building, response parsing, the
    caches, the rate limiter — so these tests exercise the real code paths. Only the
    socket is replaced.

    The route is derived from the straight-line distance so that different journeys
    give different distances; tests that care about an exact figure stub the engine
    themselves.
    """
    from app.services.external import geocoding as geocoding_module
    from app.services.external import routing as routing_module

    geocoder = geocoding_module.GeocodingService()
    router = routing_module.RoutingService()

    async def fake_geocode(path, params):
        if path == "/search":
            name = _match(str(params.get("q", "")))
            return [_row(name)] if name else []
        name = _nearest(float(params["lat"]), float(params["lon"]))
        return _row(name) if name else None

    async def fake_route(origin, destination):
        from app.services.geo import haversine_km

        km = max(1.0, haversine_km(origin, destination) * 1.6)
        return {
            "code": "Ok",
            "routes": [{"distance": km * 1000, "duration": (km / 45) * 3600}],
        }

    geocoder._get = AsyncMock(side_effect=fake_geocode)
    geocoder._throttle = AsyncMock()          # real limiter, but never actually sleep
    router._fetch_route = AsyncMock(side_effect=fake_route)

    monkeypatch.setattr(geocoding_module, "get_geocoding_service", lambda: geocoder)
    monkeypatch.setattr(routing_module, "get_routing_service", lambda: router)
    monkeypatch.setattr("app.services.geo.get_geocoding_service", lambda: geocoder)
    monkeypatch.setattr("app.services.geo.get_routing_service", lambda: router)

    # The offers backend is a real host too — a test must never call it.
    from app.services.himcab import offers_service as offers_module

    offers = offers_module.OffersService()
    offers._get = AsyncMock(return_value={"data": []})
    monkeypatch.setattr(offers_module, "get_offers_service", lambda: offers)
    monkeypatch.setattr(
        "app.agent.nodes.tool_executor.get_offers_service", lambda: offers, raising=False
    )

    yield {"geocoder": geocoder, "router": router, "offers": offers}


# ── Agent State Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def base_state():
    """Minimal agent state for testing nodes."""
    return AgentState(
        session_id="test-session-001",
        user_id="test-user-001",
        messages=[HumanMessage(content="I want to visit Manali for 5 days")],
    )


@pytest.fixture
def plan_trip_state():
    """Agent state with full entities extracted (ready for tools)."""
    return AgentState(
        session_id="test-session-002",
        user_id="test-user-001",
        messages=[
            HumanMessage(content="Plan a 3-day trip to Dharamshala for my family from Delhi"),
            AIMessage(content="Sure! Let me plan that for you."),
        ],
        intent="plan_trip",
        entities=ExtractedEntities(
            destination="Dharamshala",
            origin="Delhi",
            duration_days=3,
            group_type="family",
            group_size=4,
            budget_range="moderate",
            interests=["temples", "local food"],
        ),
        language=Language.ENGLISH,
    )


@pytest.fixture
def places_query_state():
    """Agent state for places query."""
    return AgentState(
        session_id="test-session-003",
        user_id="test-user-001",
        messages=[HumanMessage(content="What are the best places to visit in Shimla?")],
        intent="places_query",
        entities=ExtractedEntities(destination="Shimla"),
    )


# ── Mock Service Fixtures ────────────────────────────────────────────────────

@pytest.fixture
def mock_firestore():
    """Mock Firestore services."""
    with patch("app.services.firestore.get_session_service") as session_mock, \
         patch("app.services.firestore.get_user_profile_service") as profile_mock:
        session_svc = AsyncMock()
        profile_svc = AsyncMock()
        session_mock.return_value = session_svc
        profile_mock.return_value = profile_svc
        yield {"session": session_svc, "profile": profile_svc}


# ── Sample Data Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def sample_weather_data():
    """Sample weather data as stored in agent state."""
    return {
        "destination": "dharamshala",
        "current_temp_c": 22.5,
        "feels_like_c": 21.0,
        "humidity_percent": 65,
        "wind_speed_kmh": 12.3,
        "condition": "partly cloudy",
        "forecast_7day": [
            {"date": "2026-06-24", "min_c": 18.0, "max_c": 26.0, "condition": "clear sky", "rain_mm": 0, "snow_mm": 0},
            {"date": "2026-06-25", "min_c": 19.0, "max_c": 27.0, "condition": "few clouds", "rain_mm": 0, "snow_mm": 0},
        ],
        "alerts": [],
    }
