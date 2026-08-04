"""
tests/conftest.py
──────────────────
Shared test fixtures for the Him Cab AI Travel Assistant.
Provides mock services, fake states, and test helpers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import AgentState
from app.models.session import ExtractedEntities, Language


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
def mock_vertex_ai():
    """Mock the Vertex AI service to avoid real API calls."""
    with patch("app.services.vertex_ai.get_vertex_ai_service") as mock:
        service = MagicMock()
        mock.return_value = service
        yield service


@pytest.fixture
def mock_redis():
    """Mock Redis cache service."""
    with patch("app.services.redis_cache.get_redis_cache") as mock:
        cache = AsyncMock()
        cache.get.return_value = None
        cache.set.return_value = True
        cache.get_places.return_value = None
        cache.set_places.return_value = True
        cache.get_weather.return_value = None
        cache.set_weather.return_value = True
        cache.get_crowd_data.return_value = None
        cache.get_himachal_alerts.return_value = None
        cache.ping.return_value = True
        cache.check_rate_limit.return_value = (True, 1)
        cache.invalidate_session.return_value = None
        mock.return_value = cache
        yield cache


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
def sample_places_data():
    """Sample places data as returned by the Overpass service."""
    return {
        "hotels": [
            {"name": "Hotel Mountain View", "address": "Mall Road", "rating": None, "price_level": "3", "maps_url": "https://osm.org/node/123", "types": ["hotel"], "source": "OpenStreetMap"},
            {"name": "The Himalayan Retreat", "address": "Jogiwara Road", "rating": None, "price_level": "4", "maps_url": "https://osm.org/node/456", "types": ["hotel"], "source": "OpenStreetMap"},
        ],
        "restaurants": [
            {"name": "Tibet Kitchen", "address": "Jogiwara Road", "rating": None, "price_level": "unknown", "maps_url": "https://osm.org/node/789", "types": ["restaurant"], "source": "OpenStreetMap"},
        ],
        "attractions": [
            {"name": "Dalai Lama Temple Complex", "address": "McLeod Ganj", "rating": None, "price_level": "unknown", "maps_url": "https://osm.org/node/321", "types": ["attraction"], "source": "OpenStreetMap"},
        ],
    }


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
