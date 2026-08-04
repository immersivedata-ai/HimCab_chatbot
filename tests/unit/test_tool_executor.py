"""
tests/unit/test_tool_executor.py
──────────────────────────────────
Unit tests for the Tool Executor node.
Tests parallel execution, graceful degradation, and timeout handling.
"""

import asyncio

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agent.nodes.tool_executor import tool_executor_node, _safe_call


class TestSafeCall:
    """Tests for the _safe_call wrapper function."""

    @pytest.mark.asyncio
    async def test_successful_async_call(self):
        """_safe_call should return the result of a successful async call."""
        async def mock_coro():
            return {"data": "test"}

        result = await _safe_call(mock_coro(), "test_tool")
        assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_handles_sync_value(self):
        """_safe_call should return sync values directly without awaiting."""
        result = await _safe_call({"data": "sync"}, "test_sync")
        assert result == {"data": "sync"}

    @pytest.mark.asyncio
    async def test_handles_none_value(self):
        """_safe_call should handle None sync values."""
        result = await _safe_call(None, "test_none")
        assert result is None

    @pytest.mark.asyncio
    async def test_catches_exception(self):
        """_safe_call should catch exceptions and return None."""
        async def failing_coro():
            raise ValueError("API Error")

        result = await _safe_call(failing_coro(), "failing_tool")
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """_safe_call should timeout slow coroutines and return None."""
        async def slow_coro():
            await asyncio.sleep(100)
            return "never reached"

        # Patch the timeout to be very short for testing
        with patch("app.agent.nodes.tool_executor.TOOL_TIMEOUT_SECONDS", 0.1):
            result = await _safe_call(slow_coro(), "slow_tool")
        assert result is None


class TestToolExecutorNode:
    """Tests for the tool_executor_node function."""

    @pytest.mark.asyncio
    async def test_no_destination_returns_early(self, base_state):
        """Should return early with just node_trace if no destination."""
        base_state.entities.destination = None
        result = await tool_executor_node(base_state)
        assert result == {"node_trace": ["tool_executor"]}

    @pytest.mark.asyncio
    async def test_full_plan_fetches_all_categories(self, plan_trip_state):
        """Full trip plan should fetch all 5 place categories + weather."""
        mock_weather = MagicMock()
        mock_weather.model_dump.return_value = {"temp": 22}

        with patch("app.agent.nodes.tool_executor.get_weather_service") as weather_mock, \
             patch("app.agent.nodes.tool_executor.get_places_service") as places_mock, \
             patch("app.agent.nodes.tool_executor.get_redis_cache") as cache_mock:

            weather_svc = AsyncMock()
            weather_svc.get_weather.return_value = mock_weather
            weather_svc.get_snowfall_risk.return_value = {"risk": "none", "expected_mm": 0}
            weather_mock.return_value = weather_svc

            places_svc = AsyncMock()
            places_svc.get_places.return_value = [{"name": "Test Place"}]
            places_svc.get_drive_time.return_value = {"duration_minutes": 540, "distance_km": 490}
            places_mock.return_value = places_svc

            cache = AsyncMock()
            cache.get_crowd_data.return_value = None
            cache_mock.return_value = cache

            result = await tool_executor_node(plan_trip_state)

        assert "tool_executor" in result["node_trace"]
        assert result["weather_data"] is not None
        assert result["snowfall_data"] is not None
        assert len(result["places_data"]) > 0
        assert result["drive_times"] is not None

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_weather_failure(self, plan_trip_state):
        """If weather fails, other tools should still work."""
        with patch("app.agent.nodes.tool_executor.get_weather_service") as weather_mock, \
             patch("app.agent.nodes.tool_executor.get_places_service") as places_mock, \
             patch("app.agent.nodes.tool_executor.get_redis_cache") as cache_mock:

            weather_svc = AsyncMock()
            weather_svc.get_weather.side_effect = Exception("API down")
            weather_svc.get_snowfall_risk.side_effect = Exception("API down")
            weather_mock.return_value = weather_svc

            places_svc = AsyncMock()
            places_svc.get_places.return_value = [{"name": "Temple"}]
            places_svc.get_drive_time.return_value = {"duration_minutes": 300}
            places_mock.return_value = places_svc

            cache = AsyncMock()
            cache.get_crowd_data.return_value = None
            cache_mock.return_value = cache

            result = await tool_executor_node(plan_trip_state)

        # Weather should be None but places should still have data
        assert result["weather_data"] is None
        assert len(result["places_data"]) > 0

    @pytest.mark.asyncio
    async def test_places_query_intent(self, places_query_state):
        """Places query intent should only fetch attractions and restaurants."""
        with patch("app.agent.nodes.tool_executor.get_weather_service") as weather_mock, \
             patch("app.agent.nodes.tool_executor.get_places_service") as places_mock, \
             patch("app.agent.nodes.tool_executor.get_redis_cache") as cache_mock:

            weather_svc = AsyncMock()
            mock_weather = MagicMock()
            mock_weather.model_dump.return_value = {}
            weather_svc.get_weather.return_value = mock_weather
            weather_svc.get_snowfall_risk.return_value = {"risk": "none"}
            weather_mock.return_value = weather_svc

            places_svc = AsyncMock()
            places_svc.get_places.return_value = [{"name": "Place"}]
            places_mock.return_value = places_svc

            cache = AsyncMock()
            cache.get_crowd_data.return_value = None
            cache_mock.return_value = cache

            result = await tool_executor_node(places_query_state)

        # Should have fetched attractions and restaurants (PARTIAL_CATEGORIES)
        call_categories = [
            call.args[1] for call in places_svc.get_places.call_args_list
        ]
        assert "attractions" in call_categories
        assert "restaurants" in call_categories
        # Should NOT have fetched hotels, temples, cafes
        assert "hotels" not in call_categories
