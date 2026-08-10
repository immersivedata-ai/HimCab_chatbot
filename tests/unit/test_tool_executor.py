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
    async def test_fetches_weather_and_derives_snowfall_once(self, plan_trip_state):
        """Weather is fetched exactly ONCE and snowfall risk is derived from that same
        report (no second get_weather call). Also confirms weather + snowfall reach state."""
        mock_report = MagicMock()
        mock_report.model_dump.return_value = {"temp": 22}

        with patch("app.agent.nodes.tool_executor.get_weather_service") as weather_mock:
            weather_svc = MagicMock()
            weather_svc.get_weather = AsyncMock(return_value=mock_report)
            # snowfall_from_report is the pure derivation (no I/O) used instead of a 2nd fetch
            weather_svc.snowfall_from_report = MagicMock(
                return_value={"risk": "none", "expected_mm": 0}
            )
            weather_mock.return_value = weather_svc

            result = await tool_executor_node(plan_trip_state)

        assert "tool_executor" in result["node_trace"]
        assert result["weather_data"] is not None
        assert result["snowfall_data"] == {"risk": "none", "expected_mm": 0}
        # The dedup fix: get_weather called once, snowfall derived (not re-fetched).
        weather_svc.get_weather.assert_awaited_once()
        weather_svc.snowfall_from_report.assert_called_once_with(mock_report)

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_weather_failure(self, plan_trip_state):
        """A dead weather API costs detail in the reply, never the reply itself."""
        with patch("app.agent.nodes.tool_executor.get_weather_service") as weather_mock:
            weather_svc = MagicMock()
            weather_svc.get_weather = AsyncMock(side_effect=Exception("API down"))
            weather_svc.snowfall_from_report = MagicMock(return_value={"risk": "none"})
            weather_mock.return_value = weather_svc

            result = await tool_executor_node(plan_trip_state)

        assert result["weather_data"] is None
        assert result["snowfall_data"] is None
        assert "tool_executor" in result["node_trace"]
        # No derivation is attempted from a report that was never fetched.
        weather_svc.snowfall_from_report.assert_not_called()
