"""
tests/unit/test_tool_executor.py
──────────────────────────────────
Unit tests for the Tool Executor node.
Tests parallel execution, graceful degradation, and timeout handling.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agent.nodes.tool_executor import _safe_call, tool_executor_node
from app.services import geo


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


# ── The journey, measured from the rider's own position ───────────────────────

def _quiet_weather():
    """Patch context that keeps the weather call out of the way of a route assertion."""
    weather_svc = MagicMock()
    weather_svc.get_weather = AsyncMock(side_effect=Exception("not under test"))
    weather_svc.snowfall_from_report = MagicMock(return_value=None)
    return patch(
        "app.agent.nodes.tool_executor.get_weather_service", return_value=weather_svc
    )


async def _located_state(lat: float, lon: float, **entity_overrides):
    """Agent state for a rider at these coordinates.

    Async because resolving a fix is a live reverse-geocode now — there is no
    coordinate table to consult. `conftest.no_network` stubs the provider.
    """
    from app.agent.state import AgentState
    from app.domain.conversation import ExtractedEntities

    entities = {"destination": "Manali"}
    entities.update(entity_overrides)
    return AgentState(
        session_id="route-1",
        user_id="rider:1",
        intent="plan_trip",
        visitor_type="local",
        entities=ExtractedEntities(**entities),
        rider_location=await geo.resolve_rider_location(lat, lon),
    )


class TestRouteFromCoordinates:
    """`route_estimate` is what the reply quotes, what the fare is priced on and what
    the app pins on its map. All three have to describe the same journey."""

    @pytest.mark.asyncio
    async def test_a_fix_alone_is_enough_to_resolve_the_journey(self):
        """No origin was ever typed — the fix supplies it. Before coordinates, this
        rider got no distance and therefore no fare at all."""
        state = await _located_state(31.1048, 77.1734)   # Shimla

        with _quiet_weather():
            result = await tool_executor_node(state)

        route = result["route_estimate"]
        assert route is not None
        assert route["distance_km"] > 0
        assert route["destination"]["name"] == "Manali"

    @pytest.mark.asyncio
    async def test_the_pickup_pin_is_the_riders_own_fix(self):
        """Solang Valley is 14 km up the road from Manali. Handing the app the town
        centroid instead of the fix would drop the pin on the wrong village."""
        state = await _located_state(32.3200, 77.1700, destination="Kasol")

        with _quiet_weather():
            result = await tool_executor_node(state)

        origin = result["route_estimate"]["origin"]
        assert origin["lat"] == pytest.approx(32.3200)
        assert origin["lon"] == pytest.approx(77.1700)

    @pytest.mark.asyncio
    async def test_a_named_origin_elsewhere_beats_the_fix(self):
        """Their words win: someone arranging a Delhi pickup from their Shimla sofa
        must be quoted the Delhi journey, not the one starting under their feet."""
        state = await _located_state(31.1048, 77.1734, origin="Delhi")

        with _quiet_weather():
            result = await tool_executor_node(state)

        origin = result["route_estimate"]["origin"]
        assert origin["name"] == "Delhi"
        # Delhi's coordinates, not the Shimla fix the phone reported.
        assert origin["lat"] == pytest.approx(28.6139, abs=0.5)

    @pytest.mark.asyncio
    async def test_the_fare_is_priced_on_the_distance_the_reply_quotes(self):
        """The single most damaging inconsistency available here: quoting one distance
        in the text and pricing the cab on another."""
        from app.core.config import settings
        from app.integrations.himcab.models import FareCategoryFare, FareQuote

        state = await _located_state(32.3200, 77.1700, destination="Kasol")
        state.rider_jwt = "jwt-token"

        captured = {}

        async def fake_quote(quote_input, rider_jwt):
            captured["input"] = quote_input
            return FareQuote(
                base_fare_inr=1500,
                category_name="Sedan",
                trip_type=quote_input.trip_type,
                distance_km=quote_input.distance_km,
                alternatives=[FareCategoryFare(category_name="Sedan", total_fare_inr=1500)],
            )

        fare_svc = MagicMock()
        fare_svc.get_quote = fake_quote

        with _quiet_weather(), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare_svc), \
             patch.object(settings, "enable_offers", False):
            result = await tool_executor_node(state)

        assert captured["input"].distance_km == result["route_estimate"]["distance_km"]
        assert result["cab_fare"]["distance_km"] == result["route_estimate"]["distance_km"]
        # Journey context travels with the fare so a fare card needs no second lookup.
        assert result["cab_fare"]["duration_text"] == result["route_estimate"]["duration_text"]

    @pytest.mark.asyncio
    async def test_a_fix_alone_opens_the_fare_gate(self):
        """The gate used to demand a named origin, so a rider whose phone knew exactly
        where they were still saw no price until they typed the town out."""
        from app.core.config import settings

        state = await _located_state(31.1048, 77.1734)
        state.rider_jwt = "jwt-token"

        fare_svc = MagicMock()
        fare_svc.get_quote = AsyncMock(side_effect=Exception("backend down"))

        with _quiet_weather(), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare_svc), \
             patch.object(settings, "enable_offers", False):
            await tool_executor_node(state)

        fare_svc.get_quote.assert_awaited_once()
