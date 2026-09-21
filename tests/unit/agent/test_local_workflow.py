"""
tests/unit/test_local_workflow.py
──────────────────────────────────
Unit tests for the local visitor workflow —
validates router classification, planner clarification flow,
and writer local-specific content.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.router import router_node
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.user import PastTrip, UserProfile
from tests.conftest import stub_vertex


def _mock_llm_response(json_str: str):
    """Helper to create a mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = json_str
    return mock_response


# ── Router Tests ──────────────────────────────────────────────────────────────

class TestRouterClassification:
    """The router classifies INTENT (for routing) but deliberately does NOT own
    visitor_type — that is set by the persona seed (front-end page) and confirmed by the
    planner's origin-based classification. So the router must never write visitor_type,
    which guarantees a seeded value can't be flipped by an unreliable per-turn LLM guess.
    """

    @pytest.mark.asyncio
    async def test_router_classifies_intent(self):
        """Router returns the intent used for graph routing."""
        state = AgentState(
            session_id="test-1",
            user_id="user-1",
            messages=[HumanMessage(content="Shimla se Manali jaana hai 2 din ke liye")],
        )
        mock_response = _mock_llm_response(
            '{"intent": "plan_trip", "language": "hinglish", "confidence": 0.95, "visitor_type": "local"}'
        )
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["intent"] == "plan_trip"

    @pytest.mark.asyncio
    async def test_router_does_not_write_visitor_type(self):
        """Even when the LLM guesses a visitor_type, the router must not put it in state —
        so LangGraph keeps whatever the persona seed / planner set."""
        state = AgentState(
            session_id="test-2",
            user_id="user-1",
            messages=[HumanMessage(content="Coming from Delhi to Manali for 5 days")],
        )
        mock_response = _mock_llm_response(
            '{"intent": "plan_trip", "language": "en", "confidence": 0.97, "visitor_type": "outstation"}'
        )
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert "visitor_type" not in result

    @pytest.mark.asyncio
    async def test_seeded_visitor_type_not_overwritten(self):
        """A persona-seeded local visitor must survive the router untouched, even when the
        LLM guesses something else (the exact bug where 'manali' flipped it)."""
        state = AgentState(
            session_id="test-3",
            user_id="user-1",
            visitor_type="local",
            messages=[HumanMessage(content="manali")],
        )
        mock_response = _mock_llm_response(
            '{"intent": "plan_trip", "language": "en", "confidence": 0.85, "visitor_type": "outstation"}'
        )
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        # Router didn't write visitor_type, so the seeded "local" is preserved by the graph.
        assert "visitor_type" not in result


# ── Planner Tests ─────────────────────────────────────────────────────────────

def _mock_structured_llm(structured_response):
    """Helper to mock get_flash_model(...).with_structured_output(...).ainvoke()."""
    mock_llm = MagicMock()
    mock_llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=structured_response)
    return mock_llm


class TestPlannerLocalFlow:
    @pytest.mark.asyncio
    async def test_local_visitor_return_journey_confirmed(self):
        """Once return_journey_preference is known, clarification resolves for locals too."""
        from app.agent.nodes.planner import planner_node
        from tests.conftest import wire_extraction

        state = AgentState(
            session_id="test-local-planner-1",
            user_id="user-1",
            visitor_type="local",
            messages=[
                HumanMessage(content="Shimla se Manali jaana hai 2 din ke liye, is weekend, return cab book kar li hai"),
            ],
        )
        planner_response = wire_extraction(
            destination="Manali",
            origin="Shimla",
            duration_days=2,
            group_size=2,
            travel_dates="this weekend",
            return_journey_preference="booked",
            home_city="Shimla",
            origin_in_himachal=True,
            confidence=0.95,
        )

        with patch("app.agent.nodes.planner.get_vertex_ai_service") as mock_svc:
            stub_vertex(mock_svc, _mock_structured_llm(planner_response))
            result = await planner_node(state)

        assert result["needs_clarification"] is False
        assert result["visitor_type"] == "local"
        assert result["entities"].home_city == "Shimla"
        assert result["entities"].return_journey_preference == "booked"

    @pytest.mark.asyncio
    async def test_return_journey_does_not_block_the_itinerary(self):
        """The return cab is worth knowing, but it is not worth withholding the plan for.

        It used to be the sixth mandatory question, so a user who had given every
        trip detail still got interrogated instead of an itinerary. It is now carried
        in `open_questions` and asked in the itinerary's closing line instead.
        """
        from app.agent.nodes.planner import planner_node
        from tests.conftest import wire_extraction

        state = AgentState(
            session_id="test-local-planner-1b",
            user_id="user-1",
            visitor_type="local",
            messages=[
                HumanMessage(content="Shimla se Manali jaana hai 2 din ke liye, is weekend"),
            ],
        )
        planner_response = wire_extraction(
            destination="Manali",
            origin="Shimla",
            duration_days=2,
            group_size=2,
            travel_dates="this weekend",
            return_journey_preference=None,
            home_city="Shimla",
            origin_in_himachal=True,
            confidence=0.95,
        )

        with patch("app.agent.nodes.planner.get_vertex_ai_service") as mock_svc:
            stub_vertex(mock_svc, _mock_structured_llm(planner_response))
            result = await planner_node(state)

        assert result["needs_clarification"] is False
        assert result["entities"].return_journey_preference is None
        # Never silently assumed either — the writer is told nothing about it.
        assert "return_journey_preference" not in result["assumed_fields"]

    @pytest.mark.asyncio
    async def test_local_visitor_asks_home_city(self):
        """Planner should ask local visitor their home city when unknown."""
        from app.agent.nodes.planner import planner_node
        from tests.conftest import wire_extraction

        state = AgentState(
            session_id="test-local-planner-2",
            user_id="user-1",
            visitor_type="local",
            messages=[
                HumanMessage(content="Kasol jaana hai weekend pe"),
            ],
        )
        planner_response = wire_extraction(
            destination="Kasol",
            origin=None,
            duration_days=2,
            travel_dates="this weekend",
            return_journey_preference=None,
            home_city=None,
            needs_clarification=True,
            clarification_question="Sounds great! Which city in Himachal are you based in?",
            confidence=0.82,
        )

        with patch("app.agent.nodes.planner.get_vertex_ai_service") as mock_svc:
            stub_vertex(mock_svc, _mock_structured_llm(planner_response))
            result = await planner_node(state)

        assert result["needs_clarification"] is True
        # The question may be in Hindi (हिमाचल/शहर/रहते) or English (home/city/based)
        q = result["clarification_question"].lower()
        assert ("home" in q or "city" in q or "based" in q
                or "हिमाचल" in q or "शहर" in q or "रहते" in q)


# ── State & Entity Tests ──────────────────────────────────────────────────────

class TestLocalEntities:
    def test_home_city_field_exists(self):
        """ExtractedEntities should support home_city field."""
        entities = ExtractedEntities(
            destination="Manali",
            origin="Shimla",
            visitor_type="local",
            home_city="Shimla",
        )
        assert entities.home_city == "Shimla"

    def test_home_city_defaults_to_none(self):
        """home_city should default to None."""
        entities = ExtractedEntities(destination="Manali")
        assert entities.home_city is None

    def test_agent_state_carries_the_planner_persona(self):
        state = AgentState(
            session_id="test-state-1",
            user_id="user-1",
            visitor_type="local",
            entities=ExtractedEntities(visitor_type="local", home_city="Shimla"),
        )
        assert state.visitor_type == "local"
        assert state.entities.home_city == "Shimla"


# ── User Profile Tests ───────────────────────────────────────────────────────

class TestUserProfileEnhancements:
    def test_frequent_destinations(self):
        """get_frequent_destinations should return destinations visited > once."""
        profile = UserProfile(
            user_id="u-test",
            firebase_uid="fb-test",
            total_trips=4,
            past_trips=[
                PastTrip(destination="Manali", duration_days=3, travel_date="2026-01-01"),
                PastTrip(destination="Shimla", duration_days=2, travel_date="2026-02-01"),
                PastTrip(destination="Manali", duration_days=2, travel_date="2026-03-01"),
                PastTrip(destination="Kasol", duration_days=2, travel_date="2026-04-01"),
            ],
        )
        frequent = profile.get_frequent_destinations()
        assert "Manali" in frequent
        assert "Shimla" not in frequent

    def test_context_string_includes_loyalty(self):
        """to_context_string should include loyalty tier info."""
        profile = UserProfile(
            user_id="u-test",
            firebase_uid="fb-test",
            total_trips=4,
            home_city="Shimla",
        )
        ctx = profile.to_context_string()
        assert "frequent" in ctx.lower()
        assert "Shimla" in ctx
        assert "4 trips" in ctx

    def test_context_string_new_user(self):
        """New user context string should show 'new' tier."""
        profile = UserProfile(
            user_id="u-new",
            firebase_uid="fb-new",
            total_trips=0,
        )
        ctx = profile.to_context_string()
        assert "new" in ctx.lower()


# ── Writer Context Tests ──────────────────────────────────────────────────────

class TestWriterLocalContext:
    def test_a_local_gets_a_cab_recommendation_not_an_itinerary(self):
        """Residents know the place. What they are choosing is the ride, so the prompt
        must ask for a cab, a fare and a drive time — and explicitly not a day plan."""
        from app.agent.nodes.writer import _build_system_prompt

        state = AgentState(
            session_id="test-writer-3",
            user_id="user-1",
            intent="plan_trip",
            visitor_type="local",
            entities=ExtractedEntities(
                destination="Manali",
                origin="Shimla",
                group_size=4,
                visitor_type="local",
                home_city="Shimla",
            ),
            route_estimate={
                "origin": {"name": "Shimla", "lat": 31.1, "lon": 77.17},
                "destination": {"name": "Manali", "lat": 32.24, "lon": 77.19},
                "distance_km": 270, "duration_minutes": 270,
                "duration_text": "4h 30m", "is_approximate": False,
            },
        )
        prompt = _build_system_prompt(state)

        assert "FORBIDDEN" in prompt and "day-by-day plan" in prompt
        assert "ITINERARY JSON" not in prompt
        # The sightseeing brief is gone — matched on the instruction that sells it,
        # not on the words "hidden gems", which the ride prompt uses to FORBID it.
        assert "LOCAL VISITOR INSTRUCTIONS" not in prompt
        assert "prioritise hidden gems" not in prompt
        # The arrival-day brief is itinerary guidance; in a cab reply it was what
        # produced "you can enjoy light sightseeing when you arrive".
        assert "Day 1 must begin" not in prompt
        # …and it must carry the ride facts instead.
        assert "THIS JOURNEY" in prompt
        assert "4h 30m" in prompt
        assert "270 km" in prompt
        assert "Shimla" in prompt and "Manali" in prompt
        # The app owns booking; the reply must not ask for it.
        assert "Do not ask whether they want to book" in prompt

    def test_build_system_prompt_no_local_for_outstation(self):
        """System prompt for outstation visitor should NOT contain local instructions."""
        from app.agent.nodes.writer import _build_system_prompt

        state = AgentState(
            session_id="test-writer-4",
            user_id="user-1",
            visitor_type="outstation",
            entities=ExtractedEntities(
                destination="Manali",
                origin="Delhi",
                duration_days=4,
                visitor_type="outstation",
            ),
        )
        prompt = _build_system_prompt(state)
        assert "LOCAL VISITOR" not in prompt
        assert "OUTSTATION VISITOR" in prompt
