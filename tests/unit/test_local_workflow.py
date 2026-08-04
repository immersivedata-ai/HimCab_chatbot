"""
tests/unit/test_local_workflow.py
──────────────────────────────────
Unit tests for the local visitor workflow —
validates router classification, planner clarification flow,
and writer local-specific content.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from app.agent.state import AgentState
from app.agent.nodes.router import router_node
from app.models.session import ExtractedEntities, Language
from app.models.user import UserProfile, PastTrip


def _mock_llm_response(json_str: str):
    """Helper to create a mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = json_str
    return mock_response


# ── Router Tests ──────────────────────────────────────────────────────────────

class TestRouterLocalClassification:
    @pytest.mark.asyncio
    async def test_local_visitor_from_shimla(self):
        """Router should classify Shimla origin as local visitor."""
        state = AgentState(
            session_id="test-local-1",
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

        assert result["visitor_type"] == "local"
        assert result["intent"] == "plan_trip"

    @pytest.mark.asyncio
    async def test_local_visitor_from_kullu(self):
        """Router should classify Kullu origin as local visitor."""
        state = AgentState(
            session_id="test-local-2",
            user_id="user-1",
            messages=[HumanMessage(content="I'm from Kullu, want to visit Dharamshala")],
        )
        mock_response = _mock_llm_response(
            '{"intent": "plan_trip", "language": "en", "confidence": 0.93, "visitor_type": "local"}'
        )
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["visitor_type"] == "local"

    @pytest.mark.asyncio
    async def test_outstation_visitor_still_works(self):
        """Outstation classification should not be broken."""
        state = AgentState(
            session_id="test-outstation-1",
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

        assert result["visitor_type"] == "outstation"

    @pytest.mark.asyncio
    async def test_visitor_type_preserved_from_previous_state(self):
        """Router should not downgrade previously classified local visitor."""
        state = AgentState(
            session_id="test-local-preserve",
            user_id="user-1",
            visitor_type="local",
            messages=[HumanMessage(content="haan 2 din ke liye")],
        )
        mock_response = _mock_llm_response(
            '{"intent": "clarification", "language": "hinglish", "confidence": 0.85, "visitor_type": "unknown"}'
        )
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["visitor_type"] == "local"


# ── Planner Tests ─────────────────────────────────────────────────────────────

class TestPlannerLocalFlow:
    @pytest.mark.asyncio
    async def test_local_visitor_no_return_journey_question(self):
        """Planner should NOT ask about return_journey_preference for local visitors."""
        from app.agent.nodes.planner import planner_node

        state = AgentState(
            session_id="test-local-planner-1",
            user_id="user-1",
            visitor_type="local",
            messages=[
                HumanMessage(content="Shimla se Manali jaana hai 2 din ke liye, is weekend"),
            ],
        )
        planner_response = {
            "destination": "Manali",
            "origin": "Shimla",
            "duration_days": 2,
            "travel_dates": "this weekend",
            "group_type": None,
            "group_size": None,
            "budget_range": None,
            "interests": [],
            "special_requirements": [],
            "visitor_type": "local",
            "return_journey_preference": None,
            "home_city": "Shimla",
            "needs_clarification": False,
            "clarification_question": None,
            "confidence": 0.95,
        }
        import json
        mock_response = _mock_llm_response(json.dumps(planner_response))

        with patch("app.agent.nodes.planner.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_pro_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await planner_node(state)

        assert result["needs_clarification"] is False
        assert result["visitor_type"] == "local"
        assert result["entities"].home_city == "Shimla"
        assert result["entities"].return_journey_preference is None

    @pytest.mark.asyncio
    async def test_local_visitor_asks_home_city(self):
        """Planner should ask local visitor their home city when unknown."""
        from app.agent.nodes.planner import planner_node

        state = AgentState(
            session_id="test-local-planner-2",
            user_id="user-1",
            messages=[
                HumanMessage(content="Kasol jaana hai weekend pe"),
            ],
        )
        planner_response = {
            "destination": "Kasol",
            "origin": None,
            "duration_days": 2,
            "travel_dates": "this weekend",
            "group_type": None,
            "group_size": None,
            "budget_range": None,
            "interests": [],
            "special_requirements": [],
            "visitor_type": None,
            "return_journey_preference": None,
            "home_city": None,
            "needs_clarification": True,
            "clarification_question": "Sounds great! Which city in Himachal are you based in?",
            "confidence": 0.82,
        }
        import json
        mock_response = _mock_llm_response(json.dumps(planner_response))

        with patch("app.agent.nodes.planner.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_pro_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await planner_node(state)

        assert result["needs_clarification"] is True
        assert "home" in result["clarification_question"].lower() or "city" in result["clarification_question"].lower() or "based" in result["clarification_question"].lower()


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

    def test_agent_state_local_fields(self):
        """AgentState should have is_frequent_traveller and available_offers."""
        state = AgentState(
            session_id="test-state-1",
            user_id="user-1",
            is_frequent_traveller=True,
            available_offers=[{"title": "Test", "discount_percent": 10}],
        )
        assert state.is_frequent_traveller is True
        assert len(state.available_offers) == 1


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
    def test_build_offers_section_with_offers(self):
        """_build_offers_section should format offers correctly."""
        from app.agent.nodes.writer import _build_offers_section

        state = AgentState(
            session_id="test-writer-1",
            user_id="user-1",
            available_offers=[
                {
                    "title": "HP Resident Special",
                    "description": "7% off for locals",
                    "promo_code": "HIMLOCAL7",
                    "discount_percent": 7.0,
                    "applicable_to": ["cab"],
                    "conditions": "HP ID required",
                },
                {
                    "title": "Frequent Traveller 10% Off",
                    "description": "10% off for frequent users",
                    "promo_code": "HIMFREQ10",
                    "discount_percent": 10.0,
                    "applicable_to": ["cab", "package"],
                    "conditions": "Includes free sightseeing",
                },
            ],
        )
        section = _build_offers_section(state)
        assert "HIMLOCAL7" in section
        assert "HIMFREQ10" in section
        assert "7%" in section or "7.0%" in section
        assert "10%" in section or "10.0%" in section

    def test_build_offers_section_empty(self):
        """_build_offers_section should return empty string when no offers."""
        from app.agent.nodes.writer import _build_offers_section

        state = AgentState(session_id="test-writer-2", user_id="user-1")
        section = _build_offers_section(state)
        assert section == ""

    def test_build_system_prompt_has_local_instructions(self):
        """System prompt for local visitor should contain local-specific instructions."""
        from app.agent.nodes.writer import _build_system_prompt

        state = AgentState(
            session_id="test-writer-3",
            user_id="user-1",
            visitor_type="local",
            entities=ExtractedEntities(
                destination="Manali",
                origin="Shimla",
                duration_days=2,
                visitor_type="local",
                home_city="Shimla",
            ),
        )
        prompt = _build_system_prompt(state)
        assert "LOCAL VISITOR" in prompt
        assert "hidden gems" in prompt.lower()
        assert "Shimla" in prompt

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
