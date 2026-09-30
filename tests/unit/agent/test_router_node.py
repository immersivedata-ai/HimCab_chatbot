"""
tests/unit/test_router_node.py
───────────────────────────────
Unit tests for the Router node intent classification.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent.nodes.router import router_node
from app.agent.state import AgentState


@pytest.fixture
def base_state():
    return AgentState(
        session_id="test-session-123",
        user_id="test-user-456",
        messages=[HumanMessage(content="I want to visit Manali for 5 days")],
    )


def _mock_llm_response(json_str: str):
    """Helper to create a mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = json_str
    return mock_response


class TestRouterNode:
    @pytest.mark.asyncio
    async def test_intent_plan_trip(self, base_state):
        """Router should classify trip planning intent."""
        mock_response = _mock_llm_response('{"intent": "plan_trip", "language": "en", "confidence": 0.95}')

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(base_state)

        assert result["intent"] == "plan_trip"
        assert "router" in result["node_trace"]

    @pytest.mark.asyncio
    async def test_intent_weather_query(self):
        """Router should classify weather query intent."""
        state = AgentState(
            session_id="test-1",
            user_id="user-1",
            messages=[HumanMessage(content="What's the weather like in Manali right now?")],
        )

        mock_response = _mock_llm_response('{"intent": "weather_query", "language": "en", "confidence": 0.92}')

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["intent"] == "weather_query"

    @pytest.mark.asyncio
    async def test_intent_places_query(self):
        """Router should classify places query intent."""
        state = AgentState(
            session_id="test-2",
            user_id="user-1",
            messages=[HumanMessage(content="Best restaurants and cafes in Dharamshala?")],
        )

        mock_response = _mock_llm_response('{"intent": "places_query", "language": "en", "confidence": 0.88}')

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["intent"] == "places_query"

    @pytest.mark.asyncio
    async def test_intent_hotel_query(self):
        """Router should classify hotel query intent."""
        state = AgentState(
            session_id="test-3",
            user_id="user-1",
            messages=[HumanMessage(content="Where should I stay in Kasol?")],
        )

        mock_response = _mock_llm_response('{"intent": "hotel_query", "language": "en", "confidence": 0.90}')

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["intent"] == "hotel_query"

    @pytest.mark.asyncio
    async def test_intent_hinglish(self):
        """Router should detect Hinglish language."""
        state = AgentState(
            session_id="test-1",
            user_id="user-1",
            messages=[HumanMessage(content="Main Shimla jana chahta hoon 3 din ke liye")],
        )

        mock_response = _mock_llm_response('{"intent": "plan_trip", "language": "hinglish", "confidence": 0.90}')

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(state)

        assert result["language"] == "hinglish"

    @pytest.mark.asyncio
    async def test_router_fallback_on_json_error(self, base_state):
        """Router should fallback gracefully on JSON parse error."""
        mock_response = _mock_llm_response("This is not valid JSON")

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(base_state)

        assert result["intent"] == "general_query"
        assert "router" in result["node_trace"]

    @pytest.mark.asyncio
    async def test_router_handles_markdown_wrapped_json(self, base_state):
        """Router should handle JSON wrapped in markdown code blocks."""
        mock_response = _mock_llm_response(
            '```json\n{"intent": "plan_trip", "language": "en", "confidence": 0.95}\n```'
        )

        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                return_value=mock_response
            )
            result = await router_node(base_state)

        assert result["intent"] == "plan_trip"

    @pytest.mark.asyncio
    async def test_empty_message_defaults(self):
        """Router should handle empty message history."""
        state = AgentState(session_id="empty-test", user_id="u1", messages=[])
        result = await router_node(state)
        assert result["intent"] == "general_query"

    @pytest.mark.asyncio
    async def test_router_fallback_on_llm_exception(self, base_state):
        """Router should fallback gracefully when LLM call itself fails."""
        with patch("app.agent.nodes.router.get_vertex_ai_service") as mock_svc:
            mock_svc.return_value.get_flash_model.return_value.ainvoke = AsyncMock(
                side_effect=Exception("Vertex AI unavailable")
            )
            result = await router_node(base_state)

        assert result["intent"] == "general_query"
        assert "router" in result["node_trace"]


# ── A bare answer is an answer, whatever the word looks like ──────────────────


class TestAnAnswerIsNotMistakenForAQuestion:
    """The reported failure, and its root cause.

        assistant  aap himachal mein kis shehar mein rehte hain?   (which city do you live in?)
        rider      kasol
        assistant  <a write-up of Shimla's attractions, no cab, no price>

    The router classified "kasol" as `places_query` — a question ABOUT a place — and
    everything downstream then did the right thing for the wrong question:
    `places_query` is not a travel intent, so `response_mode` returned ANSWER, the fare
    was never requested, no booking handoff shipped, and a resident who wanted a cab
    from Kasol to Shimla got a brochure.

    The old guard only rescued `general_query`, so a bare place name that looked like a
    topic went straight past it. The test is now whether the message ANSWERS the
    question in flight, asked of the same parser the planner will use.
    """

    @pytest.mark.parametrize(
        "asked,reply",
        [
            ("origin", "kasol"),
            ("origin", "Shimla"),
            ("destination", "manali"),
            ("group_size", "4"),
            ("group_size", "4 people"),
        ],
    )
    def test_a_bare_answer_is_read_as_one(self, asked, reply):
        from app.agent.nodes.router import _answers_the_question

        assert _answers_the_question(asked, reply) is True

    @pytest.mark.parametrize(
        "asked,reply",
        [
            # A real topic switch keeps its own intent and is still answered.
            ("origin", "actually, what's the weather there?"),
            ("origin", "what places can I see in Manali"),
            ("destination", "how many days should I go for"),
            # An answer aimed at a different field is not an answer to this one.
            ("origin", "4"),
            ("group_size", "kasol"),
            # Nothing at all.
            ("origin", ""),
            ("origin", "   "),
        ],
    )
    def test_anything_else_keeps_its_intent(self, asked, reply):
        from app.agent.nodes.router import _answers_the_question

        assert _answers_the_question(asked, reply) is False

    def test_an_unknown_field_is_never_an_answer(self):
        """A field the slot table does not define cannot be answered, and must not
        crash the turn trying."""
        from app.agent.nodes.router import _answers_the_question

        assert _answers_the_question("not_a_slot", "kasol") is False
