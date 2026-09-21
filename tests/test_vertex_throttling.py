"""Behaviour when Vertex says it is out of capacity.

Production logged, sustained:

    Retrying langchain_google_vertexai...as it raised ResourceExhausted: 429

Gemini on Vertex runs on Dynamic Shared Quota — a shared pool, and NO quota increase can
be requested. The only levers on this side are sending fewer requests and spreading them
out. Before this, the system did the opposite of both:

  * `ChatVertexAI.max_retries` was left at its default of 6, so every logical call
    became up to six requests, invisibly.
  * Our own retries (planner extraction, writer JSON re-ask) fired IMMEDIATELY and did
    not recognise a 429, so they re-asked an empty pool at the worst possible moment.
  * Nothing bounded concurrency: 10 instances x 80 concurrency = up to 800 turns at
    once, ~3 model calls each, with no semaphore anywhere.
  * A 429 was logged as `planner_extraction_failed` — indistinguishable from a model
    answering badly, so the metric that should have screamed said nothing.

A throttled turn must therefore: not retry, be logged as throttling under its own name,
and tell the rider the truth rather than inventing an answer out of an empty extraction.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.state import AgentState
from app.core.config import settings
from app.domain.conversation import ExtractedEntities


def _429() -> ResourceExhausted:
    """The real thing: `google.api_core.exceptions.ResourceExhausted`, verified against
    a live throttled project rather than guessed at. If LangChain ever wrapped it, every
    `except ResourceExhausted` in the agent would silently stop firing."""
    return ResourceExhausted("429 Resource exhausted. Please try again later.")


class TestTheClientIsConfiguredToBeAGoodCitizen:
    """Asserted on the constructed client, because these are one-word settings whose
    absence is invisible and expensive."""

    @pytest.fixture(scope="class")
    def service(self):
        from app.services.vertex_ai import get_vertex_ai_service

        return get_vertex_ai_service()

    def test_langchains_invisible_retry_is_bounded(self, service):
        """The default is 6. Six retries into an exhausted pool is six more requests
        that will also fail, and the only sign is a warning nobody is watching."""
        for model in (service.gemini_flash, service.gemini_flash_structured):
            assert model.max_retries == settings.vertex_max_retries
            assert model.max_retries < 6

    def test_both_clients_are_paced(self, service):
        assert service.gemini_flash.rate_limiter is not None
        assert service.gemini_flash_structured.rate_limiter is not None

    def test_they_share_one_bucket(self, service):
        """Vertex enforces quota per project and model, not per Python object. Two
        independently-paced clients would simply take turns exceeding it."""
        assert (
            service.gemini_flash.rate_limiter
            is service.gemini_flash_structured.rate_limiter
        )


class TestTheConcurrencyCeiling:

    @pytest.mark.asyncio
    async def test_no_more_than_the_limit_are_in_flight_at_once(self):
        """The rate limiter paces how often a call STARTS; this bounds how many are
        open. They are different failures: when Vertex slows down, a paced trickle still
        piles up into a thundering herd built out of politeness."""
        import app.services.vertex_ai as vx

        vx._in_flight = None                       # a fresh gate on this event loop
        peak = 0
        live = 0

        async def work():
            nonlocal peak, live
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1

        await asyncio.gather(*[vx.call_model(work()) for _ in range(40)])

        assert peak <= settings.vertex_max_in_flight
        assert peak > 1, "serialised everything — that is a latency bug, not a fix"
        vx._in_flight = None

    @pytest.mark.asyncio
    async def test_a_failure_does_not_leak_a_permit(self):
        """A leaked permit is a slow strangle: the ceiling drops by one per error until
        the instance stops calling Vertex at all."""
        import app.services.vertex_ai as vx

        vx._in_flight = None

        async def boom():
            raise _429()

        for _ in range(settings.vertex_max_in_flight + 3):
            with pytest.raises(ResourceExhausted):
                await vx.call_model(boom())

        async def fine():
            return "ok"

        assert await asyncio.wait_for(vx.call_model(fine()), timeout=1) == "ok"
        vx._in_flight = None


class TestAThrottledPlannerDegradesWithoutReAsking:

    @staticmethod
    def _state():
        return AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="local",
            entities=ExtractedEntities(visitor_type="local",
                                       last_asked_field="destination"),
            messages=[AIMessage(content="Where would you like to go?"),
                      HumanMessage(content="mei kasol jana chahta hu")],
        )

    @pytest.mark.asyncio
    async def test_it_does_not_retry_a_429(self):
        """LangChain has already retried with backoff by the time this is raised. One
        more re-ask is one more request into a pool that just said it has no room."""
        from app.agent.nodes.planner import planner_node

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=_429())
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            await planner_node(self._state())

        assert llm.with_structured_output.return_value.ainvoke.await_count == 1

    @pytest.mark.asyncio
    async def test_the_turn_still_works_on_the_deterministic_readers(self):
        """Degrading is right. It was the SILENCE around it that turned a capacity
        problem into "the AI is hallucinating"."""
        from app.agent.nodes.planner import planner_node

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=_429())
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(self._state())

        assert result["entities"].destination == "Kasol"

    @pytest.mark.asyncio
    async def test_a_blank_extraction_still_gets_its_retry(self):
        """Throttling is not the only reason a call comes back empty, and the retry that
        exists for the other reason must survive this change."""
        from app.agent.nodes.planner import planner_node
        from tests.conftest import wire_extraction

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=[None, wire_extraction(destination="Kasol")]
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            await planner_node(self._state())

        assert llm.with_structured_output.return_value.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_that_retry_waits_first(self):
        """It used to re-ask instantly. Under load every client re-asks at the same
        moment, and the second attempt lands in the same congestion as the first."""
        from app.agent.nodes import planner as planner_mod

        with patch.object(planner_mod, "_backoff", new=AsyncMock()) as backoff:
            llm = MagicMock()
            llm.with_structured_output.return_value.ainvoke = AsyncMock(
                side_effect=[None, None]
            )
            with patch.object(planner_mod, "get_vertex_ai_service") as svc:
                svc.return_value.get_flash_model.return_value = llm
                await planner_mod.planner_node(self._state())

        backoff.assert_awaited()

    @pytest.mark.asyncio
    async def test_the_delay_is_jittered(self):
        """Backoff without jitter just synchronises every client onto a slower clock."""
        from app.agent.nodes.planner import _backoff

        seen = set()
        with patch("app.agent.nodes.planner.asyncio.sleep", new=AsyncMock()) as slept:
            for _ in range(8):
                await _backoff(1)
                seen.add(slept.await_args[0][0])
        assert len(seen) > 1


class TestAThrottledWriterSaysSo:

    @staticmethod
    def _state(message: str = "plan my trip"):
        # The reply language is detected from the CONVERSATION, not from a state field —
        # so the message is what decides it, exactly as in a real turn.
        return AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="outstation",
            entities=ExtractedEntities(visitor_type="outstation", origin="Delhi",
                                       destination="Manali", group_size=2,
                                       duration_days=3, travel_dates="kal"),
            messages=[HumanMessage(content=message)],
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "message,marker",
        [("plan my trip", "busy"), ("मेरी यात्रा की योजना बनाओ", "भीड़")],
    )
    async def test_it_tells_the_rider_to_try_again(self, message, marker):
        """Not "I encountered an issue" — that reads as broken and permanent. A 429 WILL
        clear, and saying so is more use to them than an apology. In their language,
        because a rider writing Hindi should not be told anything in English."""
        from app.agent.nodes.writer import writer_node

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=_429())
        with patch("app.agent.nodes.writer.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await writer_node(self._state(message))

        assert marker in result["final_response_text"]
        assert result["structured_events"] == []

    @pytest.mark.asyncio
    async def test_it_does_not_invent_a_plan(self):
        """The failure mode this replaces: a throttled turn produced an empty extraction
        and a writer left to fill the silence, which is what got reported as
        hallucination."""
        from app.agent.nodes.writer import writer_node

        llm = MagicMock()
        llm.ainvoke = AsyncMock(side_effect=_429())
        with patch("app.agent.nodes.writer.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await writer_node(self._state())

        assert result.get("structured_itinerary") is None
        for invented in ("₹", "Day 1"):
            assert invented not in result["final_response_text"]
