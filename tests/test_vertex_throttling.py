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
from tests.conftest import stub_vertex


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

    def test_langchains_invisible_retry_is_never_left_at_the_default(self, service):
        """The default is 6, and the only sign of it is a warning nobody is watching.
        Six retries into an exhausted pool is six more requests that will also fail.

        The exact number differs by role — the primary does not retry at all when a
        standby exists, see `TestTheStandbyModel` — but NOTHING here may sit at 6.
        """
        models = [service.gemini_flash, service.gemini_flash_structured]
        models += [m for m in (service.fallback_flash, service.fallback_structured) if m]
        for model in models:
            assert model.max_retries <= settings.vertex_max_retries
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
            stub_vertex(svc, llm)
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
            stub_vertex(svc, llm)
            result = await planner_node(self._state())

        assert result["entities"].destination == "Kasol"

    @pytest.mark.asyncio
    async def test_a_blank_extraction_still_gets_its_retry(self):
        """Throttling is not the only reason a call comes back empty, and the retry that
        exists for the other reason must survive this change."""
        from app.agent.nodes.planner import planner_node
        from tests.conftest import stub_vertex, wire_extraction

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            side_effect=[None, wire_extraction(destination="Kasol")]
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            stub_vertex(svc, llm)
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
                stub_vertex(svc, llm)
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
            stub_vertex(svc, llm)
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
            stub_vertex(svc, llm)
            result = await writer_node(self._state())

        assert result.get("structured_itinerary") is None
        for invented in ("₹", "Day 1"):
            assert invented not in result["final_response_text"]


class TestTheStandbyModel:
    """Quota on Vertex is PER MODEL, and the pools are not the same size.

    Measured on this project during the outage, same region, same minute:
    `gemini-2.5-flash-lite` returned 429 on 8 of 8 calls while `gemini-2.5-flash`
    answered 8 of 8. A cheap model with no capacity is not cheaper than a dearer one
    that works — it is an outage.
    """

    @pytest.fixture(scope="class")
    def service(self):
        from app.services.vertex_ai import get_vertex_ai_service

        return get_vertex_ai_service()

    def test_there_is_one(self, service):
        assert service.fallback_name
        assert service.fallback_name != settings.gemini_flash_model

    def test_the_primary_does_not_climb_a_retry_ladder_first(self, service):
        """Retrying and failing over are two answers to the same question, and paying
        for both is the worst of each. With `max_retries=2` on the primary, one planner
        call against an exhausted pool took 24 SECONDS to reach the standby — three
        calls a turn, past the per-call timeout for the slower ones. Now ~9s."""
        assert service.primary_model(streaming=False).max_retries == 0

    def test_the_standby_still_retries(self, service):
        """By then there is nowhere else to go, so a short ladder is worth walking."""
        assert service.fallback_structured.max_retries == settings.vertex_max_retries

    def test_only_a_429_fails_over(self, service):
        """A fallback that catches everything hides real faults: a bad schema would be
        retried against a second model, fail there too, cost twice as much, and report
        the error from the wrong place."""
        runnable = service.get_flash_model(streaming=False)
        assert runnable.exceptions_to_handle == (ResourceExhausted,)

    def test_the_structured_path_has_one_too(self, service):
        """The planner's path. `with_structured_output` is a method on `ChatVertexAI`
        and not on the wrapper, so the two have to be composed in this order — easy to
        get backwards, and backwards means no standby at all."""
        from app.agent.nodes.planner import _VertexExtraction

        runnable = service.structured(_VertexExtraction)
        assert runnable.exceptions_to_handle == (ResourceExhausted,)

    def test_it_can_be_turned_off(self, monkeypatch):
        """One env var, because the right answer changes with what the pools are doing
        and nobody should need a deploy to find out."""
        import app.services.vertex_ai as vx

        monkeypatch.setattr(settings, "gemini_fallback_model", "")
        monkeypatch.setattr(vx, "_vertex_ai_service", None)
        service = vx.get_vertex_ai_service()
        try:
            assert service.fallback_name == ""
            assert service.fallback_structured is None
            # No standby means the primary must go back to retrying for itself.
            assert service.primary_model(streaming=False).max_retries == settings.vertex_max_retries
        finally:
            monkeypatch.setattr(vx, "_vertex_ai_service", None)
