"""
tests/unit/test_cancellation.py
────────────────────────────────
Changing your mind: "actually, cancel that".

Reported from the app: a rider who started booking and then backed out got a reply that
had nothing to do with what they said. There was no cancellation anywhere in this system
— so "cancel" arrived as an ordinary message, was classified as trip planning, and was
read by the slot parsers as the ANSWER to whichever question was in flight.
`parse_place("cancel")` returned "Cancel", which became the pickup city, which was
geocoded, which was priced. The rider who had just called it off was shown a fare, a cab
list and a booking screen for the request they had abandoned.

Four guarantees, one per section below:

  1. a cancellation is RECOGNISED, in code, before any model runs
  2. it is never confused with a question about cancelling, or with a correction
  3. the reply says only what is true — including that a confirmed booking can only be
     cancelled from the app, which is the one thing this service must not do itself
  4. the request is CLEARED, so the next message starts a new one
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import cancellation, slots
from app.agent.graph import cancel_node, route_after_router
from app.agent.nodes.router import router_node
from app.agent.state import AgentState
from app.core.enums import CancelScope, Intent, ResponseMode
from app.domain.conversation import ExtractedEntities
from app.services.chat_session import RENTAL


def _state(message: str, **kwargs) -> AgentState:
    base = {
        "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
        "messages": [AIMessage(content="Which cab would you like?"),
                  HumanMessage(content=message)],
        "entities": ExtractedEntities(visitor_type=RENTAL, origin="Shimla",
                                   rental_hours=8, travel_dates="tomorrow",
                                   pickup_time="09:00", group_size=4,
                                   last_asked_field="cab_category"),
    }
    base.update(kwargs)
    return AgentState(**base)


# ── 1. It is recognised, in code, before any model runs ───────────────────────

class TestACancellationIsRecognised:

    @pytest.mark.parametrize(
        "message",
        [
            "cancel", "Cancel", "cancel it", "cancel that", "cancel this",
            "actually never mind", "never mind", "nevermind", "forget it",
            "I've changed my mind", "not interested", "I don't want it any more",
            "call it off", "drop it", "leave it", "abort",
            # Hindi and Hinglish — the ones riders actually type.
            "rehne do", "rehne do yaar", "cancel karo", "cancel kar do",
            "nahi chahiye", "band karo", "mat karo", "छोड़ दो", "रहने दो",
        ],
    )
    def test_the_obvious_phrasings_need_no_model(self, message):
        assert cancellation.looks_like_cancellation(message) is True

    @pytest.mark.asyncio
    async def test_the_router_decides_it_without_calling_vertex(self):
        """The detector runs BEFORE the classifier, so an unmistakable "cancel" costs no
        LLM call and cannot be reinterpreted by one having a bad day."""
        with patch("app.agent.nodes.router.get_vertex_ai_service") as svc:
            result = await router_node(_state("actually, cancel it"))

        svc.assert_not_called()
        assert result["intent"] == Intent.CANCEL_REQUEST

    @pytest.mark.asyncio
    async def test_a_phrasing_the_list_misses_is_caught_by_the_classifier(self):
        """Two detectors, either sufficient. This one reads phrasings no list can
        enumerate."""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"intent": "cancel_request", "language": "en", "confidence": 0.9}'
        ))
        with patch("app.agent.nodes.router.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await router_node(_state("you know what, scrap the whole thing"))

        assert result["intent"] == Intent.CANCEL_REQUEST

    @pytest.mark.asyncio
    async def test_it_still_works_when_vertex_is_down(self):
        with patch("app.agent.nodes.router.get_vertex_ai_service",
                   side_effect=RuntimeError("vertex down")):
            result = await router_node(_state("cancel karo"))

        assert result["intent"] == Intent.CANCEL_REQUEST

    def test_the_flow_stops_at_the_router(self):
        """Straight to `cancel_node` — no extraction model, no slot policy, no tools.
        Everything that used to go wrong went wrong downstream of here."""
        assert route_after_router(
            _state("cancel", intent=Intent.CANCEL_REQUEST)
        ) == "cancel"

    def test_a_cancellation_outranks_the_persona(self):
        """The persona wins over a drifting intent everywhere else in this system —
        `rental_query` in a local chat is still a rental. This is the one intent that
        must beat it: whatever they came for, they have said they no longer want it."""
        for persona in (None, "local", "outstation", RENTAL):
            assert slots.response_mode(Intent.CANCEL_REQUEST, persona) == ResponseMode.CANCELLED
            assert slots.needs_for(Intent.CANCEL_REQUEST, persona) == {}


# ── 2. Not everything that says "cancel" is one ───────────────────────────────

class TestWhatIsNotACancellation:

    @pytest.mark.parametrize(
        "message",
        [
            # Asking ABOUT it. Answering these by cancelling is the same bug, reversed.
            "what is your cancellation policy?",
            "is there a cancellation fee?",
            "can I cancel later?",
            "what happens if I cancel?",
            "cancel karne par kitna charge lagta hai?",
            # Wanting it FASTER, not gone. This is `proceed_requested`.
            "stop asking me questions",
            "no more questions, just book it",
            # Correcting a detail. A correction is not a cancellation.
            "no, make it 5 hours instead",
            "not Manali, Kasol",
            "a different cab please",
            # Ordinary answers to the questions we ask.
            "no", "yes", "8 hours", "luxury", "Shimla", "tomorrow",
        ],
    )
    def test_these_never_cancel_a_request(self, message):
        assert cancellation.looks_like_cancellation(message) is False

    def test_an_instruction_survives_a_question_mark(self):
        """"Can you cancel this?" names what to cancel — it is an instruction politely
        punctuated. A bare "cancel" inside a question is not."""
        assert cancellation.looks_like_cancellation("can you cancel this?") is True
        assert cancellation.looks_like_cancellation("can I cancel later?") is False

    def test_no_proceed_phrase_is_ever_read_as_a_cancellation(self):
        """The two backstops must not overlap. "Chhodo" is the reason this test exists:
        it reads like "drop it", but the planner has always treated it as "stop asking
        and just make it" — and a word that means both has to keep the meaning that
        costs less when it is wrong. Cancelling by mistake destroys a request; proceeding
        by mistake produces one."""
        from app.agent.nodes.planner import _PROCEED_PHRASES

        clashes = [p for p in _PROCEED_PHRASES if cancellation.looks_like_cancellation(p)]
        assert clashes == [], f"these mean 'get on with it', not 'cancel': {clashes}"

    def test_cancellation_is_never_cancel(self):
        """Phrases match whole words. Substring matching would make every mention of
        the policy a cancellation."""
        assert cancellation.looks_like_cancellation("tell me the cancellation terms") is False

    @pytest.mark.asyncio
    async def test_the_router_leaves_a_correction_alone(self):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(
            content='{"intent": "clarification", "language": "en", "confidence": 0.9}'
        ))
        with patch("app.agent.nodes.router.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await router_node(_state("no, make it 5 hours instead"))

        assert result["intent"] != Intent.CANCEL_REQUEST


# ── 3. The reply says only what is true ───────────────────────────────────────

class TestWhatTheRiderIsTold:

    @pytest.mark.asyncio
    async def test_it_confirms_the_request_is_off_and_nothing_was_charged(self):
        result = await cancel_node(_state("cancel it"))

        text = result["final_response_text"]
        assert "cancelled that request" in text
        assert "Nothing was booked and nothing has been charged" in text
        # No sales pitch, in either direction.
        for banned in ("₹", "Rs ", "package", "cab options", "booking screen is open"):
            assert banned not in text

    @pytest.mark.asyncio
    async def test_an_existing_booking_is_never_claimed_to_be_cancelled(self):
        """The dangerous half. This service never books anything, holds no trip id, and
        the backend's cancel is irreversible and notifies the driver — so a rider with a
        confirmed cab is sent to the screen that owns it, and is NOT told to relax."""
        result = await cancel_node(_state(
            "I booked a cab yesterday, cancel it",
            entities=ExtractedEntities(visitor_type=RENTAL),
        ))

        text = result["final_response_text"]
        assert "Trips" in text
        assert "nothing here for me to cancel" in text
        # The reassurance that would be a lie here.
        assert "cancelled that request" not in text
        assert result["structured_events"][0]["payload"]["scope"] == CancelScope.BOOKED_TRIP
        assert result["structured_events"][0]["payload"]["open_screen"] == "trips"

    @pytest.mark.asyncio
    async def test_an_ambiguous_cancellation_is_true_under_both_readings(self):
        """"Cancel my booking", typed halfway through building a request, is genuinely
        both. Rather than pick one and be wrong half the time, the reply covers both."""
        result = await cancel_node(_state("cancel my booking"))

        text = result["final_response_text"]
        assert "Nothing from this chat was booked or charged" in text
        assert "Trips" in text
        assert result["structured_events"][0]["payload"]["scope"] == CancelScope.UNCLEAR

    @pytest.mark.asyncio
    async def test_it_answers_in_the_riders_language(self):
        result = await cancel_node(_state(
            "rehne do",
            messages=[AIMessage(content="कौन सी गाड़ी चाहिए?"),
                      HumanMessage(content="mujhe nahi chahiye, rehne do")],
        ))

        assert "रद्द" in result["final_response_text"]

    @pytest.mark.asyncio
    async def test_no_fare_no_cabs_and_above_all_no_booking_screen(self):
        """The other half of the reported bug: the assistant answered a cancellation
        with a price and a pre-filled booking screen."""
        result = await cancel_node(_state(
            "cancel", cab_fare={"fare_inr": 2400, "options": [{"category_name": "Sedan"}]},
            rental_packages=[{"hours": 8, "label": "8 hrs", "from_rate_card": True}],
            offers=[{"coupon_code": "SAVE100"}],
        ))

        types = [e["type"] for e in result["structured_events"]]
        assert types == ["request_cancelled"]
        assert result["cab_fare"] is None
        assert result["offers"] == []
        assert result["rental_packages"] == []
        assert result["awaiting_cab_choice"] is False

    @pytest.mark.asyncio
    async def test_the_app_is_told_to_dismiss_the_booking_screen(self):
        """A `booking_handoff` earlier in the conversation may have opened a pre-filled
        sheet. Leaving it up is how a rider books the thing they just cancelled."""
        payload = (await cancel_node(_state("cancel")))["structured_events"][0]["payload"]

        assert payload["dismiss_booking_screen"] is True

    @pytest.mark.asyncio
    async def test_the_writer_can_never_answer_a_cancellation(self):
        """The graph routes cancellations away from the writer. This is the guard under
        that: with no branch of its own a CANCELLED turn would fall through to the
        itinerary prompt and answer "cancel my booking" with a four-day plan."""
        from app.agent.nodes.writer import writer_node

        state = _state("cancel it", intent=Intent.CANCEL_REQUEST)
        with patch("app.agent.nodes.writer.get_vertex_ai_service") as svc:
            result = await writer_node(state)

        svc.assert_not_called()
        assert "cancelled that request" in result["final_response_text"]
        assert [e["type"] for e in result["structured_events"]] == ["request_cancelled"]


# ── 4. The request is cleared, so the next message starts a new one ───────────

class TestTheRequestIsDropped:

    @pytest.mark.asyncio
    async def test_every_detail_of_the_request_goes(self):
        result = await cancel_node(_state("cancel it"))

        entities = result["entities"]
        for field in ("origin", "rental_hours", "travel_dates", "pickup_time",
                      "group_size", "cab_category", "destination"):
            assert getattr(entities, field) in (None, [], ""), field

    @pytest.mark.asyncio
    async def test_the_rider_is_kept(self):
        """Which assistant they are talking to, and where they live, were true before
        this request and are true after it."""
        result = await cancel_node(_state(
            "cancel it",
            entities=ExtractedEntities(visitor_type=RENTAL, home_city="Shimla",
                                       origin="Shimla", rental_hours=8),
        ))

        assert result["entities"].visitor_type == RENTAL
        assert result["entities"].home_city == "Shimla"

    @pytest.mark.asyncio
    async def test_the_question_counters_reset(self):
        """Otherwise a rider who cancels after being asked twice for a pickup starts
        their next request with that question already exhausted."""
        result = await cancel_node(_state(
            "cancel it",
            entities=ExtractedEntities(visitor_type=RENTAL, rental_hours=8,
                                       asked_counts={"origin": 2},
                                       no_progress_turns=3, proceed_requested=True,
                                       last_asked_field="origin"),
        ))

        entities = result["entities"]
        assert entities.asked_counts == {}
        assert entities.no_progress_turns == 0
        assert entities.proceed_requested is False
        assert entities.last_asked_field is None
        assert result["last_asked_field"] is None

    @pytest.mark.asyncio
    async def test_the_event_names_what_was_dropped(self):
        payload = (await cancel_node(_state("cancel")))["structured_events"][0]["payload"]

        assert set(payload["cleared_fields"]) == {
            "origin", "rental_hours", "travel_dates", "pickup_time", "group_size",
        }

    @pytest.mark.asyncio
    async def test_the_next_request_starts_clean(self):
        """The point of clearing. A leftover pickup or duration is how a rider who
        cancelled and started again got a reply built half out of the abandoned one."""
        cancelled = (await cancel_node(_state("cancel it")))["entities"]

        assert not cancellation.has_request_details(cancelled)
        # …and nothing is in flight, so the next message is read as a fresh start
        # rather than as an answer to a question about a request that is over.
        assert cancelled.last_asked_field is None

    def test_cancel_is_never_read_as_an_answer(self):
        """Defence in depth for a phrasing both detectors miss. This is the mechanism
        that produced the original bug: `parse_place("cancel")` returned "Cancel", and
        the pickup city became a word the rider used to back out."""
        assert slots.parse_place("cancel") is None
        assert slots.parse_place("cancel karo") is None
        assert slots.parse_place("rehne do") is None
        merged: dict = {}
        assert slots.read_answer("origin", "cancel", merged, RENTAL) is None
        assert merged == {}
