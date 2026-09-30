"""
The rental booking screen opens with everything the rider already told us.

Reported as: "the rider is not getting cabs recommended for their passengers and hours,
and on the booking screen they have to enter the cab and the hours again to get a
price." Driving the real planner, tool executor and handoff builder showed three
separate causes, each covered below:

  1. ACCEPTING the recommendation was not an answer. The cab list led with the cab that
     seats the party, but "haan book kar do" or "ok" to "which cab?" came back as "Sorry,
     I didn't quite catch that" — and no booking handoff was sent until the rider typed
     a cab name, so the app's screen opened empty.
  2. The DATE was corrupted from the second turn on. `tidy_when` re-read the stored ISO
     date as a sentence and cut "2026-09-29" to "09-29".
  3. A CALENDAR date ("15 Oct") went to the booking screen's Date field as words.

The fare backend is stubbed at its HTTP boundary with the row shape
`/api/rider/fare/estimate` returns, so the real parser and seat filter run.
"""

import json
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from app.agent import sanitize, slots
from app.agent.nodes.tool_executor import tool_executor_node
from app.agent.nodes.writer import _build_booking_handoff, _build_system_prompt
from app.agent.state import AgentState
from app.core.config import settings
from app.integrations.himcab.fare_service import FareService
from tests.unit.agent.test_conversation_flow import NOTHING, Conversation, extraction

TODAY = date(2026, 9, 21)

#: What POST /api/rider/fare/estimate returns: one row per category, `name` not
#: `categoryName`, and `seatCapacity` sometimes a range string.
ESTIMATE = {"data": [
    {"categoryId": 1, "name": "Economy", "totalFare": 2100, "seatCapacity": "4"},
    {"categoryId": 2, "name": "Sedan", "totalFare": 2400, "seatCapacity": "4"},
    {"categoryId": 3, "name": "SUV", "totalFare": 3300, "seatCapacity": "6 - 7"},
    {"categoryId": 4, "name": "Tempo Traveller", "totalFare": 5200, "seatCapacity": "12"},
]}

#: The rider's own sentence. It names a drop, which a rental does not take (it has only a
#: pickup, a 2026-09-30 decision) — the model reads it anyway, as it does in production,
#: so this also proves the drop never reaches the booking screen.
FIRST_MESSAGE = (
    "I need a car for 8 hours from Shimla tomorrow at 9 am, we are 5 people, "
    "drop at Kufri"
)
FIRST_EXTRACTION = extraction(
    rental_hours=8, origin="Shimla", destination="Kufri", travel_dates="tomorrow",
    pickup_time="09:00", group_size=5,
)
#: "tomorrow", resolved against the pinned `TODAY` below.
TOMORROW = "2026-09-22"


@pytest.fixture(autouse=True)
def _pinned_today(monkeypatch):
    """Resolve "tomorrow" against a fixed day (§5: never `date.today()`).

    These tests first asserted a hardcoded "2026-09-29" while resolving "tomorrow"
    against the real calendar, and failed the day the calendar moved on.
    """
    monkeypatch.setattr(sanitize, "today_ist", lambda: TODAY)


async def _tools(chat: Conversation, accepted: bool = False) -> tuple[dict, AgentState]:
    """Run the real tool executor on the conversation's entities; return (result, state)."""
    fare = FareService()
    fare._post = AsyncMock(return_value=ESTIMATE)
    state = AgentState(
        session_id="s", user_id="u", intent="rental_query", visitor_type="rental",
        rider_jwt="rider-token", entities=chat.entities, messages=list(chat.messages),
        accepted_recommended_cab=accepted,
    )
    with patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
         patch.object(settings, "enable_real_fares", True), \
         patch.object(settings, "enable_offers", False):
        result = await tool_executor_node(state)
    after = state.model_copy(update={k: v for k, v in result.items() if k != "node_trace"})
    if result.get("entities") is not None:
        chat.entities = result["entities"]
    return result, after


async def _quoted_conversation() -> Conversation:
    """A rental that has just been priced and asked "which cab?"."""
    chat = Conversation(persona="rental")
    await chat.say(FIRST_MESSAGE, FIRST_EXTRACTION, intent="rental_query")
    await _tools(chat)
    chat.messages[-1] = AIMessage(content="Which cab would you like? SUV Rs 3,300 ...")
    return chat


class TestTheCabIsRecommendedForTheParty:
    @pytest.mark.asyncio
    async def test_five_people_are_offered_only_cabs_that_seat_five(self):
        chat = Conversation(persona="rental")
        await chat.say(FIRST_MESSAGE, FIRST_EXTRACTION, intent="rental_query")
        result, _ = await _tools(chat)

        cab_fare = result["cab_fare"]
        assert cab_fare["category_name"] == "SUV"
        assert [o["category_name"] for o in cab_fare["options"]] == ["SUV", "Tempo Traveller"]
        assert result["awaiting_cab_choice"] is True

    @pytest.mark.asyncio
    async def test_the_reply_names_the_recommendation_and_the_party_it_seats(self):
        chat = Conversation(persona="rental")
        await chat.say(FIRST_MESSAGE, FIRST_EXTRACTION, intent="rental_query")
        _, state = await _tools(chat)

        prompt = _build_system_prompt(state)
        assert "for 5 passengers" in prompt
        assert "replying yes books the recommended one" in prompt


class TestTakingTheRecommendationIsAnAnswer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", ["haan book kar do", "ok", "yes please", "ठीक है", "aap batao"])
    async def test_it_books_the_recommended_cab_with_every_detail_filled_in(self, reply):
        chat = await _quoted_conversation()
        turn = await chat.say(reply, NOTHING, intent="rental_query")

        assert turn.produced, f"{reply!r} was re-asked instead of accepted"
        assert turn.result["accepted_recommended_cab"] is True

        _, state = await _tools(chat, accepted=True)
        handoff = _build_booking_handoff(state)

        assert handoff is not None, "no booking screen after the rider said yes"
        assert handoff["recommended_category_id"] == 3
        assert handoff["recommended_category_name"] == "SUV"
        assert handoff["category_selected_by_rider"] is True
        assert handoff["rental_hours"] == 8
        assert handoff["fare_inr"] == 3300
        assert handoff["passengers"] == 5
        assert handoff["pickup_time"] == "09:00"
        assert handoff["travel_dates"] == TOMORROW
        assert handoff["origin"]["lat"] and handoff["origin"]["lon"]
        # A rental has one place. The drop they named is not carried: the Rent a Car
        # screen has no field for it.
        assert handoff["destination"] is None
        # The choice is stored, so the next turn is still booking the same cab.
        assert chat.entities.cab_category == "suv"

    @pytest.mark.asyncio
    async def test_naming_a_cab_still_wins_over_the_recommendation(self):
        chat = await _quoted_conversation()
        await chat.say("tempo traveller", extraction(cab_category="tempo traveller"),
                       intent="rental_query")
        _, state = await _tools(chat)

        assert _build_booking_handoff(state)["recommended_category_name"] == "Tempo Traveller"

    @pytest.mark.asyncio
    async def test_a_yes_to_some_other_question_is_not_a_cab_choice(self):
        chat = Conversation(persona="rental")
        turn = await chat.say("ok", NOTHING, intent="rental_query")
        assert not turn.result.get("accepted_recommended_cab")


class TestWhatCountsAsAcceptingIt:
    @pytest.mark.parametrize("text", [
        "haan", "ok", "okay book it", "yes", "haan book kar do", "theek hai",
        "chalega", "wahi wala", "ठीक है", "हाँ बुक कर दो", "go ahead with that one",
    ])
    def test_agreement(self, text):
        assert slots.accepts_recommendation(text)

    @pytest.mark.parametrize("text", [
        "sedan", "haan but sedan", "is the suv available?", "kasol",
        "8 hours", "nahi", "", "book a sedan",
    ])
    def test_anything_carrying_content_or_a_question_is_not(self, text):
        assert not slots.accepts_recommendation(text)


class TestTheDateSurvivesEveryTurn:
    def test_a_resolved_date_is_not_read_as_a_sentence_again(self):
        assert slots.tidy_when("2026-09-29") == "2026-09-29"

    @pytest.mark.asyncio
    async def test_the_second_turn_does_not_cut_the_year_off(self):
        chat = await _quoted_conversation()
        assert chat.entities.travel_dates == TOMORROW
        await chat.say("haan", NOTHING, intent="rental_query")
        assert chat.entities.travel_dates == TOMORROW


class TestACalendarDateReachesTheBookingScreenAsADate:
    @pytest.mark.parametrize("text", ["15 Oct", "Oct 15", "15th October", "15/10", "15-10-2026"])
    def test_it_is_resolved(self, text):
        assert sanitize.travel_date(text, as_iso=True, today=TODAY) == "2026-10-15"

    def test_a_month_already_behind_us_means_next_year(self):
        assert sanitize.travel_date("15 Jan", as_iso=True, today=TODAY) == "2027-01-15"

    def test_a_day_that_went_by_this_month_is_refused_by_name(self):
        assert sanitize.travel_date("5 Sep", as_iso=True, today=TODAY) is None
        assert sanitize.day_already_gone("5 Sep", today=TODAY) == "5 Sep"

    @pytest.mark.parametrize("text", ["31/02", "next week", "after Diwali"])
    def test_a_phrase_naming_no_real_day_is_left_in_the_riders_words(self, text):
        assert sanitize.travel_date(text, as_iso=True, today=TODAY) == text

    def test_a_trip_keeps_the_riders_words(self):
        assert sanitize.travel_date("15 Oct", today=TODAY) == "15 Oct"


def test_the_handoff_is_serialisable_for_the_app():
    """The event goes over SSE as JSON — nothing in it may be a Python-only type."""
    json.dumps({"rental_hours": 8, "travel_dates": sanitize.travel_date(
        "15 Oct", as_iso=True, today=TODAY)})
