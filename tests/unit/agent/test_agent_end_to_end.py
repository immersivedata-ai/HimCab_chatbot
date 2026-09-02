"""
tests/unit/test_agent_end_to_end.py
────────────────────────────────────
Whole-turn tests against the REAL compiled LangGraph, with every external service
stubbed. Where test_conversation_flow.py covers what the planner decides, these
cover what the user ends up receiving: which nodes ran, whether an itinerary was
emitted, and — the part no unit test can see — what the writer was actually told.

That last one matters most. "It assumes things and generates an itinerary" is a
failure of the writer's INPUT, not its output: a prompt that says `Duration: None`
or omits the fact that a detail was never given is a prompt that invites invention.
So these tests capture the system prompt and assert on it directly.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import get_agent_graph
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.user import UserProfile

ITINERARY_REPLY = """Assuming a 3-day trip — tell me if you'd like a different length.

**Day 1** — Arrive and settle in.
**Day 2** — Local sightseeing.

Where will you be starting from?

```json
{"destination": "Manali", "duration_days": 3, "group_type": "family",
 "budget_range": "moderate", "days": [{"day": 1, "theme": "Arrival",
 "activities": [{"time": "Morning", "place": "Mall Road", "type": "sightseeing"}]}]}
```"""

ANSWER_REPLY = "Manali is around 8C right now with clear skies.\n\nShall I plan the full trip?"


class AgentHarness:
    """Runs one turn through the real graph and records what the writer was told."""

    def __init__(self):
        self.writer_prompt = ""

    def _router_llm(self, intent, language):
        llm = MagicMock()
        response = MagicMock()
        response.content = json.dumps(
            {"intent": intent, "language": language, "confidence": 0.95}
        )
        llm.ainvoke = AsyncMock(return_value=response)
        return llm

    def _planner_llm(self, extraction):
        from app.agent.nodes.planner import PlannerExtraction

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction(**extraction)
        )
        return llm

    def _writer_llm(self, reply):
        llm = MagicMock()

        async def ainvoke(messages):
            self.writer_prompt = messages[0].content
            response = MagicMock()
            response.content = reply
            return response

        llm.ainvoke = AsyncMock(side_effect=ainvoke)
        return llm

    async def turn(
        self,
        message: str,
        *,
        intent: str,
        extraction: dict,
        reply: str = ITINERARY_REPLY,
        persona: str = "outstation",
        entities: ExtractedEntities | None = None,
        history: list | None = None,
        language: str = "en",
        weather=None,
        rider_jwt: str | None = None,
        fare_response=None,
    ) -> dict:
        profile_svc = MagicMock()
        profile_svc.get_or_create_profile = AsyncMock(
            return_value=UserProfile(user_id="u", firebase_uid="u", total_trips=0)
        )
        profile_svc.add_past_trip = AsyncMock()

        session_svc = MagicMock()
        session_svc.append_message = AsyncMock()
        session_svc.update_itinerary = AsyncMock()
        session_svc.update_entities = AsyncMock()

        weather_svc = MagicMock()
        weather_svc.get_weather = AsyncMock(return_value=weather)
        weather_svc.snowfall_from_report = MagicMock(return_value=None)

        # Stubbed at the HTTP boundary, so the real request building, parsing and
        # headline selection all run — only the network call is replaced.
        from app.integrations.himcab.fare_service import FareService

        fare_svc = FareService()
        if isinstance(fare_response, Exception):
            fare_svc._post = AsyncMock(side_effect=fare_response)
        else:
            fare_svc._post = AsyncMock(return_value=fare_response or {"data": []})

        messages = list(history or [])
        messages.append(HumanMessage(content=message))
        state = AgentState(
            session_id="s",
            user_id="u",
            messages=messages,
            entities=entities
            or ExtractedEntities(
                visitor_type=persona,
                last_asked_field="destination",
            ),
            visitor_type=persona,
            rider_jwt=rider_jwt,
        )

        with patch("app.agent.nodes.router.get_vertex_ai_service") as router_svc, \
             patch("app.agent.nodes.planner.get_vertex_ai_service") as planner_svc, \
             patch("app.agent.nodes.writer.get_vertex_ai_service") as writer_svc, \
             patch("app.agent.nodes.context.get_user_profile_service", lambda: profile_svc), \
             patch("app.agent.nodes.persist.get_session_service", lambda: session_svc), \
             patch("app.agent.nodes.persist.get_user_profile_service", lambda: profile_svc), \
             patch("app.agent.nodes.tool_executor.get_weather_service", lambda: weather_svc), \
             patch("app.agent.nodes.tool_executor.get_fare_service", lambda: fare_svc):
            router_svc.return_value.get_flash_model.return_value = self._router_llm(
                intent, language
            )
            planner_svc.return_value.get_flash_model.return_value = self._planner_llm(extraction)
            writer_svc.return_value.get_flash_model.return_value = self._writer_llm(reply)
            return await get_agent_graph().ainvoke(state)


@pytest.fixture
def agent():
    return AgentHarness()


def _event_types(final: dict) -> list[str]:
    return [e["type"] for e in final.get("structured_events", [])]


class TestWholeTurns:

    @pytest.mark.asyncio
    async def test_a_complete_request_returns_an_itinerary_in_one_turn(self, agent):
        final = await agent.turn(
            "Plan 3 days in Manali from Delhi for 4 of us next week",
            intent="plan_trip",
            extraction={
                "destination": "Manali", "origin": "Delhi", "duration_days": 3,
                "group_size": 4, "travel_dates": "next week", "origin_in_himachal": False,
            },
        )

        assert final["node_trace"] == [
            "router", "context", "planner", "tool_executor", "writer", "persist"
        ]
        assert final.get("structured_itinerary") is not None
        assert "itinerary" in _event_types(final)
        assert final.get("assumed_fields", []) == []
        assert "DETAILS THE USER NEVER GAVE" not in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_place_question_is_answered_without_an_itinerary(self, agent):
        final = await agent.turn(
            "what's the weather in Manali?",
            intent="weather_query",
            extraction={"destination": "Manali"},
            reply=ANSWER_REPLY,
        )

        assert final.get("structured_itinerary") is None
        assert "itinerary" not in _event_types(final)
        assert "ITINERARY JSON" not in agent.writer_prompt
        assert "Do NOT write a day-by-day itinerary" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_vague_opener_asks_and_never_reaches_the_writer(self, agent):
        final = await agent.turn("plan a trip", intent="plan_trip", extraction={})

        assert final["node_trace"] == ["router", "context", "planner", "clarify", "persist"]
        assert final["entities"].destination is None, "a destination was fabricated"
        assert final.get("structured_itinerary") is None
        assert final["final_response_text"].endswith("?")

    @pytest.mark.asyncio
    async def test_giving_up_on_a_detail_hands_the_writer_an_explicit_assumption(self, agent):
        """The core of the "it assumes things" fix.

        The user has refused twice, so the agent proceeds — but the writer is told, in
        so many words, which facts it does not have, what to substitute, and that it
        must open by saying so. It is never left to fill the gaps itself.
        """
        final = await agent.turn(
            "just make it",
            intent="plan_trip",
            extraction={"destination": "Manali", "user_wants_to_proceed_now": True},
            entities=ExtractedEntities(
                destination="Manali", visitor_type="outstation",
                last_asked_field="origin",
                asked_counts={"origin": 2, "duration_days": 2, "group_size": 2,
                              "travel_dates": 2},
                no_progress_turns=1,
            ),
            history=[
                AIMessage(content="Where would you like to go?"),
                HumanMessage(content="Manali"),
                AIMessage(content="Which city are you starting from, and how many days?"),
                HumanMessage(content="hmm"),
                AIMessage(content="Which city are you starting from, and how many days?"),
            ],
        )

        assert final.get("structured_itinerary") is not None
        assert set(final["assumed_fields"]) == {
            "origin", "duration_days", "group_size", "travel_dates"
        }

        prompt = agent.writer_prompt
        assert "DETAILS THE USER NEVER GAVE" in prompt
        assert "Do NOT invent a starting city" in prompt
        assert "Plan for a 3-day trip" in prompt
        assert "START your reply with ONE short sentence naming what you assumed" in prompt
        # …and the JSON the app renders carries the same number the text discloses.
        assert final["structured_itinerary"].duration_days == 3

    @pytest.mark.asyncio
    async def test_unknown_facts_never_reach_the_model_as_the_word_none(self, agent):
        """`Duration: None days` is an invitation to invent one — and an unknown
        destination used to crash the prompt builder outright."""
        final = await agent.turn(
            "just plan something",
            intent="plan_trip",
            extraction={"destination": "Manali", "user_wants_to_proceed_now": True},
            entities=ExtractedEntities(
                destination="Manali", visitor_type="outstation",
                asked_counts={"origin": 2, "duration_days": 2, "group_size": 2,
                              "travel_dates": 2},
            ),
        )

        facts = agent.writer_prompt.split("## REAL-TIME DATA")[0]
        assert "None" not in facts, facts
        assert final.get("structured_itinerary") is not None

    @pytest.mark.asyncio
    async def test_the_itinerary_carries_the_real_weather(self, agent):
        """`Itinerary.weather` depended on the model copying the forecast into its JSON
        block, which it routinely skipped — so the app rendered a plan with no weather
        even though it had been fetched. It is stamped on from the report now."""
        report = MagicMock()
        report.model_dump.return_value = {
            "destination": "manali",
            "current_temp_c": 8.2,
            "condition": "light snow",
            "forecast_7day": [
                {"min_c": -1.0, "max_c": 9.0}, {"min_c": 0.0, "max_c": 11.0},
                {"min_c": 2.0, "max_c": 12.0},
            ],
            "alerts": [{"severity": "high", "description": "Heavy snow expected"}],
        }
        final = await agent.turn(
            "Plan 3 days in Manali from Delhi for 4 of us next week",
            intent="plan_trip",
            extraction={
                "destination": "Manali", "origin": "Delhi", "duration_days": 3,
                "group_size": 4, "travel_dates": "next week", "origin_in_himachal": False,
            },
            weather=report,
        )

        weather = final["structured_itinerary"].weather
        assert weather is not None, "itinerary shipped with no weather"
        assert "8" in weather.summary and "light snow" in weather.summary
        assert (weather.temperature_min_c, weather.temperature_max_c) == (-1.0, 12.0)
        assert weather.alert == "Heavy snow expected"
        # …and the model is told to put it in the text, not left to decide.
        assert "include a short 'Weather' line" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_the_closing_question_never_asks_for_trip_details(self, agent):
        """The reported bug: the reply ended by asking how many days and how many
        people — after an itinerary that had already been written without them."""
        final = await agent.turn(
            "Delhi to Manali, 3 days, 4 of us, next week",
            intent="plan_trip",
            extraction={
                "destination": "Manali", "origin": "Delhi", "duration_days": 3,
                "group_size": 4, "travel_dates": "next week", "origin_in_himachal": False,
            },
        )

        assert final.get("structured_itinerary") is not None
        assert "Do NOT ask for trip details" in agent.writer_prompt
        assert "moves toward a booking" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_the_itinerary_carries_a_him_cab_fare(self, agent):
        """The reported failure: itineraries arrived with no Him Cab price.

        Runs the whole turn with a stubbed backend and asserts the fare survives all
        the way to the prompt AND to the JSON the app renders — the two places the
        rider can actually see it.
        """
        final = await agent.turn(
            "Gurgaon to Manali, 3 days, 4 of us, next week",
            intent="plan_trip",
            extraction={
                "destination": "Manali", "origin": "Gurgaon", "duration_days": 3,
                "group_size": 4, "travel_dates": "next week", "origin_in_himachal": False,
            },
            rider_jwt="rider-token",
            fare_response={
                "data": [
                    {"categoryId": 1, "name": "Economy", "totalFare": 8400, "seatCapacity": 4},
                    {"categoryId": 2, "name": "Standard", "totalFare": 9600, "seatCapacity": 4},
                    {"categoryId": 3, "name": "Premium", "totalFare": 14200, "seatCapacity": 6},
                ]
            },
        )

        fare_events = [e for e in final.get("structured_events", []) if e["type"] == "cab_fare"]
        assert fare_events, "no cab_fare card for the app to render"
        assert len(fare_events[0]["payload"]["options"]) == 3

        # Gurgaon was in no surveyed table, so this is the path that used to yield
        # nothing at all. The distance now comes from the routing engine.
        assert fare_events[0]["payload"]["distance_km"] > 0
        assert "9,600" in agent.writer_prompt

        # The price also lands on the itinerary the app renders, not just in the text.
        cab_package = final["structured_itinerary"].cab_package
        assert cab_package is not None and cab_package.estimated_cost_inr == 9600

    @pytest.mark.asyncio
    async def test_a_fare_failure_never_blocks_the_itinerary(self, agent):
        """A dead fare backend must cost the price, not the plan."""
        final = await agent.turn(
            "Delhi to Manali, 3 days, 4 of us, next week",
            intent="plan_trip",
            extraction={
                "destination": "Manali", "origin": "Delhi", "duration_days": 3,
                "group_size": 4, "travel_dates": "next week", "origin_in_himachal": False,
            },
            rider_jwt="rider-token",
            fare_response=RuntimeError("fare backend down"),
        )

        assert final.get("structured_itinerary") is not None
        assert "cab_fare" not in _event_types(final)
        assert "do NOT state, estimate or imply any price" in agent.writer_prompt
        assert "Book Your Him Cab" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_bare_answer_is_not_treated_as_an_open_question(self, agent):
        """"delhi" classifies as general_query on its own; with our question in flight
        it must continue the trip rather than being answered as a topic of its own."""
        final = await agent.turn(
            "delhi",
            intent="general_query",
            extraction={"destination": "Manali", "origin": "Delhi", "origin_in_himachal": False},
            entities=ExtractedEntities(
                destination="Manali", visitor_type="outstation",
                last_asked_field="origin",
                asked_counts={"origin": 1, "duration_days": 1},
            ),
            history=[
                AIMessage(content="Which city are you starting from, and how many days?"),
            ],
        )

        assert final["intent"] == "clarification"
        assert final["entities"].origin == "Delhi"


# ── The local rider flow ──────────────────────────────────────────────────────

LOCAL_FARE_RESPONSE = {"data": [
    {"categoryId": 1, "name": "Economy", "totalFare": 3200, "seatCapacity": None},
    {"categoryId": 2, "name": "Sedan", "totalFare": 3900, "seatCapacity": None},
    {"categoryId": 3, "name": "SUV", "totalFare": 5400, "seatCapacity": None},
    {"categoryId": 4, "name": "Innova Crysta", "totalFare": 6800, "seatCapacity": None},
    {"categoryId": 5, "name": "Tempo Traveller", "totalFare": 11000, "seatCapacity": None},
    {"categoryId": 6, "name": "Mini Bus", "totalFare": 17500, "seatCapacity": None},
]}


def _local_weather():
    report = MagicMock()
    report.model_dump.return_value = {
        "destination": "manali", "current_temp_c": 9.4, "condition": "light snow",
        "forecast_7day": [{"min_c": -1.0, "max_c": 10.0}, {"min_c": 0.0, "max_c": 11.0}],
        "alerts": [],
    }
    return report


class TestLocalRiderGetsACabNotAnItinerary:
    """A Himachal resident asking to go to Manali is booking a ride, not planning a
    holiday in a place they already know. The reply is the cab, the fare, the drive
    time and the weather — then a handoff to the app's booking screen."""

    async def _local_turn(self, agent, **overrides):
        kwargs = {
            "message": "Shimla se Manali, 4 log, kal",
            "intent": "plan_trip",
            "persona": "local",
            "rider_jwt": "rider-token",
            "fare_response": LOCAL_FARE_RESPONSE,
            "weather": _local_weather(),
            "reply": "<ride reply>",
            "extraction": {
                "destination": "Manali", "origin": "Shimla", "home_city": "Shimla",
                "group_size": 4, "travel_dates": "kal", "duration_days": 2, "origin_in_himachal": True,
            },
        }
        kwargs.update(overrides)
        return await agent.turn(**kwargs)

    @pytest.mark.asyncio
    async def test_no_itinerary_is_produced_for_a_local(self, agent):
        final = await self._local_turn(agent)

        assert final.get("structured_itinerary") is None
        assert "itinerary" not in _event_types(final)
        assert "FORBIDDEN" in agent.writer_prompt
        assert "day-by-day plan" in agent.writer_prompt
        assert "ITINERARY JSON" not in agent.writer_prompt
        # The sightseeing brief that used to be added for locals is gone.
        assert "LOCAL VISITOR INSTRUCTIONS" not in agent.writer_prompt
        # The arrival-day brief is itinerary guidance; in a cab reply it was what
        # produced "you can enjoy light sightseeing when you arrive".
        assert "Day 1 must begin" not in agent.writer_prompt
        assert "travel fatigue" not in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_only_cabs_that_seat_the_group_reach_the_rider(self, agent):
        """The reported symptom: a Mini Bus recommended to four travellers."""
        final = await self._local_turn(agent)

        fare = next(e for e in final["structured_events"] if e["type"] == "cab_fare")
        offered = [o["category_name"] for o in fare["payload"]["options"]]

        assert offered == ["Economy", "Sedan", "SUV"]
        assert "Mini Bus" not in agent.writer_prompt
        assert "Tempo Traveller" not in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_the_reply_carries_the_drive_time_and_temperature(self, agent):
        final = await self._local_turn(agent)

        route = final["route_estimate"]
        assert "THIS JOURNEY" in agent.writer_prompt
        # Quoted from the resolved route, whatever the engine returned for it.
        assert route["duration_text"] in agent.writer_prompt
        assert f"{round(route['distance_km'])} km" in agent.writer_prompt
        assert "9.4" in agent.writer_prompt             # current temperature
        assert route["distance_km"] > 0

    @pytest.mark.asyncio
    async def test_the_rider_is_handed_to_the_booking_screen_with_map_pins(self, agent):
        """The app needs coordinates, not names — re-geocoding "Kasol" could land
        somewhere other than the point this fare was priced on."""
        final = await self._local_turn(agent)

        handoff = next(e for e in final["structured_events"] if e["type"] == "booking_handoff")
        payload = handoff["payload"]

        assert payload["origin"]["name"] == "Shimla"
        assert payload["destination"]["name"] == "Manali"
        for end in ("origin", "destination"):
            assert payload[end]["lat"] and payload[end]["lon"], f"{end} has no map pin"
        assert payload["passengers"] == 4
        assert payload["recommended_category_id"] is not None
        assert payload["duration_text"] == "4h 30m"

    @pytest.mark.asyncio
    async def test_the_reply_never_asks_the_rider_to_confirm_a_booking(self, agent):
        """The app owns the booking flow — the assistant only opens the door to it."""
        final = await self._local_turn(agent)

        assert "Do not ask whether they want to book" in agent.writer_prompt
        assert "shall I book" not in agent.writer_prompt.lower()
        assert final["structured_events"]

    @pytest.mark.asyncio
    async def test_a_local_is_still_asked_for_the_missing_details(self, agent):
        """"take all the questions as you were taking" — the ride mode changes the
        answer, not the interview."""
        final = await agent.turn(
            "Manali",
            intent="plan_trip",
            persona="local",
            rider_jwt="rider-token",
            extraction={"destination": "Manali", "origin_in_himachal": None},
        )

        assert final["node_trace"] == ["router", "context", "planner", "clarify", "persist"]
        assert final["final_response_text"].endswith("?")


class TestOutstationIsUnaffected:
    """The ride mode is scoped to residents; a visitor's itinerary must be untouched."""

    @pytest.mark.asyncio
    async def test_an_outstation_visitor_still_gets_a_full_itinerary(self, agent):
        final = await agent.turn(
            "Delhi to Manali, 4 of us, 3 days, next week",
            intent="plan_trip",
            persona="outstation",
            rider_jwt="rider-token",
            fare_response=LOCAL_FARE_RESPONSE,
            extraction={
                "destination": "Manali", "origin": "Delhi", "group_size": 4, "duration_days": 3,
                "travel_dates": "next week", "origin_in_himachal": False,
            },
        )

        assert final.get("structured_itinerary") is not None
        assert "Give a day-by-day itinerary" in agent.writer_prompt
        assert "ITINERARY JSON" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_visitor_is_handed_to_the_outstation_booking_screen(self, agent):
        """The plan is worth nothing until somebody drives them to it, so the arrival
        transfer gets the same Book Now card a resident's cab does — routed by the
        journey's length to the screen that prices an intercity transfer."""
        final = await agent.turn(
            "Delhi to Manali, 4 of us, 3 days, next week",
            intent="plan_trip",
            persona="outstation",
            rider_jwt="rider-token",
            fare_response=LOCAL_FARE_RESPONSE,
            extraction={
                "destination": "Manali", "origin": "Delhi", "group_size": 4, "duration_days": 3,
                "travel_dates": "next week", "origin_in_himachal": False,
            },
        )

        handoff = next(e for e in final["structured_events"] if e["type"] == "booking_handoff")
        payload = handoff["payload"]

        assert payload["origin"]["name"] == "Delhi"
        assert payload["destination"]["name"] == "Manali"
        for end in ("origin", "destination"):
            assert payload[end]["lat"] and payload[end]["lon"], f"{end} has no map pin"
        # 500+ km — the app must open its outstation screen, not the local one.
        assert payload["trip_type"] == "OUTSTATION"
        assert payload["passengers"] == 4
        # …and the reply says so rather than asking them to type it all again.
        assert "pickup and drop are already filled in" in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_visitor_with_no_starting_city_gets_no_card(self, agent):
        """Half a journey is not a booking screen: with no pickup there is nothing to
        open on, so the itinerary ships without the card rather than with a blank one."""
        final = await agent.turn(
            "3 days in Manali for 4 of us next week",
            intent="plan_trip",
            persona="outstation",
            rider_jwt="rider-token",
            fare_response=LOCAL_FARE_RESPONSE,
            extraction={
                "destination": "Manali", "group_size": 4, "duration_days": 3,
                "travel_dates": "next week", "origin_in_himachal": False,
                "user_wants_to_proceed_now": True,
            },
        )

        assert "booking_handoff" not in _event_types(final)
        assert "pickup and drop are already filled in" not in agent.writer_prompt

    @pytest.mark.asyncio
    async def test_a_visitor_is_also_spared_the_oversized_cab(self, agent):
        """Seat filtering is a plain bug fix, so it applies to both personas."""
        final = await agent.turn(
            "Delhi to Manali, 4 of us, 3 days, next week",
            intent="plan_trip",
            persona="outstation",
            rider_jwt="rider-token",
            fare_response=LOCAL_FARE_RESPONSE,
            extraction={
                "destination": "Manali", "origin": "Delhi", "group_size": 4, "duration_days": 3,
                "travel_dates": "next week", "origin_in_himachal": False,
            },
        )

        fare = next(e for e in final["structured_events"] if e["type"] == "cab_fare")
        assert not any("Bus" in o["category_name"] for o in fare["payload"]["options"])
