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
from tests.conftest import stub_vertex

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
        from tests.conftest import wire_extraction

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=wire_extraction(**extraction)
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
        # Kept so a test can prove the fare backend was never ASKED, not merely that
        # its answer went unused — a local ride and a trip plan name no cab.
        self.fare_post = fare_svc._post

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
            stub_vertex(router_svc, self._router_llm(intent, language))
            stub_vertex(planner_svc, self._planner_llm(extraction))
            stub_vertex(writer_svc, self._writer_llm(reply))
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
    async def test_the_itinerary_names_no_cab_and_asks_for_no_fare(self, agent):
        """Only a rental recommends a cab (`slots.lists_cabs`, a 2026-09-28 decision).

        This used to assert the opposite — the fare in the prompt, a `cab_fare` card and
        a priced `cab_package` on the itinerary. A visitor now picks their cab on the
        app's booking screen, so the backend is not asked, and nothing the rider sees
        names a cab: not the prompt, not the events, not the itinerary card.
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

        assert agent.fare_post.await_count == 0, "a trip plan asked for a cab fare"
        assert "cab_fare" not in _event_types(final)
        assert "9,600" not in agent.writer_prompt
        assert "chooses their cab on the booking screen" in agent.writer_prompt
        assert final["structured_itinerary"].cab_package is None

        # The journey itself is still measured — Gurgaon was in no surveyed table, and
        # the distance comes from the routing engine whether or not anything is priced.
        assert final["route_estimate"]["distance_km"] > 0

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
    """A resident asking to go to Manali is booking a ride, not planning a holiday in a
    place they already know. The reply is the drive time, the day and the weather —
    then a handoff to the app's booking screen, where they choose the cab."""

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

        # Asserted of the REPLY, not of the prompt. A ride reply is assembled rather
        # than generated (app/agent/ride_reply.py), so there is no prompt to inspect —
        # and checking the text itself is the stronger test anyway: the old version
        # only proved we had TOLD the model not to write a brochure, which it did
        # regardless, recommending Mall Road and the Ridge off its own knowledge.
        reply = final["final_response_text"].lower()
        for brochure in ("mall road", "ridge", "jakhu", "kufri", "sightseeing",
                         "hidden gem", "day 1", "itinerary"):
            assert brochure not in reply, f"the ride reply reads like a brochure: {brochure!r}"

    @pytest.mark.asyncio
    async def test_no_cab_is_recommended_to_a_local_rider(self, agent):
        """Only a rental recommends a cab (`slots.lists_cabs`, a 2026-09-28 decision).

        This test used to require the opposite — every cab that seats four, by name,
        with its price. The ride now ends at the booking screen, which lists the cabs
        and their fares itself, so the backend is not asked and no cab is named.
        """
        final = await self._local_turn(agent)

        assert agent.fare_post.await_count == 0, "a local ride asked for a cab fare"
        assert "cab_fare" not in _event_types(final)
        reply = final["final_response_text"]
        for cab in ("Economy", "Sedan", "SUV", "Innova", "Tempo Traveller", "Mini Bus", "₹"):
            assert cab not in reply, f"the local reply still recommends a cab: {cab!r}"
        assert "गाड़ी चुनें" in reply or "Choose your cab" in reply

    @pytest.mark.asyncio
    async def test_the_reply_carries_the_drive_time_and_temperature(self, agent):
        final = await self._local_turn(agent)

        route = final["route_estimate"]
        reply = final["final_response_text"]

        # Copied from the resolved route, whatever the engine returned for it — and
        # now verifiable in the reply itself rather than in an instruction. The unit is
        # asserted in either script: a local rider is answered in Hindi, so the reply
        # reads "202 किमी" rather than "202 km".
        assert route["duration_text"] in reply
        assert str(round(route["distance_km"])) in reply
        assert ("km" in reply or "किमी" in reply)
        assert "9.4" in reply                            # current temperature
        assert route["distance_km"] > 0

        # The forecast and the snowfall RISK are real data that belong in an itinerary,
        # where somebody is choosing what to do with four days. Here they were what
        # produced a seven-day outlook and a "snowfall risk" line in a reply meant to
        # be a cab and a price.
        #
        # The current CONDITION is not that, and may well be "light snow" — it is one
        # of the four facts a ride reply is for. So the check is on the forecast
        # wording, in both scripts, and on the reply staying short enough that a
        # multi-day outlook could not fit in it.
        assert "forecast" not in reply.lower()
        assert "भविष्यवाणी" not in reply          # "forecast"
        assert "जोखिम" not in reply               # "risk"
        assert len(reply.splitlines()) <= 8, f"the ride reply has grown: {reply!r}"

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
        # Nothing preselected: the rider chooses the cab on the booking screen.
        assert payload["recommended_category_id"] is None
        assert payload["recommended_category_name"] is None
        assert payload["duration_text"] == "4h 30m"

    @pytest.mark.asyncio
    async def test_the_reply_never_asks_the_rider_to_confirm_a_booking(self, agent):
        """The app owns the booking flow — the assistant only opens the door to it."""
        final = await self._local_turn(agent)

        reply = final["final_response_text"]

        # The app owns booking; this reply only opens the door. It is a statement, and
        # a statement cannot drift into "shall I book?" or into offering to plan a trip
        # the way a generated one did.
        assert "?" not in reply, f"the ride reply asks a question: {reply!r}"
        for asked in ("shall i book", "would you like", "do you want", "plan your"):
            assert asked not in reply.lower()
        assert (
            "booking screen" in reply.lower()
            or "बुकिंग स्क्रीन" in reply
        ), f"the reply does not point at the booking screen: {reply!r}"
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
    async def test_a_visitor_is_recommended_no_cab_either(self, agent):
        """Only a rental recommends a cab — a visitor chooses on the booking screen."""
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

        assert agent.fare_post.await_count == 0
        assert "cab_fare" not in _event_types(final)
        handoff = next(e for e in final["structured_events"] if e["type"] == "booking_handoff")
        assert handoff["payload"]["recommended_category_id"] is None
        assert handoff["payload"]["options"] == []
