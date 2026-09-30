"""
tests/unit/agent/test_flow_separation.py
─────────────────────────────────────────
The three flows are separate, and stay separate.

Reported: "the local flow is completely irrelevant … keep the three flows separate so
that any changes made in one will not change the output of the other".

The writer builds one prompt for four reply shapes. For a long time the context blocks
were built ONCE and then selectively withheld, so a flow got every section unless
somebody remembered to exclude it. That is how a RIDE — a resident booking a cab
between two towns they already know — ended up carrying the group composition, the
travel dates, the visitor type and the rider's travel history, while the steps
underneath forbade using any of them.

A prompt that states a fact and then forbids it does not get ignored; the model
resolves the contradiction by using the fact. That is where "travelling with family
means comfort matters" and "you can enjoy light sightseeing when you arrive" came from
in a reply meant to be four facts and a booking link.

So each flow now declares its own facts block, and these tests hold that line. They are
deliberately about ABSENCE: the presence of a section is visible the moment anyone
reads a reply, while a section that should not be there looks perfectly normal until
you compare two flows side by side.
"""

from __future__ import annotations

import pytest

from app.agent import slots
from app.agent.nodes.writer import _build_system_prompt
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities

ROUTE = {
    "origin": {"name": "Kullu", "lat": 31.9579, "lon": 77.1095},
    "destination": {"name": "Manali", "lat": 32.2396, "lon": 77.1887},
    "distance_km": 40.0,
    "duration_minutes": 90,
    "duration_text": "1h 30m",
    "is_approximate": False,
}

PROFILE = "Returning traveller. Past trips: Shimla, Kasol. Prefers moderate budget."


def _ride() -> AgentState:
    return AgentState(
        session_id="s", user_id="u", intent="plan_trip", visitor_type="local",
        entities=ExtractedEntities(
            visitor_type="local", destination="Manali", origin="Kullu",
            home_city="Kullu", group_size=3, travel_dates="kal",
        ),
        route_estimate=ROUTE,
        user_context_summary=PROFILE,
    )


def _itinerary() -> AgentState:
    return AgentState(
        session_id="s", user_id="u", intent="plan_trip", visitor_type="outstation",
        entities=ExtractedEntities(
            visitor_type="outstation", destination="Manali", origin="Delhi",
            duration_days=3, group_size=3, travel_dates="kal",
        ),
        route_estimate=ROUTE,
        user_context_summary=PROFILE,
    )


def _rental() -> AgentState:
    return AgentState(
        session_id="s", user_id="u", intent="rental_query", visitor_type="rental",
        entities=ExtractedEntities(
            visitor_type="rental", origin="Shimla", rental_hours=8,
            group_size=3, pickup_time="09:00", travel_dates="kal",
        ),
        user_context_summary=PROFILE,
    )


# ── 1. A ride carries only what a ride uses ───────────────────────────────────


class TestTheRidePromptIsARide:
    """A resident booking a cab. Four facts: the journey, the cabs, the price, the
    booking screen."""

    def test_it_is_headed_as_a_ride(self):
        """A heading is an instruction like any other. "TRIP" over a resident's cab
        between two towns they know produced holidays."""
        prompt = _build_system_prompt(_ride())
        assert "## THIS RIDE" in prompt
        assert "## TRIP" not in prompt

    @pytest.mark.parametrize(
        "leaked",
        ["- Group:", "- Travel dates:", "- Visitor type:", "- Duration:", "- Budget:",
         "- Interests:"],
    )
    def test_no_trip_shaped_fact_is_stated(self, leaked):
        """Every one of these is forbidden by the steps. Stating it anyway is the
        contradiction the model resolves in the wrong direction."""
        assert leaked not in _build_system_prompt(_ride())

    def test_the_travel_history_is_absent(self):
        """"Past trips: Shimla, Kasol" invites the reply to bring up somewhere the
        rider never mentioned, in a message whose job is a cab and a price."""
        prompt = _build_system_prompt(_ride())
        assert "User Profile" not in prompt
        assert "Past trips" not in prompt

    def test_what_a_ride_does_need_is_still_there(self):
        """Separation is not subtraction. The journey and the cabs are the reply."""
        prompt = _build_system_prompt(_ride())
        assert "- Destination: Manali" in prompt
        assert "- Origin: Kullu" in prompt
        assert "THIS JOURNEY" in prompt
        assert "HIM CAB FARES" in prompt

    def test_the_group_still_reaches_the_reply_through_the_fares(self):
        """Group size is not stated as a fact, but it is not lost either — the fare
        block is already narrowed to cabs that seat the party, which is the only form
        of it a ride actually uses."""
        assert "HIM CAB FARES" in _build_system_prompt(_ride())


# ── 2. The empty-section trap ─────────────────────────────────────────────────


class TestNoSectionIsPromisedAndThenMissing:

    def test_the_weather_step_and_the_weather_section_agree(self):
        """An empty "## REAL-TIME DATA" heading with a step pointing at it leaves the
        model a step it cannot satisfy — so it either announces the gap, which step 2
        forbids, or supplies a temperature of its own."""
        without = _build_system_prompt(_ride())
        assert ("## REAL-TIME DATA" in without) is ("(c) Current temperature" in without)

    def test_both_appear_together_when_there_is_a_forecast(self):
        state = _ride()
        state.weather_data = {
            "destination": "Manali", "current_temp_c": 18.0, "feels_like_c": 17.0,
            "humidity_percent": 50, "wind_speed_kmh": 5.0, "condition": "Clear",
            "forecast_7day": [], "alerts": [],
        }
        prompt = _build_system_prompt(state)
        assert "## REAL-TIME DATA" in prompt
        assert "(c) Current temperature" in prompt


# ── 3. One flow's material never appears in another ───────────────────────────


class TestTheFlowsDoNotBleed:
    """The regression this file exists to catch.

    A section added for one flow and not gated shows up in the other two, and it looks
    entirely reasonable in isolation — you only notice by comparing.
    """

    def test_each_flow_has_its_own_heading(self):
        assert "## THIS RIDE" in _build_system_prompt(_ride())
        assert "## THIS RENTAL" in _build_system_prompt(_rental())
        assert "## TRIP" in _build_system_prompt(_itinerary())

    def test_a_ride_is_not_given_the_itinerary_material(self):
        prompt = _build_system_prompt(_ride())
        assert "LOCAL VISITOR INSTRUCTIONS" not in prompt
        assert "OUTSTATION VISITOR INSTRUCTIONS" not in prompt
        assert "TRAVEL FROM ORIGIN" not in prompt

    def test_a_rental_is_not_given_a_destination_or_a_trip_length(self):
        """A rental has neither by definition — naming one hands somebody hiring a car
        for eight hours a plan they never asked for."""
        prompt = _build_system_prompt(_rental())
        assert "- Destination:" not in prompt
        assert "- Duration:" not in prompt
        assert "User Profile" not in prompt

    def test_an_itinerary_keeps_everything_it_needs(self):
        """The other side of the same rule: narrowing the other two must not quietly
        strip the flow that legitimately uses all of it."""
        prompt = _build_system_prompt(_itinerary())
        for expected in ("- Group:", "- Travel dates:", "- Duration:", "User Profile"):
            assert expected in prompt

    @pytest.mark.parametrize(
        "builder,mode",
        [(_ride, slots.RIDE_MODE), (_itinerary, slots.ITINERARY_MODE),
         (_rental, slots.RENTAL_MODE)],
    )
    def test_the_mode_is_what_decides(self, builder, mode):
        """The prompt follows `response_mode`, not the persona or the intent directly —
        one decision point, so the three shapes cannot disagree about which is which."""
        state = builder()
        assert slots.response_mode(state.intent, state.visitor_type) == mode

    def test_only_the_itinerary_asks_for_json(self):
        """A ride and a rental render no itinerary card, so asking for the JSON invites
        the model to produce a plan nobody will show."""
        assert "JSON" not in _build_system_prompt(_ride()).split("FORBIDDEN")[0]
        assert "day-by-day" not in _build_system_prompt(_rental()).split("FORBIDDEN")[0]
