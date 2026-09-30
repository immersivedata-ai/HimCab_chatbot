"""
Only a rental recommends a cab.

A product decision (2026-09-28): a local ride and an outstation trip hand the rider to
the app's booking screen, which lists the cabs and their fares itself, so the chat
names no cab on those flows. A rental keeps its recommendation, because its booking
screen will not price anything until a cab is chosen (see
`test_rental_booking_handoff.py` for that half).

The rule lives in one table, `slots.lists_cabs`, and the tool executor follows it by
never asking for a fare. Each test below covers a different way a cab could still
reach a local or outstation rider: the policy itself, the prompt, a model that names a
vehicle anyway, and a model that writes a cab package into the itinerary JSON.
"""

import pytest

from app.agent import slots
from app.agent.guardrails import apply_guardrails
from app.agent.nodes.writer import _build_system_prompt, _normalize_itinerary_dict
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities


def _trip_state(visitor_type: str = "outstation", **overrides) -> AgentState:
    entities = ExtractedEntities(
        destination="Manali", origin="Delhi", group_size=4, duration_days=3,
        visitor_type=visitor_type,
        home_city="Shimla" if visitor_type == "local" else None,
    )
    return AgentState(
        session_id="s", user_id="u", intent="plan_trip", visitor_type=visitor_type,
        entities=entities, **overrides,
    )


class TestThePolicy:
    def test_only_the_rental_reply_lists_cabs(self):
        assert slots.lists_cabs(slots.RENTAL_MODE)
        for mode in (slots.RIDE_MODE, slots.ITINERARY_MODE, slots.ANSWER_MODE,
                     slots.CANCELLED_MODE):
            assert not slots.lists_cabs(mode), f"{mode} would list cabs"

    def test_it_is_keyed_on_the_flow_not_the_intent(self):
        """The router calls half of a rental chat `plan_trip`; the page decides."""
        assert slots.lists_cabs(slots.response_mode("plan_trip", slots.RENTAL))
        assert not slots.lists_cabs(slots.response_mode("plan_trip", slots.LOCAL))
        assert not slots.lists_cabs(slots.response_mode("plan_trip", slots.OUTSTATION))


class TestAnOutstationReplyNamesNoCab:
    def test_the_prompt_offers_no_cab_to_recommend(self):
        prompt = _build_system_prompt(_trip_state())

        assert "recommended for this group" not in prompt
        assert "chooses their cab on the booking screen" in prompt
        assert "Name NO cab and NO vehicle" in prompt

    def test_a_vehicle_the_model_names_anyway_is_taken_out(self):
        state = _trip_state()
        text = (
            "Day 1: drive up from Delhi.\n"
            "For your group of 4 we recommend an Innova Crysta.\n"
            "Choose your cab on the booking screen."
        )
        cleaned, findings = apply_guardrails(state, text, slots.ITINERARY_MODE, "en")

        assert "Innova" not in cleaned
        assert "Day 1" in cleaned
        assert any(f.kind == "invented_cab" for f in findings)

    def test_a_cab_package_the_model_writes_into_the_json_is_dropped(self):
        normalized = _normalize_itinerary_dict(
            {
                "destination": "Manali",
                "cab_package": {"vehicle": "Innova Crysta", "estimated_cost_inr": 8000},
                "cost_breakdown": {"cab_inr": 8000, "total_inr": 8000},
            },
            _trip_state(),
        )
        assert "cab_package" not in normalized
        assert "cost_breakdown" not in normalized

    def test_a_real_rental_fare_would_still_keep_them(self):
        """The drop is for a cab nobody priced — not for one the backend quoted."""
        state = _trip_state(cab_fare={"fare_inr": 3300, "options": []})
        normalized = _normalize_itinerary_dict(
            {"destination": "Manali", "cab_package": {"vehicle": "SUV"}}, state
        )
        assert "cab_package" in normalized


class TestTheGroupSizeQuestion:
    def test_hindi_asks_who_is_with_them(self):
        first = slots.SLOTS["group_size"].ask("hi", None, 0)
        again = slots.SLOTS["group_size"].ask("hi", None, 1)

        assert first == "आपके साथ कितने लोग यात्रा कर रहे हैं?"
        assert "आपके साथ कितने लोग" in again
        assert "कुल" not in first and "कुल" not in again

    def test_english_is_unchanged(self):
        assert (
            slots.SLOTS["group_size"].ask("en", None, 0)
            == "How many people will be travelling in total?"
        )


@pytest.mark.parametrize("lang", ["en", "hi"])
def test_the_choose_cab_line_exists_in_both_languages(lang):
    from app.agent.ride_reply import _TEXT

    line = _TEXT["choose_cab"][lang]
    assert line and line != _TEXT["choose_cab"]["hi" if lang == "en" else "en"]
