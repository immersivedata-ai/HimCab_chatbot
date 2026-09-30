"""The `input_request` card: which questions get a picker, and that a picked answer parses."""

import pytest

from app.agent import slots
from app.core.enums import SlotName


def test_date_question_asks_for_a_calendar():
    assert slots.input_request_for(SlotName.TRAVEL_DATES) == {
        "field": "travel_dates",
        "kind": "date",
    }


@pytest.mark.parametrize("field", [SlotName.GROUP_SIZE, SlotName.DURATION_DAYS])
def test_count_questions_ask_for_a_number_row(field):
    request = slots.input_request_for(field)
    assert request["kind"] == "number"
    assert (request["min"], request["max"]) == (1, slots.PICKER_MAX)


@pytest.mark.parametrize(
    "field",
    [None, SlotName.DESTINATION, SlotName.ORIGIN, SlotName.PICKUP_TIME, SlotName.RENTAL_HOURS],
)
def test_other_questions_get_no_picker(field):
    assert slots.input_request_for(field) is None


def test_a_picked_date_is_read_as_the_travel_date():
    """The app sends the calendar pick as "25-09-2026"."""
    assert slots.parse_when("25-09-2026") == "25-09-2026"


@pytest.mark.parametrize("n", [1, 4, slots.PICKER_MAX])
def test_every_picked_number_is_read_by_its_parser(n):
    assert slots.parse_group_size(str(n)) == n
    assert slots.parse_duration_days(str(n)) == n


@pytest.mark.asyncio
async def test_clarify_attaches_the_picker_to_the_question():
    from app.agent.graph import clarify_node
    from app.agent.state import AgentState

    state = AgentState(
        session_id="s",
        clarification_question="How many people will be travelling in total?",
        last_asked_field=SlotName.GROUP_SIZE,
    )
    result = await clarify_node(state)
    assert {"type": "input_request", "payload": slots.input_request_for(SlotName.GROUP_SIZE)} \
        in result["structured_events"]
