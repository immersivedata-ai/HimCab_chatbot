"""A drop the rider PICKED in the app: its exact point reaches the route, the fare and the
booking screen, instead of whatever its name is found as.

The app's drop search knows the exact point of the place the rider tapped, and could only
send its name. The backend then found the name again — the town centre at best, a namesake
somewhere else at worst — and priced and pinned the ride there. These tests pin down the
path that replaces it: `drop_lat` / `drop_lon` / `drop_address` → `PickedPlace` →
`destination_place`. The pickup's twin is `test_picked_pickup.py`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import sanitize
from app.agent.nodes.planner import planner_node
from app.agent.nodes.tool_executor import _destination_point, _resolve_journey, tool_executor_node
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.location import PickedPlace
from app.integrations.external.geocoding import Lookup, LookupStatus
from app.services import geo
from tests.conftest import stub_vertex
from tests.unit.agent.test_conversation_flow import NOTHING, _stub_extractor

DROP_EN = "Where should we drop you?"
# A hotel on Mall Road, half a kilometre from the point a search for "Manali" lands on.
GATE = (32.2432, 77.1892)
ADDRESS = "Hotel Snow Peak, Mall Road, Manali, Himachal Pradesh 175131, India"
SHORT = "Hotel Snow Peak, Mall Road, Manali"


def _pick(**overrides) -> PickedPlace:
    values = {
        "address": ADDRESS, "lat": GATE[0], "lon": GATE[1],
        "town": "Manali", "in_himachal": True,
    }
    values.update(overrides)
    return PickedPlace(**values)


def _state(*, picked=None, asked="destination", text=ADDRESS, **entities) -> AgentState:
    entities.setdefault("visitor_type", "local")
    entities.setdefault("origin", "Shimla")
    return AgentState(
        session_id="s", user_id="u", intent="plan_trip",
        visitor_type=entities["visitor_type"],
        messages=[AIMessage(content=DROP_EN), HumanMessage(content=text)],
        entities=ExtractedEntities(last_asked_field=asked, **entities),
        picked_drop=picked,
    )


async def _plan(state: AgentState) -> dict:
    with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
        stub_vertex(svc, _stub_extractor(NOTHING))
        return await planner_node(state)


def _searched(geocoder) -> list[str]:
    """Every name the geocoder was asked to find by name."""
    return [
        str(call.args[1].get("q", "")) for call in geocoder._get.call_args_list
        if call.args[0] == "/search"
    ]


class TestTheShortNameOfAPickedDrop:
    def test_the_address_is_cut_at_the_town_and_passes_the_place_gate(self):
        """`destination` is validated by `place` on every turn, like `origin`."""
        name = sanitize.picked_name(ADDRESS, "Manali")
        assert name == SHORT
        assert sanitize.place(name) == name


# ── The planner files it as the answer to the drop question ───────────────────

class TestThePlannerFilesAPickedDrop:
    @pytest.mark.asyncio
    async def test_the_pick_answers_the_drop_question_with_the_exact_point(self):
        result = await _plan(_state(picked=_pick()))

        entities = result["entities"]
        assert entities.destination == SHORT
        assert (entities.destination_place.lat, entities.destination_place.lon) == GATE
        assert entities.destination_place.address == ADDRESS
        assert entities.destination_place.name == SHORT

    @pytest.mark.asyncio
    async def test_the_text_sent_with_it_does_not_replace_it(self):
        """The sentence reader would read "drop Hotel Snow Peak" out of the text and write
        that over the point."""
        result = await _plan(_state(picked=_pick(), text="drop Hotel Snow Peak"))
        assert result["entities"].destination == SHORT

    @pytest.mark.asyncio
    async def test_a_pick_is_an_answer_not_a_change_to_confirm(self):
        """Re-asked drop that already held a value: the tap answers the question. The text
        alone WOULD read as a change from Kasol, put to the rider to confirm."""
        result = await _plan(_state(picked=_pick(), destination="Kasol", text="Mall Road, Manali"))

        assert result["entities"].destination == SHORT
        assert not result["entities"].pending_changes
        assert result["entities"].no_progress_turns == 0

    @pytest.mark.asyncio
    async def test_a_pick_counts_as_progress_whatever_text_came_with_it(self):
        """Replacing a re-asked drop fills no NEW slot; without the pick itself counting,
        the turn reads as the rider telling us nothing — one step towards giving up."""
        result = await _plan(_state(picked=_pick(), destination="Kasol", text=ADDRESS))
        assert result["entities"].no_progress_turns == 0

    @pytest.mark.asyncio
    async def test_a_pick_when_the_drop_is_known_and_not_asked_changes_nothing(self):
        """Overwriting a drop the rider gave, unasked, would skip the confirmation every
        other change gets (CLAUDE.md §4.4)."""
        result = await _plan(_state(picked=_pick(), asked="travel_dates", destination="Kasol"))

        assert result["entities"].destination == "Kasol"
        assert result["entities"].destination_place is None

    @pytest.mark.asyncio
    async def test_a_rental_never_files_a_drop(self):
        """A rental has one place, the pickup. A drop filed there would leak into the
        trip-only paths `_NOT_IN_A_RENTAL` exists to keep it out of."""
        result = await _plan(_state(picked=_pick(), asked="rental_hours", visitor_type="rental"))

        assert result["entities"].destination is None
        assert result["entities"].destination_place is None

    @pytest.mark.asyncio
    async def test_a_picked_name_is_never_refused_as_unplaceable(self, monkeypatch):
        """A search for "Hotel Snow Peak, Mall Road, Manali" by name may find nothing, on
        this turn or any later one; refusing it would throw away the exact drop."""
        real_lookup = geo.look_up_place

        async def lookup(name):
            if "snow peak" in name.lower():
                return Lookup(status=LookupStatus.NOT_FOUND)
            return await real_lookup(name)

        monkeypatch.setattr(geo, "look_up_place", lookup)
        first = await _plan(_state(picked=_pick()))
        assert first["entities"].destination == SHORT

        later = _state(asked="travel_dates", text="kal",
                       destination=SHORT, destination_place=_pick(name=SHORT))
        result = await _plan(later)
        assert result["entities"].destination == SHORT
        assert result["entities"].destination_place is not None

    @pytest.mark.asyncio
    async def test_a_typed_drop_is_still_checked_against_the_map(self, monkeypatch):
        """The exemption is for the picked label only. A typo typed over it is refused
        as before, or the check would have a hole in it the shape of a pick."""
        async def lookup(name):
            return Lookup(status=LookupStatus.NOT_FOUND)

        monkeypatch.setattr(geo, "look_up_place", lookup)
        later = _state(asked="travel_dates", text="kal",
                       destination="Kasoll", destination_place=_pick(name=SHORT))
        result = await _plan(later)
        assert result["entities"].destination is None


# ── The route, the fare and the booking pin use the point ─────────────────────

def _trip_state(**entities) -> AgentState:
    entities.setdefault("origin", "Shimla")
    return AgentState(session_id="s", user_id="u", entities=ExtractedEntities(**entities))


class TestTheJourneyEndsAtThePickedPoint:
    @pytest.mark.asyncio
    async def test_the_journey_and_its_pin_are_the_picked_point_with_its_full_address(self):
        state = _trip_state(destination=SHORT, destination_place=_pick(name=SHORT))

        journey = await _resolve_journey(state)

        assert (journey.destination.lat, journey.destination.lon) == GATE
        assert journey.destination.name == ADDRESS          # what the booking screen shows
        assert journey.as_dict()["destination"] == {"name": ADDRESS, "lat": GATE[0], "lon": GATE[1]}

    @pytest.mark.asyncio
    async def test_the_drop_is_not_found_again_by_its_name(self, no_network):
        """By name, this drop is Manali's town centre — the very thing the pick replaces."""
        state = _trip_state(destination=SHORT, destination_place=_pick(name=SHORT))

        await _resolve_journey(state)

        searched = _searched(no_network["geocoder"])
        assert searched == ["Shimla"]

    @pytest.mark.asyncio
    async def test_with_both_ends_picked_nothing_is_looked_up_by_name(self, no_network):
        pickup = PickedPlace(
            name="The Ridge, Shimla", address="The Ridge, Shimla, Himachal Pradesh",
            lat=31.1041, lon=77.1729, town="Shimla", in_himachal=True,
        )
        state = _trip_state(origin="The Ridge, Shimla", origin_place=pickup,
                            destination=SHORT, destination_place=_pick(name=SHORT))

        journey = await _resolve_journey(state)

        assert (journey.origin.lat, journey.origin.lon) == (31.1041, 77.1729)
        assert (journey.destination.lat, journey.destination.lon) == GATE
        assert _searched(no_network["geocoder"]) == []

    def test_a_pick_left_behind_by_a_changed_drop_is_ignored(self):
        """The rider later changed the drop to Kasol: the old point must not price it."""
        state = _trip_state(destination="Kasol", destination_place=_pick(name=SHORT))
        assert _destination_point(state, "Kasol") is None

    @pytest.mark.asyncio
    async def test_the_weather_is_asked_for_the_drops_town(self):
        """A weather service does not know a hotel; asked for one, the reply loses its
        weather line."""
        state = AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="local",
            entities=ExtractedEntities(
                visitor_type="local", origin="Shimla",
                destination=SHORT, destination_place=_pick(name=SHORT),
            ),
        )
        weather = MagicMock()
        weather.get_weather = AsyncMock(return_value=None)
        weather.snowfall_from_report = MagicMock(return_value=None)
        with patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather):
            result = await tool_executor_node(state)

        weather.get_weather.assert_called_once_with("Manali")
        assert result["route_estimate"]["destination"]["name"] == ADDRESS
