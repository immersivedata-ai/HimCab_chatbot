"""A pickup the rider PICKED in the app: its exact point reaches the route, the fare and the
booking screen, instead of the town it sits in.

The app's pickup dialog offers a place search and "use my current location". Both know the
exact point, and both used to arrive as a town: the API could only carry words for a
pickup, so the app trimmed a pick to three of them or fell back to the town. Picked at the
hotel gate, priced and pinned at the town centre. These tests pin down the new path —
`pickup_lat` / `pickup_lon` / `pickup_address` → `PickedPlace` → `origin_place`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import sanitize
from app.agent.nodes.planner import planner_node
from app.agent.nodes.tool_executor import _origin_point, _rental_tools, _resolve_journey
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.location import PickedPlace, RiderLocation
from app.integrations.external.geocoding import GeocodingService, Lookup, LookupStatus
from app.services import geo
from tests.conftest import stub_vertex
from tests.unit.agent.test_conversation_flow import NOTHING, _stub_extractor

PICKUP_EN = "Where should we pick you up?"
GATE = (31.1035, 77.1712)          # a point on The Mall, a few hundred metres from Shimla's centre
ADDRESS = "Hotel Snow View, The Mall, Shimla, Himachal Pradesh 171001, India"
SHORT = "Hotel Snow View, The Mall, Shimla"


def _pick(**overrides) -> PickedPlace:
    values = {
        "address": ADDRESS, "lat": GATE[0], "lon": GATE[1],
        "town": "Shimla", "in_himachal": True,
    }
    values.update(overrides)
    return PickedPlace(**values)


def _state(*, picked=None, asked="origin", text=ADDRESS, **entities) -> AgentState:
    entities.setdefault("visitor_type", "local")
    entities.setdefault("destination", "Manali")
    return AgentState(
        session_id="s", user_id="u", intent="plan_trip",
        visitor_type=entities["visitor_type"],
        messages=[AIMessage(content=PICKUP_EN), HumanMessage(content=text)],
        entities=ExtractedEntities(last_asked_field=asked, **entities),
        picked_pickup=picked,
    )


async def _plan(state: AgentState) -> dict:
    with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
        stub_vertex(svc, _stub_extractor(NOTHING))
        return await planner_node(state)


# ── The short name the conversation uses ──────────────────────────────────────

class TestTheShortNameOfAPickedAddress:
    def test_the_address_is_cut_at_the_town(self):
        assert sanitize.picked_name(ADDRESS, "Shimla") == SHORT

    def test_the_state_after_the_town_is_not_part_of_the_name(self):
        """Short enough to fit, and still not where the car stops."""
        assert sanitize.picked_name("The Ridge, Shimla, Himachal Pradesh", "Shimla") == (
            "The Ridge, Shimla"
        )

    def test_a_pin_code_and_the_country_never_reach_the_name(self):
        name = sanitize.picked_name("Ridge Road, 171001, India")
        assert name == "Ridge Road"

    def test_parts_stop_being_added_before_the_name_is_too_long(self):
        """Six words is `place`'s limit; a seventh part would make the name unusable."""
        name = sanitize.picked_name(
            "Near Old Bus Stand, Lakkar Bazaar, Sanjauli, Shimla", "Shimla"
        )
        assert name == "Near Old Bus Stand, Lakkar Bazaar"

    def test_one_part_too_long_on_its_own_keeps_its_first_words(self):
        name = sanitize.picked_name("Opposite Old Bus Stand Near Lakkar Bazaar Market Road")
        assert name == "Opposite Old Bus Stand Near Lakkar"

    @pytest.mark.parametrize("address", [
        ADDRESS,
        "Near Old Bus Stand, Lakkar Bazaar, Sanjauli, Shimla",
        "Opposite Old Bus Stand Near Lakkar Bazaar Market Road",
        "Kangra Fort Road, Dharamshala",
    ])
    def test_every_name_passes_the_gate_every_other_place_passes(self, address):
        """`origin` is validated by `place` on every turn. A name it refused would empty
        the pickup on the very turn the rider picked it."""
        name = sanitize.picked_name(address, "Shimla")
        assert name and sanitize.place(name) == name

    @pytest.mark.parametrize("text", ["my current location", "", "   "])
    def test_words_that_are_not_an_address_give_no_name(self, text):
        assert sanitize.picked_name(text) is None


# ── Naming a point the app sent without an address ────────────────────────────

class TestNamingAPointAtStreetLevel:
    def test_the_address_is_built_from_the_street_up(self):
        row = {
            "name": "Hotel Snow View",
            "address": {"road": "The Mall", "suburb": "Shimla Urban", "city": "Shimla",
                        "state": "Himachal Pradesh", "postcode": "171001", "country": "India"},
        }
        assert GeocodingService._to_address(row) == (
            "Hotel Snow View, The Mall, Shimla Urban, Shimla"
        )

    def test_a_part_already_named_is_not_repeated(self):
        row = {"name": "Shimla", "address": {"city": "Shimla", "state": "Himachal Pradesh"}}
        assert GeocodingService._to_address(row) == "Shimla"

    @pytest.mark.asyncio
    async def test_it_asks_the_provider_at_street_level_not_town_level(self, no_network):
        """`reverse` stops at the town on purpose; this one must not."""
        geocoder = no_network["geocoder"]
        await geocoder.reverse_address(*GATE)
        zooms = [call.args[1].get("zoom") for call in geocoder._get.call_args_list]
        assert 18 in zooms


class TestResolvingAPick:
    @pytest.mark.asyncio
    async def test_the_point_is_kept_exactly_and_the_address_as_given(self):
        picked = await geo.resolve_picked_place(*GATE, ADDRESS)

        assert (picked.lat, picked.lon) == GATE
        assert picked.address == ADDRESS
        assert picked.town == "Shimla"
        assert picked.in_himachal is True

    @pytest.mark.asyncio
    async def test_without_an_address_the_backend_names_the_point(self, no_network):
        picked = await geo.resolve_picked_place(*GATE, None)

        assert picked is not None and picked.address
        assert (picked.lat, picked.lon) == GATE

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lat,lon", [(0.0, 0.0), (None, 77.17), (31.10, None)])
    async def test_an_unusable_point_is_no_pick_at_all(self, lat, lon):
        assert await geo.resolve_picked_place(lat, lon, ADDRESS) is None


# ── The planner files it as the answer to the pickup question ─────────────────

class TestThePlannerFilesAPickedPickup:
    @pytest.mark.asyncio
    async def test_the_pick_answers_the_pickup_question_with_the_exact_point(self):
        result = await _plan(_state(picked=_pick()))

        entities = result["entities"]
        assert entities.origin == SHORT
        assert (entities.origin_place.lat, entities.origin_place.lon) == GATE
        assert entities.origin_place.address == ADDRESS
        assert entities.origin_place.name == SHORT

    @pytest.mark.asyncio
    async def test_the_text_sent_with_it_does_not_replace_it(self):
        """The app sends the address as the message too. The sentence readers would read
        "pickup Hotel Snow View" out of it and write that over the point."""
        result = await _plan(_state(picked=_pick(), text="pickup Hotel Snow View drop Manali"))
        assert result["entities"].origin == SHORT

    @pytest.mark.asyncio
    async def test_a_pick_is_an_answer_not_a_change_to_confirm(self):
        """Re-asked pickup that already held a value: the tap answers the question. The
        text is the short wording the app sends with a pick, which on its own WOULD read
        as a change from Kufri and be put to the rider to confirm."""
        result = await _plan(_state(picked=_pick(), origin="Kufri", text="The Mall, Shimla"))

        assert result["entities"].origin == SHORT
        assert not result["entities"].pending_changes
        # And it counts as progress: a run of "no progress" turns is what makes the
        # agent stop asking and give up.
        assert result["entities"].no_progress_turns == 0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("text", [ADDRESS, "my current location"])
    async def test_a_pick_counts_as_progress_whatever_text_came_with_it(self, text):
        """Replacing a re-asked pickup fills no NEW slot, and neither of these texts reads
        as a place — so without the pick itself counting, the turn would be logged as the
        rider telling us nothing, one step towards the agent giving up on them."""
        result = await _plan(_state(picked=_pick(), origin="Kufri", text=text))
        assert result["entities"].no_progress_turns == 0

    @pytest.mark.asyncio
    async def test_a_local_riders_home_city_is_the_town_not_the_gate(self):
        """A home city is somewhere a rider lives, and is looked up by name on later
        turns; nobody lives at "Hotel Snow View, The Mall", and no geocoder finds it."""
        result = await _plan(_state(picked=_pick()))
        assert result["entities"].home_city == "Shimla"

    @pytest.mark.asyncio
    async def test_a_pick_when_the_pickup_is_known_and_not_asked_changes_nothing(self):
        """Overwriting a pickup the rider gave, unasked, would skip the confirmation
        every other change gets (CLAUDE.md §4.4)."""
        result = await _plan(_state(picked=_pick(), asked="travel_dates", origin="Kufri"))

        assert result["entities"].origin == "Kufri"
        assert result["entities"].origin_place is None

    @pytest.mark.asyncio
    async def test_saying_my_current_location_with_a_pick_uses_the_pick(self):
        """Not the town the phone is in: the app sends the exact point with it now."""
        state = _state(picked=_pick(), text="my current location")
        state.rider_location = RiderLocation(
            lat=31.1048, lon=77.1734, in_himachal=True, place_name="Shimla",
            place_distance_km=0.2, state="Himachal Pradesh",
        )
        result = await _plan(state)
        assert result["entities"].origin == SHORT

    @pytest.mark.asyncio
    async def test_where_the_rider_is_decided_from_the_point_not_the_label(self):
        """A pick outside Himachal is a visitor's journey, as a typed "Delhi" would be —
        and it is decided from the point, not by searching for the label."""
        # No town and no name the geocoder knows: only the point can say where this is.
        delhi = _pick(address="Terminal 3, IGI Airport", lat=28.5562,
                      lon=77.1000, town=None, in_himachal=False)
        result = await _plan(_state(picked=delhi))
        assert result["entities"].visitor_type == "outstation"

    @pytest.mark.asyncio
    async def test_a_picked_name_is_never_refused_as_unplaceable(self, monkeypatch):
        """A search for "Hotel Snow View, The Mall, Shimla" by name may find nothing, on
        this turn or any later one; refusing it would throw away the exact pickup."""
        # Only the picked label is unfindable; "Manali" must still be found, or the check
        # stops at the destination and never reaches the pickup at all.
        real_lookup = geo.look_up_place

        async def lookup(name):
            if "snow view" in name.lower():
                return Lookup(status=LookupStatus.NOT_FOUND)
            return await real_lookup(name)

        monkeypatch.setattr(geo, "look_up_place", lookup)
        first = await _plan(_state(picked=_pick()))
        assert first["entities"].origin == SHORT

        later = _state(asked="travel_dates", text="kal",
                       origin=SHORT, origin_place=_pick(name=SHORT))
        result = await _plan(later)
        assert result["entities"].origin == SHORT
        assert result["entities"].origin_place is not None


# ── The route, the fare and the booking pin use the point ─────────────────────

def _trip_state(**entities) -> AgentState:
    entities.setdefault("destination", "Manali")
    return AgentState(session_id="s", user_id="u", entities=ExtractedEntities(**entities))


class TestTheJourneyStartsAtThePickedPoint:
    @pytest.mark.asyncio
    async def test_the_journey_and_its_pin_are_the_picked_point_with_its_full_address(self):
        state = _trip_state(origin=SHORT, origin_place=_pick(name=SHORT))

        journey = await _resolve_journey(state)

        assert (journey.origin.lat, journey.origin.lon) == GATE
        assert journey.origin.name == ADDRESS          # what the booking screen shows
        assert journey.as_dict()["origin"] == {"name": ADDRESS, "lat": GATE[0], "lon": GATE[1]}

    def test_a_pick_left_behind_by_a_changed_pickup_is_ignored(self):
        """The rider later changed the pickup to Kufri: the old point must not price it."""
        state = _trip_state(origin="Kufri", origin_place=_pick(name=SHORT))
        assert _origin_point(state, "Kufri") is None

    def test_the_pick_beats_the_phone(self):
        """The phone is where the rider is; the pick is where the car should go."""
        state = _trip_state(origin=SHORT, origin_place=_pick(name=SHORT))
        state.rider_location = RiderLocation(
            lat=31.1048, lon=77.1734, in_himachal=True, place_name="Shimla",
            place_distance_km=0.0,
        )
        point = _origin_point(state, SHORT)
        assert (point.lat, point.lon) == GATE

    def test_without_a_pick_the_phone_is_used_as_before(self):
        state = _trip_state(origin="Shimla")
        state.rider_location = RiderLocation(
            lat=31.1048, lon=77.1734, in_himachal=True, place_name="Shimla",
            place_distance_km=0.0,
        )
        point = _origin_point(state, "Shimla")
        assert (point.lat, point.lon) == (31.1048, 77.1734)


class TestARentalCollectsFromThePickedPoint:
    @pytest.mark.asyncio
    async def test_the_pin_is_the_point_and_the_weather_is_the_towns(self):
        state = AgentState(
            session_id="s", user_id="u", visitor_type="rental",
            entities=ExtractedEntities(
                visitor_type="rental", origin=SHORT, origin_place=_pick(name=SHORT)
            ),
        )
        weather = MagicMock()
        weather.get_weather = AsyncMock(return_value=None)
        weather.snowfall_from_report = MagicMock(return_value=None)
        with patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather):
            result = await _rental_tools(state)

        assert result["pickup"] == {"name": ADDRESS, "lat": GATE[0], "lon": GATE[1]}
        weather.get_weather.assert_called_once_with("Shimla")
