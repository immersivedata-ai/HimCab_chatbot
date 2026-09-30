"""
A place the rider names that is not on the map goes back to them by name.

Reported as "the workflow either takes wrong place information or hallucinates and
gives an invalid response". Both halves were real, and both were measured against the
live geocoder before this was written:

  • WRONG PLACE — the geocoder took the provider's first hit, whatever it was. "xyz"
    came back as a residential road in Thane, "hello" as a shop in Lucknow, and "Bir"
    as a railway station in Madhya Pradesh rather than the village in Kangra. Each was
    routed, priced and put on a booking screen.
  • HALLUCINATION — a name nothing matched ("Kasoll") went ahead anyway. No route, no
    fare, and a writer handed a destination it had no facts about, which it filled in.

The provider rows below are the shapes Nominatim actually returned for those queries.
"""

from datetime import date
from typing import ClassVar
from unittest.mock import AsyncMock

import pytest

from app.agent import slots
from app.agent.nodes import planner
from app.core.config import settings
from app.integrations.external import geocoding
from app.integrations.external.geocoding import LookupStatus
from tests.unit.agent.test_conversation_flow import NOTHING, Conversation, extraction

TODAY = date(2026, 9, 21)


def _row(name, category, kind, state, rank=18, lat=31.0, lon=77.0):
    return {
        "lat": str(lat), "lon": str(lon), "name": name,
        "category": category, "type": kind, "place_rank": rank,
        "display_name": f"{name}, {state}, India",
        "address": {"town": "Manali", "state": state, "country": "India"},
    }


# ── The geocoder: which result is a place, and which one the rider meant ──────

class TestTheGeocoderOnlyAcceptsPlaces:
    def test_a_road_that_shares_the_word_is_not_a_destination(self):
        rows = [
            _row("xyz", "highway", "residential", "Maharashtra", rank=26),
            _row("XYZ", "amenity", "pub", "Punjab", rank=30),
            _row("xyz", "building", "yes", "Kerala", rank=30),
        ]
        assert geocoding.pick_candidate(rows) is None

    def test_a_shop_called_hello_is_not_a_destination(self):
        rows = [_row("Hello", "shop", "mall", "Uttar Pradesh", rank=30)]
        assert geocoding.pick_candidate(rows) is None

    def test_the_himachal_reading_of_an_ambiguous_name_wins(self):
        rows = [
            _row("Bir", "railway", "station", "Madhya Pradesh", rank=30),
            _row("Bir", "place", "village", "Himachal Pradesh", rank=19),
        ]
        picked = geocoding.pick_candidate(rows)
        assert picked["address"]["state"] == "Himachal Pradesh"

    def test_a_major_city_outside_himachal_keeps_its_own_meaning(self):
        rows = [
            _row("Delhi", "boundary", "administrative", "Delhi", rank=8),
            _row("Delhi", "place", "hamlet", "Himachal Pradesh", rank=20),
        ]
        assert geocoding.pick_candidate(rows)["address"]["state"] == "Delhi"

    def test_a_temple_or_a_valley_is_somewhere_a_cab_can_go(self):
        assert geocoding.is_destination_kind(
            _row("Hadimba Devi Temple", "amenity", "place_of_worship", "Himachal Pradesh", 30)
        )
        assert geocoding.is_destination_kind(
            _row("Solang Valley", "landuse", "residential", "Himachal Pradesh", 24)
        )

    def test_a_forward_result_is_called_what_the_rider_asked_for(self):
        """The drop is the temple, not the town around it."""
        place = geocoding.GeocodingService()._to_place(
            _row("Hadimba Devi Temple", "amenity", "place_of_worship", "Himachal Pradesh"),
            own_name_first=True,
        )
        assert place.name == "Hadimba Devi Temple"

    def test_a_reverse_result_is_still_named_by_its_town(self):
        place = geocoding.GeocodingService()._to_place(
            _row("Some Guest House", "tourism", "guest_house", "Himachal Pradesh")
        )
        assert place.name == "Manali"


class TestALookupSaysWhyItFoundNothing:
    @pytest.mark.asyncio
    async def test_no_such_place_is_not_found(self, no_network):
        lookup = await no_network["geocoder"].lookup("Kasoll")
        assert lookup.status is LookupStatus.NOT_FOUND

    @pytest.mark.asyncio
    async def test_a_provider_failure_is_unavailable_not_a_verdict_on_the_name(self, no_network):
        geocoder = no_network["geocoder"]
        geocoder._get = AsyncMock(side_effect=RuntimeError("nominatim down"))
        lookup = await geocoder.lookup("Shimla")
        assert lookup.status is LookupStatus.UNAVAILABLE

    @pytest.mark.asyncio
    async def test_an_outage_is_not_cached_as_a_miss(self, no_network):
        geocoder = no_network["geocoder"]
        real = geocoder._get
        geocoder._get = AsyncMock(side_effect=RuntimeError("nominatim down"))
        await geocoder.lookup("Shimla")
        geocoder._get = real
        assert (await geocoder.lookup("Shimla")).status is LookupStatus.FOUND


# ── The planner: an unplaceable name is refused on the turn it arrives ────────

class TestATypoedDestinationIsPutBackByName:
    @pytest.mark.asyncio
    async def test_it_is_not_kept_and_the_question_quotes_it(self):
        chat = Conversation()
        turn = await chat.say("Kasoll", extraction(destination="Kasoll"))

        assert turn.asked
        assert "Kasoll" in turn.question
        assert chat.entities.destination is None
        assert turn.result["unplaceable_place"]["value"] == "Kasoll"

    @pytest.mark.asyncio
    async def test_it_is_asked_in_hindi_for_a_hindi_speaker(self):
        chat = Conversation()
        turn = await chat.say("mujhe kasoll jana hai", NOTHING)

        assert turn.asked
        assert "जगह" in turn.question and "kasoll" in turn.question.lower()

    @pytest.mark.asyncio
    async def test_the_corrected_spelling_is_then_taken(self):
        chat = Conversation()
        await chat.say("Kasoll", extraction(destination="Kasoll"))
        await chat.say("Kasol", extraction(destination="Kasol"))

        assert chat.entities.destination == "Kasol"

    @pytest.mark.asyncio
    async def test_a_sentence_about_somewhere_real_is_not_rewritten(self):
        """"places TO eat in Manali" once became destination "Eat"."""
        chat = Conversation()
        turn = await chat.say(
            "What are the best places to eat in Manali?",
            extraction(destination="Manali"),
            intent="places_query",
        )
        assert turn.produced
        assert chat.entities.destination == "Manali"

    @pytest.mark.asyncio
    async def test_a_geocoder_outage_refuses_nothing(self, no_network):
        no_network["geocoder"]._get = AsyncMock(side_effect=RuntimeError("down"))
        chat = Conversation()
        turn = await chat.say("Kasol", extraction(destination="Kasol"))

        assert chat.entities.destination == "Kasol"
        assert turn.result["unplaceable_place"] is None


class TestAnOriginWeCannotPlace:
    @pytest.mark.asyncio
    async def test_it_is_asked_about_even_before_the_next_missing_detail(self):
        chat = Conversation()
        turn = await chat.say(
            "Manali from Xyzabc", extraction(destination="Manali", origin="Xyzabc")
        )
        assert turn.asked
        assert turn.result["last_asked_field"] == "origin"
        assert "Xyzabc" in turn.question
        assert chat.entities.destination == "Manali"
        assert chat.entities.origin is None

    @pytest.mark.asyncio
    async def test_a_resident_is_asked_for_the_pickup_not_where_they_live(self):
        """Every flow asks for a pickup since 2026-09-29 — this re-ask used to say
        "Which city do you live in?"."""
        chat = Conversation(persona="local")
        turn = await chat.say(
            "Xyzabc", extraction(origin="Xyzabc", home_city="Xyzabc", destination="Manali")
        )
        assert turn.asked
        assert "पिकअप" in turn.question or "pick you up" in turn.question
        assert "रहते" not in turn.question and "live" not in turn.question
        assert chat.entities.home_city is None and chat.entities.origin is None

    @pytest.mark.asyncio
    async def test_past_the_cap_the_trip_goes_ahead_and_says_it_left_it_out(self):
        chat = Conversation()
        chat.entities.asked_counts["origin"] = settings.max_clarification_attempts
        turn = await chat.say(
            "Manali from Xyzabc", extraction(destination="Manali", origin="Xyzabc")
        )
        assert chat.entities.origin is None
        assert "Xyzabc" in (turn.result.get("change_note") or "")


class TestARentalPickupWeCannotPlace:
    @pytest.mark.asyncio
    async def test_the_pickup_question_names_it(self):
        chat = Conversation(persona="rental")
        turn = await chat.say("8 hours from Xyzabc", extraction(rental_hours=8, origin="Xyzabc"))

        assert turn.asked
        assert turn.result["last_asked_field"] == "origin"
        assert "Xyzabc" in turn.question
        assert chat.entities.origin is None


class TestARealPlaceAnywhereIsAccepted:
    """The flow is not limited to one state, so nobody is told "we only plan Himachal"."""

    @pytest.mark.asyncio
    async def test_a_trip_outside_himachal_is_not_refused(self):
        chat = Conversation()
        turn = await chat.say("Mumbai", extraction(destination="Mumbai"))

        assert chat.entities.destination == "Mumbai"
        assert turn.result["unplaceable_place"] is None
        assert "Himachal" not in (turn.question or "")

    @pytest.mark.asyncio
    async def test_a_local_ride_may_end_outside_the_state(self):
        chat = Conversation(persona="local")
        await chat.say(
            "Shimla se Chandigarh",
            extraction(destination="Chandigarh", origin="Shimla", home_city="Shimla"),
        )
        assert chat.entities.destination == "Chandigarh"


class TestACorrectionToAPlaceWeCannotFind:
    @pytest.mark.asyncio
    async def test_the_old_value_stays_and_the_next_name_is_applied_directly(self):
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        turn = await chat.say("mujhe kasoll jana hai", NOTHING)

        assert turn.asked and "kasoll" in turn.question.lower()
        assert chat.entities.destination == "Manali"
        assert chat.entities.awaiting_change_detail

        await chat.say("Kasol", NOTHING)
        assert chat.entities.destination == "Kasol"


# ── The wording: both languages, same placeholders (§4.8) ─────────────────────

class TestTheWordingExistsInBothLanguages:
    TABLES: ClassVar[list[dict]] = [
        slots._PLACE_NOT_FOUND_Q[slots.SlotName.DESTINATION],
        slots._PLACE_NOT_FOUND_Q[slots.SlotName.ORIGIN],
        slots._PLACE_DROPPED_NOTE,
    ]

    @pytest.mark.parametrize("table", TABLES)
    def test_hindi_and_english_carry_the_same_placeholder(self, table):
        assert set(table) == {"hi", "en"}
        assert "{place}" in table["hi"] and "{place}" in table["en"]
        assert table["hi"] != table["en"]

    def test_a_long_paste_is_trimmed_before_it_is_quoted(self):
        question = slots.unplaceable_question("destination", "x" * 90, "en")
        assert "…" in question and "x" * 60 not in question

    def test_the_planner_decides_this_without_the_model(self):
        """§2: every rider-facing decision works with Vertex unavailable."""
        assert "get_vertex_ai_service" not in planner._reject_unplaceable.__code__.co_names
        assert "get_vertex_ai_service" not in planner._ask_about_unplaceable.__code__.co_names
