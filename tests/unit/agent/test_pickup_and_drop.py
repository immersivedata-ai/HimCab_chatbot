"""
Every flow asks for a pickup and a drop.

A 2026-09-29 decision. The local flow used to ask "आप किस शहर में रहते हैं?" — where the
rider LIVES — when the booking needs where the cab should come; the outstation flow
asked which city they were travelling from, the rental where the car should pick them
up, and a rental had no drop at all. Now there is one pickup question and one drop
question, in every flow.

Asking in those words changes what riders type back, so this also covers the readers:
"Manali drop kar do", "Shimla se pickup", "yahin se", and — for a rental, whose hire
often ends where it began — "same place".
"""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import slots
from app.agent.nodes.planner import planner_node
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.location import RiderLocation
from app.services import chat_session
from tests.conftest import stub_vertex
from tests.unit.agent.test_conversation_flow import (
    NOTHING,
    Conversation,
    _stub_extractor,
    extraction,
)

PICKUP_HI = "आपको कहाँ से पिकअप करना है?"
DROP_HI = "आपको कहाँ ड्रॉप करना है?"


class TestTheQuestions:
    @pytest.mark.parametrize("flow", [slots.LOCAL, slots.OUTSTATION, slots.RENTAL])
    def test_every_flow_asks_for_the_pickup_and_the_drop_in_the_same_words(self, flow):
        assert slots.SLOTS["origin"].ask("hi", flow, 0) == PICKUP_HI
        assert slots.SLOTS["destination"].ask("hi", flow, 0) == DROP_HI
        assert slots.SLOTS["origin"].ask("en", flow, 0) == "Where should we pick you up?"
        assert slots.SLOTS["destination"].ask("en", flow, 0) == "Where should we drop you?"

    def test_nobody_is_asked_where_they_live_any_more(self):
        for flow in (slots.LOCAL, slots.OUTSTATION, slots.RENTAL):
            for attempt in (0, 1):
                assert "रहते" not in slots.SLOTS["origin"].ask("hi", flow, attempt)
                assert "live" not in slots.SLOTS["origin"].ask("en", flow, attempt)

    def test_the_re_asks_differ_from_the_first_ask(self):
        for field in ("origin", "destination"):
            for lang in ("hi", "en"):
                assert slots.SLOTS[field].ask(lang, None, 0) != slots.SLOTS[field].ask(lang, None, 1)

    def test_the_labels_say_pickup_and_drop(self):
        """Used in change confirmations: "change the pickup location from X to Y?"."""
        assert slots.label_for("origin", "en") == "pickup location"
        assert slots.label_for("destination", "en") == "drop location"
        assert slots.label_for("origin", "hi") == "पिकअप की जगह"
        assert slots.label_for("destination", "hi") == "ड्रॉप की जगह"

    @pytest.mark.parametrize("flow", [slots.LOCAL, slots.OUTSTATION])
    def test_the_greeting_opens_on_the_drop_question(self, flow):
        """The greeting's question is the first one the flow asks (the drop is the
        trip's REQUIRED slot), and `/start` seeds `last_asked_field` to match it."""
        greeting = chat_session.WELCOME_MESSAGES[flow]
        assert greeting.endswith(DROP_HI) or greeting.endswith("Where should we drop you?")


class TestAnAnswerWrappedInPickupAndDropWords:
    @pytest.mark.parametrize("reply,place", [
        ("Manali drop kar do", "Manali"),
        ("drop at Kasol", "Kasol"),
        ("Shimla se pickup", "Shimla"),
        ("pick me up from Solan", "Solan"),
        ("मनाली ड्रॉप", "मनाली"),
    ])
    def test_the_framing_is_taken_off_and_the_place_kept(self, reply, place):
        assert slots.parse_location(reply) == place

    def test_a_name_containing_a_framing_word_is_left_whole(self):
        """Stripped only after the reply failed as it stands — so "Point" survives."""
        assert slots.parse_location("Scandal Point") == "Scandal Point"

    @pytest.mark.parametrize("reply", ["pickup", "drop", "drop kar do", "cancel", "rent a car"])
    def test_framing_alone_is_still_not_a_place(self, reply):
        assert slots.parse_location(reply) is None

    @pytest.mark.asyncio
    async def test_a_bare_drop_answer_fills_the_destination(self):
        chat = Conversation(persona="local")
        await chat.say("Kasol drop kar do", NOTHING)
        assert chat.entities.destination == "Kasol"


class TestThePickupAnsweredWithHere:
    """Rental-only until every flow asked for a pickup; now any flow may hear "yahin"."""

    @pytest.mark.asyncio
    async def test_a_local_rider_saying_yahin_se_is_collected_from_their_fix(self):
        state = AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="local",
            messages=[AIMessage(content=PICKUP_HI), HumanMessage(content="yahin se pickup karo")],
            entities=ExtractedEntities(
                visitor_type="local", destination="Manali", last_asked_field="origin",
            ),
            rider_location=RiderLocation(
                lat=31.1048, lon=77.1734, in_himachal=True, place_name="Shimla",
                place_distance_km=1.2, state="Himachal Pradesh",
            ),
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            stub_vertex(svc, _stub_extractor(NOTHING))
            result = await planner_node(state)

        assert result["entities"].origin == "Shimla"

    @pytest.mark.asyncio
    async def test_without_a_usable_fix_the_pickup_is_asked_again(self):
        chat = Conversation(persona="local")
        await chat.say("Manali", extraction(destination="Manali"))
        turn = await chat.say("yahin se", NOTHING)

        assert chat.entities.origin is None
        assert turn.asked and turn.result["last_asked_field"] == "origin"


class TestARentalDrop:
    @pytest.mark.parametrize("reply", ["same place", "wahi", "pickup wali jagah", "वहीं"])
    def test_same_place_means_the_pickup(self, reply):
        assert slots.means_same_as_pickup(reply)

    @pytest.mark.parametrize("reply", ["back to Shimla", "Kufri", "is it the same?", ""])
    def test_a_place_or_a_question_does_not(self, reply):
        assert not slots.means_same_as_pickup(reply)

    @pytest.mark.asyncio
    async def test_it_is_asked_after_the_pickup(self):
        chat = Conversation(persona=slots.RENTAL, intent="rental_query")
        await chat.say("8 hours", NOTHING)
        turn = await chat.say("Shimla", NOTHING)

        assert turn.result["last_asked_field"] == "destination"
        assert turn.question in (DROP_HI, "Where should we drop you?")

    @pytest.mark.asyncio
    async def test_same_place_as_the_answer_drops_them_at_the_pickup(self):
        chat = Conversation(persona=slots.RENTAL, intent="rental_query")
        await chat.say("8 hours", NOTHING)
        await chat.say("Shimla", NOTHING)
        await chat.say("same place", NOTHING)

        assert chat.entities.destination == "Shimla"

    @pytest.mark.asyncio
    async def test_a_drop_the_model_reads_off_the_drop_question_is_kept(self):
        chat = Conversation(persona=slots.RENTAL, intent="rental_query")
        await chat.say("8 hours", NOTHING)
        await chat.say("Shimla", NOTHING)
        await chat.say("Kufri please", extraction(destination="Kufri"))

        assert chat.entities.destination == "Kufri"

    @pytest.mark.asyncio
    async def test_a_place_mentioned_in_passing_is_still_not_a_drop(self):
        """The reason the model's rental destination is not trusted on its own: "around
        Manali" is where the car is driven about, not where the hire ends."""
        chat = Conversation(persona=slots.RENTAL, intent="rental_query")
        await chat.say("a car for 8 hours around Manali",
                       extraction(rental_hours=8, destination="Manali"))

        assert chat.entities.destination is None


class TestTheSentenceReaders:
    @pytest.mark.parametrize("text,origin,drop", [
        ("8 hours from Shimla, drop at Kufri", "Shimla", "Kufri"),
        ("8 hours from Shimla drop me at Kufri", "Shimla", "Kufri"),
        ("pickup Shimla drop Kufri", "Shimla", "Kufri"),
        ("Shimla se 8 ghante, Kufri pe drop", "Shimla", "Kufri"),
    ])
    def test_a_rental_sentence_yields_both_ends(self, text, origin, drop):
        found = slots.read_rental_details(text)
        assert (found.get("origin"), found.get("destination")) == (origin, drop)

    def test_a_drop_verb_is_not_read_as_the_place(self):
        """"Kufri drop kar dena" once came back as the drop "Kar Dena"."""
        assert slots.read_rental_details("Kufri drop kar dena").get("destination") == "Kufri"

    def test_an_ambiguous_run_of_names_is_read_as_nothing(self):
        """"from Shimla Kufri drop": nothing says which word is the drop, so neither end
        is guessed — a guess here was the pickup "Shimla Kufri"."""
        assert slots.read_rental_details("from Shimla Kufri drop") == {}

    @pytest.mark.parametrize("text", ["pickup Shimla drop Manali", "Shimla se pickup, Manali drop"])
    def test_a_trip_sentence_yields_both_ends(self, text):
        found = slots.read_trip_details(text)
        assert (found.get("origin"), found.get("destination")) == ("Shimla", "Manali")

    @pytest.mark.asyncio
    async def test_both_ends_are_read_with_vertex_down(self):
        """§2: a rider-facing reading works with no model at all."""
        chat = Conversation(persona="local")
        await chat.say("pickup Shimla drop Manali", None)
        assert (chat.entities.origin, chat.entities.destination) == ("Shimla", "Manali")


class TestABareReplyNamingBothEnds:
    """Asked for a drop, riders often give the whole journey. The bare-answer parser
    used to read it as ONE place — "Shimla Se Manali", "Shimla Manali" — and that value
    then outranked the sentence reader's correct two ends."""

    @pytest.mark.parametrize("reply", ["Shimla se Manali", "pickup Shimla drop Manali",
                                       "Kasol to Tosh"])
    def test_the_bare_parser_reads_no_single_place_out_of_two(self, reply):
        assert slots.parse_location(reply) is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply", ["Shimla se Manali", "pickup Shimla drop Manali",
                                       "Delhi se Manali tak"])
    async def test_the_two_ends_land_in_the_two_slots(self, reply):
        chat = Conversation(persona="local")
        await chat.say(reply, NOTHING)
        pickup = "Delhi" if reply.startswith("Delhi") else "Shimla"
        assert (chat.entities.origin, chat.entities.destination) == (pickup, "Manali")
