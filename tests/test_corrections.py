"""Changing your mind mid-conversation.

The reported bug: the rider gave a pickup and a drop, then changed one, and the
assistant carried on with the old value. The design question that came with it was
whether detecting a change needs a second intent classifier ("destination change",
"date change", "pickup change") on every turn.

It does not, and these tests are the argument. A change is a COMPARISON against what
the session already holds, not a classification of the sentence — so it costs a dict
lookup, runs with the extraction model unavailable, and cannot itself be wrong about
a value the readers never produced. `test_detection_costs_no_model_call` and
`test_detection_is_microseconds` pin those two properties down so a future refactor
cannot quietly reintroduce a round trip here.
"""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from app.agent import slots
from app.agent.nodes.router import _answers_the_question
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities


class TestASentenceThatChangesASlot:
    """The first shape of a correction: a sentence carrying its own marker."""

    @pytest.mark.parametrize(
        "message, expected",
        [
            # The exact message from the report, which used to read as the places
            # "Mujhe Palampur" and "Lekar Kasol".
            ("mujhe palampur se lekar kasol jana hai", {"origin": "Palampur", "destination": "Kasol"}),
            ("ab kasol jana hai", {"destination": "Kasol"}),
            ("I want to go to Kasol instead", {"destination": "Kasol"}),
            ("delhi se hote hue manali jana hai", {"origin": "Delhi"}),
            # A date and a group size change on the same anchored footing.
            ("parso jana hai", {"travel_dates": "parso"}),
            ("we are 6 now", {"group_size": 6}),
        ],
    )
    def test_the_new_value_is_read(self, message, expected):
        read = slots.read_trip_details(message)
        for field, value in expected.items():
            assert read.get(field) == value, f"{message!r} -> {read}"

    def test_a_pickup_change_moves_the_pickup_not_the_destination(self):
        """"change the pickup to Shimla" used to move the far end of the trip.

        The bare "to X" destination rule claimed the place, because the word that said
        which end was meant sat before the marker rather than after it.
        """
        for message in (
            "change the pickup to Shimla",
            "pickup ko Shimla kar do",
            "pickup location Shimla",
        ):
            read = slots.read_trip_details(message)
            assert read.get("origin") == "Shimla", f"{message!r} -> {read}"
            assert read.get("destination") is None, f"{message!r} -> {read}"


class TestAPlaceMentionedInPassing:
    """The guard that makes the comparison safe: no marker, no value, no change."""

    @pytest.mark.parametrize(
        "message",
        [
            "what is the weather in Manali",
            "is Kasol nice in December",
            "my friend lives in Shimla",
            "how far is Dharamshala",
        ],
    )
    def test_it_cannot_move_the_trip(self, message):
        read = slots.read_trip_details(message)
        assert read.get("destination") is None, f"{message!r} -> {read}"
        assert read.get("origin") is None, f"{message!r} -> {read}"


class TestABareReplyToTheQuestionInFlight:
    """The second shape: no marker at all, so the asked field is what disambiguates."""

    @pytest.mark.parametrize(
        "message",
        ["Kasol", "actually Kasol", "nahi Kasol", "actually make it Kasol",
         "actually, no, make it Kasol", "sorry i meant Kasol", "change it to Kasol"],
    )
    def test_it_replaces_the_asked_field(self, message):
        assert slots.read_correction_to_asked("destination", message, "Manali") == "Kasol"

    def test_restating_the_same_value_is_not_a_change(self):
        assert slots.read_correction_to_asked("destination", "actually Manali", "Manali") is None
        assert slots.read_correction_to_asked("destination", "manali", "Manali") is None

    def test_an_empty_slot_is_left_to_read_answer(self):
        """Filling is not correcting. `read_answer` owns the empty case."""
        assert slots.read_correction_to_asked("destination", "Kasol", None) is None

    def test_nothing_is_touched_when_no_question_was_asked(self):
        """`last_asked_field` is None whenever the assistant did not ask, which is
        what keeps the window one turn wide."""
        assert slots.read_correction_to_asked(None, "Kasol", "Manali") is None

    def test_a_reply_of_the_wrong_type_changes_nothing(self):
        """The parser is typed to the asked field, so it rejects rather than guesses."""
        assert slots.read_correction_to_asked("group_size", "Kasol", 4) is None
        assert slots.read_correction_to_asked("destination", "8", "Manali") is None

    @pytest.mark.parametrize("message", ["sure", "ok", "thik hai", "haan", "yes please"])
    def test_an_acknowledgement_is_not_a_correction(self, message):
        assert slots.read_correction_to_asked("destination", message, "Manali") is None

    def test_a_number_correction_is_read_the_same_way(self):
        assert slots.read_correction_to_asked("group_size", "actually 8", 4) == 8


class TestTheRouterAgreesWithThePlanner:
    """Both nodes must read a correction as an answer, or the turn is misrouted.

    The router decides the intent before the planner ever sees the message. If it reads
    "nahi, Kasol" as a topic change while the planner reads it as a correction, the
    reply comes back about something else entirely — which is the class of bug that
    made this reply nonsense in the first place.
    """

    @pytest.mark.parametrize(
        "message", ["Kasol", "nahi Kasol", "actually make it Kasol", "change it to Kasol"]
    )
    def test_a_correction_still_reads_as_an_answer(self, message):
        assert _answers_the_question("destination", message) is True

    @pytest.mark.parametrize(
        "message", ["what is the weather there", "how much will it cost", "sure"]
    )
    def test_a_genuine_topic_switch_keeps_its_own_intent(self, message):
        assert _answers_the_question("destination", message) is False


class TestWhyThisIsNotAClassifier:
    """The two properties that answer the latency question in the original request."""

    def test_detection_costs_no_model_call(self, monkeypatch):
        """Nothing in the correction path may reach for Vertex.

        Asserted by breaking the service outright: if any of this were a classifier,
        the calls below would raise instead of returning an answer.
        """
        import app.services.vertex_ai as vertex_ai

        def explode(*args, **kwargs):  # pragma: no cover - only runs on regression
            raise AssertionError("the correction path made a model call")

        monkeypatch.setattr(vertex_ai.VertexAIService, "__init__", explode, raising=False)

        assert slots.read_trip_details("ab kasol jana hai")["destination"] == "Kasol"
        assert slots.read_correction_to_asked("destination", "nahi Kasol", "Manali") == "Kasol"
        assert _answers_the_question("destination", "nahi Kasol") is True

    def test_detection_is_microseconds(self):
        """A generous ceiling — the point is the order of magnitude, not the number.

        A classifier turn is a network round trip: hundreds of milliseconds at best.
        This budget is 2 ms for a thousand messages, and the margin is what makes the
        claim safe to assert on a loaded CI box.
        """
        entities = ExtractedEntities(
            visitor_type="local",
            origin="Kullu",
            destination="Manali",
            travel_dates="kal",
            group_size=4,
        )
        messages = [
            "mujhe palampur se lekar kasol jana hai",
            "what is the weather in Manali",
            "nahi Kasol",
            "change the pickup to Shimla",
        ]
        start = time.perf_counter()
        for _ in range(250):
            for message in messages:
                slots.read_trip_details(message)
                slots.read_correction_to_asked(
                    "destination", message, entities.destination
                )
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"1000 messages took {elapsed:.3f}s"


class TestTheRentalFlowIsUntouched:
    """Corrections must not leak trip vocabulary into the rental reader.

    The three flows are kept separate on purpose — a change in one is not allowed to
    move the output of another.
    """

    def test_the_rental_reader_still_reads_a_rental_sentence(self):
        read = slots.read_rental_details("I need a car tomorrow at 9 am from Shimla for 8 hours")
        assert read.get("origin") == "Shimla"
        assert read.get("rental_hours") == 8

    def test_a_rental_pickup_change_is_read(self):
        read = slots.read_rental_details("change the pickup to Manali")
        assert read.get("origin") == "Manali"


class TestThePlannerActsOnTheChange:
    """The whole point: reading the change is worthless unless the session moves.

    Every one of these drives the real `planner_node` with the extraction model
    returning the STALE values — which is the realistic failure, because the model
    re-reports what it already believes about the conversation. If the correction only
    worked when the model caught it, it would not be a fix at all.

    A detected change is PROPOSED rather than applied — see app/agent/modification.py
    and the confirm-then-apply conversations in
    tests/unit/agent/test_conversation_flow.py. What these assert is the detection
    reaching the planner intact; that it is then put to the rider is asserted there.
    """

    @staticmethod
    def _stale_model(extraction):
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(return_value=extraction)
        return llm

    async def _plan(self, message, entities, extraction):
        from app.agent.nodes.planner import planner_node

        state = AgentState(
            session_id="correction-test",
            user_id="user-1",
            visitor_type=entities.visitor_type,
            intent="plan_trip",
            entities=entities,
            messages=[HumanMessage(content=message)],
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = self._stale_model(extraction)
            return await planner_node(state)

    @pytest.mark.asyncio
    async def test_a_sentence_moves_the_destination(self):
        from tests.conftest import wire_extraction

        stored = ExtractedEntities(
            visitor_type="local", origin="Kullu", destination="Manali",
            travel_dates="kal", group_size=4,
        )
        result = await self._plan(
            "mujhe palampur se lekar kasol jana hai",
            stored,
            # The model insists on the old trip.
            wire_extraction(destination="Manali", origin="Kullu", confidence=0.9),
        )
        assert result["entities"].pending_changes == {
            "destination": "Kasol",
            "origin": "Palampur",
        }
        # Not applied while it is being asked about.
        assert result["entities"].destination == "Manali"
        assert result["entities"].origin == "Kullu"

    @pytest.mark.asyncio
    async def test_a_bare_reply_moves_the_field_that_was_asked(self):
        from tests.conftest import wire_extraction

        stored = ExtractedEntities(
            visitor_type="local", origin="Kullu", destination="Manali",
            travel_dates="kal", group_size=4, last_asked_field="destination",
        )
        result = await self._plan(
            "nahi, Kasol",
            stored,
            wire_extraction(destination="Manali", origin="Kullu", confidence=0.9),
        )
        assert result["entities"].pending_changes == {"destination": "Kasol"}

    @pytest.mark.asyncio
    async def test_the_model_wins_on_what_is_proposed(self):
        """The text settles THAT a value changed; the model settles WHAT it changed to.

        Reading a whole conversation is the model's strength, and `parse_place` caps a
        name at three words on purpose. So the two never fight: the readers establish
        the change from the text alone, and a model that read the same reply better
        supplies the value that goes in front of the rider.
        """
        from tests.conftest import wire_extraction

        stored = ExtractedEntities(
            visitor_type="local", origin="Kullu", destination="Manali",
            travel_dates="kal", group_size=4, last_asked_field="destination",
        )
        result = await self._plan(
            "nahi, Kasol",
            stored,
            # The model already moved it, and somewhere better than a bare parse could.
            wire_extraction(destination="Kasol Village", origin="Kullu", confidence=0.9),
        )
        assert result["entities"].pending_changes == {"destination": "Kasol Village"}

    @pytest.mark.asyncio
    async def test_a_correction_counts_as_progress(self):
        """Without this, a rider actively changing details is read as going nowhere.

        `no_progress_turns` is what makes the agent stop asking and deliver. A
        correction fills no NEW slot, so it used to increment the counter and push a
        thoroughly engaged rider towards being cut off.
        """
        from tests.conftest import wire_extraction

        stored = ExtractedEntities(
            visitor_type="local", origin="Kullu", destination="Manali",
            travel_dates="kal", group_size=4, last_asked_field="destination",
            no_progress_turns=2,
        )
        result = await self._plan(
            "nahi, Kasol",
            stored,
            wire_extraction(destination="Manali", origin="Kullu", confidence=0.9),
        )
        assert result["entities"].no_progress_turns == 0

    @pytest.mark.asyncio
    async def test_a_passing_mention_leaves_the_trip_alone(self):
        """The safety half of the same mechanism, driven end to end."""
        from tests.conftest import wire_extraction

        stored = ExtractedEntities(
            visitor_type="local", origin="Kullu", destination="Manali",
            travel_dates="kal", group_size=4,
        )
        result = await self._plan(
            "what is the weather in Kasol",
            stored,
            wire_extraction(destination="Manali", origin="Kullu", confidence=0.9),
        )
        assert result["entities"].destination == "Manali"
        assert result["entities"].origin == "Kullu"
