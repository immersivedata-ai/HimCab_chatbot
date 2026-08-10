"""
tests/unit/test_conversation_flow.py
─────────────────────────────────────
Regression tests for how the agent conducts a conversation.

Three failures were reported from production, and everything here exists to keep
them fixed:

  1. "It only asks questions."   Every request — including "what's the weather in
     Manali?" — was pushed through a six-item trip checklist.
  2. "It never generates."       Six mandatory answers meant six round trips, and a
     detail the user would not give blocked the itinerary indefinitely.
  3. "It assumes things."        A free-form fallback wrote whatever the user typed
     into whichever field was next in the checklist, so "plan a trip" became a
     destination and the agent then produced an itinerary for it.

The tests are written against conversations rather than functions: each drives real
turns through the planner with a stubbed extraction model and asserts on what the
user would experience — how many questions, and whether an answer ever arrives.
"""

from collections import Counter

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage

from app.agent import slots
from app.agent.graph import route_after_planner, route_after_router
from app.agent.nodes.planner import PlannerExtraction, planner_node
from app.agent.state import AgentState
from app.core.config import settings
from app.models.session import ExtractedEntities


# ── Harness ───────────────────────────────────────────────────────────────────

def _stub_extractor(extraction):
    """Stand in for Gemini Flash. `None` simulates Vertex AI being unavailable."""
    llm = MagicMock()
    target = llm.with_structured_output.return_value
    if extraction is None:
        target.ainvoke = AsyncMock(side_effect=RuntimeError("vertex unavailable"))
    else:
        target.ainvoke = AsyncMock(return_value=PlannerExtraction(**extraction))
    return llm


class Turn:
    """One exchange, recorded for assertions."""

    def __init__(self, question: str | None, result: dict):
        self.question = question
        self.result = result

    @property
    def asked(self) -> bool:
        return self.question is not None

    @property
    def produced(self) -> bool:
        return self.question is None


class Conversation:
    """A running chat, starting from what /start actually persists.

    `chat_session._create_session` seeds `last_asked_field="destination"` because the
    welcome message asks exactly that — so turn one behaves the same here as in prod.
    """

    def __init__(self, persona: str = "outstation", intent: str = "plan_trip"):
        self.persona = persona
        self.default_intent = intent
        self.visitor_type = persona
        self.entities = ExtractedEntities(
            visitor_type=persona,
            last_asked_field="destination",
        )
        self.messages = [AIMessage(content="Where would you like to go?")]
        self.turns: list[Turn] = []

    async def say(self, text: str, extraction=None, intent: str | None = None) -> Turn:
        turn_intent = intent or self.default_intent
        self.messages.append(HumanMessage(content=text))
        state = AgentState(
            session_id="s", user_id="u", intent=turn_intent,
            messages=list(self.messages), entities=self.entities,
            visitor_type=self.visitor_type,
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = _stub_extractor(extraction)
            result = await planner_node(state)

        self.entities = result["entities"]
        self.visitor_type = result.get("visitor_type") or self.visitor_type

        routed = route_after_planner(
            AgentState(
                session_id="s", user_id="u", intent=turn_intent,
                messages=list(self.messages), entities=self.entities,
                visitor_type=self.visitor_type,
                needs_clarification=result["needs_clarification"],
            )
        )
        question = result["clarification_question"] if routed == "clarify" else None
        self.messages.append(AIMessage(content=question or "<answer>"))
        turn = Turn(question, result)
        self.turns.append(turn)
        return turn

    @property
    def questions_asked(self) -> int:
        return sum(1 for t in self.turns if t.asked)


def extraction(**kwargs) -> dict:
    """A stubbed extraction reporting exactly what the user stated — nothing more."""
    return kwargs


NOTHING: dict = {}


# ── 1. "It only asks questions" ───────────────────────────────────────────────

class TestQuestionsDoNotHijackAnswers:
    """A question about a place must be answered, not used to start an interview."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "intent,message",
        [
            ("weather_query", "What's the weather like in Manali right now?"),
            ("places_query", "What are the best places to eat in Manali?"),
            ("hotel_query", "Any good hotels in Manali?"),
        ],
    )
    async def test_a_question_about_a_known_place_is_answered_immediately(self, intent, message):
        chat = Conversation()
        turn = await chat.say(message, extraction(destination="Manali"), intent=intent)

        assert turn.produced, f"{intent} was interrogated instead of answered"
        assert chat.questions_asked == 0

    @pytest.mark.asyncio
    async def test_a_general_question_needs_nothing_at_all(self):
        """No destination, no origin, no trip — it is still answerable."""
        chat = Conversation(persona="local")
        turn = await chat.say(
            "Himachal me ghumne ka sabse acha time kaunsa hai?",
            NOTHING,
            intent="general_query",
        )

        assert turn.produced
        assert route_after_planner(
            AgentState(session_id="s", user_id="u", intent="general_query",
                       entities=chat.entities, needs_clarification=False)
        ) == "tool_executor"

    @pytest.mark.asyncio
    async def test_a_lookup_still_asks_which_place(self):
        """The one thing a place question genuinely needs is the place."""
        chat = Conversation()
        turn = await chat.say("what's the weather like?", NOTHING, intent="weather_query")

        assert turn.asked
        assert turn.result["last_asked_field"] == "destination"

    @pytest.mark.asyncio
    async def test_switching_to_a_question_mid_plan_is_answered(self):
        """The reported case: a weather question during planning restarted the checklist."""
        chat = Conversation()
        await chat.say(
            "Delhi to Shimla for 3 days, 2 people, next week",
            extraction(destination="Shimla", origin="Delhi", duration_days=3,
                       group_size=2, travel_dates="next week", origin_in_himachal=False),
        )
        turn = await chat.say(
            "is it snowing there?",
            extraction(destination="Shimla", origin="Delhi", duration_days=3,
                       group_size=2, travel_dates="next week", origin_in_himachal=False),
            intent="weather_query",
        )

        assert turn.produced
        assert chat.questions_asked == 0


# ── 2. "It never generates" ───────────────────────────────────────────────────

class TestTheAgentAlwaysGetsToAnAnswer:

    @pytest.mark.asyncio
    async def test_a_complete_request_produces_a_plan_with_no_questions(self):
        chat = Conversation()
        turn = await chat.say(
            "I want to go from Delhi to Manali for 4 days, 3 of us, next weekend",
            extraction(destination="Manali", origin="Delhi", duration_days=4,
                       group_size=3, travel_dates="next weekend", origin_in_himachal=False),
        )

        assert turn.produced
        assert chat.questions_asked == 0

    @pytest.mark.asyncio
    async def test_everything_that_shapes_the_plan_is_asked_before_the_plan(self):
        """The reported bug: the itinerary arrived, THEN it asked how many days and how
        many people — so the plan it had just written could not have used the answers."""
        chat = Conversation()
        turn = await chat.say(
            "Delhi to Manali",
            extraction(destination="Manali", origin="Delhi", origin_in_himachal=False),
        )

        assert turn.asked, "produced an itinerary before knowing how long or for whom"
        assert turn.result["last_asked_field"] == "duration_days"

    @pytest.mark.asyncio
    async def test_the_return_cab_never_blocks_the_plan(self):
        """It is a booking detail, not an itinerary input — it rides with the close."""
        chat = Conversation()
        turn = await chat.say(
            "Delhi to Manali, 4 days, 3 of us, next weekend",
            extraction(destination="Manali", origin="Delhi", duration_days=4,
                       group_size=3, travel_dates="next weekend", origin_in_himachal=False),
        )

        assert turn.produced
        assert chat.entities.return_journey_preference is None
        assert turn.result["assumed_fields"] == []

    @pytest.mark.asyncio
    async def test_answering_several_details_at_once_skips_their_questions(self):
        """Serial questions are the fallback, not a script: anything already answered
        is never asked, however it arrived."""
        chat = Conversation(persona="local")
        await chat.say("namaste", NOTHING)
        await chat.say("kasol", extraction(destination="Kasol"))
        turn = await chat.say(
            "shimla se, 2 din, 4 log, kal",
            extraction(destination="Kasol", origin="Shimla", duration_days=2,
                       group_size=4, travel_dates="kal", origin_in_himachal=True),
        )

        assert turn.produced
        assert chat.questions_asked <= 2
        assert chat.entities.home_city == "Shimla"

    @pytest.mark.asyncio
    async def test_details_are_asked_one_at_a_time_in_policy_order(self):
        """One question per turn, so each gets a focused reply. The order is the slot
        policy's, and the questions are never bundled into a single message."""
        chat = Conversation()
        asked = []

        await chat.say("Manali", extraction(destination="Manali"))
        asked.append(chat.turns[-1].result["last_asked_field"])
        await chat.say("Delhi", extraction(destination="Manali", origin="Delhi",
                                           origin_in_himachal=False))
        asked.append(chat.turns[-1].result["last_asked_field"])
        await chat.say("4 days", extraction(destination="Manali", origin="Delhi",
                                            duration_days=4, origin_in_himachal=False))
        asked.append(chat.turns[-1].result["last_asked_field"])
        await chat.say("3 people", extraction(destination="Manali", origin="Delhi",
                                              duration_days=4, group_size=3,
                                              origin_in_himachal=False))
        asked.append(chat.turns[-1].result["last_asked_field"])

        assert asked == ["origin", "duration_days", "group_size", "travel_dates"]
        # Each question is exactly one question — never a bundled list.
        for turn in chat.turns:
            if turn.asked:
                assert turn.question.count("?") == 1, turn.question

    @pytest.mark.asyncio
    async def test_serial_questions_still_reach_an_itinerary(self):
        """The whole point of asking separately is that the answers land — the run must
        still terminate in a plan, not another question."""
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        await chat.say("Delhi", extraction(destination="Manali", origin="Delhi",
                                           origin_in_himachal=False))
        await chat.say("4 days", extraction(destination="Manali", origin="Delhi",
                                            duration_days=4, origin_in_himachal=False))
        await chat.say("3 people", extraction(destination="Manali", origin="Delhi",
                                              duration_days=4, group_size=3,
                                              origin_in_himachal=False))
        turn = await chat.say("next weekend", extraction(
            destination="Manali", origin="Delhi", duration_days=4, group_size=3,
            travel_dates="next weekend", origin_in_himachal=False))

        assert turn.produced
        assert turn.result["assumed_fields"] == []

    @pytest.mark.asyncio
    async def test_a_reply_answers_only_the_question_that_was_asked(self):
        """We asked for the starting city, so "3 days" fills nothing — the date parser
        never gets a look at a reply aimed at a different field."""
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        turn = await chat.say("3 days", None)  # extractor down: code-only path

        assert turn.result["entities"].origin is None
        assert turn.result["entities"].duration_days is None

    @pytest.mark.asyncio
    async def test_the_user_can_tell_the_agent_to_just_get_on_with_it(self):
        chat = Conversation()
        await chat.say("Manali for 5 days", extraction(destination="Manali", duration_days=5))
        turn = await chat.say(
            "just make the plan, I'll fill in the details later",
            extraction(destination="Manali", duration_days=5, user_wants_to_proceed_now=True),
        )

        assert turn.produced
        assert chat.entities.proceed_requested is True
        assert "origin" in turn.result["assumed_fields"]

    @pytest.mark.asyncio
    async def test_a_user_who_will_not_answer_gets_their_plan_anyway(self):
        """Non-answers used to loop forever. Two get us to a plan, with disclosure."""
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        await chat.say("hmm", extraction(destination="Manali"))
        turn = await chat.say("ok", extraction(destination="Manali"))

        assert turn.produced
        assert set(turn.result["assumed_fields"]) == {
            "origin", "duration_days", "group_size", "travel_dates"
        }

    @pytest.mark.asyncio
    async def test_the_same_question_is_never_asked_a_third_time(self):
        chat = Conversation()
        for _ in range(5):
            await chat.say("dunno", extraction(destination="Manali"))

        asks = [t.question for t in chat.turns if t.asked]
        repeats = Counter(asks)
        assert all(n <= settings.max_clarification_attempts for n in repeats.values()), repeats

    @pytest.mark.asyncio
    async def test_giving_up_stays_given_up(self):
        """Serial questions made this reachable: the agent stopped asking, which
        cleared `last_asked_field`, which reset the patience counter, which let it
        start asking again on the very next message. Ask, ask, plan, ask, ask, plan."""
        chat = Conversation()
        for _ in range(6):
            await chat.say("dunno", extraction(destination="Manali"))

        # It gives up once and stays given up — no question after the first plan.
        first_plan = next(i for i, t in enumerate(chat.turns) if t.produced)
        assert not any(t.asked for t in chat.turns[first_plan:]), (
            "the interrogation restarted after the agent had already given up"
        )
        assert chat.questions_asked <= settings.max_clarification_attempts

    @pytest.mark.asyncio
    async def test_engaging_again_resets_the_patience_counter(self):
        """Giving up must not be permanent — a user who starts answering is followed."""
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        await chat.say("hmm", extraction(destination="Manali"))
        turn = await chat.say(
            "sorry, from Delhi",
            extraction(destination="Manali", origin="Delhi", origin_in_himachal=False),
        )

        assert turn.result["entities"].no_progress_turns == 0
        assert turn.result["entities"].origin == "Delhi"

    @pytest.mark.asyncio
    async def test_a_destination_is_never_assumed_away(self):
        """Every other detail is escapable. This one cannot be — there is nothing to
        plan without it, and inventing one is the worst possible failure."""
        chat = Conversation()
        for _ in range(4):
            turn = await chat.say("whatever you think", extraction(user_wants_to_proceed_now=True))

        assert turn.asked
        assert turn.result["last_asked_field"] == "destination"
        assert chat.entities.destination is None

    @pytest.mark.asyncio
    async def test_the_agent_still_works_when_vertex_ai_is_down(self):
        """Extraction failing degrades what the agent knows, never how it behaves.

        Every reply here is read by the code-level parsers alone — one question at a
        time is exactly the shape they handle best, since each reply has only one
        field it could be answering.
        """
        chat = Conversation()
        for reply in ("shimla", "delhi", "3 days", "4 people"):
            await chat.say(reply, None)
        turn = await chat.say("next week", None)

        assert turn.produced
        entities = chat.entities
        assert (entities.destination, entities.origin, entities.duration_days,
                entities.group_size, entities.travel_dates) == (
            "Shimla", "Delhi", 3, 4, "next week"
        )


# ── 3. "It assumes things" ────────────────────────────────────────────────────

class TestNothingIsEverInvented:

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "reply",
        ["plan a trip", "something in the hills", "I want a holiday", "somewhere nice", "idk"],
    )
    async def test_a_non_answer_never_becomes_a_destination(self, reply):
        """The exact reported bug: "plan a trip" was stored as destination "Plan A Trip"."""
        chat = Conversation()
        turn = await chat.say(reply, NOTHING)

        assert chat.entities.destination is None, f"fabricated a destination from {reply!r}"
        assert turn.asked

    @pytest.mark.asyncio
    async def test_an_unrelated_message_is_not_filed_as_an_answer(self):
        """Any text used to be accepted as the travel date, including a question."""
        chat = Conversation()
        await chat.say(
            "Delhi to Manali 3 days",
            extraction(destination="Manali", origin="Delhi", duration_days=3,
                       origin_in_himachal=False),
        )
        await chat.say("is the road open?", extraction(
            destination="Manali", origin="Delhi", duration_days=3, origin_in_himachal=False,
        ), intent="general_query")

        assert chat.entities.travel_dates is None

    @pytest.mark.asyncio
    async def test_a_reply_is_only_read_against_the_question_we_asked(self):
        """With no question in flight, a stray message changes nothing."""
        entities = ExtractedEntities(visitor_type="outstation")  # nothing asked
        merged = entities.model_dump()
        filled = slots.read_answer(None, "manali", merged, "outstation")

        assert filled is None
        assert merged["destination"] is None

    @pytest.mark.asyncio
    async def test_an_unclear_return_answer_is_left_unknown(self):
        """It used to default to "own arrangement", silently rewriting the last day."""
        merged = {"return_journey_preference": None}
        assert slots.read_answer(
            "return_journey_preference", "we'll see how it goes", merged, "outstation"
        ) is None
        assert merged["return_journey_preference"] is None

    @pytest.mark.asyncio
    async def test_what_the_agent_proceeds_without_is_reported_for_disclosure(self):
        chat = Conversation()
        await chat.say("Manali", extraction(destination="Manali"))
        await chat.say("hmm", extraction(destination="Manali"))
        turn = await chat.say("sure", extraction(destination="Manali"))

        # The writer contract: anything here MUST be stated as an assumption.
        assert turn.result["assumed_fields"]
        assert chat.entities.duration_days is None, "a guess was written into the entities"


# ── Slot parsers ──────────────────────────────────────────────────────────────

class TestAnswerParsers:
    """Biased towards rejecting: a missed answer costs one question, a wrong one
    corrupts the trip."""

    @pytest.mark.parametrize(
        "text,expected",
        [("Manali", "Manali"), ("manali", "Manali"), ("mcleod ganj", "Mcleod Ganj"),
         ("New Delhi", "New Delhi"),
         ("plan a trip", None), ("somewhere in the hills", None), ("3", None),
         ("where should I go?", None), ("yes", None), ("", None), ("a", None)],
    )
    def test_place(self, text, expected):
        assert slots.parse_place(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [("5", 5), ("5 days", 5), ("5 din", 5), ("a week", 7), ("ek hafta", 7),
         ("2 weeks", 14), ("weekend", 2),
         ("manali", None), ("how many?", None), ("0", None), ("99", None)],
    )
    def test_duration(self, text, expected):
        assert slots.parse_duration_days(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [("4", 4), ("4 people", 4), ("4 log", 4), ("four", None), ("0", None), ("kal", None)],
    )
    def test_group_size(self, text, expected):
        assert slots.parse_group_size(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [("yes", "booked"), ("haan", "booked"), ("no", "not_booked"), ("nahi", "not_booked"),
         ("we'll take the bus", "own_arrangement"), ("hmm not sure yet", None), ("", None)],
    )
    def test_return_preference(self, text, expected):
        assert slots.parse_return_preference(text) == expected

    @pytest.mark.parametrize(
        "text,expected",
        [("kal", "kal"), ("next week", "next week"), ("15 Jan", "15 Jan"),
         ("agle mahine", "agle mahine"), ("this weekend", "this weekend"),
         # travel_dates is the last field a combined question covers, so a reply that
         # answered a different one must not fall through into it.
         ("shimla", None), ("with my family", None),
         ("is it snowing?", None), ("hmm", None)],
    )
    def test_when(self, text, expected):
        assert slots.parse_when(text) == expected

    def test_each_field_only_accepts_a_reply_of_its_own_shape(self):
        """The parsers police each other's boundaries, so a reply aimed at one field
        is never filed against another."""
        blank = {"origin": None, "duration_days": None, "group_size": None,
                 "travel_dates": None}

        assert slots.read_answer("origin", "shimla", dict(blank), "local") == "origin"
        assert slots.read_answer("duration_days", "3 days", dict(blank), "local") == "duration_days"
        assert slots.read_answer("group_size", "6 people", dict(blank), "local") == "group_size"
        assert slots.read_answer("travel_dates", "next friday", dict(blank), "local") == "travel_dates"

        # …and a reply of the wrong shape fills nothing rather than something wrong.
        assert slots.read_answer("origin", "next friday", dict(blank), "local") is None
        assert slots.read_answer("origin", "3 days", dict(blank), "local") is None
        assert slots.read_answer("travel_dates", "shimla", dict(blank), "local") is None
        assert slots.read_answer("duration_days", "shimla", dict(blank), "local") is None


# ── Policy ────────────────────────────────────────────────────────────────────

class TestSlotPolicy:

    def test_only_a_destination_is_ever_mandatory(self):
        """Everything else must be escapable, or the agent can be trapped again."""
        required = {
            name
            for policy in (slots.needs_for(i) for i in
                           ("plan_trip", "weather_query", "cost_query", "general_query"))
            for name, need in policy.items()
            if need is slots.Need.REQUIRED
        }
        assert required == {"destination"}

    @pytest.mark.parametrize(
        "intent,mode",
        [("plan_trip", slots.ITINERARY_MODE), ("clarification", slots.ITINERARY_MODE),
         ("show_itinerary", slots.ITINERARY_MODE), ("cost_query", slots.ITINERARY_MODE),
         ("weather_query", slots.ANSWER_MODE), ("places_query", slots.ANSWER_MODE),
         ("hotel_query", slots.ANSWER_MODE), ("general_query", slots.ANSWER_MODE)],
    )
    def test_response_mode(self, intent, mode):
        assert slots.response_mode(intent) == mode

    def test_an_unknown_intent_falls_back_to_planning(self):
        """A router typo or a new intent must degrade to the safe, familiar flow."""
        assert slots.needs_for("brand_new_intent") == slots.needs_for("plan_trip")
        assert slots.response_mode(None) == slots.ITINERARY_MODE

    def test_every_slot_in_the_policy_actually_exists(self):
        """Guards against a typo in the policy table silently disabling a question."""
        for intent in ("plan_trip", "cost_query", "weather_query", "general_query"):
            assert set(slots.needs_for(intent)) <= set(slots.SLOTS)

    def test_every_slot_can_be_asked_in_both_languages(self):
        for name, slot in slots.SLOTS.items():
            for lang in ("en", "hi"):
                assert slot.ask(lang, "outstation", 0), f"{name} has no {lang} question"
                assert slot.ask(lang, "local", 1), f"{name} has no {lang} retry question"
                assert slots.label_for(name, lang)

    def test_a_local_is_asked_where_they_live_not_where_they_are_visiting_from(self):
        entities = ExtractedEntities(destination="Kasol", visitor_type="local")
        local = slots.decide(entities, "plan_trip", "local", "hi", max_attempts=2)
        outstation = slots.decide(entities, "plan_trip", "outstation", "hi", max_attempts=2)

        assert local.question != outstation.question
        assert "रहते" in local.question or "रहते" in slots.SLOTS["origin"].ask("hi", "local", 0)


# ── Routing ───────────────────────────────────────────────────────────────────

class TestRouting:

    def _state(self, **kwargs) -> AgentState:
        kwargs.setdefault("session_id", "s")
        kwargs.setdefault("user_id", "u")
        return AgentState(**kwargs)

    def test_show_itinerary_skips_ahead_only_when_there_is_one_to_show(self):
        with_trip = self._state(intent="show_itinerary",
                                entities=ExtractedEntities(destination="Manali"))
        without = self._state(intent="show_itinerary", entities=ExtractedEntities())

        assert route_after_router(with_trip) == "writer"
        assert route_after_router(without) == "context", (
            "jumping to the writer with no trip makes it invent one"
        )

    def test_a_plan_without_a_destination_asks_rather_than_inventing(self):
        state = self._state(intent="plan_trip", needs_clarification=False,
                            entities=ExtractedEntities())
        assert route_after_planner(state) == "clarify"

    def test_a_question_without_a_destination_is_still_answered(self):
        state = self._state(intent="general_query", needs_clarification=False,
                            entities=ExtractedEntities())
        assert route_after_planner(state) == "tool_executor"
