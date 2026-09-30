"""
tests/unit/test_reply_language.py
─────────────────────────────────
Which language the assistant replies in, and who it thinks the rider is.

The reported failure: an English-speaking rider got an English welcome, typed two or
three ordinary sentences, and the assistant switched to Hindi — and stopped producing
an itinerary. Two independent causes, both covered here:

  1. LANGUAGE. Hindi was detected by counting romanized function words, and the word
     list contained `the`, `me`, `par`, `na`, `ho`, `se` and `ki` — all ordinary
     English. "Can you plan the trip for me" scored two hits and became Hindi.

  2. PERSONA. An origin inside Himachal was read as "this rider is a resident", so a
     visitor who said they were starting from Manali was reclassified local, which
     switched them to Hindi AND to the cab-only reply with no itinerary.
"""

from typing import ClassVar

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.language import detect_reply_language
from app.agent.nodes.planner import classify_visitor_type
from app.agent.state import AgentState


def _language(*user_messages: str, persona: str = "outstation") -> str:
    messages = [AIMessage(content="Welcome to Him Cab!")]
    for text in user_messages:
        messages.append(HumanMessage(content=text))
        messages.append(AIMessage(content="<reply>"))
    return detect_reply_language(
        AgentState(session_id="s", user_id="u", visitor_type=persona, messages=messages)
    )[0]


# ── 1. An English conversation stays English ──────────────────────────────────

class TestEnglishStaysEnglish:

    @pytest.mark.parametrize(
        "message",
        [
            # Every one of these used to score two "Hindi" hits and flip the reply.
            "Can you plan the trip for me",
            "I am travelling with the family and me",
            "We want to see the snow and the mountains",
            "Plan the itinerary for me please",
            "Around the 15th, the two of us",
            "Just the two of us, me and my wife",
            "What is the best time to go to the hills",
            "The trip is for me and my parents",
        ],
    )
    def test_an_ordinary_english_sentence_is_english(self, message):
        assert _language(message) == "en"

    def test_a_whole_english_conversation_never_switches(self):
        """The reported shape: fine for a turn or two, Hindi from then on."""
        conversation = [
            "I want to plan a trip to Manali",
            "We are coming from Delhi",
            "Plan the trip for me for 4 days",
            "Just the two of us, me and my wife",
            "We want to travel in the first week of June",
        ]
        for turn in range(1, len(conversation) + 1):
            assert _language(*conversation[:turn]) == "en", (
                f"switched to Hindi at turn {turn}: {conversation[turn - 1]!r}"
            )

    def test_one_ambiguous_word_cannot_overturn_an_english_conversation(self):
        """"kal" after three English sentences is a rider talking about tomorrow, not
        switching language."""
        assert _language(
            "I want to plan a trip to Manali", "We are coming from Delhi", "4 days", "kal"
        ) == "en"


# ── 2. Hindi and Hinglish are still detected ──────────────────────────────────

class TestHindiIsStillDetected:

    @pytest.mark.parametrize(
        "message",
        [
            "mujhe manali jana hai",
            "kal 4 log ja rahe hain",
            "shimla se manali kitne din lagenge",
            "hum kal ja rahe hain",
            "kya aap plan bana sakte hain",
            "मुझे मनाली जाना है",
        ],
    )
    def test_hinglish_and_devanagari_get_hindi(self, message):
        assert _language(message, persona="local") == "hi"

    def test_devanagari_wins_regardless_of_persona(self):
        assert _language("मुझे मनाली जाना है", persona="outstation") == "hi"

    def test_a_hindi_speaker_on_the_english_screen_gets_hindi(self):
        assert _language("mujhe kal manali jana hai", persona="outstation") == "hi"


# ── 3. An explicit request beats everything ───────────────────────────────────

class TestExplicitRequests:

    def test_asking_for_english_wins_over_hinglish(self):
        assert _language("mujhe manali jana hai", "please reply in english") == "en"

    def test_asking_for_hindi_wins_over_english(self):
        assert _language("I want a trip to Manali", "hindi mein bolo") == "hi"


# ── 4. Ambiguous input follows the persona ────────────────────────────────────

class TestPersonaDefault:

    @pytest.mark.parametrize("message", ["Manali", "4", "hi", "kasol"])
    def test_a_bare_answer_follows_the_persona(self, message):
        assert _language(message, persona="local") == "hi"
        assert _language(message, persona="outstation") == "en"


# ── 5. Who the rider is ───────────────────────────────────────────────────────

class TestVisitorClassification:
    """`origin` says where the JOURNEY starts. `home_city` says where they LIVE. Only
    the second one makes somebody a resident."""

    @pytest.mark.asyncio
    async def test_an_origin_outside_the_state_makes_a_visitor(self):
        assert await classify_visitor_type(
            "Delhi", False, current="local"
        ) == "outstation"

    @pytest.mark.asyncio
    async def test_a_visitor_already_in_himachal_stays_a_visitor(self):
        """The reported bug: a tourist in Manali asking to go on to Kasol was
        reclassified as a resident, losing both English and the itinerary."""
        assert await classify_visitor_type(
            "Manali", True, current="outstation"
        ) == "outstation"

    @pytest.mark.asyncio
    async def test_a_resident_naming_an_in_state_origin_stays_a_resident(self):
        assert await classify_visitor_type("Shimla", True, current="local") == "local"

    @pytest.mark.asyncio
    async def test_a_stated_home_city_makes_a_resident_whatever_the_origin(self):
        """A Shimla resident flying out of Delhi is still a local."""
        assert await classify_visitor_type(
            "Delhi", False, home_city="Shimla", current="outstation"
        ) == "local"

    @pytest.mark.asyncio
    async def test_no_evidence_leaves_the_classification_alone(self):
        """None means "we don't know" — the caller keeps the GPS fix or the front end's
        answer rather than the planner inventing one."""
        assert await classify_visitor_type(None) is None
        assert await classify_visitor_type("Narnia") is None


# ── 6. The two failures together ──────────────────────────────────────────────

class TestTheReportedConversation:

    @pytest.mark.asyncio
    async def test_a_visitor_starting_inside_himachal_keeps_english_and_the_itinerary(self):
        """End of the causal chain: persona decides both the language default and
        whether an itinerary is produced at all."""
        from app.agent import slots

        persona = await classify_visitor_type("Manali", True, current="outstation")

        assert persona == "outstation"
        assert slots.response_mode("plan_trip", persona) == slots.ITINERARY_MODE
        assert _language("I want to go to Kasol", persona=persona) == "en"


# ── 7. A short answer is not a language sample ────────────────────────────────

class TestAnAnswerDoesNotChangeTheLanguage:
    """Reported: "all of a sudden started giving response in English instead of Hindi".

    The assistant asks a local rider its questions in Hindi. They answer the way anyone
    answers a form — "4 people", "2 of us", "tomorrow at 5" — and every one of those is
    mostly English FUNCTION words, whoever is typing them. Those words were counted as
    evidence, so the reply that a Hindi question had just solicited was itself what
    switched the conversation to English, one turn after it started.

    Two things caused it, and both are covered here:

      1. the threshold was checked against the WEIGHTED total, and the newest message
         carries the highest weight — so a single "of" scored enough to clear a bar
         meant to require two separate words
      2. short replies voted at all
    """

    ANSWERS: ClassVar[list] = [
        "4 people", "2 of us", "tomorrow at 5", "at 6 am", "from my home",
        "3 of us", "for 2 days",
    ]

    @pytest.mark.parametrize("answer", ANSWERS)
    def test_a_local_answering_a_question_stays_in_hindi(self, answer):
        assert _language("Manali", answer, persona="local") == "hi"

    @pytest.mark.parametrize("answer", ANSWERS)
    def test_the_same_answer_leaves_a_visitor_in_english(self, answer):
        """Symmetric: it is not evidence either way, so the persona still decides."""
        assert _language("Manali", answer, persona="outstation") == "en"

    def test_a_whole_local_conversation_of_short_answers_stays_hindi(self):
        """The reported shape, turn by turn: Hindi welcome, then it drifts."""
        conversation = ["Manali", "Kullu se", "4 people", "tomorrow at 5", "2 of us"]
        for turn in range(1, len(conversation) + 1):
            assert _language(*conversation[:turn], persona="local") == "hi", (
                f"switched to English at turn {turn}: {conversation[turn - 1]!r}"
            )

    def test_real_english_prose_still_switches_a_local(self):
        """The guard is about ANSWERS, not about refusing to listen. A local who
        actually writes English gets English."""
        assert _language(
            "I would like to book a cab to Manali for my family", persona="local"
        ) == "en"

    def test_an_explicit_request_still_wins_from_a_local(self):
        assert _language("Manali", "please reply in english", persona="local") == "en"

    def test_devanagari_in_a_short_answer_is_still_hindi(self):
        """Script is unambiguous at any length — it never needed the word count."""
        assert _language("मनाली", persona="outstation") == "hi"


# ── 8. A local is booking a ride, so it is never asked how many DAYS ──────────

class TestALocalIsNeverAskedHowManyDays:
    """Reported: the local flow asks "how many days are you travelling for?".

    `response_mode` already returns RIDE_MODE for a resident — the reply is a cab, a
    fare, a drive time and the weather, with no itinerary in it. But the QUESTIONS were
    keyed on the intent alone, and a local and a visitor both arrive as `plan_trip`, so
    the resident was walked through the itinerary checklist to produce a reply that
    could not use the answers.
    """

    def test_duration_is_not_asked_of_a_local(self):
        from app.agent import slots

        needs = slots.needs_for("plan_trip", "local")
        assert "duration_days" not in needs

    def test_a_visitor_is_still_asked_it(self):
        from app.agent import slots

        assert "duration_days" in slots.needs_for("plan_trip", "outstation")

    @pytest.mark.parametrize("intent", ["plan_trip", "clarification", "cost_query"])
    def test_no_travel_intent_asks_a_local_for_days(self, intent):
        from app.agent import slots

        assert "duration_days" not in slots.needs_for(intent, "local")

    def test_the_ride_still_asks_what_it_actually_needs(self):
        """Where to, where from, how many, and when — each one feeds the fare or the
        cab. Dropping the days must not quietly drop these."""
        from app.agent import slots

        needs = slots.needs_for("plan_trip", "local")
        assert needs["destination"] is slots.Need.REQUIRED
        for field in ("origin", "group_size", "travel_dates"):
            assert needs[field] is slots.Need.IMPORTANT

    def test_a_local_asking_about_the_weather_is_still_asked_nothing_but_the_place(self):
        """The ride table is scoped to travel intents — a lookup keeps its own."""
        from app.agent import slots

        assert list(slots.needs_for("weather_query", "local")) == ["destination"]

    def test_the_question_sequence_a_local_actually_sees(self):
        from app.agent import slots
        from app.domain.conversation import ExtractedEntities

        entities = ExtractedEntities(
            visitor_type="local", destination="Manali", origin="Kullu"
        )
        asked = []
        for _ in range(6):
            decision = slots.decide(
                entities, "plan_trip", "local", "hi", max_attempts=2
            )
            if not decision.ask_field:
                break
            asked.append(str(decision.ask_field))
            setattr(entities, decision.ask_field,
                    {"group_size": 3, "travel_dates": "kal"}[decision.ask_field])

        assert asked == ["group_size", "travel_dates"]
