"""Dates a rider names relative to today.

Reported as: the assistant takes "today" but not "yesterday", "day before yesterday"
or "day after tomorrow". Probing it turned up four separate faults, and there is a
class of test here for each.

  1. A SILENT OFF-BY-ONE. "day after tmrw" resolved to +2 in the validator but was READ
     out of a sentence as plain "tmrw", because the resolver's phrase table and the
     sentence reader's list of day words were maintained separately and had drifted.
     The rental was booked for Wednesday when the rider said Thursday. One table now,
     and `TestTheReadersAllUseOneTable` keeps it that way.

  2. COUNTED PHRASES DID NOT RESOLVE AT ALL. "3 din baad", "in 2 days", "2 days later"
     reached the rental booking screen's date field as those words — the exact bug
     `sanitize` was written to stop for "tomorrow", still open for every other way of
     saying the same thing.

  3. A PAST DAY WAS DROPPED IN SILENCE. Refusing it is right; nobody hires a car for
     yesterday. Doing it without a word was not: the slot went empty and the rider got
     "when are you planning to travel?" back verbatim, which from their side is
     indistinguishable from not having been understood. That is what was reported.

  4. "abhi" / "asap" / "turant" NAMED NO DAY. A rider who wants a car now is booking
     for today.

The dates here are computed from a fixed TODAY, never from the clock, so the suite does
not change meaning overnight.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from app.agent import sanitize, slots

#: A Monday, so a weekday-adjacent bug shows up as a weekday rather than by luck.
TODAY = date(2026, 9, 21)


def day(offset: int) -> str:
    return (TODAY + timedelta(days=offset)).isoformat()


class TestTheDaysARiderNames:
    """Every phrase, to the day it means. The table is the specification."""

    @pytest.mark.parametrize(
        "phrase,offset",
        [
            # Today, including the "I want it now" family that named no day at all.
            ("today", 0), ("aaj", 0), ("tonight", 0), ("tdy", 0), ("aaj hi", 0),
            ("abhi", 0), ("right now", 0), ("asap", 0), ("turant", 0),
            ("immediately", 0), ("आज", 0), ("अभी", 0),
            # Tomorrow.
            ("tomorrow", 1), ("tommorow", 1), ("tomarrow", 1), ("tmrw", 1),
            ("kal", 1), ("agle din", 1), ("कल", 1),
            # The day after. Every spelling of tomorrow needs its own entry, or the
            # shorter one wins and the cab comes a day early.
            ("day after tomorrow", 2), ("day after tmrw", 2), ("day after tommorow", 2),
            ("day after kal", 2), ("parso", 2), ("परसों", 2),
            ("day after next", 2), ("next to next day", 2),
            # Three out.
            ("narso", 3),
            # Counted, in both languages and in digits or words.
            ("in 2 days", 2), ("after 3 days", 3), ("2 days later", 2),
            ("3 days from now", 3), ("3 din baad", 3), ("do din baad", 2),
            ("teen din baad", 3), ("ek din baad", 1), ("das din baad", 10),
            ("5 दिन बाद", 5),
            # Past. Recognised so it can be refused BY NAME — see the class below.
            ("yesterday", -1), ("day before yesterday", -2), ("beeta kal", -1),
        ],
    )
    def test_it_resolves_to_the_right_calendar_day(self, phrase, offset):
        assert sanitize.resolve_relative_day(phrase, today=TODAY) == TODAY + timedelta(
            days=offset
        )

    @pytest.mark.parametrize(
        "phrase,offset",
        [("day after", 2), ("next day", 1)],
    )
    def test_a_fragment_answers_only_when_it_is_the_whole_answer(self, phrase, offset):
        """"day after" is a complete reply to "when?" and also half an English clause.

        As the whole answer it means the day after tomorrow. Buried in a sentence
        ("the day after we reach we go to Kasol") it is not a date at all, and reading
        it as one files a travel date off a sentence about something else.
        """
        assert sanitize.resolve_relative_day(phrase, today=TODAY) == TODAY + timedelta(
            days=offset
        )
        assert slots.find_relative_day(f"the {phrase} we reach we go to Kasol") is None


class TestPhrasesThatNameNoSingleDay:
    """Left alone, on purpose. A rough answer is still the rider's answer, and inventing
    a date for one would be worse than carrying the words into the reply."""

    @pytest.mark.parametrize(
        "phrase",
        ["next week", "this weekend", "next month", "sometime after Diwali",
         "ek hafte baad", "agle mahine", "15 Jan", "25/12"],
    )
    def test_no_date_is_invented(self, phrase):
        assert sanitize.resolve_relative_day(phrase, today=TODAY) is None
        # …and the phrase survives, rather than being dropped as unreadable.
        assert sanitize.travel_date(phrase, today=TODAY) == phrase


class TestSentencesAboutSomethingElse:
    """The cost of widening a date reader is that it starts seeing dates everywhere.

    Every one of these was a real false positive during the change. A travel date filed
    off one of them is a wrong date nobody typed — worse than the miss it replaced,
    because the rider has no reason to look for it.
    """

    @pytest.mark.parametrize(
        "message",
        [
            # "now" as an ordinary adverb. This one shipped: it turned "we are 5 people
            # now" into a proposed change of travel date.
            "we are 5 people now", "book it now", "I am in Delhi now",
            "jaldi karo", "subah jaldi",
            # "aaj kal" is the idiom for "these days", not today.
            "aaj kal bahut traffic hai", "आजकल बहुत traffic hai",
            # Clause fragments.
            "the day after we reach we go to Kasol", "next day return",
            # Durations. "3 din" is a trip LENGTH; only a future marker makes it a date,
            # and "3 din se" is the PAST — "for the last three days".
            "3 din ke liye", "4 days trip", "next 3 days", "for 2 days",
            "3 din se ruka hun", "do din se wait kar raha hu",
            # Ordinary traffic that happens to contain a count or a day word.
            "das rupay", "do log hain", "kitne din lagenge", "8 hours ke liye",
        ],
    )
    def test_no_date_is_read_out_of_it(self, message):
        assert slots.find_relative_day(message) is None
        assert slots.read_trip_details(message).get("travel_dates") is None


class TestTheReadersAllUseOneTable:
    """The off-by-one was not a missing phrase. It was two lists that disagreed.

    `sanitize` could resolve "day after tmrw"; the sentence reader had no pattern for it
    and matched "tmrw" instead. Both were individually defensible and the pair booked
    the wrong day. So: anything the resolver understands, every other reader must
    understand identically.
    """

    @pytest.mark.parametrize("phrase", sorted(slots.RELATIVE_DAY_PHRASES))
    def test_a_phrase_that_resolves_is_also_accepted_as_an_answer(self, phrase):
        """`parse_when` used to turn away "yesterday" and "abhi" — days the validator
        could resolve perfectly well. The value never reached the validator, so nothing
        could explain the problem and the rider just got the question again."""
        assert slots.parse_when(phrase) == phrase

    @pytest.mark.parametrize("phrase", sorted(slots.RELATIVE_DAY_PHRASES))
    def test_a_phrase_that_resolves_is_read_whole_out_of_a_sentence(self, phrase):
        """The off-by-one itself: the sentence reader must not return a SHORTER phrase
        than the one that is there, because a prefix of a day phrase is a different
        day."""
        found = slots.find_relative_day(f"mujhe {phrase} jana hai")
        assert found is not None, f"{phrase!r} is resolvable but invisible in a sentence"
        assert found[0] == phrase
        assert found[1] == slots.RELATIVE_DAY_PHRASES[phrase]

    def test_the_sentence_reader_agrees_with_the_resolver_on_every_phrase(self):
        """Stated once over the whole table, so a new entry is covered the day it is
        added rather than the day someone remembers to write a test."""
        for phrase, offset in slots.RELATIVE_DAY_PHRASES.items():
            sentence = f"Dadh se Kangra {phrase}"
            read = slots.read_trip_details(sentence).get("travel_dates")
            assert read is not None, f"{phrase!r} vanished from a sentence"
            assert sanitize.resolve_relative_day(read, today=TODAY) == TODAY + timedelta(
                days=offset
            ), f"{phrase!r} reads as {read!r}, which is a different day"

    def test_tidying_a_sentence_keeps_the_whole_phrase(self):
        """`tidy_when` reduces an answer to the day inside it for the rental booking
        screen. Reducing "day after tmrw" to "tmrw" is how the wrong date shipped."""
        assert slots.tidy_when("I want the car day after tmrw") == "day after tmrw"
        assert slots.tidy_when("mujhe 3 din baad chahiye") == "3 din baad"


class TestADayThatHasAlreadyGone:
    """Refused — but by name, which is the whole point.

    Nobody hires a car for yesterday, so dropping the value is right and stays. What
    changed is the silence around it: the rider used to get the identical "when are you
    planning to travel?" back, with nothing to say their answer had been read at all.
    """

    @pytest.mark.parametrize(
        "phrase", ["yesterday", "day before yesterday", "beeta kal"]
    )
    def test_it_is_never_booked(self, phrase):
        assert sanitize.travel_date(phrase, today=TODAY) is None
        assert sanitize.travel_date(phrase, as_iso=True, today=TODAY) is None

    @pytest.mark.parametrize(
        "phrase", ["yesterday", "day before yesterday", "beeta kal"]
    )
    def test_it_is_reported_in_the_riders_own_words(self, phrase):
        """Quoted back, not normalised: the question says which answer could not be
        used, and only their own wording tells them that."""
        assert sanitize.day_already_gone(phrase, today=TODAY) == phrase

    @pytest.mark.parametrize(
        "phrase", ["today", "tomorrow", "parso", "3 din baad", "next week", "15 Jan"]
    )
    def test_a_usable_day_is_not_reported_as_gone(self, phrase):
        assert sanitize.day_already_gone(phrase, today=TODAY) is None

    def test_today_itself_is_bookable(self):
        """The boundary. `<` not `<=` — a cab for later today is an ordinary booking,
        and off-by-one here would refuse the single most common answer there is."""
        assert sanitize.day_already_gone("today", today=TODAY) is None
        assert sanitize.travel_date("today", as_iso=True, today=TODAY) == day(0)


class TestTheQuestionThatGoesBack:

    @pytest.mark.parametrize("lang_key", ["en", "hi"])
    def test_it_quotes_the_day_that_could_not_be_used(self, lang_key):
        text = slots.when_question(lang_key, 0, past_day="yesterday")
        assert "yesterday" in text

    @pytest.mark.parametrize("lang_key", ["en", "hi"])
    def test_it_outranks_both_ordinary_wordings(self, lang_key):
        """A rider who answered precisely is owed a straight answer about THAT answer —
        not the original question, and not a hint about what a date looks like."""
        plain = slots.when_question(lang_key, 0)
        retry = slots.when_question(lang_key, 1)
        past = slots.when_question(lang_key, 1, past_day="yesterday")
        assert past != plain and past != retry

    @pytest.mark.parametrize("lang_key", ["en", "hi"])
    def test_it_still_offers_a_way_forward(self, lang_key):
        """Naming the problem is half of it. A question that only says no leaves the
        rider to guess the format."""
        text = slots.when_question(lang_key, 0, past_day="yesterday")
        assert ("today" in text.lower()) or ("आज" in text)

    def test_the_ordinary_wordings_are_untouched(self):
        assert slots.when_question("en", 0) != slots.when_question("en", 1)
        assert slots.when_question("en", 0).strip()
        assert slots.when_question("hi", 1).strip()


# ── The conversation ──────────────────────────────────────────────────────────

class TestWhatTheRiderActuallySees:
    """Driving the real planner and the real clarify node.

    The unit tests above prove the wording exists. These prove it REACHES the rider,
    which is the part that was broken: the phrase was resolved, refused, and then the
    refusal was never spoken.
    """

    @staticmethod
    async def _say(text: str, extraction: dict, *, persona: str = "local",
                   intent: str = "plan_trip", entities=None):
        from unittest.mock import AsyncMock, MagicMock, patch

        from langchain_core.messages import AIMessage, HumanMessage

        from app.agent.graph import clarify_node, route_after_planner
        from app.agent.nodes.planner import PlannerExtraction, planner_node
        from app.agent.state import AgentState
        from app.domain.conversation import ExtractedEntities

        entities = entities or ExtractedEntities(
            visitor_type=persona, origin="Dadh", destination="Kangra",
            group_size=2, last_asked_field="travel_dates",
        )
        messages = [
            AIMessage(content="When are you planning to travel?"),
            HumanMessage(content=text),
        ]
        state = AgentState(session_id="s", user_id="u", intent=intent,
                           messages=messages, entities=entities, visitor_type=persona)
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction(**extraction)
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        after = AgentState(
            session_id="s", user_id="u", intent=intent, messages=messages,
            entities=result["entities"], visitor_type=persona,
            needs_clarification=result["needs_clarification"],
            clarification_question=result["clarification_question"],
            last_asked_field=result["last_asked_field"],
            past_travel_date=result.get("past_travel_date"),
        )
        question = None
        if route_after_planner(after) == "clarify":
            question = (await clarify_node(after))["final_response_text"]
        return result, question

    @pytest.mark.asyncio
    @pytest.mark.parametrize("phrase", ["yesterday", "day before yesterday"])
    async def test_a_day_that_has_gone_is_named_not_silently_dropped(self, phrase):
        """The report, in one assertion: the rider answered, and the answer came back
        to them. Before this, they got the question they had just answered."""
        result, question = await self._say(
            phrase,
            {"destination": "Kangra", "origin": "Dadh", "group_size": 2,
             "travel_dates": phrase, "origin_in_himachal": True},
        )

        assert result["entities"].travel_dates is None, "a past day must never be booked"
        assert result["past_travel_date"] == phrase
        assert question is not None
        assert phrase in question, (
            "the rider was asked again without being told which answer failed — which "
            "is indistinguishable from not having been understood"
        )
        assert question != "आप कब यात्रा करना चाहते हैं?"

    @pytest.mark.asyncio
    async def test_the_next_answer_is_taken_normally(self):
        """The refusal must not poison the slot. A rider who says "yesterday" and then
        "kal" has given a perfectly good date on their second try."""
        from app.domain.conversation import ExtractedEntities

        first, _ = await self._say(
            "yesterday",
            {"destination": "Kangra", "origin": "Dadh", "group_size": 2,
             "travel_dates": "yesterday", "origin_in_himachal": True},
        )
        second, question = await self._say(
            "kal",
            {"destination": "Kangra", "origin": "Dadh", "group_size": 2,
             "travel_dates": "kal", "origin_in_himachal": True},
            entities=first["entities"],
        )
        assert second["entities"].travel_dates == "kal"
        assert second["past_travel_date"] is None
        assert question is None, "asked again after a usable date"
        assert isinstance(first["entities"], ExtractedEntities)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "phrase,offset",
        [("day after tmrw", 2), ("3 din baad", 3), ("in 2 days", 2), ("abhi", 0),
         ("parso", 2), ("do din baad", 2)],
    )
    async def test_a_rental_booking_screen_gets_a_real_date(self, phrase, offset):
        """The field the app renders. A phrase here is the original reported bug, and a
        phrase one day off is worse — a car arrives on the wrong morning."""
        from datetime import date as _date

        from app.agent import slots as _slots
        from app.domain.conversation import ExtractedEntities

        result, _ = await self._say(
            phrase,
            {"origin": "Shimla", "rental_hours": 8, "group_size": 4,
             "travel_dates": phrase},
            persona=_slots.RENTAL, intent="rental_query",
            entities=ExtractedEntities(
                visitor_type=_slots.RENTAL, origin="Shimla", rental_hours=8,
                pickup_time="09:00", group_size=4, last_asked_field="travel_dates",
            ),
        )
        booked = result["entities"].travel_dates
        assert booked == (sanitize.today_ist() + timedelta(days=offset)).isoformat(), (
            f"{phrase!r} put {booked!r} in the booking screen's date field"
        )
        _date.fromisoformat(booked)   # renderable, which is the whole requirement
