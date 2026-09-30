"""The wording and the reading behind a mid-conversation change.

The conversations in tests/unit/agent/test_conversation_flow.py cover the flow — a
change is proposed, confirmed, then applied. This covers the pieces it is built from:
which messages count as asking for a change, how a yes is told from a no, and that
every line the rider can be shown exists in both languages.
"""

import pytest

from app.agent import modification
from app.core.enums import ReplyLanguage
from app.domain.conversation import ExtractedEntities


def _entities(**overrides) -> ExtractedEntities:
    values = {
        "visitor_type": "local",
        "origin": "Dadh",
        "destination": "Kangra",
        "travel_dates": "kal",
        "group_size": 2,
    }
    values.update(overrides)
    return ExtractedEntities(**values)


class TestSpottingAChangeRequestWithNoFieldNamed:

    @pytest.mark.parametrize(
        "message",
        [
            "I want to change something",
            "can you modify my booking details",
            "mujhe kuch badalna hai",
            "badal do",
            "that's wrong",
            "galat ho gaya",
            "wait",
            "ek minute",
            "hold on",
            "I made a mistake",
        ],
    )
    def test_it_is_recognised(self, message):
        assert modification.wants_to_change_something(message) is True

    @pytest.mark.parametrize(
        "message",
        [
            # Asking ABOUT changing is a question about the policy. Answering it by
            # opening a modification flow is the trap `cancellation` guards against
            # with "can I cancel later?".
            "can I change the date later?",
            "kya main baad mein badal sakta hoon",
            "how do I change it after booking",
            "no change needed",
            "kuch nahi badalna",
            # Ordinary messages that share a word or two.
            "exchange rate",
            "what is the weather in Manali",
            "book it",
            "",
        ],
    )
    def test_it_is_not(self, message):
        assert modification.wants_to_change_something(message) is False


class TestReadingTheAnswerToAConfirmation:

    @pytest.mark.parametrize(
        "message", ["yes", "haan", "ji haan", "ok", "sure", "kar do", "bilkul", "हाँ"]
    )
    def test_yes(self, message):
        assert modification.reads_as_confirmation(message) is True

    @pytest.mark.parametrize(
        "message", ["no", "nahi", "nope", "rehne do", "नहीं", "mat karo"]
    )
    def test_no(self, message):
        assert modification.reads_as_confirmation(message) is False

    @pytest.mark.parametrize(
        "message",
        [
            # The important one: a reply that carries a DIFFERENT value is neither.
            # Read as "no" it would throw the change away; read as "yes" it would apply
            # the wrong one. Both are wrong, so it falls through to the slot readers.
            "nahi, mujhe kullu jaana hai",
            "no, make it Kasol instead",
            "actually change the pickup to Shimla",
            "what will it cost",
            "",
        ],
    )
    def test_neither(self, message):
        assert modification.reads_as_confirmation(message) is None

    def test_a_long_reply_is_never_read_as_a_bare_yes_or_no(self):
        """Length is the guard. "nahi" in front of a sentence is punctuation, not the
        answer — the sentence after it is what the rider means."""
        assert modification.reads_as_confirmation("nahi nahi manali hi theek hai") is None


class TestTheConfirmationText:

    def test_it_names_both_sides(self):
        """The value WE read is the thing most likely to be wrong, and the rider is the
        only one who can see that. Showing only the new value hides the misparse."""
        text = modification.confirm_question(
            {"destination": "Dharamshala"}, _entities(), ReplyLanguage.ENGLISH
        )
        assert "Kangra" in text and "Dharamshala" in text
        assert "destination" in text

    def test_several_changes_are_listed(self):
        text = modification.confirm_question(
            {"origin": "Palampur", "destination": "Kasol"},
            _entities(),
            ReplyLanguage.ENGLISH,
        )
        for value in ("Dadh", "Palampur", "Kangra", "Kasol"):
            assert value in text
        # Pickup before destination — the order a rider thinks about the journey.
        assert text.index("Palampur") < text.index("Kasol")

    def test_the_misparse_from_the_report_is_visible_in_it(self):
        """The whole argument for confirming. Had this question been asked, the rider
        would have seen the nonsense instead of a booking screen built on it."""
        text = modification.confirm_question(
            {"origin": "Mujhe Palampur", "destination": "Lekar Kasol"},
            _entities(),
            ReplyLanguage.ENGLISH,
        )
        assert "Mujhe Palampur" in text and "Lekar Kasol" in text


class TestTheWhichDetailText:

    def test_it_lists_what_we_hold(self):
        """"What would you like to change?" on its own makes the rider remember what
        they said. Showing it back turns the question into one they can point at."""
        text = modification.which_detail_question(_entities(), ReplyLanguage.ENGLISH)
        for value in ("Dadh", "Kangra", "kal", "2"):
            assert value in text

    def test_an_empty_session_still_gets_a_usable_reply(self):
        text = modification.which_detail_question(
            ExtractedEntities(visitor_type="local"), ReplyLanguage.ENGLISH
        )
        assert text.strip()
        assert "change" in text.lower()

    def test_the_card_carries_only_the_details_that_are_set(self):
        event = modification.which_detail_event(
            _entities(pickup_time=None), ReplyLanguage.ENGLISH
        )
        fields = {row["field"] for row in event["payload"]["fields"]}
        assert fields == {"origin", "destination", "travel_dates", "group_size"}


class TestBothLanguagesAreComplete:
    """A missing key is a rider shown an English line mid-Hindi conversation, or a
    KeyError. One table, checked, so a wording change lands in both or fails here."""

    def test_every_line_exists_in_both(self):
        for key, table in modification._TEXT.items():
            assert set(table) == {"en", "hi"}, key
            assert table["en"].strip() and table["hi"].strip(), key

    def test_every_label_exists_in_both(self):
        for field, table in modification._LABELS.items():
            assert set(table) == {"en", "hi"}, field

    def test_the_placeholders_match_across_languages(self):
        """A `{new}` present in one and absent in the other is a line that formats in
        English and silently drops the value in Hindi."""
        import re

        for key, table in modification._TEXT.items():
            placeholders = {
                lang: set(re.findall(r"{(\w+)}", text)) for lang, text in table.items()
            }
            assert placeholders["en"] == placeholders["hi"], key

    @pytest.mark.parametrize("lang_key", ["en", "hi"])
    def test_every_public_line_renders(self, lang_key):
        entities = _entities()
        changes = {"destination": "Dharamshala", "group_size": 5}
        assert modification.confirm_question(changes, entities, lang_key).strip()
        assert modification.which_detail_question(entities, lang_key).strip()
        assert modification.applied_message(changes, lang_key).strip()
        assert modification.kept_message(lang_key).strip()

    def test_an_unknown_language_falls_back_to_english(self):
        text = modification.confirm_question(
            {"destination": "Kasol"}, _entities(), "fr"
        )
        assert "destination" in text


class TestSayingWhatChanged:

    def test_one_change_names_the_field_and_the_new_value(self):
        text = modification.applied_message(
            {"destination": "Dharamshala"}, ReplyLanguage.ENGLISH
        )
        assert "destination" in text and "Dharamshala" in text

    def test_several_are_listed(self):
        text = modification.applied_message(
            {"origin": "Palampur", "destination": "Kasol"}, ReplyLanguage.ENGLISH
        )
        assert "Palampur" in text and "Kasol" in text


class TestNoneOfThisCallsAModel:
    """Same property the detection has, for the same reason: this runs on the turn a
    rider is waiting on, and it must work with Vertex unavailable."""

    def test_the_whole_module_is_tables_and_comparisons(self, monkeypatch):
        import app.services.vertex_ai as vertex_ai

        def explode(*args, **kwargs):  # pragma: no cover - only runs on regression
            raise AssertionError("the modification path made a model call")

        monkeypatch.setattr(vertex_ai.VertexAIService, "__init__", explode, raising=False)

        entities = _entities()
        assert modification.wants_to_change_something("mujhe kuch badalna hai") is True
        assert modification.reads_as_confirmation("haan") is True
        assert modification.confirm_question({"destination": "Kasol"}, entities, "hi")
        assert modification.which_detail_question(entities, "hi")
        assert modification.applied_message({"destination": "Kasol"}, "hi")
