"""
Nothing the rider reads names a state.

The questions used to be written for one state — "आप हिमाचल में किस शहर में रहते हैं?",
"Where would you like to go in Himachal Pradesh?" — and the assistant was told it was
"Himachal Pradesh's most trusted cab service", which it then repeated in its replies.
The flow is now generic ("आप किस शहर में रहते हैं?"). These tests fail the moment a
state name is written back into anything a rider sees or the model is told to say.

What is deliberately NOT covered: the internal check that treats a GPS fix inside the
state as a local rider (`Place.in_himachal`), which chooses a flow and is never shown,
and API names such as `/v1/alerts/himachal`, which the app depends on.
"""

import re
from pathlib import Path

import pytest

from app.agent import cancellation, modification, ride_reply, slots
from app.agent.nodes.writer import _build_system_prompt
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.user import UserProfile
from app.services import chat_session

_STATE_NAME = re.compile(r"himachal|हिमाचल|\bH\.?P\.?\b", re.IGNORECASE)
_FRONTEND = Path(__file__).resolve().parents[3] / "frontend"


def _strings(value):
    """Every string inside a (possibly nested) table."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _strings(v)


def _bilingual_tables(module):
    """Module-level tables holding rider-facing text: anything with an "en"/"hi" leaf."""
    for name, value in vars(module).items():
        if isinstance(value, (dict, tuple)) and any(
            isinstance(v, dict) and {"en", "hi"} & set(v)
            for v in ([value] if isinstance(value, dict) else [])
            + (list(value.values()) if isinstance(value, dict) else list(value))
        ):
            yield name, value


class TestNoStateNameReachesTheRider:
    @pytest.mark.parametrize(
        "module", [slots, modification, cancellation, ride_reply, chat_session],
        ids=lambda m: m.__name__,
    )
    def test_no_question_or_reply_table_names_it(self, module):
        offenders = [
            f"{name}: {text!r}"
            for name, table in _bilingual_tables(module)
            for text in _strings(table)
            if _STATE_NAME.search(text)
        ]
        assert offenders == []

    def test_the_greetings_and_flow_boxes_do_not(self):
        texts = list(_strings(chat_session.WELCOME_MESSAGES)) + list(
            _strings(chat_session._FLOWS)
        )
        assert [t for t in texts if _STATE_NAME.search(t)] == []

    def test_the_pickup_question_names_no_region(self):
        question = slots.SLOTS["origin"].ask("hi", slots.LOCAL, 0)
        assert question == "आपको कहाँ से पिकअप करना है?"
        assert slots.SLOTS["origin"].ask("en", slots.LOCAL, 0) == "Where should we pick you up?"

    def test_the_drop_question_names_no_region(self):
        assert slots.SLOTS["destination"].ask("hi", None, 0) == "आपको कहाँ ड्रॉप करना है?"
        assert slots.SLOTS["destination"].ask("en", None, 0) == "Where should we drop you?"

    @pytest.mark.parametrize("page", ["index.html", "local.html", "rental.html"])
    def test_the_test_clients_do_not(self, page):
        html = (_FRONTEND / page).read_text(encoding="utf-8")
        assert not _STATE_NAME.search(html)


class TestTheModelIsNotToldToSayIt:
    def test_the_profile_summary_in_the_writer_prompt_names_no_state(self):
        profile = UserProfile(user_id="u", firebase_uid="u", home_city="Shimla")
        summary = profile.to_context_string()
        assert "Shimla" in summary
        assert not _STATE_NAME.search(summary)

    @pytest.mark.parametrize("visitor_type,intent", [
        ("outstation", "plan_trip"), ("local", "plan_trip"),
        ("rental", "rental_query"), ("outstation", "general_query"),
    ])
    def test_the_writer_prompt_names_no_state(self, visitor_type, intent):
        state = AgentState(
            session_id="s", user_id="u", intent=intent, visitor_type=visitor_type,
            entities=ExtractedEntities(
                destination=None if visitor_type == "rental" else "Manali",
                origin="Delhi", home_city="Shimla" if visitor_type == "local" else None,
                rental_hours=8 if visitor_type == "rental" else None,
            ),
        )
        assert not _STATE_NAME.search(_build_system_prompt(state))
