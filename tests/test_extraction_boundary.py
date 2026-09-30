"""The contract between the planner and Vertex, and what happens when it is not met.

Reported as: "the AI is hallucinating a lot — it assumed 3 days, it didn't route me to
the booking screen, and this mainly happens when I type in sentences."

None of those were writer bugs. The extraction model was returning NOTHING on about half
of all live calls, and had been for as long as anyone had looked:

    planner_extraction_failed  error='NoneType' object is not iterable

`Optional[str]` compiles to `anyOf: [{"type":"string"},{"type":"null"}]`. Vertex's
`Schema` proto has no `anyOf` — in `json_mode` it raises, and in the default
`function_calling` mode it silently mangles the schema, the model returns no tool call,
and `with_structured_output` hands back `None`. `dict(None)` then raised the TypeError.

Measured on a real conversation, 25 calls per configuration:

    Optional fields + function_calling      12-19 failures
    flat scalars   + function_calling       19 failures
    flat scalars   + json_mode               0 failures

The node degraded exactly as designed on every failure, which is why this survived: it
kept working on the deterministic readers in `slots` alone. Those handle short answers
well and whole sentences only partly — so the assistant looked fine when a rider typed
"Kasol" and lost the plot when they typed "mei kasol jana chahta hu". Hence "this
happens when I type in sentences".

The suite stayed green throughout, because every test stubbed the boundary with
`PlannerExtraction` — a shape the real system never returns. That is the deeper lesson
here, and `tests/conftest.py::wire_extraction` exists so it cannot be repeated.
"""

from __future__ import annotations

import json

import pytest

from app.agent.nodes.planner import (
    _WIRE_BLANKS,
    PlannerExtraction,
    _VertexExtraction,
)
from app.domain.itinerary import BudgetRange, GroupType
from tests.conftest import stub_vertex


class TestTheWireSchemaIsSomethingVertexCanAccept:
    """Asserted on the SCHEMA, not on a live call, so it fails in CI rather than in
    production at three in the morning."""

    def test_it_contains_no_anyof(self):
        """The whole defect in one assertion. `anyOf` is what Vertex cannot represent,
        and an `Optional[...]` field added here would reintroduce it — silently, since
        function calling reports nothing."""
        schema = json.dumps(_VertexExtraction.model_json_schema())
        assert "anyOf" not in schema, (
            "a field on _VertexExtraction is Optional (or a union). Vertex drops what "
            "it cannot represent and the model then returns no object at all — about "
            "half the time, with no error anywhere but one log line."
        )

    def test_it_contains_no_nested_definitions(self):
        """`$defs` is dropped by Vertex too, with a warning nobody reads. Enums are the
        usual way one appears — hence `Literal` below rather than the domain enums."""
        assert not _VertexExtraction.model_json_schema().get("$defs")

    def test_no_enum_value_is_the_empty_string(self):
        """The second Vertex rule, and the one that reached production.

        A plain string field may default to "", but an ENUM MEMBER MAY NOT BE EMPTY:

            400 ...response_schema.properties[origin_in_himachal].enum[0]: cannot be empty

        The first version of the wire model used "" as the sentinel inside `Literal`
        too. Vertex rejected the whole schema, LangChain retried it, and a four-turn
        conversation logged 39-45 of those 400s — roughly six and a half wasted requests
        per extraction that got through. On Dynamic Shared Quota that is six times the
        volume into a pool that answers 429 when it is busy, and production started
        doing exactly that.

        Unlike the `anyOf` case this one DOES raise, so it is loud once it reaches
        Vertex — but it reaches Vertex only at runtime, and LangChain's retry buries it
        in a warning. Asking the schema is how it fails in CI instead.
        """
        for name, prop in _VertexExtraction.model_json_schema()["properties"].items():
            assert "" not in (prop.get("enum") or []), (
                f"{name} has an empty enum value; Vertex rejects the whole request "
                f"with 'enum[0]: cannot be empty'. Use a word like 'unknown' and map "
                f"it back in to_extraction()."
            )

    def test_every_field_is_a_flat_scalar_or_a_list_of_strings(self):
        allowed = {"string", "integer", "number", "boolean", "array"}
        for name, prop in _VertexExtraction.model_json_schema()["properties"].items():
            kind = prop.get("type") or ("string" if "enum" in prop else None)
            assert kind in allowed, f"{name} is {prop}, which Vertex may not accept"

    def test_the_description_sent_to_vertex_stays_small(self):
        """Pydantic ships the class docstring as the schema description, so an essay
        there is re-sent on every extraction call. The explanation lives in a comment
        above the class for exactly this reason."""
        schema = _VertexExtraction.model_json_schema()
        assert len(schema.get("description", "")) < 200


class TestTheChoicesMatchTheDomain:
    """The wire model spells its enums as `Literal` to avoid `$defs`. That is a
    transport decision, and it must not quietly drift from the domain types."""

    def test_group_type_offers_exactly_the_domain_values(self):
        literal = set(
            _VertexExtraction.model_json_schema()["properties"]["group_type"]["enum"]
        )
        assert literal == {"unknown"} | {g.value for g in GroupType}

    def test_budget_range_offers_exactly_the_domain_values(self):
        literal = set(
            _VertexExtraction.model_json_schema()["properties"]["budget_range"]["enum"]
        )
        assert literal == {"unknown"} | {b.value for b in BudgetRange}

    def test_the_sentinel_is_not_a_real_choice(self):
        """"unknown" has to mean "not stated" and nothing else. If a domain enum ever
        gained a member called `unknown`, the sentinel would silently become a value the
        rider could be assigned."""
        assert "unknown" not in {g.value for g in GroupType}
        assert "unknown" not in {b.value for b in BudgetRange}


class TestSentinelsBecomeNoneAgain:
    """`to_extraction` is the other half of the trick: the wire speaks in "" and 0, the
    node speaks in None, and nothing downstream should be able to tell."""

    def test_an_empty_wire_object_is_an_empty_extraction(self):
        node = _VertexExtraction().to_extraction()
        assert isinstance(node, PlannerExtraction)
        for field in ("destination", "origin", "travel_dates", "home_city",
                      "duration_days", "group_size", "rental_hours", "pickup_time",
                      "cab_category", "group_type", "budget_range",
                      "origin_in_himachal"):
            assert getattr(node, field) is None, field

    def test_real_values_survive(self):
        node = _VertexExtraction(
            destination="Kasol", origin="Shimla", duration_days=3, group_size=2,
            travel_dates="kal", group_type="family", budget_range="moderate",
            rental_hours=8, pickup_time="09:00", cab_category="sedan",
            home_city="Shimla", origin_in_himachal="yes",
        ).to_extraction()
        assert node.destination == "Kasol"
        assert node.origin == "Shimla"
        assert node.duration_days == 3
        assert node.group_size == 2
        assert node.travel_dates == "kal"
        assert node.group_type == GroupType.FAMILY
        assert node.budget_range == BudgetRange.MODERATE
        assert node.rental_hours == 8
        assert node.pickup_time == "09:00"
        assert node.origin_in_himachal is True

    @pytest.mark.parametrize("word", sorted(_WIRE_BLANKS - {""}))
    def test_the_model_writing_the_word_null_is_not_a_destination(self, word):
        """Observed live: `origin='null'`. Told to use "", the model sometimes writes
        the word — and a booking screen would have shown a pickup of "null"."""
        assert _VertexExtraction(destination=word, origin=word).to_extraction().destination is None
        assert _VertexExtraction(origin=word).to_extraction().origin is None

    def test_a_three_state_flag_keeps_all_three_states(self):
        """`origin_in_himachal` decides whether a rider is a visitor. Collapsing
        "unknown" into False would classify every rider we know nothing about as an
        out-of-state visitor and switch their whole flow."""
        assert _VertexExtraction(origin_in_himachal="yes").to_extraction().origin_in_himachal is True
        assert _VertexExtraction(origin_in_himachal="no").to_extraction().origin_in_himachal is False
        assert _VertexExtraction(origin_in_himachal="unknown").to_extraction().origin_in_himachal is None
        # The default, which is what arrives when the model omits the field entirely.
        assert _VertexExtraction().to_extraction().origin_in_himachal is None

    def test_zero_is_read_as_unstated_not_as_zero(self):
        """0 is the sentinel and also a nonsense answer — no trip lasts zero days, no
        cab carries zero people — so nothing real is lost by conflating them."""
        node = _VertexExtraction(duration_days=0, group_size=0, rental_hours=0).to_extraction()
        assert node.duration_days is None
        assert node.group_size is None
        assert node.rental_hours is None


class TestABlankExtractionIsRetriedNotCrashed:
    """`with_structured_output` returns None when the model declines to answer in the
    required shape. It is not an exception and must not be turned into one."""

    @staticmethod
    def _planner_with(results):
        """A planner whose extraction returns each of `results` in turn."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from langchain_core.messages import AIMessage, HumanMessage

        from app.agent.nodes.planner import planner_node
        from app.agent.state import AgentState
        from app.domain.conversation import ExtractedEntities

        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=results)
        state = AgentState(
            session_id="s", user_id="u", intent="plan_trip", visitor_type="local",
            entities=ExtractedEntities(visitor_type="local",
                                       last_asked_field="destination"),
            messages=[AIMessage(content="Where would you like to go?"),
                      HumanMessage(content="mei kasol jana chahta hu")],
        )
        ctx = patch("app.agent.nodes.planner.get_vertex_ai_service")
        return planner_node, state, ctx, llm

    @pytest.mark.asyncio
    async def test_a_none_on_the_first_call_is_retried(self):
        """One blank must not cost the rider the turn. The whole reported failure was
        this case going unretried and unnoticed."""
        from tests.conftest import stub_vertex, wire_extraction

        planner_node, state, ctx, llm = self._planner_with(
            [None, wire_extraction(destination="Kasol", origin_in_himachal=True)]
        )
        with ctx as svc:
            stub_vertex(svc, llm)
            result = await planner_node(state)

        assert result["entities"].destination == "Kasol"
        assert llm.with_structured_output.return_value.ainvoke.await_count == 2

    @pytest.mark.asyncio
    async def test_all_blank_degrades_instead_of_raising_a_typeerror(self):
        """When every attempt is blank the turn still has to work. `dict(None)` used to
        raise here, which is what put `'NoneType' object is not iterable` in the logs
        and sent everyone looking for a bug in the planner rather than in the schema."""
        planner_node, state, ctx, llm = self._planner_with([None, None])
        with ctx as svc:
            stub_vertex(svc, llm)
            result = await planner_node(state)

        # The deterministic reader still gets the destination out of the sentence, which
        # is the whole reason the failure stayed invisible for so long.
        assert result["entities"].destination == "Kasol"
        assert result["entities"] is not None

    @pytest.mark.asyncio
    async def test_json_mode_and_the_wire_model_are_what_is_asked_for(self):
        """The mode is half the fix; flat scalars alone still failed 19 times in 25.

        Asserted on the SERVICE call, because `structured()` is where the two halves are
        put together — and where the standby model is composed behind them.
        """
        from tests.conftest import stub_vertex, wire_extraction

        planner_node, state, ctx, llm = self._planner_with(
            [wire_extraction(destination="Kasol")]
        )
        with ctx as svc:
            stub_vertex(svc, llm)
            await planner_node(state)

        args, kwargs = svc.return_value.structured.call_args
        assert args[0] is _VertexExtraction
        assert kwargs.get("method") == "json_mode"


class TestAnEmptyBubbleIsNeverShipped:
    """Seen live on a local ride: the writer logged `response_length=119` and the rider
    got a blank message. The model had answered with the JSON block alone, and the
    cleaner removes everything from the fence onwards."""

    def test_cleaning_a_json_only_reply_leaves_nothing(self):
        """The mechanism, stated so the fallback below has something to stand on."""
        from app.agent.nodes.writer import _clean_response_text

        assert _clean_response_text('```json\n{"destination": "Kasol"}\n```') == ""
        assert _clean_response_text('{"destination": "Kasol"}') == ""

    @pytest.mark.parametrize("lang_key", ["en", "hi"])
    def test_there_is_always_something_to_say(self, lang_key):
        from app.agent.nodes.writer import _blank_reply_fallback

        text = _blank_reply_fallback(None, lang_key)
        assert text.strip()
        # A question, not an apology: a dead end leaves the rider with nothing to type.
        assert "?" in text

    def test_an_unknown_language_still_gets_words(self):
        from app.agent.nodes.writer import _blank_reply_fallback

        assert _blank_reply_fallback(None, "fr").strip()

    @pytest.mark.parametrize(
        "raw",
        [
            '```json\n{"destination": "Kasol"}\n```',
            '{"destination": "Kasol"}',
            # The one that shipped: the model opened its object with a different key,
            # the narrow strip missed it, and the rider got a single "{".
            '{\n  "trip": {"destination": "Kasol"}\n}',
            "{",
            "   \n  ",
        ],
    )
    def test_nothing_readable_is_recognised_as_nothing(self, raw):
        from app.agent.nodes.writer import _clean_response_text, _is_unreadable

        assert _is_unreadable(_clean_response_text(raw)), (
            f"{raw!r} cleaned to something that would be shown to a rider"
        )

    @pytest.mark.parametrize(
        "reply",
        ["Your ride from Shimla to Kasol is ready.",
         "शिमला से कसोल तक आपकी सवारी तैयार है।",
         "8 hrs"],
    )
    def test_a_real_reply_is_left_alone(self, reply):
        from app.agent.nodes.writer import _clean_response_text, _is_unreadable

        assert not _is_unreadable(_clean_response_text(reply))
        assert _clean_response_text(reply) == reply
