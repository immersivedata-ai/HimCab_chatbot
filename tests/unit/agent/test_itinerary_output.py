"""
tests/unit/test_itinerary_output.py
────────────────────────────────────
Regression tests for the itinerary the mobile app actually receives.

The bug these lock down: `ChatResponse.itinerary` came back null even on turns that
generated a complete itinerary. `Itinerary.model_validate()` was raising — the writer
LLM had put the persona ("outstation") in `group_type` and left `budget_range` null,
neither of which the strict model accepted — so the writer logged the failure, set
`structured_itinerary = None`, and shipped only the unvalidated raw dict inside
`structured_events`. The two delivery channels disagreed and the documented one was
dead.

Two independent guarantees are tested here:
  1. the models coerce off-vocabulary LLM output instead of rejecting it
  2. the writer overwrites trip-level facts with what the planner actually confirmed,
     so the persona can never reach `group_type` in the first place
"""

import pytest

from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities
from app.domain.itinerary import (
    ActivityType,
    BudgetRange,
    CrowdLevel,
    GroupType,
    Itinerary,
)

# The literal payload the app team captured from production.
REPORTED_PAYLOAD = {
    "destination": "Himachal Pradesh",
    "duration_days": 3,
    "group_type": "outstation",   # ← persona leaked into a group-type field
    "budget_range": None,         # ← user never stated a budget
    "days": [
        {
            "day": 1,
            "theme": "Arrival",
            "activities": [
                {"time": "Morning", "place": "Mall Road", "type": "sightseeing"},
            ],
        }
    ],
}


class TestItineraryValidationIsForgiving:
    def test_the_exact_reported_payload_now_validates(self):
        itinerary = Itinerary.model_validate(REPORTED_PAYLOAD)

        assert itinerary.destination == "Himachal Pradesh"
        assert itinerary.duration_days == 3
        # "outstation" is not a group type — coerced rather than raising.
        assert itinerary.group_type in {g.value for g in GroupType}
        assert itinerary.group_type != "outstation"
        assert itinerary.budget_range == BudgetRange.MODERATE.value

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("family", GroupType.FAMILY),
            ("Family with young kids", GroupType.FAMILY),
            ("business trip", GroupType.CORPORATE),
            ("honeymoon", GroupType.COUPLE),
            ("local", GroupType.FAMILY),        # persona, not a group type
            ("outstation", GroupType.FAMILY),   # persona, not a group type
            (None, GroupType.FAMILY),
            ("", GroupType.FAMILY),
        ],
    )
    def test_group_type_coercion(self, value, expected):
        itinerary = Itinerary.model_validate({**REPORTED_PAYLOAD, "group_type": value})
        assert itinerary.group_type == expected.value

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("luxury", BudgetRange.LUXURY),
            ("premium", BudgetRange.LUXURY),
            ("mid-range", BudgetRange.MODERATE),
            ("budget friendly", BudgetRange.BUDGET),
            (None, BudgetRange.MODERATE),
        ],
    )
    def test_budget_range_coercion(self, value, expected):
        itinerary = Itinerary.model_validate({**REPORTED_PAYLOAD, "budget_range": value})
        assert itinerary.budget_range == expected.value

    def test_unknown_activity_type_degrades_to_other(self):
        payload = {
            **REPORTED_PAYLOAD,
            "days": [{"day": 1, "activities": [{"place": "Hotel", "type": "check-in"}]}],
        }
        itinerary = Itinerary.model_validate(payload)
        assert itinerary.days[0].activities[0].type == ActivityType.OTHER

    def test_coercion_matches_whole_words_only(self):
        """`rest` must not swallow `restaurant` — a substring match would mislabel
        every dining activity as a rest stop."""
        payload = {
            **REPORTED_PAYLOAD,
            "days": [{"day": 1, "activities": [{"place": "X", "type": "restaurant"}]}],
        }
        itinerary = Itinerary.model_validate(payload)
        assert itinerary.days[0].activities[0].type == ActivityType.OTHER

    def test_crowd_level_synonym(self):
        payload = {
            **REPORTED_PAYLOAD,
            "days": [{"day": 1, "activities": [
                {"place": "X", "type": "sightseeing", "crowd_level": "medium"},
            ]}],
        }
        itinerary = Itinerary.model_validate(payload)
        assert itinerary.days[0].activities[0].crowd_level == CrowdLevel.MODERATE

    def test_a_sparse_payload_still_produces_an_itinerary(self):
        """Better a thin itinerary than a null one."""
        itinerary = Itinerary.model_validate({"destination": "Manali"})
        assert itinerary.destination == "Manali"
        assert itinerary.days == []

    def test_json_dump_uses_plain_strings(self):
        """The app parses these with Gson — every enum must serialise as a string."""
        dumped = Itinerary.model_validate(REPORTED_PAYLOAD).model_dump(mode="json")
        assert isinstance(dumped["group_type"], str)
        assert isinstance(dumped["budget_range"], str)
        assert isinstance(dumped["days"][0]["activities"][0]["type"], str)


class TestLeakedMetadataIsStrippedFromProse:
    """The reported screenshot: an itinerary reply whose activity bullets each carried a
    stray "◦ type: sightseeing" sub-bullet, plus a "Recommended Mode: Cab" line — the
    JSON schema's own field names echoed into the sentences the rider reads. Those belong
    only in the ```json block; in the prose they are clutter."""

    def test_the_screenshot_leaks_are_removed(self):
        from app.agent.nodes.writer import _clean_response_text

        raw = (
            "Here is a suggested itinerary for your 2-day family trip to Manali.\n"
            "\n"
            "Your Journey: Kangra to Manali\n"
            "- Distance: about 229 km\n"
            "- Recommended Mode: Cab\n"
            "\n"
            "Day 1: Arrival in Manali & Local Exploration\n"
            "- Morning/Afternoon: Begin your journey from Kangra to Manali.\n"
            "  - type: travel\n"
            "- Afternoon: Check into your accommodation and rest after your travel.\n"
            "  - type: accommodation\n"
            "  - type: rest\n"
            "- Evening: Visit the Hadimba Devi Temple.\n"
            "  - type: sightseeing\n"
            "- Dinner: Enjoy local Himachali cuisine at a restaurant in Manali.\n"
            "  - type: dining\n"
            "\n"
            "```json\n"
            '{"destination": "Manali", "duration_days": 2}\n'
            "```\n"
        )

        clean = _clean_response_text(raw)

        # None of the leaked field lines survive.
        assert "type:" not in clean
        assert "Recommended Mode" not in clean.replace("recommended mode", "Recommended Mode")
        assert "recommended_mode" not in clean.lower()
        # The JSON block is gone, as before.
        assert "```" not in clean
        assert '"destination"' not in clean
        # The real, human-readable content is untouched.
        assert "Hadimba Devi Temple" in clean
        assert "Distance: about 229 km" in clean
        assert "Begin your journey from Kangra to Manali." in clean

    def test_a_legitimate_sentence_starting_with_type_is_kept(self):
        """The strip is for one-token field leaks, not any sentence about a 'type'."""
        from app.agent.nodes.writer import _strip_leaked_metadata

        text = (
            "- This type of trek suits beginners.\n"
            "- Best time to visit: early morning to beat the crowds.\n"
        )
        assert _strip_leaked_metadata(text) == text.strip()

    def test_leaks_are_caught_under_any_bullet_style(self):
        from app.agent.nodes.writer import _strip_leaked_metadata

        text = (
            "* Visit the temple.\n"
            "  • type: spiritual\n"
            "1. Lunch at a dhaba.\n"
            "   ◦ type: dining\n"
            "- Recommended Mode: Cab\n"
        )
        cleaned = _strip_leaked_metadata(text)
        assert "type:" not in cleaned
        assert "Recommended Mode" not in cleaned
        assert "Visit the temple." in cleaned
        assert "Lunch at a dhaba." in cleaned

    def test_off_vocabulary_type_leak_is_also_removed(self):
        """A leak the model wrote with a non-schema value ('check-in') is still a leak."""
        from app.agent.nodes.writer import _strip_leaked_metadata

        text = "- Check into the hotel.\n  - type: check-in\n"
        assert "type:" not in _strip_leaked_metadata(text)
        assert "Check into the hotel." in _strip_leaked_metadata(text)


class TestStructuredEventTypesStayInSync:
    """`ChatResponse.structured_events` is typed, so an event type the writer emits but
    the enum doesn't list would fail response validation and 500 the request. This
    catches that drift at test time instead of in production."""

    def test_every_type_the_agent_emits_is_declared(self):
        """Scans the whole agent package, not just the writer.

        It used to scan `writer.py` alone, which was true when the writer was the only
        node that emitted cards. `cancellation`, `modification` and the planner all
        build their own now, and a type none of them declares 500s the request at
        response validation — the exact drift this test exists to catch, missed because
        the search was narrower than the code.
        """
        import re
        from pathlib import Path

        from app.schemas.chat import StructuredEventType

        # Anchored on the `payload` that follows it, so the activity types inside the
        # writer prompt's own JSON template are not mistaken for event types.
        pattern = re.compile(r'"type":\s*"([a-z_]+)",\s*(?:#[^\n]*\n\s*)?"payload"')
        emitted: set[str] = set()
        for path in sorted(Path("app/agent").rglob("*.py")):
            emitted |= set(pattern.findall(path.read_text(encoding="utf-8")))

        declared = {t.value for t in StructuredEventType}

        assert emitted, "no structured event types found — did the agent package move?"
        assert emitted <= declared, (
            f"the agent emits undeclared event type(s) {sorted(emitted - declared)}; "
            f"add them to StructuredEventType and document them in docs/api_reference.md"
        )

    def test_declared_types_are_documented(self):
        from pathlib import Path

        from app.schemas.chat import StructuredEventType

        doc = Path("docs/api_reference.md").read_text(encoding="utf-8")
        missing = [t.value for t in StructuredEventType if f"`{t.value}`" not in doc]
        assert not missing, f"undocumented structured event type(s): {missing}"


class TestWriterNormalisation:
    """The writer's own guard: the conversation, not the LLM, owns trip-level facts."""

    def _state(self, **entity_kwargs) -> AgentState:
        return AgentState(
            session_id="s1",
            user_id="rider:1",
            visitor_type="outstation",
            entities=ExtractedEntities(**entity_kwargs),
        )

    def test_persona_in_group_type_is_replaced_by_the_confirmed_value(self):
        from app.agent.nodes.writer import _normalize_itinerary_dict

        state = self._state(destination="Manali", duration_days=4, group_size=4)
        result = _normalize_itinerary_dict(REPORTED_PAYLOAD, state)

        assert result["group_type"] == GroupType.FAMILY.value
        assert result["destination"] == "Manali"   # planner's value wins
        assert result["duration_days"] == 4

    @pytest.mark.parametrize(
        "group_size,expected",
        [(1, GroupType.SOLO), (2, GroupType.COUPLE), (5, GroupType.FAMILY)],
    )
    def test_group_type_inferred_from_group_size(self, group_size, expected):
        from app.agent.nodes.writer import _infer_group_type

        entities = ExtractedEntities(destination="Manali", group_size=group_size)
        assert _infer_group_type(entities) == expected

    def test_explicit_group_type_beats_the_size_heuristic(self):
        from app.agent.nodes.writer import _infer_group_type

        entities = ExtractedEntities(group_type=GroupType.FRIENDS, group_size=2)
        assert _infer_group_type(entities) == GroupType.FRIENDS

    def test_a_valid_llm_group_type_is_left_alone(self):
        """Normalisation repairs bad values; it must not overwrite good ones."""
        from app.agent.nodes.writer import _normalize_itinerary_dict

        state = self._state(destination="Manali", group_size=4)
        payload = {**REPORTED_PAYLOAD, "group_type": "friends"}
        assert _normalize_itinerary_dict(payload, state)["group_type"] == "friends"

    def test_normalisation_does_not_mutate_the_input(self):
        from app.agent.nodes.writer import _normalize_itinerary_dict

        original = dict(REPORTED_PAYLOAD)
        _normalize_itinerary_dict(REPORTED_PAYLOAD, self._state(destination="Manali"))
        assert original == REPORTED_PAYLOAD

    def test_normalised_payload_validates_end_to_end(self):
        from app.agent.nodes.writer import _normalize_itinerary_dict

        state = self._state(destination="Manali", duration_days=3, group_size=4)
        itinerary = Itinerary.model_validate(
            _normalize_itinerary_dict(REPORTED_PAYLOAD, state)
        )
        assert itinerary.destination == "Manali"
        assert itinerary.group_type == GroupType.FAMILY.value
