"""
tests/unit/agent/test_sanitize.py
──────────────────────────────────
The validation gate, and the relative dates it resolves.

Two reports sit behind this module:

  1. a rental booked for "tomorrow" or "day after tomorrow" put that PHRASE into the
     booking screen's date field, which cannot render it — and "day after tmrw" was
     read as tomorrow, a silent off-by-one day
  2. nothing checked what the extraction model wrote. `slots.parse_*` guards a bare
     reply and `slots.read_rental_details` anchors a sentence, but the model's own
     output went straight into the entity bag, so a group of 500 or a pickup of "Here"
     reached the fare request unopposed

Every test here fixes a date rather than trusting the clock: a suite that depends on
what today happens to be is a suite that starts failing on a date nobody chose.
"""

from datetime import date, timedelta

import pytest

from app.agent import sanitize

#: A fixed "today" so every expectation below is exact. A Thursday, for no reason
#: other than that a real one makes the arithmetic easy to check by eye.
TODAY = date(2026, 9, 3)


def _day(offset: int) -> str:
    return (TODAY + timedelta(days=offset)).isoformat()


# ── 1. Relative days become real dates ────────────────────────────────────────

class TestRelativeDays:
    """The reported bug. A booking screen has a date field, not a sentence field."""

    @pytest.mark.parametrize(
        "text,offset",
        [
            ("today", 0), ("tonight", 0), ("aaj", 0),
            ("tomorrow", 1), ("kal", 1),
            ("day after tomorrow", 2), ("parso", 2), ("parson", 2),
        ],
    )
    def test_the_common_phrasings_resolve(self, text, offset):
        assert sanitize.resolve_relative_day(text, today=TODAY) == TODAY + timedelta(days=offset)

    @pytest.mark.parametrize(
        "typo", ["tommorow", "tomorow", "tomarrow", "tomorrw", "tmrw", "tmr", "tmw"],
    )
    def test_the_spellings_riders_actually_type(self, typo):
        """A misspelling must not cost a rider their date — they get re-asked for
        something they already answered, which reads as the assistant ignoring them."""
        assert sanitize.resolve_relative_day(typo, today=TODAY) == TODAY + timedelta(days=1)

    @pytest.mark.parametrize(
        "text", ["day after tomorrow", "day after tmrw", "the day after tomorrow",
                 "I need it day after tomorrow"],
    )
    def test_day_after_tomorrow_is_never_read_as_tomorrow(self, text):
        """The off-by-one. "day after tmrw" matched the short pattern first and became
        tomorrow: the rider asked for Saturday and the cab came on Friday."""
        assert sanitize.resolve_relative_day(text, today=TODAY) == TODAY + timedelta(days=2)

    def test_devanagari_days_resolve_too(self):
        assert sanitize.resolve_relative_day("कल", today=TODAY) == TODAY + timedelta(days=1)
        assert sanitize.resolve_relative_day("परसों", today=TODAY) == TODAY + timedelta(days=2)

    @pytest.mark.parametrize(
        "text", ["next week", "after Diwali", "15 Jan", "sometime in June", ""],
    )
    def test_a_phrase_naming_no_single_day_resolves_to_nothing(self, text):
        """None is the right answer, not a guess. "next week" is not a date, and
        inventing one would put a day in a booking the rider never chose."""
        assert sanitize.resolve_relative_day(text, today=TODAY) is None


# ── 2. What reaches the booking screen ────────────────────────────────────────

class TestTravelDate:

    @pytest.mark.parametrize(
        "text,expected_offset",
        [("today", 0), ("tomorrow", 1), ("day after tomorrow", 2), ("parso", 2)],
    )
    def test_a_rental_gets_an_iso_date(self, text, expected_offset):
        assert sanitize.travel_date(text, as_iso=True, today=TODAY) == _day(expected_offset)

    @pytest.mark.parametrize("text", ["today", "tomorrow", "day after tomorrow"])
    def test_a_trip_keeps_the_riders_own_words(self, text):
        """A trip's date is prose in a reply, not a value in a form — and "sometime
        after Diwali" is worth keeping verbatim, so trips are left alone."""
        assert sanitize.travel_date(text, as_iso=False, today=TODAY) == text

    @pytest.mark.parametrize("text", ["yesterday", "day before yesterday"])
    def test_a_day_in_the_past_is_refused(self, text):
        """Nobody hires a car for yesterday. Dropping it puts the question back to the
        rider instead of handing the backend a date it must reject."""
        assert sanitize.travel_date(text, as_iso=True, today=TODAY) is None
        assert sanitize.travel_date(text, as_iso=False, today=TODAY) is None

    @pytest.mark.parametrize("text", ["next week", "after Diwali", "first week of June"])
    def test_an_unresolvable_phrase_survives_untouched(self, text):
        assert sanitize.travel_date(text, as_iso=True, today=TODAY) == text


# ── 3. The gate on everything else ────────────────────────────────────────────

class TestBounds:

    @pytest.mark.parametrize("value", [1, 4, 60, "4"])
    def test_a_real_group_size_is_kept(self, value):
        assert sanitize.group_size(value) == int(value)

    @pytest.mark.parametrize("value", [0, -1, 61, 500, "many", None, ""])
    def test_an_impossible_group_size_is_dropped(self, value):
        assert sanitize.group_size(value) is None

    @pytest.mark.parametrize("value", [1, 3, 30])
    def test_a_real_trip_length_is_kept(self, value):
        assert sanitize.duration_days(value) == value

    @pytest.mark.parametrize("value", [0, -3, 31, 400])
    def test_an_impossible_trip_length_is_dropped(self, value):
        assert sanitize.duration_days(value) is None

    @pytest.mark.parametrize("value", [1, 8, 24])
    def test_a_plausible_hire_is_kept(self, value):
        assert sanitize.rental_hours(value) == value

    @pytest.mark.parametrize("value", [0, -8, 25, 300])
    def test_an_implausible_hire_is_dropped(self, value):
        assert sanitize.rental_hours(value) is None


class TestPickupTime:
    """The backend reads this as `bookingTime` and charges the night rate from it, so a
    value that is not a clock reading is a value that changes what the rider pays."""

    @pytest.mark.parametrize(
        "value,expected", [("09:00", "09:00"), ("9:05", "09:05"), ("23:59", "23:59"),
                           ("00:00", "00:00")],
    )
    def test_a_real_clock_reading_is_kept(self, value, expected):
        assert sanitize.pickup_time(value) == expected

    @pytest.mark.parametrize(
        "value", ["25:99", "24:00", "9 am", "morning", "half past nine", "", "-1:00"],
    )
    def test_anything_that_is_not_a_clock_is_dropped(self, value):
        assert sanitize.pickup_time(value) is None


class TestPlace:
    """Validating is not parsing — see the module docstring. This is more permissive
    than `slots.parse_place` on purpose."""

    @pytest.mark.parametrize(
        "value", ["Manali", "Sector 17 Chandigarh", "Kullu Bus Stand",
                  "Mcleod Ganj", "Kangra Fort Road"],
    )
    def test_a_real_place_survives(self, value):
        assert sanitize.place(value) == value

    def test_a_digit_does_not_disqualify_a_place(self):
        """`slots.parse_place` rejects any string with a digit, which is right for
        reading a bare reply and wrong for validating an extracted value."""
        from app.agent import slots

        assert slots.parse_place("Sector 17 Chandigarh") is None
        assert sanitize.place("Sector 17 Chandigarh") == "Sector 17 Chandigarh"

    @pytest.mark.parametrize(
        "value", ["here", "Here", "my location", "cancel", "rent a car", "", "   "],
    )
    def test_the_values_that_actually_reached_this_field_are_refused(self, value):
        assert sanitize.place(value) is None

    def test_an_extracted_paragraph_is_not_a_place(self):
        assert sanitize.place("I would like to be collected from somewhere near the "
                              "bus stand in Kullu tomorrow morning please") is None


class TestTextHygiene:

    def test_invisible_characters_are_stripped(self):
        """A zero-width space survives every eyeball review and then makes the geocoder
        miss, for a reason nobody can see in a log or a bug report."""
        assert sanitize.place("Man​ali") == "Manali"
        assert sanitize.clean_text("Shimla‮") == "Shimla"

    def test_control_characters_and_runs_of_space_collapse(self):
        assert sanitize.clean_text("Kullu\x00\x07   Bus\t\tStand") == "Kullu Bus Stand"

    def test_devanagari_is_not_mangled(self):
        assert sanitize.clean_text("मनाली") == "मनाली"

    def test_nothing_usable_becomes_none(self):
        for value in (None, "", "   ", "​​"):
            assert sanitize.clean_text(value) is None


# ── 4. The gate as one call ───────────────────────────────────────────────────

class TestSanitizeEntities:

    def test_bad_values_are_dropped_and_named(self):
        cleaned, dropped = sanitize.sanitize_entities(
            {
                "destination": "Manali",
                "origin": "Here",
                "group_size": 500,
                "duration_days": -3,
                "pickup_time": "25:99",
                "travel_dates": "yesterday",
            },
            today=TODAY,
        )
        assert cleaned["destination"] == "Manali"
        for field in ("origin", "group_size", "duration_days", "pickup_time", "travel_dates"):
            assert cleaned[field] is None
        assert set(dropped) == {
            "origin", "group_size", "duration_days", "pickup_time", "travel_dates"
        }

    def test_a_clean_bag_passes_through_unchanged(self):
        original = {
            "destination": "Manali", "origin": "Kullu", "group_size": 4,
            "duration_days": 3, "travel_dates": "next week",
        }
        cleaned, dropped = sanitize.sanitize_entities(dict(original), today=TODAY)
        assert cleaned == original
        assert dropped == []

    def test_the_caller_s_dict_is_never_mutated(self):
        original = {"group_size": 500}
        sanitize.sanitize_entities(original, today=TODAY)
        assert original == {"group_size": 500}

    def test_fields_it_does_not_know_are_left_alone(self):
        """A gate on the values the system ACTS on, not a schema for the whole bag."""
        cleaned, dropped = sanitize.sanitize_entities(
            {"proceed_requested": True, "asked_counts": {"origin": 2}, "interests": ["snow"]},
            today=TODAY,
        )
        assert cleaned["proceed_requested"] is True
        assert cleaned["asked_counts"] == {"origin": 2}
        assert cleaned["interests"] == ["snow"]
        assert dropped == []

    def test_a_rental_date_becomes_iso_and_a_trip_date_does_not(self):
        rental, _ = sanitize.sanitize_entities(
            {"travel_dates": "day after tomorrow"}, is_rental=True, today=TODAY
        )
        trip, _ = sanitize.sanitize_entities(
            {"travel_dates": "day after tomorrow"}, is_rental=False, today=TODAY
        )
        assert rental["travel_dates"] == _day(2)
        assert trip["travel_dates"] == "day after tomorrow"
