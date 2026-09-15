"""
tests/unit/agent/test_trip_sentence.py
───────────────────────────────────────
A trip stated in a sentence, read in code.

The reported conversation, on the very first turn:

    assistant   नमस्ते! … आप कहाँ जाना चाहते हैं?          (where do you want to go?)
    rider       mei shimla jana chahta hu
    assistant   आप हिमाचल में कहाँ जाना चाहते हैं?          (where do you want to go?)

The rider answered and was asked the same thing again. `parse_place` accepts at most
three words on purpose — a longer string is a sentence, and letting a sentence become a
destination is how "plan a trip" once became a place — so a bare "shimla" was recovered
in code and a sentence containing it was not. That left the whole trip flow depending on
the extraction model, and a miss was silent: the detail vanished and the question came
back.

The rental flow already had this safety net. These tests are the trip flow's, and they
run with the model DOWN, because the point is that the sentence is understood without
it.
"""

from __future__ import annotations

import pytest

from app.agent.slots import read_trip_details

# ── 1. The destination, however it is phrased ─────────────────────────────────


class TestTheDestinationIsFound:

    @pytest.mark.parametrize(
        "sentence,expected",
        [
            # The reported one, and its neighbours.
            ("mei shimla jana chahta hu", "Shimla"),
            ("mujhe manali jana hai", "Manali"),
            ("main manali ja raha hu", "Manali"),
            ("hum kasol ja rahe hain", "Kasol"),
            ("kullu ghumna hai", "Kullu"),
            ("dharamshala tak jana hai", "Dharamshala"),
            # English.
            ("I want to go to Manali", "Manali"),
            ("we are going to Kasol", "Kasol"),
            ("planning a trip to Dharamshala", "Dharamshala"),
            ("I want to visit Kufri", "Kufri"),
            ("heading to Solan", "Solan"),
        ],
    )
    def test_a_sentence_yields_its_destination(self, sentence, expected):
        assert read_trip_details(sentence).get("destination") == expected

    def test_the_pronoun_does_not_become_part_of_the_place(self):
        """"mei shimla jana hai" must not yield the place "Mei Shimla". The words
        beside a destination marker are collected until one cannot belong to a name."""
        assert read_trip_details("mei shimla jana hai")["destination"] == "Shimla"

    def test_a_date_after_the_place_does_not_swallow_it(self):
        """"a trip to Dharamshala next week" once collected three words, failed the
        place gate for containing a time word, and lost the DESTINATION rather than
        just the date."""
        found = read_trip_details("planning a trip to Dharamshala next week")
        assert found["destination"] == "Dharamshala"
        assert found["travel_dates"] == "next week"


# ── 2. Both ends of the journey out of one sentence ───────────────────────────


class TestBothEndsAreRead:

    def test_origin_and_destination_from_one_sentence(self):
        found = read_trip_details("shimla se manali jana hai")
        assert (found["origin"], found["destination"]) == ("Shimla", "Manali")

    def test_the_english_form_too(self):
        found = read_trip_details("we are going to Kasol from Delhi")
        assert (found["origin"], found["destination"]) == ("Delhi", "Kasol")

    def test_the_whole_trip_in_one_line(self):
        found = read_trip_details("we are going to Kasol from Delhi for 3 days, 4 of us")
        assert found["destination"] == "Kasol"
        assert found["origin"] == "Delhi"
        assert found["duration_days"] == 3
        assert found["group_size"] == 4

    def test_the_two_places_never_read_each_other(self):
        """`se` and `from` are the origin's markers and stop the destination scan, so
        one end can never be filed as the other."""
        found = read_trip_details("delhi se shimla jana hai")
        assert found["origin"] == "Delhi"
        assert found["destination"] == "Shimla"


# ── 3. Nothing is invented ────────────────────────────────────────────────────


class TestNothingIsInvented:

    @pytest.mark.parametrize(
        "sentence",
        [
            "plan a trip",
            "I need a car",
            "make me an itinerary",
            "yes please",
            "cancel",
            "what is the weather like",
            "kuch bhi",
            "",
            "   ",
        ],
    )
    def test_a_sentence_with_no_destination_yields_none(self, sentence):
        assert "destination" not in read_trip_details(sentence)

    def test_a_question_about_a_place_is_not_a_trip_to_it(self):
        """"what is the weather in manali" names a place and asks about it. Reading it
        as a destination would turn a lookup into a booking."""
        assert "destination" not in read_trip_details("what is the weather in manali")

    def test_a_bare_number_is_not_a_duration(self):
        """The unit is the anchor. Without it a "3" could be anything — a group size,
        a date, a cab tier — and guessing which is how the wrong field gets filled."""
        assert "duration_days" not in read_trip_details("3")

    def test_next_week_is_a_date_not_a_seven_day_trip(self):
        """"a week" is a length; "next week" is when. Reading the second as the first
        invents a trip duration the rider never gave."""
        found = read_trip_details("next week")
        assert found.get("travel_dates") == "next week"
        assert "duration_days" not in found

    def test_for_a_week_is_a_duration(self):
        assert read_trip_details("manali for a week")["duration_days"] == 7


# ── 4. The rental reader is untouched ─────────────────────────────────────────


class TestTheRentalReaderStillWorks:
    """Both readers now share one place-extraction helper, so the rental flow is
    exercised here too — a refactor that quietly broke the pickup scan would otherwise
    only show up as a rental asking for something it had just been told."""

    def test_the_rental_pickup_still_reads(self):
        from app.agent.slots import read_rental_details

        found = read_rental_details("I need a car for 8 hours from Shimla tomorrow at 9 am, 4 of us")
        assert found["origin"] == "Shimla"
        assert found["rental_hours"] == 8
        assert found["pickup_time"] == "09:00"
        assert found["group_size"] == 4

    def test_a_rental_sentence_has_no_destination_read_into_it(self):
        """A rental has no destination by design. The trip reader is not run on that
        flow, and the rental reader has never looked for one."""
        from app.agent.slots import read_rental_details

        assert "destination" not in read_rental_details("car for 8 hours from Kullu")
