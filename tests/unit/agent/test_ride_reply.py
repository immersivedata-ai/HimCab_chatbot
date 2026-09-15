"""
tests/unit/agent/test_ride_reply.py
────────────────────────────────────
The local ride reply, which is assembled rather than generated.

Reported twice. The second time, after the prompt had already been stripped of every
trip-shaped fact, the reply still came back like this:

    कसोल से शिमला तक की आपकी यात्रा के बारे में यहाँ कुछ जानकारी दी गई है:
    ...
    यात्रा के लिए कुछ सुझाव:
    • शिमला में दर्शनीय स्थलों की यात्रा के लिए यह एक बेहतरीन समय है।
    • आप मॉल रोड, रिज, जाखू मंदिर और कुफरी जैसे स्थानों पर जा सकते हैं।
    • अपने परिवार के साथ इस यात्रा का आनंद लें।
    क्या आप अपनी पूरी यात्रा की योजना बनाने में मेरी मदद चाहेंगे?

No price. No booking screen. A list of landmarks that came from no tool, and an offer
to plan a holiday — to a resident who had asked for a cab from Kasol to Shimla and
lives an hour away.

The prompt already forbade every one of those in writing. So the reply is no longer
written by a model: `app/agent/ride_reply.py` assembles it from the facts the system
already holds, the way `app/agent/cancellation.py` does and for the same reason. These
tests are about what such a reply can no longer contain.
"""

from __future__ import annotations

import pytest

from app.agent.ride_reply import build_ride_reply
from app.agent.state import AgentState
from app.domain.conversation import ExtractedEntities

ROUTE = {
    "origin": {"name": "Kasol", "lat": 32.0109, "lon": 77.3160},
    "destination": {"name": "Shimla", "lat": 31.1048, "lon": 77.1734},
    "distance_km": 215.0,
    "duration_minutes": 270,
    "duration_text": "4h 30m",
    "is_approximate": False,
}

WEATHER = {
    "destination": "Shimla",
    "current_temp_c": 19.7,
    "condition": "Clear",
    "forecast_7day": [{"date": "2026-09-16", "condition": "Light rain"}],
    "alerts": [],
}

FARE = {
    "category_name": "Sedan",
    "distance_km": 215.0,
    "options": [
        {"category_name": "Economy", "total_fare_inr": 3200, "seat_capacity": 4},
        {"category_name": "Sedan", "total_fare_inr": 4100, "seat_capacity": 4},
        {"category_name": "SUV", "total_fare_inr": 5600, "seat_capacity": 6},
    ],
}


def _state(**overrides) -> AgentState:
    kwargs = {
        "session_id": "s",
        "user_id": "u",
        "intent": "clarification",
        "visitor_type": "local",
        "entities": ExtractedEntities(
            visitor_type="local", destination="Shimla", origin="Kasol",
            home_city="Kasol", group_size=4, travel_dates="kal",
        ),
        "route_estimate": ROUTE,
        "weather_data": WEATHER,
        "cab_fare": FARE,
        "snowfall_data": {"risk": "none", "message": "no snowfall expected"},
    }
    kwargs.update(overrides)
    return AgentState(**kwargs)


# ── 1. What it contains ───────────────────────────────────────────────────────


class TestItStatesTheFacts:

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_the_journey_is_confirmed(self, lang):
        reply = build_ride_reply(_state(), lang)
        assert "Kasol" in reply and "Shimla" in reply

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_the_drive_is_quoted_from_the_route(self, lang):
        reply = build_ride_reply(_state(), lang)
        assert "4h 30m" in reply
        assert "215" in reply

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_every_priced_cab_is_listed_with_its_fare(self, lang):
        """The cabs are already narrowed to what seats the party, so all of them are
        bookable and the choice is the rider's."""
        reply = build_ride_reply(_state(), lang)
        for name, fare in [("Economy", "3,200"), ("Sedan", "4,100"), ("SUV", "5,600")]:
            assert name in reply
            assert fare in reply

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_it_ends_at_the_booking_screen(self, lang):
        reply = build_ride_reply(_state(), lang)
        last = reply.splitlines()[-1]
        assert "booking screen" in last.lower() or "बुकिंग स्क्रीन" in last

    def test_a_verified_coupon_is_named(self):
        state = _state(offers=[{
            "coupon_code": "HIMCAB300", "coupon_value_inr": 300,
            "applies_to_this_fare": True,
        }])
        assert "HIMCAB300" in build_ride_reply(state, "en")

    def test_an_unverified_coupon_is_not(self):
        """Eligibility could not be checked against a fare, so naming it would put a
        code on the booking screen that the payment step may reject."""
        state = _state(offers=[{
            "coupon_code": "MAYBE100", "coupon_value_inr": 100,
            "applies_to_this_fare": False,
        }])
        assert "MAYBE100" not in build_ride_reply(state, "en")


# ── 2. What it can no longer contain ──────────────────────────────────────────


class TestItCannotWander:
    """The reported failure, in the form it actually arrived."""

    @pytest.mark.parametrize("lang", ["en", "hi"])
    @pytest.mark.parametrize(
        "brochure",
        ["mall road", "ridge", "jakhu", "kufri", "sightseeing", "hidden gem",
         "restaurant", "hotel", "day 1", "itinerary", "enjoy"],
    )
    def test_no_brochure_content(self, lang, brochure):
        assert brochure not in build_ride_reply(_state(), lang).lower()

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_it_never_asks_a_question(self, lang):
        """The app owns booking. "Shall I book?" invites a yes this service cannot act
        on, and "would you like help planning your whole trip?" is the itinerary flow
        leaking into a cab reply."""
        assert "?" not in build_ride_reply(_state(), lang)

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_the_forecast_and_the_snowfall_risk_stay_out(self, lang):
        """Both are real data, and both belong in the itinerary reply. Here they were
        what produced a multi-day outlook in a reply meant to be a cab and a price.
        The CURRENT condition is one of the four facts and stays."""
        reply = build_ride_reply(_state(), lang)
        assert "forecast" not in reply.lower()
        assert "भविष्यवाणी" not in reply
        assert "जोखिम" not in reply
        assert "19.7" in reply          # the current temperature is still there

    def test_it_stays_short(self):
        """A ride reply is a booking summary. Growth is the failure mode."""
        assert len(build_ride_reply(_state(), "en").splitlines()) <= 8


# ── 3. Missing data is absent, never announced ────────────────────────────────


class TestGapsAreSilent:
    """Announcing our own gaps tells the rider nothing they can act on and reads as a
    broken assistant."""

    def test_no_fare_means_no_number_at_all(self):
        reply = build_ride_reply(_state(cab_fare=None), "en")
        assert "booking screen" in reply.lower()
        assert "₹" not in reply
        for digit_word in ("3,200", "4,100", "5,600"):
            assert digit_word not in reply

    def test_no_weather_means_no_weather_line(self):
        reply = build_ride_reply(_state(weather_data=None), "en")
        assert "°C" not in reply
        assert "unavailable" not in reply.lower()
        assert "could not" not in reply.lower()

    def test_no_route_still_produces_a_usable_reply(self):
        """The journey could not be measured. The cabs and the booking screen are
        still real, so the reply is still worth sending."""
        reply = build_ride_reply(_state(route_estimate=None), "en")
        assert "Kasol" in reply and "Shimla" in reply
        assert "booking screen" in reply.lower()
        assert "km" not in reply


# ── 4. The two languages stay in step ─────────────────────────────────────────


class TestBothLanguagesAreComplete:

    def test_every_line_has_both_translations(self):
        """One table, so a wording change lands in both or fails here rather than
        reaching a rider as a half-English reply."""
        from app.agent.ride_reply import _TEXT

        for key, table in _TEXT.items():
            assert set(table) == {"en", "hi"}, f"{key} is missing a language"
            assert all(v.strip() for v in table.values()), f"{key} has an empty string"

    def test_a_local_reply_is_hindi_by_default(self):
        """The persona default. A resident is answered in Hindi unless they write
        otherwise — see app/agent/language.py."""
        reply = build_ride_reply(_state(), "hi")
        assert "सवारी" in reply
        assert "your ride" not in reply.lower()
