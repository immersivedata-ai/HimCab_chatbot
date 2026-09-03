"""
tests/unit/test_rental_flow.py
──────────────────────────────
Hiring a car by the hour: /v1/chat/rental/*.

The constraint that shapes this whole flow comes from the backend, not from us. A
rental is priced off a RentalChart, and `TripService.findMatchingRentalChart` matches
one by an EXACT comparison — `rentHours == chart.getMaxHour()`. So:

  • the assistant offers the packages the rate card actually defines, fetched live
  • a duration that matches no package yields NO fare, and the reply says so rather
    than quoting a price for something Him Cab does not sell
  • every price comes from POST /fare/estimate; nothing is computed here

On top of the backend's constraint sits the product's: the Rent a Car screen sells
3, 5 and 8 hours — "max 8hrs and min 3hrs" — so a chart outside that window is not
offered even when the rate card defines one, and those three durations are what the
rider is offered when the rate card cannot be read at all.

A rental also has no destination — that is the point of hiring by the hour — so the
flow asks for a duration, a pickup, a date, a time and a cab, and nothing downstream
may demand one.
"""

from datetime import timedelta
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent import sanitize, slots
from app.agent.graph import route_after_planner
from app.agent.nodes.planner import PlannerExtraction, planner_node
from app.agent.nodes.tool_executor import tool_executor_node
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import SlotName
from app.domain.conversation import ExtractedEntities
from app.integrations.himcab.models import FareCategoryFare, FareQuote
from app.integrations.himcab.quote_builder import build_rental_quote_input, matching_package
from app.integrations.himcab.rental_service import (
    RentalPackage,
    RentalService,
    default_packages,
    distinct_packages,
)
from app.services.chat_session import RENTAL, resolve_persona, welcome_message

# What the rate card returns: a chart per category, so hours repeat across cabs. The
# 12-hour chart is deliberately present and deliberately never offered — the backend
# prices it, the product does not sell it.
#: A rate card as the Spring backend returns one. Deliberately carries charts the
#: backend can PRICE but the product does not SELL (3 and 12) alongside the ones it
#: does (4 and 8) — that gap is the whole point of `_is_sold`, and a fixture holding
#: only sellable durations would pass whether or not the filter existed.
RATE_CARD = {"data": [
    {"categoryId": 1, "name": "Economy", "rentalChartList": [
        {"maxHour": 3, "maxKilometer": 30, "farePerKilometer": 14, "gst": 5},
        {"maxHour": 4, "maxKilometer": 40, "farePerKilometer": 13, "gst": 5},
        {"maxHour": 8, "maxKilometer": 80, "farePerKilometer": 13, "gst": 5},
        {"maxHour": 12, "maxKilometer": 120, "farePerKilometer": 12, "gst": 5},
    ]},
    {"categoryId": 2, "name": "SUV", "rentalChartList": [
        {"maxHour": 8, "maxKilometer": 80, "farePerKilometer": 18, "gst": 5},
    ]},
]}

#: The durations the product sells, read from config rather than restated — a test
#: that hard-coded them would keep passing while the assistant offered something
#: else entirely, which is the failure this whole module exists to catch.
SOLD = list(settings.rental_hour_options)

#: A duration nothing sells: absent from `rental_hour_options` AND from the rate card.
UNSOLD_HOURS = 6


def _days_from_today(offset: int) -> str:
    """A rental's travel_dates as the booking screen now receives it: a real date.

    Computed rather than hard-coded, because the value depends on the day the suite
    runs — a literal would pass today and fail tomorrow, which is precisely the class
    of bug that made the offer tests start failing on a date.
    """
    return (sanitize.today_ist() + timedelta(days=offset)).isoformat()


def _parsed() -> list[RentalPackage]:
    """Everything the rate card defines, before the product's allowlist is applied."""
    return distinct_packages(RentalService()._parse_packages(RATE_CARD))


def _packages() -> list[RentalPackage]:
    """What `get_packages` hands back: the rate card narrowed to what we SELL.

    Reads `settings.rental_hour_options` rather than restating it, so changing the
    durations the product sells changes this helper too — a test that hard-coded them
    would pass while the assistant offered something else entirely.
    """
    return [p for p in _parsed() if p.hours in settings.rental_hour_options]


# ── 1. Packages come from the rate card, never from this repo ─────────────────

class TestPackagesComeFromTheRateCard:

    def test_the_rate_card_becomes_the_offered_packages(self):
        assert [p.label for p in _packages()] == [
            "4 hrs / 40 km", "8 hrs / 80 km"
        ]

    def test_the_same_hours_across_cabs_collapse_to_one_choice(self):
        """The rate card defines a chart PER CATEGORY, so 8 hours appears once per cab.
        The rider is picking a duration, not a duration-and-a-car."""
        assert [p.hours for p in _parsed()] == [3, 4, 8, 12]

    @pytest.mark.asyncio
    async def test_a_chart_outside_the_products_window_is_not_offered(self):
        """The screen says max 8 hours. A 12-hour chart the backend can still price must
        not reappear as a chip under a heading that says otherwise."""
        service = RentalService()
        service._get = AsyncMock(return_value=RATE_CARD)

        assert [p.hours for p in await service.get_packages(rider_jwt="token")] == SOLD

    @pytest.mark.parametrize(
        "envelope",
        [
            {"data": RATE_CARD["data"]},
            {"data": {"content": RATE_CARD["data"]}},
            {"data": [{"category_id": 1, "rental_chart_list": [
                {"max_hour": 8, "max_kilometer": 80}]}]},
        ],
    )
    def test_every_envelope_the_backend_uses_is_read(self, envelope):
        """This endpoint has not been exercised against a live backend, so the reader
        accepts each shape the other Him Cab endpoints use."""
        assert RentalService()._parse_packages(envelope)

    @pytest.mark.parametrize("payload", [{"nope": 1}, {"data": []}, "junk", None])
    def test_an_unrecognised_shape_yields_no_packages(self, payload):
        """Empty degrades to asking the rider, which beats inventing hours."""
        assert RentalService()._parse_packages(payload) == []

    @pytest.mark.asyncio
    async def test_a_backend_failure_yields_no_packages_rather_than_raising(self):
        service = RentalService()
        service._get = AsyncMock(side_effect=RuntimeError("backend down"))

        assert await service.get_packages(rider_jwt="token") == []

    @pytest.mark.asyncio
    async def test_a_readable_rate_card_is_cached(self):
        service = RentalService()
        service._get = AsyncMock(return_value=RATE_CARD)

        first = await service.get_packages(rider_jwt="token")
        second = await service.get_packages(rider_jwt="token")

        assert [p.hours for p in first] == [p.hours for p in second] == SOLD
        service._get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_an_unreadable_rate_card_is_never_cached(self):
        """Caching an empty parse would leave the assistant unable to quote for an hour
        AFTER the cause was fixed, and would make the retry that proves it fixed
        impossible. The whole point of a rate card we cannot read is to try again."""
        service = RentalService()
        service._get = AsyncMock(return_value={"data": []})

        assert await service.get_packages(rider_jwt="token") == []
        assert await service.get_packages(rider_jwt="token") == []
        assert service._get.await_count == 2

        # ...and the moment the backend answers properly, it takes effect.
        service._get.return_value = RATE_CARD
        assert [p.hours for p in await service.get_packages(rider_jwt="token")] == SOLD

    @pytest.mark.asyncio
    async def test_switching_rentals_off_offers_nothing_at_all(self):
        """The one case with no fallback: with the feature flag off there is no product
        to offer, so the configured durations must not stand in for one."""
        from app.integrations.himcab.rental_service import available_packages

        with patch.object(settings, "enable_rentals", False):
            assert await available_packages("rider-token") == []

    def test_an_unreadable_rate_card_falls_back_to_the_configured_durations(self):
        """The rider still chooses from the durations Him Cab sells. They carry no
        included kilometres and are flagged as not coming from the rate card, so nothing
        claims a distance allowance for them — but they are still priced, see
        `test_an_unreadable_rate_card_still_gets_a_price`."""
        fallback = default_packages()

        assert [p.hours for p in fallback] == settings.rental_hour_options
        assert not any(p.from_rate_card or p.included_km for p in fallback)


# ── 2. The duration must match a package exactly ──────────────────────────────

class TestDurationMustMatchAPackage:

    @pytest.mark.parametrize(
        "hours,found",
        # 12 is on the rate card and outside the product's window, so it never reaches
        # this list — see `test_a_chart_outside_the_products_window_is_not_offered`.
        [(4, True), (8, True), (3, False), (5, False), (6, False), (12, False),
         (None, False)],
    )
    def test_only_a_configured_duration_matches(self, hours, found):
        """`findMatchingRentalChart` compares hours exactly. Returning the nearest
        package instead would quote five hours to somebody who asked for six."""
        assert (matching_package(_packages(), hours) is not None) is found

    def test_the_quote_is_a_rental_priced_on_the_included_kilometres(self):
        entities = ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4)
        quote = build_rental_quote_input(entities, matching_package(_packages(), 8))

        assert quote.trip_type == "RENTAL"
        assert quote.rent_hours == 8
        # The endpoint needs a distance; the honest one is what the package includes.
        assert quote.distance_km == 80
        assert quote.origin == "Shimla"
        # A rental has no drop-off.
        assert quote.destination == ""

    def test_no_duration_means_no_quote(self):
        assert build_rental_quote_input(ExtractedEntities(origin="Shimla"), None) is None

    @pytest.mark.parametrize(
        "reply,hours",
        [("8", 8), ("8 hours", 8), ("8 hrs", 8), ("8 ghante", 8),
         ("half day", 4), ("full day", 8), ("12hrs", 12),
         ("8.5 hours", None), ("manali", None), ("25", None)],
    )
    def test_a_bare_duration_reply_is_read_in_code(self, reply, hours):
        """The safety net for when the extraction model misses a one-word answer.
        Half-days are rejected rather than rounded — quoting 8 for "8.5" would tell a
        rider they were priced for something other than what they asked for."""
        assert slots.parse_rental_hours(reply) == hours


# ── 3. The conversation asks rental questions, not trip questions ─────────────

class TestTheRentalConversation:

    def _state(self, **entity_kwargs) -> AgentState:
        return AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[AIMessage(content=welcome_message(RENTAL)),
                      HumanMessage(content="I need a car")],
            entities=ExtractedEntities(visitor_type=RENTAL, **entity_kwargs),
        )

    def test_a_rental_never_asks_for_a_destination(self):
        """Hiring by the hour means the rider has not chosen one, so asking would be
        asking them to invent an answer."""
        assert "destination" not in slots.needs_for("rental_query")

    def test_it_asks_for_the_duration_and_the_pickup(self):
        needs = slots.needs_for("rental_query")
        assert needs["rental_hours"] is slots.Need.REQUIRED
        assert needs["origin"] is slots.Need.REQUIRED

    def test_the_welcome_asks_the_first_question(self):
        assert "how long" in welcome_message(RENTAL).lower()

    @pytest.mark.asyncio
    async def test_the_duration_is_asked_for_first(self):
        state = self._state()
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction()
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        assert result["needs_clarification"] is True
        assert result["last_asked_field"] == "rental_hours"

    def test_a_rental_reaches_the_tools_without_a_destination(self):
        """The destination guard protects itineraries and rides. Applying it here would
        deadlock the rental flow, which never collects one."""
        state = self._state(rental_hours=8, origin="Shimla")
        state.needs_clarification = False

        assert route_after_planner(state) == "tool_executor"

    def test_the_rental_route_is_chosen_by_the_rider_not_their_location(self):
        """A GPS fix reroutes /local and /outstation. It must not cancel a rental: their
        coordinates say where to collect them, not that they stopped wanting a car."""
        from app.domain.location import RiderLocation

        outside = RiderLocation(lat=28.6, lon=77.2, in_himachal=False,
                                place_name="Delhi", state="Delhi")
        assert resolve_persona(RENTAL, outside) == (RENTAL, "requested")

    def test_a_rental_gets_the_rental_reply_whoever_asks(self):
        for persona in (RENTAL, "local", "outstation"):
            assert slots.response_mode("rental_query", persona) == slots.RENTAL_MODE

    # ── The persona outranks whatever the router made of THIS message ─────────
    #
    # The router re-classifies intent every turn, and in a rental chat "kasol" or
    # "shimla" looks exactly like trip planning. When the question policy followed that
    # drift, a rider hiring a car for four hours was asked where they wanted to GO and
    # for how many DAYS, and their reply was filed as a destination.

    @pytest.mark.parametrize(
        "drifted_intent",
        ["plan_trip", "show_itinerary", "places_query", "weather_query", None],
    )
    def test_the_questions_stay_rental_whatever_the_router_says(self, drifted_intent):
        needs = slots.needs_for(drifted_intent, RENTAL)

        assert "destination" not in needs, "asked a rental rider where they are going"
        assert "duration_days" not in needs, "asked a rental rider how many DAYS"
        assert needs["rental_hours"] is slots.Need.REQUIRED

    @pytest.mark.parametrize(
        "drifted_intent", ["plan_trip", "show_itinerary", "places_query"]
    )
    def test_the_reply_stays_rental_whatever_the_router_says(self, drifted_intent):
        assert slots.response_mode(drifted_intent, RENTAL) == slots.RENTAL_MODE

    def test_the_trip_checklist_is_untouched_for_everyone_else(self):
        """The rental tables must not have leaked into the other personas.

        `duration_days` is asked of a VISITOR, who is being handed a day-by-day plan.
        A local is booking a ride and is not asked it — that is `_RIDE_NEEDS`, not
        rental bleed — so the two are asserted apart rather than lumped together.
        """
        for persona in (None, "outstation"):
            needs = slots.needs_for("plan_trip", persona)
            assert needs["destination"] is slots.Need.REQUIRED
            assert "duration_days" in needs

        local = slots.needs_for("plan_trip", "local")
        assert local["destination"] is slots.Need.REQUIRED
        assert local["group_size"] is slots.Need.IMPORTANT
        assert "duration_days" not in local

    @pytest.mark.asyncio
    @pytest.mark.parametrize("pickup", ["Delhi", "Chandigarh", "Shimla", "Manali"])
    async def test_a_pickup_anywhere_does_not_cancel_the_rental(self, pickup):
        """A rental is collected from wherever the rider is -- Delhi, Chandigarh,
        Ambala. Reading an out-of-state pickup as 'a visitor' turned the hire into a
        holiday: they were asked where they wanted to go and handed an itinerary."""
        from app.agent.nodes.planner import classify_visitor_type

        assert await classify_visitor_type(pickup, current=RENTAL) == RENTAL

    @pytest.mark.asyncio
    async def test_naming_a_himachal_home_does_not_cancel_the_rental_either(self):
        from app.agent.nodes.planner import classify_visitor_type

        assert await classify_visitor_type(
            "Delhi", home_city="Shimla", current=RENTAL
        ) == RENTAL

    def test_a_rental_never_skips_the_planner_for_an_itinerary(self):
        """`show_itinerary` + a destination the extractor picked up in passing would
        otherwise jump straight to the writer, handing a four-hour hire a trip plan.

        A rental routes to `planner` directly (skipping the `context` node, whose user
        profile no rental reply reads) — the point is that it still reaches the planner,
        which strips the stray destination, rather than the writer."""
        from app.agent.graph import route_after_router

        state = AgentState(
            session_id="s", user_id="u", intent="show_itinerary", visitor_type=RENTAL,
            entities=ExtractedEntities(visitor_type=RENTAL, destination="Shimla",
                                       rental_hours=4),
        )
        assert route_after_router(state) == "planner"

    def test_the_itinerary_shortcut_still_works_for_a_real_trip(self):
        from app.agent.graph import route_after_router

        state = AgentState(
            session_id="s", user_id="u", intent="show_itinerary",
            visitor_type="outstation",
            entities=ExtractedEntities(destination="Shimla"),
        )
        assert route_after_router(state) == "writer"

    def test_a_rejected_answer_is_re_asked_with_a_hint_not_verbatim(self):
        """Repeating a question word for word reads as a stuck loop. A rider who
        answers "shimla" to "when are you travelling?" has misread it."""
        first = slots.SLOTS["travel_dates"].ask("en", RENTAL, 0)
        again = slots.SLOTS["travel_dates"].ask("en", RENTAL, 1)
        assert first != again
        assert "today" in again.lower() or "date" in again.lower()

        first_pickup = slots.SLOTS["origin"].ask("en", RENTAL, 0)
        again_pickup = slots.SLOTS["origin"].ask("en", RENTAL, 1)
        assert first_pickup != again_pickup


# ── 4. The fare comes from the microservice, or not at all ────────────────────

class TestTheFareComesFromTheBackend:

    def _state(self, **entity_kwargs) -> AgentState:
        entities = {"visitor_type": RENTAL, "origin": "Shimla", "group_size": 4}
        entities.update(entity_kwargs)
        return AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token", entities=ExtractedEntities(**entities),
        )

    def _services(self, fare_payload=None, rate_card=RATE_CARD):
        rental = RentalService()
        rental._get = AsyncMock(return_value=rate_card)

        fare = MagicMock()
        captured: dict = {}

        async def get_quote(quote_input, rider_jwt):
            captured["input"] = quote_input
            if fare_payload is None:
                raise RuntimeError("no fare")
            return fare_payload

        fare.get_quote = get_quote

        weather = MagicMock()
        weather.get_weather = AsyncMock(side_effect=Exception("not under test"))
        weather.snowfall_from_report = MagicMock(return_value=None)
        return rental, fare, weather, captured

    def _quote(self) -> FareQuote:
        return FareQuote(
            base_fare_inr=2400, category_name="Sedan", trip_type="RENTAL",
            alternatives=[
                FareCategoryFare(category_name="Economy", total_fare_inr=2100),
                FareCategoryFare(category_name="Sedan", total_fare_inr=2400),
            ],
        )

    async def _run(self, state, rental, fare, weather):
        from app.core.config import settings

        with patch("app.agent.nodes.tool_executor.get_rental_service", return_value=rental), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
             patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather), \
             patch.object(settings, "enable_offers", False):
            return await tool_executor_node(state)

    @pytest.mark.asyncio
    async def test_the_backend_is_asked_for_a_rental_priced_on_the_package(self):
        rental, fare, weather, captured = self._services(self._quote())

        result = await self._run(self._state(rental_hours=8), rental, fare, weather)

        sent = captured["input"]
        assert sent.trip_type == "RENTAL"
        assert sent.rent_hours == 8
        assert sent.distance_km == 80          # the package's included km
        assert result["cab_fare"]["fare_inr"] == 2400
        assert result["cab_fare"]["rental_hours"] == 8
        assert result["cab_fare"]["included_km"] == 80

    @pytest.mark.asyncio
    async def test_the_offered_packages_reach_the_reply(self):
        rental, fare, weather, _ = self._services(self._quote())

        result = await self._run(self._state(rental_hours=8), rental, fare, weather)

        assert [p["hours"] for p in result["rental_packages"]] == SOLD
        assert all(p["from_rate_card"] for p in result["rental_packages"])

    @pytest.mark.asyncio
    async def test_an_unlisted_duration_is_never_sent_to_the_backend(self):
        """6 hours matches no chart, so a quote for it would come back empty and the
        rider would be told nothing useful. It is not attempted."""
        rental, fare, weather, captured = self._services(self._quote())

        result = await self._run(
            self._state(rental_hours=UNSOLD_HOURS), rental, fare, weather
        )

        assert "input" not in captured, "asked the backend to price an unlisted duration"
        assert result["cab_fare"] is None
        # The packages still reach the reply, so it can offer the real ones.
        assert [p["hours"] for p in result["rental_packages"]] == SOLD

    @pytest.mark.asyncio
    async def test_an_unreadable_rate_card_still_gets_a_price(self):
        """The rate card is a convenience for US, not a precondition for a price.

        Reported from the app: the rider picked 3 hrs off the chips and got "no price
        came back", on a turn where the backend would have priced it perfectly well. The
        backend matches the chart itself from `rentHours` and the rental price is
        `farePerKilometer x maxKilometer` off that chart — `distanceInMeters` plays no
        part in it — so refusing to ASK because we could not parse the card ourselves
        cost the rider their price for nothing."""
        rental, fare, weather, captured = self._services(
            self._quote(), rate_card={"data": []}
        )

        result = await self._run(self._state(rental_hours=8), rental, fare, weather)

        # The rider still chooses from the durations the product sells...
        assert [p["hours"] for p in result["rental_packages"]] == SOLD
        assert not any(p["from_rate_card"] for p in result["rental_packages"])
        # ...and the backend is still asked to price the one they picked.
        assert captured["input"].rent_hours == 8
        assert captured["input"].trip_type == "RENTAL"
        assert result["cab_fare"]["fare_inr"] == 2400
        # What the package INCLUDES is unknown without the card, so nothing claims it.
        assert result["cab_fare"]["included_km"] is None
        # 8 hours IS one of the offered durations, so it is not thrown back at the rider.
        assert "unbookable_rental_hours" not in result

    @pytest.mark.asyncio
    async def test_a_duration_nothing_offers_is_still_never_priced(self):
        """The gate that remains: we ask the backend only about durations we offered."""
        rental, fare, weather, captured = self._services(
            self._quote(), rate_card={"data": []}
        )

        result = await self._run(
            self._state(rental_hours=UNSOLD_HOURS), rental, fare, weather
        )

        assert "input" not in captured
        assert result["unbookable_rental_hours"] == UNSOLD_HOURS

    @pytest.mark.asyncio
    async def test_a_dead_fare_backend_costs_the_price_not_the_reply(self):
        rental, fare, weather, _ = self._services(fare_payload=None)

        result = await self._run(self._state(rental_hours=8), rental, fare, weather)

        assert result["cab_fare"] is None
        assert result["rental_packages"]           # the turn still has something to say


# ── 5. What the reply is allowed to say ───────────────────────────────────────

class TestTheRentalReply:

    FARE: ClassVar[dict] = {
        "category_name": "Sedan", "fare_inr": 2400, "rental_hours": 8,
        "included_km": 80, "distance_km": 80,
        "options": [
            {"category_name": "Economy", "total_fare_inr": 2100, "seat_capacity": 4},
            {"category_name": "Sedan", "total_fare_inr": 2400, "seat_capacity": 4},
        ],
    }

    def _prompt(self, **state_kwargs) -> str:
        from app.agent.nodes.writer import _build_system_prompt

        base = {
            "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
            "entities": ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4),
            "rental_packages": [
                {"hours": 4, "included_km": 40, "label": "4 hrs / 40 km"},
                {"hours": 8, "included_km": 80, "label": "8 hrs / 80 km"},
            ],
        }
        base.update(state_kwargs)
        return _build_system_prompt(AgentState(**base))

    def test_it_offers_only_the_configured_packages(self):
        prompt = self._prompt(cab_fare=self.FARE)

        assert "4 hrs / 40 km, 8 hrs / 80 km" in prompt
        assert "Offer ONLY these" in prompt
        assert "The rider chose: 8 hrs / 80 km" in prompt

    def test_it_quotes_the_backend_prices_exactly(self):
        prompt = self._prompt(cab_fare=self.FARE)

        assert "2,100" in prompt and "2,400" in prompt
        assert "Do NOT round, adjust, average or invent a number" in prompt
        # Described as a package, not as a drive between two towns.
        assert "8-hour package" in prompt and "80 km included" in prompt

    def test_it_writes_no_itinerary_and_no_route(self):
        prompt = self._prompt(cab_fare=self.FARE)

        assert "ITINERARY JSON" not in prompt
        assert "THIS JOURNEY" not in prompt
        assert "FORBIDDEN" in prompt
        for banned in ("day-by-day plan", "a route", "a drive time"):
            assert banned in prompt, f"{banned!r} is not forbidden"

    def test_an_unlisted_duration_is_explained_rather_than_priced(self):
        prompt = self._prompt(
            entities=ExtractedEntities(origin="Shimla", rental_hours=5, group_size=4),
            cab_fare=None,
        )

        assert "NOT one of the packages" in prompt
        assert "quote a price for 5 hours" in prompt

    def test_a_turn_with_no_chosen_duration_asks_instead_of_confirming(self):
        """The worst bug this flow has had. Asked for 5 hours -- which Him Cab does not
        sell -- the reply described the 4-hour package as though it had been chosen, and
        announced a booking screen that (correctly) never opened. A turn with no
        bookable duration is a QUESTION; only a chosen one confirms anything."""
        prompt = self._prompt(
            entities=ExtractedEntities(origin="Shimla", rental_hours=None, group_size=4),
            unbookable_rental_hours=5,
            cab_fare=None,
        )

        assert "not one of the rental packages" in prompt
        assert "Ask which one they would like" in prompt or "ask which one" in prompt.lower()
        # The three claims that made the old reply wrong.
        assert "no mention of a booking screen" in prompt
        assert "Do not describe any single package as though they had picked it" in prompt
        assert "never quote or estimate a price for 5 hours" in prompt
        # ...and none of the confirmation scaffolding leaks in.
        assert "The booking screen is open" not in prompt
        assert "One bullet PER CAB" not in prompt

    def test_no_booking_handoff_without_a_chosen_duration(self):
        """The text and the event read the SAME gate, so a reply can never announce a
        booking screen the app does not open."""
        from app.agent.nodes.writer import _build_booking_handoff, _build_system_prompt

        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla", rental_hours=None, group_size=4),
            unbookable_rental_hours=5,
        )

        assert _build_booking_handoff(state) is None
        assert "booking screen is open" not in _build_system_prompt(state)

    def test_the_confirming_reply_speaks_to_the_rider_not_about_the_company(self):
        """"Him Cab is ready to provide your rental car" reads like a form letter. The
        reply is one side of a conversation, so it says 'we' and 'you'."""
        prompt = self._prompt(cab_fare=self.FARE)

        assert "Never refer to Him Cab in the third person" in prompt
        # The bullets used to come back as '**Package:**' with a nested cab sub-list.
        assert "no bold labels" in prompt
        assert "no indented sub-lists" in prompt

    def test_a_missing_weather_report_is_left_out_not_narrated(self):
        """Reported from a live chat: "There is no information available about the
        current temperature or conditions at your pickup location." A gap in OUR data
        is not news the rider can act on -- the line simply does not get written."""
        prompt = self._prompt(cab_fare=self.FARE, weather_data=None)

        assert "Write NO weather bullet" in prompt
        assert "do not say that they are unavailable" in prompt
        # The conditional phrasing that invited it is gone: an instruction the model
        # cannot satisfy is an instruction it will answer badly.
        assert "Skip it entirely if it does not" not in prompt

    def test_a_report_with_no_temperature_counts_as_no_report(self):
        """A truthy dict with a null temperature promised a number that was not there."""
        prompt = self._prompt(
            cab_fare=self.FARE,
            weather_data={"current_temp_c": None, "condition": None,
                          "forecast_7day": [], "alerts": []},
        )

        assert "Write NO weather bullet" in prompt

    def test_a_real_reading_is_still_required(self):
        prompt = self._prompt(
            cab_fare=self.FARE,
            weather_data={"current_temp_c": 17.5, "condition": "clear sky",
                          "forecast_7day": [], "alerts": []},
        )

        assert "Current temperature and condition at the pickup" in prompt
        assert "Write NO weather bullet" not in prompt

    def test_no_gap_of_ours_is_ever_announced(self):
        """One rule, every mode: weather, coupons, fares. Naming the situation in the
        instruction is what got the situation written down as a bullet."""
        prompt = self._prompt(cab_fare=None, weather_data=None)

        assert "never ANNOUNCE that something is missing" in prompt
        assert "Write NOTHING about coupons" in prompt
        assert "not a line saying none applies" in prompt

    def test_with_no_fare_it_must_not_invent_one(self):
        prompt = self._prompt(cab_fare=None)

        assert "do NOT state, estimate or imply any price" in prompt

    def test_it_never_asks_the_rider_to_confirm_a_booking(self):
        prompt = self._prompt(cab_fare=self.FARE)

        assert "It must NOT be a question" in prompt
        assert "do not ask them to confirm" in prompt

    def test_the_handoff_carries_the_package_and_no_destination(self):
        from app.agent.nodes.writer import _build_booking_handoff

        handoff = _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4,
                                       travel_dates="kal"),
            cab_fare=self.FARE,
        ))

        assert handoff["trip_type"] == "RENTAL"
        assert handoff["rental_hours"] == 8
        assert handoff["included_km"] == 80
        assert handoff["origin"]["name"] == "Shimla"
        assert handoff["destination"] is None
        assert handoff["fare_inr"] == 2400

    def test_no_handoff_before_a_duration_is_chosen(self):
        from app.agent.nodes.writer import _build_booking_handoff

        assert _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla"),
        )) is None


# ── No question is ever put twice in the same words ───────────────────────────
#
# Reported from a live chat: the rider typed "8 hourse", and the assistant replied with
# the identical sentence it had just sent. Two defects in one turn — the typo threw the
# answer away, and the re-ask gave no hint that anything had gone wrong.

class TestAQuestionIsNeverRepeatedVerbatim:

    @pytest.mark.parametrize(
        "reply,hours",
        # The unit is matched by stem, so how they spell "hours" is not our business.
        [("8", 8), ("8 hours", 8), ("8 hourse", 8), ("8 hrs", 8), ("8 hr", 8),
         ("8 hours.", None),                       # trailing punctuation is the LLM's job
         ("8 ghante", 8), ("8 ghantey", 8), ("8 घंटे", 8),
         ("half day", 4), ("full day", 8),
         # Still strict about the things that change the price.
         ("8.5 hours", None), ("8 days", None), ("manali", None), ("25", None)],
    )
    def test_a_typo_in_the_unit_does_not_cost_a_turn(self, reply, hours):
        assert slots.parse_rental_hours(reply) == hours

    @pytest.mark.parametrize(
        "slot",
        # Every slot the agent ever ASKS for. A slot added here without a retry variant
        # fails this test rather than shipping a chat that repeats itself.
        ["rental_hours", "pickup_time", "origin", "travel_dates",
         "duration_days", "group_size", "destination"],
    )
    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_every_slot_re_asks_in_different_words(self, slot, lang):
        first = slots.SLOTS[slot].ask(lang, RENTAL, 0)
        again = slots.SLOTS[slot].ask(lang, RENTAL, 1)

        assert first and again
        assert first != again, f"{slot} repeats itself verbatim in {lang}"

    def test_the_package_list_is_re_asked_differently_too(self):
        labels = ["4 hrs / 40 km", "8 hrs / 80 km"]
        first = slots.rental_hours_question(labels, "en", 0)
        again = slots.rental_hours_question(labels, "en", 1)

        assert first != again
        # Both still name the real packages -- the wording changes, the options do not.
        for label in labels:
            assert label in first and label in again

    def test_with_no_rate_card_the_re_ask_still_differs(self):
        assert slots.rental_hours_question([], "en", 0) != \
               slots.rental_hours_question([], "en", 1)

    @pytest.mark.parametrize("lang", ["en", "hi"])
    def test_the_third_ask_differs_from_the_second(self, lang):
        """Only a REQUIRED slot gets this far -- everything else is assumed past the
        attempt cap -- so a rider stuck here is told plainly why we keep asking."""
        asks = [
            slots.decide(
                ExtractedEntities(visitor_type=RENTAL, rental_hours=8,
                                  asked_counts={"origin": n}),
                "rental_query", RENTAL, lang, max_attempts=2,
            ).question
            for n in (0, 1, 2)
        ]

        assert len(set(asks)) == 3, f"an ask repeats verbatim: {asks}"
        assert all(a for a in asks)

    def test_the_escalation_never_hits_an_escapable_slot(self):
        """`group_size` is IMPORTANT: past the cap it is assumed, not chased. Reaching
        the third-ask wording would mean the cap had stopped working."""
        decision = slots.decide(
            ExtractedEntities(visitor_type=RENTAL, rental_hours=8, origin="Shimla",
                              asked_counts={"group_size": 2, "travel_dates": 2,
                                            "pickup_time": 2}),
            "rental_query", RENTAL, "en", max_attempts=2,
        )

        assert decision.ask_field is None
        assert "group_size" in decision.assumed


# ── The vocabulary is defined once, in code ───────────────────────────────────

class TestTheImportantValuesAreEnums:
    """Every one of these used to be a bare string, respelled in each module that
    needed it. Nothing checked the spellings matched, so a drifted value failed
    silently: the persona compare returned False and the rental chat fell back to the
    trip checklist. A typo is now an AttributeError at import, not a wrong reply."""

    def test_one_set_of_persona_objects_for_the_whole_system(self):
        from app.agent.nodes import planner
        from app.core.enums import Persona
        from app.services import chat_session

        assert slots.RENTAL is Persona.RENTAL is chat_session.RENTAL is planner.RENTAL
        assert slots.LOCAL is Persona.LOCAL is chat_session.LOCAL is planner.LOCAL

    def test_the_wire_values_are_unchanged(self):
        """These strings are a contract with the Spring backend and with the app."""
        from app.core.enums import Persona, ResponseMode, TripType

        assert Persona.RENTAL == "rental"
        assert TripType.RENTAL == "RENTAL"
        assert ResponseMode.RENTAL == "rental"
        # str() must give the value, not "TripType.RENTAL" -- these go into payloads.
        assert str(TripType.RENTAL) == "RENTAL"
        assert f"{TripType.OUTSTATION}" == "OUTSTATION"

    def test_the_rental_quote_is_typed_not_spelled(self):
        from app.core.enums import TripType

        quote = build_rental_quote_input(
            ExtractedEntities(origin="Shimla", rental_hours=8),
            matching_package(_packages(), 8),
        )
        assert quote.trip_type is TripType.RENTAL

    def test_every_policy_table_names_a_real_slot(self):
        """A needs table referencing a slot that does not exist is a question silently
        never asked. The enum keys make that impossible; this proves it."""
        from app.core.enums import Intent, SlotName

        for intent in Intent:
            for name in slots.needs_for(intent):
                assert SlotName.has(name), f"{intent}: {name!r} is not a SlotName"
                assert name in slots.SLOTS

    def test_every_intent_and_persona_pair_yields_a_known_mode(self):
        """The writer branches on exactly one mode, so an unmapped pair would fall
        through to whichever branch happened to be last."""
        from app.core.enums import Intent, Persona, ResponseMode

        seen = {
            slots.response_mode(intent, persona)
            for intent in Intent
            for persona in (None, *Persona)
        }
        assert seen <= set(ResponseMode)
        assert ResponseMode.RENTAL in seen


# ── The pickup TIME, which is not the same question as the day ────────────────
#
# The backend prices a rental off `bookingTime` and applies the night rate from it
# (`TripService.isNightTime`). "kal" alone left the quote standing on whatever time the
# rider happened to be chatting at, so a 6 a.m. pickup arranged at 10 p.m. was priced
# as a night run — and then charged the day rate at booking.

class TestThePickupTimeIsAskedFor:

    def test_the_rental_flow_asks_for_it(self):
        needs = slots.needs_for("rental_query")
        assert "pickup_time" in needs
        # The day and the hour are separate questions, both wanted.
        assert "travel_dates" in needs

    def test_the_day_question_and_the_time_question_are_different(self):
        day = slots.SLOTS["travel_dates"].ask("en", RENTAL, 0)
        time = slots.SLOTS["pickup_time"].ask("en", RENTAL, 0)
        assert day != time
        assert "time" in time.lower()

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("6", "06:00"), ("6 am", "06:00"), ("6:30am", "06:30"), ("6.30 am", "06:30"),
            ("18:00", "18:00"), ("18", "18:00"), ("00:30", "00:30"),
            ("6 baje", "06:00"), ("subah 6", "06:00"), ("subah 6 baje", "06:00"),
            ("shaam 5", "17:00"), ("7 baje shaam", "19:00"), ("raat 9", "21:00"),
            ("12 am", "00:00"), ("12 pm", "12:00"), ("9pm", "21:00"),
            ("24:00", "00:00"),
        ],
    )
    def test_a_clock_time_is_read_in_code(self, reply, expected):
        assert slots.parse_pickup_time(reply) == expected

    @pytest.mark.parametrize(
        "reply",
        # A vague part of the day is NOT a time. Mapping "morning" to 09:00 would invent
        # the one number the price is calculated from, so it is rejected and re-asked.
        ["morning", "subah", "evening", "shaam", "raat", "night", "afternoon",
         # ...and neither is a day, a place, or a non-answer.
         "kal", "tomorrow", "manali", "yes", "no", "idk", "", "  ",
         # Nor an impossible or self-contradictory reading.
         "25", "13:70", "subah 6 pm", "what time is good?"],
    )
    def test_anything_that_is_not_a_clock_time_is_rejected(self, reply):
        assert slots.parse_pickup_time(reply) is None

    def test_the_backend_is_priced_on_the_pickup_hour_not_the_chat_hour(self):
        entities = ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4,
                                     travel_dates="kal", pickup_time="06:00")
        quote = build_rental_quote_input(entities, matching_package(_packages(), 8))

        assert quote.booking_time == "06:00"

    def test_with_no_time_given_it_still_quotes(self):
        """Falling back to now is the old behaviour, and better than refusing to price."""
        entities = ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4)
        quote = build_rental_quote_input(entities, matching_package(_packages(), 8))

        assert quote.booking_time and ":" in quote.booking_time

    def test_the_handoff_carries_the_time_the_fare_was_priced_on(self):
        """The booking screen must open on the same hour we quoted, or the rider lands
        on a default time and is charged a rate they were never shown."""
        from app.agent.nodes.writer import _build_booking_handoff

        handoff = _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4,
                                       travel_dates="kal", pickup_time="06:00"),
            cab_fare={"fare_inr": 2400, "included_km": 80, "options": []},
        ))

        assert handoff["pickup_time"] == "06:00"
        assert handoff["travel_dates"] == "kal"

    def test_the_reply_is_told_to_state_it_back(self):
        from app.agent.nodes.writer import _build_system_prompt

        prompt = _build_system_prompt(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla", rental_hours=8, group_size=4,
                                       travel_dates="kal", pickup_time="06:00"),
            rental_packages=[{"hours": 8, "included_km": 80, "label": "8 hrs / 80 km"}],
            cab_fare={"fare_inr": 2400, "included_km": 80, "options": [
                {"category_name": "Sedan", "total_fare_inr": 2400, "seat_capacity": 4}]},
        ))

        assert "Pickup time: 06:00 on kal" in prompt
        assert "the fare is calculated from this time" in prompt

    def test_the_other_flows_are_not_asked_for_a_pickup_time(self):
        """Scoped to rentals: adding a question to the trip flow was not the ask."""
        for intent in ("plan_trip", "weather_query", "places_query", "cost_query"):
            assert "pickup_time" not in slots.needs_for(intent)


# ── 6. The question names what can actually be booked ─────────────────────────
#
# The rider is asked for a duration BEFORE any of it has been priced, and the backend
# matches a duration exactly. Asking an open question therefore invites an answer that
# cannot be filled — and the rider only finds that out after also giving a pickup and a
# date. So the asking turn reads the rate card too.

class TestTheDurationQuestionOffersTheRealPackages:

    def _clarify_state(self, **kwargs) -> AgentState:
        base = {
            "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
            "rider_jwt": "rider-token",
            "messages": [AIMessage(content=welcome_message(RENTAL)),
                      HumanMessage(content="I need a car tomorrow")],
            "entities": ExtractedEntities(visitor_type=RENTAL),
            "needs_clarification": True,
            "clarification_question": slots.rental_hours_question([], "en"),
            "last_asked_field": "rental_hours",
        }
        base.update(kwargs)
        return AgentState(**base)

    async def _clarify(self, state, rate_card=RATE_CARD):
        from app.agent.graph import clarify_node

        rental = RentalService()
        rental._get = AsyncMock(return_value=rate_card)
        with patch("app.integrations.himcab.rental_service.get_rental_service", return_value=rental):
            return await clarify_node(state)

    def test_the_greeting_itself_offers_the_packages(self):
        """The welcome asks for the duration, so it is the FIRST place that can offer
        an unbookable one. It gets the same list as every question after it."""
        greeting = welcome_message(RENTAL, [
            {"hours": 4, "included_km": 40, "label": "4 hrs / 40 km"},
            {"hours": 8, "included_km": 80, "label": "8 hrs / 80 km"},
        ])
        assert "4 hrs / 40 km" in greeting
        assert "8 hrs / 80 km" in greeting
        assert "Him Cab" in greeting              # still a greeting, not a bare list

    def test_the_greeting_falls_back_to_the_open_question(self):
        assert "how long" in welcome_message(RENTAL, []).lower()
        assert "how long" in welcome_message(RENTAL).lower()

    def test_only_a_rental_greeting_changes(self):
        packages = [{"hours": 8, "included_km": 80, "label": "8 hrs / 80 km"}]
        for persona in ("local", "outstation"):
            assert welcome_message(persona, packages) == welcome_message(persona)

    def test_the_wording_lists_every_package(self):
        text = slots.rental_hours_question(
            ["4 hrs / 40 km", "8 hrs / 80 km", "12 hrs / 120 km"], "en"
        )
        assert "4 hrs / 40 km" in text
        assert "8 hrs / 80 km" in text
        assert "12 hrs / 120 km" in text

    def test_no_packages_means_the_open_question_stands(self):
        """An unreachable rate card must not produce a question with an empty list in
        it — the assistant falls back to asking, exactly as it did before."""
        assert slots.rental_hours_question([], "en") == slots.rental_hours_question([], "en")
        assert "package" not in slots.rental_hours_question([], "en").lower()
        assert "how long" in slots.rental_hours_question([], "en").lower()

    def test_the_question_is_asked_in_the_riders_language(self):
        assert "घंटे" in slots.rental_hours_question([], "hi")
        assert "पैकेज" in slots.rental_hours_question(["4 hrs / 40 km"], "hi")

    @pytest.mark.asyncio
    async def test_the_asking_turn_names_the_rate_cards_packages(self):
        result = await self._clarify(self._clarify_state())

        assert "4 hrs / 40 km" in result["final_response_text"]
        assert "8 hrs / 80 km" in result["final_response_text"]
        assert "8 hrs / 80 km" in result["final_response_text"]
        # The 12-hour chart is on the rate card and outside the product's window.
        assert "12 hrs" not in result["final_response_text"]

    @pytest.mark.asyncio
    async def test_the_packages_travel_as_an_event_the_app_can_render_as_taps(self):
        result = await self._clarify(self._clarify_state())

        events = result["structured_events"]
        assert [e["type"] for e in events] == ["rental_packages"]
        assert [p["hours"] for p in events[0]["payload"]["packages"]] == SOLD
        assert result["rental_packages"] == events[0]["payload"]["packages"]

    @pytest.mark.asyncio
    async def test_an_unreachable_rate_card_still_offers_a_choice(self):
        """The rider picks from the durations the product sells rather than being asked
        an open question. Only the price is lost, and the reply never states one."""
        result = await self._clarify(self._clarify_state(), rate_card={"data": []})

        text = result["final_response_text"]
        assert "4 hrs" in text and "8 hrs" in text
        packages = result["structured_events"][0]["payload"]["packages"]
        assert not any(p["from_rate_card"] for p in packages)

    @pytest.mark.asyncio
    async def test_no_rider_token_means_no_backend_call(self):
        """A rate card is rider-authenticated. Without a token the flow falls back to the
        configured durations rather than calling the backend without one."""
        from app.agent.graph import clarify_node

        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        state = self._clarify_state(rider_jwt=None)
        with patch("app.integrations.himcab.rental_service.get_rental_service", return_value=rental):
            result = await clarify_node(state)

        rental._get.assert_not_awaited()
        assert [p["hours"] for p in result["rental_packages"]] == settings.rental_hour_options

    @pytest.mark.asyncio
    async def test_only_the_duration_question_fetches_the_rate_card(self):
        """Every other question in every other flow goes through this node too."""
        from app.agent.graph import clarify_node

        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        state = self._clarify_state(
            clarification_question="Where should the car pick you up?",
            last_asked_field="origin",
        )
        with patch("app.integrations.himcab.rental_service.get_rental_service", return_value=rental):
            result = await clarify_node(state)

        rental._get.assert_not_awaited()
        assert result["final_response_text"] == "Where should the car pick you up?"
        assert result["structured_events"] == []


# ── 7. An unbookable duration leaves the question in flight ───────────────────

class TestAnUnlistedDurationIsReAsked:

    @pytest.mark.asyncio
    async def test_the_duration_question_goes_back_in_flight(self):
        """The reply tells the rider 6 hours is not a package and lists the real ones.
        The next message is therefore an answer to that — mark it as such, or a bare
        "8" is discarded and the rider repeats themselves forever."""
        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        fare = MagicMock()
        fare.get_quote = AsyncMock(side_effect=AssertionError("must not be priced"))
        weather = MagicMock()
        weather.get_weather = AsyncMock(side_effect=Exception("not under test"))
        weather.snowfall_from_report = MagicMock(return_value=None)

        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token",
            entities=ExtractedEntities(
                visitor_type=RENTAL, origin="Shimla", rental_hours=UNSOLD_HOURS,
                last_asked_field=None,
            ),
        )
        with patch("app.agent.nodes.tool_executor.get_rental_service", return_value=rental), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
             patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather), \
             patch.object(settings, "enable_offers", False):
            result = await tool_executor_node(state)

        assert result["last_asked_field"] == "rental_hours"
        assert result["entities"].last_asked_field == "rental_hours"
        # Cleared as well as re-asked: a slot that still holds a value is never re-read
        # (slots.read_answer), so leaving 6 in place would make the correction a no-op.
        assert result["entities"].rental_hours is None
        # …but the reply still has to name what they asked for.
        assert result["unbookable_rental_hours"] == UNSOLD_HOURS

    # ── …and it is refused on the turn it arrives, not four questions later ───
    #
    # The duration used to be checked only where the fare is requested, which is after
    # the pickup, the date, the time and the headcount have all been collected. A rider
    # who opened with "6 hours" answered four more questions before being told the first
    # answer had never been usable. Now the planner checks it against the offered list
    # the moment it is extracted.

    async def _plan(self, text, entities, rate_card=None):
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token" if rate_card else None,
            messages=[AIMessage(content=welcome_message(RENTAL)),
                      HumanMessage(content=text)],
            entities=entities,
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction()
        )
        rental = RentalService()
        rental._get = AsyncMock(return_value=rate_card or {"data": []})
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc, \
             patch("app.integrations.himcab.rental_service.get_rental_service",
                   return_value=rental):
            svc.return_value.get_flash_model.return_value = llm
            return await planner_node(state)

    @pytest.mark.asyncio
    async def test_an_unsold_duration_is_refused_immediately(self):
        result = await self._plan(
            f"I need a car for {UNSOLD_HOURS} hours from Shimla tomorrow at 9 am",
            ExtractedEntities(visitor_type=RENTAL, last_asked_field="rental_hours"),
            rate_card=RATE_CARD,
        )

        assert result["unbookable_rental_hours"] == UNSOLD_HOURS
        # Cleared and back in flight, so the correction is read as a fresh answer.
        assert result["entities"].rental_hours is None
        assert result["last_asked_field"] == "rental_hours"
        # …and everything else they said is KEPT: they should not have to repeat it.
        assert result["entities"].origin == "Shimla"
        assert result["entities"].pickup_time == "09:00"

    @pytest.mark.asyncio
    async def test_the_question_names_the_duration_it_is_refusing(self):
        """A re-ask that does not mention the 6 hours reads as the assistant forgetting.
        The refusal and the alternatives belong in one message."""
        from app.agent.graph import clarify_node

        planned = await self._plan(
            f"{UNSOLD_HOURS} hours please",
            ExtractedEntities(visitor_type=RENTAL, last_asked_field="rental_hours"),
            rate_card=RATE_CARD,
        )
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token",
            entities=planned["entities"],
            needs_clarification=True,
            clarification_question=planned["clarification_question"],
            last_asked_field=planned["last_asked_field"],
            unbookable_rental_hours=planned["unbookable_rental_hours"],
        )
        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        with patch("app.integrations.himcab.rental_service.get_rental_service",
                   return_value=rental):
            result = await clarify_node(state)

        text = result["final_response_text"]
        assert f"{UNSOLD_HOURS}-hour package" in text
        assert "4 hrs / 40 km" in text and "8 hrs / 80 km" in text

    @pytest.mark.asyncio
    async def test_an_offered_duration_is_left_alone(self):
        result = await self._plan(
            "8 hours from Shimla tomorrow at 9 am, 4 of us",
            ExtractedEntities(visitor_type=RENTAL, last_asked_field="rental_hours"),
            rate_card=RATE_CARD,
        )

        assert result["unbookable_rental_hours"] is None
        assert result["entities"].rental_hours == 8
        assert result["needs_clarification"] is False

    @pytest.mark.asyncio
    async def test_a_duration_is_never_refused_on_a_guess(self):
        """With no offered list there is nothing to check against, so nothing is refused
        — the fare gate is still the backstop."""
        with patch("app.agent.nodes.planner.available_packages",
                   AsyncMock(return_value=[])):
            result = await self._plan(
                f"{UNSOLD_HOURS} hours from Shimla",
                ExtractedEntities(visitor_type=RENTAL, last_asked_field="rental_hours"),
            )

        assert result["unbookable_rental_hours"] is None
        assert result["entities"].rental_hours == UNSOLD_HOURS

    def test_the_wording_offers_the_alternatives_in_both_languages(self):
        labels = ["4 hrs / 40 km", "8 hrs / 80 km"]
        for lang in ("en", "hi"):
            text = slots.rental_hours_question(labels, lang, rejected=UNSOLD_HOURS)
            assert str(UNSOLD_HOURS) in text
            for label in labels:
                assert label in text

    def test_the_reply_names_the_duration_it_is_refusing(self):
        from app.agent.nodes.writer import _build_rental_section

        section = _build_rental_section(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(visitor_type=RENTAL, origin="Shimla"),
            unbookable_rental_hours=5,
            rental_packages=[{"hours": 4, "included_km": 40, "label": "4 hrs / 40 km"},
                             {"hours": 8, "included_km": 80, "label": "8 hrs / 80 km"}],
        ))

        assert "5 hours" in section
        assert "NOT one of the packages" in section
        assert "4 hrs / 40 km" in section

    def test_no_booking_screen_opens_on_a_package_that_does_not_exist(self):
        """Handing off to a booking screen for an unsellable duration is worse than not
        handing off at all — the rider gets there and it fails."""
        from app.agent.nodes.writer import _build_booking_handoff

        assert _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(visitor_type=RENTAL, origin="Shimla"),
            unbookable_rental_hours=5,
            pickup={"name": "Shimla", "lat": 31.1, "lon": 77.17},
        )) is None

    @pytest.mark.asyncio
    async def test_a_priced_duration_leaves_nothing_in_flight(self):
        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        fare = MagicMock()
        fare.get_quote = AsyncMock(return_value=FareQuote(
            base_fare_inr=2400, category_name="Sedan", trip_type="RENTAL",
        ))
        weather = MagicMock()
        weather.get_weather = AsyncMock(side_effect=Exception("not under test"))
        weather.snowfall_from_report = MagicMock(return_value=None)

        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token",
            entities=ExtractedEntities(
                visitor_type=RENTAL, origin="Shimla", rental_hours=8
            ),
        )
        from app.core.config import settings
        with patch("app.agent.nodes.tool_executor.get_rental_service", return_value=rental), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
             patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather), \
             patch.object(settings, "enable_offers", False):
            result = await tool_executor_node(state)

        assert "last_asked_field" not in result
        assert result["cab_fare"]["fare_inr"] == 2400


# ── 8. The booking screen opens on a real pin ─────────────────────────────────

class TestTheHandoffCarriesAPickupPin:

    async def _run(self, state):
        from app.core.config import settings

        rental = RentalService()
        rental._get = AsyncMock(return_value=RATE_CARD)
        fare = MagicMock()
        fare.get_quote = AsyncMock(return_value=FareQuote(
            base_fare_inr=2400, category_name="Sedan", trip_type="RENTAL",
        ))
        weather = MagicMock()
        weather.get_weather = AsyncMock(side_effect=Exception("not under test"))
        weather.snowfall_from_report = MagicMock(return_value=None)

        with patch("app.agent.nodes.tool_executor.get_rental_service", return_value=rental), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
             patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather), \
             patch.object(settings, "enable_offers", False):
            return await tool_executor_node(state)

    def _state(self, **kwargs) -> AgentState:
        base = {
            "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
            "rider_jwt": "rider-token",
            "entities": ExtractedEntities(
                visitor_type=RENTAL, origin="Shimla", rental_hours=8, group_size=4
            ),
        }
        base.update(kwargs)
        return AgentState(**base)

    @pytest.mark.asyncio
    async def test_the_named_pickup_is_geocoded(self):
        """A hire by the hour has no drop, so there is no journey to carry the pin —
        but the booking screen still has to open its map somewhere."""
        result = await self._run(self._state())

        assert result["pickup"]["name"]
        assert result["pickup"]["lat"] == pytest.approx(31.1048, abs=0.05)
        assert result["pickup"]["lon"] == pytest.approx(77.1734, abs=0.05)

    @pytest.mark.asyncio
    async def test_the_device_fix_beats_the_city_centre(self):
        """The fix is where the cab actually collects them; the city name is a label
        for it. Same precedence a journey's origin uses."""
        from app.domain.location import RiderLocation

        state = self._state(rider_location=RiderLocation(
            lat=31.0985, lon=77.2675, in_himachal=True,
            place_name="Shimla", state="Himachal Pradesh",
        ))
        result = await self._run(state)

        assert (result["pickup"]["lat"], result["pickup"]["lon"]) == (31.0985, 77.2675)

    @pytest.mark.asyncio
    async def test_a_fix_somewhere_else_is_not_the_pickup(self):
        """They said Delhi and their phone says Shimla — booking a pickup from your own
        sofa is a real thing people do, and their words win."""
        from app.domain.location import RiderLocation

        state = self._state(rider_location=RiderLocation(
            lat=31.1048, lon=77.1734, in_himachal=True,
            place_name="Shimla", state="Himachal Pradesh",
        ))
        state.entities.origin = "Delhi"
        result = await self._run(state)

        assert result["pickup"]["lat"] == pytest.approx(28.6139, abs=0.05)

    def test_the_handoff_uses_the_pin_when_there_is_one(self):
        from app.agent.nodes.writer import _build_booking_handoff

        handoff = _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Shimla", rental_hours=8),
            pickup={"name": "Shimla", "lat": 31.1048, "lon": 77.1734},
        ))

        assert handoff["origin"] == {"name": "Shimla", "lat": 31.1048, "lon": 77.1734}
        assert handoff["destination"] is None

    def test_an_unplaceable_pickup_still_hands_off_by_name(self):
        """A geocoder outage should cost the map its pin, not the rider their booking
        screen — the app can geocode a name itself."""
        from app.agent.nodes.writer import _build_booking_handoff

        handoff = _build_booking_handoff(AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            entities=ExtractedEntities(origin="Jibhi", rental_hours=8),
            pickup=None,
        ))

        assert handoff["origin"] == {"name": "Jibhi"}


# ── 9. The other flows are untouched ──────────────────────────────────────────

class TestARentalStaysARentalWhateverThePhoneSays:
    """The worst bug this flow has had, and the one a rider reports as "it hallucinates".

    The Rent a Car screen sends a GPS fix, and `apply_location` used to overwrite
    `entities.visitor_type` with what the fix implied — "local" inside Himachal,
    "outstation" outside. From that moment the rental was gone: `needs_for` fell back to
    the trip checklist and asked where they wanted to GO and for how many DAYS, and
    `response_mode` produced a cab quote or a day-by-day itinerary. Every other guard in
    the system (`resolve_persona`, `classify_visitor_type`, `needs_for`,
    `response_mode`) reads the value that line had already destroyed.
    """

    def _session(self):
        from app.domain.conversation import TripSession

        session = TripSession(user_id="u", persona=RENTAL)
        session.entities.visitor_type = RENTAL
        return session

    @pytest.mark.parametrize(
        "in_himachal,place", [(False, "Delhi"), (True, "Shimla")]
    )
    def test_a_fix_anywhere_leaves_the_rental_intact(self, in_himachal, place):
        from app.domain.location import RiderLocation
        from app.services.chat_session import apply_location

        session = self._session()
        apply_location(session, RiderLocation(
            lat=31.1, lon=77.17, in_himachal=in_himachal,
            place_name=place, place_distance_km=1.0, state="X",
        ))

        assert session.entities.visitor_type == RENTAL
        # …and the fix does NOT answer the pickup question. It used to: `origin` was
        # set to `place` here, so the rider was never asked where the car should
        # collect them. See TestTheRentalAlwaysAsksWhereToCollectThem below.
        assert session.entities.origin is None

    def test_the_questions_and_the_reply_survive_the_fix(self):
        """The two decisions that were being made off the clobbered value."""
        from app.domain.location import RiderLocation
        from app.services.chat_session import apply_location

        session = self._session()
        apply_location(session, RiderLocation(
            lat=28.6, lon=77.2, in_himachal=False, place_name="Delhi",
            place_distance_km=1.0, state="Delhi",
        ))
        persona = session.entities.visitor_type

        needs = slots.needs_for("plan_trip", persona)
        assert "destination" not in needs and "duration_days" not in needs
        assert slots.response_mode("plan_trip", persona) == slots.RENTAL_MODE

    def test_the_other_personas_still_follow_their_coordinates(self):
        """A visitor who has arrived in Himachal SHOULD become a local — that behaviour
        is why the line exists, and it is untouched."""
        from app.domain.conversation import TripSession
        from app.domain.location import RiderLocation
        from app.services.chat_session import apply_location

        session = TripSession(user_id="u", persona="outstation")
        session.entities.visitor_type = "outstation"
        apply_location(session, RiderLocation(
            lat=31.1, lon=77.17, in_himachal=True, place_name="Shimla",
            place_distance_km=1.0, state="Himachal Pradesh",
        ))

        assert session.entities.visitor_type == "local"


# ── 9b. The pickup is ASKED FOR, never assumed from the phone ─────────────────

class TestTheRentalAlwaysAsksWhereToCollectThem:
    """Reported: "the system is not asking for the pickup location".

    The Rent a Car screen sends a GPS fix. `apply_location` wrote the reverse-geocoded
    town into `entities.origin` for every persona, so by the time the slot policy ran
    the pickup looked ANSWERED and the question was skipped — the rider went straight
    from the duration to the date, and the fare was priced from whatever settlement
    their phone sat nearest.

    That is defensible for a trip, where `origin` means "which city are you travelling
    from" and a fix genuinely answers it. It is wrong for a rental, where `origin` is
    the address a driver is sent to. `is_precise` only promises the fix is within 15 km
    of the named town, and somebody hiring a car from an airport, a hotel or an office
    is not at the town centre.
    """

    def _rental_session(self):
        from app.domain.conversation import TripSession

        session = TripSession(user_id="u", persona=RENTAL)
        session.entities.visitor_type = RENTAL
        return session

    def _fix(self, **overrides):
        from app.domain.location import RiderLocation

        return RiderLocation(**{
            "lat": 31.1048, "lon": 77.1734, "in_himachal": True,
            "place_name": "Shimla", "place_distance_km": 3.2,
            "state": "Himachal Pradesh", **overrides,
        })

    def test_a_precise_fix_does_not_fill_the_pickup(self):
        from app.services.chat_session import apply_location

        session = self._rental_session()
        apply_location(session, self._fix())

        assert session.entities.origin is None
        assert session.entities.home_city is None
        # The coordinates are still kept — they are the map pin and the weather lookup.
        assert session.rider_location is not None

    def test_the_pickup_question_is_actually_asked(self):
        from app.services.chat_session import apply_location

        session = self._rental_session()
        apply_location(session, self._fix())
        session.entities.rental_hours = 8

        decision = slots.decide(
            session.entities, "rental_query", RENTAL, "en", max_attempts=2
        )
        assert decision.ask_field == SlotName.ORIGIN
        assert "pick you up" in decision.question

    @pytest.mark.parametrize("persona", ["local", "outstation"])
    def test_a_trip_still_takes_its_origin_from_the_fix(self, persona):
        """The convenience this removes from rentals is untouched everywhere else —
        for a journey, the starting CITY is exactly what a fix is good for."""
        from app.domain.conversation import TripSession
        from app.services.chat_session import apply_location

        session = TripSession(user_id="u", persona=persona)
        session.entities.visitor_type = persona
        apply_location(session, self._fix())

        assert session.entities.origin == "Shimla"


class TestHereMeansTheDeviceFix:
    """Once the pickup is always asked for, "here" is a reply riders actually give.

    It used to parse as a PLACE: `parse_place("here")` returned "Here", which became
    the pickup, was geocoded to something somewhere, and priced. Exactly the failure
    `_NOT_A_PLACE` already guards for "cancel" and "rent a car".
    """

    @pytest.mark.parametrize(
        "reply", ["here", "Here", "from here", "yahin", "yahin se", "my location",
                  "current location", "where i am"],
    )
    def test_deixis_never_becomes_a_place(self, reply):
        assert slots.parse_place(reply) is None
        assert slots.means_current_location(reply) is True

    @pytest.mark.parametrize(
        "reply", ["Shimla", "shimla mall road", "Kasol", "Manali bus stand"],
    )
    def test_a_real_pickup_is_still_read_normally(self, reply):
        assert slots.parse_place(reply) is not None
        assert slots.means_current_location(reply) is False

    def test_here_in_a_named_town_is_not_treated_as_deixis(self):
        """"here in Shimla" names a place. It must not be swallowed as "the fix"."""
        assert slots.means_current_location("here in shimla") is False


# ── 11. A whole sentence, read in code as well as by the model ────────────────
#
# "when the user enters in sentences the system hallucinates and gives irrelevant
# responses" — reported from the app. A sentence used to be understood ONLY by the
# extraction LLM: a field it missed was silently lost, so the assistant asked for
# something it had just been told, or answered around the gap. These patterns are
# anchored on unmistakable markers, so the same sentence is read the same way every
# time, and is still read when Vertex is unavailable.

class TestASentenceIsReadInCode:

    def test_the_whole_booking_out_of_one_sentence(self):
        found = slots.read_rental_details(
            "I need a car for 8 hours from Shimla tomorrow at 9 am, 4 of us, luxury"
        )

        assert found == {
            "rental_hours": 8,
            "origin": "Shimla",
            "travel_dates": "tomorrow",
            "pickup_time": "09:00",
            "group_size": 4,
            "cab_category": "luxury",
        }

    def test_the_same_sentence_in_hinglish(self):
        found = slots.read_rental_details(
            "Kal subah 6 baje Manali se gaadi chahiye 5 ghante ke liye, hum 3 hain"
        )

        assert found["rental_hours"] == 5
        assert found["pickup_time"] == "06:00"
        assert found["travel_dates"].lower() == "kal"
        assert found["origin"] == "Manali"
        assert found["group_size"] == 3

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("car for 3 hrs", 3),
            ("8hrs please", 8),
            ("5 ghante ke liye", 5),
            ("I want it for the full day", 8),
            ("half day rental", 4),
            # A trip length is not a rental duration, and neither is a clock time.
            ("3 days in Manali", None),
            ("pick me up at 9 am", None),
        ],
    )
    def test_only_a_duration_with_its_unit_is_read_as_hours(self, text, expected):
        assert slots.read_rental_details(text).get("rental_hours") == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("car at 9 am from Shimla", "09:00"),
            ("18:30 pickup", "18:30"),
            ("shaam 5 baje", "17:00"),
            ("6 baje chahiye", "06:00"),
            # No marker, no reading: the fare is calculated from this number, so a bare
            # "at 9" is left to the model or to the question rather than guessed at.
            ("tomorrow at 9", None),
            ("for 8 hours", None),
        ],
    )
    def test_a_clock_time_needs_an_unmistakable_marker(self, text, expected):
        assert slots.read_rental_details(text).get("pickup_time") == expected

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("from Shimla", "Shimla"),
            ("pick me up in Kasol", "Kasol"),
            ("Mandi se gaadi chahiye", "Mandi"),
            ("from Mall Road Shimla", "Mall Road Shimla"),
            # The stop words earn their place: without them the "place" swallows the
            # rest of the sentence and the geocoder fails on it.
            ("from Shimla tomorrow morning", "Shimla"),
            # Not places, however they are phrased.
            ("from 9 am", None),
            ("I want to rent a car", None),
            ("from tomorrow", None),
        ],
    )
    def test_a_pickup_is_only_read_from_an_explicit_marker(self, text, expected):
        assert slots.read_rental_details(text).get("origin") == expected

    def test_a_date_word_inside_an_ordinary_sentence_is_not_a_date(self):
        """`_TIME_WORDS` contains "start", "end" and "mid" because they help elsewhere.
        Reading them here filed "start from Shimla" as the travel date."""
        found = slots.read_rental_details("I want to start from Shimla for 8 hours")

        assert "travel_dates" not in found
        assert found["origin"] == "Shimla"
        assert found["rental_hours"] == 8

    @pytest.mark.parametrize(
        "text,expected",
        [("on 15 Jan", "15 Jan"), ("next monday", "next monday"),
         ("this weekend", "this weekend"), ("day after tomorrow", "day after tomorrow"),
         ("on 12/03", "12/03"), ("agle hafte", "agle hafte")],
    )
    def test_a_day_is_read_whole(self, text, expected):
        assert slots.read_rental_details(text).get("travel_dates") == expected

    @pytest.mark.parametrize(
        "answered,stored",
        [
            # Reported from the app: this reached the booking screen's Date field, and
            # the reply, word for word.
            ("I am planning to travel today", "today"),
            ("I want to go next monday please", "next monday"),
            ("this weekend if possible", "this weekend"),
            ("kal subah", "kal"),
            ("on 15 Jan", "15 Jan"),
            # Nothing recognisable to reduce it to, so their own words stand.
            ("sometime after Diwali", "sometime after Diwali"),
        ],
    )
    def test_a_stated_day_is_reduced_to_the_date_itself(self, answered, stored):
        assert slots.tidy_when(answered) == stored

    @pytest.mark.asyncio
    async def test_the_booking_screen_gets_a_date_not_a_sentence(self):
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[AIMessage(content="When do you need it?"),
                      HumanMessage(content="I am planning to travel today")],
            entities=ExtractedEntities(visitor_type=RENTAL, origin="Delhi",
                                       rental_hours=3, last_asked_field="travel_dates"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction(travel_dates="I am planning to travel today")
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        assert result["entities"].travel_dates == sanitize.today_ist().isoformat()

    def test_nothing_is_invented_from_a_sentence_with_no_details(self):
        assert slots.read_rental_details("I need a car please") == {}
        assert slots.read_rental_details("") == {}

    @pytest.mark.asyncio
    async def test_the_planner_reads_the_sentence_when_the_model_misses_it(self):
        """The model returning nothing is the case that used to lose the whole message.
        Everything the rider stated is still collected, so the next turn quotes instead
        of asking for details it was already given."""
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[
                AIMessage(content=welcome_message(RENTAL)),
                HumanMessage(content="8 hours from Shimla tomorrow at 9 am, 4 of us"),
            ],
            entities=ExtractedEntities(visitor_type=RENTAL, last_asked_field="rental_hours"),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction()
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        entities = result["entities"]
        assert (entities.rental_hours, entities.origin) == (8, "Shimla")
        assert (entities.pickup_time, entities.group_size) == ("09:00", 4)
        assert entities.travel_dates == _days_from_today(1)
        # Nothing left to ask, so the turn goes on to price it.
        assert result["needs_clarification"] is False

    @pytest.mark.asyncio
    async def test_the_planner_reads_the_sentence_when_vertex_is_down(self):
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[HumanMessage(content="I need a car for 4 hours from Manali kal")],
            entities=ExtractedEntities(visitor_type=RENTAL),
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service",
                   side_effect=RuntimeError("vertex down")):
            result = await planner_node(state)

        assert result["entities"].rental_hours == 4
        assert result["entities"].origin == "Manali"

    @pytest.mark.asyncio
    async def test_the_model_still_leads_and_is_never_overwritten(self):
        """Code fills GAPS. A rider correcting themselves ("make it 4, not 8") is the
        model's job to read, and this must not undo it."""
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[HumanMessage(content="actually make it 4 hours not 8 hours")],
            entities=ExtractedEntities(visitor_type=RENTAL),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction(rental_hours=4)
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        assert result["entities"].rental_hours == 4

    @pytest.mark.asyncio
    async def test_a_place_mentioned_in_passing_is_never_a_destination(self):
        """"A car for 8 hours around Manali" comes back from the extractor with
        destination="Manali" every time. Left in place it reached the writer's facts
        block, the itinerary shortcut and the rider's past-trip list — a trip nobody
        asked for, assembled out of a hire."""
        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            messages=[HumanMessage(content="a car for 8 hours around Manali")],
            entities=ExtractedEntities(visitor_type=RENTAL),
        )
        llm = MagicMock()
        llm.with_structured_output.return_value.ainvoke = AsyncMock(
            return_value=PlannerExtraction(
                rental_hours=8, destination="Manali", duration_days=2
            )
        )
        with patch("app.agent.nodes.planner.get_vertex_ai_service") as svc:
            svc.return_value.get_flash_model.return_value = llm
            result = await planner_node(state)

        assert result["entities"].destination is None
        assert result["entities"].duration_days is None
        assert result["entities"].rental_hours == 8

    def test_the_prompt_tells_the_extractor_a_rental_is_not_a_trip(self):
        from app.agent.nodes.planner import PLANNER_SYSTEM_PROMPT

        assert "A RENTAL CONVERSATION" in PLANNER_SYSTEM_PROMPT
        assert "destination and duration_days are ALWAYS null here" in PLANNER_SYSTEM_PROMPT


# ── 12. Which cab: the app's "Select Rental Cab" list, as a conversation ──────
#
# The Rent a Car screen will not book until the rider has picked a cab, so neither will
# the assistant. The quoting turn lists what was priced and asks; the answer is matched
# against the categories the backend actually returned; and the price that is confirmed,
# the coupon that is checked and the category the booking screen opens on are all the
# rider's own choice rather than our recommendation.

class TestTheRiderPicksTheCab:

    FARE: ClassVar[dict] = {
        "category_name": "Economy", "fare_inr": 2100, "rental_hours": 8,
        "included_km": 80, "distance_km": 80,
        "options": [
            {"category_name": "Economy", "category_id": 1,
             "total_fare_inr": 2100, "seat_capacity": 4},
            {"category_name": "Luxury", "category_id": 3,
             "total_fare_inr": 3400, "seat_capacity": 6},
            {"category_name": "Him Cab Mini Bus", "category_id": 5,
             "total_fare_inr": 5200, "seat_capacity": 20},
        ],
    }

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("luxury", "luxury"), ("Luxury", "luxury"),
            ("I'll take the luxury one", "luxury"),
            ("luxary please", "luxury"),                    # a typo costs no turn
            ("mini bus", "mini bus"), ("him cab mini bus", "mini bus"),
            ("economy", "economy"), ("premium car", "premium"),
            # Not a choice: a question, a place, a duration, a non-answer.
            ("which cars do you have?", None), ("manali", None),
            ("8 hours", None), ("yes", None), ("", None),
        ],
    )
    def test_a_cab_choice_is_read_from_whatever_they_type(self, reply, expected):
        assert slots.parse_cab_category(reply) == expected

    def test_mini_bus_is_never_read_as_mini(self):
        """20 seats versus 4. The tier list is matched longest-first for this reason."""
        assert slots.parse_cab_category("mini bus") == "mini bus"
        assert slots.parse_cab_category("mini") == "mini"

    @pytest.mark.parametrize(
        "text", ["8 hours from the bus stand", "pick me up at the bus stop",
                 "I'm a solo traveller", "we need it for a day"],
    )
    def test_an_address_is_never_read_as_a_cab(self, text):
        """A bare "bus" or "traveller" in an address or a description would put a
        20-seater on the booking screen. Riders who want one say "mini bus"."""
        assert slots.parse_cab_category(text) is None
        assert "cab_category" not in slots.read_rental_details(text)

    def test_the_chosen_cab_becomes_the_price_we_confirm(self):
        from app.agent.nodes.tool_executor import _select_requested_category

        fare = dict(self.FARE)
        assert _select_requested_category(fare, "luxury") is True
        assert fare["fare_inr"] == 3400
        assert fare["category_id"] == 3
        assert fare["category_name"] == "Luxury"
        assert fare["seat_capacity"] == 6
        assert fare["selected_by_rider"] is True

    def test_a_cab_that_was_not_priced_is_not_substituted(self):
        from app.agent.nodes.tool_executor import _select_requested_category

        fare = dict(self.FARE)
        assert _select_requested_category(fare, "premium") is False
        # Untouched: still our recommendation, and flagged as such by omission.
        assert fare["fare_inr"] == 2100
        assert "selected_by_rider" not in fare

    async def _run(self, entities, rate_card=RATE_CARD, fare_fails=False):
        rental = RentalService()
        rental._get = AsyncMock(return_value=rate_card)
        fare = MagicMock()
        if fare_fails:
            fare.get_quote = AsyncMock(side_effect=RuntimeError("fare service down"))
        else:
            fare.get_quote = AsyncMock(return_value=FareQuote(
                base_fare_inr=2100, category_id=1, category_name="Economy",
                trip_type="RENTAL",
                alternatives=[
                    FareCategoryFare(category_id=1, category_name="Economy",
                                     total_fare_inr=2100, seat_capacity=4),
                    FareCategoryFare(category_id=3, category_name="Luxury",
                                     total_fare_inr=3400, seat_capacity=6),
                ],
            ))
        weather = MagicMock()
        weather.get_weather = AsyncMock(side_effect=Exception("not under test"))
        weather.snowfall_from_report = MagicMock(return_value=None)

        state = AgentState(
            session_id="s", user_id="u", intent="rental_query", visitor_type=RENTAL,
            rider_jwt="rider-token", entities=entities,
        )
        with patch("app.agent.nodes.tool_executor.get_rental_service", return_value=rental), \
             patch("app.agent.nodes.tool_executor.get_fare_service", return_value=fare), \
             patch("app.agent.nodes.tool_executor.get_weather_service", return_value=weather), \
             patch.object(settings, "enable_offers", False):
            return await tool_executor_node(state)

    def _entities(self, **kwargs) -> ExtractedEntities:
        base = {"visitor_type": RENTAL, "origin": "Shimla", "rental_hours": 8}
        base.update(kwargs)
        return ExtractedEntities(**base)

    @pytest.mark.asyncio
    async def test_the_quoting_turn_asks_which_cab_and_holds_the_question(self):
        """The app will not book without a cab, so the reply asks — and the slot is put
        in flight, which is what makes a one-word "luxury" next turn an answer."""
        result = await self._run(self._entities())

        assert result["awaiting_cab_choice"] is True
        assert result["last_asked_field"] == "cab_category"
        assert result["entities"].last_asked_field == "cab_category"
        # The prices are still quoted — they are what the rider is choosing between.
        assert result["cab_fare"]["fare_inr"] == 2100

    @pytest.mark.asyncio
    async def test_choosing_a_cab_settles_the_turn(self):
        result = await self._run(self._entities(cab_category="luxury"))

        assert result["cab_fare"]["category_name"] == "Luxury"
        assert result["cab_fare"]["fare_inr"] == 3400
        assert result["cab_fare"]["selected_by_rider"] is True
        assert "awaiting_cab_choice" not in result
        assert "last_asked_field" not in result

    @pytest.mark.asyncio
    async def test_a_cab_this_package_does_not_have_is_re_asked(self):
        result = await self._run(self._entities(cab_category="mini bus"))

        assert result["unavailable_cab_category"] == "mini bus"
        assert result["awaiting_cab_choice"] is True
        assert result["last_asked_field"] == "cab_category"
        # Cleared, so their next answer is read as a fresh choice.
        assert result["entities"].cab_category is None
        # The duration stands — only the cab was wrong.
        assert result["entities"].rental_hours == 8

    @pytest.mark.asyncio
    async def test_the_question_is_bounded_like_every_other(self):
        """A rider who ignores it still reaches a booking screen: past the attempt cap
        the assistant confirms the cab it recommended instead of asking again."""
        result = await self._run(self._entities(
            asked_counts={"cab_category": settings.max_clarification_attempts}
        ))

        assert "awaiting_cab_choice" not in result
        assert "last_asked_field" not in result

    @pytest.mark.asyncio
    async def test_no_priced_options_means_no_cab_question(self):
        """Nothing to choose between. The booking screen is where they pick, and the
        reply says the price is confirmed there."""
        result = await self._run(self._entities(), fare_fails=True)

        assert result["cab_fare"] is None
        assert "awaiting_cab_choice" not in result

    # ── The reply and the event read the same gate ─────────────────────────────

    def _state(self, **kwargs) -> AgentState:
        base = {
            "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
            "entities": self._entities(),
            "rental_packages": [{"hours": 8, "included_km": 80, "label": "8 hrs / 80 km",
                              "from_rate_card": True}],
            "cab_fare": self.FARE,
        }
        base.update(kwargs)
        return AgentState(**base)

    def test_no_booking_screen_while_still_asking_which_cab(self):
        from app.agent.nodes.writer import _build_booking_handoff, _build_system_prompt

        state = self._state(awaiting_cab_choice=True)

        assert _build_booking_handoff(state) is None
        prompt = _build_system_prompt(state)
        assert "ask which of the cabs above they would like" in prompt
        assert "booking screen is open" not in prompt
        assert "has NOT chosen a cab yet" in prompt

    def test_the_chosen_cab_is_what_the_booking_screen_opens_on(self):
        from app.agent.nodes.writer import _build_booking_handoff

        chosen = {**self.FARE, "category_name": "Luxury", "category_id": 3,
                  "fare_inr": 3400, "selected_by_rider": True}
        handoff = _build_booking_handoff(self._state(
            entities=self._entities(cab_category="luxury"), cab_fare=chosen
        ))

        assert handoff["recommended_category_name"] == "Luxury"
        assert handoff["recommended_category_id"] == 3
        assert handoff["fare_inr"] == 3400
        assert handoff["category_selected_by_rider"] is True

    def test_the_confirming_reply_names_one_cab_not_three(self):
        from app.agent.nodes.writer import _build_system_prompt

        chosen = {**self.FARE, "category_name": "Luxury", "fare_inr": 3400,
                  "selected_by_rider": True}
        prompt = _build_system_prompt(self._state(cab_fare=chosen))

        assert "The rider chose this cab: Luxury" in prompt
        assert "THE cab they chose" in prompt
        assert "Do not list the other cabs" in prompt
        assert "booking screen is open with their package, their cab" in prompt

    def test_an_unavailable_cab_is_explained_rather_than_swapped(self):
        from app.agent.nodes.writer import _build_system_prompt

        prompt = _build_system_prompt(self._state(
            entities=self._entities(cab_category=None),
            unavailable_cab_category="premium",
            awaiting_cab_choice=True,
        ))

        assert "asked for a premium cab, which is NOT one of the cabs" in prompt
        assert "Do NOT quote a price for a premium" in prompt


# ── 13. A rental prompt contains no trip ──────────────────────────────────────

class TestTheRentalPromptHasNoTripInIt:

    def _prompt(self, **kwargs) -> str:
        from app.agent.nodes.writer import _build_system_prompt

        base = {
            "session_id": "s", "user_id": "u", "intent": "rental_query", "visitor_type": RENTAL,
            "entities": ExtractedEntities(visitor_type=RENTAL, origin="Shimla",
                                       rental_hours=8, travel_dates="kal",
                                       pickup_time="09:00"),
            "rental_packages": [{"hours": 8, "included_km": 80, "label": "8 hrs / 80 km",
                              "from_rate_card": True}],
        }
        base.update(kwargs)
        return _build_system_prompt(AgentState(**base))

    def test_the_facts_block_is_the_booking_not_a_trip(self):
        prompt = self._prompt()

        assert "## THIS RENTAL" in prompt
        assert "## TRIP" not in prompt
        assert "- Pickup: Shimla" in prompt
        assert "- Date: kal" in prompt
        assert "- Pickup time: 09:00" in prompt
        assert "There is NO destination and NO trip length" in prompt

    def test_no_destination_line_survives_into_a_rental(self):
        """This block used to open with "Destination: not stated — answer generally about
        Himachal Pradesh", which is an invitation to write a travel brochure in a reply
        whose whole job is a car, a duration and a price."""
        prompt = self._prompt()

        assert "answer generally about Himachal Pradesh" not in prompt
        assert "- Destination:" not in prompt
        assert "- Duration:" not in prompt and "- Budget:" not in prompt
        assert "- Interests:" not in prompt

    def test_a_destination_the_extractor_picked_up_is_still_not_named(self):
        """Belt and braces: the planner drops it, and if one ever arrives anyway the
        prompt does not repeat it back."""
        prompt = self._prompt(
            entities=ExtractedEntities(visitor_type=RENTAL, origin="Shimla",
                                       rental_hours=8, destination="Kasol"),
        )

        assert "Kasol" not in prompt

    def test_a_package_with_no_known_km_is_never_given_one(self):
        """When the rate card is unreadable the price still arrives (the backend matches
        the chart itself) but the included kilometres do not. Asking for them anyway is
        how a reply states a distance nobody sent us."""
        prompt = self._prompt(
            cab_fare={"category_name": "Economy", "fare_inr": 900, "rental_hours": 3,
                      "included_km": None, "options": [
                          {"category_name": "Economy", "total_fare_inr": 900,
                           "seat_capacity": 4}]},
            rental_packages=[{"hours": 4, "included_km": None, "label": "4 hrs",
                              "from_rate_card": False}],
        )

        assert "Say NOTHING about included kilometres" in prompt
        assert "the kilometres it includes" not in prompt

    def test_a_known_km_figure_is_still_required(self):
        prompt = self._prompt(
            cab_fare={"category_name": "Economy", "fare_inr": 2100, "rental_hours": 8,
                      "included_km": 80, "options": [
                          {"category_name": "Economy", "total_fare_inr": 2100,
                           "seat_capacity": 4}]},
        )

        assert "the kilometres it includes" in prompt
        assert "Say NOTHING about included kilometres" not in prompt

    def test_the_weather_is_labelled_with_the_pickup(self):
        """`_rental_tools` fetches weather for the PICKUP — a rental has no destination
        to fetch it for. Labelling it as a destination attached a real temperature to a
        place the rider never named."""
        prompt = self._prompt(
            weather_data={"current_temp_c": 17.5, "condition": "clear sky",
                          "forecast_7day": [], "alerts": []},
        )

        assert "Current Weather at Shimla" in prompt


class TestOtherFlowsAreUnaffected:

    @pytest.mark.parametrize(
        "intent,persona,mode",
        [
            ("plan_trip", "outstation", slots.ITINERARY_MODE),
            ("plan_trip", "local", slots.RIDE_MODE),
            ("weather_query", "local", slots.ANSWER_MODE),
            ("general_query", "outstation", slots.ANSWER_MODE),
        ],
    )
    def test_existing_modes_are_unchanged(self, intent, persona, mode):
        assert slots.response_mode(intent, persona) == mode

    def test_a_trip_still_requires_a_destination(self):
        assert slots.needs_for("plan_trip")["destination"] is slots.Need.REQUIRED
