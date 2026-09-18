"""
app/agent/nodes/writer.py
──────────────────────────
Writer Node — generates the final response using Gemini Flash with streaming.
Responsibilities:
  1. Build a rich, context-aware system prompt with all tool data
  2. Generate a natural Hinglish/English response
  3. Generate a structured JSON itinerary (Itinerary model)
  4. Emit structured SSE events for the mobile app (itinerary, alerts, cab fare)
  5. Stream tokens back to the SSE handler

Three response shapes, chosen by `slots.response_mode(intent, visitor_type)`:

  • ITINERARY — a day-by-day plan for a visitor with days to fill
  • RIDE      — a cab recommendation for a Himachal resident: vehicle, fare, drive
                time, weather, then a handoff to the app's booking screen. They know
                the place; what they are choosing is the ride.
  • RENTAL    — a car and driver by the hour: the package, the cab options and the
                price. No route and no destination, because hiring by the hour means
                the rider has not chosen one.
  • ANSWER    — a short direct answer to a question about somewhere
  • CANCELLED — the rider called it off. Handled by `cancel_node` before this node is
                reached; the guard at the top of `writer_node` is what makes sure a
                cancellation can never be answered with any of the four above.

Live data always beats generated data. Whatever the model writes, the real cab fare,
the real weather report and the measured journey are stamped onto the validated
itinerary afterwards (`_apply_real_fare`, `_apply_weather`, `_apply_travel_info`) — so
the app never renders a price, a forecast or a distance the model made up, or an empty
one it forgot to copy across. Coupons follow the same rule in reverse: the reply may
quote only the codes the backend returned (`_build_offers_block`), and when there are
none it is forbidden to mention discounts at all.

This is the most visible node — its output IS the user experience.
"""

import asyncio
import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent import slots
from app.agent.guardrails import apply_guardrails
from app.agent.language import detect_reply_language
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import TripType
from app.core.logging import get_logger
from app.domain.conversation import ExtractedEntities
from app.domain.itinerary import (
    ActivityType,
    BudgetRange,
    CabPackage,
    CostBreakdown,
    GroupType,
    Itinerary,
    TravelInfo,
    WeatherForecast,
)
from app.integrations.himcab.quote_builder import classify_trip_type
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

# Maximum retry attempts for structured output extraction
_MAX_RETRIES = 2


def _infer_group_type(entities: ExtractedEntities) -> GroupType:
    """Resolve the trip's group type from what the planner actually extracted.

    `group_type` is a property of the travelling party (solo/couple/family/…), NOT of
    the front end the chat was opened from. Resolving it here — and injecting the
    answer into the prompt — is what stops the persona ("local"/"outstation") being
    echoed back into this field by the LLM.
    """
    if entities.group_type:
        return GroupType(entities.group_type)
    size = entities.group_size
    if size:
        if size <= 1:
            return GroupType.SOLO
        if size == 2:
            return GroupType.COUPLE
    return GroupType.FAMILY


def _infer_budget_range(entities: ExtractedEntities) -> BudgetRange:
    """The user's stated budget, defaulting to moderate when they never said."""
    return BudgetRange(entities.budget_range) if entities.budget_range else BudgetRange.MODERATE


def _resolved_duration(entities: ExtractedEntities) -> int:
    """The trip length to plan for — the user's, or the configured default.

    A default is only ever reached after the planner has given up asking (the field
    is then listed in `state.assumed_fields`), and the prompt requires the reply to
    say so out loud. See `_build_assumptions_section`.
    """
    return entities.duration_days or settings.default_trip_duration_days


def _normalize_itinerary_dict(itinerary_dict: dict, state: AgentState) -> dict:
    """Overwrite the LLM's trip-level facts with the ones the planner confirmed.

    The conversation is the source of truth for destination/duration/group/budget —
    the writer LLM only fills in the day plans. This also repairs the two fields that
    used to sink validation: a persona in `group_type` and a null `budget_range`.
    """
    entities = state.entities
    normalized = dict(itinerary_dict)

    if entities.destination:
        normalized["destination"] = entities.destination
    # Always pinned, so the JSON can't disagree with the day plans in the text — the
    # resolved value is the user's when they gave one and the disclosed default otherwise.
    normalized["duration_days"] = _resolved_duration(entities)

    # Only override when the LLM gave us something that isn't a valid group type.
    llm_group = normalized.get("group_type")
    if not isinstance(llm_group, str) or llm_group.strip().lower() not in {g.value for g in GroupType}:
        normalized["group_type"] = _infer_group_type(entities).value

    llm_budget = normalized.get("budget_range")
    if not isinstance(llm_budget, str) or llm_budget.strip().lower() not in {b.value for b in BudgetRange}:
        normalized["budget_range"] = _infer_budget_range(entities).value

    normalized.setdefault("language", str(getattr(state.language, "value", state.language)))
    return normalized


def _build_fare_block(state: AgentState) -> str:
    """The HIM CAB FARES section of the prompt — the whole price story in one place.

    Fares come from the backend per cab category, priced on the route distance. When
    that distance was computed rather than surveyed the fare is an estimate, and the
    prompt says so explicitly rather than presenting it as a firm quote. No coupons or
    discounts appear here — those live in the app's own booking flow.
    """
    cf = state.cab_fare
    options = (cf or {}).get("options") or []

    if not options:
        return (
            "### HIM CAB FARES\n"
            "No fare is available for this route right now — do NOT state, estimate or "
            "imply any price. Say the exact fare will be confirmed when they book, and "
            "still recommend travelling with Him Cab.\n"
        )

    headline = cf.get("category_name")
    distance_km = cf.get("distance_km")
    # The headline is the rider's own pick once they have made one, so it must not be
    # labelled as our suggestion — a reply that "recommends" the cab they just chose
    # reads as not having heard them.
    chosen = bool(cf.get("selected_by_rider"))
    headline_flag = (
        "  (THE CAB THEY CHOSE — confirm this one)" if chosen
        else "  (recommended for this group/budget)"
    )

    lines = []
    for o in options:
        fare = o.get("total_fare_inr")
        if fare is None:
            continue
        name = o.get("category_name") or "Cab"
        seats = o.get("seat_capacity")
        seat_str = f", {seats}-seater" if seats else ""
        flag = headline_flag if name == headline else ""
        lines.append(f"  - {name}: Rs {round(fare):,}{seat_str}{flag}")

    # For a rental the distance is the package's INCLUDED kilometres, not a drive
    # between two towns — calling it a drive would misdescribe what the price covers.
    if cf.get("rental_hours"):
        route = f" for the {cf['rental_hours']}-hour package"
        if distance_km:
            route += f" ({round(distance_km)} km included)"
    else:
        route = f" for the {round(distance_km)} km drive" if distance_km else ""
    header = (
        f"### HIM CAB FARES (live prices — use these EXACT figures)\n"
        f"Him Cab fares{route}, quoted by the backend for the measured road\n"
        f"distance. Do NOT round, adjust, average or invent a number, and do NOT\n"
        f"present a range.\n"
    )
    return header + "\n".join(lines) + "\n"


def _build_offers_block(state: AgentState) -> str:
    """The HIM CAB OFFERS section — real coupons, or an instruction to stay silent.

    The reply used to be told never to mention a discount at all, which was right when
    the only coupons available were ones the model made up. Now that offers come from
    the backend (`app/integrations/himcab/offers_service.py`), staying silent means a rider
    pays full price with a valid coupon sitting in their account.

    The rule is therefore narrower, not looser: quote these codes exactly, and nothing
    else. `offers` is already filtered to this trip type, this fare and today's date,
    so anything here is a coupon the rider can genuinely apply.
    """
    offers = state.offers or []
    if not offers:
        return (
            "\n### HIM CAB OFFERS\n"
            "No coupon applies to this trip. Do NOT mention, invent or hint at any "
            "discount, promo or coupon code.\n"
        )

    lines = []
    for offer in offers:
        code = offer.get("coupon_code")
        value = offer.get("coupon_value_inr")
        after = offer.get("fare_after_offer_inr")
        name = offer.get("coupon_name") or "Him Cab offer"
        saving = f"Rs {round(value):,} off" if value else "a discount"
        landed = f", bringing it to Rs {round(after):,}" if after else ""
        minimum = offer.get("minimum_value_inr")
        min_note = f" (on fares over Rs {round(minimum):,})" if minimum else ""
        lines.append(f"  - {code}: {name} — {saving}{landed}{min_note}")

    # Eligibility can only be CHECKED against a fare. Without one — the quote failed —
    # a coupon with a minimum spend is live on the account but not proven to apply to
    # this trip, and the prompt has to say the weaker thing.
    all_verified = all(o.get("applies_to_this_fare") for o in offers)
    applicability = (
        "These coupons apply to THIS trip."
        if all_verified
        else "These coupons are live on this rider's account. No fare came back for this "
             "route, so do NOT claim they apply to this trip — say the discount is "
             "applied if the fare qualifies."
    )

    return (
        "\n### HIM CAB OFFERS (real, live on this rider's account)\n"
        f"{applicability} Mention the best one in a single short line, with its exact "
        "code and exact rupee saving as written below. Quote ONLY these codes — never "
        "invent, adapt or recall a coupon from anywhere else — and say the coupon is "
        "applied on the booking screen. Never state a total that is not written here.\n"
        + "\n".join(lines)
        + "\n"
    )


def _apply_real_fare(itinerary: Itinerary, cab_fare: dict) -> None:
    """Force the real Him Cab fare onto the validated itinerary (mutates in place).

    The LLM's numbers are advisory when a real fare exists — this guarantees the cab cost
    and total match the backend fare exactly. The only discount applied is a real coupon
    from the backend (`best_offer`), subtracted the same way the Spring booking flow
    subtracts it, so the total here matches the one the rider will be charged.
    """
    fare = cab_fare.get("fare_inr") or 0
    vehicle = cab_fare.get("vehicle_label")
    seats = cab_fare.get("seat_capacity")
    best_offer = cab_fare.get("best_offer") or {}
    after_offer = cab_fare.get("fare_after_offer_inr")
    discount = max(0.0, fare - after_offer) if after_offer is not None else 0.0

    # cab_package — create if the LLM omitted it
    if itinerary.cab_package is None:
        itinerary.cab_package = CabPackage(
            package_type="Him Cab Package",
            vehicle=vehicle or "Cab",
            estimated_cost_inr=fare,
        )
    cp = itinerary.cab_package
    cp.estimated_cost_inr = fare
    cp.base_fare_inr = fare
    cp.discounted_fare_inr = after_offer if discount else None
    cp.coupon_code = best_offer.get("coupon_code") if discount else None
    cp.fare_source = cab_fare.get("source", "himcab_api")
    if seats is not None:
        cp.seat_capacity = seats
    if vehicle:
        cp.vehicle = vehicle

    # cost_breakdown — rebuild deterministically, preserving any non-cab estimates.
    existing = itinerary.cost_breakdown
    itinerary.cost_breakdown = CostBreakdown.compute(
        cab=fare,
        hotels=existing.hotels_inr if existing else 0.0,
        food=existing.food_estimate_inr if existing else 0.0,
        entry_fees=existing.entry_fees_inr if existing else 0.0,
        misc=existing.miscellaneous_inr if existing else 0.0,
        discount=discount,
    )


def _apply_travel_info(itinerary: Itinerary, state: AgentState) -> None:
    """Stamp the measured journey onto the itinerary's travel_info (mutates in place).

    Same rule as the fare and the weather: live numbers come from the service, never
    from the model. `travel_info` used to be built by the LLM with both figures pinned
    to null, because at the time nothing measured them — so an itinerary that clearly
    knew the journey shipped with `estimated_distance_km: null` and the app had nothing
    to render on its route line.
    """
    route = state.route_estimate
    if not route:
        return

    info = itinerary.travel_info or TravelInfo()
    # Endpoints AND figures both come from the route. Keeping the model's endpoints
    # while stamping our numbers over them would let the JSON pair one journey's
    # cities with another journey's distance — a route line that is wrong in a way
    # nothing downstream can detect.
    info.origin = route["origin"]["name"]
    info.destination = route["destination"]["name"]
    info.estimated_distance_km = route["distance_km"]
    info.estimated_duration_hours = round(route["duration_minutes"] / 60, 1)
    itinerary.travel_info = info


def _apply_weather(itinerary: Itinerary, state: AgentState) -> None:
    """Fill the itinerary's weather from the live report (mutates in place).

    `Itinerary.weather` used to depend entirely on the model copying the forecast into
    its JSON block, which it frequently skipped — so the app rendered an itinerary with
    no weather even though the data had been fetched. Building it here from
    `state.weather_data` makes it deterministic, and matches how the cab fare is handled
    (`_apply_real_fare`): live numbers come from the service, never from the model.
    """
    w = state.weather_data
    if not w:
        return

    forecast = w.get("forecast_7day") or []
    trip_days = forecast[: max(1, itinerary.duration_days)] if forecast else []
    lows = [d["min_c"] for d in trip_days if d.get("min_c") is not None]
    highs = [d["max_c"] for d in trip_days if d.get("max_c") is not None]

    condition = w.get("condition")
    temp = w.get("current_temp_c")
    summary = f"Currently {round(temp)}°C, {condition}" if temp is not None else (condition or "")
    if lows and highs:
        summary += f". Expect {round(min(lows))}–{round(max(highs))}°C during the trip"

    # Weather alerts are TravelAlert objects serialised by model_dump(mode="json").
    alerts = w.get("alerts") or []
    first = alerts[0] if isinstance(alerts, list) and alerts else (
        alerts if isinstance(alerts, dict) else None
    )
    snow = state.snowfall_data or {}

    itinerary.weather = WeatherForecast(
        summary=summary,
        temperature_min_c=min(lows) if lows else None,
        temperature_max_c=max(highs) if highs else None,
        condition=condition,
        alert=(first or {}).get("description") or (snow.get("message") if snow.get("risk") in ("high", "critical") else None),
        alert_severity=(first or {}).get("severity") or (snow.get("risk") if snow.get("risk") in ("high", "critical") else None),
    )


def _has_current_weather(state: AgentState) -> bool:
    """True when there is an actual temperature to quote.

    Deliberately stricter than `bool(state.weather_data)`: a report can arrive with a
    null temperature, and a truthy-dict check then promises the writer a number that
    is not there — which is how a reply ends up describing what it does not have.
    """
    return (state.weather_data or {}).get("current_temp_c") is not None


def _rental_duration_chosen(state: AgentState) -> bool:
    """True when the rider has settled on a duration this turn can act on.

    The first gate for a rental turn. `tool_executor` clears the slot when the duration
    is not one of the offered packages, so an unsellable "6 hours" lands here as False.
    """
    return bool(state.entities.rental_hours)


def _rental_cab_chosen(state: AgentState) -> bool:
    """True when the price in front of the rider is for the cab THEY picked.

    Set by `tool_executor._select_requested_category`, which only sets it after finding
    the tier among the categories the backend actually priced — so this is never true on
    the rider's word alone.
    """
    return bool((state.cab_fare or {}).get("selected_by_rider"))


def _rental_ready_to_book(state: AgentState) -> bool:
    """True when this turn may hand the rider to the booking screen.

    THE gate for a rental handoff, and deliberately the same condition the prompt
    branches on — read by both so the reply can never announce a booking screen while
    the text above it is still asking which cab to open it on. The app's Rent a Car
    screen will not book without a cab selected, so neither will we.

    A turn with no priced options is ready as soon as the duration is settled: there is
    no list to choose from, the reply says the price is confirmed at booking, and the
    screen is where the rider picks their cab.
    """
    if not _rental_duration_chosen(state):
        return False
    return not state.awaiting_cab_choice


def _build_rental_section(state: AgentState) -> str:
    """The rental packages and what the rider has chosen, or how to ask.

    The packages come from the rate card, and the reply may offer NO others: the
    backend matches a duration against a chart exactly (`rentHours == maxHour`), so
    inventing a "6 hour" option produces a request that returns no fare and a rider who
    was quoted for something Him Cab does not sell.
    """
    packages = state.rental_packages
    hours = state.entities.rental_hours
    lines = ["\n### RENTAL"]

    if packages:
        offered = ", ".join(p["label"] for p in packages)
        lines.append(f"- Packages Him Cab offers: {offered}")
        lines.append(
            "- Offer ONLY these. Never invent a duration, and never imply one can be "
            "split or combined."
        )
    else:
        lines.append(
            "- The package list is unavailable right now, so do NOT name any duration "
            "or included distance."
        )

    # A duration the rate card does not sell. The slot has already been cleared so the
    # rider's next answer is read as a fresh choice — but the reply still has to say
    # what they asked for and that it is not sold, or the re-ask looks like the
    # assistant simply forgot.
    rejected = state.unbookable_rental_hours
    if rejected:
        lines.append(
            f"- The rider asked for {rejected} hours, which is NOT one of the packages. "
            f"Say so plainly, name the ones above, and ask them to pick one. Do NOT "
            f"quote a price for {rejected} hours."
        )
    elif hours:
        chosen = next((p for p in packages if p["hours"] == hours), None)
        if chosen:
            lines.append(f"- The rider chose: {chosen['label']}")
        elif packages:
            lines.append(
                f"- The rider asked for {hours} hours, which is NOT one of the packages. "
                f"Say so plainly, name the ones above, and ask them to pick one. Do NOT "
                f"quote a price for {hours} hours."
            )
        else:
            lines.append(f"- The rider asked for {hours} hours.")
    else:
        lines.append("- No duration chosen yet.")

    # The pickup clock time, when they gave one. Stated in the reply because it is what
    # the fare was calculated on — the rider should be able to see the figure their
    # price depends on, and spot it if we heard them wrong.
    if state.entities.pickup_time:
        when = state.entities.travel_dates
        lines.append(
            f"- Pickup time: {state.entities.pickup_time}"
            + (f" on {when}" if when else "")
            + ". Say it back to them exactly; the fare is calculated from this time."
        )
    elif state.entities.travel_dates:
        lines.append(f"- Pickup day: {state.entities.travel_dates}.")

    # ── Which cab ─────────────────────────────────────────────────────────────
    # The app's "Select Rental Cab" list, as a conversation. Three situations, and they
    # are different replies: a cab we could not find, a cab they chose, and no choice
    # yet. Merging them is how a reply ends up confirming one cab in prose while every
    # price in it belongs to another.
    unavailable = state.unavailable_cab_category
    if unavailable:
        lines.append(
            f"- The rider asked for a {unavailable} cab, which is NOT one of the cabs "
            f"available on this package. Say so plainly, name the cabs listed under HIM "
            f"CAB FARES, and ask which of THOSE they would like. Do NOT quote a price "
            f"for a {unavailable}."
        )
    elif _rental_cab_chosen(state):
        chosen_cab = (state.cab_fare or {}).get("category_name") or "the cab they named"
        lines.append(
            f"- The rider chose this cab: {chosen_cab}. Confirm THAT cab and its price. "
            f"Do not re-list the others and do not recommend a different one."
        )
    elif state.awaiting_cab_choice:
        lines.append(
            "- The rider has NOT chosen a cab yet, and the booking cannot be completed "
            "until they do. List the priced cabs and ask which one they want."
        )
    return "\n".join(lines) + "\n"


def _build_journey_section(state: AgentState) -> str:
    """Distance and drive time for the route, or nothing when it isn't known.

    Surveyed routes carry measured figures; the rest are derived from coordinates and
    are labelled, because a drive time quoted to the minute implies knowledge of the
    road that this app does not have.
    """
    route = state.route_estimate
    if not route:
        return ""

    return (
        f"\n### THIS JOURNEY\n"
        f"- {route['origin']['name']} to {route['destination']['name']}\n"
        f"- Drive time: about {route['duration_text']}\n"
        f"- Distance: about {round(route['distance_km'])} km\n"
        f"Use these exact figures. Do NOT add traffic, road or weather-delay claims — "
        f"nothing here observes them.\n"
    )


def _with_change_note(state: AgentState, text: str) -> str:
    """`text`, preceded by the line saying which detail the rider just changed.

    Everything in the reply below it is rebuilt from the entities each turn, so it
    already SHOWS the new destination or the new date. What it has no way to say is
    that the value moved, which is the one thing a rider who just asked for a change is
    looking for. Set by the planner (see `modification.applied_message`), so the
    wording is a table rather than something the model is asked to remember to mention.
    """
    note = state.change_note
    return f"{note}\n\n{text}" if note else text


def _build_booking_handoff(state: AgentState) -> Optional[dict]:
    """The payload the app needs to open its booking screen on this journey.

    Both endpoints carry coordinates so the map can drop pins without geocoding the
    names again — re-resolving "Kasol" in the app could easily land somewhere other
    than the point this fare was priced on.

    `trip_type` is the rate card the fare was quoted on, decided by the journey's
    length rather than by the persona that was talking (see
    `quote_builder.classify_trip_type`). The app reads it to keep a short hop out of
    its outstation booking screen, which refuses anything inside one state and under
    200 km — so a visitor asking about two Manali suburbs is not sent to a "Confirm
    Route" button that cannot be pressed.

    None whenever the journey is unknown: with no route there is no pickup, no drop
    and nothing for the booking screen to open on. That is the whole gate — an
    outstation visitor who never named a starting city simply gets no card.

    This is a handoff, not a booking: nothing here reserves a cab or takes payment.
    The app owns that flow, which is why the reply is told to announce the screen
    rather than ask the rider to confirm anything.
    """
    route = state.route_estimate
    fare = state.cab_fare or {}
    # A rental has no route — the booking screen opens on a pickup and a duration.
    is_rental = slots.response_mode(state.intent, state.visitor_type) == slots.RENTAL_MODE
    if not route and not is_rental:
        return None
    # Same gate the rental prompt reads, so text and event agree by construction: no
    # duration and no cab chosen means the reply is still a question, and a question
    # must not ship a button that opens a screen it has not finished filling in.
    if is_rental and not _rental_ready_to_book(state):
        return None
    # Prefill the coupon only when eligibility was actually checked against a fare.
    # Preselecting one we could not verify puts a code on the booking screen that the
    # payment step may then reject — worse than leaving the field empty.
    best_offer = next(
        (o for o in (state.offers or []) if o.get("applies_to_this_fare")), {}
    )
    if is_rental:
        pickup_name = state.entities.origin or state.entities.home_city
        # Coordinates when the pickup could be placed, so the map opens on the point we
        # priced; the bare name when it could not, so the app can geocode it itself.
        origin_pin = state.pickup or {"name": pickup_name}
        return {
            "trip_type": TripType.RENTAL,
            "origin": origin_pin,
            # A rental has no drop — the rider decides where to go once the car is theirs.
            "destination": None,
            "rental_hours": state.entities.rental_hours,
            "included_km": fare.get("included_km"),
            # True when `recommended_category_*` is the rider's own pick rather than our
            # recommendation. The app's Rent a Car screen should open with that cab
            # already selected; anything else re-asks a question they have answered.
            "category_selected_by_rider": bool(fare.get("selected_by_rider")),
            "travel_dates": state.entities.travel_dates,
            # The clock time the quote was priced on, so the booking screen opens on the
            # same one. Sending the day without the hour would let the rider land on a
            # default time and be charged a rate they were never shown.
            "pickup_time": state.entities.pickup_time,
            "passengers": state.entities.group_size,
            "recommended_category_id": fare.get("category_id"),
            "recommended_category_name": fare.get("category_name"),
            "fare_inr": fare.get("fare_inr"),
            "offer_id": best_offer.get("offer_id"),
            "coupon_code": best_offer.get("coupon_code"),
            "fare_after_offer_inr": best_offer.get("fare_after_offer_inr"),
            "options": fare.get("options", []),
        }

    # Unreachable: the guard above returns when there is no route and this is not a
    # rental, and the rental path has just returned. Spelled out so the reads below are
    # provably safe rather than safe-by-inspection.
    if route is None:
        return None

    return {
        "origin": route["origin"],
        "destination": route["destination"],
        "distance_km": route["distance_km"],
        "duration_minutes": route["duration_minutes"],
        "duration_text": route["duration_text"],
        # From the fare when there is one, and from the JOURNEY when there is not.
        # It is decided by distance either way (`quote_builder.classify_trip_type`), so
        # a missing fare is no reason to leave it null — and null is expensive here: the
        # app reads this to choose which booking screen to open, and its outstation
        # screen refuses anything inside one state and under 200 km. A 514 km Delhi
        # to Kasol run that arrives without a trip_type opens the wrong form.
        "trip_type": fare.get("trip_type") or classify_trip_type(route["distance_km"]),
        "passengers": state.entities.group_size,
        "travel_dates": state.entities.travel_dates,
        # The cab we put in front of them, so the booking screen opens on the same one.
        "recommended_category_id": fare.get("category_id"),
        "recommended_category_name": fare.get("category_name"),
        "fare_inr": fare.get("fare_inr"),
        # The coupon the reply just told them about, so the booking screen can prefill
        # it. `offerId` is exactly what POST /trip/book-ride takes.
        "offer_id": best_offer.get("offer_id"),
        "coupon_code": best_offer.get("coupon_code"),
        "fare_after_offer_inr": fare.get("fare_after_offer_inr"),
        "options": fare.get("options", []),
    }


def _build_assumptions_section(state: AgentState) -> str:
    """Tell the model exactly what the user never said, and what to do about it.

    `state.assumed_fields` is the planner's list of details it stopped asking for
    (see app/agent/slots.py). Every one of them is a fact the reply would otherwise
    invent — the reported "it assumes things and generates an itinerary" failure. So
    each gets an explicit instruction: substitute a stated default, or leave it out.
    Nothing here may be presented to the user as something they told us.
    """
    if not state.assumed_fields:
        return ""

    # What to do per field. A value means "use it and say so"; None means "there is
    # no safe substitute — work around the gap instead of filling it".
    substitutes: dict[str, str | None] = {
        "duration_days": f"a {_resolved_duration(state.entities)}-day trip",
        "origin": None,
        "group_size": None,
        "travel_dates": None,
        "return_journey_preference": None,
    }
    guidance: dict[str, str] = {
        "origin": (
            "Do NOT invent a starting city. Plan the destination side only and do not "
            "describe the journey in, or quote any travel time from, a city they never named."
        ),
        "group_size": (
            "Do NOT state a number of travellers. Keep suggestions group-neutral."
        ),
        "travel_dates": (
            "Do NOT name specific calendar dates. Say the plan works for any time they "
            "choose and keep seasonal notes general."
        ),
        "return_journey_preference": (
            "Do NOT plan or assume a return journey. Mention it as an option only."
        ),
    }

    lines = []
    for field in state.assumed_fields:
        label = slots.label_for(field, "en")
        substitute = substitutes.get(field)
        if substitute:
            lines.append(f"- {label}: never stated. Plan for {substitute}.")
        else:
            lines.append(f"- {label}: never stated. {guidance.get(field, 'Leave it out.')}")

    return (
        "\n### DETAILS THE USER NEVER GAVE (do not invent these)\n"
        + "\n".join(lines)
        + "\nSTART your reply with ONE short sentence naming what you assumed and inviting a "
        "correction. Never write or imply that the user told you any of the above.\n"
    )


def _build_followup_instruction(state: AgentState, reply_language: str) -> str:
    """The one question the reply closes with — always about booking, never about the trip.

    Every detail that shapes a plan is collected before the plan is written (see the
    slot policy in app/agent/slots.py). So closing with "how many days?" or "how many
    of you are travelling?" would mean the itinerary above it was written without the
    answer — the reported bug. Anything the user declined to give is raised in the
    assumptions line at the TOP of the reply instead, where a correction still helps.
    """
    destination = state.entities.destination or "their destination"
    extra = (
        " You may also ask whether they want the return cab included."
        if not state.entities.return_journey_preference
        else ""
    )
    # A known route means a booking_handoff ships with this reply (see
    # `_build_booking_handoff`), so the app is drawing a Book Now button underneath it
    # with both ends already filled in. Saying so is the difference between a button
    # the rider trusts and one they wonder what it will ask them for.
    handoff_note = (
        " In the same line, tell them their pickup and drop are already filled in on "
        "the booking screen, and do not ask them to enter either again."
        if state.route_estimate is not None
        else ""
    )
    return (
        f"End with ONE short, warm question that moves toward a booking — which cab "
        f"category they prefer, or whether to book their Him Cab to {destination} now."
        f"{extra}{handoff_note} Do NOT ask for trip details (dates, number of days, "
        f"group size, starting city) here: you already have them, or you have stated "
        f"your assumption at the top. Write it in {reply_language}."
    )


def _build_system_prompt(state: AgentState) -> str:
    """
    Build the full system prompt for the Writer node,
    injecting all tool results as structured context.

    Three shapes, chosen by `slots.response_mode(intent, visitor_type)`: a day-by-day
    plan for a visitor, a cab recommendation for a Himachal resident, and a short
    direct answer for a question about a place. Answering "what's the weather in
    Manali?" with a four-day itinerary is what made the agent feel like it never
    listened; telling a Kullu resident what to see in Manali is the same mistake in a
    different costume.
    """
    entities = state.entities
    mode = slots.response_mode(state.intent, state.visitor_type)
    is_local = state.visitor_type == "local"
    is_outstation = state.visitor_type == "outstation"

    # Never interpolate a bare None into the prompt: an unknown destination used to
    # crash this function outright (`None.upper()`), and an unknown duration reached
    # the model as the literal text "None days".
    destination = entities.destination or ""
    destination_display = destination or "not stated — answer generally about Himachal Pradesh"
    duration_days = _resolved_duration(entities)

    # ── Language: ONE language for the whole reply, mirroring the user ─────────
    # Shared with the Planner (app/agent/language.py) so clarifications and itineraries
    # never disagree — the root cause of replies flipping between Hindi and English.
    _lang_key, reply_language = detect_reply_language(state)
    language_instruction = (
        f"Write your ENTIRE reply in {reply_language} only. Never switch or mix languages "
        f"or scripts within the reply — every line, heading, and the follow-up question must "
        f"be in {reply_language}. Do not use emojis."
    )

    # How the pickup point is NAMED comes from the resolved route, which is the one
    # place that knows how much the fix can be trusted: it says "Shimla" for a fix in
    # Shimla and "near Mandi" for one 40 km out. Re-deriving the name here from the
    # raw fix is how the prompt ended up asserting "the traveller starts from Mandi"
    # to somebody who had never been there.
    origin_display = (
        (state.route_estimate or {}).get("origin", {}).get("name")
        or entities.origin
        or entities.home_city
        or "not specified"
    )

    weather_section = ""
    if state.weather_data:
        w = state.weather_data
        # A rental's weather is fetched for the PICKUP (`_rental_tools`) — there is no
        # destination to fetch it for. Labelling it with a destination invited the model
        # to write about somewhere the rider never named, and to attach a real
        # temperature to it.
        weather_place = (
            origin_display if mode == slots.RENTAL_MODE else destination_display
        )
        weather_section = f"""
**Current Weather at {weather_place}:**
- Temperature: {w.get('current_temp_c')}°C (feels like {w.get('feels_like_c')}°C)
- Condition: {w.get('condition')}
- Humidity: {w.get('humidity_percent')}%
- Wind: {w.get('wind_speed_kmh')} km/h
- Upcoming Forecast: {json.dumps(w.get('forecast_7day', [])[:3], indent=2)}
"""
        if w.get("alerts"):
            weather_section += f"\nWEATHER ALERT: {w['alerts']}"

    snowfall_section = ""
    if state.snowfall_data:
        s = state.snowfall_data
        snowfall_section = f"\n**Snowfall Risk:** {s.get('risk', 'none').upper()} — {s.get('message', '')}"

    # ── Travel from the origin ────────────────────────────────────────────────
    # Rendered only when BOTH ends of the journey are known, and deliberately silent
    # about distance and time when THIS JOURNEY (`_build_journey_section`) has them —
    # this section used to assert "no verified travel time or distance" while that one
    # gave both, and a prompt that contradicts itself gets whichever half the model
    # happened to read.
    route_known = state.route_estimate is not None
    travel_section = ""
    # Itinerary mode ONLY. This block tells the model to open Day 1 with the arrival
    # journey and to allow for travel fatigue — right for a visitor planning days, wrong
    # for a resident booking a cab. In RIDE_MODE it was the instruction that produced
    # "you can enjoy light sightseeing when you arrive" in a reply that was meant to be
    # four facts and a booking link.
    if destination and mode == slots.ITINERARY_MODE and (entities.origin or route_known):
        journey_from = origin_display
        travel_section = (
            f"\n**TRAVEL FROM ORIGIN — {journey_from.upper()} to {destination.upper()}:**\n"
            + (
                "- Use the distance and drive time from THIS JOURNEY above; state no others.\n"
                if route_known
                else "- No verified travel time or distance for this route — do NOT state one.\n"
            )
            + f"IMPORTANT: the traveller starts from {journey_from}, so Day 1 must begin with "
            "that journey. Account for travel fatigue — lighter activities on arrival day.\n"
        )

    # Itinerary and answer modes only. The profile summary carries "Has visited: Manali,
    # Kasol recently", "Frequent destinations: …", "Interests: trekking" and "eligible
    # for loyalty discounts". That is real material for somebody choosing what to do
    # with four days — and noise in the two replies that are a transaction.
    #
    # A rental never had it. A RIDE did, and should not: a travel history invites the
    # reply to bring up a place the rider never mentioned ("since you enjoyed Kasol…")
    # or a discount no coupon backs, in a message whose whole job is a cab and a price.
    user_context = ""
    if state.user_context_summary and mode not in (slots.RENTAL_MODE, slots.RIDE_MODE):
        user_context = f"\n**User Profile:** {state.user_context_summary}"

    # ── Outstation-specific instructions ──────────────────────────────────────
    outstation_instructions = ""
    if is_outstation and entities.origin and mode == slots.ITINERARY_MODE:
        outstation_instructions = f"""
**OUTSTATION VISITOR INSTRUCTIONS (CRITICAL):**
- This visitor is coming from **{entities.origin}** (OUTSIDE Himachal Pradesh)
- **Day 1 MUST begin with the arrival journey** from {entities.origin} to {destination_display}
- Account for travel fatigue — plan lighter activities on arrival day
- Return journey preference: **{entities.return_journey_preference or 'not specified'}**
  - If "same_cab_return" or "booked": Last day MUST include the return journey from {destination_display} back to {entities.origin}. Plan departure timing accordingly.
  - If "own_arrangement": Do NOT include return journey details. Last day should focus on sightseeing/departure prep only.
  - If not specified: Briefly mention return options but don't plan it in detail.
- Mention the recommended travel mode (cab, flight+cab, train+cab) based on distance
- If the distance is >500km, suggest flight+cab or train+cab as alternatives

"""

    # ── Local visitor-specific instructions ────────────────────────────────────
    # Itinerary mode only. A local in RIDE_MODE is booking a cab, not being sold
    # hidden gems in a valley they live next to.
    local_instructions = ""
    if is_local and mode == slots.ITINERARY_MODE:
        home_city = entities.home_city or entities.origin or "their HP home city"
        local_instructions = f"""
**LOCAL VISITOR INSTRUCTIONS (IMPORTANT):**
- This visitor is a **Himachal Pradesh resident** based in **{home_city}**
- They likely already know the popular tourist spots — **prioritise hidden gems and off-beat experiences**
- Travel time is shorter (intra-HP routes are typically 1-6 hours by cab)
- **No need to account for long-distance travel fatigue** — they can start activities sooner
- Do NOT ask about or plan return journey — locals manage their own return
- For 1-2 day trips, use a **weekend getaway format**: compact, action-packed itinerary
- **Highlight seasonal specialities** relevant to the current month (apple season, cherry blossoms, snowfall, monsoon greenery)
- Mention **local festivals or events** happening at the destination if any
- Suggest **insider tips** that a local would appreciate (best time to avoid crowds, lesser-known viewpoints, local dhabas)
- Include **local_tips** in each day's plan in the JSON output

"""

    # ── Him Cab fares (all categories, before any coupon) and the live coupons ─
    fare_block = _build_fare_block(state)
    offers_block = _build_offers_block(state)

    # ── travel_info JSON template (only when both ends of the journey are known) ──
    # The figures are given to the model rather than asked of it, and stamped over
    # afterwards by `_apply_travel_info` regardless. They stay null only when nothing
    # measured the route — the model must not fill that gap in.
    travel_info_json = ""
    if (entities.origin or state.route_estimate) and destination:
        route = state.route_estimate or {}
        json_origin = entities.origin or (route.get("origin") or {}).get("name") or ""
        distance_json = route.get("distance_km") if route else None
        hours_json = round(route["duration_minutes"] / 60, 1) if route else None
        travel_info_json = f"""
  "travel_info": {{
    "origin": "{json_origin}",
    "destination": "{destination}",
    "estimated_duration_hours": {hours_json if hours_json is not None else "null"},
    "estimated_distance_km": {distance_json if distance_json is not None else "null"},
    "recommended_mode": "cab"
  }},"""

    # ── Hidden gems JSON template (for local visitors) ────────────────────────
    local_json_fields = ""
    if is_local:
        local_json_fields = """
  "hidden_gems": ["...", "..."],"""

    # ── Enum vocabularies allowed in the JSON (derived from the models, not hardcoded) ──
    activity_types = ", ".join(f'"{t.value}"' for t in ActivityType)
    group_types = ", ".join(f'"{g.value}"' for g in GroupType)
    budget_ranges = ", ".join(f'"{b.value}"' for b in BudgetRange)

    # Resolved trip facts, injected so the model echoes them rather than inventing —
    # in particular it must never put the persona ("local"/"outstation") in group_type.
    resolved_group_type = _infer_group_type(entities).value
    resolved_budget_range = _infer_budget_range(entities).value

    # ── Shared honesty rules ──────────────────────────────────────────────────
    # First in the list on purpose: everything below is about being useful, and these
    # are about not making things up while being useful.
    language_step = (
        f"LANGUAGE: write the ENTIRE reply in {reply_language} — every line, heading, and the "
        f"closing question. No mixing languages/scripts, no emojis."
    )
    honesty_step = (
        "Use ONLY the facts given above. Never state a travel time, distance, price, date or "
        "live condition that does not appear here — say you can confirm it on booking instead."
        "\nAnd never ANNOUNCE that something is missing. A fact you were not given is simply "
        "not mentioned: no 'there is no information available about the current temperature', "
        "no 'the weather could not be retrieved', no 'details are unavailable'. Reporting our "
        "own gaps tells the rider nothing they can act on and reads as a broken assistant — "
        "leave the line out instead."
    )
    # Weather was being dropped from itineraries even when it had been fetched, so when
    # there is a report it is a required section rather than an optional aside.
    weather_step = (
        "Weather: include a short 'Weather' line near the top with the CURRENT temperature "
        "and condition from REAL-TIME DATA, and a packing or safety note if there is an "
        "alert or snowfall risk. Use only the numbers given."
        if state.weather_data
        # No loophole here: "unless it is relevant" used to sit on the end of this line,
        # and the model took it — replies came back announcing that no temperature was
        # available, which is our problem, not the rider's.
        else "Weather: no live report was available. Write NO weather line at all — no "
             "temperature, no condition, and no statement that any of it is unavailable."
    )
    assumptions_section = _build_assumptions_section(state)
    followup_step = _build_followup_instruction(state, reply_language)
    has_fares = bool((state.cab_fare or {}).get("options"))
    him_cab_step = (
        'Close with a "Book Your Him Cab" section (comfortable cabs, verified local drivers, '
        "safe hill driving). This section is REQUIRED — never drop it."
        + (
            " List EVERY cab category from the HIM CAB FARES section above as its own bullet "
            "with the exact price shown there, and recommend the flagged one. Follow the "
            "prices exactly as written."
            if has_fares
            else " No fare came back, so name NO cab and NO vehicle — not Innova, not "
                 "Tempo Traveller, not SUV, not a 'spacious' or 'larger' one, and not a "
                 "suggestion of what would suit the group size. Say only that Him Cab "
                 "will carry them and that the cab and the exact fare are confirmed on "
                 "the booking screen."
        )
    )
    # The old wording here forbade a PRICE and said to "still recommend travelling with
    # Him Cab" — which the model obliged by recommending vehicles instead: "for your
    # family of 8, we recommend a spacious Innova or a Tempo Traveller". Neither came
    # from the backend. A cab category is a claim exactly like a price is: it implies we
    # checked what is available and what it seats, and the rider plans around it. With
    # no fare we checked nothing, so there is nothing to name.
    # Coupons: quote the real ones, invent none. The blanket ban this replaces was
    # correct only while there was no source of truth for offers — now that the backend
    # supplies them, staying silent means the rider pays full price with a valid coupon
    # in their account. See `_build_offers_block`.
    has_offers = bool(state.offers)
    coupons_step = (
        "Mention the BEST coupon from HIM CAB OFFERS in one short line — its exact code "
        "and exact saving, and that it applies on the booking screen. Never quote a code "
        "that is not listed there, and never invent or adjust a discount."
        if has_offers
        # "no offer applies to this trip" described the situation, so the model wrote
        # it down: replies came back with a bullet reading "No coupon is applicable to
        # this trip." Silence about discounts means writing nothing, not announcing a
        # nothing — the rider loses a line of their reply to a fact worth zero to them.
        else "Write NOTHING about coupons, promos, offers or discounts — not a code, "
             "and not a line saying none applies. The subject does not come up."
    )

    if mode == slots.ANSWER_MODE:
        # The user asked a question. Answer that question — a day-by-day plan here reads
        # as not having listened, which is exactly what was reported.
        steps = [
            language_step,
            "Answer the user's actual question in the FIRST line, directly and specifically.",
            honesty_step,
            weather_step,
            "Keep it to 3-6 short bullets. Do NOT write a day-by-day itinerary and do NOT "
            "output any JSON.",
            coupons_step,
            "End with ONE short line offering to plan the full trip if they'd like.",
        ]
        json_block = ""

    elif mode == slots.RENTAL_MODE:
        # A car and a driver for a block of time. The rider is choosing a package and a
        # vehicle, so the reply is the package, the price and the pickup — no route, no
        # itinerary, and no destination, because hiring by the hour means they have not
        # decided one and need not.
        #
        # The two situations below are genuinely different replies, and merging them is
        # what produced the worst bug this flow has had: asked for 5 hours (which Him Cab
        # does not sell), the assistant described the 4-hour package as though it had
        # been chosen, and announced a booking screen that — correctly — never opened.
        # A turn with no bookable duration is a QUESTION; only a chosen one confirms.
        forbidden_step = (
            "FORBIDDEN — do not write any of these, in any wording: a day-by-day plan, "
            "sightseeing or things-to-do suggestions, a destination or drop-off point, "
            "a route, a distance between two towns, a drive time, hotel or food advice, "
            "or any JSON. They are hiring a car by the hour and have not said where "
            "they are going."
        )

        if not _rental_duration_chosen(state):
            rejected = state.unbookable_rental_hours
            steps = [
                language_step,
                honesty_step,
                (
                    f"Line 1: say plainly that {rejected} hours is not one of the rental "
                    f"packages Him Cab sells. State it once, without apologising at "
                    f"length, and never quote or estimate a price for {rejected} hours."
                    if rejected
                    else "Line 1: say in one sentence that you can arrange a car and "
                         "driver by the hour."
                ),
                "Line 2: list the packages from RENTAL below, exactly as written there "
                "and no others.",
                "Line 3: ask which one they would like. End on that question.",
                "Write NOTHING else. In particular: no price, no cab list, no weather, "
                "no coupon, and no mention of a booking screen — nothing has been "
                "chosen, so nothing is open and nothing is reserved. Do not describe "
                "any single package as though they had picked it.",
                forbidden_step,
                "Hard limit: 3 lines.",
            ]
        else:
            steps = [
                language_step,
                honesty_step,
                "Write to the rider as 'you', and as Him Cab as 'we'. Never refer to "
                "Him Cab in the third person — 'we'll have the car with you' reads like "
                "a person, 'Him Cab is ready to provide' reads like a form letter.",
                "Line 1: confirm what they are getting in one natural sentence — the "
                "package, the pickup and when. Do not describe WHO is travelling "
                "('your family', 'your group of 4') unless the party is stated below; "
                "a headcount nobody gave you is still an invention.",
                "Then EXACTLY these bullets, in this order, and NOTHING else. Each is "
                "ONE plain sentence at the SAME level — no bold labels like "
                "'**Package:**', no headings, and no indented sub-lists.",
                # The included kilometres are only known when the rate card was
                # readable. Asking for them regardless is how a reply ends up stating a
                # distance nobody sent us: an instruction the model cannot satisfy
                # honestly is one it will satisfy dishonestly.
                (
                    "  (a) The package: hours and the kilometres it includes, from "
                    "RENTAL below. Say plainly that distance beyond it is charged on "
                    "the meter at drop-off. If RENTAL gives a pickup time, state it in "
                    "this bullet exactly as written there — it is what the fare was "
                    "calculated on."
                    if (state.cab_fare or {}).get("included_km")
                    else "  (a) The package: the number of hours, from RENTAL below, "
                    "and the pickup time if RENTAL gives one (state it exactly as "
                    "written there — it is what the fare was calculated on). Say NOTHING "
                    "about included kilometres or a distance limit: no figure for this "
                    "package reached you, and the booking screen confirms it."
                ),
                (
                    # They picked one. Repeating the whole list here is how a reply
                    # ends up quoting three prices for a booking that has exactly one.
                    "  (b) THE cab they chose, from RENTAL below — vehicle, seats and "
                    "its exact price from HIM CAB FARES. One bullet, that cab only. Do "
                    "not list the other cabs and do not suggest a different one."
                    if _rental_cab_chosen(state)
                    else "  (b) One bullet PER CAB from HIM CAB FARES — vehicle, seats, "
                    "price — each as its own top-level bullet, never nested under a "
                    "heading. These are the cabs available on this package: list every "
                    "one of them and never mention a vehicle that is not listed."
                    if has_fares
                    else "  (b) No fare came back for this package, so say in one line "
                         "that the price is confirmed on the booking screen. Do NOT "
                         "state, estimate, range or imply any number."
                ),
                # Present ONLY when there is a reading. Asking for a conditional bullet
                # ("include it if available") invited the model to fill the slot by
                # narrating the absence: "There is no information available about the
                # current temperature at your pickup location." An instruction that
                # isn't there cannot be answered badly.
                *(
                    ["  (c) Current temperature and condition at the pickup, copied "
                     "from REAL-TIME DATA. One line."]
                    if _has_current_weather(state)
                    else ["Write NO weather bullet: there is no live reading for the "
                          "pickup. Do not mention temperature or conditions at all, and "
                          "do not say that they are unavailable."]
                ),
                coupons_step,
                forbidden_step,
                # The close is the one place the two halves of this branch differ, and
                # it reads the SAME gate the `booking_handoff` event does. Announcing an
                # open booking screen while still asking which cab is the failure this
                # pairing exists to make impossible.
                (
                    "Last line: state that the booking screen is open with their "
                    "package, their cab and their pickup already filled in. It must NOT "
                    "be a question — do not ask them to confirm, and do not ask for any "
                    "further detail."
                    if _rental_ready_to_book(state)
                    else "Last line: ask which of the cabs above they would like. End "
                         "on that question, and say NOTHING about a booking screen — "
                         "nothing opens until they have picked a cab."
                ),
                "Hard limit: at most 7 lines in total, including the bullets.",
            ]
        json_block = ""

    elif mode == slots.RIDE_MODE:
        # A Himachal resident booking a cab. They know the destination — what they are
        # deciding is which vehicle, what it costs and how long the drive is.
        #
        # This reply used to drift back into a travel brochure: bullets like "travelling
        # with family means comfort matters" and "you can enjoy light sightseeing when
        # you arrive", then a closing offer to plan their whole trip. Every one of those
        # is padding around the four facts they came for, so the steps below name the
        # bullets that are allowed and forbid the rest by exclusion.
        steps = [
            language_step,
            honesty_step,
            f"Line 1: confirm the journey — {origin_display} to {destination_display}. "
            f"One sentence, no adjectives.",
            "Then EXACTLY these bullets, in this order, and NOTHING else:",
            "  (a) Drive time and distance, copied from THIS JOURNEY above.",
            (
                "  (b) One bullet PER CAB from HIM CAB FARES — vehicle, seats, price. "
                "They are already narrowed to what seats this group: recommend all of "
                "them, and never mention a larger vehicle or one not listed."
                if has_fares
                else "  (b) No fare is available, so say in one line that the exact fare "
                     "is confirmed on the booking screen. Do NOT state, estimate, "
                     "range or imply any number."
            ),
            # Only when there is a forecast to copy. Pointing at a REAL-TIME DATA
            # section that is not in the prompt leaves the model with a step it cannot
            # satisfy, and it resolves that either by announcing the gap — the exact
            # thing step 2 forbids — or by supplying a temperature of its own.
            *(
                ["  (c) Current temperature and condition at the destination, copied "
                 "from REAL-TIME DATA. One line."]
                if _has_current_weather(state)
                else []
            ),
            coupons_step,
            "FORBIDDEN — do not write any of these, in any wording: a day-by-day plan, "
            "sightseeing or things-to-do suggestions, hidden gems, hotel or food "
            "advice, comfort/safety reassurance, travel tips, trip length, group "
            "composition, or any JSON. They live here. Every extra line pushes the "
            "price and the booking link further out of view.",
            "Last line: state that the booking screen is open with their pickup and "
            "drop already filled in. It must NOT be a question. Do not ask whether "
            "they want to book, do not ask them to confirm, do not offer to plan a "
            "trip or an itinerary, and do not ask for any further detail.",
            "Hard limit: at most 7 lines in total, including the bullets.",
        ]
        json_block = ""

    else:
        steps = [
            language_step,
            honesty_step,
            "Be concise: short bullet points, one line each. No long paragraphs or filler.",
            f"Give a day-by-day itinerary for all {duration_days} days (3-5 bullets per day: "
            "morning / afternoon / evening).",
            "Write each activity as ONE plain, human sentence. NEVER print the JSON's field "
            "names in the prose — no 'type:', 'place:', 'time:', 'description:' or "
            "'recommended_mode'/'Recommended Mode' labels, and no sub-bullets echoing them. "
            "Those exist ONLY inside the ```json block at the very end; in the text they are "
            "clutter the rider cannot use.",
            "Recommend places from REAL-TIME DATA when it lists any; otherwise use well-known "
            "spots at the destination and do not invent venue details. Add 1-2 food ideas per day.",
            weather_step,
        ]
        if is_outstation and entities.origin:
            steps.insert(3, f"Day 1 MUST begin with the arrival journey from {entities.origin}.")
        if is_local:
            steps.append(
                'LOCAL visitor: add a short "Hidden Gems" section (2-3 off-beat spots) and put '
                "local_tips in the JSON."
            )
        steps.extend([
            him_cab_step,
            coupons_step,
            followup_step,
            "After the text, append the itinerary as a ```json block (format below). Do NOT put "
            "any price, fare, cab_package, cost_breakdown, coupon, or offer fields in the JSON — "
            "the system attaches pricing separately.",
            f'Every activity "type" MUST be exactly one of: {activity_types} — never any other word.',
        ])
        json_block = f"""
## ITINERARY JSON (append at the very end, wrapped in ```json ... ```)
Copy destination, duration_days, group_type and budget_range EXACTLY as given below — they are
already resolved from the conversation. group_type describes the travelling party and must be one
of {group_types}; it is NOT the visitor type, so never write "local" or "outstation" there.
budget_range must be one of {budget_ranges}.
{{
  "destination": "{destination}",
  "duration_days": {duration_days},
  "group_type": "{resolved_group_type}",
  "budget_range": "{resolved_budget_range}",{travel_info_json}{local_json_fields}
  "days": [
    {{
      "day": 1,
      "theme": "Arrival & Settle In",
      "activities": [
        {{"time": "Morning/Afternoon", "place": "...", "type": "travel", "description": "..."}},
        {{"time": "Evening", "place": "...", "type": "sightseeing", "description": "..."}}
      ],
      "meal_suggestions": ["..."],
      "local_tips": ["..."]
    }}
  ],
  "weather": {{"summary": "...", "alert": null}}
}}

NOTE: every activity "type" must be exactly one of: {activity_types} — never any other word.
"""

    response_steps = "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))

    # Only facts the user actually gave are stated as facts; everything the planner
    # could not confirm is listed under assumptions instead of being filled in here.
    #
    # A rental gets its own list, and the omission is the point: this block used to
    # open with "Destination: not stated — answer generally about Himachal Pradesh",
    # which is an invitation to write about a place in a reply whose whole job is a car,
    # a duration and a price. Worse, when the extractor had picked a place up in passing
    # ("a car for 8 hours around Manali"), the line named it as the trip's destination.
    if mode == slots.RENTAL_MODE:
        trip_facts = [
            f"- Pickup: {origin_display}",
            f"- Date: {entities.travel_dates or 'not stated — do not name a date'}",
            f"- Pickup time: {entities.pickup_time or 'not stated — do not state a time'}",
            "- There is NO destination and NO trip length. They are hiring a car by the "
            "hour; where they go is their business and they need not have decided.",
        ]
        if entities.group_size:
            trip_facts.insert(1, f"- Party: {entities.group_size} people")
    elif mode == slots.RIDE_MODE:
        # A resident booking a cab between two places they already know. The reply is
        # four facts — the journey, the cabs, the price, the booking screen — and the
        # steps below forbid group composition, trip length and anything sightseeing-
        # shaped outright.
        #
        # So those facts are not given here either. Stating them and then forbidding
        # their use is a prompt that contradicts itself, and the model resolves that
        # by using them: "travelling with family means comfort matters", "you can
        # enjoy light sightseeing when you arrive". Group size still reaches the
        # reply — through HIM CAB FARES, which is already narrowed to cabs that seat
        # the party, which is the only form of it a ride actually needs.
        trip_facts = [
            f"- Destination: {destination_display}",
            f"- Origin: {origin_display}",
        ]
    else:
        trip_facts = [
            f"- Destination: {destination_display}",
            f"- Origin: {origin_display}",
            f"- Group: {resolved_group_type}"
            + (f" ({entities.group_size} people)" if entities.group_size
               else " (size not stated)"),
            f"- Travel dates: "
            f"{entities.travel_dates or 'not stated — do not name specific dates'}",
            f"- Visitor type: {state.visitor_type or 'unknown'}",
        ]
    if mode not in (slots.RIDE_MODE, slots.RENTAL_MODE):
        # Neither a resident booking a cab nor a rider hiring by the hour is
        # choosing a holiday length or a budget tier.
        trip_facts.insert(2, (
            f"- Duration: {duration_days} days"
            + ("" if entities.duration_days else " (ASSUMED — the user never said)")
        ))
        trip_facts.insert(4, (
            f"- Budget: {resolved_budget_range}"
            + ("" if entities.budget_range else " (not stated — keep costs mid-range)")
        ))
        trip_facts.append(
            f"- Interests: "
            f"{', '.join(entities.interests) if entities.interests else 'general sightseeing'}"
        )

    journey_section = (
        _build_rental_section(state) if mode == slots.RENTAL_MODE
        else _build_journey_section(state)
    )

    # Printed only when there is something under it. An empty "## REAL-TIME DATA"
    # heading is a promise of facts followed by none, and the steps point at it — so
    # the model either announces the gap ("weather could not be retrieved") or fills
    # it. Leaving the heading out means there is nothing to answer.
    realtime_section = (
        f"\n## REAL-TIME DATA\n{weather_section}{snowfall_section}\n"
        if (weather_section or snowfall_section)
        else ""
    )

    # A heading is an instruction like any other — the model writes what it says.
    # "TRIP" over a car hired by the hour produced trips; over a resident's cab ride
    # between two towns they know, it produced holidays.
    facts_heading = {
        slots.RENTAL_MODE: "THIS RENTAL",
        slots.RIDE_MODE: "THIS RIDE",
    }.get(mode, "TRIP")

    return f"""You are the AI travel concierge for **Him Cab** — Himachal Pradesh's most trusted local cab service. Help the user genuinely, and guide them toward booking their travel with Him Cab.

## LANGUAGE (STRICT)
{language_instruction}

## {facts_heading}
{chr(10).join(trip_facts)}
{user_context}
{assumptions_section}{journey_section}{travel_section}{outstation_instructions}{local_instructions}
{fare_block}{offers_block}{realtime_section}
## HOW TO REPLY (follow in order)
{response_steps}
{json_block}"""


def _extract_structured_itinerary(text: str) -> dict | None:
    """Extract the JSON block from the writer's response text."""
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            # Try to find any code block that looks like JSON
            blocks = text.split("```")
            for i in range(1, len(blocks), 2):  # odd indices are code blocks
                block = blocks[i].strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue
        
        # Fallback: attempt to find raw JSON without backticks
        match = re.search(r'\{\s*"destination"', text)
        if match:
            start_idx = match.start()
            json_str = text[start_idx:]
            end_idx = json_str.rfind('}')
            if end_idx != -1:
                json_str = json_str[:end_idx+1]
                return json.loads(json_str)

    except (json.JSONDecodeError, IndexError) as e:
        log.warning("structured_extraction_failed", error=str(e))
    return None


# ── Structured plumbing the model sometimes leaks into the prose ──────────────
# The itinerary JSON carries a `type` on every activity ("travel", "sightseeing",
# "dining", …) and a `recommended_mode` on the journey. Those belong in the JSON the
# app parses, NEVER in the sentences the rider reads. Asked to write both the prose and
# the JSON, the model sometimes echoes them as stray bullets under each activity:
#
#     • Evening: Visit the Hadimba Devi Temple, a pagoda-style temple...
#       ◦ type: sightseeing
#
# and prints "Recommended Mode: Cab" — which, for a cab service, says nothing at all.
# They are schema labels, not information, and the rider sees only clutter. We remove
# them in code for the same reason the fare and the forecast are stamped in code rather
# than trusted to the prompt (`_apply_real_fare`, `_apply_weather`) and for the same
# reason `guardrails` reads the finished text back: a rule the model breaks under
# pressure to be thorough is not a rule.

#: Any bullet or numbered-list marker, plus indentation, at the head of a line.
_LEADING_BULLET_RE = re.compile(r"^\s*(?:[-*•◦·▪▫○●‣⁃]+|\d+[.)])?\s*")  # noqa: RUF001 - the bullet chars are the point
#: An activity `type` echoed as its own line. Matched only when the WHOLE line is
#: "type: <one token>" — which is what a leak looks like ("type: sightseeing",
#: "type: check-in") and what a real prose sentence never does, since a sentence has
#: spaces. So a legitimate line that merely starts with the word "type" is left alone,
#: while every schema value (and off-vocabulary ones the coercer would map to `other`)
#: is caught.
_TYPE_LEAK_RE = re.compile(r"^type\s*[:=]\s*[\"']?[\w-]+[\"']?\s*$", re.IGNORECASE)
#: The journey's recommended travel mode. Always "cab" for a cab service, so the line
#: is noise whatever its value — dropped regardless of what follows the colon.
_MODE_LEAK_RE = re.compile(r"^recommended[_ ]?mode\s*[:=]", re.IGNORECASE)


def _strip_leaked_metadata(text: str) -> str:
    """Drop any line that is nothing but a leaked itinerary JSON field (see note above).

    Whole lines only: a leak arrives on its own bullet, so removing the line removes the
    leak and leaves the activity sentence above it untouched.
    """
    kept = [
        line
        for line in text.splitlines()
        if not (
            _TYPE_LEAK_RE.match(body := _LEADING_BULLET_RE.sub("", line).strip())
            or _MODE_LEAK_RE.match(body)
        )
    ]
    return "\n".join(kept).strip()


def _clean_response_text(text: str) -> str:
    """The human-readable reply: the JSON block removed, and any structured itinerary
    fields the model leaked into the prose stripped out (see `_strip_leaked_metadata`)."""
    clean = text
    if "```json" in clean:
        clean = clean.split("```json")[0].strip()
    elif "```" in clean:
        # Find the last code block and remove everything from it onwards
        parts = clean.rsplit("```", 2)
        if len(parts) >= 3:
            clean = parts[0].strip()

    # Strip raw JSON if no backticks were used
    match = re.search(r'\{\s*"destination"', clean)
    if match:
        clean = clean[:match.start()].strip()

    return _strip_leaked_metadata(clean)


async def _generate_streaming(llm, messages, token_queue) -> str:
    """
    Stream the LLM token-by-token, pushing only the human-visible text (everything
    before the trailing ```json block) onto `token_queue`, and return the full
    accumulated response for downstream JSON extraction. A small hold-back ensures a
    partial ``` fence is never emitted to the user.
    """
    full = ""
    emitted = 0
    stopped = False       # once the JSON fence appears, stop emitting to the user
    hold = 3              # chars held back so a partial "```" fence isn't shown

    async for chunk in llm.astream(messages):
        piece = getattr(chunk, "content", None) or ""
        if not piece:
            continue
        full += piece
        if stopped:
            continue
        fence = full.find("```")
        if fence != -1:
            if fence > emitted:
                await token_queue.put(full[emitted:fence])
            emitted = fence
            stopped = True
        else:
            safe = len(full) - hold
            if safe > emitted:
                await token_queue.put(full[emitted:safe])
                emitted = safe

    # No JSON fence at all → flush the held-back tail.
    if not stopped and emitted < len(full):
        await token_queue.put(full[emitted:])
    return full


async def writer_node(state: AgentState) -> dict:
    """
    LangGraph Writer Node.
    Generates the final AI response, with a structured itinerary when the intent calls
    for one. Streams real tokens when state.token_stream is set (SSE path); otherwise a
    single non-streaming call, with one retry if the JSON block failed to come through.
    """
    log.info("writer_node_entered", session_id=state.session_id, intent=state.intent)

    mode = slots.response_mode(state.intent, state.visitor_type)
    # Decided once for the whole turn, so the reply, the ride template and the guard
    # that reads the reply back can never disagree about which language this is.
    lang_key, _reply_language = detect_reply_language(state)

    # A cancellation never reaches this node — the graph routes it to `cancel_node`
    # straight out of the router. This is the guard under that, because the cost of
    # being wrong here is the whole reported bug: with no branch of its own, a
    # CANCELLED turn would fall through to the itinerary prompt below and answer
    # "cancel my booking" with a four-day plan and a booking screen. Every future
    # branch added to this graph gets that protection for free.
    if mode == slots.CANCELLED_MODE:
        from app.agent.cancellation import (
            cancel_scope,
            cancellation_event,
            cancellation_message,
            clear_request,
        )
        # NOTE: `detect_reply_language` is imported at module level and must NOT be
        # re-imported here. A function-local import of a name binds it local for the
        # WHOLE function, so re-importing it made every earlier use in this function —
        # including the ride branch above — raise UnboundLocalError.
        log.warning("writer_reached_by_a_cancellation", session_id=state.session_id)
        latest = next(
            (m.content for m in reversed(state.messages)
             if isinstance(m, HumanMessage) and isinstance(m.content, str)),
            "",
        )
        scope = cancel_scope(latest, state.entities)
        entities, cleared = clear_request(state.entities)
        text = cancellation_message(scope, lang_key)
        if state.token_stream is not None:
            await state.token_stream.put(text)
            await state.token_stream.put(None)
        return {
            "final_response_text": text,
            "messages": [AIMessage(content=text)],
            "entities": entities,
            "structured_events": [cancellation_event(scope, cleared)],
            "node_trace": ["writer"],
        }

    # ── A ride is assembled, not generated ────────────────────────────────────
    # Every line of it is a fact the system already holds: the route from the routing
    # engine, the fares from the Him Cab backend, the temperature from the weather
    # provider. There is nothing to compose, only to state.
    #
    # Asking the model to state facts it was handed kept failing. The prompt grew a
    # FORBIDDEN list — no day-by-day plan, no sightseeing, no hidden gems, no comfort
    # reassurance — and replies still arrived recommending Mall Road and the Ridge and
    # offering to plan a holiday, with no price and no booking screen in them. Not one
    # of those landmarks came from a tool.
    #
    # So it follows `cancellation`: a table the model cannot be talked out of. See
    # app/agent/ride_reply.py for what is given up in exchange.
    if mode == slots.RIDE_MODE:
        from app.agent.ride_reply import build_ride_reply

        text = _with_change_note(state, build_ride_reply(state, lang_key))
        if state.token_stream is not None:
            await state.token_stream.put(text)
            await state.token_stream.put(None)

        events: list[dict] = []
        handoff = _build_booking_handoff(state)
        if handoff:
            events.append({"type": "booking_handoff", "payload": handoff})
        if state.cab_fare:
            events.append({"type": "cab_fare", "payload": state.cab_fare})
        if state.offers:
            events.append({"type": "offers", "payload": {"offers": state.offers}})

        log.info(
            "ride_reply_assembled",
            has_fare=bool(state.cab_fare),
            cabs=len((state.cab_fare or {}).get("options") or []),
            has_route=state.route_estimate is not None,
            offers=len(state.offers or []),
            events=[e["type"] for e in events],
            session_id=state.session_id,
        )
        return {
            "final_response_text": text,
            "messages": [AIMessage(content=text)],
            "structured_events": events,
            "node_trace": ["writer"],
        }

    wants_itinerary = mode == slots.ITINERARY_MODE
    system_prompt = _build_system_prompt(state)

    # Build message history for multi-turn context
    messages = [SystemMessage(content=system_prompt)]
    for msg in state.messages[-6:]:  # last 3 turns
        messages.append(msg)

    full_response = ""
    structured_events = []
    token_queue = state.token_stream

    try:
        if token_queue is not None:
            # Live token-by-token generation for the SSE endpoint.
            llm = get_vertex_ai_service().get_flash_model(streaming=True)
            full_response = await asyncio.wait_for(
                _generate_streaming(llm, messages, token_queue),
                timeout=settings.llm_call_timeout_seconds,
            )
        else:
            llm = get_vertex_ai_service().get_flash_model(streaming=False)
            response = await asyncio.wait_for(
                llm.ainvoke(messages),
                timeout=settings.llm_call_timeout_seconds,
            )
            full_response = response.content

        # Extract structured JSON from response. A direct answer has no itinerary in
        # it by design, so don't look for one — and never spend a retry call hunting
        # for JSON the prompt explicitly asked the model not to write.
        itinerary_dict = _extract_structured_itinerary(full_response) if wants_itinerary else None

        # Retry: if structured extraction failed, ask the model to output just JSON
        if itinerary_dict is None and wants_itinerary:
            log.info("structured_extraction_retry", session_id=state.session_id)
            retry_messages = [*messages, AIMessage(content=full_response), HumanMessage(content="Please output ONLY the structured JSON itinerary from your response above. " "No explanation, just the JSON wrapped in ```json ... ```")]
            try:
                retry_response = await asyncio.wait_for(
                    llm.ainvoke(retry_messages),
                    timeout=settings.llm_call_timeout_seconds,
                )
                itinerary_dict = _extract_structured_itinerary(retry_response.content)
                if itinerary_dict:
                    log.info("structured_extraction_retry_success", session_id=state.session_id)
            except Exception as retry_err:
                log.warning("structured_extraction_retry_failed", error=str(retry_err), error_type=type(retry_err).__name__)

        # Clean the text response (remove JSON block)
        clean_text = _clean_response_text(full_response)

        # ── Validate the itinerary BEFORE building events ──────────────────────
        # The validated model is the single source of truth for both delivery
        # channels, so `structured_events[].payload` and the response's top-level
        # `itinerary` can never disagree.
        validated_itinerary: Itinerary | None = None
        if itinerary_dict:
            try:
                validated_itinerary = Itinerary.model_validate(
                    _normalize_itinerary_dict(itinerary_dict, state)
                )
            except Exception as val_err:
                log.warning(
                    "itinerary_validation_failed",
                    error=str(val_err),
                    error_type=type(val_err).__name__,
                    session_id=state.session_id,
                )
                validated_itinerary = None

        # ── Post-LLM overrides: live data beats generated data ────────────────
        # The model's figures are advisory whenever the real thing was fetched. Applied
        # before serialising, so the emitted event carries the authoritative values.
        if validated_itinerary is not None:
            if state.cab_fare:
                _apply_real_fare(validated_itinerary, state.cab_fare)
            _apply_weather(validated_itinerary, state)
            _apply_travel_info(validated_itinerary, state)

        # Build structured SSE events for mobile
        if validated_itinerary is not None:
            structured_events.append({
                "type": "itinerary",
                "payload": validated_itinerary.model_dump(mode="json"),
            })
        elif itinerary_dict:
            # Validation failed even after normalisation — ship the raw dict rather
            # than nothing, so the app still has something to render.
            structured_events.append({
                "type": "itinerary",
                "payload": itinerary_dict,
            })

        if state.weather_data and state.weather_data.get("alerts"):
            alerts = state.weather_data["alerts"]
            if isinstance(alerts, list) and alerts:
                structured_events.append({
                    "type": "weather_alert",
                    "payload": alerts[0],
                })
            elif isinstance(alerts, dict):
                structured_events.append({
                    "type": "weather_alert",
                    "payload": alerts,
                })

        if state.snowfall_data and state.snowfall_data.get("risk") in ("high", "critical"):
            structured_events.append({
                "type": "snowfall_alert",
                "payload": state.snowfall_data,
            })

        # What can be hired, restated on the answering turn as well as the asking one.
        # A rider who was quoted an 8-hour package can then switch to 4 or 12 with a
        # tap, and the app never has to remember a list from an earlier turn.
        if mode == slots.RENTAL_MODE and state.rental_packages:
            structured_events.append({
                "type": "rental_packages",
                "payload": {
                    "packages": state.rental_packages,
                    "selected_hours": state.entities.rental_hours,
                },
            })

        # Emit the real cab fare as its own structured event (fare card in the app)
        if state.cab_fare:
            structured_events.append({
                "type": "cab_fare",
                "payload": state.cab_fare,
            })

        # Live coupons that apply to this trip. Emitted only when there ARE some — an
        # empty offers card is worse than no card, and the app should never have to
        # decide whether a zero-length list means "none" or "not fetched".
        if state.offers:
            structured_events.append({
                "type": "offers",
                "payload": {
                    "trip_type": (state.cab_fare or {}).get("trip_type"),
                    "fare_inr": (state.cab_fare or {}).get("fare_inr"),
                    "offers": state.offers,
                    "best_offer": state.offers[0],
                },
            })

        # Hand the rider straight to the booking screen, with both endpoints and their
        # coordinates so the map opens on the journey we just priced. Both travelling
        # personas get it: a local is hailing the cab now, and a visitor's itinerary is
        # of no use to them until somebody drives them to it — the arrival transfer is
        # the one thing every outstation plan needs booked. A question about a place
        # (ANSWER_MODE) is not a journey, so it gets nothing.
        #
        # This is a handoff, not a booking — nothing is reserved here, which is why the
        # reply announces the screen rather than asking them to confirm.
        if mode in (slots.RIDE_MODE, slots.ITINERARY_MODE, slots.RENTAL_MODE):
            handoff = _build_booking_handoff(state)
            if handoff:
                structured_events.append({
                    "type": "booking_handoff",
                    "payload": handoff,
                })

        log.info(
            "writer_completed",
            mode=mode,
            visitor_type=state.visitor_type,
            response_length=len(full_response),
            has_itinerary=validated_itinerary is not None,
            assumed_fields=state.assumed_fields,
            offers_quoted=len(state.offers or []),
            structured_events=len(structured_events),
            session_id=state.session_id,
        )

        # ── The last gate before a rider reads it ─────────────────────────────
        # Everything above enforces the rules on the way IN: the prompt states them,
        # and the live fare, forecast and route are stamped over the itinerary JSON.
        # None of that touches the PROSE, so until here every rule the prompt stated
        # was enforced only by the model's willingness to follow it — and a prompt
        # that forbade sightseeing in writing still produced Mall Road and the Ridge.
        #
        # This reads the finished text back and checks its claims against the facts
        # this turn actually holds. A coupon code the offers endpoint never returned,
        # or a price when no fare was fetched, takes its line out; the rest is logged
        # so a pattern is visible before it becomes a complaint.
        # Findings are logged inside the guard; the caller only needs the text.
        clean_text, _findings = apply_guardrails(state, clean_text, mode, lang_key)
        clean_text = _with_change_note(state, clean_text)

        return {
            "final_response_text": clean_text,
            "structured_itinerary": validated_itinerary,
            "structured_events": structured_events,
            "messages": [AIMessage(content=clean_text)],
            "node_trace": ["writer"],
        }

    except Exception as e:
        log.error("writer_failed", error=str(e), error_type=type(e).__name__, session_id=state.session_id)
        fallback = "Sorry, I encountered an issue generating your travel plan. Please try again."
        return {
            "final_response_text": fallback,
            "structured_events": [],
            "messages": [AIMessage(content=fallback)],
            "node_trace": ["writer"],
            "error": str(e),
        }

    finally:
        # Always close the token stream so the SSE endpoint stops waiting.
        if token_queue is not None:
            await token_queue.put(None)  # None = end-of-stream sentinel
