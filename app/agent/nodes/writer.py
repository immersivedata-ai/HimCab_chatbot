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
  • ANSWER    — a short direct answer to a question about somewhere

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
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent import slots
from app.agent.language import detect_reply_language
from app.agent.state import AgentState
from app.core.config import settings
from app.core.logging import get_logger
from app.models.itinerary import (
    ActivityType,
    BudgetRange,
    CabPackage,
    CostBreakdown,
    GroupType,
    Itinerary,
    TravelInfo,
    WeatherForecast,
)
from app.models.session import ExtractedEntities
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
    approximate = bool(cf.get("is_approximate"))
    distance_km = cf.get("distance_km")

    lines = []
    for o in options:
        fare = o.get("total_fare_inr")
        if fare is None:
            continue
        name = o.get("category_name") or "Cab"
        seats = o.get("seat_capacity")
        seat_str = f", {seats}-seater" if seats else ""
        flag = "  (recommended for this group/budget)" if name == headline else ""
        lines.append(f"  - {name}: Rs {round(fare):,}{seat_str}{flag}")

    route = f" for the ~{round(distance_km)} km journey" if distance_km else ""
    if approximate:
        header = (
            f"### HIM CAB FARES (APPROXIMATE — use these EXACT figures)\n"
            f"Estimated Him Cab fares{route}, based on a computed route distance.\n"
            f"State them as approximate — say \"approx\" or \"around\" — and add that the "
            f"exact fare is confirmed at booking. Do NOT round, adjust or invent numbers.\n"
        )
    else:
        header = (
            f"### HIM CAB FARES (real prices — use these EXACT figures)\n"
            f"Live Him Cab fares{route}. Do NOT round, adjust or invent numbers.\n"
        )
    return header + "\n".join(lines) + "\n"


def _build_offers_block(state: AgentState) -> str:
    """The HIM CAB OFFERS section — real coupons, or an instruction to stay silent.

    The reply used to be told never to mention a discount at all, which was right when
    the only coupons available were ones the model made up. Now that offers come from
    the backend (`app/services/himcab/offers_service.py`), staying silent means a rider
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
    # Distinguishes a fare priced on a surveyed route from one priced on a computed
    # distance, so the app can render "approx" without re-deriving it.
    cp.fare_source = (
        "himcab_api_approx" if cab_fare.get("is_approximate")
        else cab_fare.get("source", "himcab_api")
    )
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


def _build_journey_section(state: AgentState) -> str:
    """Distance and drive time for the route, or nothing when it isn't known.

    Surveyed routes carry measured figures; the rest are derived from coordinates and
    are labelled, because a drive time quoted to the minute implies knowledge of the
    road that this app does not have.
    """
    route = state.route_estimate
    if not route:
        return ""

    qualifier = " (approximate)" if route.get("is_approximate") else ""
    return (
        f"\n### THIS JOURNEY\n"
        f"- {route['origin']['name']} to {route['destination']['name']}\n"
        f"- Drive time: about {route['duration_text']}{qualifier}\n"
        f"- Distance: about {round(route['distance_km'])} km\n"
        f"Use these exact figures. Do NOT add traffic, road or weather-delay claims — "
        f"nothing here observes them.\n"
    )


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
    if not route:
        return None

    fare = state.cab_fare or {}
    # Prefill the coupon only when eligibility was actually checked against a fare.
    # Preselecting one we could not verify puts a code on the booking screen that the
    # payment step may then reject — worse than leaving the field empty.
    best_offer = next(
        (o for o in (state.offers or []) if o.get("applies_to_this_fare")), {}
    )
    return {
        "origin": route["origin"],
        "destination": route["destination"],
        "distance_km": route["distance_km"],
        "duration_minutes": route["duration_minutes"],
        "duration_text": route["duration_text"],
        "is_approximate": route["is_approximate"],
        # Where the distance came from: "surveyed" | "gps" | "city".
        "distance_source": route.get("source"),
        "trip_type": fare.get("trip_type"),
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

    weather_section = ""
    if state.weather_data:
        w = state.weather_data
        weather_section = f"""
**Current Weather at {destination_display}:**
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
    if destination and (entities.origin or route_known):
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

    user_context = ""
    if state.user_context_summary:
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
    )
    # Weather was being dropped from itineraries even when it had been fetched, so when
    # there is a report it is a required section rather than an optional aside.
    weather_step = (
        "Weather: include a short 'Weather' line near the top with the CURRENT temperature "
        "and condition from REAL-TIME DATA, and a packing or safety note if there is an "
        "alert or snowfall risk. Use only the numbers given."
        if state.weather_data
        else "Weather: no live report was available — do not state any temperature or "
             "condition, and do not claim the forecast is unavailable unless it is relevant."
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
            "wording rule in that section about whether the prices are exact or approximate."
            if has_fares
            else " No fare is available, so promote Him Cab without stating any price and say "
                 "the exact fare is confirmed at booking."
        )
    )
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
        else "NEVER mention or invent any coupon, promo, or discount code — no offer "
             "applies to this trip."
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

    elif mode == slots.RIDE_MODE:
        # A Himachal resident booking a cab. They know the destination — what they are
        # deciding is which vehicle, what it costs, how long the drive is and what the
        # weather is doing. An itinerary here is noise wrapped around the useful part.
        steps = [
            language_step,
            honesty_step,
            f"Open with ONE line confirming the journey: {origin_display} to "
            f"{destination_display}.",
            "Then a JOURNEY line with the drive time and distance from THIS JOURNEY below, "
            "exactly as given. Say nothing about traffic or road conditions — nothing here "
            "measures them.",
            weather_step,
            (
                "Recommend the cabs from HIM CAB FARES as short bullets — vehicle, seats and "
                "price. They are ALREADY narrowed to what seats this group, so recommend every "
                "one of them and never mention a larger vehicle."
                if has_fares
                else "No fare is available for this route, so do NOT state or estimate any "
                     "price. Say the fare is confirmed when they book."
            ),
            "Do NOT write a day-by-day itinerary, sightseeing plan, hidden gems or any JSON — "
            "they live here and are booking a ride, not planning a holiday.",
            coupons_step,
            "Close with ONE short line telling them the booking screen is ready with their "
            "pickup and drop already filled in. Do NOT ask whether they want to book, do not "
            "ask them to confirm, and do not ask for any more details — the app takes over "
            "from here.",
            "Keep the whole reply under about 12 short lines.",
        ]
        json_block = ""

    else:
        steps = [
            language_step,
            honesty_step,
            "Be concise: short bullet points, one line each. No long paragraphs or filler.",
            f"Give a day-by-day itinerary for all {duration_days} days (3-5 bullets per day: "
            "morning / afternoon / evening).",
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
    trip_facts = [
        f"- Destination: {destination_display}",
        f"- Origin: {origin_display}",
        f"- Group: {resolved_group_type}"
        + (f" ({entities.group_size} people)" if entities.group_size else " (size not stated)"),
        f"- Travel dates: {entities.travel_dates or 'not stated — do not name specific dates'}",
        f"- Visitor type: {state.visitor_type or 'unknown'}",
    ]
    if mode != slots.RIDE_MODE:
        # A resident booking a cab is not choosing a holiday length or a budget tier.
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

    journey_section = _build_journey_section(state)

    return f"""You are the AI travel concierge for **Him Cab** — Himachal Pradesh's most trusted local cab service. Help the user genuinely, and guide them toward booking their travel with Him Cab.

## LANGUAGE (STRICT)
{language_instruction}

## TRIP
{chr(10).join(trip_facts)}
{user_context}
{assumptions_section}{journey_section}{travel_section}{outstation_instructions}{local_instructions}
{fare_block}{offers_block}
## REAL-TIME DATA
{weather_section}{snowfall_section}

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
        import re
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


def _clean_response_text(text: str) -> str:
    """Strip the JSON block from the human-readable response text."""
    clean = text
    if "```json" in clean:
        clean = clean.split("```json")[0].strip()
    elif "```" in clean:
        # Find the last code block and remove everything from it onwards
        parts = clean.rsplit("```", 2)
        if len(parts) >= 3:
            clean = parts[0].strip()
            
    # Strip raw JSON if no backticks were used
    import re
    match = re.search(r'\{\s*"destination"', clean)
    if match:
        clean = clean[:match.start()].strip()
        
    return clean


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
    HOLD = 3              # chars held back so a partial "```" fence isn't shown

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
            safe = len(full) - HOLD
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
            retry_messages = messages + [
                AIMessage(content=full_response),
                HumanMessage(content=(
                    "Please output ONLY the structured JSON itinerary from your response above. "
                    "No explanation, just the JSON wrapped in ```json ... ```"
                )),
            ]
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
        if mode in (slots.RIDE_MODE, slots.ITINERARY_MODE):
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
