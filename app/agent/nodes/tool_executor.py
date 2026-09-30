"""
app/agent/nodes/tool_executor.py
──────────────────────────────────
Tool Executor Node — calls the external services IN PARALLEL using asyncio.gather().

Three tools run today:
  1. Weather + snowfall risk for the destination (one fetch; the risk is derived
     from that same report rather than fetched again)
  2. A real Him Cab fare quote, when the flag is on and a rider JWT was forwarded
  3. The rider's live Him Cab coupons, filtered to ones that apply to THIS trip

It also resolves the journey itself (`_resolve_journey`) via live geocoding + routing,
because the distance, the drive time and the map coordinates all have to agree with the
fare that was quoted on them. When the app has sent a GPS fix, the journey is measured
from where the rider actually IS rather than from the centroid of the city they named —
see `_resolve_journey`.

Order matters between the three: the fare needs the distance, and the offers need the
fare (a coupon with a Rs 2,000 minimum is not an offer on a Rs 900 ride). So the route
is resolved first, in memory; weather and the fare then run concurrently; and offers
follow the fare. Total latency is one network round trip plus one, not three.

Every call is wrapped in a timeout and degrades to None instead of raising — a missing
tool result costs detail in the reply, never the reply itself.
"""

import asyncio
from typing import Any, Optional

from app.agent import slots
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import TripType
from app.core.logging import get_logger
from app.domain.conversation import ExtractedEntities
from app.integrations.external.weather import get_weather_service
from app.integrations.himcab import (
    build_fare_quote_input,
    build_rental_quote_input,
    get_fare_service,
    get_offers_service,
    get_rental_service,
    matching_package,
    offered_hours,
    offered_packages,
    packages_payload,
    usable_offers,
)
from app.integrations.himcab.models import FareQuote, Offer
from app.integrations.himcab.quote_builder import trip_type_for
from app.services import geo

log = get_logger(__name__)

# Per-tool timeout — prevents a single slow API from stalling the entire pipeline
TOOL_TIMEOUT_SECONDS = 20


async def _safe_call(coro_or_value: Any, label: str, session_id: Optional[str] = None) -> Any:
    """
    Wrapper that:
      1. Validates the argument is actually awaitable (catches sync/async mismatch bugs)
      2. Enforces a per-tool timeout
      3. Catches exceptions and logs them without failing the whole node

    `session_id` is threaded through so the two degradation lines below can be correlated
    to the turn that failed. Without it, `tool_call_failed` — the ONLY line that carries
    the error type behind a silently-missing fare — could not be tied to a conversation
    under concurrency, which is the difference between "a fare call is erroring" and
    "THIS rider's fare call errored". Optional so the many other call sites are untouched.
    """
    try:
        # Guard: if someone passes a non-coroutine (sync result), return it directly
        if not asyncio.iscoroutine(coro_or_value) and not asyncio.isfuture(coro_or_value):
            log.debug("tool_sync_result", tool=label)
            return coro_or_value

        result = await asyncio.wait_for(coro_or_value, timeout=TOOL_TIMEOUT_SECONDS)
        return result

    except TimeoutError:
        log.warning(
            "tool_call_timeout",
            tool=label,
            timeout_seconds=TOOL_TIMEOUT_SECONDS,
            session_id=session_id,
        )
        return None

    except Exception as e:
        log.warning(
            "tool_call_failed",
            tool=label,
            error=str(e),
            error_type=type(e).__name__,
            session_id=session_id,
        )
        return None


async def _resolve_journey(state: AgentState) -> Optional[geo.Journey]:
    """Resolve the journey live, measured from the best pickup we have.

    A GPS fix beats a city name whenever there is one: it is where the cab will actually
    collect them, and for a rider in Solang Valley or halfway down the Kullu road,
    "Manali" is a different journey by 15-40 km. The fix is skipped only when the rider
    has NAMED an origin somewhere else entirely — booking a Delhi pickup from a Shimla
    sofa is a real thing people do, and their words win over their phone.

    None when any part cannot be resolved. That is a real answer: no distance means the
    fare endpoint cannot be called, and the reply says the fare is confirmed at booking
    rather than quoting a guess.
    """
    entities = state.entities
    destination = entities.destination
    if not destination:
        return None

    named_origin = entities.origin or entities.home_city
    fix = state.rider_location

    if fix is not None and _fix_matches_origin(fix, named_origin):
        return await geo.resolve_journey(
            destination,
            origin_name=named_origin,
            # Re-resolved from the raw coordinates rather than the stored name: the fix
            # is where the cab goes, and the town it sits in is only a label for it.
            origin_fix=geo.LocationFix(
                lat=fix.lat,
                lon=fix.lon,
                in_himachal=fix.in_himachal,
                nearest_city=fix.place_name,
                nearest_city_km=fix.place_distance_km,
                state=fix.state,
            ),
        )
    if named_origin:
        return await geo.resolve_journey(destination, origin_name=named_origin)
    return None


def _fix_matches_origin(fix, named_origin: Optional[str]) -> bool:
    """True when the device fix and the rider's stated starting city agree.

    "Agree" means either they never named one, or the name matches the town the fix
    sits in. Anything else — they said Delhi, the phone says Shimla — means the fix is
    not the pickup point, so it must not price the trip.
    """
    if not named_origin:
        return True
    if not fix.place_name:
        return False
    return geo.same_place(named_origin, fix.place_name)


def _journey_dict(state: AgentState, journey: Optional[geo.Journey]) -> Optional[dict]:
    """Package the journey for state: both endpoints with coordinates, plus the drive.

    Coordinates travel with it because the app's booking screen drops pins for the
    pickup and the drop — handing it two place NAMES would make it geocode them again
    and possibly land somewhere other than the point we priced.
    """
    if journey is None:
        return None

    packaged = journey.as_dict()

    # A vague fix is described as "near <town>" rather than named outright: the nearest
    # place the geocoder could point at is not the same claim as where the rider is.
    fix = state.rider_location
    named_origin = state.entities.origin or state.entities.home_city
    if fix is not None and _fix_matches_origin(fix, named_origin) and not named_origin:
        if not fix.is_precise and fix.place_name:
            packaged["origin"]["name"] = f"near {fix.place_name}"
        elif not fix.place_name:
            packaged["origin"]["name"] = "your current location"
    return packaged


def _package_fare(quote: Any) -> Optional[dict]:
    """A FareQuote as the app and the writer consume it, or None when unusable.

    Shared by the trip and rental paths so a price means the same thing in both — the
    rental branch adds its own duration fields on top rather than assembling a second,
    subtly different shape.
    """
    if not isinstance(quote, FareQuote) or quote.base_fare_inr <= 0:
        return None
    return {
        "fare_inr": round(quote.base_fare_inr),
        "currency": quote.currency,
        "category_id": quote.category_id,
        "category_name": quote.category_name,
        "vehicle_label": quote.vehicle_label,
        "seat_capacity": quote.seat_capacity,
        "trip_type": quote.trip_type,
        "distance_km": quote.distance_km,
        "options": [a.model_dump() for a in quote.alternatives],  # all categories
        "source": "himcab_api",
    }


def _select_requested_category(cab_fare: dict, requested: Optional[str]) -> bool:
    """Promote the cab the rider ASKED for to the headline. True when it was found.

    The app's Rent a Car screen makes the rider pick a cab before it will book, so the
    chat has to honour the same choice: whichever cab they name becomes the price we
    confirm, the price a coupon is checked against, and the category the booking screen
    opens on. Without this the reply would agree with the rider in prose while every
    number in it, and the handoff underneath it, still described the cab WE recommended.

    Matched by tier word against what the backend actually priced — never accepted on
    the rider's word alone. False means the tier is not among this package's cabs, and
    the caller says so rather than substituting a different one.
    """
    if not requested:
        return False
    wanted = requested.strip().lower()
    for option in cab_fare.get("options") or []:
        name = (option.get("category_name") or "").lower()
        if not name or wanted not in name:
            continue
        cab_fare.update({
            "fare_inr": round(option["total_fare_inr"]),
            "category_id": option.get("category_id"),
            "category_name": option.get("category_name"),
            "vehicle_label": option.get("category_name"),
            "selected_by_rider": True,
        })
        if option.get("seat_capacity") is not None:
            cab_fare["seat_capacity"] = option["seat_capacity"]
        return True
    return False


async def _rental_tools(state: AgentState) -> dict:
    """Packages and a price for a car hired by the hour.

    Two lists, and the difference matters. `packages` is the rate card as WE could read
    it; `offered` is what the rider may choose from, which is the rate card when it is
    readable and the configured durations (3/5/8) when it is not. An answer is validated
    against `offered`, so a rider is never quietly quoted nothing for a duration they
    were offered.

    The fare is asked for whenever the duration is one we offer — NOT only when we could
    parse a package for it. The backend does its own chart lookup from `rentHours`
    (`TripService.findMatchingRentalChart`) and the rental price is
    `farePerKilometer × maxKilometer` off that chart, so `distanceInMeters` plays no part
    in it; extra distance is charged at drop-off against the odometer. Gating this call
    on our own parse of the rate card is what left a rider staring at a package they had
    chosen with no price beside it, on a turn where the backend would have priced it
    perfectly well. If the backend has no chart for those hours it returns nothing usable
    and we still show no price — which is the same honest outcome, reached by asking.

    Weather is fetched for the PICKUP city here, not a destination — an hourly rental
    has none, and the rider still wants to know what they are driving out into.
    """
    entities = state.entities
    pickup = entities.origin or entities.home_city

    packages: list = []
    if settings.enable_rentals and state.rider_jwt:
        packages = await _safe_call(
            get_rental_service().get_packages(rider_jwt=state.rider_jwt), "rental_packages"
        ) or []
    # Same rule as `available_packages`, so the asking turn and the quoting turn offer
    # the same durations: the configured ones stand in for an unreadable rate card, but
    # a switched-off feature offers nothing at all.
    offered = offered_packages(packages) if settings.enable_rentals else []

    tasks: dict[str, Any] = {}
    if pickup:
        tasks["weather"] = _safe_call(get_weather_service().get_weather(pickup), "weather")

    # The pin the booking screen opens its map on. A rental has no drop, so there is no
    # journey to resolve — only where the cab goes. Same precedence as a journey's
    # origin: the device's fix when it agrees with what they said, else the city itself.
    fix = state.rider_location
    if fix is not None and _fix_matches_origin(fix, pickup):
        tasks["pickup"] = _safe_call(
            geo.resolve_pickup(
                origin_name=pickup,
                origin_fix=geo.LocationFix(
                    lat=fix.lat,
                    lon=fix.lon,
                    in_himachal=fix.in_himachal,
                    nearest_city=fix.place_name,
                    nearest_city_km=fix.place_distance_km,
                    state=fix.state,
                ),
            ),
            "pickup",
        )
    elif pickup:
        tasks["pickup"] = _safe_call(geo.resolve_pickup(origin_name=pickup), "pickup")

    # The rate card's own entry for these hours, when we could read one. It carries the
    # included kilometres — what the package COVERS — and nothing else depends on it:
    # the price comes from the backend either way.
    package = matching_package(packages, entities.rental_hours)
    offered_payload = packages_payload(offered)
    quote_input = None
    fare_gate = {
        "enable_real_fares": settings.enable_real_fares,
        "has_rider_jwt": bool(state.rider_jwt),
        # A duration we actually offer. Deliberately NOT "a duration we parsed a package
        # for": the backend matches the chart itself, so a readable rate card is a
        # convenience for us, not a precondition for a price.
        "hours_are_offered": bool(entities.rental_hours)
        and entities.rental_hours in offered_hours(offered_payload),
    }
    if all(fare_gate.values()):
        quote_input = build_rental_quote_input(entities, package)
    if quote_input is not None:
        tasks["cab_fare"] = _safe_call(
            get_fare_service().get_quote(quote_input, rider_jwt=state.rider_jwt),
            "cab_fare",
            session_id=state.session_id,
        )
    else:
        log.info(
            "rental_fare_gate_blocked",
            session_id=state.session_id,
            requested_hours=entities.rental_hours,
            offered_hours=offered_hours(offered_payload),
            rate_card_hours=[p.hours for p in packages],
            **fare_gate,
        )

    results: dict[str, Any] = {}
    if tasks:
        results = dict(zip(tasks.keys(), await asyncio.gather(*tasks.values()), strict=False))

    weather_obj = results.get("weather")
    cab_fare = _package_fare(results.get("cab_fare"))
    # The cab the rider picked, resolved BEFORE offers: a coupon's minimum spend is
    # checked against the fare, and after this line the fare is theirs rather than ours.
    chose_cab = False
    chosen_entities: Optional[ExtractedEntities] = None
    if cab_fare:
        cab_fare["rental_hours"] = entities.rental_hours
        # None when the rate card was unreadable: the price is still the backend's, but
        # what the package INCLUDES is not something we were told, so nothing claims it.
        cab_fare["included_km"] = package.included_km if package else None
        chose_cab = _select_requested_category(cab_fare, entities.cab_category)
        # "Haan, book kar do" to the cab question: the rider has taken our
        # recommendation — the headline, which `select_cabs_for_group` chose as the
        # smallest cab that seats their party. It becomes THEIR choice exactly as a
        # typed "suv" would, so the booking screen opens with it selected and the price
        # beside it, instead of on an empty cab list they have to fill in again.
        if not chose_cab and state.accepted_recommended_cab:
            recommended = cab_fare.get("category_name")
            chose_cab = _select_requested_category(cab_fare, recommended)
            if chose_cab and recommended:
                # Persisted as the tier word `sanitize.cab_category` will keep, so the
                # choice survives the next turn's validation gate.
                entities = entities.model_copy(deep=True)
                entities.cab_category = (
                    slots.parse_cab_category(recommended) or recommended.strip().lower()
                )
                entities.last_asked_field = None
                chosen_entities = entities

    offers = await _fetch_offers(
        state, trip_type=TripType.RENTAL, fare_inr=(cab_fare or {}).get("fare_inr")
    )
    if offers and cab_fare:
        best = offers[0]
        cab_fare["best_offer"] = best
        cab_fare["fare_after_offer_inr"] = best["fare_after_offer_inr"]

    log.info(
        "rental_tools_completed",
        rate_card_packages=len(packages),
        offered_hours=offered_hours(offered_payload),
        requested_hours=entities.rental_hours,
        matched_package=package.hours if package else None,
        requested_cab=entities.cab_category,
        selected_cab=(cab_fare or {}).get("category_name") if chose_cab else None,
        has_cab_fare=cab_fare is not None,
        session_id=state.session_id,
    )

    result = {
        "weather_data": weather_obj.model_dump(mode="json") if weather_obj else None,
        "cab_fare": cab_fare,
        "offers": offers,
        "pickup": results.get("pickup"),
        "rental_packages": offered_payload,
        "node_trace": ["tool_executor"],
    }
    if chosen_entities is not None:
        # The recommendation the rider accepted, written back so `persist` stores it
        # and the next turn still knows which cab they are booking.
        result["entities"] = chosen_entities
        result["last_asked_field"] = None

    # ── Answers we cannot act on go back to the rider as questions ────────────
    # Each of these puts a field back in flight, and each needs the same two halves: the
    # slot is CLEARED, so the correction is read as a fresh answer (a slot that still
    # holds a value is never re-read — see `slots.read_answer`), and the field is marked
    # as asked, so a bare reply is not discarded. What was refused travels on state so
    # this turn's reply can name it.
    options = (cab_fare or {}).get("options") or []
    asked_for_cab = entities.asked_counts.get("cab_category", 0)

    def _ask_again(field: str, clear: bool) -> ExtractedEntities:
        asking = entities.model_copy(deep=True)
        if clear:
            setattr(asking, field, None)
        asking.last_asked_field = field
        asking.asked_counts[field] = asking.asked_counts.get(field, 0) + 1
        result["entities"] = asking
        result["last_asked_field"] = field
        return asking

    # A duration nothing on offer matches. Checked against `offered` rather than the
    # rate card, so a rider during a rate-card outage is still told that 6 hours is not
    # one of the durations Him Cab sells instead of being quietly quoted nothing. The
    # planner normally catches this first (`_reject_unsold_duration`); this is the
    # backstop that reads the LIVE rate card rather than the cached offer list.
    if entities.rental_hours and entities.rental_hours not in offered_hours(offered_payload):
        _ask_again("rental_hours", clear=True)
        result["unbookable_rental_hours"] = entities.rental_hours
        log.info(
            "rental_duration_rejected",
            session_id=state.session_id,
            requested_hours=entities.rental_hours,
            offered_hours=offered_hours(offered_payload),
        )
    # A cab tier this package does not have. The duration stands — only the cab choice
    # is re-asked, and the reply lists the cabs that were actually priced.
    elif entities.cab_category and cab_fare and not chose_cab:
        _ask_again("cab_category", clear=True)
        result["awaiting_cab_choice"] = True
        result["unavailable_cab_category"] = entities.cab_category
        log.info(
            "rental_cab_category_unavailable",
            session_id=state.session_id,
            requested=entities.cab_category,
            available=[o.get("category_name") for o in options],
        )
    # Priced, and the rider has not picked a cab. The app will not book without one, so
    # the quoting reply ends by asking which — and the slot is marked in flight here,
    # which is what makes a one-word "luxury" next turn an answer rather than noise.
    #
    # Bounded like every other question in this system (`max_clarification_attempts`):
    # past the cap the assistant stops asking and confirms the cab it recommended, so a
    # rider who ignores the question still reaches a booking screen.
    elif options and not chose_cab:
        if asked_for_cab < settings.max_clarification_attempts:
            _ask_again("cab_category", clear=False)
            result["awaiting_cab_choice"] = True
        else:
            log.info(
                "rental_cab_choice_assumed",
                session_id=state.session_id,
                recommended=cab_fare.get("category_name"),
                asked=asked_for_cab,
            )

    return result


async def tool_executor_node(state: AgentState) -> dict:
    """
    LangGraph Tool Executor Node.
    Dispatches the external tool calls in parallel with per-tool timeouts.
    """
    log.info("tool_executor_entered", session_id=state.session_id)

    entities: ExtractedEntities = state.entities
    destination = entities.destination

    mode = slots.response_mode(state.intent, state.visitor_type)

    if mode == slots.RENTAL_MODE:
        # A rental is priced on a DURATION, not a route. There is no destination to
        # look up weather for and no journey to measure, so it takes its own path.
        return await _rental_tools(state)

    if not destination:
        # A general question can reach the writer with no destination; there is simply
        # nothing to look up for it.
        log.info("tool_executor_no_destination", intent=state.intent)
        return {"node_trace": ["tool_executor"]}

    weather_svc = get_weather_service()
    intent = state.intent or "plan_trip"

    tasks: dict[str, Any] = {
        "weather": _safe_call(weather_svc.get_weather(destination), "weather"),
    }

    # ── The journey ───────────────────────────────────────────────────────────
    # Resolved ONCE, live, and reused three ways: it prices the fare, it is the drive
    # time quoted to a local rider, and its coordinates are what the app drops on its
    # map at the booking handoff. Resolving it per consumer is how a price and a map
    # end up describing different trips.
    journey = await _resolve_journey(state)
    route_estimate = _journey_dict(state, journey)

    # ── Real cab fare (Him Cab backend) ───────────────────────────────────────
    # Not asked for at all on a local ride or a trip plan: those replies name no cab
    # (`slots.lists_cabs`), and the app's booking screen prices the cabs itself. A fare
    # fetched here would only put a cab back on screen through the `cab_fare` event,
    # the itinerary's cab package or the handoff's preselected category — each one a
    # recommendation the flow no longer makes.
    #
    # Otherwise gated: flag on, a usable pickup point, pricing-relevant intent, and a
    # forwarded rider JWT. Any failure degrades gracefully to None via _safe_call.
    lists_cabs = slots.lists_cabs(mode)
    fare_gate = {
        "enable_real_fares": settings.enable_real_fares,
        # A named city, OR a GPS fix — either gives us somewhere to measure from. This
        # used to demand a name, which meant a rider whose phone knew exactly where
        # they were still got no price until they typed the town out.
        "has_origin": bool(
            entities.origin or entities.home_city or state.rider_location
        ),
        # Exactly the intents that produce a trip — the same predicate the writer uses,
        # so a new travel intent can never be added and silently lose its fare. A local
        # rider's whole reply IS the fare, so this must cover RIDE_MODE too.
        "pricing_intent": slots.response_mode(intent, state.visitor_type) in (
            slots.ITINERARY_MODE, slots.RIDE_MODE
        ),
        "has_rider_jwt": bool(state.rider_jwt),
    }
    should_quote = lists_cabs and all(fare_gate.values())
    quote_input = None
    if not lists_cabs:
        # By design, not a failure — so not `fare_gate_blocked`, which is the line read
        # when a fare that SHOULD have come back did not.
        log.debug("cab_fare_not_requested", mode=str(mode), session_id=state.session_id)
    elif not should_quote:
        # Log exactly which condition blocked it — this is the single most useful line
        # for diagnosing "why didn't I get a fare" without needing more round-trips.
        log.info("fare_gate_blocked", intent=intent, session_id=state.session_id, **fare_gate)
    else:
        # The rate card is chosen by `trip_type_for`: the LOCAL/resident flow is priced
        # LOCAL to match what the Him Cab app charges the rider, everyone else by the
        # journey. state.visitor_type is the flow's identity — the page the rider opened,
        # not a guess. The route is passed in so the price is built on the same distance
        # the reply quotes.
        quote_input = (
            build_fare_quote_input(
                entities,
                journey,
                trip_type=trip_type_for(state.visitor_type, journey.distance_km),
            )
            if journey
            else None
        )
        # distanceInMeters is required by the endpoint, so no resolved journey means no
        # quote. That happens when the geocoder cannot place a name or the routing
        # engine finds no road — both are logged below rather than guessed around.
        if quote_input is not None and quote_input.distance_km is not None:
            tasks["cab_fare"] = _safe_call(
                get_fare_service().get_quote(quote_input, rider_jwt=state.rider_jwt),
                "cab_fare",
                session_id=state.session_id,
            )
        else:
            quote_input = None
            log.warning(
                "cab_fare_skipped_no_route",
                origin=entities.origin or entities.home_city,
                destination=destination,
                session_id=state.session_id,
                hint="geocoder could not place a name, or no road route was found",
            )

    # ── Execute all in parallel ───────────────────────────────────────────────
    task_keys = list(tasks.keys())
    raw_results = await asyncio.gather(*tasks.values(), return_exceptions=False)
    results = dict(zip(task_keys, raw_results, strict=False))

    failed = [k for k, v in results.items() if v is None]
    if failed:
        log.warning(
            "tools_partial_failure",
            succeeded=[k for k, v in results.items() if v is not None],
            failed=failed,
            session_id=state.session_id,
        )

    # ── Package results into state ────────────────────────────────────────────
    weather_obj = results.get("weather")
    # Derive snowfall risk from the weather report we already have (no second fetch).
    snowfall_data = weather_svc.snowfall_from_report(weather_obj) if weather_obj else None

    # Real cab fare — gross prices straight from /fare/estimate. The endpoint takes no
    # offerId, so these are pre-discount; any coupon is applied on top, below.
    raw_fare = results.get("cab_fare")
    cab_fare = _package_fare(raw_fare)
    if cab_fare:
        # Journey context, so a fare card can show what the price is FOR without the
        # app having to join it against the route event itself.
        cab_fare["duration_text"] = (route_estimate or {}).get("duration_text")
    elif "cab_fare" in tasks:
        # The call was made and produced no usable price. Two very different things hide
        # behind that, and they need OPPOSITE fixes, so they no longer share one line:
        #
        #   • raw_fare is None            → the HTTP call itself failed and _safe_call
        #     swallowed it. An expired/wrong-environment rider JWT (401/403), an
        #     unreachable fare host, or a timeout all land here. The fix is the token or
        #     the network — nothing about this trip.
        #   • raw_fare is a FareQuote     → the backend answered but priced no category we
        #     could use (base_fare_inr 0, no alternatives). The fix is the backend data or
        #     the group, not the token.
        #
        # They were one `cab_fare_unavailable` line reading "call failed or returned no
        # priced categories", which sent whoever read it to guess which half it was. The
        # distinguishing bit is the type of raw_fare, available only here — `_package_fare`
        # collapses both to None on the next line, so it is captured before that.
        if raw_fare is None:
            log.warning(
                "cab_fare_call_failed",
                reason="fare backend rejected the request, was unreachable, or timed out",
                backend=settings.java_backend_url,
                origin=entities.origin or entities.home_city,
                destination=destination,
                session_id=state.session_id,
            )
        else:
            log.warning(
                "cab_fare_backend_empty",
                reason="backend answered but priced no usable cab category",
                base_fare_inr=getattr(raw_fare, "base_fare_inr", None),
                options=len(getattr(raw_fare, "alternatives", None) or []),
                origin=entities.origin or entities.home_city,
                destination=destination,
                session_id=state.session_id,
            )

    # ── Offers (Him Cab backend) ──────────────────────────────────────────────
    # Fetched after the fare, because whether a coupon applies depends on what the ride
    # costs. The trip type comes from the quote that was actually priced, so a LOCAL
    # hop is never offered an OUTSTATION-only coupon.
    trip_type = (cab_fare or {}).get("trip_type") or (
        quote_input.trip_type if quote_input else "ALL"
    )
    offers = await _fetch_offers(
        state, trip_type=trip_type, fare_inr=(cab_fare or {}).get("fare_inr")
    )
    if offers and cab_fare:
        best = offers[0]
        cab_fare["best_offer"] = best
        cab_fare["fare_after_offer_inr"] = best["fare_after_offer_inr"]

    log.info(
        "tools_completed",
        destination=destination,
        weather_ok=weather_obj is not None,
        has_cab_fare=cab_fare is not None,
        route_km=(route_estimate or {}).get("distance_km"),
        offers=len(offers),
        visitor_type=state.visitor_type,
        session_id=state.session_id,
    )

    return {
        "weather_data": weather_obj.model_dump(mode="json") if weather_obj else None,
        "snowfall_data": snowfall_data,
        "cab_fare": cab_fare,
        "offers": offers,
        "route_estimate": route_estimate,
        "node_trace": ["tool_executor"],
    }


async def _fetch_offers(
    state: AgentState, trip_type: str, fare_inr: Optional[float]
) -> list[dict]:
    """The rider's live coupons that apply to this trip, best saving first.

    Returns [] for every failure mode — flag off, no token, backend down, nothing
    applicable. The reply may only name what is in this list, so an empty list means
    the assistant says nothing about discounts, which is the correct behaviour when we
    cannot prove one exists.
    """
    offer_gate = {
        "enable_offers": settings.enable_offers,
        "has_rider_jwt": bool(state.rider_jwt),
        "pricing_intent": slots.response_mode(state.intent, state.visitor_type) in (
            slots.ITINERARY_MODE, slots.RIDE_MODE
        ),
    }
    if not all(offer_gate.values()):
        log.info("offers_gate_blocked", session_id=state.session_id, **offer_gate)
        return []

    raw = await _safe_call(
        get_offers_service().get_offers(
            rider_jwt=state.rider_jwt, trip_type=trip_type, rider_id=state.rider_id
        ),
        "offers",
    )
    if not raw:
        return []

    applicable = usable_offers(raw, trip_type, fare_inr)[: settings.max_offers_in_reply]
    return [_offer_payload(o, fare_inr) for o in applicable]


def _offer_payload(offer: Offer, fare_inr: Optional[float]) -> dict:
    """One coupon, as the app renders it and the prompt quotes it.

    `fare_after_offer_inr` is computed here rather than left to the writer: a discount
    the model does the arithmetic on is a discount the model can get wrong, and this is
    a number the rider will check against their receipt. It is None whenever we cannot
    state it honestly:

      • no fare came back, so there is nothing to subtract from
      • the coupon is worth at least the whole fare. `discount_on` caps at the fare, so
        the arithmetic would read "Rs 0" — a free ride the Spring booking flow will not
        honour. Quote the saving, promise no total.

    `applies_to_this_fare` tells the writer and the app whether eligibility was actually
    checked. Without a fare, the minimum-spend filter cannot run, so a coupon here is
    live on the account rather than confirmed for this trip.
    """
    verified = fare_inr is not None or offer.minimum_value_inr <= 0
    discount = offer.discount_on(fare_inr) if fare_inr is not None else 0.0
    payable = (
        round(fare_inr - discount)
        if fare_inr is not None and 0 < discount < fare_inr
        else None
    )
    return {
        "offer_id": offer.offer_id,
        "coupon_code": offer.coupon_code,
        "coupon_name": offer.coupon_name,
        "coupon_value_inr": offer.coupon_value_inr,
        "minimum_value_inr": offer.minimum_value_inr,
        "fare_after_offer_inr": payable,
        "applies_to_this_fare": verified,
        "terms_and_conditions": offer.terms_and_conditions,
        "image": offer.image,
        "expire_date": offer.expire_date.isoformat() if offer.expire_date else None,
    }
