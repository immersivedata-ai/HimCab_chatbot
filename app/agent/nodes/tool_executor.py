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
from app.core.logging import get_logger
from app.models.session import ExtractedEntities
from app.services import geo
from app.services.external.weather import get_weather_service
from app.services.himcab import (
    build_fare_quote_input,
    get_fare_service,
    get_offers_service,
    usable_offers,
)
from app.services.himcab.models import FareQuote, Offer

log = get_logger(__name__)

# Per-tool timeout — prevents a single slow API from stalling the entire pipeline
TOOL_TIMEOUT_SECONDS = 20


async def _safe_call(coro_or_value: Any, label: str) -> Any:
    """
    Wrapper that:
      1. Validates the argument is actually awaitable (catches sync/async mismatch bugs)
      2. Enforces a per-tool timeout
      3. Catches exceptions and logs them without failing the whole node
    """
    try:
        # Guard: if someone passes a non-coroutine (sync result), return it directly
        if not asyncio.iscoroutine(coro_or_value) and not asyncio.isfuture(coro_or_value):
            log.debug("tool_sync_result", tool=label)
            return coro_or_value

        result = await asyncio.wait_for(coro_or_value, timeout=TOOL_TIMEOUT_SECONDS)
        return result

    except asyncio.TimeoutError:
        log.warning(
            "tool_call_timeout",
            tool=label,
            timeout_seconds=TOOL_TIMEOUT_SECONDS,
        )
        return None

    except Exception as e:
        log.warning(
            "tool_call_failed",
            tool=label,
            error=str(e),
            error_type=type(e).__name__,
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


async def tool_executor_node(state: AgentState) -> dict:
    """
    LangGraph Tool Executor Node.
    Dispatches the external tool calls in parallel with per-tool timeouts.
    """
    log.info("tool_executor_entered", session_id=state.session_id)

    entities: ExtractedEntities = state.entities
    destination = entities.destination

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
    # Gated: flag on, a usable pickup point, pricing-relevant intent, and a forwarded
    # rider JWT. Any failure degrades gracefully to None via _safe_call → no price.
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
    should_quote = all(fare_gate.values())
    quote_input = None
    if not should_quote:
        # Log exactly which condition blocked it — this is the single most useful line
        # for diagnosing "why didn't I get a fare" without needing more round-trips.
        log.info("fare_gate_blocked", intent=intent, session_id=state.session_id, **fare_gate)
    else:
        # trip_type comes from the journey's length, not the rider's persona — a
        # Himachal resident driving 270 km is still an intercity transfer. The route is
        # passed in so the price is built on the same distance the reply quotes.
        quote_input = build_fare_quote_input(entities, journey) if journey else None
        # distanceInMeters is required by the endpoint, so no resolved journey means no
        # quote. That happens when the geocoder cannot place a name or the routing
        # engine finds no road — both are logged below rather than guessed around.
        if quote_input is not None and quote_input.distance_km is not None:
            tasks["cab_fare"] = _safe_call(
                get_fare_service().get_quote(quote_input, rider_jwt=state.rider_jwt),
                "cab_fare",
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
    results = dict(zip(task_keys, raw_results))

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
    cab_fare = None
    fq = results.get("cab_fare")
    if isinstance(fq, FareQuote) and fq.base_fare_inr > 0:
        cab_fare = {
            "fare_inr": round(fq.base_fare_inr),
            "currency": fq.currency,
            "category_id": fq.category_id,
            "category_name": fq.category_name,
            "vehicle_label": fq.vehicle_label,
            "seat_capacity": fq.seat_capacity,
            "trip_type": fq.trip_type,
            "distance_km": fq.distance_km,
            # Journey context, so a fare card can show what the price is FOR without
            # the app having to join it against the route event itself.
            "duration_text": (route_estimate or {}).get("duration_text"),
            "options": [a.model_dump() for a in fq.alternatives],  # all categories for the app
            "source": "himcab_api",
        }
    elif "cab_fare" in tasks:
        # The call was made and produced nothing usable. Distinguishing this from
        # "never attempted" is the difference between chasing a gate and chasing the
        # backend, so it gets its own line.
        log.warning(
            "cab_fare_unavailable",
            reason="call failed or returned no priced categories",
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
