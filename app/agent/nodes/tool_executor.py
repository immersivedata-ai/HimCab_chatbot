"""
app/agent/nodes/tool_executor.py
──────────────────────────────────
Tool Executor Node — calls the external services IN PARALLEL using asyncio.gather().

Two tools run today:
  1. Weather + snowfall risk for the destination (one fetch; the risk is derived
     from that same report rather than fetched again)
  2. A real Him Cab fare quote, when the flag is on and a rider JWT was forwarded

It also resolves the journey itself (`_build_route_estimate`) from local tables — no
network call — because the distance, the drive time and the map coordinates all have
to agree with the fare that was quoted on them.

Both are called concurrently, so total latency is the slower of the two rather than
their sum. Each is wrapped in a timeout and degrades to None instead of raising —
a missing tool result costs detail in the reply, never the reply itself.
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
from app.services.himcab import build_fare_quote_input, get_fare_service
from app.services.himcab.distances import resolve_route
from app.services.himcab.models import FareQuote

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


def _build_route_estimate(entities: ExtractedEntities) -> Optional[dict]:
    """Resolve the journey once: both endpoints, their coordinates, distance and time.

    Coordinates are included because the app's booking screen needs to drop pins for
    the pickup and the drop — handing it two place NAMES would make it geocode them
    again and possibly land somewhere else than we priced.
    """
    origin = entities.origin or entities.home_city
    destination = entities.destination
    if not origin or not destination:
        return None

    route = resolve_route(origin, destination)
    if route is None:
        return None

    origin_place = geo.find_city(origin)
    destination_place = geo.find_city(destination)
    return {
        "origin": {
            "name": origin.title(),
            "lat": origin_place[1][0] if origin_place else None,
            "lon": origin_place[1][1] if origin_place else None,
        },
        "destination": {
            "name": destination.title(),
            "lat": destination_place[1][0] if destination_place else None,
            "lon": destination_place[1][1] if destination_place else None,
        },
        "distance_km": route.km,
        "duration_minutes": route.duration_minutes,
        "duration_text": route.duration_text,
        # Surveyed routes carry measured figures; the rest are derived, and the reply
        # has to say so rather than quoting them as fact.
        "is_approximate": route.is_approximate,
    }


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
    # Resolved once, from local tables, and reused three ways: it prices the fare,
    # it is the drive time quoted to a local rider, and its coordinates are what the
    # app drops on its map at the booking handoff.
    route_estimate = _build_route_estimate(entities)

    # ── Real cab fare (Him Cab backend) ───────────────────────────────────────
    # Gated: flag on, origin known, pricing-relevant intent, and a forwarded rider JWT.
    # Any failure degrades gracefully to None via _safe_call → no price is quoted.
    fare_gate = {
        "enable_real_fares": settings.enable_real_fares,
        # Locals often only give home_city (not origin) — either satisfies the gate.
        "has_origin": bool(entities.origin or entities.home_city),
        # Exactly the intents that produce a trip — the same predicate the writer uses,
        # so a new travel intent can never be added and silently lose its fare. A local
        # rider's whole reply IS the fare, so this must cover RIDE_MODE too.
        "pricing_intent": slots.response_mode(intent, state.visitor_type) in (
            slots.ITINERARY_MODE, slots.RIDE_MODE
        ),
        "has_rider_jwt": bool(state.rider_jwt),
    }
    should_quote = all(fare_gate.values())
    if not should_quote:
        # Log exactly which condition blocked it — this is the single most useful line
        # for diagnosing "why didn't I get a fare" without needing more round-trips.
        log.info("fare_gate_blocked", intent=intent, session_id=state.session_id, **fare_gate)
    else:
        # trip_type comes from the journey's length, not the rider's persona — a
        # Himachal resident driving 270 km is still an intercity transfer.
        quote_input = build_fare_quote_input(entities)
        # distanceInMeters is required by the endpoint. It is now available for any
        # pair of recognised places (surveyed or computed), so this only fires when a
        # place name isn't in the geography tables at all.
        if quote_input.distance_km is not None:
            tasks["cab_fare"] = _safe_call(
                get_fare_service().get_quote(quote_input, rider_jwt=state.rider_jwt),
                "cab_fare",
            )
        else:
            log.warning(
                "cab_fare_skipped_unknown_place",
                origin=entities.origin or entities.home_city,
                destination=destination,
                session_id=state.session_id,
                hint="add the place to app/services/geo.py CITY_COORDS",
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

    # Real cab fare — gross prices straight from /fare/estimate. No coupons/discounts are
    # applied here; the app's own booking flow handles real coupons.
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
            # The rider is told the fare is approximate when the distance behind it was
            # computed rather than surveyed.
            "is_approximate": fq.distance_is_approximate,
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

    log.info(
        "tools_completed",
        destination=destination,
        weather_ok=weather_obj is not None,
        has_cab_fare=cab_fare is not None,
        fare_approximate=bool(cab_fare and cab_fare["is_approximate"]),
        route_km=(route_estimate or {}).get("distance_km"),
        visitor_type=state.visitor_type,
        session_id=state.session_id,
    )

    return {
        "weather_data": weather_obj.model_dump(mode="json") if weather_obj else None,
        "snowfall_data": snowfall_data,
        "cab_fare": cab_fare,
        "route_estimate": route_estimate,
        "node_trace": ["tool_executor"],
    }
