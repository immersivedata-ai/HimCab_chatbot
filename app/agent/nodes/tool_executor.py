"""
app/agent/nodes/tool_executor.py
──────────────────────────────────
Tool Executor Node — calls all external tools IN PARALLEL using asyncio.gather().
Responsibilities:
  1. Fetch weather + snowfall data
  2. Fetch relevant POIs (hotels, temples, cafes, restaurants, attractions)
  3. Fetch drive time estimates
  4. Fetch crowd predictions
  5. Generate personalised offers (frequent traveller / local / seasonal)
  6. Populate state with all tool results

All tools are called concurrently — total latency = slowest single call, not sum.
Failed tools degrade gracefully and return None (not an error).
Individual tool calls are wrapped with a timeout to prevent the pipeline from stalling.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.models.session import ExtractedEntities
from app.services.external.places import get_places_service
from app.services.external.weather import get_weather_service
from app.services.offers import get_offers_service
from app.services.redis_cache import get_redis_cache
from app.models.booking import TripDTORequest, DirectionDTO
from app.services.trip_booking import book_trip_api

log = get_logger(__name__)

# Which place categories to fetch for a full trip plan
FULL_TRIP_CATEGORIES = ["hotels", "restaurants", "temples", "cafes", "attractions"]
PARTIAL_CATEGORIES = ["attractions", "restaurants"]

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


def _generate_offers_sync(state: AgentState) -> dict:
    """
    Synchronous helper to generate personalised offers.
    Runs in ~0ms (no API calls) — safe for parallel execution.
    """
    try:
        offers_svc = get_offers_service()
        visitor_type = state.visitor_type or state.entities.visitor_type
        bundle = offers_svc.generate_offers(
            user_profile=state.user_profile,
            entities=state.entities,
            visitor_type=visitor_type,
        )
        return bundle.to_display_dict()
    except Exception as e:
        log.warning("offers_generation_failed", error=str(e))
        return {"offers": [], "loyalty_tier": "new", "total_trips": 0, "best_discount": 0.0}


async def _async_generate_offers(state: AgentState) -> dict:
    """Thin async wrapper around the sync offers generator."""
    return _generate_offers_sync(state)


async def tool_executor_node(state: AgentState) -> dict:
    """
    LangGraph Tool Executor Node.
    Dispatches all external tool calls in parallel with per-tool timeouts.
    Also generates personalised offers for frequent travellers and local users.
    """
    log.info("tool_executor_entered", session_id=state.session_id)

    entities: ExtractedEntities = state.entities
    destination = entities.destination

    if not destination:
        log.warning("tool_executor_no_destination")
        return {"node_trace": ["tool_executor"]}

    weather_svc = get_weather_service()
    places_svc = get_places_service()
    cache = get_redis_cache()

    intent = state.intent or "plan_trip"
    is_full_plan = intent == "plan_trip"

    # ── Define all parallel tasks ─────────────────────────────────────────────

    tasks: dict[str, Any] = {
        "weather": _safe_call(
            weather_svc.get_weather(destination),
            "weather",
        ),
        "snowfall": _safe_call(
            weather_svc.get_snowfall_risk(destination),
            "snowfall",
        ),
    }

    # Determine which place categories to fetch based on intent
    # if is_full_plan:
    #     categories = FULL_TRIP_CATEGORIES
    # elif intent == "hotel_query":
    #     categories = ["hotels"]
    # elif intent == "places_query":
    #     categories = PARTIAL_CATEGORIES
    # else:
    #     categories = ["attractions"]

    categories = []  # Overpass removed for now
    
    # for category in categories:
    #     tasks[f"places_{category}"] = _safe_call(
    #         places_svc.get_places(destination, category),
    #         f"places_{category}",
    #     )

    # Drive time from origin — ALWAYS fetch when origin is available
    # (critical for outstation visitors regardless of intent)
    # if entities.origin:
    #     tasks["drive_time"] = _safe_call(
    #         places_svc.get_drive_time(entities.origin, destination),
    #         "drive_time",
    #     )
    #     # For outstation visitors, also compute the full route plan (origin → dest → origin)
    #     if state.visitor_type == "outstation" or (entities.visitor_type == "outstation"):
    #         tasks["route_plan"] = _safe_call(
    #             places_svc.get_optimal_route(
    #                 origin=entities.origin,
    #                 destination=destination,
    #                 return_to_origin=True,
    #             ),
    #             "route_plan",
    #         )

    # Crowd data from Redis (precomputed by workers)
    current_month = datetime.now(timezone.utc).strftime("%B").lower()
    # tasks["crowd"] = _safe_call(
    #     cache.get_crowd_data(destination, current_month),
    #     "crowd",
    # )

    # Offers generation — runs synchronously (~0ms, no API calls)
    # Wrapped in _safe_call so failures are non-fatal
    tasks["offers"] = _safe_call(
        _async_generate_offers(state),
        "offers",
    )

    # ── Booking execution ─────────────────────────────────────────────────────
    if state.user_confirmed_booking and state.booking_details:
        pickup = state.booking_details.get("pickup", "")
        dropoff = state.booking_details.get("dropoff", "")
        time = state.booking_details.get("time", "12:00:00")
        if len(time) == 5:
            time += ":00" # HH:mm to HH:mm:ss
        
        trip_type = "LOCAL" if state.visitor_type == "local" else "OUTSTATION"
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d") # Use today as default if not in details
        
        request = TripDTORequest(
            fromDirectionAddress=DirectionDTO(address=pickup),
            toDirectionAddress=[DirectionDTO(address=dropoff)],
            date=date,
            bookingTime=time,
            tripType=trip_type,
            riderId=1 # mocked for now
        )
        tasks["booking_result"] = _safe_call(
            book_trip_api(request),
            "booking_result",
        )

    # ── Execute all in parallel ───────────────────────────────────────────────
    task_keys = list(tasks.keys())
    task_coroutines = list(tasks.values())
    raw_results = await asyncio.gather(*task_coroutines, return_exceptions=False)
    results = dict(zip(task_keys, raw_results))

    # ── Build tool success/failure summary for debugging ──────────────────────
    succeeded = [k for k, v in results.items() if v is not None]
    failed = [k for k, v in results.items() if v is None]
    if failed:
        log.warning(
            "tools_partial_failure",
            succeeded=succeeded,
            failed=failed,
            session_id=state.session_id,
        )

    # ── Package results into state ────────────────────────────────────────────
    weather_obj = results.get("weather")
    places_data: dict[str, list] = {}
    total_places = 0
    for category in categories:
        result = results.get(f"places_{category}")
        if result:
            places_data[category] = result
            total_places += len(result)

    drive_times = {}
    travel_estimate = None
    if results.get("drive_time") and entities.origin:
        drive_info = results["drive_time"]
        drive_times[f"{entities.origin}_to_{destination}"] = drive_info
        # Build a dedicated travel_estimate for the writer node
        travel_estimate = {
            "origin": entities.origin,
            "destination": destination,
            "duration_minutes": drive_info.get("duration_minutes"),
            "duration_hours": drive_info.get("duration_hours"),
            "distance_km": drive_info.get("distance_km"),
            "recommended_mode": drive_info.get("recommended_mode", "cab"),
            "note": drive_info.get("note"),
        }

    crowd_data = {}
    if results.get("crowd"):
        crowd_data[destination] = results["crowd"]

    # Route plan for outstation visitors
    route_plan = results.get("route_plan")

    # Offers bundle
    offers_result = results.get("offers")
    available_offers = []
    if offers_result and isinstance(offers_result, dict):
        available_offers = offers_result.get("offers", [])

    log.info(
        "tools_completed",
        destination=destination,
        weather_ok=weather_obj is not None,
        categories_fetched=list(places_data.keys()),
        total_places=total_places,
        has_travel_estimate=travel_estimate is not None,
        has_route_plan=route_plan is not None,
        offers_count=len(available_offers),
        visitor_type=state.visitor_type,
        tools_succeeded=len(succeeded),
        tools_failed=len(failed),
        session_id=state.session_id,
    )

    # Extract Booking
    booking_result = results.get("booking_result")
    final_booking_id = None
    if booking_result and not booking_result.get("error"):
        final_booking_id = booking_result.get("id")

    return {
        "weather_data": weather_obj.model_dump(mode="json") if weather_obj else None,
        "snowfall_data": results.get("snowfall"),
        "places_data": places_data,
        "drive_times": drive_times,
        "travel_estimate": travel_estimate,
        "route_plan": route_plan,
        "crowd_data": crowd_data,
        "available_offers": available_offers,
        "final_booking_id": str(final_booking_id) if final_booking_id else None,
        "node_trace": ["tool_executor"],
    }
