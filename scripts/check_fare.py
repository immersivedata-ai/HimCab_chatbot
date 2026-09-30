"""
scripts/check_fare.py — diagnose the Him Cab fare and offer path.

The agent quotes a price only when every step below succeeds, so when a reply comes
back with no cabs this script tells you WHICH step failed, without reproducing it
through the chat:

    place geocoded → journey resolved → backend reachable → categories parsed
                                                          → cabs seated → coupons filtered

Modes:

  # 1. End-to-end against the real backend (needs a rider JWT)
  python scripts/check_fare.py --jwt "<RIDER_JWT>" --origin Palampur --destination Kasol --group-size 4
  python scripts/check_fare.py --jwt "<JWT>" --distance-km 177 --trip OUTSTATION --group-size 4

  # 2. From a GPS fix, the way the app sends it
  python scripts/check_fare.py --jwt "<JWT>" --lat 32.11 --lon 76.53 --destination Kasol

  # 3. Offline: no JWT, no network — resolve the journey and show the request we'd send
  python scripts/check_fare.py --dry-run --origin Palampur --destination Kasol

The JWT can also come from HIMCAB_RIDER_JWT, and the base URL from JAVA_BACKEND_URL
(default http://34.131.37.50:9002).

NOTE: this used to import app.integrations.himcab.distances (resolve_route,
resolve_route_from_fix, ROAD_ROUTES) and a build_fare_quote_input(route=...) signature —
none of which exist any more, so every invocation died on import. It now uses the SAME
resolution the agent uses at runtime: geo.resolve_journey and build_fare_quote_input,
so what it prints is what a real turn would actually send and receive.
"""

import argparse
import asyncio
import json
import os
import sys

# Ensure the app package is importable (matches scripts/seed_data.py convention)
sys.path.insert(0, ".")


async def report_offers(quote, args, jwt: str) -> None:
    """Fetch the rider's coupons and show which survive the filter, and why.

    "The chat never mentions my offer" has four possible causes — the flag, the trip
    type, the date window and the minimum spend — and they are indistinguishable from
    the reply, which simply says nothing. So each is printed per coupon.
    """
    from app.core.config import settings
    from app.integrations.himcab.offers_service import get_offers_service, usable_offers

    if not settings.enable_offers:
        print("  ENABLE_OFFERS=false — the agent will never mention a coupon.")
        return

    service = get_offers_service()
    try:
        offers = await service.get_offers(
            rider_jwt=jwt, trip_type=quote.trip_type, rider_id=args.rider_id
        )
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
        print("  The agent degrades silently here — the reply just omits discounts.")
        return
    finally:
        await service.close()

    if args.rider_id is None:
        print("  (no --rider-id: used the unpersonalised catalogue, filtered locally)")
    if not offers:
        print("  The backend returned no usable coupons.")
        return

    fare = quote.base_fare_inr
    keep = {o.coupon_code for o in usable_offers(offers, quote.trip_type, fare)}
    for offer in offers:
        if offer.coupon_code in keep:
            after = fare - offer.discount_on(fare)
            print(f"  KEPT    {offer.coupon_code:<16} -Rs {offer.coupon_value_inr:,.0f}"
                  f"  -> Rs {after:,.0f}")
            continue
        if not offer.applies_to(quote.trip_type):
            why = f"trip type {offer.trip_types} != {quote.trip_type}"
        elif not offer.is_live():
            why = f"outside its window ({offer.live_date} .. {offer.expire_date})"
        else:
            why = f"needs a fare over Rs {offer.minimum_value_inr:,.0f}"
        print(f"  dropped {offer.coupon_code:<16} {why}")

    shown = min(len(keep), settings.max_offers_in_reply)
    print(f"  -> {shown} of {len(keep)} eligible coupon(s) reach the reply "
          f"(MAX_OFFERS_IN_REPLY={settings.max_offers_in_reply})")


async def run(args) -> None:
    """Resolve the journey the way the agent does, then quote it against the backend."""
    from app.core.config import settings
    from app.domain.conversation import ExtractedEntities
    from app.integrations.himcab.fare_service import get_fare_service
    from app.integrations.himcab.models import FareQuoteInput
    from app.integrations.himcab.quote_builder import build_fare_quote_input, classify_trip_type
    from app.services import geo

    # ── Step 0: the GPS fix, when one was given ───────────────────────────────
    fix = None
    if args.lat is not None and args.lon is not None:
        fix = await geo.locate(args.lat, args.lon)
        print("=== 0. GPS FIX ===")
        if fix is None:
            print(f"  ({args.lat}, {args.lon}) -> UNUSABLE — the agent would ignore it "
                  f"and ask which city they are travelling from")
        else:
            print(f"  ({fix.lat}, {fix.lon}) -> {fix.place_name or 'no town within reach'}"
                  f" ({fix.nearest_city_km} km away)")
            print(f"  in Himachal : {fix.in_himachal}  -> persona '{fix.visitor_type}'")
            print(f"  precise     : {fix.is_precise}")
        print()

    # ── Step 1: the journey (geocoding + routing, exactly as the agent resolves it) ──
    print("=== 1. JOURNEY ===")
    journey = None
    if fix is not None:
        journey = await geo.resolve_journey(args.destination, origin_fix=fix)
    elif args.origin:
        journey = await geo.resolve_journey(args.destination, origin_name=args.origin)
    else:
        print("  (no origin and no fix — pass --origin/--destination or --lat/--lon)")

    if journey is not None:
        print(f"  {journey.origin.name!r} -> {journey.destination.name!r}")
        print(f"  {round(journey.distance_km)} km, {journey.duration_text} drive")
    elif args.origin or fix is not None:
        print(f"  NOT RESOLVED — {args.origin or 'the fix'!r} -> {args.destination!r} could "
              f"not be geocoded/routed. The agent would skip the quote and price nothing.")

    entities = ExtractedEntities(
        origin=(None if fix else args.origin),
        destination=args.destination,
        group_size=args.group_size,
        budget_range=args.budget,
    )

    # ── Step 2: build the request ─────────────────────────────────────────────
    # With a resolved journey, use the SAME builder the agent uses. With a --distance-km
    # override (or an unresolved journey), build it directly so a raw probe by distance
    # still works — this is the fastest way to ask "is the backend returning fares at all".
    if journey is not None and args.distance_km is None:
        quote_input = build_fare_quote_input(
            entities, journey, trip_type=args.trip,
            booking_time=args.booking_time, rent_hours=args.rent_hours,
        )
    else:
        dist = args.distance_km if args.distance_km is not None else (
            journey.distance_km if journey else None)
        quote_input = FareQuoteInput(
            origin=(journey.origin.name if journey else (args.origin or "")),
            destination=(journey.destination.name if journey else args.destination),
            distance_km=dist,
            trip_type=args.trip or classify_trip_type(dist),
            booking_time=args.booking_time,
            rent_hours=args.rent_hours,
            group_size=args.group_size,
            budget_range=args.budget,
        )

    service = get_fare_service()
    url = f"{settings.java_backend_url.rstrip('/')}{settings.himcab_fare_path}"
    print("\n=== 2. REQUEST ===")
    print("  URL       :", url)
    print("  trip type :", quote_input.trip_type)
    print("  body      :", json.dumps(service._build_quote_request(quote_input)))

    if quote_input.distance_km is None:
        print("\nNo distance — pass --distance-km, or use places the geocoder can place.")
        await service.close()
        return
    if args.dry_run:
        print("\n--dry-run: stopping before the network call.")
        await service.close()
        return

    jwt = args.jwt.strip()
    if jwt.lower().startswith("bearer "):
        jwt = jwt[7:].strip()   # tolerate a pasted "Bearer <token>" — we add the prefix
    if not jwt:
        print("\nProvide --jwt or set HIMCAB_RIDER_JWT (or use --dry-run).")
        await service.close()
        return

    # ── Step 3: the call ──────────────────────────────────────────────────────
    try:
        quote = await service.get_quote(quote_input, rider_jwt=jwt)
    except Exception as e:
        print("\n=== 3. CALL FAILED ===")
        print(f"  {type(e).__name__}: {e}")
        print("  In the app this is caught by tool_executor._safe_call and turned into None,")
        print("  so the ride reply ships with NO cabs and no error — the reported symptom.")
        print("  A 401/403 here means the rider token is invalid/expired for THIS backend;")
        print("  a connect error/timeout means the fare host is unreachable from here.")
        await service.close()
        return

    try:
        print("\n=== 3. RAW RESPONSE ===")
        print(json.dumps(quote.raw, indent=2)[:2000])

        print("\n=== 4. PARSED CATEGORIES ===")
        if not quote.alternatives:
            print("  NONE — the response had no priced categories, so the agent shows no "
                  "fare.\n  Compare the raw payload above against _extract_categories: it "
                  "needs data:[{...,\"totalFare\":<num>}] as a bare list.")
        for option in quote.alternatives:
            print(f"  {(option.category_name or '?'):<22} id={option.category_id} "
                  f"fare={option.total_fare_inr} seats={option.seat_capacity}")

        if quote.alternatives and args.group_size:
            from app.integrations.himcab.vehicles import select_cabs_for_group
            fits = select_cabs_for_group(quote.alternatives, args.group_size)
            print(f"\n=== 5. CABS FOR A PARTY OF {args.group_size} (seats >= {args.group_size}) ===")
            for option in fits:
                print(f"  {(option.category_name or '?'):<22} Rs {option.total_fare_inr:,.0f}"
                      f"  seats={option.seat_capacity}")

        print("\n=== 6. OFFERS ===")
        await report_offers(quote, args, jwt)

        print("\n=== 7. WHAT THE RIDER SEES ===")
        print(f"  headline: {quote.category_name} = Rs {quote.base_fare_inr:,.0f}")
    finally:
        await service.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the Him Cab fare and offer path")
    ap.add_argument("--jwt", default=os.getenv("HIMCAB_RIDER_JWT", ""), help="rider Spring JWT")
    ap.add_argument("--base-url", default=os.getenv("JAVA_BACKEND_URL"),
                    help="fare host, e.g. http://34.131.37.50:9002")
    ap.add_argument("--origin", default="Palampur")
    ap.add_argument("--destination", default="Kasol")
    ap.add_argument("--lat", type=float, default=None,
                    help="device latitude — prices from this point instead of --origin")
    ap.add_argument("--lon", type=float, default=None, help="device longitude (with --lat)")
    ap.add_argument("--rider-id", type=int, default=None,
                    help="numeric Spring rider id, to personalise the offer lookup")
    ap.add_argument("--distance-km", type=float, default=None,
                    help="override the resolved distance (probe the backend by distance alone)")
    ap.add_argument("--trip", default=None, choices=["OUTSTATION", "LOCAL", "RENTAL"],
                    help="override the trip type (default: derived from the distance)")
    ap.add_argument("--booking-time", default="12:00")
    ap.add_argument("--rent-hours", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--budget", default=None, choices=[None, "budget", "moderate", "luxury"])
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the journey and build the request, but don't call the backend")
    args = ap.parse_args()

    if args.base_url:
        # Must happen BEFORE app.core.config is imported (cached singleton), which is why
        # run() imports settings lazily rather than this module importing it at the top.
        os.environ["JAVA_BACKEND_URL"] = args.base_url

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
