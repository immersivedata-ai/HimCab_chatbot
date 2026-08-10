"""
scripts/check_fare.py — diagnose the Him Cab fare and offer path.

The agent quotes a price only when every step below succeeds, so when one is missing
from a reply this script tells you WHICH step failed, without needing to reproduce it
through the chat:

    place recognised → distance resolved → backend reachable → categories parsed
                                                             → coupons filtered

Modes:

  # 1. End-to-end against the real backend (needs a rider JWT)
  python scripts/check_fare.py --jwt "<RIDER_JWT>" --origin Gurgaon --destination Manali
  python scripts/check_fare.py --jwt "<JWT>" --distance-km 300 --group-size 8 --budget luxury

  # 2. From a GPS fix, the way the app sends it — shows the persona it routes to and
  #    the distance measured from that exact point
  python scripts/check_fare.py --dry-run --lat 32.32 --lon 77.17 --destination Kasol

  # 3. Offline: no JWT, no network — just show what distance we would price on
  python scripts/check_fare.py --dry-run --origin Solan --destination Kasol

  # 4. Re-derive the road-winding factors in app/services/geo.py from the surveyed
  #    routes in app/services/himcab/distances.py, and report the estimator's error
  python scripts/check_fare.py --calibrate

The JWT can also come from HIMCAB_RIDER_JWT, and the base URL from JAVA_BACKEND_URL.
"""

import argparse
import asyncio
import json
import os
import statistics
import sys

# Ensure the app package is importable (matches scripts/seed_data.py convention)
sys.path.insert(0, ".")


def calibrate() -> None:
    """Fit the great-circle → road multipliers to the surveyed distances."""
    from app.services.geo import (
        HIMACHAL_COORDS,
        estimate_road_distance_km,
        find_city,
        haversine_km,
    )
    from app.services.himcab.distances import ROAD_ROUTES

    unknown, samples = set(), []
    for (origin, destination), (surveyed_km, _minutes) in ROAD_ROUTES.items():
        a, b = find_city(origin), find_city(destination)
        if a is None:
            unknown.add(origin)
            continue
        if b is None:
            unknown.add(destination)
            continue
        straight = haversine_km(a[1], b[1])
        if straight < 1:
            continue
        in_hp = (a[0] in HIMACHAL_COORDS, b[0] in HIMACHAL_COORDS)
        band = "hills" if all(in_hp) else ("mixed" if any(in_hp) else "plains")
        samples.append((band, straight, surveyed_km / straight, origin, destination))

    if unknown:
        print("!! places with no coordinates (add to geo.CITY_COORDS):", sorted(unknown))

    print(f"\n=== OBSERVED ROAD/STRAIGHT-LINE RATIO  (n={len(samples)}) ===")
    for band in ("hills", "mixed", "plains"):
        rows = [s for s in samples if s[0] == band]
        if not rows:
            print(f"  {band:7s}  no surveyed routes")
            continue
        print(f"  {band}:")
        for lo, hi in ((0, 120), (120, 300), (300, float("inf"))):
            ratios = [r[2] for r in rows if lo <= r[1] < hi]
            if ratios:
                print(f"    straight-line {lo:>4}-{hi if hi != float('inf') else '+':<5} km  "
                      f"n={len(ratios):3d}  median={statistics.median(ratios):.2f}")

    errors = []
    for _, _, _, origin, destination in samples:
        estimated = estimate_road_distance_km(origin, destination)
        surveyed = ROAD_ROUTES.get(
            (origin.lower(), destination.lower())
        ) or ROAD_ROUTES.get((destination.lower(), origin.lower()))
        if estimated and surveyed:
            surveyed_km = surveyed[0]
            errors.append((abs(estimated - surveyed_km) / surveyed_km * 100, origin,
                           destination, surveyed_km, estimated))

    if errors:
        pct = sorted(e[0] for e in errors)
        print("\n=== CURRENT ESTIMATOR ERROR vs SURVEYED ===")
        print(f"  median {statistics.median(pct):.1f}%   mean {statistics.mean(pct):.1f}%   "
              f"within 20%: {sum(1 for e in pct if e <= 20)}/{len(pct)}")
        print("  worst 5:")
        for err, origin, destination, surveyed, estimated in sorted(errors, reverse=True)[:5]:
            print(f"    {origin:<14}-> {destination:<16} surveyed={surveyed:>6.0f} "
                  f"estimated={estimated:>6.1f}  ({err:.0f}% off)")
        print("\n  Surveyed distances always win at runtime; this error applies only to the "
              "routes\n  that are NOT in the table, and those fares are labelled approximate.")


async def report_offers(quote, args, jwt: str) -> None:
    """Fetch the rider's coupons and show which survive the filter, and why.

    "The chat never mentions my offer" has four possible causes — the flag, the trip
    type, the date window and the minimum spend — and they are indistinguishable from
    the reply, which simply says nothing. So each is printed per coupon.
    """
    from app.core.config import settings
    from app.services.himcab.offers_service import get_offers_service, usable_offers

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


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the Him Cab fare and offer path")
    ap.add_argument("--jwt", default=os.getenv("HIMCAB_RIDER_JWT", ""), help="rider Spring JWT")
    ap.add_argument("--base-url", default=os.getenv("JAVA_BACKEND_URL"),
                    help="fare host, e.g. http://34.131.37.50:9002")
    ap.add_argument("--origin", default="Delhi")
    ap.add_argument("--destination", default="Manali")
    ap.add_argument("--lat", type=float, default=None,
                    help="device latitude — prices from this point instead of --origin")
    ap.add_argument("--lon", type=float, default=None, help="device longitude (with --lat)")
    ap.add_argument("--rider-id", type=int, default=None,
                    help="numeric Spring rider id, to personalise the offer lookup")
    ap.add_argument("--distance-km", type=float, default=None,
                    help="override the resolved distance")
    ap.add_argument("--trip", default=None, choices=["OUTSTATION", "LOCAL", "RENTAL"],
                    help="override the trip type (default: derived from the distance)")
    ap.add_argument("--booking-time", default="12:00")
    ap.add_argument("--rent-hours", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--budget", default=None, choices=[None, "budget", "moderate", "luxury"])
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve the distance and build the request, but don't call the backend")
    ap.add_argument("--calibrate", action="store_true",
                    help="re-derive the road-winding factors from the surveyed routes")
    args = ap.parse_args()

    if args.base_url:
        # Must happen BEFORE app.core.config is imported (cached singleton).
        os.environ["JAVA_BACKEND_URL"] = args.base_url

    if args.calibrate:
        calibrate()
        return

    from app.core.config import settings
    from app.models.session import ExtractedEntities
    from app.services import geo
    from app.services.geo import find_city
    from app.services.himcab.distances import resolve_route, resolve_route_from_fix
    from app.services.himcab.fare_service import get_fare_service
    from app.services.himcab.quote_builder import build_fare_quote_input

    fix = geo.locate(args.lat, args.lon) if args.lat is not None and args.lon is not None else None

    # ── Step 0: the GPS fix, when one was given ───────────────────────────────
    if args.lat is not None or args.lon is not None:
        print("=== 0. GPS FIX ===")
        if fix is None:
            print(f"  ({args.lat}, {args.lon}) -> UNUSABLE — the agent would ignore it "
                  f"and ask which city they are travelling from")
        else:
            print(f"  ({fix.lat}, {fix.lon}) -> {fix.place_name or 'no town within 60 km'}"
                  f" ({fix.nearest_city_km} km away)")
            print(f"  in Himachal : {fix.in_himachal}  -> persona '{fix.visitor_type}'"
                  f"{'  (NEAR THE BORDER — could go either way)' if fix.near_border else ''}")
            print(f"  precise     : {fix.is_precise}"
                  f"{'' if fix.is_precise else '  -> too vague to name as their pickup city'}")
        print()

    # ── Step 1: are both places recognised? ───────────────────────────────────
    print("=== 1. PLACES ===")
    ok = True
    places = (("destination", args.destination),) if fix else (
        ("origin", args.origin), ("destination", args.destination)
    )
    for label, name in places:
        found = find_city(name)
        if found:
            print(f"  {label:<12} {name!r} -> {found[0]!r} {found[1]}")
        else:
            print(f"  {label:<12} {name!r} -> NOT RECOGNISED "
                  f"(add it to app/services/geo.py CITY_COORDS)")
            ok = False

    # ── Step 2: distance ──────────────────────────────────────────────────────
    print("\n=== 2. DISTANCE ===")
    resolved = (
        resolve_route_from_fix(fix, args.destination, fallback_origin=args.origin)
        if fix else resolve_route(args.origin, args.destination)
    )
    if resolved:
        kind = {
            "surveyed": "SURVEYED (measured)",
            "gps": "COMPUTED from the rider's own fix",
            "city": "COMPUTED between city centres",
        }.get(resolved.source, resolved.source)
        print(f"  {resolved.km} km, {resolved.duration_text} drive  ({kind})")
    else:
        print("  none — no fare can be quoted for this route")

    entities = ExtractedEntities(
        # A precise fix names the pickup city; a vague one prices from the coordinates
        # but leaves the name to the rider, exactly as the agent does.
        origin=(fix.place_name if fix and fix.is_precise else (None if fix else args.origin)),
        destination=args.destination,
        group_size=args.group_size,
        budget_range=args.budget,
    )
    quote_input = build_fare_quote_input(
        entities, trip_type=args.trip, booking_time=args.booking_time,
        rent_hours=args.rent_hours, route=resolved,
    )
    if args.distance_km is not None:
        quote_input.distance_km = args.distance_km
        quote_input.distance_is_approximate = True

    service = get_fare_service()
    url = f"{settings.java_backend_url.rstrip('/')}{settings.himcab_fare_path}"
    print("\n=== 3. REQUEST ===")
    print("  URL       :", url)
    print("  trip type :", quote_input.trip_type)
    print("  body      :", json.dumps(service._build_quote_request(quote_input)))

    if quote_input.distance_km is None:
        raise SystemExit("\nNo distance — pass --distance-km, or add the place to geo.CITY_COORDS.")
    if args.dry_run:
        print("\n--dry-run: stopping before the network call.")
        return

    jwt = args.jwt.strip()
    if jwt.lower().startswith("bearer "):
        jwt = jwt[7:].strip()   # tolerate a pasted "Bearer <token>" — we add the prefix
    if not jwt:
        raise SystemExit("\nProvide --jwt or set HIMCAB_RIDER_JWT (or use --dry-run).")

    async def run() -> None:
        try:
            quote = await service.get_quote(quote_input, rider_jwt=jwt)
        finally:
            await service.close()

        print("\n=== 4. RAW RESPONSE ===")
        print(json.dumps(quote.raw, indent=2)[:2000])

        print("\n=== 5. PARSED CATEGORIES ===")
        if not quote.alternatives:
            print("  NONE — the response had no priced categories, so the agent will show "
                  "no fare.\n  Compare the raw payload above against _parse_quote_response().")
        for option in quote.alternatives:
            print(f"  {(option.category_name or '?'):<22} id={option.category_id} "
                  f"fare={option.total_fare_inr} seats={option.seat_capacity}")

        print("\n=== 6. OFFERS ===")
        await report_offers(quote, args, jwt)

        print("\n=== 7. WHAT THE RIDER SEES ===")
        print(f"  headline: {quote.category_name} = Rs {quote.base_fare_inr:,.0f}")
        print(f"  presented as: {'APPROXIMATE' if quote.distance_is_approximate else 'exact'}")

    asyncio.run(run())
    if not ok:
        print("\nNote: an unrecognised place above means the agent would skip the quote.")


if __name__ == "__main__":
    main()
