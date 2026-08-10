"""
scripts/check_fare.py — diagnose the Him Cab fare path.

The agent quotes a fare only when every step below succeeds, so when a price is
missing from an itinerary this script tells you WHICH step failed, without needing to
reproduce it through the chat:

    place recognised → distance resolved → backend reachable → categories parsed

Two modes:

  # 1. End-to-end against the real backend (needs a rider JWT)
  python scripts/check_fare.py --jwt "<RIDER_JWT>" --origin Gurgaon --destination Manali
  python scripts/check_fare.py --jwt "<JWT>" --distance-km 300 --group-size 8 --budget luxury

  # 2. Offline: no JWT, no network — just show what distance we would price on
  python scripts/check_fare.py --dry-run --origin Solan --destination Kasol

  # 3. Re-derive the road-winding factors in app/services/geo.py from the surveyed
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnose the Him Cab fare estimate path")
    ap.add_argument("--jwt", default=os.getenv("HIMCAB_RIDER_JWT", ""), help="rider Spring JWT")
    ap.add_argument("--base-url", default=os.getenv("JAVA_BACKEND_URL"),
                    help="fare host, e.g. http://34.131.37.50:9002")
    ap.add_argument("--origin", default="Delhi")
    ap.add_argument("--destination", default="Manali")
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
    from app.services.geo import find_city
    from app.services.himcab.distances import resolve_route
    from app.services.himcab.fare_service import get_fare_service
    from app.services.himcab.quote_builder import build_fare_quote_input

    # ── Step 1: are both places recognised? ───────────────────────────────────
    print("=== 1. PLACES ===")
    ok = True
    for label, name in (("origin", args.origin), ("destination", args.destination)):
        found = find_city(name)
        if found:
            print(f"  {label:<12} {name!r} -> {found[0]!r} {found[1]}")
        else:
            print(f"  {label:<12} {name!r} -> NOT RECOGNISED "
                  f"(add it to app/services/geo.py CITY_COORDS)")
            ok = False

    # ── Step 2: distance ──────────────────────────────────────────────────────
    print("\n=== 2. DISTANCE ===")
    resolved = resolve_route(args.origin, args.destination)
    if resolved:
        kind = "ESTIMATED from coordinates" if resolved.is_approximate else "SURVEYED"
        print(f"  {resolved.km} km, {resolved.duration_text} drive  ({kind})")
    else:
        print("  none — no fare can be quoted for this route")

    entities = ExtractedEntities(
        origin=args.origin,
        destination=args.destination,
        group_size=args.group_size,
        budget_range=args.budget,
    )
    quote_input = build_fare_quote_input(
        entities, trip_type=args.trip, booking_time=args.booking_time,
        rent_hours=args.rent_hours,
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

        print("\n=== 6. WHAT THE RIDER SEES ===")
        print(f"  headline: {quote.category_name} = Rs {quote.base_fare_inr:,.0f}")
        print(f"  presented as: {'APPROXIMATE' if quote.distance_is_approximate else 'exact'}")

    asyncio.run(run())
    if not ok:
        print("\nNote: an unrecognised place above means the agent would skip the quote.")


if __name__ == "__main__":
    main()
