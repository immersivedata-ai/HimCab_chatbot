"""
scripts/check_rental.py — diagnose why a rental has no price.

A rental quote only appears when EVERY step below succeeds, and each one fails
quietly today: the chat simply says nothing about a fare. This script runs the same
code the agent runs, against the real backend, and names the step that broke.

    rider JWT accepted -> the duration is one the chat offers -> POST /fare/estimate
      returns rows -> at least one row has a usable totalFare

The rate card (GET /fare/get-all) is checked first because it decides WHICH durations
are offered and how many kilometres each includes -- but it is NOT a precondition for
a price: /fare/estimate matches the rental chart from `rentHours` on its own.

Usage:

    # Everything: rate card, then a quote for every package on it
    python scripts/check_rental.py --jwt "<RIDER_JWT>"

    # Just one duration (the one your chat was stuck on)
    python scripts/check_rental.py --jwt "<RIDER_JWT>" --hours 8

    # Narrow the cab list the way the chat does, for a party of four
    python scripts/check_rental.py --jwt "<RIDER_JWT>" --hours 8 --group-size 4

    # See the untouched JSON, when the parse is what you suspect
    python scripts/check_rental.py --jwt "<RIDER_JWT>" --raw

The JWT can also come from HIMCAB_RIDER_JWT, and the base URL from JAVA_BACKEND_URL.
Nothing here is written to the repo — every package and price comes off the wire.
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, ".")

from app.core.config import settings                                    # noqa: E402
from app.services.himcab.fare_service import FareService                # noqa: E402
from app.services.himcab.quote_builder import (                         # noqa: E402
    build_rental_quote_input,
    matching_package,
)
from app.services.himcab.rental_service import (                        # noqa: E402
    RentalPackage,
    RentalService,
    distinct_packages,
)
from app.models.session import ExtractedEntities                        # noqa: E402

OK, BAD, WARN = "  OK  ", " FAIL ", " WARN "


def line(status: str, text: str) -> None:
    print(f"[{status}] {text}")


def blob(label: str, payload, limit: int = 1600) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if len(text) > limit:
        text = text[:limit] + f"\n  ... ({len(text)} chars total)"
    print(f"\n--- {label} ---\n{text}\n")


async def check_rate_card(jwt: str, show_raw: bool):
    """GET /fare/get-all -> the packages the assistant is allowed to offer."""
    print("=" * 78)
    print(f"1. RATE CARD   GET {settings.java_backend_url}{settings.himcab_fare_getall_path}")
    print("=" * 78)

    service = RentalService()
    token = jwt.strip()
    if token.lower().startswith("bearer "):
        token = token[len("bearer "):].strip()

    try:
        raw = await service._get(
            service._url,
            {"Accept": "application/json", "Authorization": f"Bearer {token}"},
        )
    except Exception as e:
        line(BAD, f"the request itself failed: {type(e).__name__}: {e}")
        print("\n  -> 401/403: the JWT is not accepted. 404: check himcab_fare_getall_path.")
        print("    Connection errors: check JAVA_BACKEND_URL and that the host is up.")
        await service.close()
        return None, []

    line(OK, "reachable and authorised")
    if show_raw:
        blob("raw /fare/get-all response", raw)

    rows = service._fare_rows(raw)
    if not rows:
        line(BAD, "no rate-card rows found in the response envelope")
        print("\n  -> The reader accepts `data` as a list, `data.content`, `data.fares`")
        print("    and `data.fareList`. Re-run with --raw and send me the output;")
        print("    the envelope needs adding to RentalService._fare_rows.")
        await service.close()
        return raw, []
    line(OK, f"{len(rows)} rate-card row(s) -- keys: {sorted(rows[0].keys())}")

    packages = distinct_packages(service._parse_packages(raw))
    if not packages:
        line(BAD, "rows found, but no rental charts could be read from them")
        print("\n  -> Each row needs `rentalChartList` (or rentalCharts /")
        print("    rental_chart_list) whose entries carry `maxHour`. Without an hour")
        print("    Without them the chat offers the configured durations instead and")
        print("    still asks for a price -- it just cannot say what each includes.")
        print("    Re-run with --raw so the real field names can be added.")
        await service.close()
        return raw, []

    line(OK, f"{len(packages)} chart(s) on the rate card: "
             + ", ".join(p.label for p in packages))
    missing_km = [p for p in packages if not p.included_km]
    if missing_km:
        line(WARN, f"no included km on: {', '.join(str(p.hours) + ' hrs' for p in missing_km)}")
        print("      distanceInMeters would be sent as 0 for these -- see step 2.")

    # The product's window on top of the backend's charts. A chart outside it is priced
    # by the backend and never offered by the chat, which is the difference between this
    # tool's output and what a rider sees — so it is stated rather than left implicit.
    offered = [
        p for p in packages
        if settings.rental_min_hours <= p.hours <= settings.rental_max_hours
    ]
    dropped = [p for p in packages if p not in offered]
    print(f"\n      bookable window: {settings.rental_min_hours}-"
          f"{settings.rental_max_hours} hours (rental_min_hours / rental_max_hours)")
    if dropped:
        line(WARN, "outside the window, so NEVER offered in chat: "
                   + ", ".join(p.label for p in dropped))
    if offered:
        line(OK, "offered in chat: " + ", ".join(p.label for p in offered))
    else:
        line(WARN, "nothing on the rate card is inside the bookable window")
        print("")
        print(f"  -> The chat offers the configured durations instead "
              f"({', '.join(str(h) for h in settings.rental_hour_options)} hrs).")
        print("    Step 2 shows whether the backend can price those. If it cannot,")
        print("    either widen the window (rental_min_hours / rental_max_hours) or")
        print("    configure charts for those hours on the backend.")

    await service.close()
    return raw, offered


async def check_estimate(jwt: str, packages, hours_wanted, group_size, show_raw: bool):
    """POST /fare/estimate -> the price the chat is allowed to show."""
    print()
    print("=" * 78)
    print(f"2. FARE        POST {settings.java_backend_url}{settings.himcab_fare_path}")
    print("=" * 78)

    if not packages:
        line(WARN, "no chart to read the included km from -- the chat offers the")
        print("      configured durations instead ("
              + ", ".join(str(h) for h in settings.rental_hour_options)
              + " hrs) and asks the backend to price whichever one the rider picks,")
        print("      since /fare/estimate matches the chart from rentHours itself.")
        print("      Checking those durations now -- exactly what the chat does.")
        packages = [
            RentalPackage(hours=h, from_rate_card=False)
            for h in settings.rental_hour_options
        ]

    targets = [h for h in (hours_wanted or [p.hours for p in packages])]
    service = FareService()
    any_priced = False

    for hours in targets:
        package = matching_package([p for p in packages if p.from_rate_card], hours)
        print(f"\n  -- {hours} hours " + "-" * 50)
        if package is None and not any(p.hours == hours for p in packages):
            line(BAD, f"{hours} is not offered -- the chat refuses it up front")
            print(f"      offered: {', '.join(str(p.hours) for p in packages)}")
            continue
        if package is None:
            line(WARN, "no rate-card chart for these hours on our side; asking anyway")

        entities = ExtractedEntities(origin="Shimla", rental_hours=hours,
                                     group_size=group_size)
        quote_input = build_rental_quote_input(entities, package)
        body = service._build_quote_request(quote_input)
        print(f"      request: {json.dumps(body)}")
        if body["distanceInMeters"] == 0:
            # Not a problem for a rental: the backend prices the package off the
            # chart it matches from rentHours (farePerKilometer x maxKilometer), and
            # charges extra distance at drop-off against the odometer.
            line(WARN, "distanceInMeters is 0 -- fine for a rental, the chart decides")

        try:
            quote = await service.get_quote(quote_input, rider_jwt=jwt)
        except Exception as e:
            line(BAD, f"the request failed: {type(e).__name__}: {e}")
            continue

        if show_raw:
            blob(f"raw /fare/estimate response ({hours} hrs)", quote.raw)

        rows = quote.raw.get("data") if isinstance(quote.raw, dict) else None
        if not isinstance(rows, list) or not rows:
            line(BAD, "the response carried no category rows")
            print("      -> nothing to price; re-run with --raw to see what came back.")
            continue
        line(OK, f"{len(rows)} category row(s) -- keys: {sorted(rows[0].keys())}")

        if not quote.alternatives:
            line(BAD, "rows came back but none had a usable totalFare")
            print("      -> _extract_categories reads `totalFare` and `name`. If the")
            print("        rows use other keys, that is the mismatch.")
            continue

        if quote.base_fare_inr <= 0:
            line(BAD, f"every price was zero (headline {quote.base_fare_inr})")
            print("      -> the agent DROPS a zero fare rather than showing Rs 0, which")
            print("        is why the chat shows no price. Check the rental chart's")
            print("        farePerKilometer / maxKilometer for this category.")
            continue

        any_priced = True
        line(OK, f"headline Rs {round(quote.base_fare_inr)} ({quote.category_name})")
        for alt in quote.alternatives:
            seats = f", {alt.seat_capacity} seats" if alt.seat_capacity else ""
            print(f"        - {alt.category_name}: Rs {alt.total_fare_inr}{seats}")
        if group_size:
            print(f"        (narrowed to cabs seating {group_size})")

    await service.close()

    print()
    print("=" * 78)
    if any_priced:
        print("VERDICT: the backend prices rentals correctly -- the chat will show these.")
        print("         If your UI still shows nothing, the page is almost certainly")
        print("         running on the default 'dev-test-token'. Open the Token box in")
        print("         the top bar, paste this same JWT, and hit 'Apply & restart'.")
    else:
        print("VERDICT: no usable price came back. The chat is right to stay silent --")
        print("         it never shows a fare it did not get. Fix the failing step above.")
    print("=" * 78)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the Him Cab rental fare path.")
    parser.add_argument("--jwt", default=os.getenv("HIMCAB_RIDER_JWT"),
                        help="Rider JWT (or set HIMCAB_RIDER_JWT).")
    parser.add_argument("--hours", type=int, action="append",
                        help="Only check this duration. Repeatable.")
    parser.add_argument("--group-size", type=int, default=None,
                        help="Narrow the cab list the way the chat does.")
    parser.add_argument("--raw", action="store_true",
                        help="Print the untouched JSON from both endpoints.")
    args = parser.parse_args()

    print(f"backend       : {settings.java_backend_url}")
    print(f"enable_rentals: {settings.enable_rentals}")
    print(f"real fares    : {settings.enable_real_fares}\n")

    if not settings.enable_rentals:
        line(BAD, "enable_rentals is False -- the rental flow is switched off in config.")
        return 1
    if not args.jwt:
        line(BAD, "no rider JWT. Pass --jwt or set HIMCAB_RIDER_JWT.")
        print("\n  -> Both endpoints are rider-authenticated. The 'dev-test-token' the")
        print("    UI falls back to resolves to jwt=None on purpose, which is exactly")
        print("    why a page with no token pasted into it never shows a price.")
        return 1

    _raw, packages = await check_rate_card(args.jwt, args.raw)
    await check_estimate(args.jwt, packages, args.hours, args.group_size, args.raw)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
