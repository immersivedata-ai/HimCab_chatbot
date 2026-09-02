"""
app/integrations/himcab/vehicles.py
────────────────────────────────
Which cabs can actually carry this group.

The fare endpoint prices EVERY category in one call, so the raw response always
includes vehicles the rider cannot sensibly book — a Mini Bus quoted to a party of
four. Showing that list verbatim is how the assistant ended up recommending a
20-seater for 4 travellers: the reply faithfully repeated everything the backend
returned, which is not the same as recommending something.

This module narrows the list to what fits, then orders it so the obvious choice is
first. Two rules, in this order:

  1. it must SEAT the group — a cab that cannot take everyone is not an option
  2. among those, the smallest suitable vehicle comes first — a family of four wants
     a sedan, not an empty minibus at three times the price

`seatCapacity` comes back null for most categories, so capacity is inferred from the
category name when the backend does not give one (`SEATS_BY_CATEGORY`). Inference is
deliberately conservative: an unrecognised name keeps its place in the list rather
than being filtered out on a guess, because hiding a cab the rider could have taken
is worse than showing one extra.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

    from app.integrations.himcab.models import FareCategoryFare

log = get_logger(__name__)

#: Substring in a category name → seats it takes. Checked longest-first, so
#: "tempo traveller" wins over "tempo". Used only when the backend sends no
#: seatCapacity, which is most of the time.
SEATS_BY_CATEGORY: dict[str, int] = {
    "bike": 1,
    "scooter": 1,
    "auto": 3,
    "hatchback": 4,
    "mini": 4,          # "Mini", "Mini Cab" — the small-car tier, not "Mini Bus"
    "micro": 4,
    "economy": 4,
    "sedan": 4,
    "prime": 4,
    "standard": 4,
    "comfort": 4,
    "dzire": 4,
    "etios": 4,
    "suv": 6,
    "ertiga": 6,
    "xylo": 6,
    "premium": 6,
    "luxury": 6,
    "innova": 7,
    "crysta": 7,
    "tavera": 7,
    "scorpio": 7,
    "bolero": 7,
    "tempo traveller": 12,
    "tempo": 12,
    "traveller": 12,
    "minibus": 20,
    "mini bus": 20,
    "bus": 20,
    "coach": 30,
}

#: "Mini Bus" contains "mini" (4 seats) and "bus" (20). Longest match wins, so the
#: multi-word entries must be tried first or a 20-seater is read as a hatchback.
_ORDERED_HINTS = sorted(SEATS_BY_CATEGORY, key=len, reverse=True)

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def seat_capacity_for(category_name: Optional[str]) -> Optional[int]:
    """Seats in a cab category by name, or None when the name says nothing."""
    if not category_name:
        return None
    key = " ".join(_NON_ALNUM.sub(" ", category_name.lower()).split())
    for hint in _ORDERED_HINTS:
        if hint in key:
            return SEATS_BY_CATEGORY[hint]
    return None


def effective_seats(option: FareCategoryFare) -> Optional[int]:
    """The backend's seat count, falling back to what the category name implies."""
    if option.seat_capacity and option.seat_capacity > 0:
        return option.seat_capacity
    return seat_capacity_for(option.category_name)


def select_cabs_for_group(
    options: Iterable[FareCategoryFare],
    group_size: Optional[int],
    *,
    max_options: int = 3,
) -> list[FareCategoryFare]:
    """The cabs worth showing this group, smallest suitable first.

    With no group size we cannot rule anything out, so everything is returned in the
    backend's own order — narrowing on a guess would hide a cab the rider wanted.

    When nothing on the rate card seats the group (a party of 30), the largest
    vehicles are returned rather than an empty list: the rider needs to see that more
    than one cab is required, not that Him Cab has nothing.
    """
    all_options = [o for o in options if o.total_fare_inr is not None]
    if not all_options:
        return []
    if not group_size or group_size < 1:
        return all_options[:max_options]

    fits, capacities = [], {}
    for option in all_options:
        seats = effective_seats(option)
        capacities[id(option)] = seats
        # An unknown capacity is not evidence the cab is too small.
        if seats is None or seats >= group_size:
            fits.append(option)

    if not fits:
        log.info(
            "no_cab_seats_group",
            group_size=group_size,
            largest_seats=max((c for c in capacities.values() if c), default=None),
        )
        biggest = max((c for c in capacities.values() if c), default=0)
        fits = [o for o in all_options if (capacities[id(o)] or 0) >= biggest]

    # Smallest vehicle that still fits, then cheapest — the natural recommendation
    # order. An unknown capacity sorts last so a known-good match leads.
    fits.sort(key=lambda o: (capacities[id(o)] or 99, o.total_fare_inr))
    return fits[:max_options]
