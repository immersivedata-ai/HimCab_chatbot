"""
app/domain/location.py
───────────────────────
Where the rider is, as the rest of the system carries it.

The mobile app already knows this — it has a live GPS fix for the pickup pin — and
sending it with the chat request removes two questions the assistant used to have to
ask before it could be useful:

  1. "are you a Himachal resident or visiting?"  → answered by the fix (`in_himachal`)
  2. "which city are you travelling from?"        → answered by the fix (`place_name`)

Both used to cost a round trip each, and the second one gated the fare entirely: no
origin meant no distance, and no distance meant POST /fare/estimate could not be
called at all.

`app/services/geo.py` does the resolving and hands back one of these
(`geo.resolve_rider_location`); this is the serialisable result, stored on the
session so later turns keep the location without the app resending it.

Nothing here performs I/O. `resolve()` used to live on this model and call the
geocoder, which meant an entity could not be constructed or reasoned about without
a network — and made `app/domain` depend on an HTTP client. The lookup moved to
the service; what stayed is the DATA and the one rule that is genuinely about the
data rather than about fetching it (`PRECISE_FIX_KM`).
"""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field

#: Within this of a named town, the fix IS that town — close enough to call it the
#: rider's pickup out loud. Further out the coordinates still measure the journey, but
#: the name is only the nearest thing the provider could point at, so the assistant
#: asks rather than announcing "your trip from Mandi" to somebody who has never been
#: there. Reverse geocoding returns the enclosing administrative area, so this is a
#: sanity bound on how far that area's centre may be, not a lookup radius.
#:
#: Defined here, not in the geocoding service, because it is a statement about what a
#: location MEANS rather than about how one is fetched. `app/services/geo.py` imports
#: it from here.
PRECISE_FIX_KM = 15.0


class RiderLocation(BaseModel):
    """A resolved GPS fix for the rider, as persisted on their session."""

    lat: float
    lon: float
    #: True when the fix is inside Himachal Pradesh, per the geocoder's administrative
    #: data. Decides the assistant persona.
    in_himachal: bool
    #: The town the geocoder named for this fix — the pickup used for routes and fares.
    #: None when it could name none.
    place_name: Optional[str] = None
    #: How far that town's centre is from the fix. Small means the name IS where they are.
    place_distance_km: Optional[float] = None
    #: The state the geocoder reported. Kept because it is the EVIDENCE for
    #: `in_himachal` — a support question about a misclassified rider is answerable
    #: from this without re-running the lookup.
    state: Optional[str] = None
    #: Reported by the device, when it sends one. Purely informational — a poor fix is
    #: still used, because it is better than the nothing we had before.
    accuracy_m: Optional[float] = None
    resolved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def visitor_type(self) -> str:
        """"local" | "outstation" — the persona this location implies."""
        return "local" if self.in_himachal else "outstation"

    @property
    def is_precise(self) -> bool:
        """True when the fix sits ON a town we know rather than merely near one.

        The gate on naming the rider's pickup city out loud — see `PRECISE_FIX_KM`.
        """
        return (
            self.place_name is not None
            and self.place_distance_km is not None
            and self.place_distance_km <= PRECISE_FIX_KM
        )


class PickedPlace(BaseModel):
    """A pickup or drop point the rider PICKED in the app, rather than one read from their words.

    The app's pickup dialog offers a place search and "use my current location", and its
    drop dialog a place search. Each knows the exact point, and each used to reach us as
    a name: the app sent a search pick as at most three words (falling back to the town)
    and "my current location" as the town the phone was in — because a name was all the
    chat API could carry for either end. Picked at the hotel gate, priced and pinned at
    the town centre; and a drop was re-found by its name, which can land somewhere else.

    Kept apart from `RiderLocation` on purpose. That is where the PHONE is, and it
    decides which assistant answers; this is where the rider wants the car, which may be
    somewhere else entirely, and must not move the persona.
    """

    #: The short name the conversation uses — what `ExtractedEntities.origin` (or
    #: `destination`) holds and the reply says. Short enough for `sanitize.place`,
    #: specific enough to recognise: "Hotel Snow View, The Mall, Shimla" rather than the
    #: whole postal address. None until the planner files the pick: the rule that
    #: shortens it is `sanitize`'s, next to the place limits it has to satisfy, not the
    #: geocoding service's.
    name: Optional[str] = None
    #: The full address, as the app showed it to the rider — the booking screen's label.
    address: str
    lat: float
    lon: float
    #: The town the point sits in. For a lookup that wants a town rather than a gate —
    #: the rental's weather line — since a weather service will not find a hotel.
    town: Optional[str] = None
    #: Whether the point is inside Himachal Pradesh. None when the geocoder could not
    #: say: a missing answer must never be read as "outside".
    in_himachal: Optional[bool] = None
