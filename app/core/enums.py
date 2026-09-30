"""
app/core/enums.py
──────────────────
The vocabulary the whole system agrees on, in one place.

WHY THIS EXISTS
───────────────
Every one of these values was previously a bare string, spelled out again in each
module that needed it — `"rental"` in `slots`, in `chat_session`, in `planner`;
`"RENTAL"` in the quote builder, the tool executor and the writer. Nothing checked
that the spellings matched, so a value that drifted failed SILENTLY and in the worst
possible way: the persona compare returned False, the rental conversation fell back to
the trip checklist, and a rider hiring a car by the hour was asked how many days they
were travelling for. No exception, no log, just the wrong conversation.

These are `str` enums, so `TripType.RENTAL == "RENTAL"` is True and
`json.dumps` writes the plain string. Nothing on the wire changes; what changes is
that a typo is now an AttributeError at import time instead of a wrong reply in
production.

RULE: if a value is compared, branched on, or sent to the Him Cab backend, it belongs
here — not in a prompt and not in a literal. Prompt text may QUOTE these values, but
must never be the only place a value is written down: an LLM that omits or misspells
one has to be caught by code, and code can only catch what it can name.
"""

from enum import Enum


class StrEnum(str, Enum):
    """A string enum whose members ARE their values.

    `str(TripType.RENTAL)` gives "RENTAL" rather than "TripType.RENTAL", which matters
    because these values get interpolated into prompts, log lines and JSON payloads.
    """

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def values(cls) -> tuple[str, ...]:
        return tuple(m.value for m in cls)

    @classmethod
    def has(cls, value: object) -> bool:
        """True when `value` is one of ours. For validating anything off the wire."""
        return isinstance(value, str) and value in cls.values()


class Persona(StrEnum):
    """Which assistant the rider is talking to — fixed by the route they called.

    Durably stored on the session (`TripSession.persona`) and mirrored onto
    `ExtractedEntities.visitor_type`. LOCAL and OUTSTATION are guesses about WHO the
    rider is, so a GPS fix may override them. RENTAL is a SERVICE they chose, so
    nothing overrides it — see `chat_session.resolve_persona` and
    `planner.classify_visitor_type`.
    """

    LOCAL = "local"
    OUTSTATION = "outstation"
    RENTAL = "rental"


class TripType(StrEnum):
    """The rate card a fare is priced on — `tripType` in the Him Cab fare API.

    Decided by the JOURNEY, never by the persona: a resident driving to Delhi is an
    OUTSTATION trip, and a visitor crossing Manali is a LOCAL one. Sent verbatim to
    POST /api/rider/fare/estimate and used to filter which coupons apply, so the
    spelling here is a contract with the Spring backend, not a label.
    """

    LOCAL = "LOCAL"
    OUTSTATION = "OUTSTATION"
    RENTAL = "RENTAL"


class ResponseMode(StrEnum):
    """What shape the reply takes. Chosen in code by `slots.response_mode`.

    This is the single decision the writer's prompt is built from, which is why it is
    an enum and not a phrase in a prompt: each mode forbids what the others require.
    An ITINERARY_MODE reply to a rental rider is the day-by-day plan bug; a RENTAL_MODE
    reply to a visitor is a package with no trip.
    """

    ITINERARY = "itinerary"
    RIDE = "ride"
    ANSWER = "answer"
    RENTAL = "rental"
    #: Nothing is being sold. The rider called it off, so the reply confirms that and
    #: says what is and is not booked — no fare, no cab list, no itinerary, no booking
    #: screen. Written from a table rather than by the model (`slots.cancellation_message`):
    #: a cancellation is three lines of fact, and the model is the thing that was
    #: getting them wrong.
    CANCELLED = "cancelled"


class Intent(StrEnum):
    """What the rider's message is FOR, as classified by the router each turn.

    Advisory, not authoritative. The router re-reads every message, so in a rental chat
    "kasol" comes back as PLAN_TRIP — which is why the persona outranks this for both
    `slots.needs_for` and `slots.response_mode`. Anything unrecognised is treated as
    PLAN_TRIP.
    """

    PLAN_TRIP = "plan_trip"
    CLARIFICATION = "clarification"
    SHOW_ITINERARY = "show_itinerary"
    COST_QUERY = "cost_query"
    WEATHER_QUERY = "weather_query"
    PLACES_QUERY = "places_query"
    HOTEL_QUERY = "hotel_query"
    GENERAL_QUERY = "general_query"
    RENTAL_QUERY = "rental_query"
    #: The rider is calling it off — "actually, never mind", "cancel that", "rehne do".
    #: The one intent that outranks the persona: a rental chat that classifies a
    #: cancellation as RENTAL_QUERY carries on collecting details for a request the
    #: rider has just abandoned, which is what they experience as being ignored.
    CANCEL_REQUEST = "cancel_request"


class SlotName(StrEnum):
    """Every detail the agent can collect. The keys of `slots.SLOTS`.

    Named so a policy table cannot reference a field that does not exist: the needs
    tables are checked against this in tests, and a slot renamed in one place and not
    the other stops being a question that is silently never asked.
    """

    DESTINATION = "destination"
    ORIGIN = "origin"
    DURATION_DAYS = "duration_days"
    GROUP_SIZE = "group_size"
    TRAVEL_DATES = "travel_dates"
    PICKUP_TIME = "pickup_time"
    RENTAL_HOURS = "rental_hours"
    #: Which cab the rider picked off the priced list — the app's "Select Rental Cab"
    #: radio group. Never asked before there are prices to choose between, so it is
    #: OPTIONAL in the needs table and put to the rider by the quoting turn instead.
    CAB_CATEGORY = "cab_category"
    RETURN_JOURNEY_PREFERENCE = "return_journey_preference"


class ReplyLanguage(StrEnum):
    """The language key the question tables and the writer are keyed on.

    Deliberately just the two the service supports; `app/agent/language.py` decides
    which, per turn, from what the rider actually writes.
    """

    ENGLISH = "en"
    HINDI = "hi"


class CancelScope(StrEnum):
    """WHAT the rider wants called off. The distinction is not cosmetic.

    Telling somebody with a confirmed cab that "nothing was booked, don't worry" is a
    dangerous thing to be wrong about — their driver is still coming. Telling somebody
    who has just changed their mind mid-chat to "go to your Trips screen" sends them
    looking for a booking that does not exist. So the reply says only what is true of
    the case in hand, and UNCLEAR — the common one — says what is true of BOTH.
    """

    #: The request being assembled in this chat. Nothing was ever booked; the details
    #: are dropped and the rider can start again.
    REQUEST = "request"
    #: A trip they have already booked in the app. This service never made it and holds
    #: no trip id for it, so it cannot and must not cancel it — see `cancellation_message`.
    BOOKED_TRIP = "booked_trip"
    #: Could be either. The reply covers both without claiming either.
    UNCLEAR = "unclear"
