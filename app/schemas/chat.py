"""
app/schemas/chat.py
────────────────────
The public shape of the chat API — every model the mobile app sends or receives.

This is a CONTRACT. A field renamed here is a breaking change for a shipped app
build, which is exactly why these models no longer sit in the same file as the
entities they are built from: an entity may be refactored freely, a wire format may
not. See app/api/v1/routes/chat.py for the endpoint-by-endpoint contract, and
app/domain/conversation.py for what is actually persisted.

`StructuredEvent` is the one to read first — it is how every renderable card
(itinerary, fare, offers, rental packages, booking handoff, cancellation) reaches
the app, on both the streaming and non-streaming paths.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.core.enums import Persona


class RiderCoordinates(BaseModel):
    """The device's GPS fix, sent with a chat request.

    Optional everywhere. Sending it is what lets the assistant skip "are you local or
    visiting?" and "which city are you travelling from?", and price the trip from where
    the rider actually is. Omit it and every endpoint still works — the assistant just
    asks, as it did before.
    """

    lat: float = Field(..., ge=-90, le=90, description="Latitude in decimal degrees.")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in decimal degrees.")
    accuracy_m: Optional[float] = Field(
        default=None,
        ge=0,
        description=(
            "Device-reported accuracy in metres, if you have it. Informational — a "
            "coarse fix is still used, since it beats having none."
        ),
    )


class ChatMessageRequest(BaseModel):
    """Body for POST /v1/chat/{persona}/message."""

    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = Field(
        default=None,
        description=(
            "Optional. Omit it and the backend resolves the conversation from your "
            "Bearer token — that is the normal case and it keeps multi-turn context. "
            "Pass the `session_id` from /start to pin a specific conversation, which "
            "is what you want when one account is signed in on more than one device "
            "(otherwise both devices share a single conversation). An unknown id, or "
            "one belonging to another rider, is ignored rather than rejected."
        ),
    )
    lat: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
        description=(
            "Optional device latitude. Sent together with `lon`, it sets the pickup "
            "point: the fare is priced from here, and a fix inside Himachal marks the "
            "rider as a local. Resent on every turn is fine — the newest fix wins."
        ),
    )
    lon: Optional[float] = Field(
        default=None, ge=-180, le=180, description="Optional device longitude. See `lat`."
    )
    accuracy_m: Optional[float] = Field(
        default=None, ge=0, description="Optional device-reported accuracy in metres."
    )


class StructuredEventType(str, Enum):
    """Every `type` the backend can emit in `structured_events` / the SSE `structured`
    event. This list is exhaustive — the app can switch on it safely, but should still
    ignore unrecognised values so new types can ship without breaking old builds."""

    ITINERARY = "itinerary"
    WEATHER_ALERT = "weather_alert"
    SNOWFALL_ALERT = "snowfall_alert"
    CAB_FARE = "cab_fare"
    OFFERS = "offers"
    BOOKING_HANDOFF = "booking_handoff"
    RENTAL_PACKAGES = "rental_packages"
    REQUEST_CANCELLED = "request_cancelled"
    #: The selectable services shown before a flow is chosen — the in-chat replacement
    #: for the old persona dropdown. Payload: {flows: [FlowOption]}. Tapping one is
    #: POST /v1/chat/select with `flow` = the box's `persona`.
    FLOW_OPTIONS = "flow_options"
    #: The rider is changing a detail they already gave, and is being asked to confirm
    #: it before anything moves. Payload: {changes: [{field, label, current, proposed}]}.
    CHANGE_CONFIRMATION = "change_confirmation"
    #: The rider asked to change something without saying what. Payload:
    #: {fields: [{field, label, current}]} — the details we hold, to pick from.
    CHANGE_OPTIONS = "change_options"
    #: The question in flight wants one date or one small number, so the app can offer
    #: a calendar / number picker instead of the keyboard. Payload: see StructuredEvent.
    INPUT_REQUEST = "input_request"


class StructuredEvent(BaseModel):
    """A renderable card that accompanies the assistant's text reply.

    `payload` shape depends on `type`:
      - `itinerary`      → the full `Itinerary` model (same object as the response's
                           top-level `itinerary` field — always identical, never a
                           different serialisation of it)
      - `weather_alert`  → one weather alert: {event, severity, description, ...}
      - `snowfall_alert` → {risk: "high"|"critical", message, ...}
      - `cab_fare`       → Him Cab quote: {options[], category_name, fare_inr,
                           vehicle_label, seat_capacity, trip_type, distance_km,
                           duration_text, source, selected_by_rider,
                           best_offer, fare_after_offer_inr}. `options` is already
                           narrowed to cabs that seat the party. `is_approximate` is
                           true when the route distance was computed rather than
                           surveyed — render the price as "approx" in that case.
                           On a RENTAL, `options` is the "Select Rental Cab" list:
                           render it as a radio group and send the rider's pick back as
                           its `category_name`. `selected_by_rider` is true once they
                           have picked one, and `fare_inr` / `category_name` are then
                           THEIR cab rather than our recommendation.

      - `offers`         → live Him Cab coupons that apply to this trip:
                           {trip_type, fare_inr, offers: [{offer_id, coupon_code,
                           coupon_name, coupon_value_inr, minimum_value_inr,
                           fare_after_offer_inr, terms_and_conditions, image,
                           expire_date}], best_offer}. Already filtered to the trip
                           type, the live date window and the quoted fare, best saving
                           first. Absent when the rider has no applicable coupon —
                           never a placeholder or an empty card.
      - `rental_packages`→ the hourly blocks the rider may choose from, RENTAL chats
                           only: {packages: [{hours, included_km, label,
                           from_rate_card}]}, shortest first. Render them as taps: the
                           backend matches a rental duration exactly, so a typed answer
                           that is not one of these cannot be priced. The list is the
                           rate card narrowed to the durations the product sells (min 3,
                           max 8 hours); when the rate card cannot be read at all it is
                           the configured durations instead, flagged
                           `from_rate_card: false` — still choosable and still priceable,
                           but with `included_km: null`, so show no distance allowance
                           for one. Sent both when the duration is ASKED for and when it
                           is quoted, so the list is never stale on screen.
      - `request_cancelled`→ the rider called it off: {scope, dismiss_booking_screen,
                           cleared_fields[], open_screen}. ACT ON IT: dismiss any
                           booking sheet a previous `booking_handoff` opened, and clear
                           the fare/package cards for this conversation — the request
                           they describe no longer exists. `scope` is "request" (this
                           chat's request, now dropped), "booked_trip" (they mean a cab
                           already booked in the app) or "unclear" (either — the reply
                           covers both). `open_screen` is "trips" when they need the
                           app's own cancellation flow, else null. Nothing is cancelled
                           on the backend by this event: a confirmed booking is
                           cancelled only from the Trips screen, which is what notifies
                           the driver.
      - `booking_handoff`→ open the booking screen on this journey. `origin` and
                           `destination` each carry {name, lat, lon} so the map can
                           drop both pins without geocoding again; plus distance_km,
                           duration_text, passengers, travel_dates, the
                           recommended_category_id to preselect and, when one applies,
                           offer_id / coupon_code to prefill. `trip_type` is the rate
                           card it was priced on — "OUTSTATION" for an intercity
                           transfer, "LOCAL" for a short run — decided by the
                           journey's length, never by the persona; use it to pick the
                           booking screen the journey will actually be accepted on.
                           Sent to both LOCAL and OUTSTATION riders once a route is
                           known; absent when the pickup was never established.
                           Nothing is reserved — the app owns the booking, and the
                           reply never asks the rider to confirm.
      - `change_confirmation`→ the rider changed a detail they had already given, and is
                           being asked to confirm it BEFORE it is applied:
                           {changes: [{field, label, current, proposed}]}. Render the
                           pairs with a Yes and a No; Yes is the word "yes"/"haan" sent
                           as an ordinary message, No likewise. Nothing has changed yet
                           when this arrives — any `booking_handoff` already on screen
                           is still the OLD journey and must not be updated until the
                           rider confirms. Both sides are sent because the `proposed`
                           value is our reading of their sentence and showing it is how
                           a misparse gets caught before a car is sent on it.
      - `change_options` → the rider asked to change something without saying what:
                           {fields: [{field, label, current}]}, the details we hold.
                           Render them as tappable rows; tapping one should send the
                           field and its new value as an ordinary message ("destination
                           Kasol"). A detail given in reply to THIS card is applied
                           directly — no `change_confirmation` follows, because the
                           rider has already said they want a change.
      - `flow_options`   → the selectable services shown before a flow is chosen, the
                           in-chat replacement for the persona dropdown:
                           {flows: [{persona, title, subtitle, base_path, language}]}.
                           Render each as a tappable box; when the rider taps one, call
                           POST /v1/chat/select with `flow` = that box's `persona`, and
                           the backend starts that flow and returns its greeting. Carried
                           on GET /v1/chat/flows and never mid-turn today, but shaped as a
                           structured event so the same card-rendering code handles it.
      - `input_request`  → the question just asked wants a single date or a single
                           number: {field, kind: "date"} or {field, kind: "number", min,
                           max, unit}. `field` is "travel_dates", "group_size" or
                           "duration_days". Offer a calendar (no past days) or a min..max
                           number picker beside the reply, and send the pick back as an
                           ordinary message — "25-09-2026" for a date, the bare number
                           ("4") for a number. Typing stays allowed; the picker is only a
                           shortcut. Sent only on the asking turn, never on an answer.

    Zero or more events accompany any reply; clarification turns carry none.
    """

    type: StructuredEventType
    payload: dict = Field(default_factory=dict)


class ResolvedLocation(BaseModel):
    """What the backend made of the coordinates you sent."""

    lat: float
    lon: float
    in_himachal: bool = Field(
        description=(
            "True when the fix falls inside Himachal Pradesh, per the geocoder's "
            "administrative boundaries."
        )
    )
    place_name: Optional[str] = Field(
        default=None,
        description=(
            "The town the geocoder named for this fix, used as the pickup city for "
            "distances and fares. Null when it could not name one."
        ),
    )
    place_distance_km: Optional[float] = Field(
        default=None, description="How far that town's centre is from the fix."
    )
    state: Optional[str] = Field(
        default=None,
        description=(
            "The administrative state the fix resolved to — the evidence behind "
            "`in_himachal`. Informational; the rider's own words still override the "
            "persona if they say they are travelling from elsewhere."
        ),
    )


class PersonaRouteResponse(BaseModel):
    """Response for POST /v1/chat/route — which assistant this rider belongs to.

    Call it once with the device's GPS fix before opening the chat, then use the
    returned `persona` for every /v1/chat/{persona}/* call. A rider inside Himachal
    gets the LOCAL assistant (Hindi, cab-first replies); everyone else gets OUTSTATION
    (English, itinerary-first).
    """

    persona: str = Field(description='"local" or "outstation" — use this in the path.')
    base_path: str = Field(
        description="Prefix for this rider's chat endpoints, e.g. /v1/chat/local."
    )
    reason: str = Field(
        description=(
            'Why: "coordinates" when the fix decided it, "default" when no usable fix '
            "was sent and the requested persona was kept."
        )
    )
    location: Optional[ResolvedLocation] = None


class StartChatResponse(BaseModel):
    """Response for POST /v1/chat/{persona}/start — render `message` as the first
    assistant bubble. Keep `session_id` if you want to pin this conversation on later
    calls; otherwise the backend tracks it for you."""
    session_id: str
    message: str
    language: str
    persona: str = Field(
        description=(
            "The persona this conversation actually runs as. It matches the path you "
            "called UNLESS you sent coordinates that say otherwise — a fix outside "
            "Himachal on /local/start is served as outstation, and this field tells "
            "you so. Use it for the rest of the conversation."
        )
    )
    location: Optional[ResolvedLocation] = None
    structured_events: list[StructuredEvent] = Field(
        default_factory=list,
        description=(
            "Cards to render alongside the greeting, same shapes as on a normal turn. "
            "Only `/rental/start` sends any today: a `rental_packages` event, because "
            "the greeting asks for a duration and the backend matches durations "
            "exactly — the rider should be able to tap one rather than guess. Empty "
            "everywhere else; handle it with the same code that handles a turn's "
            "events."
        ),
    )


class ChatResponse(BaseModel):
    """Response model for the non-streaming POST /v1/chat/{persona}/message endpoint."""

    session_id: str = Field(description="The conversation this turn belongs to.")
    response: str = Field(description="Assistant reply text — render as a chat bubble.")
    persona: str = Field(
        default="",
        description=(
            "The persona this turn was served as — see StartChatResponse.persona. "
            "Differs from the path only when coordinates said otherwise."
        ),
    )
    structured_events: list[StructuredEvent] = Field(
        default_factory=list,
        description="Renderable cards for this turn. See StructuredEvent.",
    )
    itinerary: Optional[dict] = Field(
        default=None,
        description=(
            "The full itinerary when this turn produced one, else null. Identical to "
            "the payload of the `itinerary` entry in `structured_events`; read "
            "whichever is convenient."
        ),
    )
    node_trace: list[str] = Field(
        default_factory=list,
        description="Agent nodes that ran, for debugging. Not part of the UI contract.",
    )


class ChatHistoryMessage(BaseModel):
    """One turn in a replayed conversation."""

    role: str = Field(description='"user" or "assistant" — drives bubble alignment.')
    content: str
    timestamp: datetime


class ChatHistoryResponse(BaseModel):
    """Response for GET /v1/chat/{persona}/history.

    Lets the app restore a chat panel without ever having stored a session id: the
    conversation is resolved from the Bearer token, exactly as /message resolves it.
    An empty `messages` list means the rider has no conversation yet — not an error.
    """

    session_id: str
    persona: str
    language: str
    messages: list[ChatHistoryMessage] = Field(default_factory=list)
    total_messages: int = Field(
        description="Total stored on the conversation, which may exceed len(messages)."
    )


class GreetingResponse(BaseModel):
    """Response for GET /v1/chat/{persona}/greeting — the opening bubble, and nothing else.

    Exists so the app can render its first bubble without waiting for `/start`, which
    has to reverse-geocode a fix, read the rental rate card and write a session to
    Firestore before it can answer. None of that work changes this text.

    The app should SHIP a copy of these strings and render from the copy on launch —
    that is the only thing that survives a cold start, when the container itself is
    still booting and no endpoint can answer. This endpoint is how that copy is kept
    honest: fetch it in the background, and if it differs from the bundled string, the
    copy is stale and the newer text wins.
    """

    persona: str = Field(description='"local", "outstation" or "rental".')
    message: str = Field(
        description=(
            "The opening line, ready to render as the first assistant bubble. For "
            "rental this is the GENERIC wording — `/start` returns a better one that "
            "names the real hourly packages, so replace this when that arrives."
        )
    )
    language: str = Field(
        description="The persona's default reply language. The agent adapts from the rider's first message onward."
    )
    is_final: bool = Field(
        description=(
            "True when `/start` will return this same text, so there is nothing to "
            "replace. False for rental, where the greeting is upgraded once the rate "
            "card has been read."
        )
    )


class FlowOption(BaseModel):
    """One selectable service in the flow chooser — a box the rider taps to start it.

    The in-chat replacement for the old persona dropdown. `persona` is what the tap
    starts: send it back as `flow` on POST /v1/chat/select (or call {base_path}/start
    directly). Title and subtitle are already localized to the language /flows was asked
    for.
    """

    persona: str = Field(
        description='"local", "outstation" or "rental" — pass this as `flow` to POST /v1/chat/select.'
    )
    title: str = Field(description="Short label for the box, e.g. 'Rent by the hour'.")
    subtitle: str = Field(description="One line describing what this flow does.")
    base_path: str = Field(
        description=(
            "This flow's chat endpoints, e.g. /v1/chat/rental. A tap is POST "
            "/v1/chat/select with `flow`=persona; {base_path}/start does the same thing "
            "for a client that prefers per-persona paths."
        )
    )
    language: str = Field(description="The flow's default reply language ('en' | 'hi').")


class FlowChooserResponse(BaseModel):
    """Response for GET /v1/chat/flows — the neutral opening and the selectable flows.

    Render `message` as the first assistant bubble and `flows` as tappable boxes (the
    replacement for the persona dropdown). When the rider taps one, call POST
    /v1/chat/select with `flow` = the box's `persona`; the backend starts that flow and
    returns its greeting, exactly as /start does. `structured_events` carries the same
    list as a single `flow_options` card, so the boxes render through the same code as
    every other structured event.

    No `Authorization` header, no session created, no backend called — it is a constant,
    like `/greeting`, so it can be shown the instant the chat opens.
    """

    message: str = Field(description="Neutral opening line to show above the boxes.")
    language: str = Field(description="The language `message` and the boxes are in.")
    flows: list[FlowOption]
    structured_events: list[StructuredEvent] = Field(
        default_factory=list,
        description="A single `flow_options` event mirroring `flows`, for card-based rendering.",
    )


class FlowSelectionRequest(BaseModel):
    """Body for POST /v1/chat/select — the rider tapped a flow box.

    Coordinates are optional and behave exactly as on /start: they set the pickup point
    and, for local/outstation, a fix decides local-vs-visiting the same way the screen
    the chat was opened from used to. Rental is never re-routed by a fix.
    """

    flow: Persona = Field(
        description='The chosen flow: "local", "outstation" or "rental" (a box\'s `persona`).'
    )
    lat: Optional[float] = Field(
        default=None, ge=-90, le=90, description="Optional device latitude — see ChatMessageRequest.lat."
    )
    lon: Optional[float] = Field(
        default=None, ge=-180, le=180, description="Optional device longitude."
    )
    accuracy_m: Optional[float] = Field(
        default=None, ge=0, description="Optional device-reported accuracy in metres."
    )
