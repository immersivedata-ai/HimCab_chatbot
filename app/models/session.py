"""
app/models/session.py
──────────────────────
Session, Message, and conversation state models.
Persisted in Firestore.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.models.itinerary import BudgetRange, GroupType, Itinerary
from app.models.location import RiderLocation


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Language(str, Enum):
    ENGLISH = "en"
    HINDI = "hi"
    HINGLISH = "hinglish"


class Message(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    role: MessageRole
    content: str
    language: Language = Language.ENGLISH
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedEntities(BaseModel):
    """
    Entities extracted by the Planner node from the conversation.
    Progressively filled as the user provides more info.
    """
    destination: Optional[str] = None
    origin: Optional[str] = None
    duration_days: Optional[int] = None
    travel_dates: Optional[str] = None
    group_type: Optional[GroupType] = None
    group_size: Optional[int] = None
    budget_range: Optional[BudgetRange] = None
    interests: list[str] = Field(default_factory=list)
    special_requirements: list[str] = Field(default_factory=list)
    visitor_type: Optional[str] = None  # "outstation" | "local" | None
    return_journey_preference: Optional[str] = None  # "same_cab_return" | "own_arrangement" | None — outstation only
    home_city: Optional[str] = None  # For local visitors — their base city within HP

    # ── Conversation control (not trip facts) ────────────────────────────────
    # Persisted alongside the entities because the planner's decision on the next
    # turn depends on what happened on this one. See app/agent/slots.py.
    #: The field our last question was about. A reply is only ever read as an
    #: answer to THIS field — the agent never reinterprets an unrelated message.
    last_asked_field: Optional[str] = None
    #: field name → how many times we have asked for it. Bounds the conversation:
    #: past `max_clarification_attempts` the agent proceeds instead of re-asking.
    asked_counts: dict[str, int] = Field(default_factory=dict)
    #: Set once the user tells us to stop asking and get on with it ("just make
    #: the plan", "kuch bhi", "you decide"), in any language. Sticky for the session.
    proceed_requested: bool = False
    #: Consecutive turns where the user replied but told us nothing new. Caps the
    #: conversation as a whole, not just per field: past the limit the agent stops
    #: asking and answers with what it has. Resets the moment they engage again.
    no_progress_turns: int = 0


class SessionStatus(str, Enum):
    ACTIVE = "active"
    PLANNING = "planning"
    ITINERARY_READY = "itinerary_ready"
    COMPLETED = "completed"
    EXPIRED = "expired"


class TripSession(BaseModel):
    """
    A full trip planning session stored in Firestore.
    Acts as the long-term memory unit for a single planning conversation.
    """
    model_config = ConfigDict(use_enum_values=True)

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    status: SessionStatus = SessionStatus.ACTIVE
    language: Language = Language.ENGLISH
    # Which front-end app opened this chat: "local" | "outstation". Fixed at creation
    # and never changed — unlike entities.visitor_type, which the planner may correct
    # from a real origin city. Used to resume the right conversation per app.
    persona: Optional[str] = None
    messages: list[Message] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
    #: The rider's last known position, from the app's GPS fix. Stored so later turns
    #: keep the pickup point without the app resending it, and refreshed whenever it
    #: does. None for a client that sends no coordinates — everything still works, the
    #: assistant just has to ask which city they are starting from.
    rider_location: Optional[RiderLocation] = None
    current_itinerary: Optional[Itinerary] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now(timezone.utc)
        return msg

    def to_firestore_dict(self) -> dict:
        """Serialize for Firestore storage."""
        return self.model_dump(mode="json")

    @classmethod
    def from_firestore_dict(cls, data: dict) -> "TripSession":
        return cls.model_validate(data)


# ── API Request/Response DTOs ─────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    language: Language = Language.ENGLISH
    initial_message: Optional[str] = None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime
    status: str


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
                           duration_text, distance_source, is_approximate, source,
                           best_offer, fare_after_offer_inr}. `options` is already
                           narrowed to cabs that seat the party. `is_approximate` is
                           true when the route distance was computed rather than
                           surveyed — render the price as "approx" in that case.
                           `distance_source` says which: "surveyed" | "gps" | "city".
      - `offers`         → live Him Cab coupons that apply to this trip:
                           {trip_type, fare_inr, offers: [{offer_id, coupon_code,
                           coupon_name, coupon_value_inr, minimum_value_inr,
                           fare_after_offer_inr, terms_and_conditions, image,
                           expire_date}], best_offer}. Already filtered to the trip
                           type, the live date window and the quoted fare, best saving
                           first. Absent when the rider has no applicable coupon —
                           never a placeholder or an empty card.
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

    Zero or more events accompany any reply; clarification turns carry none.
    """

    type: StructuredEventType
    payload: dict = Field(default_factory=dict)


class ResolvedLocation(BaseModel):
    """What the backend made of the coordinates you sent."""

    lat: float
    lon: float
    in_himachal: bool = Field(
        description="True when the fix falls inside Himachal Pradesh."
    )
    place_name: Optional[str] = Field(
        default=None,
        description=(
            "Nearest town we know, used as the pickup city for distances and fares. "
            "Null when the fix is far from every place in our geography tables."
        ),
    )
    place_distance_km: Optional[float] = Field(
        default=None, description="How far that town is from the fix."
    )
    near_border: bool = Field(
        default=False,
        description=(
            "True within a few km of the state line, where the classification could "
            "go either way. Treat the persona as a default the rider may correct."
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
