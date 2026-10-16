"""
app/domain/conversation.py
───────────────────────────
The conversation itself, as the system stores and reasons about it.

These are ENTITIES, not wire formats. `TripSession` is the Firestore document,
`ExtractedEntities` is what the planner fills in and the slot policy reads, and
`Message` is one turn of it. Nothing here is shaped by what an HTTP response needs
to look like — that lives in app/schemas/, and the two are allowed to differ.

The split exists because they were one file, and a field added for the app to render
kept ending up persisted, while a field the agent needed kept leaking into the public
API shape. A change to either side should not be able to silently change the other.
"""

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.itinerary import BudgetRange, GroupType, Itinerary
from app.domain.location import PickedPlace, RiderLocation


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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
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
    #: Hours the rider wants the car for (rental flow only). Matched EXACTLY against a
    #: package on the rate card — see app/integrations/himcab/rental_service.py.
    rental_hours: Optional[int] = None
    #: What time the cab should arrive, normalised to "HH:mm" (24-hour). NOT the same
    #: question as `travel_dates`, which is only the day: the backend reads a clock time
    #: as `bookingTime` and applies the night rate from it, so a 6 a.m. pickup asked for
    #: at 10 p.m. was being priced on the hour the rider happened to be typing.
    pickup_time: Optional[str] = None
    #: Which cab the rider chose, as a lower-case tier word ("luxury", "mini bus") —
    #: the chat equivalent of the app's "Select Rental Cab" radio group. Matched against
    #: the categories the fare endpoint actually priced (tool_executor), never trusted
    #: on its own: a tier we cannot find in the quote is told to the rider rather than
    #: booked.
    cab_category: Optional[str] = None
    #: The exact pickup point, when the rider PICKED it in the app rather than naming
    #: it — see `PickedPlace`. Read only while `origin` is still its `name`: a pickup
    #: changed later, by a correction or a cancellation, leaves this behind as stale,
    #: and a point that no longer matches the pickup must never price or pin the trip.
    origin_place: Optional[PickedPlace] = None
    #: The exact drop point, when the rider PICKED it in the app — the same as
    #: `origin_place`, for `destination`, and stale the same way once the drop changes.
    destination_place: Optional[PickedPlace] = None

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
    #: field -> proposed new value, waiting for the rider to say yes. A detail that
    #: CHANGES is never applied on the turn it is read: an answer fills a blank, but a
    #: change overwrites something the rider is already looking at on a booking screen,
    #: and the value we read is the thing most likely to be wrong. See
    #: app/agent/modification.py.
    pending_changes: dict[str, Any] = Field(default_factory=dict)
    #: Set when the rider has said they want to modify something but not yet what, so
    #: the next detail they give is applied WITHOUT asking again — they have already
    #: confirmed the intent, and asking twice is how a correction turns into an
    #: interrogation.
    awaiting_change_detail: bool = False


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
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    expires_at: Optional[datetime] = None

    def add_message(self, role: MessageRole, content: str, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        self.messages.append(msg)
        self.updated_at = datetime.now(UTC)
        return msg

    def to_firestore_dict(self) -> dict:
        """Serialize for Firestore storage."""
        return self.model_dump(mode="json")

    @classmethod
    def from_firestore_dict(cls, data: dict) -> "TripSession":
        return cls.model_validate(data)
