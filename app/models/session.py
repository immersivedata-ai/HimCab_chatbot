"""
app/models/session.py
──────────────────────
Session, Message, and conversation state models.
Persisted in Firestore, hot-cached in Redis.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.models.itinerary import BudgetRange, GroupType, Itinerary


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
    messages: list[Message] = Field(default_factory=list)
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)
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


class ChatRequest(BaseModel):
    session_id: str
    message: str
    language: Language = Language.ENGLISH


class ChatResponse(BaseModel):
    """Response model for the non-streaming POST /v1/chat/message endpoint."""
    session_id: str
    response: str
    structured_events: list[dict] = Field(default_factory=list)
    itinerary: Optional[dict] = None
    node_trace: list[str] = Field(default_factory=list)


class SessionSummary(BaseModel):
    session_id: str
    destination: Optional[str]
    duration_days: Optional[int]
    status: str
    created_at: datetime
    updated_at: datetime
