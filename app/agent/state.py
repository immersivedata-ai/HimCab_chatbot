"""
app/agent/state.py
───────────────────
LangGraph AgentState — the single shared state object passed between nodes.
Every node reads from and writes to this state.
This is what makes the graph stateful.
"""

from enum import Enum
from typing import Annotated, Any, Optional
from operator import add

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app.models.itinerary import Itinerary
from app.models.session import ExtractedEntities, Language
from app.models.user import UserProfile


class IntentType(str, Enum):
    """All recognised user intents for the router node."""
    PLAN_TRIP = "plan_trip"
    WEATHER_QUERY = "weather_query"
    HOTEL_QUERY = "hotel_query"
    PLACES_QUERY = "places_query"
    COST_QUERY = "cost_query"
    SHOW_ITINERARY = "show_itinerary"
    GENERAL_QUERY = "general_query"
    CLARIFICATION = "clarification"


class AgentState(BaseModel):
    """
    Full shared state for the LangGraph travel planning agent.

    Field annotations:
      - add_messages: LangGraph built-in reducer — appends messages safely
      - Annotated[list, add]: custom reducer — appends to lists across nodes
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # ── Conversation ──────────────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages] = Field(default_factory=list)
    session_id: str = ""
    user_id: str = ""
    language: Language = Language.ENGLISH

    # ── Routing ───────────────────────────────────────────────────────────────
    intent: Optional[str] = None          # Classified intent from Router node
    visitor_type: Optional[str] = None    # "outstation" | "local" — classified by Router/Planner
    needs_clarification: bool = False     # True if Planner needs more info
    clarification_question: Optional[str] = None

    # ── User context (from Context node) ─────────────────────────────────────
    user_profile: Optional[UserProfile] = None
    user_context_summary: str = ""        # Natural language summary injected into prompt

    # ── Extracted information (from Planner node) ─────────────────────────────
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)

    # ── Booking ───────────────────────────────────────────────────────────────
    booking_intent_active: bool = False
    booking_details: dict = Field(default_factory=dict)
    user_confirmed_booking: bool = False
    final_booking_id: Optional[str] = None

    # ── Tool results (from Tool Executor node) ────────────────────────────────
    weather_data: Optional[dict] = None
    snowfall_data: Optional[dict] = None
    places_data: dict[str, list] = Field(default_factory=dict)  # category → places
    drive_times: dict[str, dict] = Field(default_factory=dict)
    travel_estimate: Optional[dict] = None   # origin→destination travel info (duration, distance, mode)
    route_plan: Optional[dict] = None        # Full multi-leg route plan for outstation visitors
    crowd_data: dict[str, dict] = Field(default_factory=dict)
    available_offers: list[dict] = Field(default_factory=list)  # Personalised offers from OffersService

    # ── Frequent Traveller (from Context node) ────────────────────────────────
    is_frequent_traveller: bool = False

    # ── Generated output (from Writer node) ───────────────────────────────────
    final_response_text: str = ""
    structured_itinerary: Optional[Itinerary] = None
    structured_events: Annotated[list[dict], add] = Field(default_factory=list)

    # ── Streaming ─────────────────────────────────────────────────────────────
    streaming_tokens: Annotated[list[str], add] = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────────────
    error: Optional[str] = None
    tokens_used: int = 0
    node_trace: Annotated[list[str], add] = Field(default_factory=list)
