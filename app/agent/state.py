"""
app/agent/state.py
───────────────────
LangGraph AgentState — the single shared state object passed between nodes.
Every node reads from and writes to this state.
This is what makes the graph stateful.

Every field here is written by one node and read by another. A channel that nothing
reads is a channel that quietly costs serialisation, confuses the next reader, and
hides the fact that a feature was never wired up — so if a field stops being used,
it is removed rather than left as a placeholder.
"""

from operator import add
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field

from app.domain.conversation import ExtractedEntities, Language
from app.domain.itinerary import Itinerary
from app.domain.location import RiderLocation
from app.domain.user import UserProfile


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
    #: One line saying which detail just changed, set by the planner when a rider
    #: confirms a modification. Prepended to the reply by the writer: everything
    #: downstream is rebuilt from the entities, so the reply already SHOWS the new
    #: value, but nothing in it says the value moved or why.
    change_note: Optional[str] = None
    last_asked_field: Optional[str] = None  # Which entity field was asked last turn — prevents re-asking

    # ── What the Planner could NOT confirm (from app/agent/slots.py) ──────────
    # Details the agent gave up asking for and is proceeding without. The Writer is
    # required to state each one as an assumption — this is what separates "assuming
    # a 3-day trip, tell me if that's wrong" from silently inventing a trip length.
    # Everything that shapes a plan is asked for BEFORE the plan is written, so this
    # list is only ever populated when the user declined to answer.
    assumed_fields: list[str] = Field(default_factory=list)

    # ── User context (from Context node) ─────────────────────────────────────
    user_profile: Optional[UserProfile] = None
    user_context_summary: str = ""        # Natural language summary injected into prompt

    # ── Extracted information (from Planner node) ─────────────────────────────
    entities: ExtractedEntities = Field(default_factory=ExtractedEntities)

    # ── Rider identity for Him Cab backend (the caller's Bearer token) ────────
    # exclude=True: never serialized by model_dump() (SSE/Firestore safe).
    # repr=False: kept out of reprs/logs.
    rider_jwt: Optional[str] = Field(default=None, exclude=True, repr=False)
    #: The rider's numeric Spring id, when their JWT carries one. Personalises the
    #: offer lookup; None falls back to the unpersonalised catalogue.
    rider_id: Optional[int] = None

    # ── Where the rider is (from the app's GPS fix) ───────────────────────────
    # Set by the chat API when the app sends coordinates, and carried on the session
    # afterwards. Three things read it: the persona (local vs outstation), the origin
    # the route is measured FROM, and the pickup pin in the booking handoff.
    rider_location: Optional[RiderLocation] = None

    # ── Live token streaming ──────────────────────────────────────────────────
    # An asyncio.Queue set by the SSE endpoint; the Writer pushes real LLM tokens
    # onto it for token-by-token delivery. None on the non-streaming path.
    # exclude=True: runtime-only, never serialized.
    token_stream: Optional[Any] = Field(default=None, exclude=True, repr=False)

    # ── Tool results (from Tool Executor node) ────────────────────────────────
    weather_data: Optional[dict] = None
    snowfall_data: Optional[dict] = None
    cab_fare: Optional[dict] = None          # Real fare quote (base + cab details) or None
    #: Live Him Cab coupons that apply to THIS trip, best saving first, already
    #: filtered to the trip type, the date window and the quoted fare. Empty when the
    #: backend has none — the reply may name only what is in here.
    offers: list[dict] = Field(default_factory=list)
    # The journey itself: both endpoints with coordinates, road distance and drive
    # time. Feeds the drive time quoted to a local rider AND the booking handoff the
    # app uses to drop pins on its map, so both come from one resolution of the route.
    route_estimate: Optional[dict] = None
    #: A rental duration the rider asked for that the rate card does not sell. Set by
    #: the tool executor, which ALSO clears `entities.rental_hours` — so the reply can
    #: still say "5 hours isn't one of our packages" while the slot is empty again and
    #: their next answer is read as a fresh duration rather than ignored as a repeat.
    unbookable_rental_hours: Optional[int] = None
    #: A cab tier the rider asked for that this package does not have. Set by the tool
    #: executor alongside a cleared `entities.cab_category`, exactly like
    #: `unbookable_rental_hours` — so the reply can say "we don't run a Premium on this
    #: package" and list the ones that were priced, rather than silently booking another.
    unavailable_cab_category: Optional[str] = None
    #: True while the reply's job is to ask WHICH CAB. Set by the tool executor once a
    #: fare has come back with options and the rider has not picked one (and cleared once
    #: they have, or once the question has been asked its allotted times). The writer's
    #: prompt and `_build_booking_handoff` both read it, so a reply can never announce a
    #: booking screen while still asking which cab to open it on.
    awaiting_cab_choice: bool = False
    #: Where the cab collects them, as {name, lat, lon} — rental flow only. A hire by
    #: the hour has no destination, so there is no `route_estimate` to carry the pin,
    #: and the booking screen still has to open its map somewhere.
    pickup: Optional[dict] = None
    #: Rental blocks from the rate card ({hours, included_km, label}), rental flow only.
    #: The reply offers exactly these, because the backend matches a duration against
    #: them exactly — see app/integrations/himcab/rental_service.py.
    rental_packages: list[dict] = Field(default_factory=list)

    # ── Generated output (from Writer node) ───────────────────────────────────
    final_response_text: str = ""
    structured_itinerary: Optional[Itinerary] = None
    structured_events: Annotated[list[dict], add] = Field(default_factory=list)

    # ── Metadata ──────────────────────────────────────────────────────────────
    error: Optional[str] = None
    node_trace: Annotated[list[str], add] = Field(default_factory=list)
