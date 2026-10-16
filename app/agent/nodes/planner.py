"""
app/agent/nodes/planner.py
───────────────────────────
Planner Node — turns the conversation into structured trip entities and decides
whether the agent has enough to answer.

Split of responsibility:
  • the LLM EXTRACTS — it reads the conversation and reports what the user said
  • this node DECIDES — it merges, classifies, and applies the slot policy

The LLM is never asked whether to ask a question or what to ask. It was, once, and
it produced questions in the wrong language, questions for details the user had
already given, and questions for details the request never needed. The policy now
lives in code (app/agent/slots.py) where it is deterministic and testable, and the
model is left to do the one thing it is good at: reading the conversation.

Three guarantees this node makes, each covering a reported failure:

  1. A message is only ever read as an answer to a question we actually asked, and
     only when it has the right shape (`slots.read_answer`). Nothing is inferred
     from an unrelated message, so "plan a trip" can never become a destination.
  2. The agent asks for any one detail at most `max_clarification_attempts` times,
     and stops asking altogether the moment the user says to get on with it. It
     cannot loop.
  3. Whatever it proceeds without is returned in `assumed_fields`, so the writer
     states the assumption instead of quietly inventing a fact.
"""

import asyncio
import json
import random
from datetime import UTC, datetime
from typing import Any, Literal, Optional

from google.api_core.exceptions import ResourceExhausted
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.agent import modification, sanitize, slots
from app.agent.language import detect_reply_language
from app.agent.state import AgentState
from app.core.config import settings
from app.core.enums import SlotName
from app.core.logging import get_logger
from app.domain.conversation import ExtractedEntities
from app.domain.itinerary import BudgetRange, GroupType
from app.domain.location import PickedPlace
from app.integrations.external.geocoding import LookupStatus
from app.integrations.himcab import available_packages, offered_hours
from app.services import geo
from app.services.vertex_ai import call_model, get_vertex_ai_service

log = get_logger(__name__)

# Aliases onto app/core/enums.Persona, via slots — one set of objects for the whole
# system, so a persona compare here cannot disagree with one in chat_session.
LOCAL = slots.LOCAL
OUTSTATION = slots.OUTSTATION
#: Hiring a car by the hour. Not a kind of person, so `classify_visitor_type` never
#: reclassifies it — see the note there.
RENTAL = slots.RENTAL

#: How many times a blank extraction is retried before the turn degrades to the
#: deterministic readers. Two, because a rider is waiting on every one of these.
_EXTRACTION_ATTEMPTS = 2


async def _backoff(attempt: int) -> None:
    """Wait before re-asking, a little longer each time and never the same twice.

    This retry used to fire instantly. That is fine for the case it was written for — a
    model that answered in the wrong shape — and wrong for the case that actually
    happens under load, where every client re-asks at the same moment and the second
    attempt lands in the same congestion as the first. The jitter is the point: without
    it, backoff merely synchronises everyone onto the same slower clock.
    """
    await asyncio.sleep(0.25 * (2 ** attempt) * (1 + random.random()))

# Bare greetings, recognised so "hi" costs no LLM call. Matched on the whole
# message after punctuation is stripped — never on a substring, so "hi, plan
# Manali" still goes to the extractor.
_GREETINGS = {
    "hi", "hii", "hiii", "hello", "helo", "hey", "yo", "hola", "namaste",
    "namaskar", "namaste ji", "good morning", "good afternoon", "good evening",
    "hi there", "hello there", "start", "hey there",
}

# Backstop for "stop asking and just do it" when the extraction LLM is unavailable.
# The model is the primary detector (`user_wants_to_proceed_now`), which is what
# makes this work in languages and phrasings no list can enumerate.
_PROCEED_PHRASES = (
    "just make", "just plan", "just give", "just show", "just do", "go ahead",
    "anything is fine", "anything works", "you decide", "you choose", "up to you",
    "whatever you", "dont know", "don't know", "no idea", "not sure", "skip",
    "bas bana", "bana do", "bana dijiye", "kuch bhi", "aap batao", "aap hi",
    "tum batao", "pata nahi", "nahi pata", "jo bhi", "chhodo", "chalega",
)


async def classify_visitor_type(
    origin: Optional[str],
    origin_in_himachal: Optional[bool] = None,
    *,
    home_city: Optional[str] = None,
    current: Optional[str] = None,
) -> Optional[str]:
    """Decide whether this rider is a Himachal resident or a visitor.

    The evidence is asymmetric, and treating it symmetrically was a bug:

      • an origin OUTSIDE Himachal is strong evidence of a visitor. Nobody drives in
        from Delhi and needs to be told about their own state.
      • an origin INSIDE Himachal proves nothing on its own. It fits a resident, and
        it equally fits a tourist who is already here — someone in Manali asking to
        go on to Kasol. Reading it as residence flipped English-speaking visitors to
        the local persona the moment they named their starting city, which switched
        them to Hindi and replaced their itinerary with a cab quote.

    So residence has to be stated, not inferred: `home_city` is where they LIVE (the
    extractor only fills it for a Himachal town), while `origin` is merely where this
    journey starts. An in-state origin leaves `current` untouched.

    Returns None when there is no evidence either way, so the caller keeps whatever
    classification it already had rather than guessing.
    """
    # A rental is a SERVICE the rider chose, not a guess about who they are, so no
    # amount of geography cancels it. Rentals are picked up from anywhere — Delhi,
    # Chandigarh, Ambala — and reading that pickup as "a visitor" turned the hire into
    # a holiday: the rider was asked where they wanted to go and for how many days,
    # and got an itinerary instead of a car. They leave this flow by opening a
    # different tab, never by naming a city.
    if current == RENTAL:
        return RENTAL

    # Residence beats everything, including an origin somewhere else — a Shimla
    # resident flying out of Delhi is still a local.
    if home_city and await geo.place_in_himachal(home_city):
        return LOCAL

    if not origin:
        return None

    in_himachal = (
        origin_in_himachal
        if origin_in_himachal is not None
        else await geo.place_in_himachal(origin)
    )
    if in_himachal is None:
        return None
    if not in_himachal:
        return OUTSTATION

    # In-state origin: consistent with a resident AND with a visitor already here.
    # Not enough on its own — keep what we had (the GPS fix or the front end they
    # opened), and let them say "I live in Manali" if they want to be treated as one.
    return current


class PlannerExtraction(BaseModel):
    """What the model reports about the conversation. Facts only — no flow control.

    Note there is no `needs_clarification`/`clarification_question` here any more:
    those were asked for, then discarded by the node, and the model spent tokens
    producing questions nobody read.

    This is the model the NODE works in, with `None` meaning "not stated". It is not
    the shape sent to Vertex — see `_VertexExtraction` for why it cannot be.
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
    return_journey_preference: Optional[str] = None
    home_city: Optional[str] = None
    rental_hours: Optional[int] = Field(
        default=None,
        description="Hours the rider wants to hire a car for, if they said (rental flow).",
    )
    pickup_time: Optional[str] = Field(
        default=None,
        description=(
            'The clock time the cab should arrive, as "HH:mm" on the 24-hour clock. '
            'Only from an actual time the user gave. Null for a vague "morning".'
        ),
    )
    cab_category: Optional[str] = Field(
        default=None,
        description=(
            "The cab tier the rider chose, lower-cased and nothing else "
            '("luxury", "economy", "mini bus"). Only when they picked one.'
        ),
    )
    origin_in_himachal: Optional[bool] = Field(
        default=None,
        description="True if `origin` is a town inside Himachal Pradesh, False if outside, null if unknown.",
    )
    user_wants_to_proceed_now: bool = Field(
        default=False,
        description="True if the user asked to stop being asked questions and just get the plan.",
    )
    confidence: float = 0.5


#: Values the model writes when it means "not stated". It is told to use "" and
#: sometimes writes the word instead; both are the absence of an answer, and letting
#: either through would put the string "null" in a destination field.
_WIRE_BLANKS = {"", "null", "none", "n/a", "na", "unknown", "not stated", "-"}


# ── Why there are two extraction models ───────────────────────────────────────
#
# THIS EXISTS BECAUSE OF ONE INCOMPATIBILITY, and the cost of not having it was the
# whole extraction layer silently failing about half the time.
#
# `Optional[str]` compiles to `anyOf: [{"type": "string"}, {"type": "null"}]`, and
# Vertex's `Schema` proto has no `anyOf`. In `json_mode` it says so outright —
# `ValueError: Unknown field for Schema: anyOf`. In the default `function_calling` mode
# it says nothing at all: the schema is mangled on the way through, the model returns no
# tool call, and `with_structured_output` hands back `None`. Measured over 25 calls on a
# real conversation, that was 12 to 19 failures. Every one logged a single
# `planner_extraction_failed` line and degraded exactly as designed, which is why it
# survived so long — the node kept working, on the deterministic readers alone.
#
# What a rider saw was an assistant that followed short answers and lost the plot on
# whole sentences, because a sentence is precisely what those readers cannot fully parse
# and the model was supposed to. Reported as hallucination; it was an empty extraction
# every other turn and a writer left to fill the silence.
#
# So `_VertexExtraction` is flat scalars with sentinel defaults — no `anyOf`, no
# `$defs` — and is used with `json_mode`. `to_extraction` puts the `None`s back, so
# nothing downstream knows.
#
# THE SENTINEL IS NOT THE SAME FOR EVERY TYPE, and the reason is a second Vertex rule
# found the hard way. A plain string field may default to "", but an ENUM MEMBER MAY
# NOT BE EMPTY:
#
#     400 GenerateContentRequest.generation_config.response_schema
#         .properties[origin_in_himachal].enum[0]: cannot be empty
#
# The first version of this class used "" as the sentinel inside `Literal` too. Vertex
# rejected the schema, LangChain retried it — `max_retries` was 6 at the time — and a
# four-turn conversation logged 39 to 45 of those 400s, about six and a half rejected
# requests for every extraction that eventually got through. On Dynamic Shared Quota
# that is not just waste; it is six times the request volume into a pool that answers
# 429 when it is busy, which is exactly what production then started returning.
#
# So enum sentinels are the WORD "unknown", and `_WIRE_BLANKS` maps it back to None
# along with the other ways a model says "nothing". `test_extraction_boundary.py`
# asserts no enum value is empty, beside the `anyOf` guard — same class of defect, and
# neither is visible without asking the schema.
#
# KEEP THE DOCSTRING BELOW SHORT. Pydantic sends it to Vertex as the schema description,
# so every word here would be re-sent on every extraction call.
class _VertexExtraction(BaseModel):
    """Trip facts as flat scalars. "", 0 and "unknown" mean "the user did not say"."""

    destination: str = ""
    origin: str = ""
    duration_days: int = 0
    travel_dates: str = ""
    group_type: Literal[
        "unknown", "solo", "couple", "family", "friends", "corporate"
    ] = "unknown"
    group_size: int = 0
    budget_range: Literal["unknown", "budget", "moderate", "luxury"] = "unknown"
    interests: list[str] = Field(default_factory=list)
    special_requirements: list[str] = Field(default_factory=list)
    return_journey_preference: str = ""
    home_city: str = ""
    rental_hours: int = 0
    pickup_time: str = ""
    cab_category: str = ""
    #: Three-state, as a string. A bare `bool` cannot say "no origin yet", and
    #: `Optional[bool]` would reintroduce the `anyOf` this class exists to avoid.
    origin_in_himachal: Literal["unknown", "yes", "no"] = "unknown"
    user_wants_to_proceed_now: bool = False
    confidence: float = 0.5

    def to_extraction(self) -> PlannerExtraction:
        """Back to `None`-means-unstated, which is what the node is written against."""

        def text(value: str) -> Optional[str]:
            cleaned = (value or "").strip()
            return None if cleaned.lower() in _WIRE_BLANKS else cleaned

        def count(value: int) -> Optional[int]:
            # 0 is the sentinel AND a nonsense answer for every count here — no trip
            # lasts zero days and no cab carries zero people — so nothing real is lost.
            return value if value and value > 0 else None

        # `text` already treats "unknown" as blank, so the enum sentinel goes through
        # the same door as every other way the model says nothing.
        chosen = text(self.group_type)
        budget = text(self.budget_range)
        himachal = text(self.origin_in_himachal)

        return PlannerExtraction(
            destination=text(self.destination),
            origin=text(self.origin),
            duration_days=count(self.duration_days),
            travel_dates=text(self.travel_dates),
            group_type=GroupType(chosen) if chosen else None,
            group_size=count(self.group_size),
            budget_range=BudgetRange(budget) if budget else None,
            interests=[i for i in self.interests if text(i)],
            special_requirements=[s for s in self.special_requirements if text(s)],
            return_journey_preference=text(self.return_journey_preference),
            home_city=text(self.home_city),
            rental_hours=count(self.rental_hours),
            pickup_time=text(self.pickup_time),
            cab_category=text(self.cab_category),
            origin_in_himachal=None if himachal is None else himachal == "yes",
            user_wants_to_proceed_now=self.user_wants_to_proceed_now,
            confidence=self.confidence,
        )


PLANNER_SYSTEM_PROMPT = """You are an information extractor for a cab and travel booking service.

You are NOT talking to the user. You read the conversation and report what the user
has actually said. Another system decides what to ask next — never write a question.

Extract these fields (null when the user has not stated them):
- destination: the DROP location — where they want to go / be dropped
- origin: the PICKUP location — where the cab should collect them
- origin_in_himachal: true if `origin` is a town/city inside Himachal Pradesh, false if
  it is outside (Delhi, Chandigarh, Mumbai, Amritsar, Dehradun...), null if no origin yet.
  Use your own geography knowledge. This says where the JOURNEY starts, not who they are.
- duration_days: integer, ONLY if the user gave a number of days (or a length like "a week")
- travel_dates: their rough travel time, verbatim ("next week", "kal", "15 Jan")
- group_type: one of [solo, couple, family, friends, corporate]
- group_size: integer, ONLY if the user stated how many people
- budget_range: one of [budget, moderate, luxury]
- interests: things they said they want to do or see
- special_requirements: constraints they stated (accessibility, dietary, pets...)
- return_journey_preference: "booked" | "not_booked" | "own_arrangement" | null
- rental_hours: integer — hours they want to HIRE a car for, only in a rental
  conversation and only if they gave a number ("8 hours", "aadha din" = 4, "full day"
  = 8). This is a duration, never a distance and never a trip length in days.
- cab_category: the cab tier they picked from a list you offered them ("standard",
  "economy", "luxury", "mini bus", "premium"), lower-cased. Null unless they chose one.
- pickup_time: the clock time the cab should ARRIVE, as "HH:mm" on the 24-hour clock.
  "6 am" -> "06:00", "shaam 5 baje" -> "17:00", "18:30" -> "18:30", "subah 6" -> "06:00".
  A DAY is not a time: "kal" and "tomorrow" are travel_dates, not this. Neither is a
  vague part of the day — "morning", "subah" and "evening" with no hour are null here,
  because the fare is calculated from this figure and a guessed hour changes the price.
  A phrase can fill both: "kal subah 6 baje" is travel_dates="kal" and pickup_time="06:00".
- home_city: where they LIVE, if that is inside Himachal Pradesh, and ONLY when they
  have actually said so ("I'm from Kullu", "main Shimla mein rehta hoon", "we live in
  Mandi"). This is the one field that marks somebody as a resident rather than a
  visitor, so do NOT fill it from where their trip happens to start: a tourist saying
  "I'm in Manali right now" is origin=Manali with home_city=null.
- user_wants_to_proceed_now: true when the user asks you to stop asking and just produce
  the plan, says the choice is up to you, or says they do not know / it does not matter.
  Recognise this in ANY language or phrasing ("just make it", "bas bana do", "kuch bhi",
  "aap decide karo", "I don't mind", "skip that"). Otherwise false.
- confidence: 0-1

MERGING (the conversation is multi-turn):
- The latest user message is often a SHORT answer to the assistant's previous question.
  Read that question to understand it: "How many days?" + "5" means duration_days = 5;
  "Where should we pick you up?" + "delhi" means origin = "Delhi";
  "Where should we drop you?" + "manali" means destination = "Manali".
- Previously extracted values are given to you. Repeat them in your output. Never replace
  a known value with null, and never overwrite one unless the user has just changed it.

A RENTAL CONVERSATION (the request says so above) IS NOT A TRIP:
- The rider is hiring a car and a driver BY THE HOUR. They have no destination, no drop,
  no trip length in days and no itinerary, and they are not expected to have one.
- destination and duration_days are ALWAYS null here, however the message is phrased.
  "A car for 8 hours around Manali" is rental_hours=8 with destination null — Manali is
  where the car will be driven about, not a trip they asked you to plan.
- A place they name is the PICKUP (`origin`) only when they said so: "from Shimla",
  "pick me up in Kullu", "Mandi se". A place mentioned any other way fills nothing.
- What matters is exactly what their booking screen asks for: the duration, the pickup
  address, the date, the clock time and which cab. Nothing else.

NEVER INVENT:
- Every field the user has not stated is null. Not a plausible guess, not a common
  default. No days, no group size, no dates, no budget unless they said so.
- A greeting ("hi", "namaste") carries no information — everything is null.
- Words describing the request are not places: "plan a trip", "somewhere in the hills",
  and "I want a holiday" all mean destination = null.
- Interpret relative dates against TODAY'S DATE from the request, and never produce a
  date in the past. A rough answer like "next month" is a perfectly good travel_dates.
"""


def _latest_user_text(state: AgentState) -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            return m.content.strip()
    return ""


def _looks_like_proceed_request(text: str) -> bool:
    """Backstop for the LLM's `user_wants_to_proceed_now` — see `_PROCEED_PHRASES`."""
    low = text.lower()
    return any(phrase in low for phrase in _PROCEED_PHRASES)


def _is_bare_greeting(text: str) -> bool:
    cleaned = text.strip().strip("!.,?").lower()
    return cleaned in _GREETINGS


def _has_trip_facts(entities: ExtractedEntities) -> bool:
    """True once the user has told us anything at all about the trip."""
    return any(
        slots.SLOTS[name].read(entities) for name in slots.SLOTS
    )


#: Trip-shaped fields that CANNOT exist in a rental conversation. The extraction model
#: still reports them — "a car for 8 hours around Manali" reliably comes back with
#: destination="Manali" — and every one of them then leaks somewhere it does not belong:
#: into the writer's TRIP block, into the itinerary shortcut in `route_after_router`.
#: So these are dropped at the door rather than guarded against one consumer at a time.
#:
#: The destination is here because a rental has one place, the pickup (a 2026-09-30
#: decision). For one day it was kept as the rental's DROP, and the chat showed a second
#: location search card for it — see `slots._RENTAL_NEEDS`. A picked drop point goes
#: with it: the planner never files one on a rental, so this is only the belt to that.
_NOT_IN_A_RENTAL = (
    "destination", "destination_place", "duration_days", "return_journey_preference"
)


def _strip_trip_fields(merged: dict, session_id: str) -> None:
    """Remove trip-only facts from a rental conversation's entities (mutates)."""
    dropped = [f for f in _NOT_IN_A_RENTAL if merged.get(f) is not None]
    for field in dropped:
        merged.pop(field, None)
    if dropped:
        log.info("rental_trip_fields_dropped", fields=dropped, session_id=session_id)


async def _reject_unsold_duration(
    merged: dict, rider_jwt: Optional[str], session_id: str
) -> Optional[int]:
    """Clear a rental duration Him Cab does not sell, returning what it was.

    Asked and answered in the SAME turn, which is the whole point. The duration used to
    be validated only where the fare is requested — after the pickup, the date, the time
    and the headcount had all been collected — so a rider who opened with "6 hours" was
    put through four more questions before being told the first answer had never been
    usable. Clearing it here puts the duration question back in flight immediately, and
    `clarify_node` names the packages in the same breath.

    The offered list is TTL-cached in the rental service, so this costs a network call
    once an hour rather than once a turn. An empty list means we cannot tell, and
    nothing is rejected on a guess — the fare gate is still there as the backstop.
    """
    hours = merged.get("rental_hours")
    if not hours:
        return None
    offered = await available_packages(rider_jwt)
    if not offered or hours in offered_hours(offered):
        return None

    merged["rental_hours"] = None
    log.info(
        "rental_duration_rejected_early",
        requested_hours=hours,
        offered_hours=offered_hours(offered),
        session_id=session_id,
    )
    return hours


#: Place fields checked against the map before anything acts on them, and the slot a
#: rejection re-asks. `home_city` re-asks the ORIGIN: for a resident the two are one
#: question (`slots._write_origin`), and there is no separate "where do you live?".
_PLACE_SLOTS: dict[str, SlotName] = {
    "destination": SlotName.DESTINATION,
    "origin": SlotName.ORIGIN,
    "home_city": SlotName.ORIGIN,
}


async def _place_problem(value: str, field: str, visitor_type: Optional[str]) -> Optional[str]:
    """Why this place name cannot be used, or None when it can — or when we cannot tell.

    `sanitize.place` answers "is this shaped like a place name?"; this answers "is it a
    place?", which only the map knows. Every place a rider names reached the fare, the
    booking screen and the writer unchecked: a typo that the geocoder could not place
    went ahead with no route and a writer free to describe somewhere that does not
    exist, and a word it COULD match went ahead as whatever it matched.

    A provider outage is NOT a rejection. It says nothing about the name, and refusing
    "Shimla" because Nominatim was slow would be the same failure pointed the other way.
    """
    lookup = await geo.look_up_place(value)
    if lookup.status is LookupStatus.UNAVAILABLE:
        return None
    if lookup.status is LookupStatus.NOT_FOUND or lookup.place is None:
        return slots.PLACE_NOT_FOUND
    # Deliberately no "is it in Himachal?" test. A real place anywhere is a place a
    # rider may book, and the flow no longer tells anyone it only plans Himachal trips.
    return None


#: Which end of the ride a picked place fills, as the logs name it.
_PICK_ROLES = {"origin": "pickup", "destination": "drop"}


def _accept_pick(
    picked: Optional[PickedPlace],
    field: str,
    state: AgentState,
    existing_entities: dict,
    previously_asked: Optional[str],
) -> Optional[PickedPlace]:
    """A pickup or drop picked in the app this turn, named and ready to file — or None.

    `field` is the end it fills: `origin` for the pickup dialog, `destination` for the
    drop dialog. Accepted as an ANSWER only: to that end's question, or into that end
    while it is still empty. The app opens each dialog under its question, so this is
    the normal case. A pick arriving when that end is already known and was not being
    asked about would overwrite something the rider is looking at, and a change is never
    applied without confirming it (CLAUDE.md §4.4); it is logged and left alone rather
    than guessed at.

    The short name is `sanitize.picked_name`'s, so it passes the same gate as any other
    place; failing that the town, because the exact point is worth keeping under a
    vaguer label. Neither usable means nothing to file, and the turn reads the text.
    """
    if picked is None:
        return None
    role = _PICK_ROLES[field]
    if previously_asked != _PLACE_SLOTS[field] and existing_entities.get(field):
        log.info("picked_place_ignored", role=role, reason="already_given", session_id=state.session_id)
        return None
    name = sanitize.picked_name(picked.address, picked.town) or sanitize.place(picked.town)
    if not name:
        log.info("picked_place_ignored", role=role, reason="no_usable_name", session_id=state.session_id)
        return None
    log.info(
        "place_picked_in_app",
        role=role,
        in_himachal=picked.in_himachal,
        named_from_address=name != picked.town,
        session_id=state.session_id,
    )
    return picked.model_copy(update={"name": name})


def _filed_pick(merged: dict, field: str = "origin") -> Optional[PickedPlace]:
    """The picked place the session holds for `field`, while it is still that end — or None.

    Every turn, not just the one it was picked on. The classification and the place
    check both run every turn, and each used to treat the picked label like a typed
    name: a later turn looked it up by its whole label, stored that label as the rider's
    home city, and the place check then refused the home city — clearing the pickup with
    it, because the two are one answer. The place check would refuse a picked drop's
    label the same way.
    """
    raw = merged.get(f"{field}_place")
    if isinstance(raw, dict):
        raw = PickedPlace.model_validate(raw)
    if not isinstance(raw, PickedPlace) or not geo.same_place(merged.get(field), raw.name):
        return None
    return raw


async def _reject_unplaceable(
    merged: dict,
    corrected: dict[str, Any],
    state: AgentState,
    visitor_type: Optional[str],
) -> Optional[dict]:
    """Clear the first place name that is not on the map. Returns what it was, or None.

    Checked in one place, after every writer of `merged` and after the sanitize gate,
    for the same reason that gate is where it is: there is no route around it. A place
    that was only PROPOSED as a change (`corrected`) is withdrawn from the proposal
    instead — the value the rider already had stays, because a typo in a correction is
    not a reason to lose the original.

    The geocoder's cache is shared with the tool executor, so a name checked here is
    routed there without a second request.
    """
    fix_name = state.rider_location.place_name if state.rider_location else None
    filed = _filed_pick(merged)
    filed_drop = _filed_pick(merged, "destination")

    for field, slot in _PLACE_SLOTS.items():
        # A proposed change is checked first: it is what the rider has JUST said.
        for source, is_correction in ((corrected, True), (merged, False)):
            value = source.get(field)
            if not isinstance(value, str) or not value.strip():
                continue
            # Named by the reverse geocoder from the device's own fix — already a place.
            if fix_name and geo.same_place(value, fix_name):
                continue
            # Picked on the app's map — already a point, on every later turn too. A
            # search for "Hotel Snow View, The Mall, Shimla" by name may well find
            # nothing, and refusing it would throw away the one pickup we know exactly.
            if filed and field in ("origin", "home_city") and geo.same_place(value, filed.name):
                continue
            if filed_drop and field == "destination" and geo.same_place(value, filed_drop.name):
                continue
            reason = await _place_problem(value, field, visitor_type)
            if reason is None:
                continue
            if is_correction:
                corrected.pop(field, None)
            else:
                merged[field] = None
                # A resident's origin and home city are one answer; clearing one and
                # keeping the other would keep the name we just refused.
                if field in ("origin", "home_city") and geo.same_place(
                    merged.get("origin" if field == "home_city" else "home_city"), value
                ):
                    merged["origin" if field == "home_city" else "home_city"] = None
            log.info(
                "place_rejected",
                field=field,
                value=value,
                reason=reason,
                was_correction=is_correction,
                session_id=state.session_id,
            )
            return {
                "field": field,
                "slot": str(slot),
                "value": value,
                "reason": reason,
                "was_correction": is_correction,
            }
    return None


def _build_entities(merged: dict) -> ExtractedEntities:
    """Rebuild the entity bag from the merged dict, preserving conversation control.

    `last_asked_field` is deliberately NOT carried over — it is set by `_resolve`
    from this turn's decision, so a stale question can never be used to reinterpret a
    later message.

    `pending_changes` and `awaiting_change_detail` are dropped for the same reason and
    are re-set only by the modification branch. A proposal that outlived the turn it
    was made on would be a change the rider never confirmed, applied later against a
    message that was never about it.
    """
    return ExtractedEntities(
        destination=merged.get("destination"),
        origin=merged.get("origin"),
        duration_days=merged.get("duration_days"),
        travel_dates=merged.get("travel_dates"),
        group_type=merged.get("group_type"),
        group_size=merged.get("group_size"),
        budget_range=merged.get("budget_range"),
        interests=merged.get("interests") or [],
        special_requirements=merged.get("special_requirements") or [],
        visitor_type=merged.get("visitor_type"),
        return_journey_preference=merged.get("return_journey_preference"),
        home_city=merged.get("home_city"),
        rental_hours=merged.get("rental_hours"),
        pickup_time=merged.get("pickup_time"),
        cab_category=merged.get("cab_category"),
        origin_place=merged.get("origin_place"),
        destination_place=merged.get("destination_place"),
        asked_counts=dict(merged.get("asked_counts") or {}),
        proceed_requested=bool(merged.get("proceed_requested")),
        no_progress_turns=int(merged.get("no_progress_turns") or 0),
    )


def _known_slots(entities: ExtractedEntities) -> set[str]:
    """Which slots currently hold a value — used to tell progress from repetition."""
    return {name for name, slot in slots.SLOTS.items() if slot.read(entities)}


def _same_value(a: Any, b: Any) -> bool:
    """True when two slot values are the same, ignoring case for the string ones.

    Place names arrive title-cased from the code readers ("Dharamshala") and however
    the extraction model happened to capitalise them; "Kangra" and "kangra" are one
    slot value, and reading them as different would flag a change that never happened.
    """
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return bool(a == b)


def _resolve(
    state: AgentState,
    entities: ExtractedEntities,
    visitor_type: Optional[str],
    lang_key: str,
    unbookable_hours: Optional[int] = None,
    past_travel_date: Optional[str] = None,
) -> dict:
    """Apply the slot policy and package the node's return value.

    Shared by the LLM path and the LLM-failure path so a Vertex outage changes what
    the agent KNOWS, never how it BEHAVES.

    `unbookable_hours` is a duration the rider asked for that Him Cab does not sell, and
    `past_travel_date` a day they named that has already gone. Both travel on state so
    the question that goes back to them can NAME what could not be used — see
    `graph.clarify_node`. A question that just repeats itself is the thing both exist to
    prevent.
    """
    decision = slots.decide(
        entities,
        state.intent,
        visitor_type,
        lang_key,
        max_attempts=settings.max_clarification_attempts,
    )

    # Count the ask now, so both caps hold even when the user ignores the question.
    if decision.ask_field:
        entities.asked_counts[decision.ask_field] = (
            entities.asked_counts.get(decision.ask_field, 0) + 1
        )
    entities.last_asked_field = decision.ask_field

    return {
        "entities": entities,
        "needs_clarification": decision.needs_clarification,
        "clarification_question": decision.question,
        "visitor_type": visitor_type,
        "last_asked_field": decision.ask_field,
        "assumed_fields": list(decision.assumed),
        "unbookable_rental_hours": unbookable_hours,
        "past_travel_date": past_travel_date,
        "unplaceable_place": None,
        "node_trace": ["planner"],
    }


def _ask_about_the_change(
    state: AgentState,
    entities: ExtractedEntities,
    question: str,
    event: dict,
    visitor_type: Optional[str],
) -> dict:
    """A turn that asks about a modification and does nothing else.

    Bypasses `_resolve` on purpose. `slots.decide` answers "what is still missing?",
    and during a modification nothing is missing — the rider is changing something
    they already gave us. Routing this through the slot policy would replace the
    confirmation with whatever question happened to be next in the table.

    `last_asked_field` is cleared for the same reason: the reply coming back is a yes,
    a no, or a new value, and reading it as the answer to a slot question is how "nahi"
    would have become a destination.
    """
    entities.last_asked_field = None
    return {
        "entities": entities,
        "needs_clarification": True,
        "clarification_question": question,
        "visitor_type": visitor_type,
        "last_asked_field": None,
        "assumed_fields": [],
        "unbookable_rental_hours": None,
        "structured_events": [event],
        "node_trace": ["planner"],
    }


def _reask_unread(
    state: AgentState,
    entities: ExtractedEntities,
    visitor_type: Optional[str],
    lang_key: str,
    field: str,
    slot: slots.Slot,
    latest_text: str,
) -> dict:
    """Re-ask the one open question when the rider's reply could not be used at all.

    Bypasses `slots.decide` on purpose: `decide` answers "what is still missing?" from
    THIS turn's intent, and the whole problem is that the intent is a misread of an
    off-topic or typo'd reply — so the policy would ask the wrong thing, or nothing at
    all. Here we already know which question is open (`field`, the one we asked last
    turn) and simply put it again, prefaced with the rider's own words so it reads as "I
    didn't catch that" rather than the identical question fired twice.

    The attempt count is read BEFORE it is bumped, so the retry wording escalates on the
    same schedule as `slots.decide`, and the bump keeps the per-field cap honest.
    """
    attempt = entities.asked_counts.get(field, 0)
    question = slot.ask(lang_key, visitor_type, attempt)
    entities.asked_counts[field] = attempt + 1
    entities.last_asked_field = field
    log.info(
        "reply_unread_reasked",
        field=field,
        no_progress_turns=entities.no_progress_turns,
        session_id=state.session_id,
    )
    return {
        "entities": entities,
        "needs_clarification": True,
        "clarification_question": (
            f"{slots.unread_reply_preamble(latest_text, lang_key)} {question}"
        ),
        "visitor_type": visitor_type,
        "last_asked_field": field,
        "assumed_fields": [],
        "unbookable_rental_hours": None,
        "past_travel_date": None,
        "node_trace": ["planner"],
    }


def _ask_about_unplaceable(
    state: AgentState,
    entities: ExtractedEntities,
    visitor_type: Optional[str],
    lang_key: str,
    unplaceable: dict,
) -> dict:
    """Put a place we could not find back to the rider, quoting what they typed.

    Bypasses `slots.decide` for the reason `_reask_unread` does: the policy answers
    "what is missing next?", and here we already know which answer failed. Left to the
    policy, a rejected ORIGIN with no destination yet would ask for the destination and
    the rider would never learn their starting city had been dropped.

    Bounded like every other question. A REQUIRED slot (a trip's destination, a
    rental's pickup) is asked until it is answered — nothing can be produced without
    it, and inventing one is the failure this exists to prevent. Anything else stops
    being asked at the usual caps; the turn then goes ahead WITHOUT the place, and the
    reply says so (`slots.unplaceable_note`) rather than planning around a name the map
    has never heard of.
    """
    slot = unplaceable["slot"]
    needs = slots.needs_for(state.intent, visitor_type)
    required = needs.get(slot) is slots.Need.REQUIRED
    cap = settings.max_clarification_attempts
    patience_left = (
        entities.asked_counts.get(slot, 0) < cap
        and entities.no_progress_turns < cap
        and not entities.proceed_requested
    )

    if not (required or patience_left):
        log.info(
            "place_rejected_and_left_out",
            field=unplaceable["field"],
            value=unplaceable["value"],
            session_id=state.session_id,
        )
        result = _resolve(state, entities, visitor_type, lang_key)
        result["change_note"] = slots.unplaceable_note(unplaceable["value"], lang_key)
        result["unplaceable_place"] = unplaceable
        return result

    entities.asked_counts[slot] = entities.asked_counts.get(slot, 0) + 1
    entities.last_asked_field = slot
    if unplaceable["was_correction"]:
        # They asked to CHANGE this field and the new name did not land. The next reply
        # is the name they meant, so it is applied as the change it is instead of being
        # put to them for confirmation a second time (see the modification branch).
        entities.awaiting_change_detail = True

    return {
        "entities": entities,
        "needs_clarification": True,
        "clarification_question": slots.unplaceable_question(
            slot, unplaceable["value"], lang_key, unplaceable["reason"]
        ),
        "visitor_type": visitor_type,
        "last_asked_field": slot,
        "assumed_fields": [],
        "unbookable_rental_hours": None,
        "past_travel_date": None,
        "unplaceable_place": unplaceable,
        "node_trace": ["planner"],
    }


async def planner_node(state: AgentState) -> dict:
    """LangGraph Planner Node — extract, merge, classify, then decide."""
    log.info("planner_node_entered", session_id=state.session_id, intent=state.intent)

    lang_key, reply_language = detect_reply_language(state)
    latest_text = _latest_user_text(state)
    # Hiring a car by the hour, decided by the SERVICE the rider opened rather than by
    # this turn's intent — the router re-reads every message and calls half of a rental
    # chat `plan_trip`. Two things hang off it: the trip-shaped fields are dropped, and
    # the message is read for rental details in code as well as by the model.
    is_rental = RENTAL in (state.visitor_type, state.entities.visitor_type)
    # The single detail our last question asked about. The reply is only ever read
    # as an answer to THIS field.
    previously_asked = state.entities.last_asked_field
    known_before = _known_slots(state.entities)
    existing_entities = state.entities.model_dump(exclude_none=True)

    # ── Greeting fast path ────────────────────────────────────────────────────
    # A bare "hi" with nothing known yet has nothing to extract. Skipping the LLM
    # here saves a call and removes its best opportunity to hallucinate a trip.
    if _is_bare_greeting(latest_text) and not _has_trip_facts(state.entities):
        log.info("greeting_shortcircuit", session_id=state.session_id)
        greeting_entities = state.entities.model_copy(deep=True)
        # A greeting in reply to a question is a turn that told us nothing — it counts
        # towards the patience limit exactly like any other non-answer.
        if previously_asked:
            greeting_entities.no_progress_turns = state.entities.no_progress_turns + 1
        return _resolve(state, greeting_entities, state.visitor_type, lang_key)

    # ── A change we asked about last turn ─────────────────────────────
    # Read before the extraction model, not after: "haan" is unmistakable in code, and
    # sending it to a model to be classified adds a round trip and a way to be wrong
    # about the one turn where being wrong rewrites a booking screen.
    pending = dict(state.entities.pending_changes or {})
    confirmed_changes: dict[str, Any] = {}
    if pending:
        verdict = modification.reads_as_confirmation(latest_text)
        if verdict is True:
            confirmed_changes = pending
            log.info(
                "change_confirmed", fields=sorted(pending), session_id=state.session_id
            )
        elif verdict is False:
            # Declined. Keep every old value, and ask what they DID mean — a rider who
            # says "no" to our reading has still told us something is wrong, and
            # dropping it there leaves them exactly where the reported bug left them.
            declined = state.entities.model_copy(deep=True)
            declined.pending_changes = {}
            declined.awaiting_change_detail = True
            declined.no_progress_turns = 0
            log.info(
                "change_declined", fields=sorted(pending), session_id=state.session_id
            )
            return _ask_about_the_change(
                state,
                declined,
                modification.kept_message(lang_key)
                + "\n"
                + modification.which_detail_question(declined, lang_key),
                modification.which_detail_event(declined, lang_key),
                state.visitor_type,
            )
        else:
            # Neither yes nor no. The likeliest reason is that they answered with a
            # DIFFERENT value ("no, make it Kullu"), which the readers below handle far
            # better than a yes/no test can. The old proposal is dropped rather than
            # left hanging, so it cannot be applied by a later turn that was never
            # about it.
            log.info(
                "change_confirmation_unclear",
                fields=sorted(pending),
                session_id=state.session_id,
            )

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.messages[-10:]
    )
    today = datetime.now(UTC).strftime("%A, %d %B %Y")
    asked_context = (
        f"\nThe assistant's last question asked about: {previously_asked}. The "
        f"user's latest message is most likely the answer to it (it may be a short "
        f"reply or contain a typo)."
        if previously_asked else ""
    )

    # Named in the prompt rather than left to be inferred from the transcript: the
    # extractor's rules for a rental are different, and "rental" in the persona line was
    # being read as just another kind of traveller.
    service_line = (
        "\nRequest type: RENTAL — a car and driver BY THE HOUR. There is no destination "
        "and no trip length. Read the RENTAL CONVERSATION rules below."
        if is_rental else ""
    )

    prompt = f"""Today's date: {today}
Conversation language: {reply_language}
Visitor classification so far: {state.visitor_type or 'unknown'}{service_line}{asked_context}

Conversation so far:
{conversation_text}

Already extracted (carry these forward; do not null them out):
{json.dumps(existing_entities, indent=2, default=str)}

Report what the user has stated. Use null for anything they have not."""

    try:
        # `json_mode` rather than the default function calling, and `_VertexExtraction`
        # rather than `PlannerExtraction` — both are required together, and the note on
        # that class says why. In short: the default combination returned None on about
        # half of all calls and this one has not returned None yet.
        # `structured` rather than `get_flash_model(...).with_structured_output(...)`:
        # same runnable, with the standby model behind it for when the primary's quota
        # pool is empty. See `vertex_ai.structured`.
        llm = get_vertex_ai_service().structured(_VertexExtraction, method="json_mode")
        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        raw_result = None
        for attempt in range(_EXTRACTION_ATTEMPTS):
            raw_result = await asyncio.wait_for(
                call_model(llm.ainvoke(messages)),
                timeout=settings.llm_call_timeout_seconds,
            )
            if raw_result is not None:
                break
            # A None here is the model declining to answer in the required shape. It is
            # not an error and raises nothing — it used to reach `dict(None)` and come
            # out as a TypeError, which read in the logs like a bug in this file.
            # Retried because it is transient; counted because a rising rate means the
            # schema and the model have drifted apart again.
            log.warning(
                "planner_extraction_empty",
                attempt=attempt + 1,
                session_id=state.session_id,
            )
            if attempt + 1 < _EXTRACTION_ATTEMPTS:
                await _backoff(attempt)

        if raw_result is None:
            raise ValueError(
                f"the model returned no parseable object in "
                f"{_EXTRACTION_ATTEMPTS} attempts"
            )
        if not isinstance(raw_result, _VertexExtraction):
            # LangChain types this return as `dict | BaseModel`. Anything else here is
            # a version change in how structured output is handed back, and guessing at
            # it would put whatever arrived straight into the rider's entities.
            raise TypeError(
                f"extraction returned {type(raw_result).__name__}, "
                f"expected _VertexExtraction"
            )

        result_dict = raw_result.to_extraction().model_dump()
        llm_ok = True

    except ResourceExhausted as e:
        # THROTTLED, not broken. Logged under its own name because the two need
        # different responses and used to be indistinguishable: a rising
        # `planner_extraction_failed` rate means the schema and the model have drifted
        # apart, a rising `vertex_throttled` means the quota pool is full and no amount
        # of staring at this file will help.
        #
        # Nothing is retried here. LangChain has already retried with backoff by the
        # time this is raised, and a further re-ask would be one more request into a
        # pool that has just said it has no room.
        log.warning(
            "vertex_throttled",
            node="planner",
            error=str(e)[:200],
            session_id=state.session_id,
        )
        result_dict = {}
        llm_ok = False

    except Exception as e:
        # Degrade, don't derail: keep what we know, read the reply in code, and let
        # the same policy decide. The user sees a slightly less clever turn, not an error.
        log.error(
            "planner_extraction_failed",
            error=str(e),
            error_type=type(e).__name__,
            session_id=state.session_id,
        )
        result_dict = {}
        llm_ok = False

    # ── Merge: new facts win, known facts are never nulled out ────────────────
    merged = {**existing_entities}
    control_keys = ("confidence", "user_wants_to_proceed_now", "origin_in_himachal")
    for key, val in result_dict.items():
        if key not in control_keys and val not in (None, []):
            merged[key] = val

    # A change the rider confirmed last turn, applied over whatever the model still
    # believes. It has to land AFTER the merge above: the extraction model re-reports
    # the whole conversation every turn and will happily re-assert the old value it has
    # seen five times, which is exactly how the confirmed change used to evaporate.
    if confirmed_changes:
        merged.update(confirmed_changes)

    if is_rental:
        _strip_trip_fields(merged, state.session_id)

    # ── A pickup the rider PICKED in the app ──────────────────────────────────
    # The app's pickup dialog — a place search, or "use my current location" — knows the
    # exact point, and it used to reach us as a town name: the only thing the API could
    # carry for a pickup was words, so the app trimmed a pick to three of them, or to the
    # town. Picked at the hotel gate, priced and pinned at the town centre. Now the point
    # itself arrives (`state.picked_pickup`), and is filed here, ahead of everything that
    # reads `origin`, so the classification below is made from it and not from whatever
    # the extraction model made of the address text.
    picked = _accept_pick(state.picked_pickup, "origin", state, existing_entities, previously_asked)
    if picked is not None:
        merged["origin"] = picked.name
        merged["origin_place"] = picked.model_dump()

    # ── A drop the rider PICKED in the app ────────────────────────────────────
    # The same, from the other end. The app's drop search knows the exact point and could
    # only send its name, which was then found again by name — the town centre at best,
    # a namesake somewhere else at worst — and priced and pinned there. Never on a
    # rental: it has no drop (`_NOT_IN_A_RENTAL`).
    picked_drop = (
        None if is_rental
        else _accept_pick(state.picked_drop, "destination", state, existing_entities, previously_asked)
    )
    if picked_drop is not None:
        merged["destination"] = picked_drop.name
        merged["destination_place"] = picked_drop.model_dump()

    # ── Visitor classification (code, from what they told us) ─────────────────
    # `current` is passed so an in-state ORIGIN cannot promote a visitor to resident;
    # only a stated home city does that. See `classify_visitor_type`.
    # A rental never uses this: `classify_visitor_type` returns RENTAL before it reads
    # `origin_in_himachal` (the pickup says nothing about who the rider is), and the
    # `home_city` promotion below is LOCAL-only. Geocoding the pickup here would be a
    # wasted Nominatim round-trip on the planner's critical path.
    origin_in_hp = result_dict.get("origin_in_himachal")
    # A picked pickup answers this from the point itself, and is looked up — if it has
    # to be — by its town: the geocoder finds "Shimla", not "Hotel Snow View, The Mall".
    # On every turn it is still the pickup, not only the one it was picked on.
    origin_for_lookup = merged.get("origin")
    filed = _filed_pick(merged)
    if filed is not None:
        if filed.in_himachal is not None:
            origin_in_hp = filed.in_himachal
        origin_for_lookup = filed.town or filed.name
    if origin_in_hp is None and origin_for_lookup and not is_rental:
        origin_in_hp = await geo.place_in_himachal(origin_for_lookup)

    visitor_type = (
        await classify_visitor_type(
            origin_for_lookup,
            origin_in_hp,
            home_city=merged.get("home_city"),
            current=state.visitor_type,
        )
        or state.visitor_type
    )
    if visitor_type:
        merged["visitor_type"] = visitor_type
        # When a rider we ALREADY treat as a resident names an in-state pickup, it is
        # recorded as their home city too, so the classification survives later turns —
        # a resident who later changes the pickup to Chandigarh is still on the local
        # flow they opened. (This dates from when the local flow asked where they LIVE;
        # it now asks for the pickup, and the recording is kept for that stickiness.)
        # Deliberately not done for a visitor: their in-state origin is where the trip
        # starts, not where they live, and writing it here used to turn them into a local.
        # The town, for a picked pickup: a home city is somewhere a rider lives, and
        # nobody lives at "Hotel Snow View, The Mall".
        if visitor_type == LOCAL and origin_in_hp and not merged.get("home_city"):
            merged["home_city"] = origin_for_lookup

    # ── "Here" — the pickup answered by pointing rather than naming ───────────
    # Every flow, and only as a reply to that exact question. Every flow now asks
    # "आपको कहाँ से पिकअप करना है?", and "here" / "yahin se" / "my location" is a reply
    # riders genuinely give — which used to parse as a PLACE called "Here", which the
    # geocoder resolved to something and the car was sent to. (Rental-only until every
    # flow asked for a pickup: a resident asked where they LIVE did not answer "here".)
    #
    # The fix is the honest reading of it, and a better one than the pickup this flow
    # used to assume: exact coordinates the rider has just confirmed, rather than a
    # town centre nobody was asked about. It is accepted only when the fix is precise
    # — `is_precise` is what makes `place_name` a name worth saying out loud, and the
    # reply announces the pickup. Without one we would be guessing at a name and
    # sending a car on it, so the question is simply asked again.
    if (
        latest_text
        and picked is None  # a pick already said exactly where; "here" adds nothing
        and previously_asked == SlotName.ORIGIN
        and not merged.get("origin")
        and slots.means_current_location(latest_text)
    ):
        fix = state.rider_location
        if fix is not None and fix.is_precise and fix.place_name:
            merged["origin"] = fix.place_name
            log.info(
                "pickup_from_device_fix",
                place=fix.place_name,
                distance_km=fix.place_distance_km,
                session_id=state.session_id,
            )
        else:
            # Deliberately fills nothing: `decide` then re-asks, and the second wording
            # ("just the pickup location — a city, town or area") is exactly the right
            # thing to say to somebody whose "here" we could not place.
            log.info(
                "pickup_here_without_usable_fix",
                has_fix=fix is not None,
                is_precise=bool(fix and fix.is_precise),
                session_id=state.session_id,
            )

    # ── Read the reply as an answer to the question we asked ──────────────────
    # Strictly scoped: only the fields that question covered, only if the reply has
    # the right shape. A safety net for a missed one-word answer, not an interpreter.
    corrected: dict[str, Any] = {}
    bare_correction: Optional[str] = None

    recovered = slots.read_answer(previously_asked, latest_text, merged, visitor_type) if latest_text else None
    if recovered:
        log.info("short_answer_recovered", field=recovered, session_id=state.session_id)
        # Skipped for a rental for the same reason as the classification above: the
        # pickup does not decide the persona (`classify_visitor_type` returns RENTAL),
        # so geocoding it and re-classifying would change nothing but the latency.
        if recovered == "origin" and not is_rental:
            # An origin recovered in code carries the same weight as an extracted one.
            recovered_in_hp = await geo.place_in_himachal(merged["origin"])
            visitor_type = await classify_visitor_type(
                merged.get("origin"),
                recovered_in_hp,
                home_city=merged.get("home_city"),
                current=visitor_type,
            ) or visitor_type
            if visitor_type:
                merged["visitor_type"] = visitor_type
                if visitor_type == LOCAL and recovered_in_hp and not merged.get("home_city"):
                    merged["home_city"] = merged.get("origin")

    # ── Read the reply as a CORRECTION to the question we asked ────────────
    # `read_answer` above only fills an EMPTY slot. A bare reply that changes one that
    # is already filled — "nahi, Kasol" after the destination is in — landed nowhere.
    # Guarded to the field the assistant itself asked last turn — that is what makes a
    # bare reply unambiguous. Like the sentence reader below it PROPOSES rather than
    # writes, because it overwrites a value the rider already gave us.
    if latest_text and previously_asked:
        asked_slot = slots.SLOTS.get(previously_asked)
        stored = existing_entities.get(previously_asked)
        if asked_slot:
            # Read against `stored`, not against `merged`: the text is what establishes
            # that a change was asked for, and it says so whether or not the extraction
            # model also noticed. Reading against `merged` made the behaviour depend on
            # the model — the same reply confirmed on one turn and applied silently on
            # the next, for no reason the rider could see.
            new_value = slots.read_correction_to_asked(
                previously_asked, latest_text, stored
            )
            if new_value is not None:
                # The model's reading of the same reply wins WHAT is proposed when it
                # moved the field too; the text only had to establish THAT it moved.
                model_value = merged.get(previously_asked)
                if model_value is not None and not _same_value(model_value, stored):
                    new_value = model_value
                # Proposed, not written. Same rule as the sentence block below: this
                # overwrites a value the rider already gave, so they get to see our
                # reading of it before a booking screen is built on it.
                corrected[previously_asked] = new_value
                bare_correction = previously_asked
                log.info(
                    "bare_answer_correction_proposed",
                    field=previously_asked,
                    session_id=state.session_id,
                )

    # ── Read the whole sentence, in code ──────────────────────────────────────
    # `read_answer` above handles a BARE reply to the question in flight ("manali").
    # This handles the other thing riders actually type — a sentence — which used to
    # depend entirely on the extraction model. A miss there was silent: the detail
    # vanished, and the assistant asked for something it had just been told, which is
    # what a rider experiences as not being listened to.
    #
    # Reported twice, once per flow. For rentals it was "I need a car tomorrow at 9 am
    # from Shimla for 8 hours". For trips it was the very first turn: the greeting asks
    # "आप कहाँ जाना चाहते हैं?", the rider answers "mei shimla jana chahta hu", and the
    # same question comes back — because `parse_place` caps an answer at three words on
    # purpose, so a bare "shimla" was recovered and a sentence containing it was not.
    #
    # Every value here is anchored on an unmistakable marker — see the two readers in
    # `slots`. A marker-anchored value fills an EMPTY slot, and it also CORRECTS a slot
    # that already holds one when the rider changes their mind — see `corrected` below.
    if latest_text:
        reader = slots.read_rental_details if is_rental else slots.read_trip_details
        filled: dict[str, Any] = {}
        for field, value in reader(latest_text).items():
            stored = existing_entities.get(field)
            if stored in (None, "", [], {}):
                # Nothing there yet — an ANSWER. Fills the blank, takes nothing away,
                # and needs no confirmation.
                #
                # Unless the model filled the same blank this turn with something else.
                # The readers cap a place at a few words after a marker and cannot tell
                # "go TO Kasol" from "places TO eat in Manali": the second came back as
                # destination "Eat", overwrote the model's "Manali", and was planned as
                # a trip. On a blank slot the model's reading of the whole sentence
                # wins, as it already does below for a change. The reader is the
                # safety net for when the model says nothing, not a second opinion.
                #
                # The MODEL's value — `result_dict`, not `merged`. `merged` also holds
                # what the bare-answer parser just wrote, and that parser is weaker than
                # this reader on a sentence: "pickup Shimla drop Manali" parsed there as
                # the one place "Shimla Manali", and deferring to it threw away this
                # reader's correct two ends.
                model_value = result_dict.get(field)
                if model_value not in (None, "", [], {}) and not _same_value(
                    model_value, value
                ):
                    continue
                filled[field] = value
            elif not _same_value(value, stored):
                # A CHANGE: the rider has restated a slot as something different from
                # what the session holds. Every detail may already be in and the
                # booking already on screen; "mujhe dharamshala jaana hai" moves it.
                #
                # Compared against `stored` — the value as it was BEFORE this turn —
                # rather than against `merged`, which the extraction model has already
                # written to. That is what makes the detection uniform: whether the
                # model caught the change or missed it, the rider sees the same
                # confirmation. It used to depend on which, so the same sentence was
                # sometimes applied silently and sometimes not.
                #
                # The model's value wins WHAT is proposed when it moved the field too:
                # reading a whole sentence is its strength, and these regexes cap a
                # place at three words. The reader's job here is to establish THAT
                # something changed, which it can do from the text alone — a
                # model-only disagreement is far more often a re-wording ("kal" ->
                # "tomorrow") than a change the rider asked for.
                model_value = merged.get(field)
                corrected[field] = (
                    model_value
                    if model_value is not None and not _same_value(model_value, stored)
                    else value
                )

        # `filled` goes in unconditionally and `corrected` does not. That is the whole
        # distinction this flow turns on: filling a blank takes nothing away, while
        # changing a value overwrites something the rider already gave us and is
        # already looking at. The second one is proposed and confirmed below; see
        # app/agent/modification.py for why it is worth the extra turn.
        #
        # Neither may touch a picked pickup: the address text the app sent with it would
        # be read as a place here ("pickup Hotel Snow View, The Mall" → "Hotel Snow View")
        # and written over the point the rider actually chose. Nor is the pick itself a
        # change to confirm — the bare-answer reader above may have proposed one from
        # that same text — because the rider tapped it in reply to the pickup question.
        if picked is not None:
            filled.pop("origin", None)
            corrected.pop("origin", None)
        # A picked drop, the same way: "drop Hotel Snow View" read out of its address
        # text must not be written over the point, nor proposed as a change.
        if picked_drop is not None:
            filled.pop("destination", None)
            corrected.pop("destination", None)
        if filled:
            merged.update(filled)
        for field in corrected:
            # Put back what the session held. The model may have written its own change
            # into `merged` several steps ago, and leaving it there would apply the very
            # change we are about to ask about — the confirmation would be theatre.
            merged[field] = existing_entities.get(field)
        if filled or corrected:
            log.info(
                "sentence_read_in_code",
                flow="rental" if is_rental else "trip",
                filled=sorted(filled),
                proposed=sorted(corrected),
                llm_ok=llm_ok,
                session_id=state.session_id,
            )

        # A changed pickup re-decides who the rider is, exactly as a recovered or
        # extracted origin does above: "actually, start from Delhi" turns a resident's
        # ride into a visitor's transfer. Skipped for a rental, where the pickup never
        # sets the persona (`classify_visitor_type` returns RENTAL regardless).
        if "origin" in filled and not is_rental:
            corrected_in_hp = await geo.place_in_himachal(merged["origin"])
            visitor_type = await classify_visitor_type(
                merged.get("origin"),
                corrected_in_hp,
                home_city=merged.get("home_city"),
                current=visitor_type,
            ) or visitor_type
            if visitor_type:
                merged["visitor_type"] = visitor_type
                if visitor_type == LOCAL and corrected_in_hp and not merged.get("home_city"):
                    merged["home_city"] = merged.get("origin")

    # ── "Stop asking and just do it" ──────────────────────────────────────────
    if result_dict.get("user_wants_to_proceed_now") or (
        not llm_ok and _looks_like_proceed_request(latest_text)
    ):
        merged["proceed_requested"] = True
        log.info("proceed_requested", session_id=state.session_id)

    # ── "The one you suggested is fine" ───────────────────────────────────────
    # The cab question lists the cabs that seat THIS party, recommended one first. A
    # rider who answers "haan book kar do", "ok" or "aap batao" has chosen — the cab we
    # recommended — and used to be told we had not understood them, with no booking
    # screen until they typed its name back. Read here only as a reply to that exact
    # question; the tool executor, which knows which cab was recommended, applies it.
    accepted_recommended_cab = bool(
        is_rental
        and latest_text
        and previously_asked == SlotName.CAB_CATEGORY
        and not merged.get("cab_category")
        and (
            slots.accepts_recommendation(latest_text)
            or _looks_like_proceed_request(latest_text)
            or result_dict.get("user_wants_to_proceed_now")
        )
    )
    if accepted_recommended_cab:
        log.info("recommended_cab_accepted", session_id=state.session_id)

    # The booking screen wants a DATE in its date field, not the sentence the rider
    # happened to answer with. Rentals only: a trip's "sometime after Diwali" is worth
    # keeping word for word, and nothing renders it into a form.
    if is_rental and merged.get("travel_dates"):
        merged["travel_dates"] = slots.tidy_when(merged["travel_dates"])

    # ── The validation gate ───────────────────────────────────────────────────
    # Everything above WRITES into `merged` — the extraction model, the answer parsers
    # and the sentence reader — and until this existed only the last two checked what
    # they wrote. The model's output went straight through, so a group of 500, a
    # negative trip length, a pickup of "Here" or a travel date of "yesterday" reached
    # the fare request and the booking screen unopposed.
    #
    # Placed here, after every writer and before `_build_entities`, so there is exactly
    # one gate and no route around it. A value that fails is dropped rather than
    # corrected: the slot is empty again, so `slots.decide` asks for it on this very
    # turn instead of the rider discovering the problem at the booking screen.
    #
    # For a rental this is also where "tomorrow" becomes a real date — see
    # `sanitize.travel_date`.
    # Asked BEFORE the gate, because the gate is what empties the field: afterwards
    # there is nothing left to tell the rider we could not use.
    past_day = sanitize.day_already_gone(merged.get("travel_dates"))

    merged, rejected = sanitize.sanitize_entities(merged, is_rental=is_rental)
    if rejected:
        log.info(
            "entity_values_rejected",
            fields=rejected,
            session_id=state.session_id,
        )
    if past_day:
        log.info("travel_date_already_gone", value=past_day, session_id=state.session_id)

    # ── A place that is not on the map, refused on the turn it arrives ────────
    # After the gate, so only a well-formed name costs a lookup, and before anything
    # can act on it. See `_reject_unplaceable`.
    unplaceable = await _reject_unplaceable(merged, corrected, state, visitor_type)

    # ── A duration we do not sell, refused on the turn it arrives ─────────────
    unbookable_hours = (
        await _reject_unsold_duration(merged, state.rider_jwt, state.session_id)
        if is_rental else None
    )

    entities = _build_entities(merged)

    # ── Progress tracking ─────────────────────────────────────────────────────
    # "They replied but told us nothing new." Counting these is what lets the agent
    # notice it is getting nowhere and deliver instead of asking again.
    #
    # The condition is deliberately NOT "…and we asked a question last turn". It was,
    # and that made giving up last exactly one turn: the agent stopped asking, which
    # cleared `last_asked_field`, which reset the streak, which let it start asking
    # again on the very next message. The user got interrogated, then a plan, then
    # interrogated again. Only real progress — a slot that gained a value — resets it.
    # A correction is progress too, even though it fills no NEW slot: the rider handed
    # us a real, different value. Without `or corrected`, a turn that only changes an
    # already-known detail would count as no progress and inch the agent towards giving
    # up on a rider who is actively engaged.
    made_progress = bool(
        (_known_slots(entities) - known_before)
        or corrected
        or bare_correction
        or confirmed_changes
        # Taking the recommended cab is an answer, even though it fills no slot until
        # the tool executor writes the cab in — without this the reply read as no
        # progress and the "didn't catch that" re-ask fired on a perfectly clear yes.
        or accepted_recommended_cab
        # A pickup or drop tapped in the app is an answer even when it replaces one the
        # question was re-asking about, which fills no NEW slot.
        or picked is not None
        or picked_drop is not None
    )
    if made_progress:
        entities.no_progress_turns = 0
    else:
        entities.no_progress_turns = state.entities.no_progress_turns + 1

    # ── A place we could not find goes back to the rider, by name ─────────────
    # Ahead of the modification branch: "nahi, Kasoll" is a change the rider asked for
    # with a name we cannot place, and without this it fell through to "which detail
    # would you like to change?" — asking about the one thing they had just told us.
    if unplaceable and not corrected:
        return _ask_about_unplaceable(state, entities, visitor_type, lang_key, unplaceable)

    # ── A change the rider has asked for ───────────────────────────────
    # `corrected` holds values that would OVERWRITE something already on the session.
    # Three outcomes, and which one applies is decided in code:
    #
    #   They already told us they wanted to change something and we asked what — this
    #   IS the what, so it goes straight in. Asking "are you sure?" about an answer to
    #   "what would you like to change?" is how a correction becomes an interrogation.
    #
    #   Otherwise we put our reading of it to them before acting on it. The value we
    #   read is the thing most likely to be wrong, and the reported bug is what a bad
    #   one looks like once it reaches a booking screen: a ride from "Mujhe Palampur"
    #   to "Lekar Kasol". Shown as a question, that misparse is obvious to the one
    #   person who can fix it.
    #
    #   And a rider who says they want a change without saying which gets asked which,
    #   with the current details listed so they can point at one rather than remember.
    if corrected and state.entities.awaiting_change_detail:
        for field, value in corrected.items():
            setattr(entities, field, value)
        log.info(
            "change_applied_on_request",
            fields=sorted(corrected),
            session_id=state.session_id,
        )
    elif corrected:
        pending_entities = entities.model_copy(deep=True)
        pending_entities.pending_changes = dict(corrected)
        log.info(
            "change_proposed",
            fields=sorted(corrected),
            session_id=state.session_id,
        )
        return _ask_about_the_change(
            state,
            pending_entities,
            # Against `state.entities` — the values as they stood BEFORE this turn.
            # `entities` already carries anything newly filled, and a confirmation that
            # says "destination: Kasol → Kasol" tells the rider nothing.
            modification.confirm_question(corrected, state.entities, lang_key),
            modification.confirmation_event(corrected, state.entities, lang_key),
            visitor_type,
        )
    elif latest_text and modification.wants_to_change_something(latest_text):
        asking = entities.model_copy(deep=True)
        asking.awaiting_change_detail = True
        asking.no_progress_turns = 0
        log.info("change_requested_without_a_field", session_id=state.session_id)
        return _ask_about_the_change(
            state,
            asking,
            modification.which_detail_question(asking, lang_key),
            modification.which_detail_event(asking, lang_key),
            visitor_type,
        )

    # ── A reply we could not use at all ────────────────────────────────────────
    # The rider answered, but nothing in it was usable — not the field we asked about,
    # not any other slot, not a correction, not "just get on with it". In a booking flow
    # the old behaviour handed this turn to the free-text writer, which then answered a
    # question the rider never asked: the reported "typo / something off-topic → the flow
    # hallucinates an irrelevant reply". It happens because a misread reply is classified
    # as a general question, and `slots.decide` then asks for none of the booking slots.
    #
    # So the pending question outranks a misclassified intent here: re-ask it, NAMING that
    # we could not read the reply so it does not read as a robotic repeat (§7). Wholly
    # deterministic — no model runs on this path, so a typo can never become an invented
    # answer. Bounded by the SAME patience cap as every other question
    # (`no_progress_turns`), so a rider who keeps typing past us still reaches a plan or a
    # booking screen rather than an endless re-ask. Scoped to the booking flows the report
    # is about; an outstation visitor's general question is still answered.
    reasked_slot = slots.SLOTS.get(previously_asked) if previously_asked else None
    if (
        previously_asked is not None
        and reasked_slot is not None
        and not made_progress
        and not entities.proceed_requested
        and visitor_type in (slots.LOCAL, slots.RENTAL)
        and not reasked_slot.read(entities)
        and entities.no_progress_turns < settings.max_clarification_attempts
        # A day that has gone or a duration Him Cab does not sell was CAUGHT — the rider
        # answered, the answer just cannot be used — and `_resolve` re-asks NAMING that
        # (§7). Those beat this generic "didn't catch that", which is for a reply we could
        # not read at all.
        and unbookable_hours is None
        and past_day is None
        # A genuine question is answered, never turned into an interview — the deliberate
        # `TestQuestionsDoNotHijackAnswers` behaviour. This guard is only for a reply that
        # was an attempted ANSWER and failed (a typo, an off-topic line), which is where
        # the free-text writer was inventing irrelevant prose.
        and not slots._looks_like_a_question(latest_text)
    ):
        return _reask_unread(
            state, entities, visitor_type, lang_key,
            previously_asked, reasked_slot, latest_text,
        )

    result = _resolve(
        state, entities, visitor_type, lang_key, unbookable_hours, past_day
    )
    result["accepted_recommended_cab"] = accepted_recommended_cab

    # A change that just landed is stated in the reply, so the rider can see it took.
    # Everything downstream — route, fare, weather, booking handoff — is rebuilt from
    # the entities each turn, so the reply already reflects the new value; what it has
    # no way to say on its own is that the value MOVED, and why.
    if confirmed_changes or (corrected and state.entities.awaiting_change_detail):
        result["change_note"] = modification.applied_message(
            confirmed_changes or corrected, lang_key
        )

    log.info(
        "entities_extracted",
        destination=entities.destination,
        origin=entities.origin,
        duration=entities.duration_days,
        group_size=entities.group_size,
        travel_dates=entities.travel_dates,
        rental_hours=entities.rental_hours,
        pickup_time=entities.pickup_time,
        cab_category=entities.cab_category,
        visitor_type=visitor_type,
        intent=state.intent,
        needs_clarification=result["needs_clarification"],
        asking=result["last_asked_field"],
        previously_asked=previously_asked,
        assumed=result["assumed_fields"],
        proceed_requested=entities.proceed_requested,
        no_progress_turns=entities.no_progress_turns,
        llm_ok=llm_ok,
        session_id=state.session_id,
    )
    if not llm_ok:
        result["error"] = "planner extraction unavailable"
    return result


__all__ = [
    "PLANNER_SYSTEM_PROMPT",
    "PlannerExtraction",
    "classify_visitor_type",
    "planner_node",
]
