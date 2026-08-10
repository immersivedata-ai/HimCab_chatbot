"""
app/agent/slots.py
───────────────────
What the agent needs to know — the single source of truth for the conversation.

Every piece of information the agent collects is a SLOT. This module owns three
things about each slot, in one place, so they can never drift apart:

  1. how important it is  — per kind of request (see `Need` and `_NEEDS`)
  2. how it is asked      — the question text, per reply language and persona
  3. how a reply is read  — a strict parser for the case where the extraction LLM
                            misses a bare one-word answer ("5", "manali", "kal")

The planner owns *when* to ask; this module owns *what* and *whether*.

WHY THIS EXISTS
───────────────
The clarification flow used to be a fixed six-item checklist that every request
was pushed through, regardless of what the user had actually asked for. Three
failures came out of that, and this module is shaped to make each impossible:

  • "It only asks questions."      A weather question was answered with "which
    city are you travelling from?", because the trip checklist ran for every
    intent. Fixed by `Need.UNUSED`: a slot that does not matter for this kind of
    request is never asked. See `_NEEDS`.

  • "It never generates."          Six mandatory answers meant six round trips,
    and any field the user would not answer blocked the itinerary forever. Fixed
    by `Need.IMPORTANT`/`Need.OPTIONAL`: only a destination is truly mandatory,
    everything else is escapable — after a bounded number of attempts, or the
    moment the user asks us to get on with it.

  • "It assumes things."           A free-form fallback wrote whatever the user
    typed into whichever field was next on the checklist ("plan a trip" became
    the destination). Fixed by `parse`: each slot validates the shape of an
    answer and returns None when it does not fit, so a bad reply produces no
    value rather than a wrong one. Rejecting a real answer only costs one extra
    question; accepting a non-answer corrupts the whole trip — the parsers are
    deliberately biased towards rejecting.

Anything the agent proceeds without is reported back as `Decision.assumed` so the
writer can state the assumption out loud instead of quietly inventing a fact.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional

from app.models.session import ExtractedEntities

# ── Personas ──────────────────────────────────────────────────────────────────
LOCAL = "local"


class Need(str, Enum):
    """How badly a slot is needed for a given kind of request."""

    REQUIRED = "required"    # nothing useful exists without it — asked until answered
    IMPORTANT = "important"  # materially better with it — asked, but escapable
    OPTIONAL = "optional"    # nice to have — never blocks, offered as a follow-up
    UNUSED = "unused"        # irrelevant to this request — never asked


# ── Answer parsers ────────────────────────────────────────────────────────────
# These run ONLY on a direct reply to a question we just asked, and only when the
# extraction LLM failed to pick the answer up. Each returns None unless the reply
# clearly has the right shape. See the module docstring on why they reject freely.

# Replies that are grammatically fine but carry no information. Used to reject,
# never to interpret — an unlisted non-answer just costs one more question.
_NON_ANSWERS = {
    "yes", "no", "ok", "okay", "sure", "hmm", "hm", "idk", "dunno", "maybe",
    "anything", "any", "whatever", "nothing", "none", "later", "dont", "don't",
    "know", "not", "sure?", "hi", "hello", "hey", "yo", "hola",
    "haan", "ha", "han", "ji", "nahi", "nahin", "nhi", "na", "pata", "malum",
    "kuch", "bhi", "koi", "batao", "bata", "aap", "tum", "namaste", "namaskar",
}

# Words that mean the user is describing the *request* rather than naming a place.
# A place parser that accepts these is how "plan a trip" became a destination.
_NOT_A_PLACE = {
    "plan", "planning", "trip", "tour", "travel", "travelling", "traveling",
    "holiday", "vacation", "visit", "go", "going", "want", "wanna", "need",
    "make", "create", "generate", "show", "give", "tell", "help", "book",
    "somewhere", "something", "anywhere", "hills", "hill", "mountains",
    "mountain", "himachal", "india", "place", "places", "city", "town",
    "ghumna", "jaana", "jana", "chahiye", "banao", "bana", "do", "karo",
}

_QUESTION_HINTS = ("?", "kya ", "kaise", "kahan ", "kaunsa", "what ", "which ", "how ", "why ")

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _looks_like_a_question(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in _QUESTION_HINTS)


def _is_non_answer(text: str) -> bool:
    toks = _tokens(text)
    return not toks or all(t in _NON_ANSWERS for t in toks)


def parse_place(text: str) -> Optional[str]:
    """A place name, or None. Accepts 1-3 alphabetic words that name something.

    Rejects date words too. A reply does not always answer the question that was
    asked — someone asked for their city may volunteer "next friday" — and filing
    that as the starting city is worse than not parsing it at all.
    """
    text = text.strip().strip(".,!")
    if not text or _looks_like_a_question(text) or any(c.isdigit() for c in text):
        return None
    toks = _tokens(text)
    if not 1 <= len(toks) <= 3 or len("".join(toks)) < 3:
        return None
    if any(t in _NOT_A_PLACE or t in _NON_ANSWERS or t in _TIME_WORDS for t in toks):
        return None
    return text.title()


def parse_duration_days(text: str) -> Optional[int]:
    """A trip length in days, or None. Handles '5', '5 days', '5 din', 'a week'."""
    low = text.strip().lower()
    if _looks_like_a_question(low):
        return None

    m = re.fullmatch(r"(\d{1,2})\s*(days?|day|din|dino|dinon|दिन)?", low)
    if m:
        return _in_range(int(m.group(1)), 1, 30)

    m = re.fullmatch(r"(a|an|one|1|ek|do|two|2|teen|three|3)\s*(weeks?|hafta|hafte|हफ्ता|हफ्ते)", low)
    if m:
        weeks = {"a": 1, "an": 1, "one": 1, "1": 1, "ek": 1, "do": 2, "two": 2, "2": 2,
                 "teen": 3, "three": 3, "3": 3}[m.group(1)]
        return weeks * 7

    if re.fullmatch(r"(a\s+)?(week\s*end|weekend|weekend\s*trip)", low):
        return 2
    return None


def parse_group_size(text: str) -> Optional[int]:
    """A headcount, or None. Handles '4', '4 people', '4 log'."""
    low = text.strip().lower()
    if _looks_like_a_question(low):
        return None
    m = re.fullmatch(
        r"(\d{1,2})\s*(people|persons?|pax|members?|adults?|guests?|log|logo|लोग)?", low
    )
    return _in_range(int(m.group(1)), 1, 60) if m else None


# Words that mark a reply as being about WHEN. Used both to accept a date and, in
# `parse_place`, to reject one — the parsers police each other's boundaries so a
# reply can never be filed against a field it doesn't answer.
_TIME_WORDS = {
    "today", "tomorrow", "tonight", "weekend", "week", "weeks", "month", "months",
    "next", "this", "coming", "after", "soon", "later", "end", "start", "mid",
    "holidays", "vacation", "diwali", "christmas", "newyear", "summer", "winter",
    "monsoon", "spring", "season",
    "kal", "aaj", "parso", "parson", "agle", "agla", "agli", "iss", "is", "hafte",
    "hafta", "mahine", "mahina", "saal", "baad", "sardi", "garmi", "barish", "chutti",
    "chuttiyan",
    "jan", "january", "feb", "february", "mar", "march", "apr", "april", "may",
    "jun", "june", "jul", "july", "aug", "august", "sep", "sept", "september",
    "oct", "october", "nov", "november", "dec", "december",
    "mon", "monday", "tue", "tues", "tuesday", "wed", "wednesday", "thu", "thur",
    "thurs", "thursday", "fri", "friday", "sat", "saturday", "sun", "sunday",
    "कल", "आज", "परसों", "अगले", "अगला", "इस", "हफ्ते", "हफ्ता", "महीने", "महीना",
}


def parse_when(text: str) -> Optional[str]:
    """A rough travel time ('kal', 'next week', '15 Jan'), kept verbatim, or None.

    Requires the reply to actually look like a time — a date, or a word from
    `_TIME_WORDS`. Anything else is left to the extraction model rather than
    guessed at.
    """
    stripped = text.strip()
    if not stripped or _looks_like_a_question(stripped) or _is_non_answer(stripped):
        return None
    if len(_tokens(stripped)) > 6:
        return None
    if any(c.isdigit() for c in stripped):        # "15 Jan", "3rd week of May"
        return stripped
    if set(_tokens(stripped)) & _TIME_WORDS:
        return stripped
    return None


_RETURN_BOOKED = {"yes", "yeah", "yep", "yup", "booked", "haan", "ha", "han", "ji", "done"}
_RETURN_NOT_BOOKED = {"no", "nope", "nahi", "nahin", "nhi", "na"}
#: Whole phrases, matched before single words, so "not sure yet" (they don't know)
#: is never mistaken for "not yet" (they haven't booked).
_RETURN_NOT_BOOKED_PHRASES = {"not yet", "no not yet", "abhi nahi", "nahi abhi", "not booked"}
_RETURN_OWN = {"own", "self", "myself", "bus", "train", "flight", "car", "khud", "apne", "apni"}
_RETURN_UNSURE = {"sure", "idk", "dunno", "maybe", "pata", "dekhenge", "later", "decide"}


def parse_return_preference(text: str) -> Optional[str]:
    """'booked' | 'not_booked' | 'own_arrangement', or None when the reply is unclear.

    Unlike the old fallback, an unrecognised reply yields None instead of being filed
    as "own arrangement" — that guess silently rewrote the trip's last day. Kept to
    short, direct replies: a longer sentence is left to the extraction model.
    """
    toks = _tokens(text)
    if not toks:
        return None
    normalized = " ".join(toks)

    if normalized in _RETURN_NOT_BOOKED_PHRASES:
        return "not_booked"
    if set(toks) & _RETURN_OWN:
        return "own_arrangement"
    # "hmm, not sure yet" is an answer about their certainty, not about the booking.
    if set(toks) & _RETURN_UNSURE:
        return None
    if len(toks) > 3:
        return None
    if set(toks) & _RETURN_NOT_BOOKED:
        return "not_booked"
    if set(toks) & _RETURN_BOOKED:
        return "booked"
    return None


def _in_range(value: int, lo: int, hi: int) -> Optional[int]:
    return value if lo <= value <= hi else None


# ── Slot definitions ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Slot:
    """One piece of information, with everything the agent knows about asking for it."""

    name: str
    #: (lang_key, visitor_type, attempt) -> the question to put to the user
    ask: Callable[[str, Optional[str], int], str]
    #: read the current value out of the entity bag (None when unknown)
    read: Callable[[ExtractedEntities], Any]
    #: strict parser for a direct reply; None when the reply doesn't fit
    parse: Callable[[str], Any]
    #: write a parsed value into the merged-entities dict
    write: Callable[[dict, Any, Optional[str]], None]
    #: short label for the writer when we proceed without it
    label: Callable[[str], str]


def _q(table: dict[str, str], lang_key: str) -> str:
    return table.get(lang_key, table["en"])


_DESTINATION_Q = {
    "hi": "आप हिमाचल में कहाँ जाना चाहते हैं?",
    "en": "Where would you like to go in Himachal Pradesh?",
}
# Used after a first attempt failed to land — same question, with examples, so the
# user can see the shape of answer we need instead of being asked the same words.
_DESTINATION_Q_RETRY = {
    "hi": "मुझे बस जगह का नाम बता दीजिए — जैसे मनाली, शिमला, धर्मशाला या कसोल।",
    "en": "Just the place name is enough — for example Manali, Shimla, Dharamshala or Kasol.",
}
_ORIGIN_Q = {
    "hi": "आप किस शहर से यात्रा शुरू कर रहे हैं?",
    "en": "Which city are you travelling from?",
}
_ORIGIN_Q_LOCAL = {
    "hi": "आप हिमाचल में किस शहर में रहते हैं?",
    "en": "Which city in Himachal are you based in?",
}
_DURATION_Q = {
    "hi": "आप कितने दिनों के लिए यात्रा करना चाहते हैं?",
    "en": "How many days are you planning your trip for?",
}
_GROUP_SIZE_Q = {
    "hi": "आपके साथ कुल कितने लोग यात्रा कर रहे हैं?",
    "en": "How many people will be travelling in total?",
}
_WHEN_Q = {
    "hi": "आप कब यात्रा करना चाहते हैं?",
    "en": "When are you planning to travel?",
}
_RETURN_Q = {
    "hi": "क्या आपने वापसी के लिए कैब बुक कर ली है, या अपने साधन से लौटेंगे?",
    "en": "Have you already booked your return cab, or will you be making your own way back?",
}

_LABELS = {
    "destination": {"hi": "गंतव्य", "en": "destination"},
    "origin": {"hi": "शुरुआती शहर", "en": "starting city"},
    "duration_days": {"hi": "यात्रा के दिन", "en": "trip length"},
    "group_size": {"hi": "यात्रियों की संख्या", "en": "group size"},
    "travel_dates": {"hi": "यात्रा की तारीख", "en": "travel dates"},
    "return_journey_preference": {"hi": "वापसी की व्यवस्था", "en": "return journey"},
}

def _write_origin(merged: dict, value: Any, visitor_type: Optional[str]) -> None:
    """The pickup city. For a Himachal resident it doubles as their home city, which
    is why there is one slot here and not two that can disagree."""
    merged["origin"] = value
    if visitor_type == LOCAL:
        merged["home_city"] = value


def _simple_write(field: str) -> Callable[[dict, Any, Optional[str]], None]:
    def write(merged: dict, value: Any, _visitor_type: Optional[str]) -> None:
        merged[field] = value
    return write


def _label(field: str) -> Callable[[str], str]:
    return lambda lang_key: _q(_LABELS[field], lang_key)


SLOTS: dict[str, Slot] = {
    "destination": Slot(
        name="destination",
        ask=lambda lang, _vt, attempt: _q(
            _DESTINATION_Q_RETRY if attempt else _DESTINATION_Q, lang
        ),
        read=lambda e: e.destination,
        parse=parse_place,
        write=_simple_write("destination"),
        label=_label("destination"),
    ),
    "origin": Slot(
        name="origin",
        ask=lambda lang, vt, _a: _q(_ORIGIN_Q_LOCAL if vt == LOCAL else _ORIGIN_Q, lang),
        read=lambda e: e.origin or e.home_city,
        parse=parse_place,
        write=_write_origin,
        label=_label("origin"),
    ),
    "duration_days": Slot(
        name="duration_days",
        ask=lambda lang, _vt, _a: _q(_DURATION_Q, lang),
        read=lambda e: e.duration_days,
        parse=parse_duration_days,
        write=_simple_write("duration_days"),
        label=_label("duration_days"),
    ),
    "group_size": Slot(
        name="group_size",
        ask=lambda lang, _vt, _a: _q(_GROUP_SIZE_Q, lang),
        read=lambda e: e.group_size,
        parse=parse_group_size,
        write=_simple_write("group_size"),
        label=_label("group_size"),
    ),
    "travel_dates": Slot(
        name="travel_dates",
        ask=lambda lang, _vt, _a: _q(_WHEN_Q, lang),
        read=lambda e: e.travel_dates,
        parse=parse_when,
        write=_simple_write("travel_dates"),
        label=_label("travel_dates"),
    ),
    "return_journey_preference": Slot(
        name="return_journey_preference",
        ask=lambda lang, _vt, _a: _q(_RETURN_Q, lang),
        read=lambda e: e.return_journey_preference,
        parse=parse_return_preference,
        write=_simple_write("return_journey_preference"),
        label=_label("return_journey_preference"),
    ),
}


# ── Per-intent policy ─────────────────────────────────────────────────────────
# The order of keys here IS the order questions are asked. Everything above is
# mechanism; this table is the policy, and it is the only place to change it.
#
# Reading the table: a trip plan cannot exist without a destination, so that is the
# single REQUIRED slot. A cab company also wants the pickup city and the trip
# length — IMPORTANT, so they are asked, but the agent gives up rather than loops.
# The rest ride along with the itinerary as a closing follow-up question.

_PLAN_TRIP_NEEDS: dict[str, Need] = {
    "destination": Need.REQUIRED,
    # Everything that SHAPES the plan is collected before it is written — the trip
    # length sets the number of days, the starting city sets day 1 and the fare, the
    # group size sets the cab and the recommendations, the dates set the season.
    # Asking for any of these after handing over an itinerary means the itinerary was
    # written without them, which is the wrong order however politely it is phrased.
    "origin": Need.IMPORTANT,
    "duration_days": Need.IMPORTANT,
    "group_size": Need.IMPORTANT,
    "travel_dates": Need.IMPORTANT,
    # Not an itinerary input — it belongs to booking, so it rides along with the
    # closing "shall I book your Him Cab?" rather than gating the plan.
    "return_journey_preference": Need.OPTIONAL,
}

#: A question about a place needs the place and nothing else. This one line is what
#: stops "what's the weather in Manali?" turning into a trip interrogation.
_LOOKUP_NEEDS: dict[str, Need] = {"destination": Need.REQUIRED}

_COST_NEEDS: dict[str, Need] = {
    "destination": Need.REQUIRED,
    "origin": Need.IMPORTANT,
    "duration_days": Need.IMPORTANT,
    "group_size": Need.IMPORTANT,
}

_NEEDS: dict[str, dict[str, Need]] = {
    "plan_trip": _PLAN_TRIP_NEEDS,
    # A reply to one of our own questions is part of the trip conversation.
    "clarification": _PLAN_TRIP_NEEDS,
    "show_itinerary": _PLAN_TRIP_NEEDS,
    "cost_query": _COST_NEEDS,
    "weather_query": _LOOKUP_NEEDS,
    "places_query": _LOOKUP_NEEDS,
    "hotel_query": _LOOKUP_NEEDS,
    # An open question is answered from knowledge; it needs nothing collected first.
    "general_query": {},
}

#: Intents that are about GOING somewhere rather than asking about it. What the
#: reply looks like then depends on who is travelling — see `response_mode`.
_TRAVEL_INTENTS = frozenset({"plan_trip", "clarification", "show_itinerary", "cost_query"})

#: A full day-by-day plan plus the structured itinerary JSON. Outstation visitors:
#: they are choosing what to do with several days in an unfamiliar state.
ITINERARY_MODE = "itinerary"
#: A cab recommendation — vehicle, fare, drive time, weather — and nothing else.
#: Himachal residents already know the place; what they want is the ride, so an
#: itinerary telling them about Mall Road is noise around the useful part.
RIDE_MODE = "ride"
#: A short direct answer to a question about a place. No trip, no cab.
ANSWER_MODE = "answer"


def needs_for(intent: Optional[str]) -> dict[str, Need]:
    """The slot policy for an intent. Unknown intents plan a trip (the safe default)."""
    return _NEEDS.get(intent or "", _PLAN_TRIP_NEEDS)


def response_mode(intent: Optional[str], visitor_type: Optional[str] = None) -> str:
    """What shape the reply takes, from the intent and who is asking.

    The same "plan my trip" means different things to the two personas: a visitor
    wants an itinerary, a resident wants a cab. Questions about a place get a direct
    answer either way.
    """
    if (intent or "plan_trip") not in _TRAVEL_INTENTS:
        return ANSWER_MODE
    return RIDE_MODE if visitor_type == LOCAL else ITINERARY_MODE


# ── The decision ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Decision:
    """What the planner should do this turn."""

    #: the one field being asked about, or None to go ahead and answer
    ask_field: Optional[str] = None
    #: the question to put to the user (set iff ask_field is set)
    question: Optional[str] = None
    #: slots we wanted but gave up on — the writer MUST say what it assumed
    assumed: tuple[str, ...] = ()

    @property
    def needs_clarification(self) -> bool:
        return self.ask_field is not None


def decide(
    entities: ExtractedEntities,
    intent: Optional[str],
    visitor_type: Optional[str],
    lang_key: str,
    *,
    max_attempts: int,
) -> Decision:
    """Choose between asking for ONE missing detail and producing the answer.

    Exactly one question per turn, in the order of the policy table, so each gets a
    focused reply. Bundling several into one message reads as a form to fill in and
    invites a partial answer that then has to be chased anyway.

    A slot is asked only when it is unknown, relevant to `intent`, and has been put
    to the user fewer than `max_attempts` times. Because the questions are now serial,
    the limits below are what keep the conversation finite rather than merely tidy:

      • per field  — `entities.asked_counts`, so one detail is never chased forever
      • per person — `entities.no_progress_turns`, so a user who keeps replying
        without answering gets their plan instead of another question
      • on request — `entities.proceed_requested`, set the moment they tell us to
        get on with it, which drops every remaining question at once

    None of the three can skip a REQUIRED slot: with no destination there is nothing
    to produce, and inventing one is the failure this whole module prevents.
    """
    needs = needs_for(intent)
    give_up = bool(entities.proceed_requested) or entities.no_progress_turns >= max_attempts

    required: Optional[str] = None
    important: Optional[str] = None
    assumed: list[str] = []

    for name, need in needs.items():
        if need is Need.UNUSED or need is Need.OPTIONAL:
            continue
        if _has_value(SLOTS[name].read(entities)):
            continue

        if need is Need.REQUIRED:
            # Never escapable — not by the attempt cap, not by "just make it".
            required = required or name
        elif give_up or entities.asked_counts.get(name, 0) >= max_attempts:
            # IMPORTANT: worth a question, never worth a loop.
            assumed.append(name)
        else:
            important = important or name

    # A missing REQUIRED slot always jumps the queue — it is the anchor the rest of
    # the questions only make sense against.
    pending = required or important
    if pending is None:
        return Decision(assumed=tuple(assumed))

    return Decision(
        ask_field=pending,
        question=SLOTS[pending].ask(
            lang_key, visitor_type, entities.asked_counts.get(pending, 0)
        ),
        assumed=tuple(assumed),
    )


def _has_value(value: Any) -> bool:
    return value not in (None, "", [], {})


def read_answer(
    field: Optional[str],
    text: str,
    merged: dict,
    visitor_type: Optional[str],
) -> Optional[str]:
    """Read `text` as the answer to the question we just asked about `field`.

    Returns the field that was filled, or None. Nothing happens unless we actually
    asked about it and it is still unknown — the agent never reinterprets a message
    as an answer to a question it did not ask, which is what let "plan a trip" become
    a destination. The per-slot parser then has to accept the shape of the reply, so
    an off-topic answer fills nothing rather than something wrong.
    """
    if not field or field not in SLOTS:
        return None
    if _has_value(merged.get(field)):
        return None
    if field == "origin" and _has_value(merged.get("home_city")):
        return None

    value = SLOTS[field].parse(text)
    if value is None:
        return None
    SLOTS[field].write(merged, value, visitor_type)
    return field


def label_for(field: str, lang_key: str) -> str:
    """Human-readable name of a slot, for messages shown to the user."""
    slot = SLOTS.get(field)
    return slot.label(lang_key) if slot else field
