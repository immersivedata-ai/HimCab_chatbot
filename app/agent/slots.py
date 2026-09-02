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
from difflib import get_close_matches
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from app.core.enums import CancelScope, Intent, Persona, ResponseMode, SlotName
from app.models.session import ExtractedEntities

# ── Personas ──────────────────────────────────────────────────────────────────
# Aliases onto app/core/enums.py, which is the one place these values are written
# down. Kept as module names because this module is where the policy lives and the
# short names read better in the tables below — but they are the SAME objects, so a
# persona compare here can never disagree with one in chat_session or the planner.
LOCAL = Persona.LOCAL
OUTSTATION = Persona.OUTSTATION
RENTAL = Persona.RENTAL


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
    "cancel", "cancelled", "canceled", "nevermind", "forget", "abort",
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
    # The rental vocabulary. "Where should the car pick you up?" answered with "rent a
    # car" is three ordinary words and used to pass this parser outright — the pickup
    # became "Rent A Car", the geocoder placed it somewhere, and the fare was quoted
    # from there. None of these is a Himachal town.
    "rent", "rental", "rentals", "hire", "hourly", "car", "cars", "cab", "cabs",
    "taxi", "driver", "gaadi", "gadi", "vehicle", "booking", "address", "location",
    "pickup", "pick", "drop", "time", "date", "hour", "hours", "ghanta", "ghante",
    # Calling it off. "Where should the car pick you up?" answered with "cancel" used
    # to pass this parser: the pickup became "Cancel", the geocoder placed it
    # somewhere, and the rider who had just backed out was quoted a fare from it.
    # The router catches a cancellation before any of this runs — this is the net
    # under that, for a phrasing it misses.
    "cancel", "cancelled", "canceled", "cancellation", "abort", "nevermind",
    "forget", "rehne", "rehnedo", "chhodo", "chodo", "band",
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
    if any(t in _NOT_A_PLACE or t in _NON_ANSWERS or _looks_like_time_word(t) for t in toks):
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


def parse_rental_hours(text: str) -> Optional[int]:
    """A rental duration in hours, or None. Handles "8", "8 hours", "8 ghante".

    Deliberately NOT validated against the rate card here: the packages are fetched
    live and this parser is a code-level safety net that runs without them. A duration
    that matches no package is caught where the fare is requested, which is the only
    place that knows what the packages are.

    Half-days are rejected rather than rounded — "8.5 hours" would be sent as 8, and a
    rider told they were quoted for something other than what they asked for.
    """
    low = text.strip().lower()
    if _looks_like_a_question(low):
        return None
    # The unit is matched by STEM, not spelled out exactly: a rider typing "8 hourse"
    # (as one did) had their answer thrown away and the identical question put back to
    # them. The number is the answer; how they spell "hours" is not our business, and a
    # typo must not cost them a turn.
    m = re.fullmatch(
        r"(\d{1,2})\s*(?:hour\w*|hr\w*|ghant\w*|घंट[ेा])?", low
    )
    if m:
        return _in_range(int(m.group(1)), 1, 24)
    if re.fullmatch(r"(half|aadha)\s*(a\s*)?day", low):
        return 4
    if re.fullmatch(r"(full|pura|whole)\s*(a\s*)?day", low):
        return 8
    return None


#: Nouns that mark a bare number as a headcount. A reply must be ENTIRELY a number and
#: (optionally) one of these — "4 hours" and "4 days" match nothing here, so they cannot
#: be filed as a group size even when that is the question in flight.
_GROUP_NOUNS = r"(?:people|persons?|pax|members?|adults?|guests?|of\s+us|log|logo|लोग)"


def parse_group_size(text: str) -> Optional[int]:
    """A headcount, or None. Handles '4', '4 people', '4 of us', 'we are 4', '4 log'.

    The leading and trailing words are the ones riders actually type when answering
    "how many people?" — everything else still fails, because a wrong headcount picks
    the wrong cab.
    """
    low = text.strip().lower()
    if _looks_like_a_question(low):
        return None
    m = re.fullmatch(
        rf"(?:(?:we|hum)\s+(?:are|hain|hai)\s+)?"      # "we are 4", "hum hain 4"
        rf"(?:just|only|total)?\s*"                    # "just 2"
        rf"(\d{{1,2}})"
        rf"(?:\s*{_GROUP_NOUNS})?"
        rf"(?:\s*(?:hain|hai))?",                      # "4 log hain"
        low,
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
    # Abbreviations and stock alternate spellings fuzzy matching cannot reach — too
    # short, or too far from the dictionary spelling for the ratio to clear the cutoff
    # ("tmrw" vs "tomorrow", "tonite" vs "tonight" both score below it).
    "tmrw", "tmr", "tmw", "tdy", "tonite", "tonit",
    "कल", "आज", "परसों", "अगले", "अगला", "इस", "हफ्ते", "हफ्ता", "महीने", "महीना",
}

#: Full words riders routinely misspell when answering "when?". Only present/future
#: days — "yesterday" is left out on purpose, so a past day is never silently taken as
#: a travel date. Matched fuzzily below; the exact spellings already live in `_TIME_WORDS`.
_TIME_TYPO_TARGETS = ("tomorrow", "today", "tonight")


def _looks_like_time_word(token: str) -> bool:
    """True if `token` is a time word, or an obvious misspelling of one.

    "tommorow", "tomorow" and "tomarrow" all mean tomorrow, and a typo in the answer
    must not cost the rider a turn — the same rule `parse_rental_hours` applies to
    "8 hourse". Kept deliberately tight (length >= 4, a high cutoff) so a place name is
    never mistaken for a date: both parsers read this, so widening it here would loosen
    the boundary they police for each other.
    """
    if token in _TIME_WORDS:
        return True
    if len(token) < 4:
        return False
    return bool(get_close_matches(token, _TIME_TYPO_TARGETS, n=1, cutoff=0.8))


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
    if any(_looks_like_time_word(t) for t in _tokens(stripped)):
        return stripped
    return None


#: Words that place a time of day but are NOT a clock reading. Recognised so they can
#: be rejected on purpose: the backend reads `bookingTime` as HH:mm and charges the
#: night rate from it, so turning "morning" into 09:00 would invent the very number the
#: price depends on. Rejecting costs one more question; guessing costs the rider money.
_VAGUE_TIME_WORDS = {
    "morning", "afternoon", "evening", "night", "noon", "midnight", "dawn", "dusk",
    "early", "late", "subah", "subeh", "dopahar", "shaam", "sham", "raat", "savere",
    "सुबह", "दोपहर", "शाम", "रात",
}

#: "shaam 5" and "5 baje shaam" both mean 17:00 — the marker can sit on either side.
_PM_WORDS = {"pm", "p.m", "shaam", "sham", "raat", "evening", "night", "शाम", "रात"}
_AM_WORDS = {"am", "a.m", "subah", "subeh", "savere", "morning", "सुबह"}


def parse_pickup_time(text: str) -> Optional[str]:
    """A clock time as "HH:mm" (24-hour), or None.

    Handles "6", "6 am", "6:30am", "18:00", "6 baje", "subah 6", "shaam 5 baje".

    A bare hour with no am/pm marker is read on the 24-hour clock ("18" -> 18:00), and
    1-11 with no marker is taken at face value ("6" -> 06:00) because that is what a
    rider answering "what time?" means. "morning" and friends are REJECTED rather than
    mapped to an hour — see `_VAGUE_TIME_WORDS`.
    """
    low = text.strip().lower()
    # NOT `_is_non_answer` here: it tokenises words only, so a perfectly good "6" or
    # "18:00" has no tokens and looks like silence. A time may legitimately be all
    # digits, and the digit requirement below rejects the fillers on its own.
    if not low or _looks_like_a_question(low):
        return None

    toks = set(re.findall(r"[a-zऀ-ॿ.]+", low))
    has_digit = any(c.isdigit() for c in low)
    # "morning" places the day, not the hour. Rejected explicitly so the rule is
    # enforced by name rather than falling out of the digit search below.
    if toks & _VAGUE_TIME_WORDS and not has_digit:
        return None

    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{2})|(\d{1,2})", low)
    if not m:
        return None
    if m.group(1) is not None:
        hour, minute = int(m.group(1)), int(m.group(2))
    else:
        hour, minute = int(m.group(3)), 0
    if not (0 <= hour <= 24 and 0 <= minute <= 59):
        return None

    pm = bool(toks & _PM_WORDS)
    am = bool(toks & _AM_WORDS)
    if pm and am:                       # "subah 6 pm" is not a time, it is a mistake
        return None
    if pm and 1 <= hour <= 11:
        hour += 12
    elif am and hour == 12:
        hour = 0
    if hour == 24:
        hour = 0
    if hour > 23:
        return None
    return f"{hour:02d}:{minute:02d}"


#: Cab-tier words, longest first so "mini bus" is never read as "mini" (4 seats vs 20).
#: This is SHAPE recognition, not a catalogue: the authoritative list of cabs is
#: whatever the fare endpoint priced for this package, and `tool_executor` matches the
#: word found here against that. A tier we recognise but the quote does not contain is
#: reported to the rider ("we don't have a Premium on this package") rather than booked.
#
#: A bare "bus" and a bare "traveller" are deliberately NOT here. "8 hours from the bus
#: stand" and "I'm a solo traveller" are a pickup and a description, and reading either
#: as a cab choice would put a 20-seater on the booking screen. Riders who want one say
#: "mini bus" or "tempo", both of which are.
_CAB_TIER_WORDS = (
    "tempo traveller", "mini bus", "minibus", "hatchback", "standard",
    "economy", "premium", "luxury", "comfort", "scorpio", "crysta", "innova",
    "bolero", "tavera", "ertiga", "dzire", "etios", "coach", "sedan", "prime",
    "micro", "tempo", "xylo", "mini", "suv",
)

#: Single words riders misspell when picking a cab. Matched fuzzily below; kept tight
#: (length >= 5, a high cutoff) so a place name is never read as a cab tier.
_CAB_TYPO_TARGETS = tuple(w for w in _CAB_TIER_WORDS if " " not in w and len(w) >= 5)


def parse_cab_category(text: str) -> Optional[str]:
    """The cab tier the rider picked, as a lower-case word, or None.

    Read out of a whole sentence rather than a bare reply — "I'll take the luxury one"
    and "luxury" must both work, because the app's radio group has no equivalent of a
    one-word answer and riders type what they would have tapped.

    Rejects a question ("is luxury available?"), because that is a request for
    information rather than a choice, and rejects anything with no tier word in it at
    all — a place name or a duration must never end up in this slot.
    """
    low = text.strip().lower()
    if not low or _looks_like_a_question(low) or _is_non_answer(low):
        return None

    normalized = " ".join(_tokens(low))
    for tier in _CAB_TIER_WORDS:
        if tier in normalized:
            return tier

    # Then the misspellings, token by token. Kept tight — a short reply, a token of at
    # least five letters, a high cutoff — so no Himachal place name reaches a cab tier:
    # "manali", "kasol", "dharamshala", "chandigarh" and "morning" all score below it,
    # while "luxary", "standerd", "primium" and "ecconomy" land on the right cab.
    toks = _tokens(low)
    if len(toks) <= 4:
        for token in toks:
            if len(token) < 5:
                continue
            match = get_close_matches(token, _CAB_TYPO_TARGETS, n=1, cutoff=0.8)
            if match:
                return match[0]
    return None


# ── Reading a whole sentence (rental flow) ────────────────────────────────────
# The parsers above read a REPLY to a question. This reads a SENTENCE — "I need a car
# tomorrow at 9 am from Shimla for 8 hours, 4 of us" — which is what riders actually
# type first, and which used to depend entirely on the extraction LLM. When that model
# missed a field or was unavailable, the detail was silently lost and the assistant
# either asked for something it had just been told or answered around it.
#
# Every pattern here is ANCHORED on an unmistakable marker: a number followed by an
# hours unit, a clock time with am/pm/baje or HH:MM, a day word, a headcount noun, or a
# "from <place>". Nothing is inferred from position or from a bare number, because the
# cost of a wrong reading (a fare priced on the wrong hour) is far higher than the cost
# of leaving a field for the LLM or for a question.

_SENTENCE_HOURS_RE = re.compile(
    r"(\d{1,2})\s*(?:hours?|hrs?|hr|ghante?y?|ghanta|घंटे|घंटा)\b", re.IGNORECASE
)
_SENTENCE_HALF_DAY_RE = re.compile(r"\b(?:half|aadha|adha)\s*(?:a\s*)?day\b", re.IGNORECASE)
_SENTENCE_FULL_DAY_RE = re.compile(
    r"\b(?:full|whole|pura|poora)\s*(?:a\s*)?day\b", re.IGNORECASE
)

#: Part-of-day words that qualify a bare hour. "shaam 5" and "5 baje shaam" are both
#: 17:00 — `parse_pickup_time` already knows that, so the whole snippet is handed to it.
_DAYPART = r"subah|subeh|savere|morning|dopahar|afternoon|shaam|sham|evening|raat|night"
#: A clock reading, in the four shapes that cannot be anything else.
_SENTENCE_TIME_RES = (
    re.compile(rf"(?:{_DAYPART})\s*(\d{{1,2}}(?:[:.]\d{{2}})?)", re.IGNORECASE),
    re.compile(rf"(\d{{1,2}}(?:[:.]\d{{2}})?)\s*(?:baje|बजे)?\s*(?:{_DAYPART})", re.IGNORECASE),
    re.compile(r"(\d{1,2}(?:[:.]\d{2})?)\s*(?:a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE),
    re.compile(r"(\d{1,2}(?:[:.]\d{2})?)\s*(?:baje|बजे)\b", re.IGNORECASE),
    re.compile(r"\b(\d{1,2}[:.]\d{2})\b"),
)

_MONTHS = (
    "jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|"
    "sep|sept|september|oct|october|nov|november|dec|december"
)
_WEEKDAYS = (
    "monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    "mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
)
#: A DAY, unambiguously. Deliberately much tighter than `_TIME_WORDS`, which contains
#: "start", "end" and "mid" — words that appear in sentences about anything ("start
#: from Shimla") and would otherwise be filed as the travel date.
_SENTENCE_WHEN_RES = (
    re.compile(r"\bday\s+after\s+tomorrow\b", re.IGNORECASE),
    re.compile(r"\b(?:\d{1,2})[/-](?:\d{1,2})(?:[/-]\d{2,4})?\b"),
    re.compile(rf"\b\d{{1,2}}\s*(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:next|this|coming|agle|agli|iss?)\s+"
        rf"(?:week|weekend|month|hafte|hafta|mahine|mahina|{_WEEKDAYS})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:{_WEEKDAYS})\b", re.IGNORECASE),
    re.compile(
        r"\b(?:today|tonight|tonite|tomorrow|tommorow|tomorow|tomarrow|tmrw|tmr|tmw"
        r"|aaj|kal|parso|parson|weekend)\b",
        re.IGNORECASE,
    ),
)

_SENTENCE_GROUP_RES = (
    re.compile(rf"(\d{{1,2}})\s*{_GROUP_NOUNS}", re.IGNORECASE),
    # "we are 4" and "hum 3 hain" — the verb sits either side of the number in Hindi.
    re.compile(r"\b(?:we|hum)\s+(?:are|hain|hai)\s+(\d{1,2})\b", re.IGNORECASE),
    re.compile(r"\b(?:we|hum)\s+(\d{1,2})\s+(?:are|hain|hai)\b", re.IGNORECASE),
)

#: The pickup and which side of its marker the name sits on. English puts it after
#: ("from Shimla"); the Hindi postposition puts it before ("Shimla se"), and reading
#: that one left-to-right captured the words in front of the place instead of the place
#: — "kal subah 6 baje Manali se" gave a pickup of "Baje Manali".
_SENTENCE_FROM_RES = (
    (
        re.compile(
            r"\b(?:from|starting\s+(?:at|from)|pick\s*me\s*up\s*(?:from|at|in)?"
            r"|pick\s*up\s*(?:from|at|in)|pickup\s*(?:from|at|in)?)\s+"
            r"([a-z][a-z' ]{2,39})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        re.compile(r"\b([a-z][a-z' ]{2,39})\s+se\b", re.IGNORECASE),
        "before",
    ),
)

#: Where a captured place name stops. Without these, "from Shimla tomorrow morning"
#: becomes a three-word "place" that the geocoder then fails on.
_PLACE_STOP_WORDS = {
    "at", "on", "for", "in", "to", "and", "with", "by", "around", "about", "please",
    "tomorrow", "today", "tonight", "kal", "aaj", "parso", "morning", "evening",
    "afternoon", "night", "subah", "subeh", "shaam", "sham", "raat", "savere",
    "baje", "बजे", "hours", "hour", "hrs", "ghante", "ghanta", "day",
    "chahiye", "chahta", "chahti", "hai", "hain", "ko", "ke", "ka", "mein", "liye",
    "car", "cab", "gaadi", "gadi", "taxi", "rent", "rental", "hire", "want", "need",
}


def _first_group(patterns, text: str) -> Optional[str]:
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _clock_from_sentence(text: str) -> Optional[str]:
    """The first unambiguous clock time in `text`, normalised by `parse_pickup_time`."""
    for pattern in _SENTENCE_TIME_RES:
        match = pattern.search(text)
        if match:
            reading = parse_pickup_time(match.group(0))
            if reading:
                return reading
    return None


def _place_from_sentence(text: str) -> Optional[str]:
    """The pickup named in `text` ("from Shimla", "Shimla se"), or None.

    Words are collected AWAY from the marker and stop at the first one that cannot be
    part of a place name, so the name is taken and nothing around it.
    """
    for pattern, side in _SENTENCE_FROM_RES:
        match = pattern.search(text)
        if not match:
            continue
        candidate = match.group(1).split()
        if side == "before":
            candidate = candidate[::-1]
        words: list[str] = []
        for word in candidate:
            if word.lower() in _PLACE_STOP_WORDS:
                break
            words.append(word)
            if len(words) == 3:
                break
        if side == "before":
            words = words[::-1]
        # `parse_place` is the gate, so every rejection this module already makes —
        # request words, the rental vocabulary, date words — applies here too.
        place = parse_place(" ".join(words)) if words else None
        if place:
            return place
    return None


def tidy_when(text: Optional[str]) -> Optional[str]:
    """Reduce a stated travel time to the DAY inside it, when there is one.

    "I am planning to travel today" is a perfectly good answer and a terrible value: it
    reaches the booking screen's Date field verbatim and reaches the reply as "your
    pickup day: I am planning to travel today". Both the extraction model and
    `parse_when` keep an answer whole on purpose — for a trip, "sometime after Diwali"
    is worth keeping word for word — so the trimming happens here, where a rental needs
    a date rather than a sentence.

    Anything with no recognisable day in it is returned untouched: a phrase we cannot
    reduce is still the rider's own answer, and dropping it would be worse than leaving
    it long.
    """
    if not text or not text.strip():
        return text
    found = _first_group(_SENTENCE_WHEN_RES, text.strip())
    return found.strip() if found else text.strip()


def read_rental_details(text: str) -> dict[str, Any]:
    """Every rental detail stated unambiguously in `text`. Absent keys mean "not said".

    Deterministic, so a sentence is understood identically every time and understood at
    all when the extraction model is unavailable. Returns only what is anchored on a
    real marker — see the note above this function.
    """
    if not text or not text.strip():
        return {}

    low = text.strip().lower()
    found: dict[str, Any] = {}

    hours_match = _SENTENCE_HOURS_RE.search(low)
    if hours_match:
        hours = _in_range(int(hours_match.group(1)), 1, 24)
        if hours:
            found["rental_hours"] = hours
    elif _SENTENCE_FULL_DAY_RE.search(low):
        found["rental_hours"] = 8
    elif _SENTENCE_HALF_DAY_RE.search(low):
        found["rental_hours"] = 4

    clock = _clock_from_sentence(low)
    if clock:
        found["pickup_time"] = clock

    when = _first_group(_SENTENCE_WHEN_RES, text.strip())
    if when:
        found["travel_dates"] = when.strip()

    for pattern in _SENTENCE_GROUP_RES:
        match = pattern.search(low)
        if match:
            size = _in_range(int(match.group(1)), 1, 60)
            if size:
                found["group_size"] = size
            break

    place = _place_from_sentence(text.strip())
    if place:
        found["origin"] = place

    cab = parse_cab_category(text)
    if cab:
        found["cab_category"] = cab

    return found


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
_DURATION_Q_RETRY = {
    "hi": "दिनों की संख्या बता दें — जैसे 3, या एक हफ्ता।",
    "en": "Just the number of days — for example 3, or a week.",
}
_GROUP_SIZE_Q = {
    "hi": "आपके साथ कुल कितने लोग यात्रा कर रहे हैं?",
    "en": "How many people will be travelling in total?",
}
_GROUP_SIZE_Q_RETRY = {
    "hi": "कुल लोगों की संख्या बता दें — जैसे 2 या 4।",
    "en": "Just the number of people — for example 2 or 4.",
}
_WHEN_Q = {
    "hi": "आप कब यात्रा करना चाहते हैं?",
    "en": "When are you planning to travel?",
}
# Asked again after a reply that was not a date at all. Repeating the question word
# for word reads as a stuck loop — a rider who answers "shimla" here has misread it,
# and saying what a date looks like fixes that in one line.
_WHEN_Q_RETRY = {
    "hi": "बस दिन बता दें — जैसे आज, कल, या कोई तारीख।",
    "en": "Just the day is enough — today, tomorrow, or a date.",
}
_RENTAL_PICKUP_Q_RETRY = {
    "hi": "बस शहर का नाम बता दें, जहाँ से गाड़ी लेनी है।",
    "en": "Just the name of the city or town the car should come to.",
}
#: The DAY is `travel_dates`; this is the hour. Asked separately because the backend
#: prices a rental off `bookingTime` and applies the night rate from it — "tomorrow"
#: alone left the quote standing on whatever time the rider happened to be chatting at.
_PICKUP_TIME_Q = {
    "hi": "गाड़ी आपको कितने बजे चाहिए?",
    "en": "What time should the car reach you?",
}
_PICKUP_TIME_Q_RETRY = {
    "hi": "समय बता दें — जैसे सुबह 6 बजे, या 18:30।",
    "en": "A clock time is what we need — for example 6 am, or 18:30.",
}
_RENTAL_HOURS_Q = {
    "hi": "आपको गाड़ी कितने घंटे के लिए चाहिए?",
    "en": "How long do you need the car for?",
}
#: Asked again after a reply we could not read as a duration. The old behaviour was to
#: repeat the question above word for word, which is what a rider saw after answering
#: "8 hourse" — the same sentence twice, with no hint that anything had gone wrong or
#: what would work instead. This is the fallback for when the rate card is unreadable;
#: when packages ARE known, `rental_hours_question` names them instead.
_RENTAL_HOURS_Q_RETRY = {
    "hi": "घंटों में बता दें — जैसे 3, 5 या 8।",
    "en": "Just the number of hours — for example 3, 5 or 8.",
}
#: The same question asked once the rate card has said what can actually be booked.
#: Naming the options matters here in a way it does not for a destination: the backend
#: matches a rental duration EXACTLY (`rentHours == chart.getMaxHour()`), so an open
#: question invites an answer that cannot be priced, and the rider only finds that out
#: after they have also given a pickup and a date. `{options}` is filled from the live
#: rate card — which packages exist is never written down here.
_RENTAL_HOURS_OPTIONS_Q = {
    "hi": "हमारे पास ये पैकेज हैं: {options}। आपको कौन सा चाहिए?",
    "en": "We have these packages: {options}. Which one would you like?",
}
#: Same list, asked again. Repeating the first wording verbatim is what made the chat
#: look stuck; leading with "please pick one" says the last reply did not land.
_RENTAL_HOURS_OPTIONS_Q_RETRY = {
    "hi": "कृपया इनमें से एक चुनें: {options}।",
    "en": "Please pick one of these: {options}.",
}
#: The same list, after a duration we do not sell. Said the moment they say it, rather
#: than after they have also given a pickup, a date and a time — which is how the rider
#: used to find out, four questions later, that the answer had never been usable.
_RENTAL_HOURS_REJECTED_Q = {
    "hi": "{hours} घंटे का पैकेज हमारे पास नहीं है। हमारे पास ये हैं: {options}। "
          "आपको कौन सा चाहिए?",
    "en": "We don't have a {hours}-hour package. What we do have is {options} — "
          "which would you like?",
}
_RENTAL_PICKUP_Q = {
    "hi": "गाड़ी आपको कहाँ से चाहिए?",
    "en": "Where should the car pick you up?",
}
#: Which cab. Never the FIRST thing asked — the rider is choosing between prices, so
#: the quoting turn asks it (see the writer's rental steps) and this wording is only
#: the re-ask. Naming no cab on purpose: the list comes from the fare the backend just
#: returned, and a tier written down here would eventually be one it no longer prices.
_CAB_CATEGORY_Q = {
    "hi": "आप कौन सी गाड़ी लेना चाहेंगे?",
    "en": "Which cab would you like?",
}
_CAB_CATEGORY_Q_RETRY = {
    "hi": "ऊपर दी गई गाड़ियों में से किसी एक का नाम बता दें।",
    "en": "Just name one of the cabs listed above.",
}
#: Prepended from the third ask onward — see `decide`. Only a REQUIRED slot reaches
#: that many attempts, so this is a statement of fact rather than a nag.
_STILL_NEEDED = {
    "hi": "इसके बिना मैं आगे नहीं बढ़ पाऊँगा —",
    "en": "I can't go ahead without this —",
}
_RETURN_Q = {
    "hi": "क्या आपने वापसी के लिए कैब बुक कर ली है, या अपने साधन से लौटेंगे?",
    "en": "Have you already booked your return cab, or will you be making your own way back?",
}

_LABELS = {
    "destination": {"hi": "गंतव्य", "en": "destination"},
    "rental_hours": {"hi": "किराये के घंटे", "en": "rental duration"},
    "origin": {"hi": "शुरुआती शहर", "en": "starting city"},
    "duration_days": {"hi": "यात्रा के दिन", "en": "trip length"},
    "group_size": {"hi": "यात्रियों की संख्या", "en": "group size"},
    "travel_dates": {"hi": "यात्रा की तारीख", "en": "travel dates"},
    "pickup_time": {"hi": "पिकअप का समय", "en": "pickup time"},
    "cab_category": {"hi": "गाड़ी का चुनाव", "en": "cab choice"},
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


def rental_hours_question(
    option_labels: Sequence[str],
    lang_key: str,
    attempt: int = 0,
    rejected: Optional[int] = None,
) -> str:
    """The rental-duration question, naming the packages when they are known.

    Takes plain labels rather than package objects so this module stays free of any
    dependency on the Him Cab client — and so an empty list falls back to the open
    question, which is exactly what should happen when the rate card is unreachable.

    `rejected` is a duration the rider asked for that is not on offer. Naming it turns a
    re-ask that looks like the assistant forgot into an answer: "we don't have a 6-hour
    package, we have these".
    """
    labels = [str(label).strip() for label in option_labels if str(label).strip()]
    if not labels:
        return _q(_RENTAL_HOURS_Q_RETRY if attempt else _RENTAL_HOURS_Q, lang_key)
    if len(labels) == 1:
        options = labels[0]
    else:
        joiner = " या " if lang_key == "hi" else " or "
        options = ", ".join(labels[:-1]) + joiner + labels[-1]
    if rejected:
        return _q(_RENTAL_HOURS_REJECTED_Q, lang_key).format(
            hours=rejected, options=options
        )
    return _q(
        _RENTAL_HOURS_OPTIONS_Q_RETRY if attempt else _RENTAL_HOURS_OPTIONS_Q, lang_key
    ).format(options=options)


SLOTS: dict[str, Slot] = {
    SlotName.DESTINATION: Slot(
        name=SlotName.DESTINATION,
        ask=lambda lang, _vt, attempt: _q(
            _DESTINATION_Q_RETRY if attempt else _DESTINATION_Q, lang
        ),
        read=lambda e: e.destination,
        parse=parse_place,
        write=_simple_write("destination"),
        label=_label("destination"),
    ),
    SlotName.ORIGIN: Slot(
        name=SlotName.ORIGIN,
        ask=lambda lang, vt, attempt: _q(
            (_RENTAL_PICKUP_Q_RETRY if attempt else _RENTAL_PICKUP_Q) if vt == RENTAL
            else _ORIGIN_Q_LOCAL if vt == LOCAL
            else _ORIGIN_Q,
            lang,
        ),
        read=lambda e: e.origin or e.home_city,
        parse=parse_place,
        write=_write_origin,
        label=_label("origin"),
    ),
    SlotName.DURATION_DAYS: Slot(
        name=SlotName.DURATION_DAYS,
        ask=lambda lang, _vt, attempt: _q(
            _DURATION_Q_RETRY if attempt else _DURATION_Q, lang
        ),
        read=lambda e: e.duration_days,
        parse=parse_duration_days,
        write=_simple_write("duration_days"),
        label=_label("duration_days"),
    ),
    SlotName.RENTAL_HOURS: Slot(
        name=SlotName.RENTAL_HOURS,
        ask=lambda lang, _vt, attempt: _q(
            _RENTAL_HOURS_Q_RETRY if attempt else _RENTAL_HOURS_Q, lang
        ),
        read=lambda e: e.rental_hours,
        parse=parse_rental_hours,
        write=_simple_write("rental_hours"),
        label=_label("rental_hours"),
    ),
    SlotName.GROUP_SIZE: Slot(
        name=SlotName.GROUP_SIZE,
        ask=lambda lang, _vt, attempt: _q(
            _GROUP_SIZE_Q_RETRY if attempt else _GROUP_SIZE_Q, lang
        ),
        read=lambda e: e.group_size,
        parse=parse_group_size,
        write=_simple_write("group_size"),
        label=_label("group_size"),
    ),
    SlotName.TRAVEL_DATES: Slot(
        name=SlotName.TRAVEL_DATES,
        ask=lambda lang, _vt, attempt: _q(
            _WHEN_Q_RETRY if attempt else _WHEN_Q, lang
        ),
        read=lambda e: e.travel_dates,
        parse=parse_when,
        write=_simple_write("travel_dates"),
        label=_label("travel_dates"),
    ),
    SlotName.PICKUP_TIME: Slot(
        name=SlotName.PICKUP_TIME,
        ask=lambda lang, _vt, attempt: _q(
            _PICKUP_TIME_Q_RETRY if attempt else _PICKUP_TIME_Q, lang
        ),
        read=lambda e: e.pickup_time,
        parse=parse_pickup_time,
        write=_simple_write("pickup_time"),
        label=_label("pickup_time"),
    ),
    SlotName.CAB_CATEGORY: Slot(
        name=SlotName.CAB_CATEGORY,
        ask=lambda lang, _vt, attempt: _q(
            _CAB_CATEGORY_Q_RETRY if attempt else _CAB_CATEGORY_Q, lang
        ),
        read=lambda e: e.cab_category,
        parse=parse_cab_category,
        write=_simple_write("cab_category"),
        label=_label("cab_category"),
    ),
    SlotName.RETURN_JOURNEY_PREFERENCE: Slot(
        name=SlotName.RETURN_JOURNEY_PREFERENCE,
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
    SlotName.DESTINATION: Need.REQUIRED,
    # Everything that SHAPES the plan is collected before it is written — the trip
    # length sets the number of days, the starting city sets day 1 and the fare, the
    # group size sets the cab and the recommendations, the dates set the season.
    # Asking for any of these after handing over an itinerary means the itinerary was
    # written without them, which is the wrong order however politely it is phrased.
    SlotName.ORIGIN: Need.IMPORTANT,
    SlotName.DURATION_DAYS: Need.IMPORTANT,
    SlotName.GROUP_SIZE: Need.IMPORTANT,
    SlotName.TRAVEL_DATES: Need.IMPORTANT,
    # Not an itinerary input — it belongs to booking, so it rides along with the
    # closing "shall I book your Him Cab?" rather than gating the plan.
    SlotName.RETURN_JOURNEY_PREFERENCE: Need.OPTIONAL,
}

#: A question about a place needs the place and nothing else. This one line is what
#: stops "what's the weather in Manali?" turning into a trip interrogation.
_LOOKUP_NEEDS: dict[str, Need] = {"destination": Need.REQUIRED}

_COST_NEEDS: dict[str, Need] = {
    SlotName.DESTINATION: Need.REQUIRED,
    SlotName.ORIGIN: Need.IMPORTANT,
    SlotName.DURATION_DAYS: Need.IMPORTANT,
    SlotName.GROUP_SIZE: Need.IMPORTANT,
}

#: A rental is a car and a driver for a block of time. There is no destination — that
#: is the whole point of hiring by the hour — so asking for one would be asking the
#: rider to invent an answer. What the fare needs is the package and the pickup.
_RENTAL_NEEDS: dict[str, Need] = {
    SlotName.RENTAL_HOURS: Need.REQUIRED,
    SlotName.ORIGIN: Need.REQUIRED,
    # The day and the hour are two questions, not one. `travel_dates` takes "tomorrow";
    # `pickup_time` takes the clock time the car should arrive, which the backend reads
    # as `bookingTime` and applies the night rate from. Without it the quote stood on
    # whatever time the rider happened to be chatting at, so a 6 a.m. pickup arranged
    # at 10 p.m. was priced as a night run.
    SlotName.TRAVEL_DATES: Need.IMPORTANT,
    SlotName.PICKUP_TIME: Need.IMPORTANT,
    SlotName.GROUP_SIZE: Need.IMPORTANT,
    # The app's "Select Rental Cab" radio group. OPTIONAL means `decide` never asks for
    # it, which is deliberate: the rider is choosing between PRICES, and no price exists
    # until the duration and the pickup have been collected and the fare fetched. The
    # quoting turn puts the question instead (see the writer's rental steps) and marks
    # the slot as in flight, so "luxury" on the next message is read as the answer.
    SlotName.CAB_CATEGORY: Need.OPTIONAL,
}

#: Nothing is collected for a cancellation. Every question would be about a request
#: that is over — and a cancellation that gets asked "which city are you starting
#: from?" is precisely the reported failure.
_CANCEL_NEEDS: dict[str, Need] = {}

_NEEDS: dict[str, dict[str, Need]] = {
    "cancel_request": _CANCEL_NEEDS,
    "rental_query": _RENTAL_NEEDS,
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
_TRAVEL_INTENTS = frozenset({
    Intent.PLAN_TRIP, Intent.CLARIFICATION, Intent.SHOW_ITINERARY, Intent.COST_QUERY,
})

#: Hiring a car by the hour. Its own intent because it needs a different set of
#: answers from every other request — a duration instead of a destination.
RENTAL_INTENT = Intent.RENTAL_QUERY

#: Calling it off. The one intent that outranks the persona in BOTH tables below: a
#: rental chat that treats a cancellation as a rental carries on collecting details for
#: a request the rider has just abandoned, which is exactly what they report as the
#: assistant ignoring them. See app/agent/cancellation.py.
CANCEL_INTENT = Intent.CANCEL_REQUEST

# Reply shapes. Aliases onto app/core/enums.ResponseMode — see the note on the persona
# aliases above. Each mode forbids what the others require, so the writer branches on
# exactly one of these and never on a phrase.
#: A full day-by-day plan plus the structured itinerary JSON. Outstation visitors:
#: they are choosing what to do with several days in an unfamiliar state.
ITINERARY_MODE = ResponseMode.ITINERARY
#: A cab recommendation — vehicle, fare, drive time, weather — and nothing else.
#: Himachal residents already know the place; what they want is the ride, so an
#: itinerary telling them about Mall Road is noise around the useful part.
RIDE_MODE = ResponseMode.RIDE
#: A short direct answer to a question about a place. No trip, no cab.
ANSWER_MODE = ResponseMode.ANSWER
#: Nothing is being sold: the rider called it off. No fare, no cab list, no itinerary
#: and no booking screen — see `cancellation.cancellation_message`.
CANCELLED_MODE = ResponseMode.CANCELLED
#: A rental quote — package, cab options and price. No itinerary, no route: the rider
#: has not said where they are going, and hiring by the hour means they need not.
RENTAL_MODE = ResponseMode.RENTAL


def needs_for(
    intent: Optional[str], visitor_type: Optional[str] = None
) -> dict[str, Need]:
    """The slot policy for an intent. Unknown intents plan a trip (the safe default).

    A RENTAL session is decided by the persona, NOT by this turn's intent — the same
    rule `response_mode` already applies, and for the same reason. The router
    re-classifies intent from every message, so "kasol" or "shimla" in a rental chat
    comes back as `plan_trip`; keying the questions off that alone asked a rider hiring
    a car by the hour where they wanted to GO and for how many DAYS, then read their
    reply as a destination. A rental has neither: it is hours, a pickup and a car.
    """
    # A cancellation is checked FIRST, before the persona: the rider is not asking for
    # a rental or a trip any more, so every question in either table is a question about
    # something that no longer exists.
    if intent == CANCEL_INTENT:
        return _CANCEL_NEEDS
    if intent == RENTAL_INTENT or visitor_type == RENTAL:
        return _RENTAL_NEEDS
    return _NEEDS.get(intent or "", _PLAN_TRIP_NEEDS)


def response_mode(intent: Optional[str], visitor_type: Optional[str] = None) -> str:
    """What shape the reply takes, from the intent and who is asking.

    The same "plan my trip" means different things to the two personas: a visitor
    wants an itinerary, a resident wants a cab. Questions about a place get a direct
    answer either way.
    """
    intent = intent or "plan_trip"
    # Before everything, including the persona: whatever they came for, they have said
    # they no longer want it. A reply that sells anything here is the reported bug.
    if intent == CANCEL_INTENT:
        return CANCELLED_MODE
    # A rental is a rental whoever asks: resident or visitor, the reply is a package,
    # a cab and a price. The persona only decides the language.
    if intent == RENTAL_INTENT or visitor_type == RENTAL:
        return RENTAL_MODE
    if intent not in _TRAVEL_INTENTS:
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
    needs = needs_for(intent, visitor_type)
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

    attempt = entities.asked_counts.get(pending, 0)
    question = SLOTS[pending].ask(lang_key, visitor_type, attempt)
    # Third time and beyond, the retry wording alone starts repeating too. Only a
    # REQUIRED slot ever gets this far — everything else is assumed once it passes
    # `max_attempts` — so the escalation can say the plain truth: we are stuck without
    # it. One place, every slot, rather than a third phrasing per question.
    if attempt >= 2:
        question = f"{_q(_STILL_NEEDED, lang_key)} {question}"

    return Decision(ask_field=pending, question=question, assumed=tuple(assumed))


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
