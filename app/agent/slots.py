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
from typing import TYPE_CHECKING, Any, Optional

from app.core.enums import Intent, Persona, ResponseMode, SlotName

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from app.domain.conversation import ExtractedEntities

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
    # Deixis. "Where should the car pick you up?" answered with "here" used to pass
    # this parser outright: the pickup became the place "Here", the geocoder found
    # something somewhere, and the car was sent to it. These words REFER to the
    # rider's position, they do not name it — `means_current_location` reads them
    # against the device's fix instead. Same failure as "cancel" above, same fix.
    "here", "there", "loc", "position", "spot", "yahan", "yahin",
    "idhar", "yaha", "yhan", "wahan", "jagah", "यहाँ", "यहीं", "इधर",
    # Agreeing. Same failure as "cancel" and "here": "thik hai" is two ordinary words,
    # so it passed this parser and the rider who had just said "fine" had their
    # destination set to the town of Thik Hai. The bare ones ("ok", "haan", "yes") are
    # already in `_NON_ANSWERS` above; these are the phrasings that were not.
    "thik", "theek", "sahi", "achha", "accha", "acha", "okey", "yeah", "yep",
    "fine", "good", "great", "ठीक", "सही",
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
    # A reply that POINTS at the rider's position does not name a place. Deferring to
    # the one recognizer rather than repeating its vocabulary in `_NOT_A_PLACE` keeps
    # a phrase from being rejected by one and accepted by the other — "where i am"
    # parsed as the place "Where I Am" precisely because the two lists disagreed.
    if means_current_location(text):
        return None
    toks = _tokens(text)
    if not 1 <= len(toks) <= 3 or len("".join(toks)) < 3:
        return None
    if any(t in _NOT_A_PLACE or t in _NON_ANSWERS or _looks_like_time_word(t) for t in toks):
        return None
    return text.title()


#: The framing a rider wraps round a place when answering the pickup or drop question:
#: "Manali drop kar do", "Shimla se pickup", "pick me up from Solan". `_NOT_A_PLACE`
#: rejects "pickup" and "drop" outright — right when the WHOLE reply is that vocabulary
#: ("pickup" alone is not a place), wrong once the question itself says "pickup",
#: because riders echo the word back and every such answer would be thrown away.
#: Whole words only, and stripped only after the reply failed to read as it stands, so
#: a name that contains one of them ("Scandal Point") is never cut short.
_LOCATION_FRAMING = frozenset({
    "pickup", "pick", "up", "drop", "dropoff", "location", "from", "at", "to", "me",
    "us", "please", "pls", "se", "pe", "par", "mein", "tak", "kar", "karo", "kardo",
    "kr", "kro", "dena", "do", "dijiye", "de", "chahiye", "hai", "karna", "lena", "lo",
    "पिकअप", "ड्रॉप", "से", "पे", "पर", "में", "तक", "कर", "करो", "दो", "दीजिए", "देना",
    "है", "चाहिए", "करना",
})


def parse_location(text: str) -> Optional[str]:
    """A place from a reply to the pickup or drop question, or None.

    `parse_place` first, as it stands; then once more with the pickup/drop framing
    taken off. Anything that is not a place after that is still rejected — the framing
    is the only thing this forgives.
    """
    words = [w.strip(".,!?") for w in (text or "").split()]
    words = [w for w in words if w]
    kept_at = [i for i, w in enumerate(words) if w.lower() not in _LOCATION_FRAMING]
    # The place words must sit together. "Shimla se Manali" and "pickup Shimla drop
    # Manali" have a framing word BETWEEN them, so they name two places — read as one
    # they were "Shimla Se Manali" and "Shimla Manali". The sentence reader reads both
    # ends of those; this reads neither, and checks it before anything else.
    if kept_at and kept_at[-1] - kept_at[0] + 1 != len(kept_at):
        return None
    direct = parse_place(text)
    if direct is not None:
        return direct
    if not kept_at or len(kept_at) == len(words):
        return None
    return parse_place(" ".join(words[i] for i in kept_at))


#: Ways a rider says "collect me where I am" rather than naming a place. Matched as
#: whole words against the normalised reply, so "here" fires and "Herebad" does not.
_CURRENT_LOCATION_WORDS = frozenset({
    "here", "yahan", "yahin", "yaha", "yhan", "idhar", "yahi",
    "यहाँ", "यहीं", "इधर",
})

#: Phrases where the words alone are not enough — "my location", "current location".
_CURRENT_LOCATION_PHRASES = (
    "my location", "current location", "my current location", "this location",
    "where i am", "where im", "where i m", "same place", "same location",
    "my position", "present location", "meri location", "abhi jahan",
    "abhi jaha", "jahan hu", "jaha hu", "jahan hoon", "jaha hoon",
)


def means_current_location(text: str) -> bool:
    """True when the reply POINTS at the rider's position instead of naming a place.

    Only meaningful for the pickup question, and only answerable when the app actually
    sent a fix — which is why this returns a boolean and resolves nothing. The caller
    (`planner_node`) is where `state.rider_location` lives, and it decides whether
    there is a usable fix to turn this into a pickup, or whether to ask again.

    Deliberately narrow. A phrase this misses costs one more question; a phrase it
    wrongly matches sends a car to the wrong place.
    """
    low = " ".join((text or "").lower().split())
    if not low:
        return False
    if any(phrase in low for phrase in _CURRENT_LOCATION_PHRASES):
        return True
    tokens = _tokens(low)
    if not tokens or len(tokens) > 4:
        return False
    # Every word must be deixis or filler — "here" and "from here" yes, "here in
    # Shimla" no, because that one names a place and `parse_place` should have it. The
    # pickup words are filler since the question says "pickup": "yahin se pickup karo"
    # points at the rider exactly as "yahin" does.
    filler = {"from", "se", "pe", "par", "at", "on", "in", "the", "my", "mere", "meri",
              "pickup", "pick", "up", "me", "kar", "karo", "lo", "please"}
    return any(t in _CURRENT_LOCATION_WORDS for t in tokens) and all(
        t in _CURRENT_LOCATION_WORDS or t in filler for t in tokens
    )


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


# ── Relative days ─────────────────────────────────────────────────────────────
#
# Every phrase that names ONE day relative to the rider's today, in one table.
#
# It used to be three, and they disagreed. `sanitize._RELATIVE_DAYS` could resolve
# "day after tmrw" to a date; the sentence reader below had a pattern for "day after
# tomorrow" but none for the abbreviation, so it fell through to the bare-word pattern,
# matched "tmrw" alone, and the rental was booked for Wednesday when the rider said
# Thursday. A silent off-by-one on a real cab, produced by nothing but two lists that
# had drifted apart. So there is now one list, and `test_slots.py` asserts the readers
# are all built from it.
#
# `kal` and `parso` each mean a day in BOTH directions in Hindi — kal is yesterday and
# tomorrow, parso is the day before yesterday and the day after tomorrow. Which is meant
# comes from the verb tense, which we do not parse. The future reading is taken because
# this is a booking: somebody arranging a cab means the coming day.
#
# Past days are HERE ON PURPOSE, even though nobody can book one. Leaving "yesterday"
# out meant it read as no date at all, so the rider got the identical "when are you
# travelling?" question back and no hint of what was wrong with their answer — which is
# indistinguishable from not being understood. Recognising it is what lets
# `sanitize.travel_date` refuse it and `clarify_node` say why.
RELATIVE_DAY_PHRASES: dict[str, int] = {
    # Past — recognised so it can be refused by name, never to book against.
    "day before yesterday": -2,
    "beeta kal": -1, "bita kal": -1, "pichhla kal": -1, "pichla kal": -1,
    "yesterday": -1, "ystrday": -1, "yday": -1,
    # Today. "abhi"/"asap"/"right now" are a day answer, not a clock reading — a rider
    # who wants a car NOW is booking for today, and `pickup_time` is asked separately.
    "today": 0, "todya": 0, "tonight": 0, "tonite": 0, "tdy": 0, "2day": 0,
    "aaj": 0, "aj": 0, "aaj hi": 0, "ajj": 0,
    # A bare "now" and a bare "jaldi" are NOT here, and must not be: "we are 5 people
    # now" and "jaldi karo" are not dates, and reading them as one turned an ordinary
    # correction into a proposed change of travel date. Only the unambiguous forms.
    "right now": 0, "just now": 0, "abhi": 0, "abhi hi": 0,
    "asap": 0, "immediately": 0, "turant": 0,
    # Tomorrow.
    "tomorrow": 1, "tommorow": 1, "tomorow": 1, "tomarrow": 1, "tomorrw": 1,
    "tommorrow": 1, "tomm": 1, "tmrw": 1, "tmrow": 1, "tmr": 1, "tmw": 1,
    "kal": 1, "kl": 1, "agle din": 1,
    # The day after. Every spelling of "tomorrow" above needs its own entry here, or
    # the shorter one wins and the booking lands a day early — see the note above.
    "day after tomorrow": 2, "day after tommorow": 2, "day after tomorow": 2,
    "day after tomarrow": 2, "day after tommorrow": 2, "day after tmrw": 2,
    "day after tmr": 2, "day after tmw": 2, "day after kal": 2,
    "day after next": 2, "next to next day": 2,
    "parso": 2, "parson": 2, "parsoo": 2, "porso": 2, "parsu": 2,
    # Three days out. A real word in Hindi and the natural next step after "parso".
    "narso": 3, "narson": 3, "atparso": 3, "terso": 3,
}

#: The same phrases in Devanagari. Matched by substring rather than word boundary,
#: because `\b` is written for Latin script and does not fire between Devanagari and
#: punctuation. Longest first, for the same reason the Latin table is sorted.
DEVANAGARI_DAY_PHRASES: dict[str, int] = {
    "परसों से पहले": -2,
    "बीता कल": -1, "पिछला कल": -1,
    "परसों": 2, "नरसों": 3,
    "आज": 0, "अभी": 0, "तुरंत": 0,
    "कल": 1,
}

#: Phrases that name a day ONLY when they are the whole answer, or end it.
#:
#: "day after" and "next day" are complete answers to "when?" and also ordinary English
#: clause fragments — "the day after we reach we go to Kasol" and "next day return" are
#: not dates, and reading them as one files the wrong travel date off a sentence about
#: something else. Anchoring costs nothing a rider would notice: somebody answering the
#: question types the phrase and stops.
_ANCHORED_DAY_PHRASES: dict[str, int] = {
    "day after": 2,
    "next day": 1,
}

#: Phrases built out of day words that are NOT days. Blanked before anything else runs,
#: so the words inside them cannot be matched by the tables above: "aaj kal" is the
#: idiom for "these days" ("aaj kal bahut traffic hai"), not today.
_DECOY_DAY_RES: tuple[re.Pattern, ...] = (
    re.compile(r"(?<!\w)aaj\s*kal(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)aajkal(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)day\s+after\s+day(?!\w)", re.IGNORECASE),
    re.compile(r"(?<!\w)din\s+ba\s+din(?!\w)", re.IGNORECASE),
    re.compile(r"आजकल|आज\s*कल"),
)

#: Spelled-out counts, for "do din baad" / "three days later". Riders type the word as
#: often as the digit, and a count we cannot read is a phrase that reaches the booking
#: screen's date field verbatim — the bug this whole area exists to prevent.
_COUNT_WORDS: dict[str, int] = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
    "chhe": 6, "che": 6, "chah": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5, "छह": 6, "सात": 7,
    "आठ": 8, "नौ": 9, "दस": 10,
}

_COUNT_ALTERNATION = "|".join(
    re.escape(word) for word in sorted(_COUNT_WORDS, key=len, reverse=True)
)

#: "in 3 days", "3 days later", "teen din baad", "3 दिन बाद" — a count plus a marker
#: that puts it in the FUTURE. The marker is required: without it "3 days" is a trip
#: length, and reading a duration as a date would book the wrong day from the answer to
#: a different question entirely.
_COUNTED_DAY_RES: tuple[re.Pattern, ...] = (
    re.compile(
        rf"(?<!\w)(?:in|after|within)\s+(\d{{1,3}}|{_COUNT_ALTERNATION})\s*"
        rf"(?:days?|din|dino|dinon|दिन)(?!\w)",
        re.IGNORECASE,
    ),
    # `से` ("se") is deliberately NOT a marker here: "3 din se" means "for the last
    # three days", the opposite direction, and reading it as "in three days" would book
    # a cab off a sentence complaining about the weather.
    re.compile(
        rf"(?<!\w)(\d{{1,3}}|{_COUNT_ALTERNATION})\s*(?:days?|din|dino|dinon|दिन)\s*"
        rf"(?:later|after|baad|bad|बाद|from\s+now|from\s+today)(?!\w)",
        re.IGNORECASE,
    ),
)

_ANCHORED_DAY_RES: list[tuple[re.Pattern, int]] = [
    (
        re.compile(
            r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"\s*[.!?]*\s*$",
            re.IGNORECASE,
        ),
        offset,
    )
    for phrase, offset in sorted(
        _ANCHORED_DAY_PHRASES.items(), key=lambda kv: -len(kv[0])
    )
]

#: Longest phrase first, so "day after tomorrow" is matched before "tomorrow" and
#: "aaj hi" before "aaj". Matching the short one first is the silent off-by-one.
_RELATIVE_DAY_RES: list[tuple[re.Pattern, int]] = [
    (
        re.compile(
            r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)",
            re.IGNORECASE,
        ),
        offset,
    )
    for phrase, offset in sorted(
        RELATIVE_DAY_PHRASES.items(), key=lambda kv: -len(kv[0])
    )
]

_DEVANAGARI_DAY_RES: list[tuple[str, int]] = sorted(
    DEVANAGARI_DAY_PHRASES.items(), key=lambda kv: -len(kv[0])
)

#: How far out a counted phrase is still a booking. "in 500 days" is a typo or a joke,
#: and `sanitize` caps the resolved date anyway; this only stops the reader claiming
#: the phrase names a day when it plainly names nonsense.
_MAX_COUNTED_DAYS = 365


def find_relative_day(text: str) -> Optional[tuple[str, int]]:
    """The relative-day phrase in `text` and its offset in days, or None.

    Returns the matched TEXT as well as the offset, because two callers need different
    halves: `sanitize.resolve_relative_day` wants the number so it can produce a real
    date, and `tidy_when` wants the words so it can reduce "I am planning to travel day
    after tomorrow" to the part that goes in a date field.

    The offset may be NEGATIVE. This function reads; it does not decide what is
    bookable. `sanitize.travel_date` owns that, and having the reading separate from
    the policy is what lets a past date be refused by name instead of silently.

    The EARLIEST phrase in the string wins, and at a given position the longest. Both
    halves matter: "day after tmrw" must not be read as the "tmrw" inside it, and a
    sentence that mentions two days means the first one. A compound like "kal se 3 din
    baad" ("three days after tomorrow") is read as its first phrase rather than summed —
    rare enough not to be worth the ambiguity of trying, and the rider sees the day we
    chose in the reply.
    """
    if not text or not text.strip():
        return None
    low = text.strip().lower()
    # Blanked, not removed, so every offset below still indexes into the original and
    # the matched text handed back is the rider's own.
    for decoy in _DECOY_DAY_RES:
        low = decoy.sub(lambda m: " " * len(m.group(0)), low)

    best: Optional[tuple[int, str, int]] = None   # (start, matched, offset)
    for pattern, offset in _ANCHORED_DAY_RES:
        match = pattern.search(low)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.group(0).strip(" .!?"), offset)
    for pattern, offset in _RELATIVE_DAY_RES:
        match = pattern.search(low)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), match.group(0), offset)
            if match.start() == 0:
                break
    for phrase, offset in _DEVANAGARI_DAY_RES:
        index = low.find(phrase)
        if index != -1 and (best is None or index < best[0]):
            best = (index, phrase, offset)
    for pattern in _COUNTED_DAY_RES:
        match = pattern.search(low)
        if not match:
            continue
        raw = match.group(1)
        count = int(raw) if raw.isdigit() else _COUNT_WORDS.get(raw.lower(), 0)
        if not 0 < count <= _MAX_COUNTED_DAYS:
            continue
        # At the SAME position a counted phrase wins, being the more specific reading.
        # Earlier in the string still beats later, like every other reader here.
        if best is None or match.start() <= best[0]:
            best = (match.start(), match.group(0), count)

    return (best[1], best[2]) if best else None


def relative_day_offset(text: str) -> Optional[int]:
    """How many days from today `text` names, or None if it names no single day."""
    found = find_relative_day(text)
    return found[1] if found else None


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

# Every word of every phrase the resolver knows, so a reply it CAN resolve is never
# turned away here as "not a date". These two used to be maintained by hand and
# disagreed: "yesterday" and "abhi" resolved to real days that `parse_when` then
# refused to accept, so the value never reached the validator that would have explained
# the problem — the rider just got the same question again.
_TIME_WORDS |= {
    word
    for phrase in (
        *RELATIVE_DAY_PHRASES, *DEVANAGARI_DAY_PHRASES, *_ANCHORED_DAY_PHRASES
    )
    for word in phrase.split()
}
#: "baad" carries a counted phrase ("3 din baad"); "din" is its noun. Listed separately
#: because they are words of no fixed phrase above. "days"/"day" are deliberately NOT
#: here: "4 days" is a trip LENGTH, and every counted date phrase already clears
#: `parse_when` on its digit or on "later"/"baad".
_TIME_WORDS |= {"din", "dino", "dinon", "baad", "bad", "दिन", "बाद"}

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


#: Words that, ON THEIR OWN, say "the one you suggested is fine". A reply to "which
#: cab?" made only of these is taking our recommendation. It used to be read as no
#: answer at all: the rider typed "haan book kar do", got "Sorry, I didn't quite catch
#: that — just name one of the cabs listed above", and no booking screen until they
#: typed a cab name back to us — the cab we had just picked for their party size.
_ACCEPT_STRONG = frozenset({
    "yes", "yeah", "yep", "ok", "okay", "okk", "fine", "sure", "done", "confirm",
    "book", "haan", "han", "ha", "haa", "ji", "theek", "thik", "chalega", "sahi",
    "wahi", "vahi", "recommended", "suggested", "perfect", "good", "ahead",
    "हाँ", "हां", "ठीक", "चलेगा", "बुक", "वही", "ओके", "सही",
})
#: Words that may sit beside those without changing what they mean.
_ACCEPT_FILLER = frozenset({
    "it", "that", "this", "one", "the", "please", "pls", "go", "with", "same",
    "cab", "car", "gaadi", "gadi", "is", "for", "me", "us", "now", "then", "so", "do",
    "kar", "karo", "kardo", "kr", "kro", "dijiye", "de", "dena", "hai", "h", "ye",
    "yeh", "wala", "wali", "vala", "vali", "bhi", "ji", "sir", "fir", "phir", "just",
    "कर", "करो", "दो", "दीजिए", "है", "ये", "यह", "वाला", "वाली", "जी", "गाड़ी",
})
#: Devanagari-safe split (§5): `\w` drops the combining marks, so `[^\W\d_]+` would
#: shred "ठीक है" into single letters and none of the Hindi entries could ever match.
_ACCEPT_TOKEN_RE = re.compile(r"[\wऀ-ॿ]+")


def accepts_recommendation(text: str) -> bool:
    """True when a reply to "which cab?" means "the one you recommended".

    Deliberately narrow: EVERY word must be agreement or filler, and at least one must
    be agreement. "haan but sedan" names a cab, and a question is a question — both go
    to the normal readers, never here.
    """
    low = (text or "").strip().lower()
    if not low or _looks_like_a_question(low):
        return False
    tokens = _ACCEPT_TOKEN_RE.findall(low)
    if not tokens or len(tokens) > 8:
        return False
    if any(t not in _ACCEPT_STRONG and t not in _ACCEPT_FILLER for t in tokens):
        return False
    return any(t in _ACCEPT_STRONG for t in tokens)


#: A rental's drop answered by pointing back at the pickup: "same place", "wahi",
#: "pickup wali jagah", "wapas yahin", "वहीं". An hourly hire often ends where it
#: began, and without this each of those replies read as a place we could not parse —
#: the rider was told we had not understood them, and then the drop was left out.
_SAME_AS_PICKUP_STRONG = frozenset({
    "same", "wahi", "vahi", "wohi", "wahin", "vahin", "usi", "pickup", "wapas",
    "vapas", "back", "return", "here", "yahin", "yahi",
    "वही", "वहीं", "उसी", "पिकअप", "वापस", "यहीं",
})
_SAME_AS_PICKUP_FILLER = frozenset({
    "place", "location", "jagah", "as", "the", "se", "par", "pe", "hi", "wali", "wala",
    "vali", "vala", "point", "to", "at", "on", "drop", "kar", "karo", "do", "dena",
    "dijiye", "please", "me", "us", "is", "it", "spot",
    "जगह", "वाली", "वाला", "पर", "पे", "से", "ही", "ड्रॉप", "कर", "दो", "दीजिए",
})


def means_same_as_pickup(text: str) -> bool:
    """True when a reply to the drop question means "where you picked me up".

    Every word must be same-place vocabulary or filler, and one must actually point
    back — "back to Shimla" names a place and goes to the place reader instead.
    """
    low = (text or "").strip().lower()
    if not low or _looks_like_a_question(low):
        return False
    tokens = _ACCEPT_TOKEN_RE.findall(low)
    if not tokens or len(tokens) > 6:
        return False
    if any(t not in _SAME_AS_PICKUP_STRONG and t not in _SAME_AS_PICKUP_FILLER for t in tokens):
        return False
    return any(t in _SAME_AS_PICKUP_STRONG for t in tokens)


#: Words that mark a message as being ABOUT the drop. A rental keeps a destination the
#: extraction model reports only from a turn that was — see `planner._strip_trip_fields`.
_DROP_WORDS = frozenset({"drop", "dropped", "dropoff", "ड्रॉप"})


def mentions_drop(text: str) -> bool:
    """True when the message talks about where the car should drop them."""
    return any(t in _DROP_WORDS for t in _ACCEPT_TOKEN_RE.findall((text or "").lower()))


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
    re.compile(r"\b(?:\d{1,2})[/-](?:\d{1,2})(?:[/-]\d{2,4})?\b"),
    re.compile(rf"\b\d{{1,2}}\s*(?:st|nd|rd|th)?\s+(?:{_MONTHS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:{_MONTHS})\s+\d{{1,2}}\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:next|this|coming|agle|agli|iss?)\s+"
        rf"(?:week|weekend|month|hafte|hafta|mahine|mahina|{_WEEKDAYS})\b",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:{_WEEKDAYS})\b", re.IGNORECASE),
    re.compile(r"\bweekend\b", re.IGNORECASE),
)


def _sentence_when(text: str) -> Optional[str]:
    """The DAY inside a sentence, as the rider wrote it, or None.

    Relative days are read by `find_relative_day` rather than by a pattern of their own,
    so this reader and the resolver can never disagree about which words name which day.
    They did: "day after tmrw" resolved to +2 but was READ here as "tmrw", and the
    rental was booked a day early.

    Tried first, before the calendar patterns, because a relative phrase is the more
    specific reading — "day after tomorrow" must not be reduced to a weekday match
    hiding inside some other word.
    """
    found = find_relative_day(text)
    if found:
        return found[0]
    return _first_group(_SENTENCE_WHEN_RES, text)

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
    # WHERE THEY LIVE. The local flow asks "आप हिमाचल में किस शहर में रहते हैं?" — which
    # city do you live in — and a rider answers it with a sentence: "mei himachal mei
    # shimla mei rehta hu", "I live in Shimla". None of the markers below fire on that,
    # so the answer was read as nothing at all. The turn then had no origin, the fare
    # gate refused (`has_origin=False`), no booking screen shipped, and the rider who
    # had answered the question exactly was handed a generic write-up instead.
    #
    # Listed FIRST so residence wins over a stray "se" later in the same sentence.
    (
        re.compile(
            r"\b(?:i\s+(?:live|stay|reside)\s+in|i(?:'m|\s+am)\s+from"
            r"|my\s+home\s+(?:is\s+)?in)\s+"
            r"([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        # "<place> mein rehta/rehti/rehte hun". Only ONE word is captured: the sentence
        # stacks postpositions ("himachal mei shimla mei rehta hu") and a greedy capture
        # swallows "himachal mei shimla" as a single place name.
        re.compile(
            r"\b([a-z][a-z']*)\s+m(?:e|ei|ain|ein|en)\s+reh(?:ta|ti|te)\b",
            re.IGNORECASE,
        ),
        "before",
    ),
    (
        re.compile(
            r"\b(?:from|starting\s+(?:at|from)|pick\s*me\s*up\s*(?:from|at|in)?"
            r"|pick\s*up\s*(?:from|at|in)|pickup\s*(?:from|at|in)?)\s+"
            r"([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        re.compile(r"\b([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})\s+se\b", re.IGNORECASE),
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
    # Adverbs that sit right before a corrected place. Trimmed rather than rejected:
    # "ab kasol jana hai" is a genuine destination with "now" in front of it, and
    # rejecting the pair loses the destination instead of the adverb.
    "ab", "abhi", "just", "only", "still", "phir", "dubara", "wapas",
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


def _place_beside_marker(text: str, patterns, stop_words: set) -> Optional[str]:
    """The place name sitting next to one of `patterns`, or None.

    Words are collected AWAY from the marker and stop at the first one that cannot be
    part of a place name, so the name is taken and nothing around it. `stop_words`
    differs between the pickup scan and the destination scan: a pronoun is harmless
    beside "from" and fatal beside "jana", where "mei shimla jana hai" would otherwise
    yield the place "Mei Shimla".

    Patterns are tried in order and the first that yields a place `parse_place` accepts
    wins, so a specific marker gets its turn before a general one. EVERY occurrence of a
    pattern is tried, not just the first: in "I want to go from Delhi to Kasol" the
    earliest "to" is "to go", which the gate rightly rejects — and stopping there lost
    "to Kasol" and the destination with it.
    """
    for pattern, side in patterns:
        for match in pattern.finditer(text):
            candidate = match.group(1).split()
            if side == "before":
                candidate = candidate[::-1]
            words: list[str] = []
            for word in candidate:
                if word.lower() in stop_words:
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


def _place_from_sentence(text: str) -> Optional[str]:
    """The pickup named in `text` ("from Shimla", "Shimla se"), or None."""
    return _place_beside_marker(text, _SENTENCE_FROM_RES, _PLACE_STOP_WORDS)



# ── Reading a whole sentence (trip flow) ──────────────────────────────────────
#
# The reported bug: the greeting asks "आप कहाँ जाना चाहते हैं?", the rider answers
# "mei shimla jana chahta hu", and the assistant asks where they want to go again.
#
# `parse_place` is the code-level safety net for a reply to a question we asked, and it
# accepts at most three words on purpose — a longer string is a sentence, and letting a
# sentence become a destination is how "plan a trip" once became a place. So a bare
# "shimla" was recovered and a sentence containing it was not, which left the whole
# trip flow depending on the extraction model. When the model missed a phrasing, the
# detail was silently lost and the same question came back.
#
# The rental flow already had this net (`read_rental_details`); trips never did. This
# is the same idea, with the same rule: every value is anchored on an unmistakable
# marker, and `parse_place` is still the gate, so every rejection this module already
# makes applies here too. A phrasing this misses costs one more question. A wrong
# destination costs somebody the wrong trip, so the patterns stay narrow.

#: Words that cannot be part of a place name when reading a TRIP sentence. Extends the
#: pickup set with the pronouns and trip verbs that sit right beside a destination —
#: without "mei", "mei shimla jana hai" yields the place "Mei Shimla".
_TRIP_PLACE_STOP_WORDS = _PLACE_STOP_WORDS | {
    # First-person, the usual left-hand neighbour of a Hinglish destination.
    "mai", "main", "mei", "me", "mujhe", "mujhko", "mera", "meri", "mere",
    "hum", "humein", "hamein", "hamara", "humara", "apne", "apna",
    "i", "we", "us", "our", "you", "your", "lets", "let",
    # The journey itself, not a place in it.
    "se", "from", "tak", "trip", "tour", "travel", "travelling", "traveling",
    "holiday", "vacation", "journey", "plan", "planning", "visit", "visiting",
    "go", "going", "goes", "chalo", "chalein", "ja", "jaa", "jana", "jaana",
    "ghumna", "ghoomna", "dekhna", "jaunga", "jaungi", "jaenge", "jayenge",
    "chahta", "chahti", "chahte", "raha", "rahi", "rahe", "hu", "hun", "hoon",
    # When words. `parse_place` rejects any candidate containing one of these, so
    # without them "a trip to Dharamshala next week" collects three words, fails the
    # gate, and loses the DESTINATION rather than just the date sitting after it.
    "next", "this", "coming", "week", "weekend", "month", "days", "din", "dino",
    "nights", "agle", "agla", "agli", "hafte", "hafta", "mahine", "mahina", "baad",
    # Connectors that sit BETWEEN two places. "palampur se lekar kasol jana hai" is
    # "from Palampur through to Kasol", and without "lekar" the destination scan
    # collected it into the name and answered "Lekar Kasol".
    "lekar", "leke", "hoke", "hokar", "hote", "hue", "via", "through", "towards", "toward",
    "upto", "till", "until", "then", "aur", "phir",
    # Changing your mind. These sit immediately before the corrected place, so without
    # them "actually Manali" reads as the place "Actually Manali" -- which differs from
    # the stored "Manali" and reports a change the rider never made.
    "actually", "instead", "rather", "sorry", "meant", "mean", "correction", "scratch",
    "change", "changed", "changing", "update", "updated", "make", "made", "nahi",
    "nahin", "nhi", "galat", "badal", "badlo", "badalna", "karo", "kar", "do", "dijiye",
}

#: A place named as the PICKUP but carrying a marker the destination rules also claim.
#: "change the pickup to Shimla" and "pickup ko Shimla kar do" both put the place after
#: a `to`/`ko`, so the bare "to X" rule read Shimla as the far end of the trip and moved
#: the wrong end of it. Scanned before the destination, and it wins the place it finds.
_PICKUP_MARKER_RES = (
    (
        re.compile(
            r"\b(?:pick\s*up|pickup|boarding|origin)\s*"
            r"(?:location|point|spot|place|city|address)?\s*"
            r"(?:ko|to|as|is|at|=|:)?\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
)

#: A rental's DROP, read from a sentence. Every flow asks for a pickup and a drop since
#: 2026-09-29, and a rental sentence now carries both: "8 hours from Shimla, drop at
#: Kufri". Without this the drop was the one detail the code could not read, so with
#: Vertex down the rider who had said it was asked for it anyway (§2).
#:
#: Narrow on purpose. The place AFTER "drop" ("drop at Kufri", "drop me at Kufri", "drop
#: location Kufri"), or a place that STARTS a clause before it ("Kufri drop kar dena",
#: "…, Kufri drop"). An unanchored "X drop" would read "from Shimla Kufri drop" as the
#: place "Shimla Kufri"; that one is left to the extraction model.
_DROP_MARKER_RES = (
    (
        # The word after "drop" must not be the verb it is part of: "Kufri drop kar
        # dena" read the drop as the place "Kar Dena" before this lookahead.
        re.compile(
            r"\bdrop(?:\s*off)?(?:\s+(?:me|us))?"
            r"(?:\s+(?:location|point|spot|place|city|address))?"
            r"(?:\s+(?:ko|to|as|is|at|in|on))?"
            r"\s+(?!(?:kar|karo|karna|kardo|kr|kro|dena|do|dijiye|de)\b)"
            r"([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        # The second word may not be the postposition: "Kufri pe drop" read "Kufri Pe".
        re.compile(
            r"(?:^|[,.;]\s*|\band\s+)"
            r"([a-z][a-z']*(?:\s+(?!(?:pe|par|tak)\b)[a-z][a-z']*)?)"
            r"\s+(?:pe\s+|par\s+|tak\s+)?drop\b",
            re.IGNORECASE,
        ),
        "before",
    ),
)

#: A pickup or drop word that INTRODUCES a place ("… drop at Kufri", "… pickup Shimla").
#: `_break_at_pickup_drop` puts a clause break in front of it, so "pickup Shimla drop
#: Kufri" reads as two places rather than the one place "Shimla Drop Kufri", which
#: `parse_place` rightly rejects — losing both ends. Only when a place FOLLOWS: in "from
#: Shimla Kufri drop" nothing says which word is the drop, and cutting before "drop"
#: there would hand back the pickup "Shimla Kufri". Left whole, it is rejected and the
#: model (or a question) settles it.
_PICKUP_DROP_INTRO_RE = re.compile(
    r"\s+(?=(?:drop(?:\s*off)?|pick\s*up|pickup)"
    r"(?:\s+(?:me|us|location|point|at|to|is|ko))*"
    r"\s+(?!(?:kar|karo|karna|kardo|kr|kro|dena|do|dijiye|de)\b)[a-z])",
    re.IGNORECASE,
)


def _break_at_pickup_drop(text: str) -> str:
    """`text` with a clause break before each pickup/drop word that introduces a place."""
    return _PICKUP_DROP_INTRO_RE.sub(", ", text)

#: Where a DESTINATION sits relative to its marker. Ordered most specific first: the
#: first pattern that yields a place `parse_place` accepts wins, so "I want to go to
#: Manali" is read by the "go to" rule rather than by the bare "to" rule finding "to go".
_SENTENCE_TO_RES = (
    # English — the place follows the marker.
    (
        re.compile(
            r"\b(?:go(?:ing)?|head(?:ing|ed)?|fly(?:ing)?|driv(?:e|ing)"
            r"|travel(?:ling|ing)?|reach(?:ing)?)\s+to\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        re.compile(
            r"\b(?:trip|tour|holiday|vacation|journey|getaway)\s+to\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    (
        re.compile(
            r"\b(?:visit(?:ing)?|explor(?:e|ing)|see)\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})",
            re.IGNORECASE,
        ),
        "after",
    ),
    # Hindi and Hinglish — the place PRECEDES the marker.
    (
        re.compile(
            r"\b([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})\s+(?:jana|jaana|jaunga|jaungi|jaenge|jayenge"
            r"|ghumna|ghoomna|ghoom|dekhna|chalna|chalenge|chalein)\b",
            re.IGNORECASE,
        ),
        "before",
    ),
    (
        # "manali ja rahe hain" — the present continuous, and the commonest way a
        # Hinglish sentence says where somebody is headed.
        re.compile(r"\b([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})\s+ja\s+rah[aeiou]\b", re.IGNORECASE),
        "before",
    ),
    (
        re.compile(r"\b([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})\s+tak\b", re.IGNORECASE),
        "before",
    ),
    (
        # "Shimla se Manali" — the whole reply is the two ends, pickup first: the bare
        # form riders give once they are asked for a pickup and a drop. It used to be
        # read as the ONE place "Shimla Se Manali". Anchored to the end of the message,
        # so "shimla se lekar kasol jana hai" is left to the `jana` rule above.
        re.compile(r"\bse\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})[\s.!]*$", re.IGNORECASE),
        "after",
    ),
    # Last resort. Deliberately last: a bare "to" also matches "to go", which
    # `parse_place` then rejects as request vocabulary, so the specific rules above
    # get their turn first.
    (
        re.compile(r"\bto\s+([a-z][a-z']*(?:\s+[a-z][a-z']*){0,2})", re.IGNORECASE),
        "after",
    ),
)

#: "3 days", "5 din", "a week". Anchored on the unit, so a bare number is never a trip
#: length — the same rule `parse_rental_hours` follows for hours.
_SENTENCE_DAYS_RE = re.compile(
    r"\b(\d{1,2})\s*(?:day|days|din|dino|raat|nights?)\b", re.IGNORECASE
)
#: "for a week", "a week", "one week" — a LENGTH. Deliberately requires the article or
#: a number, so "next week" and "this week" stay what they are: a date, not a seven-day
#: trip the rider never asked for.
_SENTENCE_WEEK_RE = re.compile(r"\b(?:for\s+)?(?:a|one|1)\s+week\b", re.IGNORECASE)


def read_trip_details(text: str) -> dict[str, Any]:
    """Every trip detail stated unambiguously in `text`. Absent keys mean "not said".

    The trip-flow twin of `read_rental_details`, and the reason a sentence like "mei
    shimla jana chahta hu" no longer loses its destination when the extraction model
    misses it. Deterministic, so the same sentence reads the same way every time and
    still reads at all when Vertex is down.

    Returns only what is anchored on a real marker. Nothing here guesses: a sentence
    with no destination marker in it yields no destination.
    """
    if not text or not text.strip():
        return {}

    stripped = text.strip()
    low = stripped.lower()
    found: dict[str, Any] = {}

    # The pickup claims its place first -- see `_PICKUP_MARKER_RES`. Without this,
    # "change the pickup to Shimla" moved the DESTINATION to Shimla.
    # Place scans read the sentence with a break before each "drop <place>" / "pickup
    # <place>" — see `_break_at_pickup_drop`. The rest of the reading uses it as typed.
    clauses = _break_at_pickup_drop(stripped)
    named_pickup = _place_beside_marker(clauses, _PICKUP_MARKER_RES, _TRIP_PLACE_STOP_WORDS)

    # "drop at Manali" before the general destination markers: the question itself now
    # says "drop", so riders answer in those words (see `_DROP_MARKER_RES`).
    destination = _place_beside_marker(
        clauses, _DROP_MARKER_RES, _TRIP_PLACE_STOP_WORDS
    ) or _place_beside_marker(clauses, _SENTENCE_TO_RES, _TRIP_PLACE_STOP_WORDS)
    if destination and destination != named_pickup:
        found["destination"] = destination
    else:
        destination = None

    # "shimla se manali jana hai" — the same sentence carries both ends. `se`/`from` is
    # the origin marker and is a stop word for the destination scan, so the two cannot
    # read each other's place.
    # The TRIP stop words, not the pickup ones. `_place_from_sentence` is the rental
    # pickup scan and knows nothing of "mujhe" — so "mujhe palampur se ... jana hai"
    # collected backwards from `se` and produced the place "Mujhe Palampur", which then
    # went into the reply heading verbatim.
    origin = named_pickup or _place_beside_marker(
        clauses, _SENTENCE_FROM_RES, _TRIP_PLACE_STOP_WORDS
    )
    if origin and origin != destination:
        found["origin"] = origin

    when = _sentence_when(stripped)
    if when:
        found["travel_dates"] = when.strip()

    days_match = _SENTENCE_DAYS_RE.search(low)
    if days_match:
        days = _in_range(int(days_match.group(1)), 1, 30)
        if days:
            found["duration_days"] = days
    elif _SENTENCE_WEEK_RE.search(low):
        found["duration_days"] = 7

    for pattern in _SENTENCE_GROUP_RES:
        match = pattern.search(low)
        if match:
            size = _in_range(int(match.group(1)), 1, 60)
            if size:
                found["group_size"] = size
            break

    return found


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
    # Already a date. The planner runs this on the STORED value every turn, and the
    # stored value is ISO once `sanitize` has resolved it — which the sentence reader
    # then read as a phrase and cut to "09-29". Turn one handed the booking screen a
    # date; every turn after it handed over something the Date field could not show,
    # and the rider had to type the day in again.
    if ISO_DATE_RE.match(text.strip()):
        return text.strip()
    found = _sentence_when(text.strip())
    return found.strip() if found else text.strip()


#: A calendar date as the booking screen takes it. Shared with `sanitize` so the two
#: modules agree on what "already resolved" means (§4.6: one reader, one fact).
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


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

    when = _sentence_when(text.strip())
    if when:
        found["travel_dates"] = when.strip()

    for pattern in _SENTENCE_GROUP_RES:
        match = pattern.search(low)
        if match:
            size = _in_range(int(match.group(1)), 1, 60)
            if size:
                found["group_size"] = size
            break

    # `_PICKUP_MARKER_RES` first, for the same reason as in `read_trip_details`: it
    # reads "change the pickup to Manali", where the place follows a `to` that the
    # plain pickup patterns do not cover.
    clauses = _break_at_pickup_drop(text.strip())
    place = _place_beside_marker(
        clauses, _PICKUP_MARKER_RES, _PLACE_STOP_WORDS
    ) or _place_from_sentence(clauses)
    if place:
        found["origin"] = place

    # The drop, when the sentence marks one — see `_DROP_MARKER_RES`. A drop that came
    # out equal to the pickup is the pickup read twice, not a drop.
    drop = _place_beside_marker(clauses, _DROP_MARKER_RES, _PLACE_STOP_WORDS)
    if drop and drop != found.get("origin"):
        found["destination"] = drop

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



# ── Changing your mind ──────────────────────────────────────────────
#
# "the rider gave their pickup and drop, then changed their mind to update the location"
# — and the assistant carried on with the old one.
#
# The obvious design is a second classifier: ask a model whether this turn is a
# destination change, a date change, a pickup change. It is also the wrong one. It puts
# a network round trip on EVERY turn to decide something the system can already see, it
# can be wrong, and it stops working when Vertex is down.
#
# A change is not a classification. It is a COMPARISON. Every slot the rider has
# answered is already stored on the session, and the readers above already pull values
# out of a sentence with no model at all. So a change is simply: this turn states a
# value for a slot that already holds a DIFFERENT one. That is a dict lookup — tens of
# microseconds, no tokens, no failure mode — exactly as `looks_like_cancellation` is a
# lookup rather than an "is this a cancellation?" call.
#
# Two shapes of correction, handled in two places:
#
#   A sentence — "ab kasol jana hai", "change the pickup to Shimla". The readers above
#   anchor every value on an unmistakable marker, so the marker IS the signal that the
#   rider means to replace rather than merely mention; a place named without one
#   ("what's the weather in Shimla?") yields nothing and can overwrite nothing. The
#   planner compares what they read against what is stored. See `planner._resolve`.
#
#   A bare reply — "actually Kasol", "nahi, Kasol", or just "Kasol". No marker to
#   anchor on, so it is only unambiguous while the assistant's own question is in
#   flight. That is `read_correction_to_asked`, below.

#: Phrasings that introduce a replacement and are not part of the answer itself.
#: Stripped from a bare reply before it is parsed: left in, "actually make it Kasol" is
#: four words, `parse_place` rejects it for length, and the correction is lost.
_CORRECTION_PREFIXES = (
    "sorry i meant", "sorry i mean", "scratch that", "changed my mind",
    "i meant", "i mean", "change it to", "change to", "change the", "change",
    "update it to", "update to", "update", "make it", "lets make it", "let's make it",
    "correction", "actually", "instead", "rather", "sorry", "nope", "no",
    "nahi", "nahin", "nhi", "galat", "badal do", "badal", "badlo", "arre", "oh",
)


def strip_correction_prefix(text: str) -> str:
    """`text` with any leading "actually" / "make it" / "nahi" phrasing removed.

    Applied repeatedly, because riders stack them: "actually, no, make it Kasol".
    """
    out = text.strip()
    while True:
        low = out.lower()
        for prefix in _CORRECTION_PREFIXES:
            if low.startswith(prefix) and (
                len(low) == len(prefix) or not low[len(prefix)].isalnum()
            ):
                out = out[len(prefix) :].lstrip(" ,.!-:—")
                break
        else:
            return out


def read_correction_to_asked(
    asked_field: Optional[str], text: str, current: Any
) -> Any:
    """The new value for `asked_field` when a bare reply changes it, else None.

    `read_answer` deliberately refuses a field that already holds a value — it is a
    net for a MISSED answer, and letting it fire on a filled slot would let any stray
    word overwrite a confirmed detail. But that refusal is also why "nahi, Kasol"
    landed nowhere once a destination was in.

    The narrow case where a bare reply is unambiguous is this one: the assistant asked
    this exact field last turn. `last_asked_field` is rewritten by `_resolve` every
    turn and is None when nothing was asked, so the window is one turn wide. Within it,
    `SLOTS[asked_field].parse` is typed to that field — a reply it cannot read yields
    nothing and changes nothing.
    """
    if not asked_field or asked_field not in SLOTS or not text or not text.strip():
        return None
    if current in (None, "", [], {}):
        return None  # Nothing to correct; filling an empty slot is `read_answer`'s job.

    bare = strip_correction_prefix(text)
    if not bare:
        return None

    stated = SLOTS[asked_field].parse(bare)
    if stated is None:
        return None
    if isinstance(stated, str) and isinstance(current, str):
        if stated.strip().casefold() == current.strip().casefold():
            return None  # Restating what we hold is not a change.
    elif stated == current:
        return None
    return stated


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


#: The rider is asked for a PICKUP and a DROP, the way a cab app asks — one wording in
#: every flow. These replaced four questions for two details: "आप किस शहर में रहते
#: हैं?" (local), "Which city are you travelling from?" (outstation), "Where should the
#: car pick you up?" (rental) and "Where would you like to go?". A resident was being
#: asked where they LIVE when what the booking needs is where the cab should come.
_DESTINATION_Q = {
    "hi": "आपको कहाँ ड्रॉप करना है?",
    "en": "Where should we drop you?",
}
# Used after a first attempt failed to land — same question, with examples, so the
# user can see the shape of answer we need instead of being asked the same words.
_DESTINATION_Q_RETRY = {
    "hi": "बस ड्रॉप की जगह का नाम बता दें — जैसे मनाली, शिमला, धर्मशाला या कसोल।",
    "en": "Just the drop location is enough — for example Manali, Shimla, "
          "Dharamshala or Kasol.",
}
_ORIGIN_Q = {
    "hi": "आपको कहाँ से पिकअप करना है?",
    "en": "Where should we pick you up?",
}
_ORIGIN_Q_RETRY = {
    "hi": "बस पिकअप की जगह का नाम बता दें — कोई शहर, कस्बा या इलाका।",
    "en": "Just the pickup location is enough — a city, town or area.",
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
    "hi": "आपके साथ कितने लोग यात्रा कर रहे हैं?",
    "en": "How many people will be travelling in total?",
}
_GROUP_SIZE_Q_RETRY = {
    "hi": "आपके साथ कितने लोग हैं? बस संख्या बता दें — जैसे 2 या 4।",
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
#: Asked when the rider named a day that has already passed. It outranks both wordings
#: above, for the same reason a duration we do not sell outranks them: they gave a
#: specific answer, we understood it exactly, and the one thing they must not be told is
#: the original question again. `{day}` is quoted back in their own words so it is clear
#: WHICH answer could not be used — the difference between "we can't book that" and the
#: silence that got this reported.
_WHEN_Q_PAST = {
    "hi": "“{day}” तो बीत चुका है — किस दिन के लिए बुक करूँ? आज, कल, या कोई तारीख।",
    "en": "“{day}” has already passed — which day should I book for? "
          "Today, tomorrow, or a date.",
}


def when_question(lang_key: str, attempt: int = 0, *, past_day: Optional[str] = None) -> str:
    """The "when?" question, worded for what just happened.

    `past_day` is a day the rider named that has already gone. It wins over the retry
    wording: a rider who answered precisely is owed a straight answer about THAT answer,
    not a repeat of the question or a hint about what a date looks like.
    """
    if past_day:
        return _q(_WHEN_Q_PAST, lang_key).format(day=past_day)
    return _q(_WHEN_Q_RETRY if attempt else _WHEN_Q, lang_key)


#: Why a place name could not be used: the map has no such place — a typo, or a word
#: that is not a place.
PLACE_NOT_FOUND = "not_found"

#: Asked when a place the rider named cannot be found on the map. Before this existed
#: there were two outcomes, both reported as hallucination: the geocoder's first hit was
#: taken whatever it was ("xyz" became a road in Thane and was priced), or nothing was
#: found and the turn went ahead anyway, leaving the writer a destination it had no facts
#: about. `{place}` is quoted in the rider's own words (§4.7), so it is clear WHICH answer
#: we could not place — and why the question is back.
_PLACE_NOT_FOUND_Q: dict[str, dict[str, str]] = {
    SlotName.DESTINATION: {
        "hi": "“{place}” नाम की कोई जगह मुझे नहीं मिली। कृपया ड्रॉप की जगह की स्पेलिंग "
              "जाँच कर दोबारा बताएँ — जैसे मनाली, शिमला, धर्मशाला या कसोल।",
        "en": "I couldn't find a place called “{place}”. Could you check the spelling "
              "of your drop location — for example Manali, Shimla, Dharamshala or Kasol?",
    },
    SlotName.ORIGIN: {
        "hi": "“{place}” नाम की कोई जगह मुझे नहीं मिली। आपको कहाँ से पिकअप करना है? "
              "कृपया नाम की स्पेलिंग जाँच लें।",
        "en": "I couldn't find a place called “{place}”. Where should we pick you up? "
              "Please check the spelling.",
    },
}
#: Said when we have stopped asking (the per-field cap or the rider's patience ran out)
#: and go ahead without the place. The reply still has to say the name was not used —
#: silently proceeding is how the writer came to plan a trip nobody asked for.
_PLACE_DROPPED_NOTE = {
    "hi": "“{place}” मुझे नक्शे पर नहीं मिला, इसलिए मैंने इसे छोड़ दिया है।",
    "en": "I couldn't find “{place}” on the map, so I've left it out.",
}


def unplaceable_question(
    field: str,
    place: str,
    lang_key: str,
    reason: str = PLACE_NOT_FOUND,
) -> str:
    """The question that goes back when a place the rider named cannot be used.

    Deterministic, like every other question: no model decides how to say "I couldn't
    find that", so the wording cannot drift into a guess about what they meant. One
    wording per field in every flow, matching the pickup and drop questions.
    """
    table = _PLACE_NOT_FOUND_Q.get(field, _PLACE_NOT_FOUND_Q[SlotName.DESTINATION])
    return _q(table, lang_key).format(place=_quotable(place))


def unplaceable_note(place: str, lang_key: str) -> str:
    """One line for the reply when we went ahead without a place we could not find."""
    return _q(_PLACE_DROPPED_NOTE, lang_key).format(place=_quotable(place))


def _quotable(text: str) -> str:
    quoted = " ".join((text or "").split())
    return quoted if len(quoted) <= 40 else quoted[:40].rstrip() + "…"
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

#: Prepended when the rider's reply could not be used at all — a typo, or a line about
#: something else — so the re-ask reads as "I didn't understand that", not a robotic
#: repeat of the same question (§7: the rider's own words go back to them). See the
#: planner's unread-reply guard. `{text}` is the rider's message, quoted and trimmed.
_UNREAD_REPLY = {
    "hi": "क्षमा करें, मैं “{text}” ठीक से समझ नहीं पाया।",
    "en": "Sorry, I didn't quite catch “{text}”.",
}
#: Same, when there is no text worth quoting back (an empty or whitespace message).
_UNREAD_REPLY_BARE = {
    "hi": "क्षमा करें, मैं ठीक से समझ नहीं पाया।",
    "en": "Sorry, I didn't quite catch that.",
}


def unread_reply_preamble(text: str, lang_key: str) -> str:
    """A short "I didn't understand that" line, in the rider's language.

    Quotes what they typed so the re-ask that follows is clearly a re-ask and not a
    repeat — repeating the identical question is the thing §7 warns reads as not having
    listened. The quote is trimmed so a long or messy paste does not swamp the question,
    and dropped entirely when there is nothing to quote.
    """
    quoted = " ".join((text or "").split())
    if not quoted:
        return _q(_UNREAD_REPLY_BARE, lang_key)
    if len(quoted) > 40:
        quoted = quoted[:40].rstrip() + "…"
    return _q(_UNREAD_REPLY, lang_key).format(text=quoted)
_RETURN_Q = {
    "hi": "क्या आपने वापसी के लिए कैब बुक कर ली है, या अपने साधन से लौटेंगे?",
    "en": "Have you already booked your return cab, or will you be making your own way back?",
}

_LABELS = {
    "destination": {"hi": "ड्रॉप की जगह", "en": "drop location"},
    "rental_hours": {"hi": "किराये के घंटे", "en": "rental duration"},
    "origin": {"hi": "पिकअप की जगह", "en": "pickup location"},
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
        parse=parse_location,
        write=_simple_write("destination"),
        label=_label("destination"),
    ),
    SlotName.ORIGIN: Slot(
        name=SlotName.ORIGIN,
        ask=lambda lang, _vt, attempt: _q(
            _ORIGIN_Q_RETRY if attempt else _ORIGIN_Q, lang
        ),
        read=lambda e: e.origin or e.home_city,
        parse=parse_location,
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
        # `clarify_node` re-words this when the rider named a day that has gone — see
        # `when_question`. The slot itself cannot: it is handed the entities, and a
        # value that was dropped is exactly the thing no longer in them.
        ask=lambda lang, _vt, attempt: when_question(lang, attempt),
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

#: A Himachal resident going somewhere is booking a RIDE, not planning a holiday —
#: `response_mode` already returns RIDE_MODE for them and the reply is a cab, a fare, a
#: drive time and the weather, with no itinerary anywhere in it.
#:
#: So there are no DAYS. "How many days are you travelling for?" is a question about an
#: itinerary, and asking a resident how many days their drive to Manali lasts is the
#: same mistake as asking a rental rider where they are going: a question whose answer
#: the reply has no use for, about a thing they did not ask for. It was being asked
#: because this table was keyed on the INTENT alone, and a local and a visitor both
#: arrive here as `plan_trip`.
#:
#: Everything else a ride genuinely needs stays: where to, where from (the fare is
#: priced on that leg), how many (it picks the cab) and when (the fare's booking time
#: and the forecast).
_RIDE_NEEDS: dict[str, Need] = {
    SlotName.DESTINATION: Need.REQUIRED,
    SlotName.ORIGIN: Need.IMPORTANT,
    SlotName.GROUP_SIZE: Need.IMPORTANT,
    SlotName.TRAVEL_DATES: Need.IMPORTANT,
    # Deliberately absent, not UNUSED-by-omission: see above. A ride has no duration in
    # days, and `_PLAN_TRIP_NEEDS` is where that question belongs.
    SlotName.RETURN_JOURNEY_PREFERENCE: Need.OPTIONAL,
}

#: A rental is a car and a driver for a block of time. What the fare needs is the
#: package and the pickup; the drop does not change the price.
_RENTAL_NEEDS: dict[str, Need] = {
    SlotName.RENTAL_HOURS: Need.REQUIRED,
    SlotName.ORIGIN: Need.REQUIRED,
    # The drop — where the hire ends — asked right after the pickup, like every flow
    # (a 2026-09-29 decision; a rental used to have no drop at all). IMPORTANT, never
    # REQUIRED: plenty of hires end wherever the day ends, so a rider who does not say
    # still reaches the booking screen, and "same place" is read as the pickup
    # (`means_same_as_pickup`). Only a drop the rider actually named is kept — see
    # `planner._strip_trip_fields`.
    SlotName.DESTINATION: Need.IMPORTANT,
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
#: A ride — route, drive time, day, weather, and the booking screen — and nothing
#: else. Residents already know the place; what they want is the ride, so an itinerary
#: telling them about Mall Road is noise around the useful part. It names no cab: see
#: `lists_cabs`.
RIDE_MODE = ResponseMode.RIDE
#: A short direct answer to a question about a place. No trip, no cab.
ANSWER_MODE = ResponseMode.ANSWER
#: Nothing is being sold: the rider called it off. No fare, no cab list, no itinerary
#: and no booking screen — see `cancellation.cancellation_message`.
CANCELLED_MODE = ResponseMode.CANCELLED
#: A rental quote — package, cab options and price. No itinerary, no route: the rider
#: has not said where they are going, and hiring by the hour means they need not.
RENTAL_MODE = ResponseMode.RENTAL

#: The replies that list and recommend cabs. Only a rental's.
#:
#: A product decision (2026-09-28), and a reversal of the one before it: a local ride
#: used to end in every cab that seats the party with its price, and a trip plan in a
#: "Book Your Him Cab" list with one cab flagged as recommended. Now a local rider and a
#: visitor are handed to the app's booking screen, which lists the cabs and their fares
#: itself, and the chat names none. A rental keeps its recommendation because its
#: booking screen will not price anything until a cab is chosen, and a rider who had to
#: pick one there typed every detail in again.
#:
#: One table, read by the tool executor, which then never asks for a fare the reply
#: would not show. Nothing downstream re-decides it: with no fare on state, the reply,
#: the itinerary, the `cab_fare` event and the handoff all carry no cab — and the
#: guardrail takes out any vehicle the model names anyway.
_CAB_LISTING_MODES = frozenset({RENTAL_MODE})


def lists_cabs(mode: Optional[str]) -> bool:
    """True when this reply lists cabs and recommends one. See `_CAB_LISTING_MODES`."""
    return mode in _CAB_LISTING_MODES


def needs_for(
    intent: Optional[str], visitor_type: Optional[str] = None
) -> dict[str, Need]:
    """The slot policy for an intent. Unknown intents plan a trip (the safe default).

    The PERSONA decides, not just this turn's intent — the same rule `response_mode`
    already applies, and for the same reason. That holds for a rental (hours, not a
    destination) and equally for a local (a ride, so no trip length). The router
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
    # Same rule, one persona along: a LOCAL travelling is booking a ride, and a ride has
    # no days. Scoped to the travel intents so a resident asking about the weather in
    # Manali still gets `_LOOKUP_NEEDS` and is asked nothing but the place.
    if visitor_type == LOCAL and intent in _TRAVEL_INTENTS:
        return _RIDE_NEEDS
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
