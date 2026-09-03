"""
app/agent/sanitize.py
──────────────────────
The one gate every user-derived value passes before the system acts on it.

WHY THIS EXISTS
───────────────
A value reached the entity bag by three different routes and only one of them was
checked. `slots.parse_*` validates a bare reply strictly, and `slots.read_rental_details`
anchors each field on an unmistakable marker — but the EXTRACTION MODEL wrote straight
into the bag. So `group_size=500`, `duration_days=-3`, a pickup of "Here" and a travel
date of "yesterday" all had a clear run to the fare request, the booking screen and the
reply. The bounds existed; they were simply on the path the model did not take.

So this module is the gate, and `planner._resolve` puts every field through it
regardless of where the value came from. A value that does not survive is DROPPED, not
corrected: the slot goes back to empty, `slots.decide` sees it missing and asks again.
Asking one more question is cheap. Sending a car to a place called "Here", quoting a
fare for a party of 500, or putting "yesterday" in a booking screen's date field is not.

VALIDATING IS NOT PARSING
─────────────────────────
Deliberately more permissive than `slots.parse_*`, and the difference matters. Those
parsers answer "is this bare reply an answer to the question I just asked?", where a
wrong yes corrupts the request — so they reject freely, and rejecting a real answer
costs only one more question. This module answers "is this value usable at all?" about
something already identified as a destination or a group size. Applying the parsers'
strictness here would throw away good data: `parse_place` rejects any string containing
a digit, which would lose "Sector 17 Chandigarh".

DATES
─────
The reported bug: a rental booked for "tomorrow" or "day after tomorrow" carried that
phrase into the booking screen's date field, which cannot render it. Relative days are
resolved to a real date here (`resolve_relative_day`), and a date in the PAST is
rejected outright — nobody hires a car for yesterday, and a rider who typed it has made
a mistake worth putting back to them as a question rather than passing to a backend.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from app.agent import slots
from app.core.enums import SlotName
from app.core.logging import get_logger

log = get_logger(__name__)

#: Him Cab operates in India, and a booking's "today" is the rider's today. Using the
#: server's date would move the boundary between "tomorrow" and "day after tomorrow" by
#: five and a half hours — wrong for anyone booking late at night.
_IST = timezone(timedelta(hours=5, minutes=30))

#: Longest a place name may be. Generous: "Kangra Fort Road, Dharamshala" is fine, an
#: extracted paragraph is not.
_MAX_PLACE_CHARS = 60
_MAX_PLACE_WORDS = 6

#: Bounds. Each one is a value the rest of the system already assumes.
_GROUP_SIZE_RANGE = (1, 60)
_DURATION_DAYS_RANGE = (1, 30)
_RENTAL_HOURS_RANGE = (1, 24)
#: A pickup further out than this is not a booking anyone is making now.
_MAX_DAYS_AHEAD = 365

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
#: Zero-width and bidi marks. Invisible, so they survive every eyeball review and then
#: break an exact-match comparison or a geocoder lookup for reasons nobody can see.
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠-⁯﻿]")
_WHITESPACE_RE = re.compile(r"\s+")


def today_ist() -> date:
    """The rider's today. See `_IST`."""
    return datetime.now(_IST).date()


def clean_text(value: Any, *, max_chars: int = 200) -> Optional[str]:
    """A trimmed, normalised string, or None when nothing usable is left.

    NFKC so visually identical inputs compare equal, then invisible and control
    characters removed and whitespace collapsed. Applied to every string field: a
    zero-width space inside a place name is invisible in a log, in a bug report and in
    a code review, and it makes the geocoder miss.
    """
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_RE.sub(" ", _INVISIBLE_RE.sub("", text))
    text = _WHITESPACE_RE.sub(" ", text).strip()
    if not text or len(text) > max_chars:
        return None
    return text


def _bounded_int(value: Any, low: int, high: int) -> Optional[int]:
    """`value` as an int inside [low, high], or None. Accepts "4" as well as 4."""
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if low <= number <= high else None


#: Words that carry no meaning of their own in a place name. Discounted before the
#: "is this ALL request vocabulary?" test in `place`, so an article cannot rescue a
#: phrase that is otherwise entirely about the request.
_FILLER_WORDS = frozenset({
    "a", "an", "the", "for", "from", "to", "of", "at", "in", "on",
    "my", "me", "please", "ka", "ki", "ke", "se", "mein", "par",
})


def place(value: Any) -> Optional[str]:
    """A usable place name, or None.

    Rejects the three things that have actually reached this field and been acted on:
    deixis ("here", "my location" — see `slots.means_current_location`), words that
    describe the REQUEST rather than name somewhere ("rent a car", "cancel"), and
    anything too long to be a place. Digits are allowed, unlike in `slots.parse_place`
    — see VALIDATING IS NOT PARSING.
    """
    text = clean_text(value, max_chars=_MAX_PLACE_CHARS)
    if text is None:
        return None
    if slots.means_current_location(text):
        return None
    tokens = slots._tokens(text)
    if not tokens or len(tokens) > _MAX_PLACE_WORDS:
        return None
    # A name made ENTIRELY of request vocabulary is not a place. One such word inside a
    # real name is fine — "Cancel" alone is the bug, "Kullu Bus Stand" is not.
    #
    # Fillers are discounted first, or the rule is defeated by an article: "rent a car"
    # has three tokens, two of them request words and one an "a", so "every token" was
    # false and the phrase passed as a place name.
    content = [t for t in tokens if t not in _FILLER_WORDS]
    if content and all(
        t in slots._NOT_A_PLACE or t in slots._NON_ANSWERS for t in content
    ):
        return None
    return text


#: Relative day expressions, mapped to an offset in days from the rider's today.
#:
#: `kal` and `parso` each mean a day in BOTH directions in Hindi — kal is yesterday and
#: tomorrow, parso is the day before yesterday and the day after tomorrow. Which one is
#: meant comes from the verb tense, which we do not parse. The future reading is taken
#: because this is a booking: somebody arranging a cab means the coming day, and the
#: past reading is not something anyone can book anyway.
_RELATIVE_DAYS: dict[str, int] = {
    "day before yesterday": -2,
    "yesterday": -1,
    "today": 0, "tonight": 0, "tonite": 0, "tdy": 0, "aaj": 0, "aj": 0,
    "tomorrow": 1, "tommorow": 1, "tomorow": 1, "tomarrow": 1, "tomorrw": 1,
    "tmrw": 1, "tmr": 1, "tmw": 1, "kal": 1,
    "day after tomorrow": 2, "day after tmrw": 2, "day after": 2,
    "parso": 2, "parson": 2, "parsoo": 2,
}

#: Longest first, so "day after tomorrow" is matched before "tomorrow" — the whole
#: point. Matching the short one first is a silent off-by-one day: the rider asks for
#: Thursday and the cab is booked for Wednesday.
_RELATIVE_DAY_PATTERNS: list[tuple[re.Pattern, int]] = [
    (
        re.compile(
            r"(?<!\w)" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?!\w)",
            re.IGNORECASE,
        ),
        offset,
    )
    for phrase, offset in sorted(_RELATIVE_DAYS.items(), key=lambda kv: -len(kv[0]))
]

#: The same words in Devanagari, matched by substring because the word-boundary guard
#: above is written for Latin script. Longest first, for the same reason.
_DEVANAGARI_DAYS: list[tuple[str, int]] = [
    ("परसों", 2),   # parson
    ("आज", 0),                     # aaj
    ("कल", 1),                     # kal
]


def resolve_relative_day(text: Any, *, today: Optional[date] = None) -> Optional[date]:
    """The calendar date a relative day phrase names, or None if it names none.

    "tomorrow", "day after tomorrow", "parso", "tmrw" — the phrases riders actually
    type into a date field. Returns a real date so the booking screen has something it
    can render; a phrase this cannot resolve ("next week", "after Diwali") is not a
    single day and is deliberately left alone by the callers.
    """
    cleaned = clean_text(text)
    if cleaned is None:
        return None
    base = today or today_ist()
    for pattern, offset in _RELATIVE_DAY_PATTERNS:
        if pattern.search(cleaned):
            return base + timedelta(days=offset)
    for word, offset in _DEVANAGARI_DAYS:
        if word in cleaned:
            return base + timedelta(days=offset)
    return None


def travel_date(
    value: Any, *, as_iso: bool = False, today: Optional[date] = None
) -> Optional[str]:
    """A travel date we are willing to act on, or None.

    Two jobs. It refuses a day in the PAST, whatever wording arrived at it — "yesterday"
    is not a booking, it is a mistake, and dropping it puts the question back to the
    rider instead of sending a backend a date it must reject. And with `as_iso` it hands
    back YYYY-MM-DD for the phrases that name one exact day, which is what the rental
    booking screen needs in its date field: it cannot render "day after tomorrow".

    A phrase naming no single day ("next week", "sometime after Diwali") survives
    untouched. It is still the rider's own answer, still useful in the reply, and
    inventing a date for it would be worse than carrying the words.
    """
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    base = today or today_ist()
    resolved = resolve_relative_day(cleaned, today=base)
    if resolved is None:
        return cleaned
    if resolved < base:
        log.info(
            "travel_date_in_the_past_dropped", value=cleaned, resolved=str(resolved)
        )
        return None
    if (resolved - base).days > _MAX_DAYS_AHEAD:
        log.info("travel_date_too_far_ahead_dropped", value=cleaned)
        return None
    return resolved.isoformat() if as_iso else cleaned


def group_size(value: Any) -> Optional[int]:
    """How many people. Above `_GROUP_SIZE_RANGE` nothing Him Cab prices can carry them."""
    return _bounded_int(value, *_GROUP_SIZE_RANGE)


def duration_days(value: Any) -> Optional[int]:
    """Trip length in days."""
    return _bounded_int(value, *_DURATION_DAYS_RANGE)


def rental_hours(value: Any) -> Optional[int]:
    """A plausible number of hours. Whether we SELL it is checked separately.

    `planner._reject_unsold_duration` owns that, because it needs the rate card and is
    async. This is only the sanity bound — a negative or 300-hour hire is not a rental
    whatever the rate card says.
    """
    return _bounded_int(value, *_RENTAL_HOURS_RANGE)


_CLOCK_RE = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


def pickup_time(value: Any) -> Optional[str]:
    """A real 24-hour clock reading as HH:MM, or None.

    The backend reads this as `bookingTime` and applies the night rate from it, so a
    value that is not a clock time is a value that changes what the rider pays.
    """
    text = clean_text(value, max_chars=10)
    if text is None:
        return None
    match = _CLOCK_RE.match(text)
    if not match:
        return None
    return f"{int(match.group(1)):02d}:{match.group(2)}"


def cab_category(value: Any) -> Optional[str]:
    """A cab tier we recognise, lower-cased, or None.

    Matched against the same vocabulary the quoting turn uses. A tier this accepts may
    still not be in the fare the backend returned — the tool executor checks that and
    tells the rider — but a value that is not a cab tier at all stops here.
    """
    text = clean_text(value, max_chars=40)
    if text is None:
        return None
    return slots.parse_cab_category(text)


#: Free-text fields validated as place names. A field absent from the gate below is
#: passed through untouched: this guards the values the system ACTS on, and is not a
#: schema for the whole bag.
_PLACE_FIELDS = ("destination", "origin", "home_city")


def sanitize_entities(
    merged: dict,
    *,
    is_rental: bool = False,
    today: Optional[date] = None,
) -> tuple[dict, list[str]]:
    """Validate every user-derived field in `merged`. Returns (cleaned, dropped names).

    Mutates nothing: the caller gets a new dict and the list of fields that did not
    survive, which is what it logs. A dropped field is left EMPTY rather than corrected,
    so the slot policy sees it missing and asks for it again.

    `is_rental` only decides whether a resolvable travel date is rewritten to ISO. A
    rental's date goes into the booking screen's date field and has to be a date; a
    trip's "sometime after Diwali" is worth keeping word for word.
    """
    cleaned = dict(merged)
    dropped: list[str] = []

    def apply(field: str, validator) -> None:
        if field not in cleaned or cleaned[field] in (None, "", [], {}):
            return
        after = validator(cleaned[field])
        if after is None:
            cleaned[field] = None
            dropped.append(str(field))
        else:
            cleaned[field] = after

    for field in _PLACE_FIELDS:
        apply(field, place)

    apply(SlotName.GROUP_SIZE, group_size)
    apply(SlotName.DURATION_DAYS, duration_days)
    apply(SlotName.RENTAL_HOURS, rental_hours)
    apply(SlotName.PICKUP_TIME, pickup_time)
    apply(SlotName.CAB_CATEGORY, cab_category)
    apply(
        SlotName.TRAVEL_DATES,
        lambda v: travel_date(v, as_iso=is_rental, today=today),
    )

    return cleaned, dropped
