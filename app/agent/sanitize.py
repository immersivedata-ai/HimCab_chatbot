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


def resolve_relative_day(text: Any, *, today: Optional[date] = None) -> Optional[date]:
    """The calendar date a relative day phrase names, or None if it names none.

    "tomorrow", "day after tomorrow", "parso", "tmrw", "3 din baad" — the phrases riders
    actually type into a date field. Returns a real date so the booking screen has
    something it can render; a phrase naming no single day ("next week", "after Diwali")
    is left alone by the callers.

    The phrase table lives in `slots.RELATIVE_DAY_PHRASES`, with the rest of the
    reading. It used to live here, duplicated against the sentence reader's own list of
    day words, and the two drifted: "day after tmrw" resolved correctly here and was
    read as plain "tmrw" over there, so the booking landed a day early. Reading in one
    place and the CALENDAR in this one is what stops that recurring.

    A date in the PAST comes back as a past date, not as None. Refusing it is
    `travel_date`'s job, and the two are kept apart so the refusal can name the day the
    rider actually said instead of pretending not to have understood them.
    """
    cleaned = clean_text(text)
    if cleaned is None:
        return None
    offset = slots.relative_day_offset(cleaned)
    if offset is None:
        return None
    return (today or today_ist()) + timedelta(days=offset)


_MONTH_NUMBERS = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"), ("feb", "february"), ("mar", "march"),
            ("apr", "april"), ("may",), ("jun", "june"), ("jul", "july"),
            ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"),
            ("nov", "november"), ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}
_MONTH_ALT = "|".join(sorted(_MONTH_NUMBERS, key=len, reverse=True))
#: Whole-value patterns only: this resolves a travel date the readers have already
#: reduced to the day (`slots.tidy_when`), not a sentence. Day before month in the
#: numeric form, because that is how dates are written in India — "15/10" is October.
_CALENDAR_RES = (
    re.compile(rf"^(?P<d>\d{{1,2}})\s*(?:st|nd|rd|th)?\s+(?P<m>{_MONTH_ALT})\.?(?:,?\s+(?P<y>\d{{4}}))?$", re.I),
    re.compile(rf"^(?P<m>{_MONTH_ALT})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?$", re.I),
    re.compile(r"^(?P<d>\d{1,2})[/.-](?P<mn>\d{1,2})(?:[/.-](?P<y>\d{2}|\d{4}))?$"),
)


#: How far back a year-less date may lie and still be read as THIS year's — i.e. as a
#: mistake to refuse rather than next year's. A month covers "the 5th" typed on the
#: 20th; anything older is almost always a date in the coming year.
_RECENTLY_GONE_DAYS = 31


def resolve_calendar_day(text: Any, *, today: Optional[date] = None) -> Optional[date]:
    """The date a calendar phrase names ("15 Oct", "Oct 15", "15/10"), or None.

    The companion to `resolve_relative_day`. Relative days were already turned into a
    date for the booking screen; a rider who typed "15 Oct" had those words sent into
    its Date field instead, which cannot show them — so they typed the day in again.

    With no year it is the NEXT such day — "15 Jan" typed in September is next January
    — except within `_RECENTLY_GONE_DAYS` of today. A day that went by a fortnight ago
    is a mistake, not a booking for next year, so it comes back as the past date it
    names and `travel_date` refuses it by name. Rolling that into next year would
    quietly book a car eleven months out for a rider who mistyped the month.
    """
    cleaned = clean_text(text)
    if cleaned is None:
        return None
    base = today or today_ist()
    for pattern in _CALENDAR_RES:
        match = pattern.match(cleaned)
        if not match:
            continue
        parts = match.groupdict()
        month = (
            _MONTH_NUMBERS[parts["m"].lower()] if parts.get("m") else int(parts["mn"])
        )
        year = int(parts["y"]) if parts.get("y") else base.year
        if year < 100:
            year += 2000
        try:
            day = date(year, month, int(parts["d"]))
        except ValueError:
            return None        # "31/02" names no day
        if not parts.get("y") and day < base - timedelta(days=_RECENTLY_GONE_DAYS):
            try:
                day = day.replace(year=base.year + 1)
            except ValueError:
                return None    # 29 Feb, and next year has none
        return day
    return None


def _resolve_day(text: str, today: date) -> Optional[date]:
    """Any single day `text` names — relative or on the calendar — or None."""
    if slots.ISO_DATE_RE.match(text):
        return date.fromisoformat(text)
    return resolve_relative_day(text, today=today) or resolve_calendar_day(text, today=today)


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
    resolved = _resolve_day(cleaned, base)
    if resolved is None:
        return cleaned
    if resolved < base:
        log.info(
            "travel_date_in_the_past_dropped", value=cleaned, resolved=str(resolved)
        )
        # Dropped, but see `day_already_gone` — the caller asks that too, so the
        # question that goes back can name the day instead of repeating itself.
        return None
    if (resolved - base).days > _MAX_DAYS_AHEAD:
        log.info("travel_date_too_far_ahead_dropped", value=cleaned)
        return None
    return resolved.isoformat() if as_iso else cleaned


def day_already_gone(value: Any, *, today: Optional[date] = None) -> Optional[str]:
    """The rider's own words for a day that has already passed, or None.

    The companion to `travel_date` rejecting one. Dropping the value is right — nobody
    hires a car for yesterday — but dropping it SILENTLY was not: the slot went empty,
    the policy asked "when are you planning to travel?", and the rider who had just
    answered that question got it back word for word with no sign of what was wrong.
    Reported as the assistant not detecting the date at all, which is exactly what it
    looks like from the other side.

    Returns the phrase as they typed it, so the question can quote it back.
    """
    cleaned = clean_text(value)
    if cleaned is None:
        return None
    base = today or today_ist()
    resolved = _resolve_day(cleaned, base)
    return cleaned if resolved is not None and resolved < base else None


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
