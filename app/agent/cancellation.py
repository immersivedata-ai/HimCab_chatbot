"""
app/agent/cancellation.py
──────────────────────────
Calling it off — everything the agent knows about a rider changing their mind.

WHY THIS EXISTS
───────────────
A rider who backs out is not a rider with a new requirement, and the difference used to
be invisible to this system. There was no cancellation anywhere in it: "cancel" arrived
as an ordinary message, the router classified it as trip planning, and the slot parsers
read it as an ANSWER to whatever question was in flight — `parse_place("cancel")`
returned "Cancel", which became the pickup city, which was geocoded, which was priced.
The rider who had just called it off got a fare, a cab list and a booking screen for a
request they had abandoned.

Four things fix that, and they are all here so they cannot drift apart:

  1. `looks_like_cancellation` — recognised in CODE, before any model call, so an
     unmistakable "cancel" cannot be reinterpreted by a classifier having a bad day
     (and still works when Vertex is down).
  2. `cancel_scope` — WHAT they want called off. Getting this wrong is dangerous in
     both directions; see `CancelScope`.
  3. `cancellation_message` — the reply, from a table. Every line of it is a claim
     about somebody's money or somebody's driver, and the writer LLM is precisely the
     component that was getting those wrong, so it is not given the opportunity.
  4. `clear_request` — the details are dropped, so the next message starts a genuinely
     new request rather than one built half out of the abandoned one.

WHAT THIS DELIBERATELY DOES NOT DO
──────────────────────────────────
It never cancels a real trip. The backend has one
(`DELETE /api/rider/trip/cancel-ride/{id}`), and it is irreversible: it flips the trip
to CANCELLED, stores a reason and pushes a notification to the driver. This service
never books anything — it hands off to the app, which creates the trip — so it does not
hold a trip id and would have to guess which booking to destroy from a sentence an LLM
classified. A rider with a confirmed cab is sent to the screen that owns it instead.
"""

from __future__ import annotations

import re

from app.core.enums import CancelScope, ReplyLanguage
from app.models.session import ExtractedEntities

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Asking ABOUT cancelling is not cancelling. Matched before anything else, because
#: "can I cancel later?" answered by cancelling is the same bug pointing the other way.
_QUESTION_HINTS = ("?", "kya ", "kaise", "kahan ", "kaunsa", "what ", "which ", "how ",
                   "why ", "can i ", "kar sakta", "kar sakte", "sakti")

#: Phrases that NAME what is being called off. An instruction whatever the punctuation
#: — "can you cancel this?" is not a question about our policy. Padded on both sides
#: when matched, so "cancellation" never matches "cancel".
_EXPLICIT_CANCEL_PHRASES = (
    "cancel it", "cancel that", "cancel this", "cancel my booking",
    "cancel the booking", "cancel the cab", "cancel the ride", "cancel my ride",
    "cancel my trip", "cancel booking", "cancel the request", "cancel my request",
    "never mind", "nevermind", "forget it", "forget this", "leave it",
    "not interested", "no longer interested", "changed my mind", "change my mind",
    "call it off", "call off", "abort", "drop it", "scrap it",
    "i dont want it", "i don t want it", "dont want it", "don t want it",
    "i dont want the cab", "dont need it", "don t need it", "no need now",
    "not needed", "no longer need it", "start over", "start again", "reset",
    # Hindi and Hinglish — what riders actually type.
    "rehne do", "rehnedo", "rahne do", "rehne dijiye",
    "cancel karo", "cancel kar do", "cancel kardo",
    "cancel kar dijiye", "cancel karna hai", "band karo", "band kar do",
    "nahi chahiye", "nhi chahiye", "nahin chahiye", "ab nahi chahiye", "mat karo",
    "mat kijiye", "rehne de", "छोड़ दो", "रहने दो", "कैंसल", "कैंसल करो",
    "नहीं चाहिए", "बंद करो",
)

#: On their own these mean the same thing, but they also appear inside genuine
#: questions ("can I cancel later?"), so a question mark outranks them.
_BARE_CANCEL_WORDS = ("cancel", "abort", "reset", "cancelled", "canceled", "कैंसल")

#: Says a booking ALREADY EXISTS, in the past tense — present only in the reading where
#: the rider means a trip this chat did not make.
_ALREADY_BOOKED_MARKERS = (
    "already booked", "i booked", "we booked", "have booked", "had booked",
    "i have a booking", "my booking", "my existing booking", "the cab i booked",
    "the ride i booked", "the trip i booked", "booked yesterday", "booked earlier",
    "booked a cab", "booked a ride", "booked cab", "confirmed booking",
    "driver is coming", "driver assigned", "driver has been assigned",
    "book kiya", "book kar diya", "book kar di", "booking kar di", "booking hai",
    "pehle se book", "बुक किया", "बुकिंग है",
)

#: Never a cancellation, however much it looks like one. Telling us to stop ASKING
#: means get on with it — that is `_PROCEED_PHRASES` in the planner — and treating it
#: as a cancellation would throw away a request the rider still wants.
#:
#: For the same reason "chhodo" is absent from the list above, though it reads like
#: "drop it": the planner already treats it as "stop asking and just make it", and a
#: word that means both must keep the meaning that costs less when it is wrong.
_NOT_A_CANCELLATION = (
    "stop asking", "stop the questions", "no more questions", "dont ask", "don t ask",
    "cancellation policy", "cancellation charge", "cancellation charges",
    "cancellation fee", "cancellation fees", "cancel karne par", "cancel karne ka",
    "if i cancel", "if we cancel", "before i cancel",
)

#: Everything that describes the REQUEST rather than the RIDER. Cleared on a
#: cancellation so the next message starts a genuinely new one.
#:
#: `visitor_type` and `home_city` are deliberately absent: which assistant they are
#: talking to and where they live are facts about the rider, true before the request
#: and true after it. `origin` IS cleared — for a rental it is the pickup address for
#: this hire, not a property of the person.
CLEARED_ON_CANCEL: tuple[str, ...] = (
    "destination", "origin", "duration_days", "travel_dates", "pickup_time",
    "rental_hours", "cab_category", "group_size", "group_type", "budget_range",
    "return_journey_preference", "interests", "special_requirements",
)


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _padded(text: str) -> str:
    """Tokenised, lower-cased and space-padded, for whole-phrase matching."""
    return f" {' '.join(_tokens(text))} "


def _looks_like_a_question(text: str) -> bool:
    low = text.lower()
    return any(hint in low for hint in _QUESTION_HINTS)


def _contains(text: str, phrases: tuple[str, ...]) -> bool:
    padded = _padded(text)
    return any(f" {' '.join(_tokens(phrase))} " in padded for phrase in phrases)


def looks_like_cancellation(text: str) -> bool:
    """True when this message calls the request off.

    Deliberately conservative, because a false positive throws away a request the rider
    still wants:

      • the policy questions are excluded outright ("what's your cancellation policy?",
        "is there a cancellation fee?", "cancel karne par kya lagta hai?")
      • "stop asking me questions" means get on with it, not give up
      • phrases match whole WORDS, so "cancellation" is never "cancel"

    The question mark is weighed rather than obeyed. A phrase that names what to cancel
    is an instruction however politely it is punctuated — "can you cancel this?" is not
    a request for information — while a bare "cancel" inside a question ("can I cancel
    later?") genuinely is one, so only the multi-word phrases survive a question.

    A phrasing this misses is not lost: the router's classifier reads the message too,
    and either detector is enough (`router_node`).
    """
    if not text or not text.strip():
        return False
    if _contains(text, _NOT_A_CANCELLATION):
        return False
    if _contains(text, _EXPLICIT_CANCEL_PHRASES):
        return True
    return _contains(text, _BARE_CANCEL_WORDS) and not _looks_like_a_question(text)


def mentions_an_existing_booking(text: str) -> bool:
    """True when the rider is talking about a trip that is ALREADY booked."""
    return _contains(text, _ALREADY_BOOKED_MARKERS)


def has_request_details(entities: ExtractedEntities) -> bool:
    """True when this chat had a request in progress worth clearing."""
    return any(
        getattr(entities, field, None) not in (None, "", [], {})
        for field in CLEARED_ON_CANCEL
    )


def cancel_scope(text: str, entities: ExtractedEntities) -> str:
    """What the rider wants called off — see `CancelScope`.

    The only evidence available for "is there a real booking?" is what they said: this
    service never books anything, so it cannot look one up and must not assume. So each
    reading is claimed only when it is the ONLY one available:

      • they name a booking and this chat has nothing in progress → it can only be a
        real one, so send them to the screen that owns it
      • they name no booking and this chat HAS something in progress → it can only be
        that
      • anything else — including "cancel my booking" typed halfway through building a
        request, which is genuinely both — resolves to UNCLEAR, whose reply is true
        under either reading rather than confident and wrong half the time
    """
    names_a_booking = mentions_an_existing_booking(text)
    in_progress = has_request_details(entities)
    if names_a_booking and not in_progress:
        return CancelScope.BOOKED_TRIP
    if in_progress and not names_a_booking:
        return CancelScope.REQUEST
    return CancelScope.UNCLEAR


def clear_request(
    entities: ExtractedEntities,
) -> tuple[ExtractedEntities, list[str]]:
    """Drop the request, keep the rider. Returns fresh entities and what was dropped.

    The conversation-control fields go too: a cancelled request must not carry its
    question counters into the next one, or a rider who cancels after being asked twice
    for a pickup begins the new request with that question already exhausted.
    """
    cleared = [
        field for field in CLEARED_ON_CANCEL
        if getattr(entities, field, None) not in (None, "", [], {})
    ]
    fresh = entities.model_copy(deep=True)
    for field in CLEARED_ON_CANCEL:
        setattr(fresh, field, [] if isinstance(getattr(fresh, field), list) else None)
    fresh.last_asked_field = None
    fresh.asked_counts = {}
    fresh.proceed_requested = False
    fresh.no_progress_turns = 0
    return fresh, cleared


# ── What the rider is told ────────────────────────────────────────────────────
# Written here rather than generated. Three facts, each of which the writer LLM was
# getting wrong: what has been dropped, what was never booked, and what only the app
# can cancel. `graph.cancel_node` sends these verbatim.

_CANCELLED_REQUEST = {
    "en": (
        "Done — I've cancelled that request and cleared the details. Nothing was "
        "booked and nothing has been charged.\n\n"
        "Just tell me when you'd like to start again."
    ),
    "hi": (
        "ठीक है — मैंने वह रिक्वेस्ट रद्द कर दी है और सारी जानकारी हटा दी है। "
        "न कोई बुकिंग हुई थी, न कोई पैसा लिया गया है।\n\n"
        "जब भी दोबारा शुरू करना हो, बस बता दीजिए।"
    ),
}

_CANCELLED_BOOKED_TRIP = {
    "en": (
        "Nothing from this chat is booked, so there's nothing here for me to cancel.\n\n"
        "If you've already booked a cab, cancel it from the **Trips** screen in the "
        "Him Cab app — that's the only place a confirmed booking can be cancelled, and "
        "it's what lets your driver know."
    ),
    "hi": (
        "इस चैट से कुछ बुक नहीं हुआ है, इसलिए यहाँ रद्द करने को कुछ नहीं है।\n\n"
        "अगर आपने पहले से कैब बुक कर रखी है, तो उसे ऐप की **Trips** स्क्रीन से रद्द करें — "
        "कन्फर्म बुकिंग वहीं से रद्द होती है और ड्राइवर को भी वहीं से पता चलता है।"
    ),
}

_CANCELLED_UNCLEAR = {
    "en": (
        "Done — I've cleared what we were putting together. Nothing from this chat was "
        "booked or charged.\n\n"
        "If you've already booked a cab in the app, cancel that one from the **Trips** "
        "screen — I can't cancel a confirmed booking from here."
    ),
    "hi": (
        "ठीक है — जो हम बना रहे थे, मैंने वह हटा दिया है। इस चैट से न कुछ बुक हुआ था, "
        "न कोई पैसा लिया गया।\n\n"
        "अगर ऐप में पहले से कैब बुक है, तो उसे **Trips** स्क्रीन से रद्द करें — "
        "कन्फर्म बुकिंग मैं यहाँ से रद्द नहीं कर सकता।"
    ),
}

_MESSAGES = {
    CancelScope.REQUEST: _CANCELLED_REQUEST,
    CancelScope.BOOKED_TRIP: _CANCELLED_BOOKED_TRIP,
    CancelScope.UNCLEAR: _CANCELLED_UNCLEAR,
}


def cancellation_message(scope: str, lang_key: str) -> str:
    """What to say when the rider calls it off. Never generated, never improvised."""
    table = _MESSAGES.get(scope, _CANCELLED_UNCLEAR)
    return table.get(lang_key, table[ReplyLanguage.ENGLISH])


def cancellation_event(scope: str, cleared: list[str]) -> dict:
    """The card the app renders, and the instruction it acts on.

    `dismiss_booking_screen` is the important half: a `booking_handoff` earlier in the
    conversation may have opened a pre-filled booking sheet, and leaving it up after
    the rider has called it off is how they end up booking the thing they cancelled.
    """
    return {
        "type": "request_cancelled",
        "payload": {
            "scope": scope,
            "dismiss_booking_screen": True,
            "cleared_fields": cleared,
            # Only a confirmed booking needs somewhere else to go; a dropped request is
            # finished here.
            "open_screen": "trips" if scope != CancelScope.REQUEST else None,
        },
    }


__all__ = [
    "CLEARED_ON_CANCEL",
    "cancel_scope",
    "cancellation_event",
    "cancellation_message",
    "clear_request",
    "has_request_details",
    "looks_like_cancellation",
    "mentions_an_existing_booking",
]
