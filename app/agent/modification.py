"""
app/agent/modification.py
──────────────────────────
Changing your mind mid-conversation — detected in code, confirmed before it is acted on.

WHY THIS EXISTS
───────────────
A rider gave a pickup and a drop, then changed one, and got this back:

    Mujhe Palampur से Lekar Kasol तक आपकी सवारी
    • सटीक किराया बुकिंग स्क्रीन पर पुष्टि होगा।
    आपकी बुकिंग स्क्रीन पिकअप और ड्रॉप के साथ तैयार है।

Two separate failures in four lines. The sentence parsed into places that do not exist
("Mujhe Palampur", "Lekar Kasol") — fixed in `slots`. And the system took a change it
had only half understood and went straight to a booking screen with it. Nobody was
asked whether that was even the right change.

`slots` can now tell that a detail CHANGED, for free, with no model call
(`read_trip_details` compared against what the session holds — see the "Changing your
mind" section there). This module decides what to DO about it, and the answer is: not
apply it silently.

A change is different in kind from an answer. An answer fills a blank; there was
nothing there to lose. A change overwrites something the rider already told us and is
already looking at on a booking screen. When the parse is right, a silent overwrite is
merely abrupt. When the parse is WRONG — and the screenshot is what wrong looks like —
a silent overwrite sends a car to a town that does not exist. Confirming costs one tap
and makes a misparse visible to the only person who can correct it, which is why the
confirmation is worth a turn even though it reads like politeness.

THE THREE SHAPES, AND WHAT EACH GETS
────────────────────────────────────
  1. A change we can name — "ab kasol jana hai" when the destination is Manali. We
     know the field and the new value, so we ask about THAT: "Change the destination
     from Manali to Kasol?" Confirm, then apply. `confirm_question`.

  2. A change we cannot name — "mujhe kuch badalna hai", "wait", "that's wrong". The
     rider wants to modify something and has not said what. Asking "yes or no?" here
     is useless, so we ask WHICH, and list what we hold so they can point at one.
     `which_detail_question`.

  3. The answer to either question. `reads_as_confirmation` for the first,
     and for the second the ordinary slot readers — a rider who has just been asked
     what to change and answers "destination Kasol" has already confirmed the intent,
     so that one applies directly rather than asking again.

Everything here is a table and a comparison. No model call, in any of the three, for
the same reason as `cancellation` and `ride_reply`: these lines decide what a rider's
booking screen says, the model is the component that was getting that wrong, and a
table cannot be talked out of it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Optional

from app.core.enums import ReplyLanguage, SlotName

if TYPE_CHECKING:
    from app.domain.conversation import ExtractedEntities

#: Splits on anything that is neither a word character nor Devanagari. The obvious
#: `[^\W\d_]+` does NOT work here: Python's `\w` excludes combining marks, so a Hindi
#: vowel sign is dropped and "ठीक है" tokenises to the fragments "ठ", "क", "ह" — every
#: Devanagari phrase in the tables below would be unreachable.
_SPLIT_RE = re.compile(r"[^\wऀ-ॿ]+", re.UNICODE)

#: Dropped before splitting, so "that's" is one token and matches "thats". All three
#: shapes, because a phone keyboard substitutes the curly one without being asked.
_APOSTROPHES = ("'", "’", "ʼ")  # noqa: RUF001 - the lookalikes are what this strips


# ── Recognising "I want to change something" ──────────────────────────────────
#
# Only for the case where NO value came with it. A message that names its new value is
# already handled by the readers in `slots` and never reaches these phrases.

#: Says a previous answer is wrong or should move, without saying what to. Matched as
#: padded substrings, so "change" never fires on "exchange".
_CHANGE_REQUEST_PHRASES = (
    "change", "modify", "edit", "update", "revise", "correct", "amend",
    "change something", "change it", "make a change", "changed my mind",
    "not right", "not correct", "thats wrong", "that's wrong", "is wrong",
    "mistake", "by mistake", "wrongly", "wrong entry", "redo", "start over",
    "badal", "badalna", "badlo", "badal do", "badalna hai", "badalni hai",
    "galat", "galti", "galat ho gaya", "sahi nahi", "theek nahi", "thik nahi",
    "बदल", "बदलना", "गलत", "गलती", "सही नहीं",
)

#: "Hold on" — a rider interrupting themselves. Means the same thing in practice, and
#: is how people actually phrase it before they have worked out what to say next.
_HESITATION_PHRASES = (
    "wait", "hold on", "hang on", "one sec", "one second", "one minute",
    "ek minute", "ek sec", "ruko", "ruk", "thoda ruko", "arre nahi", "arey nahi",
    "रुको", "एक मिनट",
)

#: Never a change request, however much the words overlap. Asking ABOUT changing
#: ("can I change it later?") is a question about policy, and answering it by opening
#: a modification flow is the same bug `cancellation` guards against.
_NOT_A_CHANGE_REQUEST = (
    "can i change", "can we change", "kya main badal", "badal sakta", "badal sakte",
    "badal sakti", "how to change", "how do i change", "change karne ke liye",
    "no change", "koi change nahi", "kuch nahi badalna", "don't change", "dont change",
    "no changes", "nothing to change",
)


def _tokens(text: str) -> list[str]:
    low = (text or "").lower()
    for mark in _APOSTROPHES:
        low = low.replace(mark, "")
    return [token for token in _SPLIT_RE.split(low) if token]


def _padded(text: str) -> str:
    return f" {' '.join(_tokens(text))} "


def _contains(padded: str, phrases: tuple[str, ...]) -> bool:
    return any(f" {phrase} " in padded for phrase in phrases)


def wants_to_change_something(text: str) -> bool:
    """True when the rider asks to modify a detail without saying which.

    Deliberately narrow. A message carrying its own value ("ab kasol jana hai") is read
    by `slots` and never needs this, so the only job here is the vaguer opening —
    "I want to change something", "wait, that's wrong" — where the useful reply is a
    question rather than a booking screen.
    """
    if not text or not text.strip():
        return False
    padded = _padded(text)
    if _contains(padded, _NOT_A_CHANGE_REQUEST):
        return False
    return _contains(padded, _CHANGE_REQUEST_PHRASES) or _contains(
        padded, _HESITATION_PHRASES
    )


# ── Reading the answer to "shall I change it?" ────────────────────────────────

#: Read as whole TOKENS, not as substrings of a sentence. A yes/no answer is a reply
#: that consists of nothing but these words — see `reads_as_confirmation` for why that
#: is the rule rather than "contains one of them".
_YES_TOKENS = frozenset({
    "yes", "yeah", "yep", "yup", "ya", "sure", "ok", "okay", "okey", "right",
    "correct", "confirm", "confirmed", "perfect", "great", "fine", "done",
    "please", "go", "ahead", "update", "change", "it", "that", "this", "the", "to",
    "haan", "han", "haa", "ha", "ji", "bilkul", "sahi", "thik", "theek", "hai", "ho",
    "kar", "karo", "kardo", "de", "do", "dijiye", "badal", "badlo",
    "हाँ", "हां", "ठीक", "है", "बिलकुल", "सही", "जी", "करो", "बदल", "दो",
})

#: A negation anywhere in a short reply governs it, whatever else is in there:
#: "mat karo" is made of a no word and a yes word and means no.
_NO_TOKENS = frozenset({
    "no", "nope", "nah", "not", "dont", "never", "cancel", "leave", "keep",
    "rehne", "rehnedo", "nahi", "nahin", "nhi", "na", "mat", "galat",
    "नहीं", "नही", "मत", "रहने", "गलत",
})


def reads_as_confirmation(text: str) -> Optional[bool]:
    """True for yes, False for no, None when the reply is neither.

    None is the common and important case, and getting it right matters more than
    either of the other two: a rider who answers a confirmation with a DIFFERENT change
    ("nahi, Kullu jaana hai") is not saying no. Read as a no, their new destination is
    thrown away and they are asked what they meant — which is the same "it ignored me"
    the whole flow exists to fix.

    So the test is not "does this contain a yes word". It is "is this reply made of
    NOTHING BUT yes and no words". A single token outside that vocabulary means the
    message is carrying content, and content goes to the slot readers instead.
    """
    tokens = _tokens(text)
    if not tokens:
        return None
    if any(token not in _YES_TOKENS and token not in _NO_TOKENS for token in tokens):
        return None
    if any(token in _NO_TOKENS for token in tokens):
        return False
    return True if any(token in _YES_TOKENS for token in tokens) else None


# ── The wording ───────────────────────────────────────────────────────────────

#: Every string this module can say, in both languages. One table, so a wording change
#: lands in both or fails the test that says they carry the same keys.
_TEXT: dict[str, dict[str, str]] = {
    "confirm_one": {
        "en": "Just to confirm — you want to change the {label} from {old} to {new}?",
        "hi": "पुष्टि कर लें — आप {label} {old} से बदलकर {new} करना चाहते हैं?",
    },
    "confirm_many": {
        "en": "Just to confirm — you want to change these?",
        "hi": "पुष्टि कर लें — आप ये बदलना चाहते हैं?",
    },
    "confirm_line": {
        "en": "{label}: {old} → {new}",
        "hi": "{label}: {old} → {new}",
    },
    "confirm_tail": {
        "en": "Reply yes to update, or tell me the right one.",
        "hi": "हाँ कहें तो बदल दूँ, या सही जानकारी बता दें।",
    },
    "which_detail": {
        "en": "Sure — what would you like to change?",
        "hi": "ज़रूर — आप क्या बदलना चाहते हैं?",
    },
    "which_line": {
        "en": "{label}: {value}",
        "hi": "{label}: {value}",
    },
    "which_tail": {
        "en": "Tell me which one and what it should be.",
        "hi": "बताइए कौन सा, और क्या होना चाहिए।",
    },
    "which_nothing": {
        "en": "Sure — what would you like to change? Nothing is set yet, so tell me what you need.",
        "hi": "ज़रूर — आप क्या बदलना चाहते हैं? अभी कुछ तय नहीं है, बताइए क्या चाहिए।",
    },
    "applied_one": {
        "en": "Done — {label} is now {new}.",
        "hi": "हो गया — {label} अब {new} है।",
    },
    "applied_many": {
        "en": "Done — updated:",
        "hi": "हो गया — बदल दिया:",
    },
    "applied_line": {
        "en": "{label}: {new}",
        "hi": "{label}: {new}",
    },
    "kept": {
        "en": "No problem, I have left that as it was.",
        "hi": "कोई बात नहीं, मैंने उसे वैसा ही रहने दिया।",
    },
}

#: What each field is called, for a rider. Kept here rather than imported from `slots`
#: because these read as a NOUN in a sentence about changing it ("the destination"),
#: where the slot labels read as the subject of a question.
_LABELS: dict[str, dict[str, str]] = {
    SlotName.DESTINATION: {"en": "destination", "hi": "गंतव्य"},
    SlotName.ORIGIN: {"en": "pickup", "hi": "पिकअप"},
    SlotName.TRAVEL_DATES: {"en": "travel date", "hi": "यात्रा की तारीख"},
    SlotName.PICKUP_TIME: {"en": "pickup time", "hi": "पिकअप का समय"},
    SlotName.DURATION_DAYS: {"en": "trip length", "hi": "यात्रा के दिन"},
    SlotName.GROUP_SIZE: {"en": "number of travellers", "hi": "यात्रियों की संख्या"},
    SlotName.RENTAL_HOURS: {"en": "rental duration", "hi": "किराये के घंटे"},
    SlotName.CAB_CATEGORY: {"en": "cab choice", "hi": "गाड़ी का चुनाव"},
    SlotName.RETURN_JOURNEY_PREFERENCE: {"en": "return journey", "hi": "वापसी की व्यवस्था"},
}

#: The order details are listed in when the rider is asked WHICH to change. The order
#: a rider thinks about the trip, not the order the fields happen to be declared in.
_LISTING_ORDER: tuple[str, ...] = (
    SlotName.ORIGIN,
    SlotName.DESTINATION,
    SlotName.TRAVEL_DATES,
    SlotName.PICKUP_TIME,
    SlotName.RENTAL_HOURS,
    SlotName.DURATION_DAYS,
    SlotName.GROUP_SIZE,
    SlotName.CAB_CATEGORY,
)


def _t(key: str, lang_key: str, **values: Any) -> str:
    table = _TEXT[key]
    return table.get(lang_key, table[ReplyLanguage.ENGLISH]).format(**values)


def label_for(field: str, lang_key: str) -> str:
    """What to call `field` in a sentence, in the rider's language."""
    table = _LABELS.get(field)
    if not table:
        return field.replace("_", " ")
    return table.get(lang_key, table[ReplyLanguage.ENGLISH])


def _shown(value: Any) -> str:
    """A slot value as a rider reads it back."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def confirm_question(
    changes: dict[str, Any], entities: ExtractedEntities, lang_key: str
) -> str:
    """Ask whether a change we HAVE understood is the change they meant.

    Names both sides — what it is now and what it would become — because the value we
    read is the thing most likely to be wrong. A rider shown "destination: Manali →
    Lekar Kasol" can see the parse failed; a rider shown a booking screen cannot.
    """
    ordered = [f for f in _LISTING_ORDER if f in changes]
    ordered += [f for f in changes if f not in ordered]

    if len(ordered) == 1:
        field = ordered[0]
        return "\n".join(
            [
                _t(
                    "confirm_one",
                    lang_key,
                    label=label_for(field, lang_key),
                    old=_shown(getattr(entities, field, None)),
                    new=_shown(changes[field]),
                ),
                _t("confirm_tail", lang_key),
            ]
        )

    lines = [_t("confirm_many", lang_key)]
    for field in ordered:
        lines.append(
            "- "
            + _t(
                "confirm_line",
                lang_key,
                label=label_for(field, lang_key),
                old=_shown(getattr(entities, field, None)),
                new=_shown(changes[field]),
            )
        )
    lines.append(_t("confirm_tail", lang_key))
    return "\n".join(lines)


def which_detail_question(entities: ExtractedEntities, lang_key: str) -> str:
    """Ask WHICH detail to change, listing what we currently hold.

    The list is the point. "What would you like to change?" on its own makes the rider
    remember what they told us and in what words; showing it back turns the same
    question into something they can answer by pointing.
    """
    lines = []
    for field in _LISTING_ORDER:
        value = getattr(entities, field, None)
        if value in (None, "", [], {}):
            continue
        lines.append(
            "- "
            + _t(
                "which_line",
                lang_key,
                label=label_for(field, lang_key),
                value=_shown(value),
            )
        )

    if not lines:
        return _t("which_nothing", lang_key)
    return "\n".join([_t("which_detail", lang_key), *lines, _t("which_tail", lang_key)])


def applied_message(
    changes: dict[str, Any], lang_key: str
) -> str:
    """Say what was just changed. Prepended to the turn's real reply.

    A change the rider confirmed still has to be stated: the reply that follows is a
    fresh quote or plan, and without this line there is nothing in it that says the
    update was the reason it moved.
    """
    ordered = [f for f in _LISTING_ORDER if f in changes]
    ordered += [f for f in changes if f not in ordered]

    if len(ordered) == 1:
        field = ordered[0]
        return _t(
            "applied_one",
            lang_key,
            label=label_for(field, lang_key),
            new=_shown(changes[field]),
        )

    lines = [_t("applied_many", lang_key)]
    for field in ordered:
        lines.append(
            "- "
            + _t(
                "applied_line",
                lang_key,
                label=label_for(field, lang_key),
                new=_shown(changes[field]),
            )
        )
    return "\n".join(lines)


def kept_message(lang_key: str) -> str:
    """Acknowledge a change the rider declined, so the turn is not silent about it."""
    return _t("kept", lang_key)


# ── The structured events ─────────────────────────────────────────────────────

def confirmation_event(
    changes: dict[str, Any], entities: ExtractedEntities, lang_key: str
) -> dict:
    """The card the app draws for a pending change: each field, both values, two taps.

    Sent so the rider can confirm without typing. The text carries the same question,
    so a client that renders no cards is not left with a dead end.
    """
    return {
        "type": "change_confirmation",
        "payload": {
            "changes": [
                {
                    "field": field,
                    "label": label_for(field, lang_key),
                    "current": _shown(getattr(entities, field, None)),
                    "proposed": _shown(value),
                }
                for field, value in changes.items()
            ],
        },
    }


def which_detail_event(entities: ExtractedEntities, lang_key: str) -> dict:
    """The card for "what would you like to change?" — the details, as tappable rows."""
    return {
        "type": "change_options",
        "payload": {
            "fields": [
                {
                    "field": field,
                    "label": label_for(field, lang_key),
                    "current": _shown(getattr(entities, field, None)),
                }
                for field in _LISTING_ORDER
                if getattr(entities, field, None) not in (None, "", [], {})
            ],
        },
    }


__all__ = [
    "applied_message",
    "confirm_question",
    "confirmation_event",
    "kept_message",
    "label_for",
    "reads_as_confirmation",
    "wants_to_change_something",
    "which_detail_event",
    "which_detail_question",
]
