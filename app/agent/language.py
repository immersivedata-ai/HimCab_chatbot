"""
app/agent/language.py
──────────────────────
Single source of truth for which language the agent replies in.

Both the Planner (clarification questions) and the Writer (itinerary) use this so they
never disagree. The decision is made HERE from what the user actually wrote and the
visitor persona — deliberately NOT from the router's per-turn `state.language` LLM
guess, which is unreliable for short inputs (a bare place name like "manali" was being
misclassified as Hindi and flipping the whole conversation).

WHY THIS IS NOT A WORD COUNT ANY MORE
─────────────────────────────────────
It used to be: "two or more romanized-Hindi function words → reply in Hindi". The word
list included `the`, `me`, `par`, `na`, `ho`, `se` and `ki` — every one of them also an
ordinary English word, and `the` the most common word in the language. So:

    "Can you plan the trip for me"   → the + me   → 2 hits → Hindi
    "Just the two of us, me and my wife" → the + me → 2 hits → Hindi

An English speaker got an English welcome, typed two or three ordinary sentences, and
the assistant switched to Hindi and stayed there. Three things now prevent that:

  1. the Hindi markers are words that are ONLY Hindi — no English collisions
  2. English markers are counted too, so a mixed message goes to whichever side has
     more evidence rather than to Hindi on a technicality
  3. the evidence is read across the recent conversation, not just the last message,
     so one stray token cannot flip a conversation that has been running in English
  4. only messages long enough to be PROSE are counted. The short answers the
     assistant's own questions solicit — "4 people", "2 of us", "tomorrow at 5" — are
     mostly English function words whoever is typing them, and counting them flipped
     Hindi conversations to English one answer after the assistant asked in Hindi

Rule, in order:

    explicit request ("reply in English", "hindi mein bolo")  → that language
    any Devanagari in a recent message                        → Hindi
    Hindi markers, and at least as many as English markers    → Hindi
    English markers and no Hindi markers                      → English
    otherwise (a place name, a number, a short answer)        → the PERSONA DEFAULT
                                                                 local → Hindi
                                                                 outstation → English

`state.visitor_type` starts from the rider's GPS fix or the front end they opened, and
is confirmed from their own words by the planner (app/agent/nodes/planner.py).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage

if TYPE_CHECKING:
    from app.agent.state import AgentState

_LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi (Devanagari script)",
}

_ENGLISH_REQUEST_PHRASES = (
    "speak in english", "in english", "reply in english", "answer in english",
    "respond in english", "english mein bolo", "english me bolo", "english mein baat",
    "english me baat", "english mein bat", "english me bat",
)
_HINDI_REQUEST_PHRASES = (
    "hindi mein bolo", "hindi me bolo", "hindi mein baat", "hindi me baat",
    "reply in hindi", "answer in hindi", "respond in hindi", "in hindi", "speak in hindi",
    "हिंदी में", "हिंदी में बोलो", "हिंदी में बात",
)

# Devanagari Unicode block (U+0900–U+097F).
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_WORD_RE = re.compile(r"[a-z']+")

# Romanized Hindi that is NOT also an English word. Every entry here is evidence on its
# own, which is only true because the collisions were removed: `the` (था/थे), `me` (में),
# `par` (पर), `na` (ना), `ho` (हो), `se` (से) and `ki` (की) all read as ordinary English
# and were what turned English sentences into Hindi replies.
#
# Losing them costs nothing in practice: Hinglish almost never leans on those alone —
# "mujhe manali jana hai" and "kal 4 log ja rahe hain" are still unmistakable.
_HINDI_MARKERS = {
    "hai", "hain", "hoon", "hun", "tha", "thi", "hoga", "hogi", "honge",
    "ka", "ke", "ko", "mein", "aur", "ya", "nahi", "nahin", "nhi", "haan",
    "kya", "kyu", "kyun", "kaise", "kaisa", "kaha", "kahan", "kab", "kaun",
    "kitna", "kitne", "kitni", "din", "raat", "subah", "sham",
    "mujhe", "mujhko", "mera", "meri", "mere", "hum", "hamein", "humein",
    "aap", "aapko", "aapka", "tum", "tumhe", "tumhara",
    "chahiye", "chahta", "chahti", "jana", "jaana", "jaunga", "jaungi", "jaoge",
    "chalna", "chalo", "chalein", "chalte", "karna", "karo", "karenge", "kar",
    "raha", "rahi", "rahe", "wala", "wali", "wale", "bata", "batao", "bataye",
    "lekin", "magar", "phir", "abhi", "kal", "aaj", "parso",
    "acha", "accha", "achha", "theek", "thik", "yaar", "bhai", "ji",
    "log", "saath", "liye", "bhi", "koi", "kuch", "sab", "bana", "banao",
}

# Ordinary English function words. Counted so a message with markers on both sides goes
# to whichever language has more of them, instead of to Hindi on a single coincidence.
_ENGLISH_MARKERS = {
    "the", "a", "an", "is", "are", "am", "was", "were", "be", "been", "will", "would",
    "can", "could", "should", "shall", "do", "does", "did", "have", "has", "had",
    "i", "we", "you", "my", "our", "your", "us", "me", "it", "they", "them",
    "and", "or", "but", "for", "with", "from", "to", "of", "in", "on", "at", "by",
    "want", "need", "like", "please", "plan", "trip", "days", "day", "people",
    "going", "go", "visit", "travel", "there", "here", "about", "what", "when",
    "where", "which", "how", "many", "much", "some", "any", "this", "that",
}

#: Two markers is enough to be sure — one can be a name, a typo or a coincidence.
#: Counted RAW, not weighted. Comparing the weighted total against this is what let a
#: single word decide: the newest message carries the highest weight, so one "of" in
#: the latest reply scored 4 and cleared a bar meant to need two separate words.
_MIN_MARKERS = 2

#: Below this many words a message is an ANSWER, not a language sample, and it casts no
#: vote — the persona decides instead.
#:
#: This is the fix for a local conversation flipping to English mid-way. The assistant
#: asks its questions in Hindi and the rider answers them the way anyone answers a
#: form: "4 people", "2 of us", "tomorrow at 5". Every one of those is mostly English
#: FUNCTION words ("of", "us", "at", "people"), so the reply that a Hindi question
#: solicited was itself the evidence used to switch away from Hindi — and the rider saw
#: the assistant change language for no reason they could see.
#:
#: A bare answer already followed the persona when it contained no markers at all
#: ("Manali", "4", "kasol"); this only extends the same rule to the short answers that
#: happen to contain one. Real prose — "mujhe manali jana hai", "Can you plan the trip
#: for me" — clears it comfortably, which is what keeps a single sentence able to set
#: the language on the first turn.
_MIN_SAMPLE_WORDS = 4

#: How many recent user messages the decision reads. A conversation that has been
#: running in one language should not turn over on a single short reply.
_RECENT_TURNS = 4


def _user_messages(state: AgentState, limit: int) -> list[str]:
    """The last `limit` user messages, newest last."""
    texts = [
        m.content for m in state.messages
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ]
    return texts[-limit:]


def _count_markers(text: str) -> tuple[int, int]:
    """(hindi, english) marker counts for one message."""
    tokens = _WORD_RE.findall(text.lower())
    hindi = sum(1 for t in tokens if t in _HINDI_MARKERS)
    english = sum(1 for t in tokens if t in _ENGLISH_MARKERS)
    return hindi, english


def detect_reply_language(state: AgentState) -> tuple[str, str]:
    """Return (lang_key, human_name) for the one language the whole reply should use.

    lang_key is 'en' | 'hi'; human_name is a phrase to drop into a prompt.
    """
    recent = _user_messages(state, _RECENT_TURNS)
    joined = " ".join(recent).lower()

    # An explicit instruction outranks everything, including the script they typed it in.
    if any(p in joined for p in _ENGLISH_REQUEST_PHRASES):
        return "en", _LANGUAGE_NAMES["en"]
    if any(p in joined for p in _HINDI_REQUEST_PHRASES):
        return "hi", _LANGUAGE_NAMES["hi"]

    # Devanagari is unambiguous — nobody types it by accident.
    if _DEVANAGARI_RE.search(joined):
        return "hi", _LANGUAGE_NAMES["hi"]

    # Only messages long enough to BE a language sample vote. See _MIN_SAMPLE_WORDS.
    samples = [t for t in recent if len(_WORD_RE.findall(t.lower())) >= _MIN_SAMPLE_WORDS]

    # Weighted so the newest message leads but cannot single-handedly overturn the
    # conversation: a rider who has written three English sentences and then types
    # "kal" is asking about tomorrow, not switching language.
    #
    # The raw totals are what the thresholds are checked against, and the weighted ones
    # only decide which side wins once both have real evidence. Mixing the two is what
    # let one word in the newest message cross a bar meant to need two.
    hindi = english = 0
    raw_hindi = raw_english = 0
    for weight, text in enumerate(samples, start=1):
        h, e = _count_markers(text)
        raw_hindi += h
        raw_english += e
        hindi += h * weight
        english += e * weight

    if raw_hindi >= _MIN_MARKERS and hindi >= english:
        return "hi", _LANGUAGE_NAMES["hi"]
    if raw_english >= _MIN_MARKERS and raw_hindi == 0:
        return "en", _LANGUAGE_NAMES["en"]

    # Ambiguous — a place name, a bare number, a greeting. Follow the persona.
    if state.visitor_type == "local":
        return "hi", _LANGUAGE_NAMES["hi"]
    return "en", _LANGUAGE_NAMES["en"]
