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

Rule, in order:

    explicit request ("reply in English", "hindi mein bolo")  → that language
    any Devanagari in a recent message                        → Hindi
    Hindi markers, and at least as many as English markers    → Hindi
    English markers and no Hindi markers                      → English
    otherwise (a place name, a number, a greeting)            → the PERSONA DEFAULT
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
    "log", "saath", "liye", "bhi", "koi", "kuch", "sab", "bana", "banao", "banao",
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
_MIN_MARKERS = 2

#: How many recent user messages the decision reads. A conversation that has been
#: running in one language should not turn over on a single short reply.
_RECENT_TURNS = 4


def _user_messages(state: "AgentState", limit: int) -> list[str]:
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


def detect_reply_language(state: "AgentState") -> tuple[str, str]:
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

    # Weighted so the newest message leads but cannot single-handedly overturn the
    # conversation: a rider who has written three English sentences and then types
    # "kal" is asking about tomorrow, not switching language.
    hindi = english = 0
    for weight, text in enumerate(recent, start=1):
        h, e = _count_markers(text)
        hindi += h * weight
        english += e * weight

    if hindi >= _MIN_MARKERS and hindi >= english:
        return "hi", _LANGUAGE_NAMES["hi"]
    if english >= _MIN_MARKERS and hindi == 0:
        return "en", _LANGUAGE_NAMES["en"]

    # Ambiguous — a place name, a bare number, a greeting. Follow the persona.
    if state.visitor_type == "local":
        return "hi", _LANGUAGE_NAMES["hi"]
    return "en", _LANGUAGE_NAMES["en"]
