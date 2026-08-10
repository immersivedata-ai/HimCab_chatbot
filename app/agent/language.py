"""
app/agent/language.py
──────────────────────
Single source of truth for which language the agent replies in.

Both the Planner (clarification questions) and the Writer (itinerary) use this so they
never disagree. The decision is made HERE from the user's actual message (script +
content) and the visitor persona — deliberately NOT from the router's per-turn
`state.language` LLM guess, which is unreliable for short/ambiguous inputs (a bare place
name like "manali" was being misclassified as Hindi and flipping the whole conversation).

Rule (anchored on the persona default, only deviating on a STRONG signal):

  Strong signals (apply to BOTH personas, checked first):
    - explicit "reply in English" / "hindi mein bolo" request  → that language
    - the message contains Devanagari                          → Hindi
    - clear Hinglish (>= 2 Hindi function words)               → Hindi
    - a real English sentence (>= 4 English words, no Hindi)   → English

  Otherwise (ambiguous: a single word, place name, greeting) → the PERSONA DEFAULT:
    - LOCAL visitor (Himachal resident)  → Hindi
    - OUTSTATION / unknown persona       → English

`state.visitor_type` is seeded from which front-end page the user is on (see the
`persona` param in app/api/v1/routes/chat.py) and then confirmed from a real origin city
by the planner (app/agent/nodes/planner.py).
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

# Common romanized-Hindi function words. Two or more of these in one message is a strong
# "this person is writing Hindi/Hinglish" signal. Kept to unambiguous grammar words so a
# lone place name (delhi/manali/shimla) never trips it.
_HINDI_WORDS = {
    "hai", "hain", "ho", "hoon", "hun", "tha", "thi", "the", "hoga", "hogi", "honge",
    "ka", "ki", "ke", "ko", "se", "mein", "me", "par", "pe", "aur", "ya", "nahi", "nahin",
    "na", "haan", "kya", "kyu", "kyun", "kaise", "kaisa", "kaha", "kahan", "kab", "kaun",
    "kitna", "kitne", "kitni", "din", "raat", "subah", "sham", "mujhe", "mujhko", "mera",
    "meri", "mere", "hum", "hamein", "humein", "aap", "aapko", "aapka", "tum", "tumhe",
    "chahiye", "chahta", "chahti", "jana", "jaana", "jaunga", "jaungi", "jaoge", "chalna",
    "chalo", "chalein", "chalte", "karna", "karo", "karenge", "raha", "rahi", "rahe",
    "wala", "wali", "wale", "bata", "batao", "bataye", "lekin", "magar", "phir", "abhi",
    "kal", "aaj", "parso", "acha", "accha", "achha", "theek", "thik", "yaar", "bhai",
}
_MIN_HINDI_WORDS = 2
_MIN_ENGLISH_WORDS = 4


def _latest_user_text(state: "AgentState") -> str:
    for m in reversed(state.messages):
        if isinstance(m, HumanMessage) and isinstance(m.content, str):
            return m.content
    return ""


def _recent_user_text(state: "AgentState", n: int = 5) -> str:
    return " ".join(
        m.content for m in state.messages[-n:]
        if isinstance(m, HumanMessage) and isinstance(m.content, str)
    ).lower()


def detect_reply_language(state: "AgentState") -> tuple[str, str]:
    """Return (lang_key, human_name) for the one language the whole reply should use.

    lang_key is 'en' | 'hi'; human_name is a phrase to drop into a prompt.
    """
    recent = _recent_user_text(state)
    if any(p in recent for p in _ENGLISH_REQUEST_PHRASES):
        return "en", _LANGUAGE_NAMES["en"]
    if any(p in recent for p in _HINDI_REQUEST_PHRASES):
        return "hi", _LANGUAGE_NAMES["hi"]

    text = _latest_user_text(state)
    if _DEVANAGARI_RE.search(text):
        return "hi", _LANGUAGE_NAMES["hi"]

    tokens = _WORD_RE.findall(text.lower())
    hindi_hits = sum(1 for t in tokens if t in _HINDI_WORDS)
    english_hits = sum(1 for t in tokens if t not in _HINDI_WORDS)

    if hindi_hits >= _MIN_HINDI_WORDS:            # clear Hinglish
        return "hi", _LANGUAGE_NAMES["hi"]
    if english_hits >= _MIN_ENGLISH_WORDS and hindi_hits == 0:  # clear English sentence
        return "en", _LANGUAGE_NAMES["en"]

    # Ambiguous (single word / place name / greeting) → persona default.
    if state.visitor_type == "local":
        return "hi", _LANGUAGE_NAMES["hi"]
    return "en", _LANGUAGE_NAMES["en"]
