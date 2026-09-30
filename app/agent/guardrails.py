"""
app/agent/guardrails.py
────────────────────────
The last thing a generated reply passes before a rider reads it.

WHY THIS EXISTS
───────────────
Everything in this codebase that can be made deterministic already has been. The ride
reply is assembled (`ride_reply.py`), a cancellation is read from a table
(`cancellation.py`), the fare, the forecast and the measured journey are stamped over
whatever the model wrote (`writer._apply_real_fare` and friends), and the coupons are
filtered to what the backend actually returned.

What was never checked is the PROSE. For an itinerary, a rental quote or a direct
answer, whatever the model wrote went to the rider untouched — so every rule the prompt
stated was enforced only by the model's willingness to follow it. Repeatedly, it was
not: a prompt that forbade sightseeing in writing still produced Mall Road and the
Ridge, and a prompt that forbade prices when no fare was fetched is exactly the kind of
rule a model breaks under pressure to be helpful.

So this module reads the finished text back and checks its CLAIMS against the facts the
turn actually has. It is deliberately narrow: it only flags things that can be decided
by comparison, never by taste.

  invented_coupon       a code the offers endpoint never returned
  invented_cab          a vehicle the fare endpoint never priced
  unsupported_fare      a rupee figure when no fare was fetched at all
  contradicted_fare     a rupee figure that is not one of the quoted ones
  contradicted_distance a distance that disagrees with the measured route
  wrong_language        Devanagari in a reply that should be English, or the reverse
  itinerary_in_a_quote  a day-by-day plan inside a transactional reply

WHAT IT DOES NOT DO
───────────────────
It does not judge whether a reply is "on topic" in general. A places answer legitimately
names a dozen things this module has never heard of, and a rule list that tried to
decide which of them belong would reject good replies far more often than bad ones.
Staying on topic is handled where it can be handled honestly: by giving each flow its
own prompt with its own facts (`writer._build_system_prompt`), and by assembling the
replies that are pure fact rather than asking for them. This is the net under that, for
the claims a machine can actually check.

REMEDIATION
───────────
A critical finding removes the LINE that carries it, and nothing else — the rest of the
reply is usually fine and throwing it away costs the rider more than the bad line did.
If removal would empty the reply, the reply is kept and the finding is logged loudly
instead: an empty message is worse than a flawed one, and at that point a human needs
to see it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from app.agent import slots
from app.core.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable

log = get_logger(__name__)

#: Findings at this level have their line removed. The rest are logged so a pattern can
#: be spotted before it becomes a complaint.
CRITICAL = "critical"
WARN = "warn"

#: A rupee figure, in the forms replies actually use.
_MONEY_RE = re.compile(
    r"(?:₹|\bRs\.?|\bINR)\s*([\d][\d,]*)|([\d][\d,]*)\s*(?:रुपये|rupees)",
    re.IGNORECASE,
)

#: A coupon-code-shaped token: upper case, at least five characters, and containing a
#: digit. The digit is what keeps ordinary words out — "SUV", "MANALI" and "HIMACHAL"
#: are not codes, and a reply is full of them.
_CODE_RE = re.compile(r"\b(?=[A-Z0-9]*\d)[A-Z][A-Z0-9]{4,19}\b")

_DISTANCE_RE = re.compile(
    r"([\d]+(?:\.\d+)?)\s*(?:km\b|kms\b|किमी|किलोमीटर|kilometers?|kilometres?)",
    re.IGNORECASE,
)

_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_LATIN_WORD_RE = re.compile(r"[A-Za-z]{3,}")

#: Cab and vehicle names a reply might state. Taken from the seat-capacity table, so
#: this is exactly the vocabulary the backend deals in — anything it can price, this can
#: recognise. Longest first, so "tempo traveller" is matched before "tempo".
_CAB_WORDS: tuple[str, ...] = ()


def _cab_vocabulary() -> tuple[str, ...]:
    """The cab names worth checking for, longest first.

    Built lazily from `SEATS_BY_CATEGORY` rather than copied, because a category added
    there is a category the backend has started returning — and a second hand-maintained
    list would quietly stop matching it.
    """
    global _CAB_WORDS
    if not _CAB_WORDS:
        from app.integrations.himcab.vehicles import SEATS_BY_CATEGORY

        # Single generic words are excluded: "bus", "auto" and "mini" appear in ordinary
        # sentences ("the bus stand", "automatically"), and a false positive here would
        # strip a line the rider needed.
        names = {n for n in SEATS_BY_CATEGORY if len(n) >= 5 and n not in {"micro", "prime"}}
        _CAB_WORDS = tuple(sorted(names, key=len, reverse=True))
    return _CAB_WORDS


#: Phrases that mean a day-by-day plan. Only consulted for the transactional flows,
#: where the reply is a quote and a plan has no business in it.
_ITINERARY_MARKERS = (
    "day 1", "day 2", "day-by-day", "दिन 1", "दिन 2", "पहला दिन", "दूसरा दिन",
    "itinerary", "यात्रा कार्यक्रम", "sightseeing", "दर्शनीय स्थल",
)


@dataclass(frozen=True)
class Finding:
    """One claim in the reply that the turn's facts do not support."""

    kind: str
    severity: str
    claim: str
    supported: str
    line: str

    def as_log(self) -> dict:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "claim": self.claim,
            "supported": self.supported,
        }


def _numbers(values: Iterable[Any]) -> set[int]:
    """Whole-rupee forms of everything we are allowed to quote."""
    out: set[int] = set()
    for value in values:
        try:
            out.add(round(float(value)))
        except (TypeError, ValueError):
            continue
    return out


def _money_in(text: str) -> list[int]:
    found: list[int] = []
    for prefixed, suffixed in _MONEY_RE.findall(text):
        raw = prefixed or suffixed
        try:
            found.append(int(raw.replace(",", "")))
        except ValueError:
            continue
    return found


def _supported_amounts(state) -> set[int]:
    """Every rupee figure this turn is entitled to say out loud."""
    fare = state.cab_fare or {}
    amounts: list[Any] = [fare.get("fare_inr"), fare.get("fare_after_offer_inr")]
    for option in fare.get("options") or []:
        amounts.append(option.get("total_fare_inr"))
    for offer in state.offers or []:
        amounts.append(offer.get("coupon_value_inr"))
        amounts.append(offer.get("fare_after_offer_inr"))
        amounts.append(offer.get("minimum_value_inr"))
    return _numbers(a for a in amounts if a is not None)


def _supported_codes(state) -> set[str]:
    return {
        str(o.get("coupon_code")).upper()
        for o in (state.offers or [])
        if o.get("coupon_code")
    }


def inspect_reply(state, text: str, mode: str, lang_key: str) -> list[Finding]:
    """Every claim in `text` the turn's facts do not support. Never mutates anything."""
    if not text or not text.strip():
        return []

    findings: list[Finding] = []
    lines = text.splitlines()
    supported_amounts = _supported_amounts(state)
    supported_codes = _supported_codes(state)
    has_fare = bool((state.cab_fare or {}).get("options"))

    # A quote is transactional: every figure in it is one the rider may act on, and a
    # day-by-day plan has no business in it. An itinerary is prose about a holiday, so
    # an approximate figure there is part of the product rather than a defect — the
    # same claim is graded differently depending on which reply it turned up in.
    is_quote = mode in (slots.RENTAL_MODE, slots.RIDE_MODE)
    quote_severity = CRITICAL if is_quote else WARN

    for line in lines:
        # ── Coupon codes ──────────────────────────────────────────────────────
        # The rider takes this to the payment screen. A code we invented is rejected
        # there, which is a worse experience than never mentioning a discount.
        for code in _CODE_RE.findall(line):
            if code.upper() not in supported_codes:
                findings.append(Finding(
                    kind="invented_coupon", severity=CRITICAL, claim=code,
                    supported=", ".join(sorted(supported_codes)) or "no offers returned",
                    line=line,
                ))

        # ── Cab categories ────────────────────────────────────────────────────
        # A vehicle is a claim exactly like a price: naming one implies we checked what
        # is available and what it seats, and the rider plans around it. The reported
        # failure was "for your family of 8, we recommend a spacious Innova or a Tempo
        # Traveller" in a reply where no fare had been fetched at all — so nothing had
        # been checked and neither vehicle came from anywhere.
        #
        # Only checked when the turn has no priced options. With a fare the list IS the
        # answer and the reply is supposed to name every one of them; the money check
        # above already covers quoting the wrong figure against them.
        if not has_fare:
            lowered_line = line.lower()
            named = next((c for c in _cab_vocabulary() if c in lowered_line), None)
            if named:
                findings.append(Finding(
                    kind="invented_cab", severity=CRITICAL, claim=named,
                    supported="no fare was fetched, so no cab was priced for this trip",
                    line=line,
                ))

        # ── Money ─────────────────────────────────────────────────────────────
        for amount in _money_in(line):
            if not has_fare:
                findings.append(Finding(
                    kind="unsupported_fare", severity=CRITICAL, claim=f"INR {amount:,}",
                    supported="no fare was fetched for this turn", line=line,
                ))
            elif amount not in supported_amounts:
                # Tolerated in an itinerary, where the model is asked for a cost
                # breakdown and an approximate hotel or meal figure is part of the
                # product. Never tolerated in a quote, where every number is the one
                # the rider will be charged.
                severity = quote_severity
                findings.append(Finding(
                    kind="contradicted_fare", severity=severity, claim=f"INR {amount:,}",
                    supported=", ".join(f"INR {a:,}" for a in sorted(supported_amounts)),
                    line=line,
                ))

        # ── Distance ──────────────────────────────────────────────────────────
        route = state.route_estimate or {}
        measured = route.get("distance_km")
        if measured is not None:
            for raw in _DISTANCE_RE.findall(line):
                try:
                    stated = float(raw)
                except ValueError:
                    continue
                # A tolerance, not an equality: the reply may round, and "about 215 km"
                # for a measured 214.6 is right rather than wrong.
                if abs(stated - float(measured)) > max(5.0, float(measured) * 0.1):
                    findings.append(Finding(
                        kind="contradicted_distance", severity=quote_severity,
                        claim=f"{stated:g} km",
                        supported=f"{round(float(measured))} km measured", line=line,
                    ))

    # ── A plan inside a quote ─────────────────────────────────────────────────
    # Per line, so the line can be removed. Scanning the whole text only ever produced
    # a log entry, and the sightseeing stayed in the reply.
    if is_quote:
        for line in lines:
            lowered = line.lower()
            marker = next((m for m in _ITINERARY_MARKERS if m in lowered), None)
            if marker:
                findings.append(Finding(
                    kind="itinerary_in_a_quote", severity=CRITICAL, claim=marker,
                    supported="this flow quotes a cab; it does not plan days",
                    line=line,
                ))

    # ── Language ──────────────────────────────────────────────────────────────
    # One language for the whole reply is a decision made in app/agent/language.py and
    # stated in every prompt. A reply that mixes them is the bug riders reported as the
    # assistant "changing language on its own".
    has_devanagari = bool(_DEVANAGARI_RE.search(text))
    if lang_key == "hi" and not has_devanagari:
        findings.append(Finding(
            kind="wrong_language", severity=WARN, claim="reply is not in Devanagari",
            supported="Hindi was chosen for this rider", line="",
        ))
    elif lang_key == "en" and has_devanagari:
        findings.append(Finding(
            kind="wrong_language", severity=WARN, claim="reply contains Devanagari",
            supported="English was chosen for this rider", line="",
        ))

    return findings


def apply_guardrails(
    state, text: str, mode: str, lang_key: str
) -> tuple[str, list[Finding]]:
    """The reply as it should ship, plus what was found in it.

    Critical findings take their LINE out and leave the rest. Most of a flawed reply is
    usually fine, and discarding all of it costs the rider more than the bad line did.

    If that would empty the reply, the original is kept: an empty message helps nobody,
    and a finding that severe is something a person needs to look at rather than
    something to paper over.
    """
    findings = inspect_reply(state, text, mode, lang_key)
    if not findings:
        return text, []

    bad_lines = {f.line for f in findings if f.severity == CRITICAL and f.line}
    cleaned = text
    if bad_lines:
        kept = [line for line in text.splitlines() if line not in bad_lines]
        candidate = "\n".join(kept).strip()
        if candidate:
            cleaned = candidate
        else:
            log.error(
                "guardrails_would_empty_the_reply",
                findings=[f.as_log() for f in findings],
                session_id=getattr(state, "session_id", None),
            )

    level = log.warning if bad_lines else log.info
    level(
        "guardrails_findings",
        mode=str(mode),
        removed_lines=len(bad_lines),
        findings=[f.as_log() for f in findings],
        session_id=getattr(state, "session_id", None),
    )
    return cleaned, findings


__all__ = ["CRITICAL", "WARN", "Finding", "apply_guardrails", "inspect_reply"]
