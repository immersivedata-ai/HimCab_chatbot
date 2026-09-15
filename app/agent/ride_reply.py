"""
app/agent/ride_reply.py
────────────────────────
The reply a Himachal resident gets when they book a cab. Assembled, not generated.

WHY THIS IS NOT WRITTEN BY THE MODEL
────────────────────────────────────
A ride reply has exactly one job: confirm the journey, show what the cabs cost, and
say the booking screen is open. Every line of it is a fact the system already holds —
the route came from the routing engine, the fares from the Him Cab backend, the
temperature from the weather provider. There is nothing to compose, only to state.

Asking a language model to state facts it has been handed turns out to be the hard
way. The prompt for this reply grew a FORBIDDEN list — no day-by-day plan, no
sightseeing, no hidden gems, no hotel or food advice, no comfort reassurance, no
trip length — and the replies kept arriving with all of it anyway:

    ...
    यात्रा के लिए कुछ सुझाव:
    • शिमला में दर्शनीय स्थलों की यात्रा के लिए यह एक बेहतरीन समय है।
    • आप मॉल रोड, रिज, जाखू मंदिर और कुफरी जैसे स्थानों पर जा सकते हैं।
    • अपने परिवार के साथ इस यात्रा का आनंद लें।
    क्या आप अपनी पूरी यात्रा की योजना बनाने में मेरी मदद चाहेंगे?

Not one of those places came from any tool. The rider had asked for a cab from Kasol
to Shimla and was handed a brochure and an offer to plan a holiday — and no price, and
no booking screen, which is the part they came for.

So this module does what `app/agent/cancellation.py` already does, for the same
reason: the lines are claims about somebody's money and somebody's driver, the model
was the component getting them wrong, and a deterministic table cannot be talked out
of them. A ride reply now cannot invent a landmark, cannot quote a fare nobody
returned, cannot offer to plan a trip, and cannot drift when another flow's prompt
changes.

WHAT THIS DELIBERATELY GIVES UP
───────────────────────────────
Natural phrasing. The reply reads like a booking summary because that is what it is.
The itinerary flow, where the writing is the product, still goes to the model.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.enums import ReplyLanguage

#: Every string the reply can contain, in both languages. One table, so a wording
#: change lands in both or fails the test that says they carry the same keys.
_TEXT: dict[str, dict[str, str]] = {
    "heading": {
        "en": "Your ride from {origin} to {destination}",
        "hi": "{origin} से {destination} तक आपकी सवारी",
    },
    "drive": {
        "en": "Drive time: about {duration} ({distance} km)",
        "hi": "ड्राइव समय: लगभग {duration} ({distance} किमी)",
    },
    "cab": {
        "en": "{name}{seats}: ₹{fare}",
        "hi": "{name}{seats}: ₹{fare}",
    },
    "seats": {
        "en": " ({count} seats)",
        "hi": " ({count} सीटें)",
    },
    "no_fare": {
        "en": "The exact fare is confirmed on the booking screen.",
        "hi": "सटीक किराया बुकिंग स्क्रीन पर पुष्टि होगा।",
    },
    "weather": {
        "en": "{place} right now: {temp}°C, {condition}",
        "hi": "{place} में अभी: {temp}°C, {condition}",
    },
    "offer": {
        "en": "Use {code} to save ₹{saving}",
        "hi": "{code} से ₹{saving} की बचत",
    },
    "handoff": {
        "en": "Your booking screen is open with the pickup and drop already filled in.",
        "hi": "आपकी बुकिंग स्क्रीन पिकअप और ड्रॉप के साथ तैयार है।",
    },
}


def _t(key: str, lang_key: str, **values: Any) -> str:
    """One line of the reply, in the rider's language."""
    table = _TEXT[key]
    return table.get(lang_key, table[ReplyLanguage.ENGLISH]).format(**values)


def _round_inr(value: Any) -> Optional[str]:
    """A rupee figure as the rider reads it, or None when there is not one.

    Never rounds to something prettier than the truth: the number here is the one the
    booking screen will charge, and a reply that disagrees with the receipt by ten
    rupees is a reply the rider stops trusting.
    """
    try:
        return f"{round(float(value)):,}"
    except (TypeError, ValueError):
        return None


def build_ride_reply(state, lang_key: str = ReplyLanguage.ENGLISH) -> str:
    """The whole reply, as lines. Every value comes from state; nothing is composed.

    Ordered the way a rider reads it: what the journey is, what it costs, what it is
    like there, and where to press. A section with no data behind it is simply absent —
    never announced as missing, which is the other way a reply wastes the rider's time.
    """
    entities = state.entities
    route = state.route_estimate or {}
    fare = state.cab_fare or {}

    origin = (
        (route.get("origin") or {}).get("name")
        or entities.origin
        or entities.home_city
        or ""
    )
    destination = (
        (route.get("destination") or {}).get("name") or entities.destination or ""
    )

    lines: list[str] = [
        _t("heading", lang_key, origin=origin, destination=destination)
    ]

    # ── The drive ─────────────────────────────────────────────────────────────
    if route.get("duration_text") and route.get("distance_km") is not None:
        lines.append(
            "- "
            + _t(
                "drive",
                lang_key,
                duration=route["duration_text"],
                distance=round(route["distance_km"]),
            )
        )

    # ── The cabs ──────────────────────────────────────────────────────────────
    # Already narrowed to what seats the party (`himcab/vehicles.py`), so every option
    # here is one the rider can actually book. Listing them all is the choice they came
    # to make.
    options = [o for o in (fare.get("options") or []) if o.get("total_fare_inr") is not None]
    for option in options:
        amount = _round_inr(option.get("total_fare_inr"))
        if amount is None:
            continue
        seats = option.get("seat_capacity")
        lines.append(
            "- "
            + _t(
                "cab",
                lang_key,
                name=option.get("category_name") or "Cab",
                seats=_t("seats", lang_key, count=seats) if seats else "",
                fare=amount,
            )
        )
    if not options:
        # No number at all rather than an estimate. A fare the backend did not return
        # is a fare nobody can honour.
        lines.append("- " + _t("no_fare", lang_key))

    # ── Conditions there, if anyone measured them ─────────────────────────────
    # Current temperature only. The forecast and the snowfall risk are real data and
    # belong in the itinerary reply, where somebody is choosing what to do with four
    # days — here they were the material behind "मौसम की भविष्यवाणी" and "हिमपात का
    # जोखिम" in a reply that was meant to be a cab and a price.
    weather = state.weather_data or {}
    if weather.get("current_temp_c") is not None and weather.get("condition"):
        lines.append(
            "- "
            + _t(
                "weather",
                lang_key,
                place=weather.get("destination") or destination,
                temp=round(float(weather["current_temp_c"]), 1),
                condition=weather["condition"],
            )
        )

    # ── A coupon, only when one was verified against this fare ────────────────
    best = next(
        (o for o in (state.offers or []) if o.get("applies_to_this_fare")), None
    )
    if best and best.get("coupon_code"):
        saving = _round_inr(best.get("coupon_value_inr"))
        if saving:
            lines.append(
                "- " + _t("offer", lang_key, code=best["coupon_code"], saving=saving)
            )

    # ── Where to press ────────────────────────────────────────────────────────
    # A statement, never a question. The app draws the button from `booking_handoff`;
    # asking "shall I book?" invites a yes this service has no way to act on.
    lines.append(_t("handoff", lang_key))

    return "\n".join(lines)


__all__ = ["build_ride_reply"]
