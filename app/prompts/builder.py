"""
app/prompts/builder.py
───────────────────────
Prompt assembly engine — loads YAML templates and builds context-aware
system prompts for each LangGraph node.

This centralises all prompt logic so that:
  1. Prompt changes don't require code changes
  2. Language selection is handled in one place
  3. Templates are versioned and auditable
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from app.core.logging import get_logger

if TYPE_CHECKING:
    from app.agent.state import AgentState

log = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ── Template Loading ──────────────────────────────────────────────────────────

@lru_cache(maxsize=10)
def _load_template(name: str) -> dict[str, Any]:
    """Load and cache a YAML template by name (without extension)."""
    path = _TEMPLATES_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_language_config() -> dict[str, Any]:
    """Load language instruction templates."""
    return _load_template("language_instructions")


# ── Router Prompt ─────────────────────────────────────────────────────────────

def build_router_prompt() -> str:
    """
    Build the system prompt for the Router node.
    Returns the static classification prompt (no dynamic state needed).
    """
    tmpl = _load_template("router_system")
    return tmpl["system_prompt"]


# ── Planner Prompt ────────────────────────────────────────────────────────────

def build_planner_prompt() -> str:
    """
    Build the system prompt for the Planner node.
    Returns the static entity extraction prompt (no dynamic state needed).
    """
    tmpl = _load_template("planner_system")
    return tmpl["system_prompt"]


# ── Language Selection ────────────────────────────────────────────────────────

def resolve_language_instruction(state: AgentState) -> str:
    """
    Determine the correct language instruction based on visitor type
    and explicit user requests.

    Priority:
      1. User explicitly asked for English → English (all visitor types)
      2. Local visitor → ALWAYS Hindi (Devanagari)
      3. Detected language is English (non-local) → English
      4. Default → Hindi (Devanagari)
    """
    from langchain_core.messages import HumanMessage

    lang_config = _load_language_config()
    is_local = state.visitor_type == "local"

    # Check if user explicitly requested English in recent messages
    trigger_phrases = lang_config.get("english_trigger_phrases", [])
    user_requested_english = False
    for msg in state.messages[-5:]:
        if isinstance(msg, HumanMessage):
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in trigger_phrases):
                user_requested_english = True
                break

    # Apply priority rules
    if user_requested_english:
        return lang_config["english_explicit"]
    elif is_local:
        return lang_config["hindi_local"]
    elif (state.language or "hi") == "en":
        return lang_config["english_detected"]
    else:
        return lang_config["hindi_default"]


# ── Writer Prompt ─────────────────────────────────────────────────────────────

def build_writer_prompt(state: AgentState) -> str:
    """
    Build the full system prompt for the Writer node.
    Assembles the static template with dynamic context from agent state.
    """
    tmpl = _load_template("writer_system")
    entities = state.entities
    is_local = state.visitor_type == "local"
    is_outstation = state.visitor_type == "outstation"

    # ── Language instruction ──────────────────────────────────────────────
    language_instruction = resolve_language_instruction(state)

    # ── Weather section ───────────────────────────────────────────────────
    weather_section = ""
    if state.weather_data:
        w = state.weather_data
        weather_section = f"""
**Current Weather at {entities.destination}:**
- Temperature: {w.get('current_temp_c')}°C (feels like {w.get('feels_like_c')}°C)
- Condition: {w.get('condition')}
- Humidity: {w.get('humidity_percent')}%
- Wind: {w.get('wind_speed_kmh')} km/h
- Upcoming Forecast: {json.dumps(w.get('forecast_7day', [])[:3], indent=2)}
"""
        if w.get("alerts"):
            weather_section += f"\nWEATHER ALERT: {w['alerts']}"

    # ── Snowfall section ──────────────────────────────────────────────────
    snowfall_section = ""
    if state.snowfall_data:
        s = state.snowfall_data
        snowfall_section = f"\n**Snowfall Risk:** {s.get('risk', 'none').upper()} — {s.get('message', '')}"

    # ── Places section ────────────────────────────────────────────────────
    places_section = ""
    if state.places_data:
        for category, places in state.places_data.items():
            if places:
                top_places = places[:5]
                places_section += f"\n**{category.title()} (from OpenStreetMap data):**\n"
                for p in top_places:
                    name = p.get('name', 'Unknown')
                    rating = p.get('rating', 'N/A')
                    price = p.get('price_level', '')
                    address = p.get('address', '')
                    addr_str = f" — {address}" if address else ""
                    places_section += f"  - {name}{addr_str} (Rating: {rating}, {price})\n"
    else:
        places_section = "\n**Places Data:** No real-time place data available. Provide general activity suggestions based on your knowledge of the destination.\n"

    # ── Drive times section ───────────────────────────────────────────────
    drive_section = ""
    if state.drive_times:
        drive_section = "\n**Drive Times:**\n"
        for route, info in state.drive_times.items():
            if info.get("duration_minutes"):
                hours = info["duration_minutes"] // 60
                mins = info["duration_minutes"] % 60
                drive_section += f"  - {route}: ~{hours}h {mins}m ({info.get('distance_km')} km)\n"

    # ── Route plan section ────────────────────────────────────────────────
    route_plan_section = _build_route_plan_section(state)

    # ── Travel estimate section ───────────────────────────────────────────
    origin_display = entities.origin or "not specified"
    travel_section = _build_travel_section(state, route_plan_section, origin_display)

    # ── User context ──────────────────────────────────────────────────────
    user_context = ""
    if state.user_context_summary:
        user_context = f"\n**User Profile:** {state.user_context_summary}"

    # ── Outstation instructions ───────────────────────────────────────────
    outstation_instructions = ""
    if is_outstation and entities.origin:
        outstation_instructions = tmpl["outstation_instructions"].format(
            origin=entities.origin,
            destination=entities.destination,
            return_journey_preference=entities.return_journey_preference or "not specified",
        )

    # ── Local instructions ────────────────────────────────────────────────
    local_instructions = ""
    if is_local:
        home_city = entities.home_city or entities.origin or "their HP home city"
        local_instructions = tmpl["local_instructions"].format(home_city=home_city)

    # ── Offers section ────────────────────────────────────────────────────
    offers_section = _build_offers_section(state)

    # ── Booking status ────────────────────────────────────────────────────
    booking_status_section = ""
    if state.final_booking_id:
        booking_status_section = tmpl["booking_confirmed"].format(
            booking_id=state.final_booking_id,
        )

    # ── JSON template fields ──────────────────────────────────────────────
    travel_info_json = _build_travel_info_json(entities, state)
    route_plan_json = _build_route_plan_json(state)
    offers_json = _build_offers_json(state)
    local_json_fields = '\n  "hidden_gems": ["...", "..."],' if is_local else ""

    # ── Assemble full prompt ──────────────────────────────────────────────
    instructions = tmpl["instructions"]
    # Add dynamic instruction lines
    instructions += f"\n  15. At the end of your response, keep a loop by asking the user if they would like to book a himcab service for their journey from {origin_display} to {entities.destination}.\n"
    if is_outstation:
        instructions += f"\n  16. If outstation visitor with return_journey_preference = \"same_cab_return\": include return journey on last day.\n     If \"own_arrangement\": skip return details on last day.\n"
        instructions += f"\n  17. Day 1 starts with arrival from {origin_display}\n"

    json_format = tmpl["json_format"].format(
        travel_info_json=travel_info_json,
        route_plan_json=route_plan_json,
        offers_json=offers_json,
        local_json_fields=local_json_fields,
    )

    return f"""{tmpl['role_definition']}

{language_instruction}

**Trip Request:**
- Destination: {entities.destination}
- Duration: {entities.duration_days} days
- Group: {entities.group_type} ({entities.group_size or 'N/A'} people)
- Budget: {entities.budget_range}
- Interests: {', '.join(entities.interests) if entities.interests else 'general sightseeing'}
- Origin: {origin_display}
- Visitor Type: {state.visitor_type or 'unknown'}
{user_context}
{route_plan_section}
{travel_section}
{outstation_instructions}
{local_instructions}
{offers_section}
{booking_status_section}

**Real-time Data Available:**
{weather_section}
{snowfall_section}
{places_section}
{drive_section}

**Instructions:**
{instructions}

**Structured JSON format to include at the end:**
{json_format}
"""


# ── Private Helpers ───────────────────────────────────────────────────────────

def _build_route_plan_section(state: AgentState) -> str:
    """Build a concise route summary for outstation visitors."""
    if not state.route_plan:
        return ""

    rp = state.route_plan
    if not rp.get("legs"):
        return ""

    section = "\n**ROUTE SUMMARY (brief mention only — do NOT list every leg):**\n"
    if rp.get("summary"):
        section += f"- Route: {rp['summary']}\n"
    if rp.get("total_distance_km"):
        section += f"- Total Distance: ~{rp['total_distance_km']} km\n"
    if rp.get("total_duration_hours"):
        section += f"- Estimated Travel Time: ~{rp['total_duration_hours']} hours\n"

    first_leg = rp["legs"][0] if rp["legs"] else {}
    mode = first_leg.get("recommended_mode", "cab")
    section += f"- Recommended Mode: {mode.upper().replace('+', ' + ')}\n"
    section += "NOTE: Include ONLY these 2-3 route bullets in your response. Do NOT expand or list individual road legs.\n"

    return section


def _build_travel_section(
    state: AgentState,
    route_plan_section: str,
    origin_display: str,
) -> str:
    """Build the travel estimate section."""
    entities = state.entities

    if not route_plan_section and state.travel_estimate:
        te = state.travel_estimate
        duration_h = te.get("duration_hours")
        distance = te.get("distance_km")
        mode = te.get("recommended_mode", "cab")
        note = te.get("note", "")

        if duration_h:
            return f"""
**TRAVEL FROM ORIGIN — {te.get('origin', origin_display).upper()} to {entities.destination.upper()}:**
- Estimated Travel Time: ~{duration_h} hours ({te.get('duration_minutes')} minutes)
- Distance: ~{distance} km
- Recommended Mode: {mode.upper().replace('+', ' + ')}
{f'- Route Note: {note}' if note else ''}

IMPORTANT: Since the traveller is coming from {te.get('origin', origin_display)}, Day 1 of the itinerary MUST begin with their arrival/journey from {te.get('origin', origin_display)}. Account for travel fatigue — plan lighter activities on arrival day.
"""
        else:
            return f"""
**TRAVEL FROM ORIGIN — {te.get('origin', origin_display).upper()} to {entities.destination.upper()}:**
- Exact travel time estimate not available for this route
- Recommended: Contact Him Cab for precise route timing
- Day 1 should still account for arrival travel
"""

    elif not route_plan_section and entities.origin:
        return f"""
**TRAVEL FROM ORIGIN — {entities.origin.upper()} to {entities.destination.upper()}:**
- Exact travel time estimate not available for this route
- Day 1 should start with arrival from {entities.origin}
"""

    return ""


def _build_offers_section(state: AgentState) -> str:
    """Build a formatted offers section for the system prompt."""
    if not state.available_offers:
        return ""

    section = "\n**AVAILABLE OFFERS FOR THIS CUSTOMER:**\n"

    for i, offer in enumerate(state.available_offers[:4], 1):
        title = offer.get("title", "Special Offer")
        desc = offer.get("description", "")
        code = offer.get("promo_code", "")
        discount = offer.get("discount_percent", 0)
        applicable = ", ".join(offer.get("applicable_to", ["cab"]))
        conditions = offer.get("conditions", "")

        section += f"\n  **Offer {i}: {title}**\n"
        section += f"  - {desc}\n"
        section += f"  - Discount: {discount}% off on {applicable}\n"
        section += f"  - Promo Code: **{code}**\n"
        if conditions:
            section += f"  - Terms: {conditions}\n"

    section += "\nIMPORTANT: Mention the BEST offer (highest discount) prominently in your response. Include the promo code so the user can use it.\n"

    return section


def _build_travel_info_json(entities: Any, state: AgentState) -> str:
    """Build the travel_info JSON template for the structured output."""
    if not entities.origin:
        return ""
    te = state.travel_estimate or {}
    return f"""
  "travel_info": {{
    "origin": "{entities.origin}",
    "destination": "{entities.destination}",
    "estimated_duration_hours": {te.get('duration_hours') or 'null'},
    "estimated_distance_km": {te.get('distance_km') or 'null'},
    "recommended_mode": "{te.get('recommended_mode', 'cab')}",
    "route_notes": "..."
  }},"""


def _build_route_plan_json(state: AgentState) -> str:
    """Build the route_plan JSON template for the structured output."""
    if not state.route_plan:
        return ""
    rp = state.route_plan
    return f"""
  "route_plan": {{
    "legs": {json.dumps(rp.get('legs', []))},
    "total_distance_km": {rp.get('total_distance_km') or 'null'},
    "total_duration_hours": {rp.get('total_duration_hours') or 'null'},
    "optimal_order": {json.dumps(rp.get('optimal_order', []))},
    "summary": "{rp.get('summary', '')}"
  }},"""


def _build_offers_json(state: AgentState) -> str:
    """Build the offers JSON template for the structured output."""
    if not state.available_offers:
        return ""
    return f"""
  "offers": {json.dumps(state.available_offers[:3])},"""
