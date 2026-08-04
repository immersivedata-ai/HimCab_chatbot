"""
app/agent/nodes/writer.py
──────────────────────────
Writer Node — generates the final response using Gemini Flash with streaming.
Responsibilities:
  1. Build a rich, context-aware system prompt with all tool data
  2. Generate a natural Hinglish/English response
  3. Generate a structured JSON itinerary (Itinerary model)
  4. Emit structured SSE events for the mobile app (itinerary, route_plan, alerts)
  5. Stream tokens back to the SSE handler

For outstation visitors:
  - Prominently display the full route plan with per-leg details
  - Day 1 always starts with arrival journey from origin
  - Last day includes return journey information
  - Include road conditions, scenic highlights, and recommended stops

This is the most visible node — its output IS the user experience.
"""

import json
from datetime import datetime, timezone

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.models.itinerary import Itinerary
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

# Maximum retry attempts for structured output extraction
_MAX_RETRIES = 2


def _build_route_plan_section(state: AgentState) -> str:
    """
    Build a concise route summary for outstation visitors.
    Only shows total distance, time, and recommended mode — 2-3 bullets max.
    Full leg-by-leg data is sent separately as a structured SSE event.
    """
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

    # Pick the dominant travel mode from the first leg
    first_leg = rp["legs"][0] if rp["legs"] else {}
    mode = first_leg.get("recommended_mode", "cab")
    section += f"- Recommended Mode: {mode.upper().replace('+', ' + ')}\n"
    section += "NOTE: Include ONLY these 2-3 route bullets in your response. Do NOT expand or list individual road legs.\n"

    return section


def _build_offers_section(state: AgentState) -> str:
    """
    Build a formatted offers section for the system prompt.
    Displays personalised offers with promo codes for the writer to include in the response.
    """
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


def _build_system_prompt(state: AgentState) -> str:
    """
    Build the full system prompt for the Writer node,
    injecting all tool results as structured context.
    """
    entities = state.entities
    is_local = state.visitor_type == "local"

    # ── Check if user has explicitly requested English in recent messages ──
    user_requested_english = False
    for msg in state.messages[-5:]:
        from langchain_core.messages import HumanMessage as HM
        if isinstance(msg, HM):
            content_lower = msg.content.lower()
            if any(phrase in content_lower for phrase in [
                "speak in english", "english mein bolo", "in english please",
                "reply in english", "english me bolo", "english me jawab do",
                "answer in english", "respond in english", "talk in english",
                "english mein bat karo", "english me bat karo",
                "english mein baat karo", "english me baat karo",
            ]):
                user_requested_english = True
                break

    # ── Language logic ────────────────────────────────────────────────────
    # LOCAL visitors: ALWAYS respond in pure Hindi (Devanagari), regardless
    #   of whatever language they type in (English, Hinglish, Hindi).
    #   The ONLY exception is when the user explicitly asks for English.
    # OUTSTATION / UNKNOWN visitors: Use detected language. If detected as
    #   English → respond in English. Otherwise → Hindi (Devanagari).
    detected_lang = state.language or "hi"

    if user_requested_english:
        # User explicitly asked for English — honour it for ALL visitor types
        language_instruction = (
            "The user has requested English. Respond in clear, friendly English. "
            "Do NOT use any emojis in your response."
        )
    elif is_local:
        # LOCAL visitors: STRICT Hindi (Devanagari) — no matter what language they type in
        language_instruction = (
            "सख्त निर्देश: यह यूज़र हिमाचल प्रदेश का स्थानीय निवासी है। "
            "हमेशा शुद्ध हिंदी में जवाब दें, देवनागरी लिपि में। "
            "चाहे यूज़र अंग्रेज़ी में लिखे, हिंगलिश में लिखे, या हिंदी में — "
            "आपका जवाब हमेशा देवनागरी हिंदी में होना चाहिए। "
            "English या Hinglish (Roman script में Hindi) बिल्कुल मत लिखें। "
            "केवल तभी अंग्रेज़ी में जवाब दें जब यूज़र ने स्पष्ट रूप से कहा हो: "
            "'speak in English', 'English mein bolo', या इसी तरह कुछ। "
            "किसी भी emoji का उपयोग न करें।"
        )
    elif detected_lang == "en":
        # Non-local user typing in English — respond in English
        language_instruction = (
            "The user is typing in English. Respond in clear, friendly English. "
            "Do NOT use any emojis in your response."
        )
    else:
        # Non-local user typing in Hindi/Hinglish — respond in Hindi (Devanagari)
        language_instruction = (
            "सख्त निर्देश: हमेशा शुद्ध हिंदी में जवाब दें, देवनागरी लिपि में। "
            "चाहे यूज़र हिंगलिश में लिखे, रोमन लिपि में लिखे, या हिंदी में — "
            "आपका जवाब हमेशा देवनागरी हिंदी में होना चाहिए। "
            "Hinglish (Roman script में Hindi) बिल्कुल मत लिखें। "
            "केवल तभी अंग्रेज़ी में जवाब दें जब यूज़र ने स्पष्ट रूप से कहा हो: "
            "'speak in English', 'English mein bolo', या इसी तरह कुछ। "
            "किसी भी emoji का उपयोग न करें।"
        )

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

    snowfall_section = ""
    if state.snowfall_data:
        s = state.snowfall_data
        snowfall_section = f"\n**Snowfall Risk:** {s.get('risk', 'none').upper()} — {s.get('message', '')}"

    places_section = ""
    if state.places_data:
        for category, places in state.places_data.items():
            if places:
                top_places = places[:5]  # Show more places for better recommendations
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

    drive_section = ""
    if state.drive_times:
        drive_section = "\n**Drive Times:**\n"
        for route, info in state.drive_times.items():
            if info.get("duration_minutes"):
                hours = info["duration_minutes"] // 60
                mins = info["duration_minutes"] % 60
                drive_section += f"  - {route}: ~{hours}h {mins}m ({info.get('distance_km')} km)\n"

    # ── Route plan for outstation visitors (prominent display) ────────────────
    route_plan_section = _build_route_plan_section(state)

    # ── Travel estimate from origin (fallback if no route_plan) ───────────────
    travel_section = ""
    origin_display = entities.origin or "not specified"
    if not route_plan_section and state.travel_estimate:
        te = state.travel_estimate
        duration_h = te.get("duration_hours")
        distance = te.get("distance_km")
        mode = te.get("recommended_mode", "cab")
        note = te.get("note", "")

        if duration_h:
            travel_section = f"""
**TRAVEL FROM ORIGIN — {te.get('origin', origin_display).upper()} to {entities.destination.upper()}:**
- Estimated Travel Time: ~{duration_h} hours ({te.get('duration_minutes')} minutes)
- Distance: ~{distance} km
- Recommended Mode: {mode.upper().replace('+', ' + ')}
{f'- Route Note: {note}' if note else ''}

IMPORTANT: Since the traveller is coming from {te.get('origin', origin_display)}, Day 1 of the itinerary MUST begin with their arrival/journey from {te.get('origin', origin_display)}. Account for travel fatigue — plan lighter activities on arrival day.
"""
        else:
            travel_section = f"""
**TRAVEL FROM ORIGIN — {te.get('origin', origin_display).upper()} to {entities.destination.upper()}:**
- Exact travel time estimate not available for this route
- Recommended: Contact Him Cab for precise route timing
- Day 1 should still account for arrival travel

"""
    elif not route_plan_section and entities.origin:
        travel_section = f"""
**TRAVEL FROM ORIGIN — {entities.origin.upper()} to {entities.destination.upper()}:**
- Exact travel time estimate not available for this route
- Day 1 should start with arrival from {entities.origin}
"""

    user_context = ""
    if state.user_context_summary:
        user_context = f"\n**User Profile:** {state.user_context_summary}"

    # ── Outstation-specific instructions ──────────────────────────────────────
    is_outstation = state.visitor_type == "outstation"
    outstation_instructions = ""
    if is_outstation and entities.origin:
        outstation_instructions = f"""
**OUTSTATION VISITOR INSTRUCTIONS (CRITICAL):**
- This visitor is coming from **{entities.origin}** (OUTSIDE Himachal Pradesh)
- **Day 1 MUST begin with the arrival journey** from {entities.origin} to {entities.destination}
- Account for travel fatigue — plan lighter activities on arrival day
- Return journey preference: **{entities.return_journey_preference or 'not specified'}**
  - If "same_cab_return": Last day MUST include return journey from {entities.destination} back to {entities.origin}. Plan departure timing accordingly.
  - If "own_arrangement": Do NOT include return journey details. Last day should focus on sightseeing/departure prep only.
  - If not specified: Briefly mention return options but don't plan it in detail.
- Mention the recommended travel mode (cab, flight+cab, train+cab) based on distance
- If the distance is >500km, suggest flight+cab or train+cab as alternatives

"""

    # ── Local visitor-specific instructions ────────────────────────────────────
    local_instructions = ""
    if is_local:
        home_city = entities.home_city or entities.origin or "their HP home city"
        local_instructions = f"""
**LOCAL VISITOR INSTRUCTIONS (IMPORTANT):**
- This visitor is a **Himachal Pradesh resident** based in **{home_city}**
- They likely already know the popular tourist spots — **prioritise hidden gems and off-beat experiences**
- Travel time is shorter (intra-HP routes are typically 1-6 hours by cab)
- **No need to account for long-distance travel fatigue** — they can start activities sooner
- Do NOT ask about or plan return journey — locals manage their own return
- For 1-2 day trips, use a **weekend getaway format**: compact, action-packed itinerary
- **Highlight seasonal specialities** relevant to the current month (apple season, cherry blossoms, snowfall, monsoon greenery)
- Mention **local festivals or events** happening at the destination if any
- Suggest **insider tips** that a local would appreciate (best time to avoid crowds, lesser-known viewpoints, local dhabas)
- Include **local_tips** in each day's plan in the JSON output

"""

    # ── Offers section ────────────────────────────────────────────────────────
    offers_section = _build_offers_section(state)

    # ── travel_info JSON template (only if origin is known) ───────────────────
    travel_info_json = ""
    if entities.origin:
        te = state.travel_estimate or {}
        travel_info_json = f"""
  "travel_info": {{
    "origin": "{entities.origin}",
    "destination": "{entities.destination}",
    "estimated_duration_hours": {te.get('duration_hours') or 'null'},
    "estimated_distance_km": {te.get('distance_km') or 'null'},
    "recommended_mode": "{te.get('recommended_mode', 'cab')}",
    "route_notes": "..."
  }},"""

    # ── route_plan JSON template ──────────────────────────────────────────────
    route_plan_json = ""
    if state.route_plan:
        rp = state.route_plan
        route_plan_json = f"""
  "route_plan": {{
    "legs": {json.dumps(rp.get('legs', []))},
    "total_distance_km": {rp.get('total_distance_km') or 'null'},
    "total_duration_hours": {rp.get('total_duration_hours') or 'null'},
    "optimal_order": {json.dumps(rp.get('optimal_order', []))},
    "summary": "{rp.get('summary', '')}"
  }},"""

    # ── Offers JSON template ──────────────────────────────────────────────────
    offers_json = ""
    if state.available_offers:
        offers_json = f"""
  "offers": {json.dumps(state.available_offers[:3])},"""

    # ── Hidden gems + local_tips JSON template (for local visitors) ───────────
    local_json_fields = ""
    if is_local:
        local_json_fields = """
  "hidden_gems": ["...", "..."],"""

    # ── Booking Confirmation Status ───────────────────────────────────────────
    booking_status_section = ""
    if state.final_booking_id:
        booking_status_section = f"""
**CRITICAL INSTRUCTION (BOOKING CONFIRMED):**
A booking was just confirmed! You must enthusiastically tell the user their HimCab is confirmed.
Give them their Booking ID: #{state.final_booking_id}.
DO NOT ask if they want to book again. Congratulate them on the booking.
"""

    return f"""You are the AI travel concierge for Him Cab — Himachal Pradesh's most trusted local cab service.

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
1. Keep it SHORT and CONCISE. Use simple bullet points throughout — no long paragraphs, no essays.
2. Briefly mention route info (distance, time, mode) in 2-3 bullet points max.
3. Create a concise day-by-day itinerary using ONLY bullet points:
   - Each day: 3-5 bullet points max (morning, afternoon, evening activities)
   - Keep each bullet to one line — just the place/activity name and a short note
   - Day 1 starts with arrival from {origin_display}
4. If outstation visitor with return_journey_preference = "same_cab_return": include return journey on last day.
   If "own_arrangement": skip return details on last day.
5. Add a quick "Him Cab Package" bullet with estimated cost.
6. **CRITICAL: ONLY recommend places from the "Real-time Data Available" section. Do NOT invent places.**
7. Include 1-2 food suggestions per day (inline, not separate section).
8. Mention weather warnings in a single bullet if relevant.
9. Quick cost summary: cab + hotels + food + total (bullet points, not a table).
10. Be warm but brief — no filler text. Every line should add value.
11. At the END of your response, output a JSON block wrapped in ```json ... ```
    containing the structured itinerary (see format below)
12. Give approximate pricing and not exact pricing. DO NOT give exact prices.
13. If the user is a LOCAL visitor, include a "Hidden Gems" section with 2-3 off-beat spots and add local_tips per day in the JSON.
14. If there are OFFERS available, mention the best offer prominently near the end of your response (before JSON). Include the promo code.
15. Always recommend 'himcab' services for the journey.
16. At the end of your response, keep a loop by asking the user if they would like to book a himcab service for their journey from {origin_display} to {entities.destination}.

**Structured JSON format to include at the end:**
{{
  "destination": "...",
  "duration_days": N,
  "group_type": "...",
  "budget_range": "...",{travel_info_json}{route_plan_json}{offers_json}{local_json_fields}
  "days": [
    {{
      "day": 1,
      "theme": "Arrival & Settle In",
      "activities": [
        {{"time": "Morning/Afternoon", "place": "...", "type": "travel", "description": "..."}},
        {{"time": "Evening", "place": "...", "type": "sightseeing", "description": "..."}}
      ],
      "meal_suggestions": ["..."],
      "local_tips": ["..."]
    }}
  ],
  "cab_package": {{
    "package_type": "...",
    "vehicle": "Innova Crysta",
    "estimated_cost_inr": 0,
    "includes": ["..."]
  }},
  "hotels": [{{"name": "...", "category": "moderate", "price_per_night_inr": 0}}],
  "weather": {{"summary": "...", "alert": null}},
  "cost_breakdown": {{
    "cab_inr": 0,
    "hotels_inr": 0,
    "food_estimate_inr": 0,
    "total_inr": 0
  }}
}}
"""


def _extract_structured_itinerary(text: str) -> dict | None:
    """Extract the JSON block from the writer's response text."""
    try:
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
            return json.loads(json_str)
        elif "```" in text:
            # Try to find any code block that looks like JSON
            blocks = text.split("```")
            for i in range(1, len(blocks), 2):  # odd indices are code blocks
                block = blocks[i].strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                if block.startswith("{"):
                    try:
                        return json.loads(block)
                    except json.JSONDecodeError:
                        continue
        
        # Fallback: attempt to find raw JSON without backticks
        import re
        match = re.search(r'\{\s*"destination"', text)
        if match:
            start_idx = match.start()
            json_str = text[start_idx:]
            end_idx = json_str.rfind('}')
            if end_idx != -1:
                json_str = json_str[:end_idx+1]
                return json.loads(json_str)

    except (json.JSONDecodeError, IndexError) as e:
        log.warning("structured_extraction_failed", error=str(e))
    return None


def _clean_response_text(text: str) -> str:
    """Strip the JSON block from the human-readable response text."""
    clean = text
    if "```json" in clean:
        clean = clean.split("```json")[0].strip()
    elif "```" in clean:
        # Find the last code block and remove everything from it onwards
        parts = clean.rsplit("```", 2)
        if len(parts) >= 3:
            clean = parts[0].strip()
            
    # Strip raw JSON if no backticks were used
    import re
    match = re.search(r'\{\s*"destination"', clean)
    if match:
        clean = clean[:match.start()].strip()
        
    return clean


async def writer_node(state: AgentState) -> dict:
    """
    LangGraph Writer Node.
    Generates the final AI response with structured itinerary.
    Includes retry logic for structured output extraction.
    Enhanced for outstation visitors with route plan display.
    Enhanced for local visitors with hidden gems and offers.
    """
    log.info("writer_node_entered", session_id=state.session_id)

    system_prompt = _build_system_prompt(state)

    # Build message history for multi-turn context
    messages = [SystemMessage(content=system_prompt)]
    for msg in state.messages[-6:]:  # last 3 turns
        messages.append(msg)

    full_response = ""
    structured_events = []

    try:
        llm = get_vertex_ai_service().get_flash_model(streaming=False)
        response = await llm.ainvoke(messages)
        full_response = response.content

        # Extract structured JSON from response
        itinerary_dict = _extract_structured_itinerary(full_response)

        # Retry: if structured extraction failed, ask the model to output just JSON
        if itinerary_dict is None and state.intent == "plan_trip":
            log.info("structured_extraction_retry", session_id=state.session_id)
            retry_messages = messages + [
                AIMessage(content=full_response),
                HumanMessage(content=(
                    "Please output ONLY the structured JSON itinerary from your response above. "
                    "No explanation, just the JSON wrapped in ```json ... ```"
                )),
            ]
            try:
                retry_response = await llm.ainvoke(retry_messages)
                itinerary_dict = _extract_structured_itinerary(retry_response.content)
                if itinerary_dict:
                    log.info("structured_extraction_retry_success", session_id=state.session_id)
            except Exception as retry_err:
                log.warning("structured_extraction_retry_failed", error=str(retry_err))

        # Clean the text response (remove JSON block)
        clean_text = _clean_response_text(full_response)

        # Build structured SSE events for mobile
        if itinerary_dict:
            structured_events.append({
                "type": "itinerary",
                "payload": itinerary_dict,
            })

        # Emit route_plan as a separate structured event for outstation visitors
        if state.route_plan:
            structured_events.append({
                "type": "route_plan",
                "payload": state.route_plan,
            })

        if state.weather_data and state.weather_data.get("alerts"):
            alerts = state.weather_data["alerts"]
            if isinstance(alerts, list) and alerts:
                structured_events.append({
                    "type": "weather_alert",
                    "payload": alerts[0],
                })
            elif isinstance(alerts, dict):
                structured_events.append({
                    "type": "weather_alert",
                    "payload": alerts,
                })

        if state.snowfall_data and state.snowfall_data.get("risk") in ("high", "critical"):
            structured_events.append({
                "type": "snowfall_alert",
                "payload": state.snowfall_data,
            })

        # Emit offers as a separate structured event
        if state.available_offers:
            structured_events.append({
                "type": "offers",
                "payload": {
                    "offers": state.available_offers,
                    "visitor_type": state.visitor_type,
                    "is_frequent_traveller": state.is_frequent_traveller,
                },
            })

        # Validate the itinerary dict against the Pydantic model
        validated_itinerary = None
        if itinerary_dict:
            try:
                validated_itinerary = Itinerary.model_validate(itinerary_dict)
            except Exception as val_err:
                log.warning(
                    "itinerary_validation_failed",
                    error=str(val_err),
                    session_id=state.session_id,
                )
                # Still include the raw dict in structured events even if validation fails
                validated_itinerary = None

        log.info(
            "writer_completed",
            response_length=len(full_response),
            has_itinerary=validated_itinerary is not None,
            has_route_plan=state.route_plan is not None,
            visitor_type=state.visitor_type,
            is_local=state.visitor_type == "local",
            offers_count=len(state.available_offers),
            structured_events=len(structured_events),
            places_categories=list(state.places_data.keys()) if state.places_data else [],
            session_id=state.session_id,
        )

        return {
            "final_response_text": clean_text,
            "structured_itinerary": validated_itinerary,
            "structured_events": structured_events,
            "messages": [AIMessage(content=clean_text)],
            "node_trace": ["writer"],
        }

    except Exception as e:
        log.error("writer_failed", error=str(e), error_type=type(e).__name__, session_id=state.session_id)
        fallback = "Sorry, I encountered an issue generating your travel plan. Please try again."
        return {
            "final_response_text": fallback,
            "structured_events": [],
            "messages": [AIMessage(content=fallback)],
            "node_trace": ["writer"],
            "error": str(e),
        }
