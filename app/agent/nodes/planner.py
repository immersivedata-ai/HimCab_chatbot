"""
app/agent/nodes/planner.py
───────────────────────────
Planner Node — extracts structured trip entities using Gemini Pro.
Responsibilities:
  1. Extract destination, duration, group type, budget, interests from conversation
  2. Classify visitor_type (outstation vs local)
  3. Detect if key information is missing
  4. Generate a clarifying question if necessary (outstation-aware priority)
  5. Update state.entities with confirmed information

Uses Gemini 1.5 Pro for better entity extraction accuracy (complex JSON output).
"""

import json
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from app.agent.state import AgentState
from app.core.logging import get_logger
from app.models.session import ExtractedEntities
from app.services.vertex_ai import get_vertex_ai_service

log = get_logger(__name__)

PLANNER_SYSTEM_PROMPT = """You are an expert Himachal Pradesh travel planner working for Him Cab.

Your job is to extract structured travel information from the conversation and return valid JSON.

Extract the following fields (use null if not mentioned):
- destination: string (e.g., "Dharamshala", "Manali", "Shimla")
- origin: string (where they're coming from — CRITICAL for outstation visitors, also useful for locals)
- duration_days: integer
- travel_dates: string (e.g., "December 2024", "next week", "Jan 15-18")
- group_type: one of [solo, couple, family, friends, corporate]
- group_size: integer
- budget_range: one of [budget, moderate, luxury]
- interests: list of strings (e.g., ["temples", "trekking", "local food", "snow"])
- special_requirements: list of strings (e.g., ["wheelchair accessible", "vegetarian only"])
- visitor_type: one of ["outstation", "local", null]
- return_journey_preference: one of ["same_cab_return", "own_arrangement", null] — ONLY for outstation visitors
- home_city: string — ONLY for local visitors (their base city within HP, e.g. "Shimla", "Mandi")

Also return:
- needs_clarification: boolean
- clarification_question: string (in the same language as the user if needs_clarification=true)
- confidence: float 0-1
- booking_intent_active: boolean (true if the user explicitly mentions booking a cab now)
- user_confirmed_booking: boolean (true if the user explicitly confirmed the booking after being asked)
- booking_pickup: string (exact pickup address if booking_intent_active)
- booking_dropoff: string (exact dropoff address if booking_intent_active)
- booking_time: string (HH:mm format, if booking_intent_active)

VISITOR TYPE CLASSIFICATION:
- Himachal Pradesh cities/towns: Shimla, Manali, Dharamshala, McLeod Ganj, Kullu, Mandi, Solan, Bilaspur, Hamirpur, Una, Kangra, Chamba, Kinnaur, Lahaul, Spiti, Kasol, Dalhousie, Palampur, Nahan, Sirmaur
- If origin is a Himachal city → visitor_type = "local", and set home_city = origin
- If origin is OUTSIDE Himachal (Delhi, Chandigarh, Mumbai, Bangalore, Kolkata, Jaipur, Amritsar, Ludhiana, Dehradun, Hyderabad, Pune, etc.) → visitor_type = "outstation"
- If origin is unknown → visitor_type = null (inherit from previous state)

CLARIFICATION PRIORITY FOR OUTSTATION VISITORS (follow this exact order):
1. If destination is NOT known → ask destination FIRST
   "Which place in Himachal Pradesh would you like to visit? (Manali, Shimla, Dharamshala, Kasol, etc.)"
2. If destination IS known but origin is NOT known → ask origin NEXT
   "Where are you travelling from? Knowing your starting city helps us plan the best route and travel time."
3. If destination AND origin are known but duration is NOT known → ask duration
   "How many days are you planning for this trip?"
4. If destination, origin, AND duration are known but travel_dates is NOT known → ask travel dates
   "When are you planning to visit? (e.g., next week, July 15, etc.)"
5. If destination, origin, duration, AND travel_dates are known but return_journey_preference is NOT known → ask about return cab
   "Have you booked your return cab from [destination] back to [origin]?"
6. If all above are known → needs_clarification = false

CLARIFICATION PRIORITY FOR LOCAL VISITORS (follow this exact order):
1. If destination is NOT known → ask destination FIRST
   "Which place would you like to explore? Since you're from Himachal, we can suggest some hidden gems too!"
2. If destination IS known but origin/home_city is NOT known → ask where they're based
   "Which city in Himachal are you based in? This helps us plan the best route for you."
3. If destination AND origin known but duration is NOT known → ask duration
   "How many days are you planning? Weekend getaway (1-2 days) or a longer trip?"
4. If destination, origin, duration known but travel_dates NOT known → ask travel dates
   "When are you planning to go? (e.g., this weekend, next Friday, etc.)"
5. If all above known → needs_clarification = false
NOTE: Do NOT ask return_journey_preference for local visitors — they typically manage their own return.
NOTE: For locals, suggest shorter trips (1-2 days) as default if duration seems unclear.

IMPORTANT:
- Himachal Pradesh destinations: Dharamshala, McLeod Ganj, Manali, Shimla, Kasol, Spiti, Dalhousie, Kufri, Bir Billing, Khajjiar, Kullu, Chail, Kinnaur, Chitkul, Tirthan Valley, Jibhi, Sangla, Narkanda, Kalpa, Palampur
- Origin cities can be anywhere in India — common ones: Delhi, Chandigarh, Amritsar, Ludhiana, Dehradun, Jaipur, Mumbai, Kolkata, Pathankot, Bangalore, Hyderabad, Pune
- IMPORTANT LANGUAGE RULE: ALWAYS write the clarification_question in pure Hindi (Devanagari script हिंदी). This applies to ALL visitor types — outstation AND local. Do NOT write in English or Hinglish under any circumstances, UNLESS the conversation history shows the user has explicitly asked to speak in English.
- Never ask more than ONE clarification question at a time
- Be warm and friendly in your clarification questions
- For outstation visitors, emphasize that knowing origin helps plan the optimal route and travel time
- For local visitors, acknowledge their familiarity and suggest off-beat experiences

Respond ONLY with valid JSON.

Example 1 — destination known, but no origin (outstation):
Input: "Main Dharamshala 3 din ke liye ja raha hoon family ke saath"
Output:
{
  "destination": "Dharamshala",
  "origin": null,
  "duration_days": 3,
  "travel_dates": null,
  "group_type": "family",
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": null,
  "return_journey_preference": null,
  "home_city": null,
  "needs_clarification": true,
  "clarification_question": "अच्छा विकल्प! आप किस शहर से आ रहे हैं? (जैसे दिल्ली, चंडीगढ़, आदि) इससे हम आपके लिए सबसे अच्छा रास्ता और यात्रा समय तय कर सकते हैं।",
  "confidence": 0.85
}

Example 2 — destination, origin, duration known but no travel dates (outstation):
Input: "I'm coming from Delhi to Manali for 4 days with friends, moderate budget"
Output:
{
  "destination": "Manali",
  "origin": "Delhi",
  "duration_days": 4,
  "travel_dates": null,
  "group_type": "friends",
  "group_size": null,
  "budget_range": "moderate",
  "interests": [],
  "special_requirements": [],
  "visitor_type": "outstation",
  "return_journey_preference": null,
  "home_city": null,
  "needs_clarification": true,
  "clarification_question": "आप मनाली कब जाना चाहते हैं? (जैसे अगले सप्ताह, 15 जुलाई, आदि)",
  "confidence": 0.92
}

Example 3 — travel dates provided, now asking about return cab:
Input: "July 20 se"
Output:
{
  "destination": null,
  "origin": null,
  "duration_days": null,
  "travel_dates": "July 20",
  "group_type": null,
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": null,
  "return_journey_preference": null,
  "home_city": null,
  "needs_clarification": true,
  "clarification_question": "क्या आपने मनाली से दिल्ली वापसी के लिए कैब बुक की है?",
  "confidence": 0.90
}

Example 4 — answering return cab question:
Input: "No, not yet"
Output:
{
  "destination": null,
  "origin": null,
  "duration_days": null,
  "travel_dates": null,
  "group_type": null,
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": null,
  "return_journey_preference": "not_booked",
  "home_city": null,
  "needs_clarification": false,
  "clarification_question": null,
  "confidence": 0.90
}

Example 5 — everything known upfront (outstation):
Input: "I'm coming from Delhi to Manali for 4 days starting July 20, already booked return"
Output:
{
  "destination": "Manali",
  "origin": "Delhi",
  "duration_days": 4,
  "travel_dates": "July 20",
  "group_type": null,
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": "outstation",
  "return_journey_preference": "booked",
  "home_city": null,
  "needs_clarification": false,
  "clarification_question": null,
  "confidence": 0.95
}

Example 6 — local visitor, destination known but no home_city:
Input: "Kasol jaana hai weekend pe"
Output:
{
  "destination": "Kasol",
  "origin": null,
  "duration_days": 2,
  "travel_dates": "this weekend",
  "group_type": null,
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": null,
  "return_journey_preference": null,
  "home_city": null,
  "needs_clarification": true,
  "clarification_question": "आप हिमाचल प्रदेश के किस शहर में रहते हैं? इससे हम आपके लिए कासोल का सबसे अच्छा रास्ता ढूंढ सकते हैं।",
  "confidence": 0.82
}

Example 7 — local visitor, everything known:
Input: "Shimla se Manali jaana hai 2 din ke liye, is weekend"
Output:
{
  "destination": "Manali",
  "origin": "Shimla",
  "duration_days": 2,
  "travel_dates": "this weekend",
  "group_type": null,
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": "local",
  "return_journey_preference": null,
  "home_city": "Shimla",
  "needs_clarification": false,
  "clarification_question": null,
  "confidence": 0.95
}

Example 8 — local visitor, Kullu to Dharamshala:
Input: "I'm from Kullu, want to visit Dharamshala for 3 days next week with family"
Output:
{
  "destination": "Dharamshala",
  "origin": "Kullu",
  "duration_days": 3,
  "travel_dates": "next week",
  "group_type": "family",
  "group_size": null,
  "budget_range": null,
  "interests": [],
  "special_requirements": [],
  "visitor_type": "local",
  "return_journey_preference": null,
  "home_city": "Kullu",
  "needs_clarification": false,
  "clarification_question": null,
  "confidence": 0.95
}
"""


async def planner_node(state: AgentState) -> dict:
    """
    LangGraph Planner Node.
    Extracts structured trip entities from the full conversation history.
    Classifies visitor_type and determines outstation/local-specific clarification flow.
    """
    log.info("planner_node_entered", session_id=state.session_id)

    # Build conversation context for the model
    conversation_text = "\n".join([
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in state.messages[-10:]  # last 10 messages for context
    ])

    # Include previous entities to avoid re-asking
    existing_entities = state.entities.model_dump(exclude_none=True)

    # Pass visitor_type context so planner doesn't re-classify unnecessarily
    visitor_context = ""
    if state.visitor_type:
        visitor_context = f"\nPreviously classified visitor_type: {state.visitor_type} (do NOT change unless new info contradicts this)"

    prompt = f"""Conversation so far:
{conversation_text}

Previously extracted entities (already known, DO NOT ask again):
{json.dumps(existing_entities, indent=2)}{visitor_context}

Extract any NEW information from the latest message and update the entity JSON."""

    try:
        llm = get_vertex_ai_service().get_pro_model()
        response = await llm.ainvoke([
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])

        raw = response.content.strip()
        # Strip markdown code blocks if model wraps output
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw)

        # Merge with existing entities (don't overwrite confirmed values with null)
        merged = {**existing_entities}
        ignore_keys = ("needs_clarification", "clarification_question", "confidence", "booking_intent_active", "user_confirmed_booking", "booking_pickup", "booking_dropoff", "booking_time")
        for key, val in result.items():
            if key not in ignore_keys and val is not None:
                merged[key] = val

        entities = ExtractedEntities(**{
            k: merged.get(k)
            for k in ExtractedEntities.model_fields
        })

        needs_clarification = result.get("needs_clarification", False)
        clarification_question = result.get("clarification_question")

        # Determine visitor_type: prefer planner's classification, fall back to router's
        planner_visitor_type = result.get("visitor_type")
        visitor_type = planner_visitor_type or state.visitor_type or entities.visitor_type

        # Sync visitor_type into entities
        if visitor_type:
            entities.visitor_type = visitor_type

        log.info(
            "entities_extracted",
            destination=entities.destination,
            origin=entities.origin,
            duration=entities.duration_days,
            visitor_type=visitor_type,
            home_city=entities.home_city,
            needs_clarification=needs_clarification,
            session_id=state.session_id,
        )

        booking_intent_active = result.get("booking_intent_active", False)
        user_confirmed_booking = result.get("user_confirmed_booking", False)
        
        booking_details = {}
        if result.get("booking_pickup"): booking_details["pickup"] = result.get("booking_pickup")
        if result.get("booking_dropoff"): booking_details["dropoff"] = result.get("booking_dropoff")
        if result.get("booking_time"): booking_details["time"] = result.get("booking_time")

        return {
            "entities": entities,
            "needs_clarification": needs_clarification,
            "clarification_question": clarification_question,
            "visitor_type": visitor_type,
            "booking_intent_active": booking_intent_active,
            "user_confirmed_booking": user_confirmed_booking,
            "booking_details": booking_details,
            "node_trace": ["planner"],
        }

    except (json.JSONDecodeError, Exception) as e:
        log.error("planner_failed", error=str(e), session_id=state.session_id)
        return {
            "needs_clarification": True,
            "clarification_question": "आप अपनी यात्रा की जगह और कितने दिनों का प्लान है, बता सकते हैं?",
            "node_trace": ["planner"],
            "error": str(e),
        }
