# Him Cab AI Travel Assistant — API Reference

## Base URL

- **Development**: `http://localhost:8000/v1`
- **Production**: `https://himcab-ai-assistant-<hash>.run.app/v1`

## Authentication

**One credential for the entire v1 API** — `/v1/chat/*`, `/v1/session/*` and
`/v1/user/*` all accept the same header:

```
Authorization: Bearer <him_cab_rider_jwt>
```

The same token the app already holds from its Him Cab cab-booking login. It does two
jobs: it identifies the caller (owning their chat session) and it is forwarded to the
Him Cab fare service to authorize real fare quotes. Nothing else is required — no
session id, no persona, no rider id.

> **Changed:** `/v1/session/*` and `/v1/user/*` previously required a *Firebase* ID
> token, so a token accepted by `/v1/chat/*` was rejected with 401 everywhere else.
> That inconsistency broke conversation restore, since `/history` is how a client
> rehydrates a chat it already owns. Both groups now use the rider JWT.

> ⚠️ The API decodes this token to read its claims but does **not** verify its
> signature (we don't hold Him Cab's signing key). The fare service is the real
> enforcement point. To make identity itself tamper-proof, we need the JWT public
> key/secret or a token-introspection endpoint from the Java team.

**Dev mode bypass**: `Authorization: Bearer dev-test-token` works in development
(rejected in production). On the chat API this yields a working conversation but no
real fares, since a dev token can't authorize a quote.

---

## Endpoints

### Health Check

```
GET /v1/health
```

No authentication required. Used for Cloud Run readiness/liveness probes.

**Response**:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 1234.5,
  "timestamp": "2026-08-03T12:00:00Z",
  "services": {
    "firestore": "ok",
    "api": "ok"
  },
  "conversation_memory": "ok"
}
```

`status` is one of:

| Status | Meaning | Chat usable? |
|---|---|---|
| `healthy` | Firestore reachable | Yes |
| `unhealthy` | Firestore unavailable | No — Firestore is the system of record for conversations |

Firestore is the only external dependency worth probing: caching and rate limiting
are in-process, so there is no cache tier that can be separately degraded.

Read `conversation_memory` rather than `status` if all you need to know is whether
multi-turn chat works.

---

### Session Management

#### Create Session
```
POST /v1/session
```

Create a new trip planning session before opening the chat.

**Request Body**:
```json
{
  "language": "en",
  "initial_message": "I want to visit Manali"
}
```

**Response**: `CreateSessionResponse`
```json
{
  "session_id": "uuid",
  "created_at": "2026-08-03T12:00:00Z",
  "status": "active"
}
```

#### Get Session
```
GET /v1/session/{session_id}
```

#### Get Conversation History
```
GET /v1/session/{session_id}/history?limit=50
```

For restoring a chat panel prefer `GET /v1/chat/{persona}/history`, which resolves the
conversation from the token — the app doesn't need to have kept a session id.

#### Get Latest Itinerary
```
GET /v1/session/{session_id}/itinerary
```

#### Delete Session
```
DELETE /v1/session/{session_id}
```

---

### Chat

This is the whole mobile integration. Every call carries one header —
`Authorization: Bearer <rider JWT>`. The backend owns the conversation and resolves it
from that token, so the app doesn't need to create, store, or send a session id (it
may optionally pin one — see [Pinning a conversation](#pinning-a-conversation)).

Pick the route group that matches the rider:

| Rider | Route prefix | Default language | Reply shape |
|-------|--------------|------------------|-------------|
| Inside Himachal | `/v1/chat/local` | Hindi | Cab, fare, drive time |
| Travelling in | `/v1/chat/outstation` | English | Day-by-day itinerary |

Don't guess which — [ask](#0-which-assistant-route-by-location), or just send the
device's coordinates and let the backend route itself.

#### 0. Which assistant? Route by location
```
POST /v1/chat/route
```

Send the device's GPS fix; use the `persona` that comes back for every
`/v1/chat/{persona}/*` call afterwards. A fix inside Himachal Pradesh is a **local**
rider; anywhere else is **outstation**.

**Request Body**: `RiderCoordinates`
```json
{ "lat": 31.1048, "lon": 77.1734, "accuracy_m": 25 }
```

**Response**: `PersonaRouteResponse`
```json
{
  "persona": "local",
  "base_path": "/v1/chat/local",
  "reason": "coordinates",
  "location": {
    "lat": 31.1048, "lon": 77.1734,
    "in_himachal": true,
    "place_name": "Shimla",
    "place_distance_km": 0.0,
    "near_border": false
  }
}
```

Nothing is created here — no session, no message, no state. It is a pure lookup, safe
to call on app start or on a location change. `reason` is `"coordinates"` when the fix
decided it and `"default"` when no usable fix was sent.

`near_border` is true within a few km of the state line, where the call could go either
way; treat the persona as a default the rider may correct. The rider's own words always
win over their coordinates — someone who says *"I'm coming from Delhi"* is served as a
visitor whatever their phone reports.

Skipping this endpoint is fine: `/start`, `/stream` and `/message` all accept the same
coordinates and route themselves, returning the persona they actually used.

#### 1. Open the chat dialog
```
POST /v1/chat/{local|outstation}/start?lat=31.1048&lon=77.1734
```

Call this when the chat UI opens. Starts a **fresh** conversation and returns the
welcome greeting — render `message` as the first assistant bubble.

Query params (all optional): `lat`, `lon`, `accuracy_m`.

**Response**: `StartChatResponse`
```json
{
  "session_id": "uuid",
  "message": "नमस्ते! Him Cab में आपका स्वागत है 🙏 …",
  "language": "hi",
  "persona": "local",
  "location": { "lat": 31.1048, "lon": 77.1734, "in_himachal": true, "place_name": "Shimla", "place_distance_km": 0.0, "near_border": false }
}
```

**`persona` is authoritative.** It matches the path you called *unless* you sent
coordinates that say otherwise — a fix outside Himachal on `/local/start` is served as
`outstation`, and this field tells you so. Use it for the rest of the conversation.
`location` is `null` when you sent no usable coordinates.

Keep `session_id` if you want to pin this conversation explicitly (see
[Pinning a conversation](#pinning-a-conversation)); otherwise the backend tracks it
for you and the app never needs to send it back.

#### 2a. Send a message — streaming (preferred)
```
GET /v1/chat/{local|outstation}/stream?message=<text>&lat=31.1048&lon=77.1734
```

Server-Sent Events stream. Note this needs `fetch` (or any client that can set
headers), not the browser's native `EventSource`, which cannot send an
`Authorization` header.

**SSE Event Types**:

| Event | Data | Description |
|-------|------|-------------|
| `token` | `{"token": "..."}` | Raw text chunk for chat animation |
| `structured` | `{"type": "<see below>", "payload": {...}}` | A renderable card |
| `done` | `{"session_id": "...", "persona": "local", "node_trace": [...]}` | Stream complete |
| `error` | `{"code": "...", "message": "..."}` | Error occurred |

Query params: `message` (required), the optional `session_id` described under
[Pinning a conversation](#pinning-a-conversation), and the optional `lat` / `lon` /
`accuracy_m` described under [Sending coordinates](#sending-coordinates).

#### 2b. Send a message — non-streaming (fallback)
```
POST /v1/chat/{local|outstation}/message
```

**Request Body**: `ChatMessageRequest`
```json
{ "message": "Plan a 3-day trip to Manali", "lat": 28.6139, "lon": 77.2090 }
```

`lat`, `lon` and `accuracy_m` are optional — see
[Sending coordinates](#sending-coordinates).

**Response**: `ChatResponse`
```json
{
  "session_id": "uuid",
  "response": "Here's a 3-day plan for Manali…",
  "persona": "outstation",
  "structured_events": [
    { "type": "itinerary", "payload": { "destination": "Manali", "...": "..." } }
  ],
  "itinerary": { "destination": "Manali", "...": "..." },
  "node_trace": ["router", "planner", "tool_executor", "writer"]
}
```

`itinerary` and the `itinerary` entry in `structured_events` are **the same object** —
read whichever is convenient. `itinerary` is `null` on turns that didn't produce one
(clarification turns, weather questions).

> **Fixed:** `itinerary` was previously always `null`, even on turns that generated a
> full plan. Itinerary validation was failing — the model was putting the persona in
> `group_type` and leaving `budget_range` null — so only the unvalidated raw dict in
> `structured_events` survived. Validation is now tolerant, trip facts are taken from
> the confirmed conversation state, and the two channels are guaranteed identical.

#### 3. Restore a chat panel
```
GET /v1/chat/{local|outstation}/history?limit=50
```

Replays the caller's current conversation so the app can rehydrate its UI on reopen.
Resolves the same conversation `/message` would, from the token — the client never
needs to have stored a session id. Use this instead of `/start` when reopening a panel
the user was already using; `/start` deliberately begins a fresh conversation.

**Response**: `ChatHistoryResponse`
```json
{
  "session_id": "uuid",
  "persona": "local",
  "language": "hi",
  "messages": [
    { "role": "assistant", "content": "नमस्ते! …", "timestamp": "2026-08-07T09:00:00Z" },
    { "role": "user", "content": "Manali", "timestamp": "2026-08-07T09:00:12Z" }
  ],
  "total_messages": 2
}
```

An empty `messages` list means the rider has no conversation yet — not an error.

#### Structured event types

`structured_events[]` (and the SSE `structured` event) carry zero or more of these.
This list is exhaustive, but **ignore unrecognised types** so new ones can ship
without breaking older app builds.

| `type` | `payload` | Emitted when |
|---|---|---|
| `itinerary` | Full `Itinerary` — `destination`, `duration_days`, `group_type`, `budget_range`, `days[]`, `cab_package`, `hotels[]`, `weather`, `travel_info`, `cost_breakdown`, `hidden_gems[]`, `generated_at`, `language`. `weather` and `cab_package` are filled from the live services, not from the model. | A trip plan was generated |
| `weather_alert` | One alert — `event`, `severity`, `description` | Destination has an active weather alert |
| `snowfall_alert` | `risk` (`high`\|`critical`), `message` | Snowfall risk at the destination |
| `cab_fare` | `{options[], category_name, fare_inr, vehicle_label, seat_capacity, trip_type, distance_km, duration_text, distance_source, is_approximate, source, best_offer, fare_after_offer_inr, selected_by_rider}`. On a rental, `options[]` is the **Select Rental Cab** list: render it as a radio group and send the rider's pick back as its `category_name`. `selected_by_rider` is true once they have picked one, and then `fare_inr` / `category_name` are THEIR cab rather than our recommendation. When `is_approximate` is true the route distance was computed rather than surveyed — render the price as "approx". `distance_source` is `surveyed` \| `gps` \| `city`. `best_offer` / `fare_after_offer_inr` are present only when a live coupon applies. | A fare was quoted for the route |
| `offers` | `{trip_type, fare_inr, offers: [{offer_id, coupon_code, coupon_name, coupon_value_inr, minimum_value_inr, fare_after_offer_inr, applies_to_this_fare, terms_and_conditions, image, expire_date}], best_offer}`. Real coupons from `/api/rider/offer/*`, already filtered to this trip type, today's date and the quoted fare, **best saving first**. | The rider has at least one coupon that applies to this trip |
| `rental_packages` | `{packages: [{hours, included_km, label, from_rate_card}], selected_hours}`. The hourly blocks the rider may choose from, shortest first; `selected_hours` is present once they have chosen one. **Render these as taps.** The backend matches a rental duration exactly (`rentHours == chart.getMaxHour()`), so a typed duration that is not on this list cannot be priced. The list is the rate card narrowed to the durations the product sells (`RENTAL_HOUR_OPTIONS`, currently 4 and 8 hours) — a chart for any other duration is priced by the backend but never offered here, so do not render one. When the rate card cannot be read at all, those same durations are sent with `from_rate_card: false` and `included_km: null`. Offer these as taps like any other — a `cab_fare` can still arrive for them, because the backend matches the rental chart from `rentHours` itself — but show no distance allowance beside one. Sent on the turn that ASKS for the duration and on every rental turn after it, so the list on screen is never stale. | `/v1/chat/rental/*` only |
| `request_cancelled` | `{scope, dismiss_booking_screen, cleared_fields[], open_screen}`. The rider called the request off. **Act on this**: dismiss any booking sheet a previous `booking_handoff` opened and clear the fare/package cards for this conversation — what they describe no longer exists. `scope` is `request` (this chat's request, now dropped), `booked_trip` (they mean a cab already booked in the app) or `unclear` (either; the reply covers both). `open_screen` is `trips` when they need the app's own cancellation flow, else null. **Nothing is cancelled on the backend by this event** — a confirmed booking is cancelled only from the Trips screen (`DELETE /api/rider/trip/cancel-ride/{id}`), which is what notifies the driver. | The rider said cancel / never mind / rehne do |
| `booking_handoff` | `{origin: {name, lat, lon}, destination: {name, lat, lon}, distance_km, duration_minutes, duration_text, is_approximate, distance_source, trip_type, passengers, travel_dates, recommended_category_id, recommended_category_name, fare_inr, offer_id, coupon_code, fare_after_offer_inr, options[]}`. On a rental it also carries `rental_hours`, `included_km`, `pickup_time` (`"HH:mm"`, the hour the fare was priced on — open the screen on exactly this time) and `category_selected_by_rider`, with `destination: null` since an hourly hire has no drop. A rental handoff is sent only once the rider has settled a duration AND picked a cab (or there was no priced list to pick from), because the Rent a Car screen cannot book without both — so the reply never announces a screen while still asking which cab. Open the booking screen with both pins already dropped on the map, the recommended cab preselected and the coupon prefilled (`offer_id` is exactly what `POST /api/rider/trip/book-ride` takes). `trip_type` (`OUTSTATION` \| `LOCAL`, decided by the journey's length rather than by the persona) is the rate card it was priced on — use it to open the booking screen that will accept the journey. Nothing is reserved — this is a handoff, and the assistant never asks the rider to confirm a booking. | A travelling rider's route was resolved — either persona |

The `offers` event is **absent**, never empty, when nothing applies. The reply text and
this event agree by construction: the assistant may name only the coupons listed here,
and when the list is empty it is forbidden to mention discounts at all.

Two fields carry the honesty guarantees, and the app should respect both:

- **`applies_to_this_fare`** — false when no fare came back for the route, so the
  minimum-spend check could not run. The coupon is live on the rider's account, but
  eligibility for *this* trip is unconfirmed; the reply says so, and `booking_handoff`
  does **not** prefill it.
- **`fare_after_offer_inr`** — null whenever a total cannot be stated honestly: no fare
  to subtract from, or a coupon worth at least the whole fare (which would render as
  "₹0", a free ride the booking flow will not honour). Show the saving, not a total.

Enum vocabularies inside an `itinerary` payload:

- `group_type`: `solo` \| `couple` \| `family` \| `friends` \| `corporate` — the
  travelling party, **never** the persona
- `budget_range`: `budget` \| `moderate` \| `luxury`
- `activities[].type`: `sightseeing` \| `pickup` \| `drop` \| `food` \| `adventure` \|
  `spiritual` \| `shopping` \| `rest` \| `travel` \| `accommodation` \| `leisure` \|
  `trekking` \| `dining` \| `other`
- `crowd_level`: `low` \| `moderate` \| `high` \| `very_high`

#### Session handling (what the backend does for you)

- `/start` **always** begins a new conversation, so reopening the dialog never drops
  the user into a stale half-finished plan.
- `/stream` and `/message` reuse the caller's current conversation, resolved from the
  JWT. If there isn't one — app restarted, cache expired, or `/start` was never
  called — one is created on the spot.
- Because of that, these endpoints **never** return a "session not found / expired"
  error the app has to recover from.

The "which conversation is this rider in" pointer is stored in Firestore as a direct
document read requiring no index, so multi-turn memory does not depend on any cache
or on a composite index being present.

#### Sending coordinates

Optional on every chat endpoint, and worth sending on all of them. `lat` / `lon` /
`accuracy_m` are query params on `/start` and `/stream`, body fields on `/message`, and
the whole body of `/route`.

What they buy:

| Without coordinates | With coordinates |
|---|---|
| The screen the chat was opened from picks the persona | The rider's actual position picks it |
| The assistant asks "which city are you travelling from?" | The pickup city is already known |
| No fare until they type that city out | Fare quoted on the first message |
| Distance measured between two city centroids | Distance measured from where they are |

Resend them on every turn — riders move between messages, and the newest fix wins. The
resolved fix is stored on the conversation, so a client that sends them once still gets
the benefit on later turns; it will just be pricing from a stale pickup point.

Rules the backend applies:

- **`(0, 0)` is rejected.** It is what a phone reports before it has a fix, and it is in
  the Atlantic. A rider in Shimla must not be classified as a visitor 6,000 km away.
- **A vague fix is not turned into a place name.** The pickup city is filled in only
  when the fix sits within ~15 km of a town we know; further out, the coordinates still
  measure the distance but the assistant asks which city they mean.
- **Their words beat their phone.** Once the rider names a starting city, that is the
  origin — booking a Delhi pickup from a Shimla sofa is a real thing people do.

#### Pinning a conversation

Both `/stream` (query param) and `/message` (body field) accept an optional
`session_id`.

- **Omit it** for a single device. The token resolves the conversation; this is the
  normal case and needs no client state.
- **Pass the id from `/start`** when one account may be signed in on several devices
  and you don't want them sharing one conversation — without it, sessions are keyed on
  the rider alone and both devices talk into the same chat.

An unknown id, or one belonging to another rider, is **ignored** — resolution falls
through to the caller's own conversation. It never returns 4xx, so a stale id on the
client can't wedge the UI, and a guessed id can't reach someone else's chat.

#### Example

```bash
TOKEN="<rider jwt>"

# 0. Which assistant does this rider belong to?
curl -X POST http://localhost:8000/v1/chat/route \
     -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"lat": 31.1048, "lon": 77.1734}'
# → {"persona":"local","base_path":"/v1/chat/local", ...}

# 1. Open the dialog on that persona, with the same fix.
curl -X POST "http://localhost:8000/v1/chat/local/start?lat=31.1048&lon=77.1734" \
     -H "Authorization: Bearer $TOKEN"

# 2. Send a message. The fare, distance and any coupon come back with the reply.
curl -N "http://localhost:8000/v1/chat/local/stream?message=Manali%20jaana%20hai&lat=31.1048&lon=77.1734" \
     -H "Authorization: Bearer $TOKEN"
```

---

### Alerts

#### Regional Alerts (Public)
```
GET /v1/alerts/himachal
```

No authentication required. Returns aggregated weather/snowfall alerts for the region.

#### Destination Alerts
```
GET /v1/alerts/{destination}
```

---

### User Profile

#### Get Profile
```
GET /v1/user/profile
```

#### Update Preferences
```
PUT /v1/user/profile
```

**Request Body**: `TravelPreferences`
```json
{
  "preferred_budget": "moderate",
  "preferred_group_type": "family",
  "interests": ["temples", "trekking"],
  "dietary_restrictions": ["vegetarian"],
  "preferred_language": "hi"
}
```

#### Past Trips
```
GET /v1/user/trips?limit=10
```

---

## Error Response Format

All errors follow a consistent structure:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable message",
    "details": null
  }
}
```

**Error Codes**:
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid or expired JWT |
| `FORBIDDEN` | 403 | User doesn't own the resource |
| `SESSION_NOT_FOUND` | 404 | Invalid session_id |
| `SESSION_EXPIRED` | 410 | Session TTL exceeded |
| `VALIDATION_ERROR` | 422 | Bad request body |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `AGENT_EXECUTION_FAILED` | 500 | AI agent error |
| `EXTERNAL_SERVICE_ERROR` | 502 | Weather/Places API failure |

---

## Rate limiting

**30 requests per minute per rider**, counted on `/v1/chat/*/message` and
`/v1/chat/*/stream`. The window is a fixed 60 seconds.

Over the limit, both endpoints return **429** with a `Retry-After` header:

```
HTTP/1.1 429 Too Many Requests
Retry-After: 60
```
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded: 31/30 requests per minute. Please wait a moment before sending another message.",
    "details": { "retry_after": 60, "limit": 30 }
  }
}
```

Client contract: honour `Retry-After` (seconds) and retry — a 429 is back-pressure,
not a fatal error. Don't clear the chat or sign the user out.

Rate limiting is enforced **per server instance**, so the effective ceiling for a
rider is the configured limit times the number of instances behind the load balancer.
Don't rely on 429 as a correctness boundary — treat it purely as back-pressure.

### Latency expectations

| Turn type | Typical |
|---|---|
| Clarification (`clarify` in `node_trace`) | ~1–2 s |
| Full itinerary (`writer` in `node_trace`) | ~5–15 s |

Use `/stream` where possible — first token arrives well before the full reply. On
`/message`, set a client read timeout of at least 60 s; the server caps any single
LLM call at 45 s.
