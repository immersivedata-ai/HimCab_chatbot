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

Pick the route group that matches the app:

| App | Route prefix | Default language |
|-----|--------------|------------------|
| Local / in-state riders | `/v1/chat/local` | Hindi |
| Outstation visitors | `/v1/chat/outstation` | English |

#### 1. Open the chat dialog
```
POST /v1/chat/{local|outstation}/start
```

Call this when the chat UI opens. Starts a **fresh** conversation and returns the
welcome greeting — render `message` as the first assistant bubble.

**Response**: `StartChatResponse`
```json
{
  "session_id": "uuid",
  "message": "Welcome to Him Cab! 🙏 I can help you plan your trip to Himachal Pradesh. Where would you like to go?",
  "language": "en",
  "persona": "outstation"
}
```

Keep `session_id` if you want to pin this conversation explicitly (see
[Pinning a conversation](#pinning-a-conversation)); otherwise the backend tracks it
for you and the app never needs to send it back.

#### 2a. Send a message — streaming (preferred)
```
GET /v1/chat/{local|outstation}/stream?message=<text>
```

Server-Sent Events stream. Note this needs `fetch` (or any client that can set
headers), not the browser's native `EventSource`, which cannot send an
`Authorization` header.

**SSE Event Types**:

| Event | Data | Description |
|-------|------|-------------|
| `token` | `{"token": "..."}` | Raw text chunk for chat animation |
| `structured` | `{"type": "<see below>", "payload": {...}}` | A renderable card |
| `done` | `{"session_id": "...", "node_trace": [...]}` | Stream complete |
| `error` | `{"code": "...", "message": "..."}` | Error occurred |

Query params: `message` (required) and the optional `session_id` described under
[Pinning a conversation](#pinning-a-conversation).

#### 2b. Send a message — non-streaming (fallback)
```
POST /v1/chat/{local|outstation}/message
```

**Request Body**: `ChatMessageRequest`
```json
{ "message": "Plan a 3-day trip to Manali" }
```

**Response**: `ChatResponse`
```json
{
  "session_id": "uuid",
  "response": "Here's a 3-day plan for Manali…",
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
| `cab_fare` | `{options[], category_name, fare_inr, vehicle_label, seat_capacity, trip_type, distance_km, is_approximate, source}`. When `is_approximate` is true the route distance was computed rather than surveyed — render the price as "approx". | A fare was quoted for the route |

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

curl -X POST http://localhost:8000/v1/chat/local/start \
     -H "Authorization: Bearer $TOKEN"

curl -N "http://localhost:8000/v1/chat/local/stream?message=Manali%20jaana%20hai" \
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
