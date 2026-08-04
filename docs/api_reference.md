# Him Cab AI Travel Assistant — API Reference

## Base URL

- **Development**: `http://localhost:8000/v1`
- **Production**: `https://himcab-ai-assistant-<hash>.run.app/v1`

## Authentication

All endpoints (except `/v1/health` and `/v1/alerts/himachal`) require a Firebase Auth JWT token:

```
Authorization: Bearer <firebase_id_token>
```

**Dev mode bypass**: Use `Authorization: Bearer dev-test-token` in development.

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
    "redis": "ok",
    "api": "ok"
  }
}
```

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

#### SSE Stream (Primary)
```
GET /v1/chat/stream?session_id=X&message=Y&language=en
```

Server-Sent Events stream. The primary endpoint consumed by the mobile app.

**SSE Event Types**:

| Event | Data | Description |
|-------|------|-------------|
| `token` | `{"token": "..."}` | Raw text chunk for chat animation |
| `structured` | `{"type": "itinerary", "payload": {...}}` | Structured JSON card |
| `structured` | `{"type": "route_plan", "payload": {...}}` | Route plan for outstation |
| `structured` | `{"type": "weather_alert", "payload": {...}}` | Weather alert card |
| `structured` | `{"type": "offers", "payload": {...}}` | Personalized offers |
| `done` | `{"session_id": "...", "tokens_used": 0}` | Stream complete |
| `error` | `{"code": "...", "message": "..."}` | Error occurred |

#### Sync Message (Fallback)
```
POST /v1/chat/message
```

Non-streaming fallback for clients that don't support SSE.

**Request Body**: `ChatRequest`
```json
{
  "session_id": "uuid",
  "message": "Plan a 3-day trip to Manali",
  "language": "en"
}
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
