# Him Cab — AI Tourist Travel Assistant (Backend)

Enterprise-grade AI travel concierge backend powering the Him Cab rider app.

## Stack
- **Runtime**: Python 3.11 + FastAPI
- **AI Orchestration**: LangGraph (stateful multi-node agent)
- **LLM**: Vertex AI — Gemini Flash (routing, extraction, writing/streaming)
- **Memory**: Cloud Firestore — the only stateful dependency, and the system of record
  for conversations. Caching and rate limiting are in-process
  (`app/services/local_cache.py`), so there is no cache server to run or lose.
- **Auth**: Him Cab rider JWT, across the whole v1 API
- **Deploy**: Cloud Run (auto-scaling)
- **Streaming**: Server-Sent Events (SSE)

## Project Structure
```
travel_assistant/
├── app/
│   ├── agent/                  # LangGraph agent (nodes, graph, state)
│   │   ├── nodes/              # Individual graph nodes (planner, router, writer, ...)
│   │   ├── graph.py            # Graph assembly & compilation
│   │   ├── slots.py            # What the agent asks for, and when (conversation policy)
│   │   ├── language.py         # Which language the reply is written in
│   │   └── state.py            # AgentState schema
│   ├── api/
│   │   ├── v1/
│   │   │   ├── routes/         # FastAPI route handlers
│   │   │   └── router.py
│   │   └── middleware/         # Auth, rate-limit, logging middleware
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── logging.py          # Structured logging
│   │   └── exceptions.py       # Custom exception classes
│   ├── services/
│   │   ├── chat_session.py     # Which conversation a rider is in
│   │   ├── firestore.py        # Firestore session & profile service
│   │   ├── local_cache.py      # In-process TTL cache + rate limiter
│   │   ├── vertex_ai.py        # Vertex AI / Gemini client
│   │   ├── geo.py              # Coordinates → local/visitor, nearest town, distances
│   │   ├── himcab/             # Him Cab backend clients (fares, offers, distances)
│   │   └── external/           # 3rd party API wrappers (weather)
│   ├── models/                 # Pydantic models (session, itinerary, user, alerts, location)
│   ├── prompts/                # Prompt templates & builder
│   └── main.py                 # FastAPI app entrypoint
├── tests/
│   └── unit/
├── secrets/                     # GCP/Firebase credential JSONs (gitignored, not committed)
├── deploy/
│   └── cloudrun.yaml
├── docs/                       # Architecture docs & reference material
├── scripts/                    # Dev setup & data seeding scripts
├── frontend/                   # Static demo/test HTML clients
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── pyproject.toml
```

## Quick Start

```bash
# 1. Clone and create venv
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env  # Fill in your API keys
mkdir -p secrets       # Place your GCP service-account & Firebase admin SDK JSON keys here

# 4. Run locally
uvicorn app.main:app --reload --port 8000

# 5. Run with Docker
docker-compose up
```

## API Endpoints

### Chat — the mobile integration

Auth is a single header, `Authorization: Bearer <him cab rider jwt>`. No session id,
persona, or rider id is ever sent: the backend owns the session and resolves it from
the token.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/route` | Send the device's GPS fix → which persona to use |
| `POST` | `/v1/chat/{local\|outstation}/start` | Open the dialog → returns welcome greeting |
| `GET` | `/v1/chat/{local\|outstation}/stream` | SSE: send `?message=` , stream the reply |
| `POST` | `/v1/chat/{local\|outstation}/message` | Non-streaming fallback |
| `GET` | `/v1/chat/{local\|outstation}/history` | Replay the conversation to restore a chat panel |

**Which persona?** Don't guess from the screen the rider tapped from — send coordinates.
A fix inside Himachal gets the `local` assistant (Hindi, cab-first: they know the
place, they want a ride); anywhere else gets `outstation` (English, itinerary-first).
`/start`, `/stream` and `/message` all accept `lat`/`lon` too and route themselves,
returning the persona they actually served.

Coordinates are optional everywhere and worth sending everywhere — they are what let
the assistant skip *"are you local or visiting?"* and *"which city are you starting
from?"*, and price the trip from where the rider actually is rather than a city
centroid. Send nothing and everything still works; the assistant just asks.

See [docs/api_reference.md](docs/api_reference.md) for the full contract.

### Other endpoints

The same rider JWT authenticates these — there is one credential for the whole v1 API.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/session` | Create trip planning session |
| `GET` | `/v1/session/{id}` | Get session details |
| `GET` | `/v1/session/{id}/history` | Get conversation history |
| `GET` | `/v1/session/{id}/itinerary` | Get latest itinerary |
| `GET` | `/v1/alerts/himachal` | Live weather/snowfall alerts |
| `GET` | `/v1/user/trips` | User's past trips |
| `GET` | `/v1/health` | Health check |

## Streaming Contract (SSE)

```
event: token
data: {"token": "Based on your 3-day trip..."}

event: structured
data: {"type": "itinerary", "payload": {...}}

event: structured
data: {"type": "cab_fare", "payload": {"fare_inr": 5100, "distance_km": 270, ...}}

event: structured
data: {"type": "offers", "payload": {"best_offer": {"coupon_code": "HIMCAB300", ...}}}

event: structured
data: {"type": "weather_alert", "payload": {"severity": "high"}}

event: done
data: {"session_id": "uuid", "persona": "local", "node_trace": ["router", "context", "planner", "tool_executor", "writer", "persist"]}
```

Fares and coupons are the real ones from the Him Cab Spring backend
(`/api/rider/fare/estimate`, `/api/rider/offer/*`), priced on the route distance and
filtered to what this rider can actually use. The assistant may quote only what those
endpoints returned — with no live coupon it is forbidden to mention discounts at all.
