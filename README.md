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
│   │   ├── himcab/             # Him Cab backend clients (fare quotes, distances)
│   │   └── external/           # 3rd party API wrappers (weather)
│   ├── models/                 # Pydantic models (session, itinerary, user, alerts)
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
the token. Use the `local` routes for in-state riders, `outstation` for visitors.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/chat/{local\|outstation}/start` | Open the dialog → returns welcome greeting |
| `GET` | `/v1/chat/{local\|outstation}/stream` | SSE: send `?message=` , stream the reply |
| `POST` | `/v1/chat/{local\|outstation}/message` | Non-streaming fallback |
| `GET` | `/v1/chat/{local\|outstation}/history` | Replay the conversation to restore a chat panel |

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
data: {"type": "weather_alert", "payload": {"severity": "high"}}

event: done
data: {"session_id": "uuid", "node_trace": ["router", "context", "planner", "tool_executor", "writer", "persist"]}
```
