# Him Cab — AI Tourist Travel Assistant (Backend)

Enterprise-grade AI travel concierge backend powering the Him Cab rider app.

## Stack
- **Runtime**: Python 3.11 + FastAPI
- **AI Orchestration**: LangGraph (stateful multi-node agent)
- **LLM**: Vertex AI — Gemini 1.5 Pro (planning) + Gemini 1.5 Flash (writing/streaming)
- **Memory**: Cloud Firestore (persistent) + Redis/Memorystore (hot cache)
- **Auth**: Firebase Auth (JWT)
- **Deploy**: Cloud Run (auto-scaling)
- **Streaming**: Server-Sent Events (SSE)

## Project Structure
```
travel_assistant/
├── app/
│   ├── agent/              # LangGraph agent (nodes, graph, state)
│   │   ├── nodes/          # Individual graph nodes
│   │   ├── tools/          # Gemini function-calling tools
│   │   ├── graph.py        # Graph assembly & compilation
│   │   └── state.py        # AgentState schema
│   ├── api/
│   │   ├── v1/
│   │   │   ├── routes/     # FastAPI route handlers
│   │   │   └── router.py
│   │   └── middleware/     # Auth, rate-limit, logging middleware
│   ├── core/
│   │   ├── config.py       # Settings (pydantic-settings)
│   │   ├── logging.py      # Structured logging
│   │   └── exceptions.py   # Custom exception classes
│   ├── services/
│   │   ├── firestore.py    # Firestore session & profile service
│   │   ├── redis_cache.py  # Redis cache service
│   │   ├── vertex_ai.py    # Vertex AI / Gemini client
│   │   └── external/       # 3rd party API wrappers
│   ├── models/
│   │   ├── session.py      # Session / conversation models
│   │   ├── itinerary.py    # Itinerary structured output models
│   │   ├── user.py         # User profile models
│   │   └── alerts.py       # Weather / snowfall alert models
│   └── main.py             # FastAPI app entrypoint
├── workers/
│   └── alert_refresher.py  # Cloud Run Job: snowfall/weather cache warmer
├── tests/
│   ├── unit/
│   └── integration/
├── deploy/
│   ├── cloudrun.yaml
│   └── pubsub_config.yaml
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
cp .env.example .env  # Fill in your GCP credentials & API keys

# 4. Run locally
uvicorn app.main:app --reload --port 8000

# 5. Run with Docker
docker-compose up
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/v1/session` | Create trip planning session |
| `GET` | `/v1/session/{id}` | Get session details |
| `GET` | `/v1/session/{id}/history` | Get conversation history |
| `GET` | `/v1/session/{id}/itinerary` | Get latest itinerary |
| `GET` | `/v1/chat/stream` | SSE: Stream AI response |
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
data: {"session_id": "uuid", "tokens_used": 1240}
```
