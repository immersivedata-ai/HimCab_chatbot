# Him Cab AI Travel Assistant — System Architecture

## Overview

The Him Cab AI Travel Assistant is an enterprise-grade, LangGraph-powered conversational AI backend
that plans Himachal Pradesh trips, generates structured itineraries, and integrates with the
Him Cab rider app for end-to-end cab booking.

## High-Level Architecture

```mermaid
graph TB
    subgraph Client["Mobile App / Web UI"]
        APP["Him Cab Rider App"]
        WEB["Test Frontend"]
    end

    subgraph API["FastAPI Backend"]
        GW["API Gateway<br/>(FastAPI + Auth)"]
        SSE["SSE Stream<br/>(/v1/chat/stream)"]
        REST["REST Endpoints<br/>(/v1/session, /v1/alerts)"]
    end

    subgraph Agent["LangGraph Agent Pipeline"]
        R["Router Node"]
        CTX["Context Node"]
        P["Planner Node"]
        TE["Tool Executor"]
        W["Writer Node"]
        PS["Persist Node"]
        CL["Clarify Node"]
    end

    subgraph Services["Service Layer"]
        VAI["Vertex AI<br/>(Gemini Flash)"]
        FS["Firestore<br/>(Sessions + Users)"]
        WS["Weather Service<br/>(OpenWeatherMap)"]
        FARE["Him Cab Fare Service<br/>(Java Backend)"]
        OFF["Him Cab Offers<br/>(Java Backend)"]
        GEO["Geography<br/>(coords → local/visitor, distance)"]
    end

    APP --> GW
    WEB --> GW
    GW --> SSE
    GW --> REST
    SSE --> R
    R --> CTX --> P
    P -->|needs_clarification| CL --> PS
    P -->|ready| TE --> W --> PS
    TE --> WS
    TE --> GEO
    TE --> FARE
    TE --> OFF
    GW --> GEO
    R --> VAI
    P --> VAI
    W --> VAI
    PS --> FS
    CTX --> FS
```

## Data Flow

```mermaid
sequenceDiagram
    participant App as Mobile App
    participant API as FastAPI
    participant Agent as LangGraph
    participant LLM as Gemini
    participant Tools as External APIs
    participant DB as Firestore

    App->>API: POST /v1/session (create)
    API->>DB: Create TripSession
    API-->>App: session_id

    App->>API: GET /v1/chat/stream?session_id=X&message=Y
    API->>DB: Load session + history
    API->>Agent: Run graph with initial state

    Agent->>LLM: Router: classify intent + language
    Agent->>DB: Context: load user profile
    Agent->>LLM: Planner: extract entities

    alt Needs Clarification
        Agent-->>API: Clarification question
        API-->>App: SSE: token events
    else Ready to Plan
        Agent->>Tools: Parallel: weather + Him Cab fare (priced on the rider's GPS fix)
        Agent->>Tools: Him Cab offers, filtered to that fare
        Agent->>LLM: Writer: generate response + JSON
        Agent->>DB: Persist: save message + itinerary
        Agent-->>API: Final response + structured events
        API-->>App: SSE: token + structured + done events
    end
```

## Infrastructure (GCP)

| Component | GCP Service | Purpose |
|-----------|------------|---------|
| API Server | Cloud Run | Auto-scaling containerized backend |
| Database | Firestore | Session + user profile persistence — the only stateful dependency |
| AI Models | Vertex AI (Gemini) | LLM inference |
| Auth | Firebase Auth | JWT-based user authentication |
| Secrets | Secret Manager | API keys, credentials |
| Container Registry | Artifact Registry | Docker images |

## Key Design Decisions

1. **LangGraph over raw LangChain**: Provides stateful, graph-based agent execution with clear node boundaries and conditional routing.

2. **Flash everywhere**: routing and extraction are classification tasks and the writer is latency-critical, so all three use Gemini Flash. A Pro client was being constructed at startup and never called.

3. **Firestore only, no cache tier**: caching and rate limiting are in-process (`app/services/local_cache.py`). The cache held a conversation pointer that Firestore already mirrored, an alerts response, and a request counter — none worth an external dependency that could be down. The trade is that the rate limit is per instance; see `docs/api_reference.md`.

4. **Parallel tool execution**: external calls run concurrently via `asyncio.gather()` — total latency = slowest single call, not their sum.

5. **The conversation policy lives in code, not in a prompt**: `app/agent/slots.py` decides what to ask and when; the LLM only extracts what the user said. Letting the model own the flow produced questions in the wrong language, for details already given, and for details the request never needed.

6. **Live data overrides generated data**: the real fare, the real weather report and the measured journey are stamped onto the itinerary after validation, so a price, a forecast or a distance can never be something the model invented — or something it forgot to include. Coupons are the same rule stated the other way: the reply may quote only the codes the backend returned, and when there are none it is forbidden to mention discounts at all.

7. **Coordinates decide the persona, the conversation corrects it**: a GPS fix inside Himachal serves the local assistant and one outside serves the outstation assistant (`app/services/geo.py`, `POST /v1/chat/route`) — a better answer than which screen the chat bubble was tapped from. The fix also supplies the pickup point, so the fare is priced from where the rider actually is instead of a city centroid, and a rider who never types their city out still gets a price. Whatever they say about their origin still outranks it.

8. **SSE streaming**: Token-by-token streaming for chat UX, with structured JSON events for mobile card rendering.

9. **Prompt templates in YAML**: Prompts are the most iterated component — extracting them into versioned YAML templates enables prompt engineering without code changes.
