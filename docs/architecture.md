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
        VAI["Vertex AI<br/>(Gemini Pro + Flash)"]
        FS["Firestore<br/>(Sessions + Users)"]
        RC["Redis Cache<br/>(Hot Data)"]
        WS["Weather Service<br/>(OpenWeatherMap)"]
        PS2["Places Service<br/>(Overpass API)"]
        OF["Offers Engine<br/>(Local Compute)"]
        BK["Booking Service<br/>(Java Backend)"]
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
    TE --> PS2
    TE --> OF
    TE --> BK
    R --> VAI
    P --> VAI
    W --> VAI
    PS --> FS
    PS --> RC
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
        Agent->>Tools: Parallel: weather, places, offers
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
| Database | Firestore | Session + user profile persistence |
| Cache | Memorystore (Redis) | Hot cache, rate limiting |
| AI Models | Vertex AI (Gemini) | LLM inference |
| Auth | Firebase Auth | JWT-based user authentication |
| Secrets | Secret Manager | API keys, credentials |
| Container Registry | Artifact Registry | Docker images |
| Scheduler | Cloud Scheduler | Alert refresh jobs |

## Key Design Decisions

1. **LangGraph over raw LangChain**: Provides stateful, graph-based agent execution with clear node boundaries and conditional routing.

2. **Gemini Pro for planning, Flash for writing**: Pro handles complex JSON extraction (planner), Flash handles fast response generation (writer, router).

3. **Redis as hot cache, Firestore as cold store**: Redis for sub-second reads (weather, places, sessions), Firestore for durable persistence.

4. **Parallel tool execution**: All external API calls (`weather`, `places`, `offers`) run concurrently via `asyncio.gather()` — total latency = slowest single call.

5. **SSE streaming**: Token-by-token streaming for chat UX, with structured JSON events for mobile card rendering.

6. **Prompt templates in YAML**: Prompts are the most iterated component — extracting them into versioned YAML templates enables prompt engineering without code changes.
