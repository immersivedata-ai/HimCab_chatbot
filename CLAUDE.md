# CLAUDE.md — working agreement for this repository

Him Cab's AI travel assistant. A LangGraph agent behind a FastAPI service, serving
three conversation flows to riders in Himachal Pradesh, in Hindi, Hinglish and English.

This file exists because the same bugs kept coming back in new clothes. Every rule
below is here because breaking it produced a specific, reported failure — those are
named, because a rule with its reason attached survives and a bare rule does not.

**Read this before changing anything in `app/agent/`.**

---

## 1. What the system is

Three flows, one graph, one set of nodes:

| flow | who | endpoint | what a good turn ends in |
|---|---|---|---|
| `outstation` | a visitor planning a trip | `/v1/chat/outstation/*` | a day-by-day itinerary |
| `local` | a Himachal resident booking a ride | `/v1/chat/local/*` | a route, a fare, a booking screen |
| `rental` | anyone hiring a car by the hour | `/v1/chat/rental/*` | a package, a fare, a booking screen |

The flow is chosen by which page the rider opened, seeded into `visitor_type`, and is
**not** a guess the model gets to revise. A rental is a service someone picked, not a
kind of person — see `classify_visitor_type`.

### The graph

```
router → planner → ┬→ clarify      (needs a detail; asks ONE question)
                   ├→ cancel       (the rider called it off)
                   └→ tool_executor → writer → persist
```

- **router** — intent + language. One LLM call.
- **planner** — merges what is known, decides what is still needed. The LLM here
  EXTRACTS; the node DECIDES.
- **tool_executor** — geocoding, routing, weather, fare, offers. All gated.
- **writer** — prose + structured events. One LLM call.
- **persist** — session to Firestore.

---

## 2. The division of labour, which is the whole design

> **The LLM reads. Code decides.**

The model is never asked *whether* to ask a question, *what* to ask, or what the rules
are. It once was, and it produced questions in the wrong language, questions for details
the rider had already given, and questions the request never needed.

Policy lives in `app/agent/slots.py` — deterministic, testable, no model call.

**Corollary, and it is not optional:** every rider-facing decision must work with Vertex
AI unavailable. `slots`, `sanitize`, `cancellation`, `modification` and `ride_reply` are
all tables and comparisons, with tests asserting they make no model call. A rider is
waiting on every one of these.

---

## 3. The LLM boundary — read this before touching any structured output

This is where the worst bug in the project's history lived, undetected, for a long time.

### Never put `Optional[...]` in a schema sent to Vertex

`Optional[str]` compiles to `anyOf: [{"type":"string"},{"type":"null"}]`. Vertex's
`Schema` proto has no `anyOf`:

- in `json_mode` it raises `ValueError: Unknown field for Schema: anyOf`
- in the default `function_calling` mode **it says nothing at all** — the schema is
  mangled, the model returns no tool call, and `with_structured_output` returns `None`

Measured on one real conversation, 25 calls per configuration:

| schema | method | failures |
|---|---|---|
| `Optional` fields | function_calling | 12–19 / 25 |
| flat scalars | function_calling | 19 / 25 |
| **flat scalars** | **`json_mode`** | **0 / 25** |

Nested definitions (`$defs`, which pydantic emits for `Enum` fields) are dropped the
same way, with a warning nobody reads.

### And never use `""` as an enum member

A string field may default to `""`. An **enum member may not be empty**:

```
400 ...response_schema.properties[origin_in_himachal].enum[0]: cannot be empty
```

The first version of the wire model used `""` as the sentinel inside `Literal` too.
Vertex rejected the whole schema, LangChain retried it, and a four-turn conversation
logged **39-45 of those 400s — about 6.5 wasted requests per extraction that got
through**. On shared quota that is not just waste, it is 6x the volume into a pool that
answers 429 when it is busy. Enum sentinels are the word `"unknown"`, mapped back by
`_WIRE_BLANKS`.

**So:**

- the wire model is flat scalars with sentinel defaults — `""` and `0` for strings and
  counts, the word `"unknown"` for enums — see `planner._VertexExtraction`
- it is called with `method="json_mode"`
- a `to_extraction()` converts sentinels back to `None` so the rest of the code is
  unaffected
- `Literal[...]`, never `Enum`, on the wire — with a test asserting the literals still
  match the domain enum values
- keep the wire model's **docstring short**: pydantic ships it to Vertex as the schema
  description, so an essay there is re-sent on every call

`tests/test_extraction_boundary.py` asserts all of this against the schema, so it fails
in CI rather than in production.

### A `None` from `with_structured_output` is not an exception

It is the model declining to answer in the required shape. Retry it
(`_EXTRACTION_ATTEMPTS`), log it, then degrade. It used to reach `dict(None)` and
surface as `TypeError: 'NoneType' object is not iterable`, which reads like a bug in the
planner and sent everyone looking in the wrong file.

### Why this mattered so much

The planner degrades gracefully when extraction fails — it falls back to the
deterministic readers in `slots`. Those handle **short answers** well and **whole
sentences** only partly. So with extraction dead half the time, the assistant looked
fine when a rider typed `Kasol` and lost the plot when they typed
`mei kasol jana chahta hu`, then filled the silence with invented details.

It was reported as *"the AI is hallucinating and it's worse when I type in sentences"*.
It was an empty extraction every other turn.

**Graceful degradation hides failures. Anything that degrades must also be counted and
alarmed, or it will degrade forever in silence.**

---

## 3b. Vertex runs on shared quota — do not amplify a 429

Gemini on Vertex uses **Dynamic Shared Quota**. A 429 `ResourceExhausted` means the
shared pool is busy. **You cannot request a quota increase for a DSQ model** — that is
the documented answer, not a workaround. The levers are: fewer requests, spread out;
Provisioned Throughput; or waiting.

Production hit sustained 429s. The arithmetic: 3 model calls per turn, up to 800
concurrent turns (10 instances x 80 concurrency), and LangChain silently retrying each
call **six times** by default. A pathological turn was 30 requests, and nothing bounded
the total.

So, on every Vertex call:

- **Everything goes through `vertex_ai.call_model()`**, which holds a process-wide
  semaphore (`vertex_max_in_flight`). Not a decorator, so it cannot be forgotten.
- **`max_retries` is set explicitly** on both clients. Never leave it at the default.
- **Both clients share ONE `InMemoryRateLimiter`.** Quota is per project and model, not
  per Python object; two independently-paced clients just take turns exceeding it.
- **A 429 is caught as `ResourceExhausted` and never retried by us.** LangChain has
  already retried with backoff by then. It is logged as `vertex_throttled`, distinct
  from `planner_extraction_failed` — one means the pool is full, the other means the
  schema drifted, and they used to be the same log line.
- **Our own retries back off with jitter.** Without jitter, backoff just synchronises
  every client onto a slower clock.
- **A throttled turn tells the rider to try again.** It does not invent an answer out of
  an empty extraction, which is how a capacity problem got reported as hallucination.

All of these are per-process. With `--max-instances=10` the fleet rate is ten times the
per-instance rate; a true global cap belongs at the gateway or in the instance ceiling.

Tested and rejected: `location="global"` returns 501/404 with this SDK version. Another
region may have capacity when `us-central1` does not — that is a deploy-time change
(`VERTEX_AI_LOCATION`), not a code one.

## 4. Non-negotiable invariants

Each one is a real incident.

1. **Never ship an empty or unreadable reply.** A blank bubble is worse than a bad
   answer — the rider cannot tell it from a crash and has nothing to reply to. The
   writer's cleaner strips the JSON block; when the model answers with JSON *alone*,
   that removes everything. Guarded by `_is_unreadable` + `_blank_reply_fallback`.
   *(Shipped: `writer_completed response_length=119` and a blank screen.)*

2. **Never invent a fact the rider did not give.** No prices without a fare, no coupon
   codes the offers endpoint did not return, no sightseeing on a point-to-point ride.
   Enforced on the way out by `guardrails.apply_guardrails`, not by asking the prompt
   nicely. Whatever the agent proceeds without goes in `assumed_fields` and is stated.

3. **Never put a phrase in a date field.** The booking screen cannot render
   `"3 din baad"`. `sanitize.travel_date(..., as_iso=True)` resolves it. A date in the
   past is refused **by name** — dropping it silently gave the rider back the question
   they had just answered.

4. **Never apply a change to a detail the rider already gave without confirming it.**
   Filling a blank takes nothing away; overwriting does. See `app/agent/modification.py`.
   *(Shipped: "mujhe palampur se lekar kasol jana hai" produced a booking screen for a
   ride from "Mujhe Palampur" to "Lekar Kasol".)*

5. **A cancellation is decided in code, before the model.** `cancellation.py`.

6. **One reader, one fact.** When two places can recognise the same thing, they will
   drift and the drift will be silent. The relative-day tables drifted and booked a cab
   a day early. Derive one from the other and assert the parity in a test.

7. **The rider's own words go back to them.** When we cannot use an answer, quote it:
   *"'yesterday' has already passed"*, *"5 hours isn't one of our packages"*. Repeating
   the original question reads as not having been understood.

8. **Both languages, always.** Every rider-facing string exists in `hi` and `en`, in one
   table, with a test asserting the keys and the `{placeholders}` match. Never build a
   sentence by concatenating translated fragments.

---

## 5. Conventions

### Comments
Comments say **why**, and name the failure they prevent. This codebase's comments are
unusually long on purpose — they are the institutional memory. Match the surrounding
density; do not strip them.

### Tests
- Named as sentences about behaviour (`test_a_past_day_is_named_not_silently_dropped`),
  grouped in classes that describe the situation.
- **Stub what the system actually returns.** Tests stubbed the extraction boundary with
  `PlannerExtraction` — a shape the real code never produces — and stayed green while
  extraction failed half the time in production. Use
  `tests/conftest.py::wire_extraction`. *If a test double is easier to construct than
  the real thing, ask why, and make sure the difference cannot hide a bug.*
- Fix dates (`TODAY = date(2026, 9, 21)`), never `date.today()`.
- Mutation-check anything important: break the code, confirm the test fails, restore.

### Logging
`structlog`, snake_case events, `session_id` on everything. A degradation path logs at
`warning` or `error` with enough context to count it.

### Structured events
Any new type goes in `StructuredEventType`, is documented in `docs/api_reference.md`,
and is rendered in all three clients in `frontend/`. Two tests enforce this.

### Layering
`slots` (lowest, pure) ← `sanitize` ← nodes. `sanitize` imports `slots`; **`slots` must
never import `sanitize`**.

### Tokenising Hindi
Python's `\w` **excludes Devanagari combining marks**, so `[^\W\d_]+` turns `ठीक है`
into the fragments `ठ`, `क`, `ह` and every Devanagari phrase becomes unreachable. Use
`[^\wऀ-ॿ]+` to split. *(`cancellation.py` still has this bug — its Hindi
entries are dead. Unfixed.)*

---

## 6. Running and testing locally

```powershell
.\make.ps1 test          # full suite
.\make.ps1 check         # everything CI checks except tests
.\make.ps1 dev           # server on :8000
```

Test clients at `/local`, `/rental`, `/outstation` — served only when
`APP_ENV != production`. Auth: paste a rider JWT, or use `dev-test-token`
(non-production only, `app/api/middleware/auth.py`).

**The ladder, fastest first:**

1. the suite — seconds, no server
2. one function in a REPL — `sanitize.resolve_relative_day("3 din baad")`
3. a node or the compiled graph, with the LLM stubbed
4. the live server with a real Vertex call — **the only rung that catches the class of
   bug in §3**, because every rung above it stubs the boundary away

Rung 4 is not optional for anything touching the model boundary. Drive a real
conversation and read the logs for `planner_extraction_failed`, `*_gate_blocked`,
`writer_reply_cleaned_to_nothing`.

`mypy` is configured but **not enforcing** (`pyproject.toml`); the baseline is 52
pre-existing errors. Do not add to it.

---

## 7. Known open issues

Honest list. Do not rediscover these.

- **The router classifies a residence statement as `general_query`.** "mei himachal mei
  shimla mei rehta hu" → `intent=general_query`, so the turn takes the answer path with
  no fare and no booking handoff. `_answers_the_question` catches this only when that
  field is the one in flight.
- **The writer sometimes returns JSON with no prose** on `general_query` turns. The
  blank-reply guard catches it and apologises, which is honest but not good.
- **`cancellation.py` has the Devanagari tokenising bug** described in §5.
- **The deploy passes 17 env vars; `.env` has 30.** `OPENWEATHER_API_KEY` is among the
  missing, which is why deployed replies carry no weather line. Check
  `deploy.ps1:57` against `.env` before blaming the code for a missing feature.
- **No cab options without a rider JWT.** The fare gate requires one, `dev-test-token`
  maps to `jwt=None`, and no fare means no cabs to choose from. The three test clients
  now say so on screen.

---

## 8. Before you commit

- [ ] the full suite passes
- [ ] `ruff` clean; no new `mypy` errors
- [ ] anything touching the model boundary tested against the **live** model, not a stub
- [ ] new rider-facing strings in both languages
- [ ] new structured events declared, documented, rendered in all three clients
- [ ] comments explain why, and name the failure prevented
- [ ] the commit message says what broke and why the fix is shaped as it is
