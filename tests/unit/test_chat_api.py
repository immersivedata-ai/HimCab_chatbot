"""
tests/unit/test_chat_api.py
────────────────────────────
Contract tests for the JWT-only chat API.

These lock down the promises made to the front-end team in app/api/v1/routes/chat.py:
  - the ONLY input is `Authorization: Bearer <rider JWT>` (+ the message text)
  - no session id / persona / rider id is ever accepted or required
  - /start returns a persona-appropriate welcome greeting
  - sending a message without calling /start still works (no 404 to recover from)
  - the caller's token reaches the agent as `rider_jwt` (so fares can be quoted)
"""

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app.models.session import Language, MessageRole, TripSession


# ── Test doubles ──────────────────────────────────────────────────────────────

class FakeSessionService:
    """In-memory stand-in for the Firestore SessionService."""

    def __init__(self) -> None:
        self.sessions: dict[str, TripSession] = {}
        # The durable active-conversation pointer, keyed (user_id, persona).
        self.pointers: dict[tuple[str, str], str] = {}
        self.list_should_raise = False
        self.pointers_should_raise = False

    async def create_session(self, session: TripSession) -> TripSession:
        self.sessions[session.session_id] = session
        return session

    async def get_session(self, session_id: str) -> TripSession:
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id} not found")
        return self.sessions[session_id]

    async def append_message(self, session_id: str, role, content: str, **kwargs) -> None:
        session = self.sessions[session_id]
        session.add_message(MessageRole(role), content)

    async def list_user_sessions(self, user_id: str, limit: int = 10) -> list[TripSession]:
        if self.list_should_raise:
            raise RuntimeError("missing composite index")
        owned = [s for s in self.sessions.values() if s.user_id == user_id]
        return sorted(owned, key=lambda s: s.updated_at, reverse=True)[:limit]

    async def set_active_session(self, user_id: str, persona: str, session_id: str) -> None:
        if self.pointers_should_raise:
            return  # the real service swallows write failures
        self.pointers[(user_id, persona)] = session_id

    async def get_active_session_id(self, user_id: str, persona: str) -> str | None:
        if self.pointers_should_raise:
            return None
        return self.pointers.get((user_id, persona))


class FakeGraph:
    """Stand-in for the compiled LangGraph — records the state it was invoked with."""

    def __init__(self, response_text: str = "Sure, let's plan that.") -> None:
        self.response_text = response_text
        self.last_state = None
        # Overridable per-test — the writer node is what populates this in production.
        self.structured_events: list[dict] = []

    async def ainvoke(self, state):
        self.last_state = state
        return {
            "final_response_text": self.response_text,
            "structured_events": list(self.structured_events),
            "node_trace": ["router", "planner"],
            "tokens_used": 11,
        }


def _fake_jwt(claims: dict) -> str:
    """Build an unsigned-but-well-formed JWT carrying `claims`."""
    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{seg({'alg': 'HS256', 'typ': 'JWT'})}.{seg(claims)}.signature"


@pytest.fixture
def api(monkeypatch):
    """TestClient wired to in-memory fakes. Yields (client, sessions, graph, limiter)."""
    from app.api.v1.routes import chat as chat_route
    from app.services import chat_session as chat_session_svc

    sessions = FakeSessionService()
    graph = FakeGraph()

    monkeypatch.setattr(chat_session_svc, "get_session_service", lambda: sessions)
    monkeypatch.setattr(chat_route, "get_session_service", lambda: sessions)
    monkeypatch.setattr(chat_route, "get_agent_graph", lambda: graph)
    # The limiter is process-wide, so a leftover count would leak between tests.
    chat_route._rate_limiter.reset()

    from app.main import app

    with TestClient(app) as client:
        yield client, sessions, graph, chat_route._rate_limiter


AUTH = {"Authorization": "Bearer dev-test-token"}


# ── Auth: the single credential ───────────────────────────────────────────────

class TestAuth:
    def test_missing_authorization_is_rejected(self, api):
        client, _, _, _ = api
        resp = client.post("/v1/chat/local/start")
        assert resp.status_code == 401

    def test_dev_token_accepted_outside_production(self, api):
        client, _, _, _ = api
        assert client.post("/v1/chat/local/start", headers=AUTH).status_code == 200

    def test_real_jwt_is_accepted_and_forwarded_to_the_agent(self, api):
        """The caller's token must reach the agent as rider_jwt — without it the fare
        gate blocks every quote (the exact bug that made fares silently unavailable)."""
        client, _, graph, _ = api
        token = _fake_jwt({"riderId": 4242})

        resp = client.post(
            "/v1/chat/local/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "Manali jaana hai"},
        )

        assert resp.status_code == 200
        assert graph.last_state.rider_jwt == token

    def test_doubled_bearer_prefix_is_tolerated(self, api):
        """An app that stores the whole header value would otherwise send
        'Bearer Bearer <jwt>' and get a silent 401 from the fare service."""
        client, _, graph, _ = api
        token = _fake_jwt({"riderId": 7})

        resp = client.post(
            "/v1/chat/local/message",
            headers={"Authorization": f"Bearer Bearer {token}"},
            json={"message": "hi"},
        )

        assert resp.status_code == 200
        assert graph.last_state.rider_jwt == token

    def test_same_rider_keeps_one_conversation_across_calls(self, api):
        """Identity comes from the token's claims, so the session must be reused."""
        client, sessions, _, _ = api
        headers = {"Authorization": f"Bearer {_fake_jwt({'riderId': 99})}"}

        client.post("/v1/chat/local/start", headers=headers)
        client.post("/v1/chat/local/message", headers=headers, json={"message": "one"})
        client.post("/v1/chat/local/message", headers=headers, json={"message": "two"})

        assert len(sessions.sessions) == 1

    def test_different_riders_get_separate_conversations(self, api):
        client, sessions, _, _ = api
        for rider_id in (1, 2):
            client.post(
                "/v1/chat/local/message",
                headers={"Authorization": f"Bearer {_fake_jwt({'riderId': rider_id})}"},
                json={"message": "hello"},
            )
        assert len(sessions.sessions) == 2


# ── /start: opening the chat dialog ───────────────────────────────────────────

class TestStart:
    def test_local_start_returns_hindi_welcome(self, api):
        client, _, _, _ = api
        body = client.post("/v1/chat/local/start", headers=AUTH).json()

        assert body["persona"] == "local"
        assert body["language"] == Language.HINDI.value
        assert "Him Cab" in body["message"]
        assert body["session_id"]

    def test_outstation_start_returns_english_welcome(self, api):
        client, _, _, _ = api
        body = client.post("/v1/chat/outstation/start", headers=AUTH).json()

        assert body["persona"] == "outstation"
        assert body["language"] == Language.ENGLISH.value
        assert "Welcome to Him Cab" in body["message"]

    def test_welcome_is_persisted_as_conversation_context(self, api):
        """The greeting asks 'where would you like to go?'. It must be stored as a real
        assistant message, otherwise the planner can't read a one-word reply like
        'manali' as the answer to it."""
        client, sessions, _, _ = api
        session_id = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]

        messages = sessions.sessions[session_id].messages
        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT.value
        assert messages[0].content

    def test_start_seeds_visitor_type_from_the_route(self, api):
        client, sessions, _, _ = api
        session_id = client.post("/v1/chat/outstation/start", headers=AUTH).json()["session_id"]
        assert sessions.sessions[session_id].entities.visitor_type == "outstation"

    def test_reopening_the_dialog_starts_a_fresh_conversation(self, api):
        client, sessions, _, _ = api
        first = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]
        second = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]

        assert first != second
        assert len(sessions.sessions) == 2


# ── /message and /stream: no session id anywhere ──────────────────────────────

class TestSendMessage:
    def test_message_needs_only_the_text(self, api):
        client, _, _, _ = api
        client.post("/v1/chat/local/start", headers=AUTH)

        resp = client.post(
            "/v1/chat/local/message", headers=AUTH, json={"message": "Manali"}
        )

        assert resp.status_code == 200
        assert resp.json()["response"] == "Sure, let's plan that."

    def test_message_without_start_still_works(self, api):
        """Robustness promise: the front end can never land in a broken state that
        needs a manual reset."""
        client, sessions, _, _ = api
        resp = client.post(
            "/v1/chat/outstation/message", headers=AUTH, json={"message": "Shimla"}
        )

        assert resp.status_code == 200
        assert len(sessions.sessions) == 1

    def test_conversation_history_is_carried_into_the_agent(self, api):
        client, _, graph, _ = api
        client.post("/v1/chat/local/start", headers=AUTH)
        client.post("/v1/chat/local/message", headers=AUTH, json={"message": "Manali"})

        # welcome (assistant) + "Manali" (user) must both be visible to the agent
        contents = [m.content for m in graph.last_state.messages]
        assert len(contents) == 2
        assert contents[-1] == "Manali"

    def test_unknown_session_id_in_the_body_is_ignored_not_an_error(self, api):
        """A stale session_id on the client must never wedge the UI — it is ignored
        and resolution falls through to the caller's real conversation."""
        client, sessions, _, _ = api
        started = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]

        resp = client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "hi", "session_id": "whatever-the-app-had"},
        )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == started
        assert len(sessions.sessions) == 1

    def test_empty_message_is_rejected(self, api):
        client, _, _, _ = api
        resp = client.post("/v1/chat/local/message", headers=AUTH, json={"message": ""})
        assert resp.status_code == 422

    def test_stream_endpoint_emits_sse_events(self, api):
        client, _, _, _ = api
        with client.stream(
            "GET", "/v1/chat/local/stream", headers=AUTH, params={"message": "Manali"}
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())

        assert "event: token" in body
        assert "event: done" in body

    def test_stream_requires_auth(self, api):
        client, _, _, _ = api
        resp = client.get("/v1/chat/local/stream", params={"message": "hi"})
        assert resp.status_code == 401


# ── Session resolution edge cases ─────────────────────────────────────────────

class TestSessionResolution:
    def test_conversation_survives_a_lost_pointer_document(self, api):
        """A missing active-chat pointer must not silently wipe an in-flight chat —
        the recent-session scan is the last line of defence."""
        client, sessions, _, _ = api
        started = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]

        sessions.pointers.clear()   # pointer gone; the recent-session scan should find it

        client.post("/v1/chat/local/message", headers=AUTH, json={"message": "Manali"})

        assert len(sessions.sessions) == 1
        assert started in sessions.sessions

    def test_new_session_created_when_nothing_is_resumable(self, api):
        client, sessions, _, _ = api
        sessions.pointers.clear()
        sessions.list_should_raise = True   # e.g. Firestore index missing

        resp = client.post("/v1/chat/local/message", headers=AUTH, json={"message": "hi"})

        assert resp.status_code == 200
        assert len(sessions.sessions) == 1

    def test_local_and_outstation_chats_do_not_collide(self, api):
        client, sessions, _, _ = api
        client.post("/v1/chat/local/start", headers=AUTH)
        client.post("/v1/chat/outstation/start", headers=AUTH)

        personas = sorted(s.persona for s in sessions.sessions.values())
        assert personas == ["local", "outstation"]


    def test_multi_turn_memory_survives_a_broken_recent_session_scan(self, api):
        """The production failure this guards against:

        the (user_id, created_at) composite index missing meant every /message minted a
        fresh session, so the agent never remembered the previous turn — it asked "where
        do you want to go?" forever. The durable pointer document, which needs no index,
        must hold the conversation together on its own.
        """
        client, sessions, _, _ = api
        sessions.list_should_raise = True  # the recent-session scan can't help

        client.post("/v1/chat/local/start", headers=AUTH)
        ids = {
            client.post(
                "/v1/chat/local/message", headers=AUTH, json={"message": text}
            ).json()["session_id"]
            for text in ("I want to go to Manali", "3 days, family of 4", "make it")
        }

        assert len(ids) == 1, f"session churned across turns: {ids}"
        assert len(sessions.sessions) == 1

    def test_explicit_session_id_pins_the_conversation(self, api):
        """Two devices on one account can each pin their own conversation."""
        client, sessions, _, _ = api
        first = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]
        second = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]
        assert first != second

        resp = client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "back to the first chat", "session_id": first},
        )

        assert resp.json()["session_id"] == first
        assert len(sessions.sessions) == 2

    def test_cannot_pin_another_riders_session(self, api):
        """Security: passing someone else's session_id must not hand it over."""
        client, sessions, _, _ = api
        victim = {"Authorization": f"Bearer {_fake_jwt({'riderId': 'victim'})}"}
        attacker = {"Authorization": f"Bearer {_fake_jwt({'riderId': 'attacker'})}"}

        victim_session = client.post("/v1/chat/local/start", headers=victim).json()["session_id"]

        resp = client.post(
            "/v1/chat/local/message",
            headers=attacker,
            json={"message": "give me their chat", "session_id": victim_session},
        )

        assert resp.status_code == 200
        assert resp.json()["session_id"] != victim_session
        assert sessions.sessions[victim_session].user_id == "rider:victim"


# ── /history: restoring a chat panel ──────────────────────────────────────────

class TestHistory:
    def test_history_replays_the_conversation(self, api):
        """Phase 2 conversation restore: reopen the panel with no client-side state."""
        client, _, _, _ = api
        started = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]
        client.post("/v1/chat/local/message", headers=AUTH, json={"message": "Manali"})

        body = client.get("/v1/chat/local/history", headers=AUTH).json()

        assert body["session_id"] == started
        assert body["persona"] == "local"
        # The assistant's reply is written by the graph's persist node, which the
        # FakeGraph stands in for — so this harness sees welcome + user turn only.
        roles = [m["role"] for m in body["messages"]]
        assert roles == ["assistant", "user"]
        assert body["messages"][1]["content"] == "Manali"

    def test_history_uses_the_same_rider_auth_as_chat(self, api):
        """/v1/chat/*, /v1/session/* and /v1/user/* must all accept ONE credential —
        the inconsistency here is what broke conversation restore."""
        client, _, _, _ = api
        assert client.get("/v1/chat/local/history").status_code == 401
        assert client.get("/v1/chat/local/history", headers=AUTH).status_code == 200
        assert client.get("/v1/user/profile", headers=AUTH).status_code != 401

    def test_history_respects_the_limit(self, api):
        client, _, _, _ = api
        client.post("/v1/chat/local/start", headers=AUTH)
        for i in range(3):
            client.post("/v1/chat/local/message", headers=AUTH, json={"message": f"m{i}"})

        body = client.get("/v1/chat/local/history", headers=AUTH, params={"limit": 2}).json()

        assert len(body["messages"]) == 2
        assert body["total_messages"] == 4  # welcome + 3 user turns (see note above)


# ── Response payload contract ─────────────────────────────────────────────────

class TestResponsePayload:
    def test_itinerary_field_mirrors_the_structured_event(self, api):
        """`itinerary` used to be null even on turns that produced a full itinerary,
        because Itinerary validation failed and the raw dict only reached
        structured_events. Both channels must now carry the same object."""
        client, _, graph, _ = api
        payload = {
            "destination": "Manali",
            "duration_days": 3,
            "group_type": "family",
            "budget_range": "moderate",
            "days": [],
        }
        graph.structured_events = [{"type": "itinerary", "payload": payload}]

        body = client.post(
            "/v1/chat/outstation/message", headers=AUTH, json={"message": "plan it"}
        ).json()

        assert body["itinerary"] == payload
        assert body["structured_events"][0]["payload"] == body["itinerary"]

    def test_itinerary_is_null_when_the_turn_produced_none(self, api):
        client, _, _, _ = api
        body = client.post(
            "/v1/chat/local/message", headers=AUTH, json={"message": "hi"}
        ).json()

        assert body["itinerary"] is None
        assert body["structured_events"] == []

    def test_unknown_structured_event_type_is_rejected_at_the_boundary(self, api):
        """StructuredEventType is the documented, exhaustive list — a typo in a new
        event type should fail loudly here rather than reach the app."""
        from app.models.session import StructuredEventType

        assert {t.value for t in StructuredEventType} == {
            "itinerary", "weather_alert", "snowfall_alert", "cab_fare",
        }

    def test_rate_limit_returns_429_with_retry_after(self, api):
        """The app needs a back-off contract, not a bare 429."""
        client, _, _, limiter = api

        for _ in range(limiter.limit):
            assert client.post(
                "/v1/chat/local/message", headers=AUTH, json={"message": "hi"}
            ).status_code == 200

        resp = client.post("/v1/chat/local/message", headers=AUTH, json={"message": "hi"})

        assert resp.status_code == 429
        assert resp.headers["Retry-After"] == "60"
        body = resp.json()
        assert body["error"]["code"] == "RATE_LIMIT_EXCEEDED"
        assert body["error"]["details"]["retry_after"] == 60

    def test_the_rate_limit_is_per_rider(self, api):
        """One noisy rider must not lock everyone else out."""
        client, _, _, limiter = api
        noisy = {"Authorization": "Bearer " + _fake_jwt({"riderId": "noisy"})}
        quiet = {"Authorization": "Bearer " + _fake_jwt({"riderId": "quiet"})}

        for _ in range(limiter.limit + 1):
            client.post("/v1/chat/local/message", headers=noisy, json={"message": "hi"})

        assert client.post(
            "/v1/chat/local/message", headers=quiet, json={"message": "hi"}
        ).status_code == 200


# ── The removed surface stays removed ─────────────────────────────────────────

class TestLegacyEndpointsAreGone:
    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/v1/chat/stream"),
            ("post", "/v1/chat/message"),
        ],
    )
    def test_persona_param_endpoints_are_removed(self, api, method, path):
        client, _, _, _ = api
        resp = getattr(client, method)(path, headers=AUTH)
        assert resp.status_code == 404
