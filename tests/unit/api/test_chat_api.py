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

from app.core.config import settings
from app.domain.conversation import Language, MessageRole, TripSession

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

    async def update_persona(self, session_id: str, persona: str, language: str) -> None:
        from app.domain.conversation import Language

        session = self.sessions[session_id]
        session.persona = persona
        session.language = Language(language)

    async def update_location(self, session_id: str, location: dict, entities=None) -> None:
        from app.domain.conversation import ExtractedEntities
        from app.domain.location import RiderLocation

        session = self.sessions[session_id]
        session.rider_location = RiderLocation.model_validate(location)
        if entities is not None:
            session.entities = ExtractedEntities.model_validate(entities)

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

    def test_rental_start_offers_the_rate_cards_packages(self, api, no_network):
        """The rental greeting asks for a duration, and the backend matches durations
        exactly — so /start ships the bookable ones with it, for the app to render as
        taps rather than have the rider guess a number."""
        no_network["rental"]._get.return_value = {"data": [
            {"categoryId": 1, "name": "Economy", "rentalChartList": [
                {"maxHour": 4, "maxKilometer": 40},
                {"maxHour": 8, "maxKilometer": 80},
            ]},
        ]}
        client, _, _, _ = api
        # A real rider token: the rate card is rider-authenticated, and the dev token
        # deliberately carries no JWT so dev mode never reaches the Him Cab backend.
        rider = {"Authorization": f"Bearer {_fake_jwt({'riderId': 4242})}"}
        body = client.post("/v1/chat/rental/start", headers=rider).json()

        assert body["persona"] == "rental"
        assert "8 hrs / 80 km" in body["message"]
        assert [e["type"] for e in body["structured_events"]] == ["rental_packages"]
        packages = body["structured_events"][0]["payload"]["packages"]
        assert [p["hours"] for p in packages] == [4, 8]

    def test_an_unreachable_rate_card_still_offers_the_configured_durations(self, api):
        """The rate card is the Him Cab backend. If it is down the rider is still offered
        the durations the product sells (3/5/8 hrs) rather than an open question they can
        only answer wrongly — the price is what is lost, not the choice."""
        client, _, _, _ = api
        rider = {"Authorization": f"Bearer {_fake_jwt({'riderId': 4242})}"}
        body = client.post("/v1/chat/rental/start", headers=rider).json()

        assert "4 hrs" in body["message"] and "8 hrs" in body["message"]
        packages = body["structured_events"][0]["payload"]["packages"]
        assert [p["hours"] for p in packages] == settings.rental_hour_options
        # Flagged, so the app never draws a price next to one of these.
        assert not any(p["from_rate_card"] for p in packages)

    def test_the_dev_token_gets_no_packages_and_therefore_no_prices(self, api, no_network):
        """The commonest reason a test page shows no rental price. `dev-test-token`
        resolves to jwt=None by design, and both the rate card and the fare endpoint are
        rider-authenticated — so there is nothing to quote. The rider is still offered
        the configured durations, unpriced, and the backend is never called without a
        credential."""
        no_network["rental"]._get.return_value = {"data": [
            {"categoryId": 1, "name": "Economy",
             "rentalChartList": [{"maxHour": 8, "maxKilometer": 80}]},
        ]}
        client, _, _, _ = api
        body = client.post("/v1/chat/rental/start", headers=AUTH).json()

        packages = body["structured_events"][0]["payload"]["packages"]
        assert [p["hours"] for p in packages] == settings.rental_hour_options
        assert not any(p["from_rate_card"] or p["included_km"] for p in packages)
        no_network["rental"]._get.assert_not_awaited()

    def test_the_other_personas_ship_no_events_with_their_greeting(self, api):
        client, _, _, _ = api
        for persona in ("local", "outstation"):
            body = client.post(f"/v1/chat/{persona}/start", headers=AUTH).json()
            assert body["structured_events"] == []

    def test_reopening_the_dialog_starts_a_fresh_conversation(self, api):
        client, sessions, _, _ = api
        first = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]
        second = client.post("/v1/chat/local/start", headers=AUTH).json()["session_id"]

        assert first != second
        assert len(sessions.sessions) == 2


# ── /flows and /select: the in-chat flow picker ───────────────────────────────

class TestFlowChooser:
    """The selectable service boxes that replace the persona dropdown. `/flows` offers
    them; `/select` starts the one the rider tapped."""

    def test_flows_lists_the_three_services_in_order(self, api):
        client, _, _, _ = api
        body = client.get("/v1/chat/flows").json()

        assert [f["persona"] for f in body["flows"]] == ["local", "outstation", "rental"]
        assert body["message"]
        for flow in body["flows"]:
            assert flow["title"] and flow["subtitle"]
            assert flow["base_path"] == f"/v1/chat/{flow['persona']}"

    def test_flows_also_ships_the_boxes_as_a_structured_event(self, api):
        """So the app can render them with the same code as every other card."""
        client, _, _, _ = api
        events = client.get("/v1/chat/flows").json()["structured_events"]

        assert [e["type"] for e in events] == ["flow_options"]
        payload = events[0]["payload"]["flows"]
        assert [f["persona"] for f in payload] == ["local", "outstation", "rental"]

    def test_flows_needs_no_auth_and_creates_nothing(self, api):
        """Like /greeting: it is a constant, shown before the app has a token."""
        client, sessions, _, _ = api
        resp = client.get("/v1/chat/flows")

        assert resp.status_code == 200
        assert sessions.sessions == {}
        assert sessions.pointers == {}

    def test_flows_localizes_to_hindi(self, api):
        client, _, _, _ = api
        body = client.get("/v1/chat/flows", params={"lang": "hi"}).json()

        assert body["language"] == "hi"
        # Devanagari in the neutral greeting and the box titles.
        assert any("ऀ" <= ch <= "ॿ" for ch in body["message"])
        assert any(
            "ऀ" <= ch <= "ॿ" for ch in body["flows"][0]["title"]
        )

    def test_selecting_rental_starts_the_rental_flow(self, api):
        client, sessions, _, _ = api
        body = client.post("/v1/chat/select", headers=AUTH, json={"flow": "rental"}).json()

        assert body["persona"] == "rental"
        assert "car and driver" in body["message"] or "How long" in body["message"]
        assert body["session_id"] in sessions.sessions
        assert sessions.sessions[body["session_id"]].persona == "rental"

    def test_selecting_local_starts_the_local_flow_in_hindi(self, api):
        client, _, _, _ = api
        body = client.post("/v1/chat/select", headers=AUTH, json={"flow": "local"}).json()

        assert body["persona"] == "local"
        assert body["language"] == Language.HINDI.value

    def test_select_requires_auth(self, api):
        """/flows is open; /select creates a session, so it needs the rider token."""
        client, _, _, _ = api
        assert client.post("/v1/chat/select", json={"flow": "local"}).status_code == 401

    def test_an_invalid_flow_is_rejected(self, api):
        client, _, _, _ = api
        resp = client.post("/v1/chat/select", headers=AUTH, json={"flow": "spaceship"})
        assert resp.status_code == 422

    def test_select_applies_the_gps_override_like_start(self, api):
        """A deliberate tap resolves the same way a screen tap does: a Delhi fix on a
        'local' selection is served as outstation, and the response says so. Keeps one
        rule for local-vs-visiting instead of two that can disagree."""
        client, _, _, _ = api
        body = client.post(
            "/v1/chat/select", headers=AUTH, json={"flow": "local", **DELHI}
        ).json()

        assert body["persona"] == "outstation"
        assert body["location"]["in_himachal"] is False

    def test_selecting_rental_is_never_rerouted_by_a_fix(self, api):
        """A rental is a service they chose; their coordinates say where to collect
        them, not that they stopped wanting a car."""
        client, _, _, _ = api
        body = client.post(
            "/v1/chat/select", headers=AUTH, json={"flow": "rental", **SHIMLA}
        ).json()

        assert body["persona"] == "rental"


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


# ── Coordinates: which assistant, and where from ──────────────────────────────

SHIMLA = {"lat": 31.1048, "lon": 77.1734}
DELHI = {"lat": 28.6139, "lon": 77.2090}


class TestPersonaRouting:
    """The screen a rider tapped from is a guess; their coordinates are not.

    A visitor browsing the home screen lands on /local and would be greeted in Hindi
    about a state they have never driven in; a Shimla resident opening the outstation
    screen would be handed a three-day itinerary for the town they live in. A GPS fix
    settles it, so when one is sent it decides the persona whichever path was called.
    """

    def test_route_endpoint_sends_a_himachal_rider_to_local(self, api):
        client, _, _, _ = api
        body = client.post("/v1/chat/route", headers=AUTH, json=SHIMLA).json()

        assert body["persona"] == "local"
        assert body["base_path"] == "/v1/chat/local"
        assert body["reason"] == "coordinates"
        assert body["location"]["in_himachal"] is True
        assert body["location"]["place_name"] == "Shimla"

    def test_route_endpoint_sends_a_visitor_to_outstation(self, api):
        client, _, _, _ = api
        body = client.post("/v1/chat/route", headers=AUTH, json=DELHI).json()

        assert body["persona"] == "outstation"
        assert body["base_path"] == "/v1/chat/outstation"
        assert body["location"]["place_name"] == "Delhi"

    def test_route_endpoint_creates_nothing(self, api):
        """A pure lookup — safe to call on app start or on every location change."""
        client, sessions, _, _ = api
        client.post("/v1/chat/route", headers=AUTH, json=SHIMLA)
        assert sessions.sessions == {}
        assert sessions.pointers == {}

    def test_route_endpoint_requires_the_same_single_credential(self, api):
        client, _, _, _ = api
        assert client.post("/v1/chat/route", json=SHIMLA).status_code == 401

    def test_unusable_coordinates_keep_the_requested_persona(self, api):
        """(0, 0) is what a phone reports before it has a fix. Trusting it would put a
        Shimla rider in the Atlantic and hand them an English itinerary."""
        client, _, _, _ = api
        body = client.post(
            "/v1/chat/route", headers=AUTH, json={"lat": 0.0, "lon": 0.0}
        ).json()

        assert body["reason"] == "default"
        assert body["location"] is None

    def test_start_on_the_wrong_path_is_served_as_the_right_persona(self, api):
        """/local/start with a Delhi fix returns persona=outstation — and the greeting
        that goes with it, in English."""
        client, _, _, _ = api
        body = client.post("/v1/chat/local/start", headers=AUTH, params=DELHI).json()

        assert body["persona"] == "outstation"
        assert body["language"] == "en"
        assert body["location"]["in_himachal"] is False

    def test_start_without_coordinates_keeps_the_path_persona(self, api):
        """No fix, no change — a client that sends nothing behaves exactly as before."""
        client, _, _, _ = api
        body = client.post("/v1/chat/local/start", headers=AUTH).json()

        assert body["persona"] == "local"
        assert body["language"] == "hi"
        assert body["location"] is None

    def test_message_reports_the_persona_it_served(self, api):
        client, _, _, _ = api
        body = client.post(
            "/v1/chat/outstation/message",
            headers=AUTH,
            json={"message": "Manali jaana hai", **SHIMLA},
        ).json()

        assert body["persona"] == "local"


class TestLocationReachesTheAgent:
    """Coordinates are only worth sending if they change what the agent can do: they
    are the pickup point the fare is priced from and the pin the booking screen opens
    on. Every one of these asserts they arrived somewhere that matters."""

    def test_fix_is_attached_to_the_agent_state(self, api):
        client, _, graph, _ = api
        client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "Manali", **SHIMLA},
        )

        location = graph.last_state.rider_location
        assert location is not None
        assert location.place_name == "Shimla"
        assert location.in_himachal is True

    def test_a_precise_fix_answers_the_starting_city_question(self, api):
        """The assistant used to have to ask "which city are you travelling from?"
        before it could price anything. A fix on a known town answers it."""
        client, _, graph, _ = api
        client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "Manali", **SHIMLA},
        )

        entities = graph.last_state.entities
        assert entities.origin == "Shimla"
        assert entities.home_city == "Shimla"

    def test_a_vague_fix_does_not_invent_a_starting_city(self, api):
        """A point 26 km from Mandi is fine for measuring a distance and useless as a
        place name — announcing "your trip from Mandi" to someone who has never been
        there is worse than asking."""
        client, _, graph, _ = api
        client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "Manali", "lat": 31.50, "lon": 77.05},
        )

        assert graph.last_state.entities.origin is None
        assert graph.last_state.rider_location is not None

    def test_a_named_origin_outranks_the_phone(self, api):
        """Booking a Delhi pickup from a Shimla sofa is a real thing people do."""
        client, sessions, graph, _ = api
        started = client.post("/v1/chat/outstation/start", headers=AUTH).json()
        session = sessions.sessions[started["session_id"]]
        session.entities.origin = "Delhi"

        client.post(
            "/v1/chat/outstation/message",
            headers=AUTH,
            json={"message": "Manali", **SHIMLA},
        )

        assert graph.last_state.entities.origin == "Delhi"

    def test_a_persona_switch_carries_the_conversation_across(self, api):
        """The likeliest way a rider's persona changes is that they ARRIVED — they
        planned the trip from Delhi and are now standing in Manali. Dropping them into
        an empty chat at that point loses everything they already told us."""
        client, sessions, graph, _ = api
        started = client.post("/v1/chat/outstation/start", headers=AUTH).json()
        client.post(
            "/v1/chat/outstation/message",
            headers=AUTH,
            json={"message": "3 days in Manali for 4 people"},
        )

        body = client.post(
            "/v1/chat/outstation/message",
            headers=AUTH,
            json={"message": "we've landed", **SHIMLA},
        ).json()

        assert body["persona"] == "local"
        assert body["session_id"] == started["session_id"]
        # Re-filed durably, not just in memory: `list_user_sessions` consumers filter
        # on the stored persona, so a document left saying "outstation" becomes
        # unresumable the moment the active-chat pointer is lost.
        adopted = sessions.sessions[started["session_id"]]
        assert adopted.persona == "local"
        assert adopted.language == "hi"
        # The history came with it — that is the whole point.
        assert any(
            "3 days in Manali" in m.content for m in graph.last_state.messages
        )

    def test_the_newest_fix_wins(self, api):
        """Riders move between turns — they open the chat at home and send the next
        message from the taxi rank."""
        client, _, graph, _ = api
        client.post(
            "/v1/chat/local/message", headers=AUTH, json={"message": "hi", **SHIMLA}
        )
        client.post(
            "/v1/chat/local/message",
            headers=AUTH,
            json={"message": "actually from here", "lat": 32.2396, "lon": 77.1887},
        )

        assert graph.last_state.rider_location.place_name == "Manali"

    def test_a_fix_persists_across_turns_without_being_resent(self, api):
        client, _, graph, _ = api
        client.post(
            "/v1/chat/local/message", headers=AUTH, json={"message": "hi", **SHIMLA}
        )
        client.post("/v1/chat/local/message", headers=AUTH, json={"message": "Manali"})

        assert graph.last_state.rider_location.place_name == "Shimla"

    def test_rider_id_from_the_jwt_reaches_the_agent(self, api):
        """It personalises the offer lookup; without it the agent falls back to the
        unpersonalised catalogue rather than showing no offers."""
        client, _, graph, _ = api
        token = _fake_jwt({"riderId": 4321, "sub": "9816000000"})
        client.post(
            "/v1/chat/local/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "Manali"},
        )

        assert graph.last_state.rider_id == 4321

    def test_a_phone_number_subject_is_not_read_as_a_rider_id(self, api):
        """`sub` on this backend is the rider's phone number. Sending it as a riderId
        would query offers for whichever account happens to have that id."""
        client, _, graph, _ = api
        token = _fake_jwt({"sub": "9816000000"})
        client.post(
            "/v1/chat/local/message",
            headers={"Authorization": f"Bearer {token}"},
            json={"message": "Manali"},
        )

        assert graph.last_state.rider_id is None


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
        from app.schemas.chat import StructuredEventType

        assert {t.value for t in StructuredEventType} == {
            "itinerary", "weather_alert", "snowfall_alert", "cab_fare", "offers",
            "booking_handoff", "rental_packages", "request_cancelled", "flow_options",
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
