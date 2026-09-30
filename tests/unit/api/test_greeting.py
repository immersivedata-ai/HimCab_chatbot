"""
tests/unit/api/test_greeting.py
────────────────────────────────
The opening bubble, and the promise that nothing stands in front of it.

`/start` cannot be fast. It reverse-geocodes the rider's fix, reads the rental rate
card (with retries, when that backend is slow) and writes a session to Firestore before
it can answer — and none of that work changes the greeting TEXT. An app that waited for
`/start` to render its first bubble was waiting on three round trips for a string that
was already known, and on a cold start it was waiting for a container to boot as well.

So the greeting is available two ways, and both are tested here:

  1. `GET /v1/chat/{persona}/greeting` — the same constant, with no I/O and no auth.
  2. A copy bundled into the client, rendered with no network call at all. This is the
     only one that can appear while the container is still starting, because at that
     point no endpoint can answer.

The second buys the latency; the first is what stops the copy going stale. A bundled
greeting that has drifted shows the rider one message and then swaps it for another a
moment later, which is a worse experience than the slow version it replaced — so the
drift check below is the point of the endpoint existing at all.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.chat_session import DEFAULT_LANGUAGE, WELCOME_MESSAGES

PERSONAS = ("local", "outstation", "rental")

CLIENTS = (
    "frontend/index.html",
    "frontend/local.html",
    "frontend/rental.html",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ── 1. The endpoint ───────────────────────────────────────────────────────────

class TestTheGreetingEndpoint:

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_it_returns_the_backend_s_own_constant(self, client, persona):
        body = client.get(f"/v1/chat/{persona}/greeting").json()
        assert body["message"] == WELCOME_MESSAGES[persona]

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_it_needs_no_authorization_header(self, client, persona):
        """The point is that it answers before the app has done anything — which
        includes before it has a token. Requiring one would put the thing the app needs
        first behind the thing it gets last."""
        response = client.get(f"/v1/chat/{persona}/greeting")
        assert response.status_code == 200

    @pytest.mark.parametrize("persona", PERSONAS)
    def test_the_language_is_a_code_not_an_enum_repr(self, client, persona):
        """`str(Language.HINDI)` is "Language.HINDI". Elsewhere the value comes off a
        TripSession, which stores enums by value already, so `str()` is right there and
        would be silently wrong here."""
        language = client.get(f"/v1/chat/{persona}/greeting").json()["language"]
        assert language == DEFAULT_LANGUAGE[persona].value
        assert language in ("en", "hi", "hinglish")

    @pytest.mark.parametrize("persona", ["local", "outstation"])
    def test_a_trip_greeting_is_final(self, client, persona):
        """`/start` returns this same text for these two, so the app has nothing to
        replace and the bubble never flickers."""
        assert client.get(f"/v1/chat/{persona}/greeting").json()["is_final"] is True

    def test_the_rental_greeting_is_not_final(self, client):
        """A rental duration is matched EXACTLY by the backend, so `/start` upgrades
        this open question to one naming the packages the rate card actually sells.
        Both are answerable, which is why showing the generic one first costs nothing —
        but the app has to know a better one is coming."""
        assert client.get("/v1/chat/rental/greeting").json()["is_final"] is False

    def test_it_touches_nothing(self, client, monkeypatch):
        """The guarantee that makes it worth calling. If this endpoint ever grows a
        Firestore read or a rate-card fetch it stops being the fast path, and the
        regression would otherwise show up only as latency nobody can place."""
        def explode(*_args, **_kwargs):
            raise AssertionError("the greeting endpoint must not perform I/O")

        monkeypatch.setattr("app.repositories.firestore.get_firestore_client", explode)
        monkeypatch.setattr("app.repositories.firestore.get_session_service", explode)
        monkeypatch.setattr("app.services.geo.resolve_rider_location", explode)
        monkeypatch.setattr("app.integrations.himcab.available_packages", explode)
        monkeypatch.setattr("app.api.v1.routes.chat.available_packages", explode)
        monkeypatch.setattr("app.services.vertex_ai.get_vertex_ai_service", explode)

        for persona in PERSONAS:
            assert client.get(f"/v1/chat/{persona}/greeting").status_code == 200

    def test_it_is_not_rate_limited(self, client):
        """It does no work, and the app may call it on every launch. Spending a rider's
        30-per-minute budget on a constant would be a strange way to lose a fare."""
        for _ in range(40):
            assert client.get("/v1/chat/local/greeting").status_code == 200


# ── 2. The bundled copies ─────────────────────────────────────────────────────

def _bundled_greetings(path: pathlib.Path) -> dict[str, str]:
    """The `GREETINGS` object out of a client page.

    Parsed as JSON rather than executed: the block is emitted as JSON precisely so this
    check needs no JavaScript engine and can run anywhere the suite runs.
    """
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const GREETINGS = (\{.*?\n    \});", html, re.S)
    assert match, f"{path.name}: no GREETINGS block — the instant greeting is gone"
    return json.loads(match.group(1))


@pytest.mark.parametrize("relative", CLIENTS)
class TestTheBundledCopyMatchesTheBackend:
    """A stale copy is worse than a slow greeting.

    The bundled string renders instantly and the server's arrives a moment later. When
    they agree the rider sees one steady message; when they have drifted the rider
    watches the assistant change its mind, which reads as a glitch. These clients are
    the reference implementation of the pattern, so they are also where the drift is
    caught.
    """

    def test_every_persona_is_bundled(self, relative):
        bundled = _bundled_greetings(REPO_ROOT / relative)
        assert set(bundled) == set(WELCOME_MESSAGES)

    def test_each_string_is_exact(self, relative):
        bundled = _bundled_greetings(REPO_ROOT / relative)
        for persona, text in WELCOME_MESSAGES.items():
            assert bundled[persona] == text, (
                f"{relative}: the bundled '{persona}' greeting has drifted from "
                f"app/services/chat_session.py. Update the GREETINGS block."
            )

    def test_the_greeting_is_rendered_before_start_is_called(self, relative):
        """The whole latency win is the ORDER. Calling `/start` first and rendering
        afterwards is the behaviour this replaced, and it looks identical in a diff
        unless something checks."""
        html = (REPO_ROOT / relative).read_text(encoding="utf-8")
        assert "renderBundledGreeting(" in html
        assert "reconcileGreeting(" in html
        # The old shape: the first bubble built straight out of the /start payload.
        assert 'addMessage("assistant", data.message)' not in html


# ── 3. The bootstrap has to actually run ──────────────────────────────────────

_BOOT_CALL_RE = re.compile(
    r"(?:renderBundledGreeting|checkGreetingDrift)\(\s*([^)]*?)\s*\)"
)
#: A raw string on purpose: written as a normal one, the trailing word-boundary
#: escape becomes a literal backspace character, the pattern then matches nothing,
#: and every page looks broken.
_DECLARED_RE = r"(?:const|let|var)\s+{name}\b"


@pytest.mark.parametrize("relative", CLIENTS)
class TestTheGreetingBootstrapCanRun:
    """The instant greeting is worth nothing if the line that draws it throws.

    index.html was patched with `renderBundledGreeting(PERSONA)` — but that page has no
    PERSONA variable, because it IS the outstation client and nothing can switch it.
    The ReferenceError stopped the two statements under it, so `startChat()` never ran:
    the page opened blank and never asked the rider anything. Reported as "outstation
    does not initiate with the initial message and does not take the parameters".

    The test above this one passed throughout, because it only checked that the string
    "renderBundledGreeting(" appeared in the file. Asserting that a call is PRESENT
    says nothing about whether it can execute, so this asserts what it is called with.
    """

    def test_every_greeting_call_is_given_something_that_exists(self, relative):
        html = (REPO_ROOT / relative).read_text(encoding="utf-8")

        for raw_arg in _BOOT_CALL_RE.findall(html):
            arg = raw_arg.strip()
            if not arg or arg.startswith(('"', "'")):
                continue                      # a literal always resolves
            if arg == "persona":
                continue                      # the parameter inside the definitions
            assert re.search(_DECLARED_RE.format(name=re.escape(arg)), html), (
                f"{relative}: greeting bootstrap is passed {arg!r}, which the page "
                f"never declares — that is a ReferenceError, and it stops the "
                f"statements after it from running at all."
            )

    def test_the_persona_it_greets_as_is_the_one_it_chats_as(self, relative):
        """A page that greets as one persona and calls another's endpoints would show
        the right opening line and then behave like a different product."""
        html = (REPO_ROOT / relative).read_text(encoding="utf-8")
        expected = {
            "frontend/index.html": "outstation",
            "frontend/local.html": "local",
            "frontend/rental.html": "rental",
        }[relative]
        assert f"/v1/chat/{expected}" in html
