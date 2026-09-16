"""
tests/unit/api/test_browser_clients.py
───────────────────────────────────────
The three browser test clients, and the switch that takes them away.

`frontend/` holds one single-file page per workflow — outstation trips, local rides and
hourly rentals. They call the same public, rider-authenticated endpoints any client
calls, so they are the only way to drive a whole conversation by hand, and they ship in
the image for exactly that reason.

They are also a page that looks like the product sitting on the same origin as the API.
That is fine while a team is testing a flow and wrong once the URL is a contract
somebody integrates against — so `app/main.py` serves them on any non-production
`APP_ENV` and nothing at all on production. Both halves are asserted here, because the
half that matters is the one nobody exercises: a guard that has quietly stopped
guarding looks identical from the inside.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: path → the page it must serve, one per workflow.
PAGES = {
    "/outstation": "index.html",
    "/local": "local.html",
    "/rental": "rental.html",
}


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestTheClientsAreServedForTesting:
    """The default. The suite runs with APP_ENV unset, which is development."""

    @pytest.mark.parametrize("path", sorted(PAGES))
    def test_each_workflow_has_a_page(self, client, path):
        response = client.get(path)
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    @pytest.mark.parametrize("path,filename", sorted(PAGES.items()))
    def test_the_right_page_is_behind_each_path(self, client, path, filename):
        """A route pointing at the wrong file is a 200 that wastes a tester's time —
        they drive a whole conversation before noticing it was the wrong persona."""
        from pathlib import Path

        expected = (
            Path(__file__).resolve().parents[3] / "frontend" / filename
        ).read_text(encoding="utf-8")
        assert client.get(path).text == expected

    def test_the_root_names_them(self, client):
        """The usual reason somebody opens the bare URL is to find these."""
        body = client.get("/").json()
        assert body["test_clients"] == {
            "outstation": "/outstation",
            "local": "/local",
            "rental": "/rental",
        }

    def test_the_static_mount_is_there_too(self, client):
        """So a page can pull a sibling asset if one is ever added next to it."""
        assert client.get("/frontend/index.html").status_code == 200


def _app_under(app_env: str) -> dict:
    """Boot the app in a subprocess with `APP_ENV` set, and report what it serves.

    A subprocess because `settings` is an lru_cached singleton read at import time —
    monkeypatching it after the fact would change the object without re-running the
    route registration that consulted it, which would test nothing.
    """
    script = textwrap.dedent(
        """
        import json
        from fastapi.testclient import TestClient
        from app.main import app
        from app.core.config import settings

        client = TestClient(app)
        print(json.dumps({
            "is_production": settings.is_production,
            "codes": {p: client.get(p).status_code
                      for p in ("/outstation", "/local", "/rental", "/frontend/index.html")},
            "root": client.get("/").json(),
        }))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "APP_ENV": app_env},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestProductionServesTheApiAlone:
    """The promise made when this URL was handed to an integrating team.

    Nothing that looks like the product, and no second copy of the flow to keep in step
    with the contract — enforced by configuration, so turning it off needs no code
    change and no rebuild.
    """

    @pytest.fixture(scope="class")
    def served(self) -> dict:
        return _app_under("production")

    def test_it_really_is_production(self, served):
        assert served["is_production"] is True

    @pytest.mark.parametrize(
        "path", ["/outstation", "/local", "/rental", "/frontend/index.html"]
    )
    def test_every_page_is_gone(self, served, path):
        assert served["codes"][path] == 404

    def test_the_root_stops_advertising_them(self, served):
        """A signpost pointing at three 404s is worse than no signpost."""
        assert "test_clients" not in served["root"]

    def test_the_api_itself_is_untouched(self, served):
        """The gate removes the pages, not the service."""
        assert served["root"]["health"] == "/v1/health"


# ── The startup path, which is what a cold start pays for ─────────────────────


class TestStartupStaysOffTheCriticalPath:
    """A ~40s cold start was the reported "greeting takes 30-40 seconds".

    This container serves the browser test pages as well as the API, so nothing can be
    fetched — not the HTML, not the bundled greeting inside it — until the application
    has finished importing and starting. The fixes were to stop importing a
    language-model SDK nobody had asked for yet, and to stop the warm-up holding the
    port. These tests hold both, because both are the kind of thing a later import
    quietly undoes.
    """

    def test_the_vertex_sdk_is_not_imported_at_module_scope(self):
        """`vertexai` and `langchain_google_vertexai` are about three seconds of Python
        between them, and importing this module used to pay it. Nothing above the client
        constructor needs the SDK, so nothing above it should wait for one."""
        source = (
            REPO_ROOT / "app" / "services" / "vertex_ai.py"
        ).read_text(encoding="utf-8")
        header = source.split("class VertexAIService")[0]
        for heavy in ("\nimport vertexai", "\nfrom vertexai", "\nfrom langchain_google_vertexai"):
            assert heavy not in header, (
                f"{heavy.strip()!r} is back at module scope in vertex_ai.py — that is "
                f"~3s added to every cold start, before the page can even be served."
            )

    def test_importing_the_app_does_not_drag_the_sdk_in(self):
        """The check that actually matters: whatever the import graph looks like, the
        SDK must not be resident merely because the app was imported."""
        import subprocess
        import sys

        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, app.main; "
             "print('vertexai' in sys.modules or 'langchain_google_vertexai' in sys.modules)"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr[-1500:]
        assert result.stdout.strip() == "False", (
            "importing app.main pulled in the Vertex SDK; a cold start now pays for it "
            "before it can serve a single page"
        )

    def test_the_warm_up_does_not_block_the_port(self):
        """uvicorn does not accept a connection until startup returns, so anything slow
        left inline there queues every request behind it — including the ones that need
        none of it."""
        source = (REPO_ROOT / "app" / "main.py").read_text(encoding="utf-8")
        lifespan = source.split("async def lifespan")[1].split("yield")[0]
        assert "create_task" in lifespan, "the warm-up is inline again"

    def test_the_container_runs_one_worker(self):
        """Cloud Run scales by adding INSTANCES. Extra workers inside one instance each
        import the whole app and contend for the single vCPU it gets by default, which
        multiplies the cold start rather than serving anyone faster."""
        dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "--workers 1" in dockerfile
        assert "--workers 4" not in dockerfile

    def test_the_deploy_keeps_an_instance_warm(self):
        """The only thing that removes a cold start entirely, rather than shortening it."""
        deploy = (REPO_ROOT / "deploy.ps1").read_text(encoding="utf-8", errors="replace")
        assert "--min-instances=1" in deploy
        assert "--cpu-boost" in deploy
