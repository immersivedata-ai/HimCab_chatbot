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
import subprocess
import sys
import textwrap

import pytest
from fastapi.testclient import TestClient

from app.main import app

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
