"""The three test clients must at least PARSE.

`TestTheGreetingBootstrapCanRun` in test_greeting.py exists because a page was shipped
with `renderBundledGreeting(PERSONA)` where no PERSONA was declared: a ReferenceError
that stopped every statement after it, so the page opened blank. That test catches an
undeclared identifier in the one call it knows about.

This catches the wider version of the same mistake — a syntax error anywhere in a page.
A browser does not report one anywhere the developer will see it; the script simply
does not run, and the page is a dead shell that still serves 200. These files are
edited by hand from Python, which is exactly the circumstance that produces an unclosed
template literal or a stray brace, so the check is cheap insurance.

Skipped when node is unavailable, so the suite still runs on a machine without it.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess

import pytest

CLIENTS = (
    "frontend/index.html",
    "frontend/local.html",
    "frontend/rental.html",
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Inline scripts only. A `src=` script is a file of its own and is not in these pages.
_SCRIPT_RE = re.compile(
    r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", re.DOTALL | re.IGNORECASE
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


@pytest.mark.parametrize("relative", CLIENTS)
def test_every_inline_script_parses(relative: str, tmp_path: pathlib.Path):
    html = (REPO_ROOT / relative).read_text(encoding="utf-8")
    scripts = _SCRIPT_RE.findall(html)
    assert scripts, f"{relative}: no inline script found — did the page change shape?"

    for index, source in enumerate(scripts):
        # `node --check` parses without executing, so browser globals (document,
        # fetch, navigator) are irrelevant here — only the syntax is.
        path = tmp_path / f"{pathlib.Path(relative).stem}_{index}.js"
        path.write_text(source, encoding="utf-8")
        result = subprocess.run(
            ["node", "--check", str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, (
            f"{relative}: inline script #{index} does not parse — the browser would "
            f"run none of it and the page would be a dead shell.\n{result.stderr}"
        )
