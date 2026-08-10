"""
app/prompts/builder.py
───────────────────────
Prompt assembly for the Router node — loads its YAML template so prompt text
changes don't require code changes.

The Planner and Writer nodes build their own prompts inline (see
app/agent/nodes/planner.py, app/agent/nodes/writer.py) since their prompts need
dynamic per-request context (entities, weather, fares, language) that's simpler
to assemble alongside the node logic itself.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_TEMPLATES_DIR = Path(__file__).parent / "templates"


@lru_cache(maxsize=10)
def _load_template(name: str) -> dict[str, Any]:
    """Load and cache a YAML template by name (without extension)."""
    path = _TEMPLATES_DIR / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_router_prompt() -> str:
    """
    Build the system prompt for the Router node.
    Returns the static classification prompt (no dynamic state needed).
    """
    tmpl = _load_template("router_system")
    return tmpl["system_prompt"]
