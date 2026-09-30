"""
tests/unit/test_architecture.py
────────────────────────────────
The layering, enforced.

A directory structure documents an intention; only a test keeps it true. Every rule
below was a real coupling before the packages were split, and each would come back the
first time someone reaches for the nearest import that makes their line work:

  • an entity importing a response model, so a field added for the app to render
    quietly becomes a field that is persisted
  • a domain object importing an HTTP client, so constructing one needs a network
  • the agent importing FastAPI, so the graph can only run inside a request

These tests read imports statically (`ast`), so they cost nothing and cannot be
defeated by a module that happens not to be imported at runtime.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

APP = pathlib.Path(__file__).resolve().parents[2] / "app"


def _modules(package: str) -> list[pathlib.Path]:
    return sorted(p for p in (APP / package).rglob("*.py") if "__pycache__" not in str(p))


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every `app.*` module name this file imports, however it spells the import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("app."))
        elif (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.level == 0
            and node.module.startswith("app.")
        ):
            found.add(node.module)
    return found


def _violations(package: str, forbidden: tuple[str, ...]) -> list[str]:
    out = []
    for path in _modules(package):
        for imported in sorted(_imported_modules(path)):
            if any(imported == f or imported.startswith(f + ".") for f in forbidden):
                out.append(f"{path.relative_to(APP.parent)} imports {imported}")
    return out


class TestLayering:
    """Inner layers may not know about outer ones."""

    def test_domain_imports_nothing_outward(self):
        """Entities are the innermost layer.

        A change to an HTTP response, a Firestore collection or a third-party provider
        must not be able to reach in here. `app.core` is allowed — it is the shared
        vocabulary (enums, config, errors), which every layer including this one needs.
        """
        assert _violations(
            "domain",
            ("app.api", "app.schemas", "app.integrations", "app.repositories",
             "app.services", "app.agent"),
        ) == []

    def test_schemas_depend_only_on_domain(self):
        """Wire formats may be built from entities, never from I/O or the agent."""
        assert _violations(
            "schemas",
            ("app.api", "app.integrations", "app.repositories", "app.services", "app.agent"),
        ) == []

    def test_integrations_do_not_import_the_api_layer(self):
        """An HTTP client must be usable without a request in flight."""
        assert _violations("integrations", ("app.api", "app.schemas", "app.agent")) == []

    def test_repositories_do_not_import_the_api_layer(self):
        """Persistence stores entities, not response bodies."""
        assert _violations("repositories", ("app.api", "app.schemas", "app.agent")) == []

    def test_agent_does_not_import_the_api_layer(self):
        """The graph runs from a test, a script or a queue worker — not only a request.

        It may read `app.schemas` for the structured-event shapes it emits, but nothing
        in `app.api` may be needed to execute a turn.
        """
        assert _violations("agent", ("app.api",)) == []


class TestNoWebFrameworkBelowTheApiLayer:
    """FastAPI belongs to delivery. Nothing below it should need the framework."""

    @pytest.mark.parametrize("package", ["domain", "integrations", "repositories", "agent"])
    def test_no_fastapi_import(self, package: str):
        offenders = []
        for path in _modules(package):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                names = (
                    [a.name for a in node.names] if isinstance(node, ast.Import)
                    else [node.module or ""] if isinstance(node, ast.ImportFrom)
                    else []
                )
                if any(n.split(".")[0] in ("fastapi", "starlette") for n in names):
                    offenders.append(str(path.relative_to(APP.parent)))
        assert offenders == []


class TestPackagesAreDocumented:
    """Each package says what belongs in it, so the next person puts it in the right one."""

    @pytest.mark.parametrize(
        "package", ["domain", "schemas", "repositories", "integrations"]
    )
    def test_package_has_a_docstring(self, package: str):
        init = APP / package / "__init__.py"
        assert init.exists(), f"{package} is not a package"
        tree = ast.parse(init.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"app/{package}/__init__.py has no docstring"


class TestNoStalePackages:
    """The packages the restructure replaced must not quietly reappear."""

    @pytest.mark.parametrize("gone", ["models", "services/himcab", "services/external"])
    def test_removed_package_is_absent(self, gone: str):
        assert not (APP / gone).exists(), (
            f"app/{gone} is back — it was split; see app/domain, app/schemas, "
            f"app/integrations"
        )
