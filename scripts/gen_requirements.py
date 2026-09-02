"""
scripts/gen_requirements.py
────────────────────────────
Regenerate requirements.txt and requirements-dev.txt from pyproject.toml.

WHY BOTH FORMS EXIST
────────────────────
pyproject.toml is where a version is decided; the requirements files are what the
Docker build installs, as their own layer, before the source is copied — so a source
change does not invalidate the dependency layer and rebuilds stay fast.

Two files stating the same versions drift, and the drift is silent: the image pins one
version, a developer's venv another, and the difference surfaces as a bug that will not
reproduce locally. So these files are GENERATED, never hand-edited, and CI runs
`--check` to fail a pull request that changes one without the other.

RUNTIME AND DEV ARE SPLIT DELIBERATELY
──────────────────────────────────────
The image installs requirements.txt only, so pytest, ruff, mypy and pre-commit never
reach Cloud Run. Every package in a production image is attack surface and cold-start
weight, and a test runner is neither needed nor wanted there.

    python scripts/gen_requirements.py            # rewrite both files
    python scripts/gen_requirements.py --check    # exit 1 if either is out of date
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS = ROOT / "requirements.txt"
REQUIREMENTS_DEV = ROOT / "requirements-dev.txt"

RULE = "\u2500" * 77

HEADER = (
    f"# {RULE}\n"
    "# GENERATED FILE - do not edit.\n"
    "#\n"
    "# Source of truth is pyproject.toml. Regenerate with:\n"
    "#     python scripts/gen_requirements.py\n"
    "# CI fails if this file and pyproject.toml disagree.\n"
    f"# {RULE}\n"
)


def _render(note: str, packages: list[str]) -> str:
    body = "\n".join(sorted(packages, key=str.lower))
    return f"{HEADER}#\n# {note}\n\n{body}\n"


def _deps() -> tuple[list[str], list[str]]:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    return (
        list(project["dependencies"]),
        list(project.get("optional-dependencies", {}).get("dev", [])),
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync requirements files with pyproject.toml.")
    ap.add_argument(
        "--check",
        action="store_true",
        help="Exit 1 if a requirements file is out of date, instead of writing it.",
    )
    args = ap.parse_args()

    runtime, dev = _deps()
    targets = [
        (REQUIREMENTS, _render(
            "Runtime dependencies. These, and only these, are installed in the image.",
            runtime,
        )),
        (REQUIREMENTS_DEV, _render(
            "Development and testing only. Never installed in the image.\n"
            "# Install with:  pip install -r requirements.txt -r requirements-dev.txt",
            dev,
        )),
    ]

    if args.check:
        stale = [
            path.name
            for path, rendered in targets
            if (path.read_text(encoding="utf-8") if path.exists() else "") != rendered
        ]
        if stale:
            print(f"Out of date with pyproject.toml: {', '.join(stale)}", file=sys.stderr)
            print("Run: python scripts/gen_requirements.py", file=sys.stderr)
            return 1
        print("requirements.txt and requirements-dev.txt are in sync with pyproject.toml.")
        return 0

    for path, rendered in targets:
        path.write_text(rendered, encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
