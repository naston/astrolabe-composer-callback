"""Check whether a dependency has a newer version than what we've tested.

Used by ``.github/workflows/dep-watch.yml``. Hits PyPI's JSON API,
compares the latest non-prerelease version against ``tested-versions.json``,
and prints two outputs to GitHub Actions:

- ``has_new_version=true|false``
- ``latest=<version>``

If a newer version exists, the workflow installs it, re-runs the
relevant integration tests, and either commits the updated
``tested-versions.json`` (on pass) or opens a ``dep-drift`` issue (on
fail). See the workflow file for the full plumbing.

Usage::

    python scripts/check_dep_drift.py <package_name>

Exit code is always 0 (so the workflow proceeds even when no newer
version exists). Decisions are made via the printed outputs, not
exit codes.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VERSIONS_FILE = REPO_ROOT / "tested-versions.json"


def load_tested_versions() -> dict:
    with VERSIONS_FILE.open() as f:
        return json.load(f)


def fetch_latest_pypi_version(package: str) -> str:
    """Return the latest non-prerelease version on PyPI."""
    url = f"https://pypi.org/pypi/{package}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = json.load(resp)
    # ``data["info"]["version"]`` is the latest *stable* release per
    # PyPI. Prereleases are excluded by default; this is what we want
    # — drift-watching against alpha/RC builds would produce noise.
    return data["info"]["version"]


def parse_version(v: str) -> tuple[int, ...]:
    """Naive PEP 440 sort key — works for the common ``X.Y.Z`` shape."""
    parts = []
    for chunk in v.split("."):
        # Strip any non-numeric suffix (e.g. "1.0.0rc1" → "1.0.0r" — we
        # discard prereleases but defensively handle stray characters).
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def is_known_bad(version: str, known_bad: dict, package: str) -> bool:
    bad_versions = known_bad.get(package, [])
    if isinstance(bad_versions, str):
        bad_versions = [bad_versions]
    return version in bad_versions


def write_github_output(key: str, value: str) -> None:
    """Append to GITHUB_OUTPUT so the workflow can read it via ``steps.<id>.outputs.<key>``."""
    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a") as f:
            f.write(f"{key}={value}\n")
    else:
        # Local dev — print to stdout so a human running this can see.
        print(f"::{key}={value}")


def main(package: str) -> None:
    tested = load_tested_versions()
    last_tested = tested.get(package)
    if last_tested is None:
        print(f"WARNING: {package} not tracked in tested-versions.json")
        write_github_output("has_new_version", "false")
        write_github_output("latest", "")
        return

    latest = fetch_latest_pypi_version(package)
    write_github_output("latest", latest)
    write_github_output("last_tested", last_tested)

    known_bad = tested.get("_known_bad", {})
    if is_known_bad(latest, known_bad, package):
        print(
            f"{package} {latest} is in _known_bad; skipping drift test"
        )
        write_github_output("has_new_version", "false")
        return

    if parse_version(latest) > parse_version(last_tested):
        print(f"{package}: new version {latest} > tested {last_tested}")
        write_github_output("has_new_version", "true")
    else:
        print(f"{package}: latest {latest} ≤ tested {last_tested}")
        write_github_output("has_new_version", "false")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_dep_drift.py <package>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
