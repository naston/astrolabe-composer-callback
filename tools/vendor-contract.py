#!/usr/bin/env python3
"""Vendor ``astrolabe/contract.py`` from the engine repo.

The engine↔callback contract lives in the engine repo as the source of
truth. Every callback library — first-party (this one) and any
third-party — vendors a verbatim copy. This script does the copying.

Why vendoring instead of a published ``astrolabe-contract`` package:
keeps the project at three packages (engine, callback, dashboard), and
keeps the training environment free of the engine library and its
dependencies.

Usage::

    python tools/vendor-contract.py

Reads the pinned engine ref from ``tools/vendor-contract.json``,
downloads ``astrolabe/contract.py`` at that ref from GitHub, validates
it's stdlib-only, and writes it to
``src/astrolabe_callbacks/contract.py``.

To update which engine version is vendored, edit
``tools/vendor-contract.json`` and re-run.

See ``plans/version-contract.md`` in the astrolabe repo (the section
"Vendoring mechanism") for the full operating model.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIDECAR_PATH = REPO_ROOT / "tools" / "vendor-contract.json"
DEST_PATH = REPO_ROOT / "src" / "astrolabe_callbacks" / "contract.py"

# Default GitHub raw URL template. ``{ref}`` is the engine ref (tag,
# branch, or SHA). The path part is fixed because ``contract.py`` lives
# at a known location in the engine repo.
DEFAULT_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/naston/astrolabe/{ref}/astrolabe/contract.py"
)

# Python 3.10+ stdlib module names — sourced from sys.stdlib_module_names
# so we don't have to maintain a list.
STDLIB_MODULES = set(sys.stdlib_module_names) | {"__future__"}


class VendorError(Exception):
    """Raised when vendoring can't proceed (network, validation, write)."""


def _load_sidecar() -> tuple[str, str]:
    """Return (engine_ref, url_template) from the sidecar.

    Sidecar shape::

        {
          "vendored_from": "astrolabe@v1.7.0",
          "url_template": "https://..."  // optional, has a sensible default
        }

    The ``vendored_from`` string is ``"astrolabe@<ref>"`` where ``<ref>``
    is any git ref the GitHub raw endpoint accepts (tag, branch, SHA).
    """
    if not SIDECAR_PATH.exists():
        raise VendorError(
            f"sidecar {SIDECAR_PATH.relative_to(REPO_ROOT)} not found. "
            f"Create it with: {{\"vendored_from\": \"astrolabe@<ref>\"}}"
        )
    data = json.loads(SIDECAR_PATH.read_text())
    vendored_from = data.get("vendored_from")
    if not isinstance(vendored_from, str) or "@" not in vendored_from:
        raise VendorError(
            f"sidecar 'vendored_from' must be a string of the form "
            f"'astrolabe@<ref>'; got {vendored_from!r}"
        )
    _, ref = vendored_from.split("@", 1)
    if not ref:
        raise VendorError("sidecar 'vendored_from' has empty ref")
    url_template = data.get("url_template", DEFAULT_URL_TEMPLATE)
    return ref, url_template


def _download(url: str) -> str:
    """Fetch the contract.py source at the given URL.

    The astrolabe repo is private, so we attach a GitHub token from
    ``$GITHUB_TOKEN`` (or ``$GH_TOKEN``) when present. In GitHub
    Actions, the workflow's ``GITHUB_TOKEN`` provides repo-read scope.
    Locally, a personal access token works fine.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status != 200:
                raise VendorError(f"HTTP {resp.status} fetching {url}")
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise VendorError(
                f"404 from {url}.\n"
                f"  - Confirm the ref exists at "
                f"https://github.com/naston/astrolabe/tree/<ref>\n"
                f"  - The astrolabe repo is private; export GITHUB_TOKEN "
                f"with repo:read scope (or GH_TOKEN). gh CLI auth works: "
                f"export GITHUB_TOKEN=$(gh auth token)"
            ) from exc
        raise VendorError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except Exception as exc:
        raise VendorError(f"failed to fetch {url}: {exc}") from exc


def _validate_stdlib_only(src: str) -> None:
    """Parse imports; refuse if any non-stdlib module is referenced.

    The engine's own CI enforces stdlib-only at PR time (see the engine
    repo's ``tools/check-contract-stdlib-only.py``). We re-validate here
    as defense in depth — a malicious or buggy mirror of the contract
    file would be caught before it lands in our package.
    """
    tree = ast.parse(src)
    non_stdlib: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top not in STDLIB_MODULES:
                    non_stdlib.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0 or node.module is None:
                continue
            top = node.module.split(".")[0]
            if top not in STDLIB_MODULES:
                non_stdlib.append(node.module)
    if non_stdlib:
        raise VendorError(
            f"vendored contract.py has non-stdlib imports: {non_stdlib}. "
            f"Refusing to write."
        )


def _extract_version(src: str) -> str:
    m = re.search(r'^CONTRACT_VERSION\s*=\s*"([^"]+)"', src, re.MULTILINE)
    if not m:
        raise VendorError("vendored contract.py has no CONTRACT_VERSION")
    return m.group(1)


def main() -> int:
    try:
        ref, url_template = _load_sidecar()
    except VendorError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    url = url_template.format(ref=ref)
    print(f"Downloading {url}")
    try:
        src = _download(url)
        _validate_stdlib_only(src)
        version = _extract_version(src)
    except VendorError as exc:
        print(f"✗ {exc}", file=sys.stderr)
        return 1

    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEST_PATH.write_text(src)
    rel = DEST_PATH.relative_to(REPO_ROOT)
    print(f"✓ Vendored astrolabe/contract.py from astrolabe@{ref}")
    print(f"  → {rel}")
    print(f"  CONTRACT_VERSION: {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
