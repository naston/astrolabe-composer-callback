#!/usr/bin/env python3
"""Vendor ``astrolabe/contract.py`` from the engine repo.

The engine↔callback contract lives in the engine repo as the source of
truth. Every callback library — first-party (this one) and any
third-party — vendors a verbatim copy. This script does the copying
and records the file's sha256 alongside the source ref.

Why hash-pinning (not a CI network fetch): the engine repo is private,
which would require the callbacks-repo CI to hold a cross-repo
read-token. That violates the spirit of the three-package boundary
work — callbacks shouldn't depend on the engine repo at runtime OR at
build time once the file has been vendored. The sha256 in
``vendor-contract.json`` IS the integrity check: CI verifies the
in-tree file matches the recorded hash. Hand-edits, accidental
corruption, or any drift from the canonical contents fails the check
without ever touching the network.

Re-vendoring is a deliberate maintainer action. To pick up a newer
contract from the engine repo:

    1. Edit ``tools/vendor-contract.json`` — bump ``vendored_from`` to
       the new engine ref (e.g. ``astrolabe@v1.8.0``).
    2. Run ``GITHUB_TOKEN=$(gh auth token) python tools/vendor-contract.py``.
    3. Commit ``src/astrolabe_callbacks/contract.py`` + the updated
       ``tools/vendor-contract.json`` (the script rewrites the hash).

CI never runs this script; it only verifies the recorded hash.

See ``plans/version-contract.md`` in the astrolabe repo for the full
operating model.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import urllib.error
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
          "sha256": "<64-char hex>",       // rewritten by this script
          "url_template": "https://..."    // optional, has a default
        }
    """
    if not SIDECAR_PATH.exists():
        raise VendorError(
            f"sidecar {SIDECAR_PATH.relative_to(REPO_ROOT)} not found. "
            f"Create it with: "
            f'{{"vendored_from": "astrolabe@<ref>", "sha256": ""}}'
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
    ``$GITHUB_TOKEN`` (or ``$GH_TOKEN``) when present. Locally:
    ``export GITHUB_TOKEN=$(gh auth token)``.

    This is only ever run by a maintainer at re-vendor time — CI does
    not invoke this function (CI verifies the recorded sha256 only).
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
                f"with repo:read scope (e.g. "
                f"export GITHUB_TOKEN=$(gh auth token))"
            ) from exc
        raise VendorError(f"HTTP {exc.code} fetching {url}: {exc.reason}") from exc
    except Exception as exc:
        raise VendorError(f"failed to fetch {url}: {exc}") from exc


def _validate_stdlib_only(src: str) -> None:
    """Parse imports; refuse if any non-stdlib module is referenced.

    Defense in depth — the engine's CI guards this at PR time too. If
    a vendored copy ever shows a non-stdlib import, that's either a
    broken engine PR getting merged or a man-in-the-middle on the
    GitHub fetch; either way, refusing is the right move.
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


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    digest = _sha256(src)
    DEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEST_PATH.write_text(src)

    # Rewrite the sidecar with the new hash. Preserve any other keys
    # (url_template, _comment, etc.) the maintainer set by hand.
    data = json.loads(SIDECAR_PATH.read_text())
    data["vendored_from"] = f"astrolabe@{ref}"
    data["sha256"] = digest
    SIDECAR_PATH.write_text(json.dumps(data, indent=2) + "\n")

    rel = DEST_PATH.relative_to(REPO_ROOT)
    print(f"✓ Vendored astrolabe/contract.py from astrolabe@{ref}")
    print(f"  → {rel}")
    print(f"  CONTRACT_VERSION: {version}")
    print(f"  sha256: {digest}")
    print("")
    print("Commit both files:")
    print(f"  git add {rel} tools/vendor-contract.json")
    print(f"  git commit -m 'chore: re-vendor contract.py from astrolabe@{ref}'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
