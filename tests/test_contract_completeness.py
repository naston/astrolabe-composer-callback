"""Completeness tests for the vendored ``contract`` module.

Mirror of the engine repo's
``tests/test_contract_completeness.py``, applied here on the
callback-consumer side. Guards the same drift class:

- The vendored contract module loads (catches install / vendor
  corruption).
- No raw contract-literal strings appear in callback code outside
  the vendored ``contract.py`` itself — every consumer routes through
  ``contract.ENV_*`` / ``contract.TAG_*`` + the helpers.

See ``plans/version-contract.md`` in the engine repo for the
operating model. The intra-repo hash check
(``.github/workflows/contract-sync.yml``) is a separate concern —
it catches hand-edits to the vendored file at PR time. This test
catches code that bypasses the vendored constants.
"""

from __future__ import annotations

import re
from pathlib import Path

from astrolabe_callbacks import contract


REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "src" / "astrolabe_callbacks" / "contract.py"


def _callback_python_files() -> list[Path]:
    """All .py files under src/, excluding the vendored contract.py."""
    src_root = REPO_ROOT / "src"
    return [
        p for p in src_root.rglob("*.py")
        if p.resolve() != CONTRACT_PATH.resolve()
    ]


def _contract_env_values() -> set[str]:
    return {v for name, v in vars(contract).items()
            if name.startswith("ENV_") and isinstance(v, str)}


def _contract_tag_values() -> set[str]:
    return {v for name, v in vars(contract).items()
            if name.startswith("TAG_") and isinstance(v, str)}


def test_contract_module_exposes_expected_constants():
    """The vendored module must define the constants callback code
    depends on. Catches the case where the vendored file is corrupted
    or pinned to an engine version that predates a constant we now
    require.
    """
    assert hasattr(contract, "CONTRACT_VERSION")
    assert hasattr(contract, "ENV_AIM_RUN_TAGS")
    assert hasattr(contract, "ENV_EXPERIMENT_NAME")
    assert hasattr(contract, "ENV_CALLBACK_STATS_PATH")
    assert hasattr(contract, "ENV_AIM_REPO_PATH")
    assert hasattr(contract, "format_aim_run_tags")
    assert hasattr(contract, "parse_aim_run_tags")


def test_no_raw_contract_literals_outside_contract():
    """Callback code (excluding the vendored contract.py) must not
    inline contract literals — every reference goes through
    ``contract.ENV_X`` / ``contract.TAG_X`` + the format/parse
    helpers.

    Mirrors the engine repo's test of the same name. Without this,
    a callback consumer could read the env var via a bare literal
    and parse the value with an inline format, sidestepping the
    vendored contract entirely.
    """
    contract_strings = _contract_env_values() | _contract_tag_values()
    pat = re.compile(r'["\']([A-Z_a-z.]+)["\']')
    violations: list[str] = []
    for path in _callback_python_files():
        src = path.read_text()
        for lineno, line in enumerate(src.splitlines(), start=1):
            for m in pat.finditer(line):
                literal = m.group(1)
                if literal in contract_strings:
                    rel = path.relative_to(REPO_ROOT)
                    violations.append(f"{rel}:{lineno}: bare literal {literal!r}")
    assert not violations, (
        "Found raw contract literals outside src/astrolabe_callbacks/"
        "contract.py:\n  "
        + "\n  ".join(violations)
        + "\n\nReplace each bare literal with the constant "
          "(contract.ENV_X / contract.TAG_X) and route any value "
          "encoding through contract.format_*() / contract.parse_*(). "
          "If the literal is callback-internal (not part of the engine "
          "contract), choose a different name to avoid collision."
    )


def test_contract_version_is_semver():
    """CONTRACT_VERSION must look like MAJOR.MINOR.PATCH."""
    assert re.match(
        r"^\d+\.\d+\.\d+(?:-[a-zA-Z0-9.]+)?$",
        contract.CONTRACT_VERSION,
    ), f"CONTRACT_VERSION={contract.CONTRACT_VERSION!r} is not semver"
