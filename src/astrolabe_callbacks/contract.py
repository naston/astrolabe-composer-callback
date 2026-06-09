"""Astrolabe contract between engine and callback.

Source of truth: this file in the engine repo. Vendored verbatim into
``astrolabe-callbacks`` (and any third-party callback library) via
``tools/vendor-contract.py``.

This file holds the **names** (env vars, Aim tag keys, metric
namespaces, default values) AND the **canonical formatters/parsers**
for any value with a non-trivial encoding. Both sides of the contract
go through the same helpers so the wire format can't drift across the
engine/callback split.

Behavioral expectations live in ``docs/callback-contract.md``.

Rules enforced by CI:

- Stdlib-only imports. Adding any third-party import is a contract
  violation; ``tools/check-contract-stdlib-only.py`` blocks merge.
- Modifying this file requires bumping ``CONTRACT_VERSION``;
  ``tools/check-contract-bump.py`` blocks merge.
- Every ``ASTROLABE_*``/``AIM_*`` env var the engine sets and every
  ``astrolabe.*`` Aim tag the engine reads must appear as a constant
  here; ``tests/test_contract_completeness.py`` enforces.
- Bare contract-literal strings outside this file are a violation;
  the engine must route through ``contract.ENV_*``/``contract.TAG_*``
  + the format/parse helpers (no inline ``json.dumps`` of contract
  values, no inline ``"astrolabe.user"`` keys).

See ``plans/version-contract.md`` for the operating model.
"""

from __future__ import annotations

# Contract version (semver).
#
# Bump pattern:
#   MAJOR — breaking change (rename or remove an env var / tag, change
#           the wire format of a value)
#   MINOR — backward-compatible addition (new optional env var / tag,
#           new helper)
#   PATCH — doc-only / comment-only edit
#
# A callback library's vendored copy carries the engine version it was
# vendored from; the engine refuses submits whose pinned callback was
# vendored against a contract older than what this engine version
# requires.
CONTRACT_VERSION = "1.1.0"

# --- Env vars: ENGINE sets in the training process -------------------------
#
# The engine writes these into the training process's environment before
# the training command runs. A contract-compliant callback reads them
# (directly or via helpers) to wire itself to the orchestration.

# Unique identifier for one submit (one `astrolabe submit` invocation).
# Callbacks tag the Aim run with this so the dashboard can link the run
# back to the submit row in the state DB.
ENV_SUBMIT_ID = "ASTROLABE_SUBMIT_ID"

# Human-readable experiment name from the YAML. Callbacks may use this
# as the Aim run name when one isn't set explicitly.
ENV_EXPERIMENT_NAME = "ASTROLABE_EXPERIMENT_NAME"

# Tag dict the engine wants applied to every run produced under this
# submit. Wire format: ``key1=val1,key2=val2`` (NOT JSON — keys and
# values are pasted directly into the env var, comma-separated). Use
# :func:`format_aim_run_tags` and :func:`parse_aim_run_tags` to read
# and write this — never inline the encoding.
ENV_AIM_RUN_TAGS = "AIM_RUN_TAGS"

# Filesystem path to the local Aim repo the callback should write
# through. Set only when the NUC has ``aim_local_mode: true`` in
# ``/etc/astrolabe/config.yaml`` (v1.7.0+). When unset, callbacks fall
# back to the tunneled Aim server at ``aim://localhost:43800``.
# Engine constructs the value via :func:`format_local_aim_repo_path`.
ENV_AIM_REPO_PATH = "ASTROLABE_AIM_REPO_PATH"

# Path to a jsonl file the callback appends structured events to (run
# open/close, schema finalize, dropped batches, etc.). Used by the
# canary verifier harness for cross-checking claimed side effects.
ENV_CALLBACK_STATS_PATH = "ASTROLABE_CALLBACK_STATS_PATH"

# Directory the engine has provisioned for per-rank stdout/stderr logs
# during distributed training. Callbacks (and frameworks) write rank-N
# logs into ``$ASTROLABE_RANK_LOGS_DIR/rank-N.{stdout,stderr}``.
ENV_RANK_LOGS_DIR = "ASTROLABE_RANK_LOGS_DIR"

# --- Aim run tags: CALLBACK writes, ENGINE + dashboard read ----------------
#
# Callbacks apply these to the Aim run at open time. The engine reads
# them back via the Aim API (state DB lookups, dashboard rendering,
# Linear report generation, etc.). Renaming any of these is a MAJOR
# contract bump.

TAG_SUBMIT_ID = "astrolabe.submit_id"
TAG_USER = "astrolabe.user"
TAG_VERSION = "astrolabe.version"
TAG_EXPERIMENT = "astrolabe.experiment"
TAG_GPU_TYPE = "astrolabe.gpu_type"
TAG_GPU_RATE_CENTS_PER_HOUR = "astrolabe.gpu_rate_cents_per_hour"

# Final outcome — set by the engine on terminal state ("success" /
# "failure" / "cancelled"). Callbacks don't write this; it's listed
# here so engine code that reads it goes through the constant.
TAG_OUTCOME = "astrolabe.outcome"

# --- Metric namespaces -----------------------------------------------------
#
# Conventions, not strictly enforceable from the engine side, but the
# dashboard groups metrics by these prefixes.

NAMESPACE_TRAIN = "train/"           # during-training metrics
NAMESPACE_VAL = "val/"               # during-training validation
NAMESPACE_EVAL = "eval/"             # post-training benchmarks

# Engine-synthesized metric: wall-clock time at each step. Callbacks
# don't write this themselves; the engine derives it from Aim's
# per-step timestamps when the Go API serves the run.
SYNTHESIZED_WALL_TIME = "wall_time"

# --- Defaults --------------------------------------------------------------

# Default Aim repo path template (local-aim mode, v1.7.0+). Substitute
# the submit_id via :func:`format_local_aim_repo_path` — do not call
# ``LOCAL_AIM_REPO_PATH_TEMPLATE.format(...)`` directly at call sites.
LOCAL_AIM_REPO_PATH_TEMPLATE = "/tmp/aim-local-{submit_id}"

# --- Canonical formatters / parsers ---------------------------------------
#
# Both engine and callback go through these. The wire format lives in
# exactly one place, so a change to the encoding requires editing this
# file — which triggers the CONTRACT_VERSION-bump CI guard.
#
# Why these specifically: only values with non-trivial encodings need a
# helper. A constant like ``ENV_SUBMIT_ID`` is just a name; the value
# is just a string passed through, no encoding involved. ``AIM_RUN_TAGS``
# encodes a dict into a single string, and ``ASTROLABE_AIM_REPO_PATH``
# templates a submit_id into a path — both are encodings, both need
# canonical helpers.


def format_aim_run_tags(tags: dict[str, str]) -> str:
    """Encode a tag dict into the ``AIM_RUN_TAGS`` wire format.

    Wire format is ``key1=val1,key2=val2``. Keys and values are
    inserted literally — callers must not include ``=`` or ``,`` in
    keys or values. In practice astrolabe's tag keys are all
    ``astrolabe.*`` literals (no ``=`` or ``,``) and values are
    submit_ids / version labels / GPU types / integer rates (none of
    which contain those characters either).

    Parameters
    ----------
    tags : dict[str, str]
        The tag dict to encode.

    Returns
    -------
    str
        The env-var-shaped wire format. ``""`` for an empty dict.
    """
    return ",".join(f"{k}={v}" for k, v in tags.items())


def parse_aim_run_tags(raw: str | None) -> dict[str, str]:
    """Decode the ``AIM_RUN_TAGS`` wire format into a tag dict.

    Inverse of :func:`format_aim_run_tags`. Forgiving rather than
    strict — this reads a value a researcher may have pasted into a
    shell, not a machine-validated payload. Entries without ``=``,
    with empty keys, or duplicate keys (last wins) are tolerated.
    Whitespace around keys/values is stripped.

    Parameters
    ----------
    raw : str | None
        The raw env var value, or ``None``/empty.

    Returns
    -------
    dict[str, str]
        Parsed tags. ``{}`` if ``raw`` is empty or unparseable.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        out[key] = value
    return out


def format_local_aim_repo_path(submit_id: str) -> str:
    """Construct the per-submit local Aim repo path.

    Substitutes ``submit_id`` into :data:`LOCAL_AIM_REPO_PATH_TEMPLATE`.
    Engine sets the env var via this helper in local-aim mode.
    """
    return LOCAL_AIM_REPO_PATH_TEMPLATE.format(submit_id=submit_id)
