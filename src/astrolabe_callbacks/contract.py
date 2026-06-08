"""Astrolabe contract between engine and callback.

Source of truth: this file in the engine repo. Vendored verbatim into
``astrolabe-callbacks`` (and any third-party callback library) via
``tools/vendor-contract.py``.

This file holds **identifiers only** — the env var names, Aim tag names,
metric namespaces, and default values that participate in the contract.
Behavioral expectations live in ``docs/callback-contract.md``.

Rules enforced by CI:

- Stdlib-only imports. Adding any third-party import is a contract
  violation; ``tools/check-contract-stdlib-only.py`` blocks merge.
- Modifying this file requires bumping ``CONTRACT_VERSION``;
  ``tools/check-contract-bump.py`` blocks merge.
- Every ``ASTROLABE_*``/``AIM_*`` env var the engine sets and every
  ``astrolabe.*`` Aim tag the engine reads must appear as a constant
  here; ``tests/test_contract_completeness.py`` enforces.

See ``plans/version-contract.md`` for the operating model.
"""

from __future__ import annotations

# Contract version (semver).
#
# Bump pattern:
#   MAJOR — breaking change (rename or remove an env var / tag)
#   MINOR — backward-compatible addition (new optional env var / tag)
#   PATCH — doc-only / comment-only edit
#
# A callback library's vendored copy carries the engine version it was
# vendored from; the engine refuses submits whose pinned callback was
# vendored against a contract older than what this engine version
# requires.
CONTRACT_VERSION = "1.0.0"

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

# JSON-encoded mapping of Aim run tags the engine wants applied to
# every run produced under this submit. Callbacks merge this into
# whatever tags they apply locally. Shape: ``{"key": "value", ...}``.
ENV_AIM_RUN_TAGS = "AIM_RUN_TAGS"

# Filesystem path to the local Aim repo the callback should write
# through. Set only when the NUC has ``aim_local_mode: true`` in
# ``/etc/astrolabe/config.yaml`` (v1.7.0+). When unset, callbacks fall
# back to the tunneled Aim server at ``aim://localhost:43800``.
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

# Default Aim repo path template (local-aim mode, v1.7.0+). The engine
# substitutes the actual submit_id when writing the env var.
LOCAL_AIM_REPO_PATH_TEMPLATE = "/tmp/aim-local-{submit_id}"
