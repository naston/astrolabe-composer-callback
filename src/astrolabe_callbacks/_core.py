"""Shared spine for every framework callback.

Every per-framework callback (Composer, Lightning, HuggingFace, raw
PyTorch) is a thin shell that delegates to this module. Putting the
contract here means a fix or feature lands in one place and propagates
to every framework automatically.

Failure handling contract
-------------------------

The default posture of every astrolabe-callbacks callback is **graceful
degradation, not training crashes.** A misconfigured Aim URL, a network
blip, or a missing optional dependency must never take down a training
job. The escape hatch is the ``ASTROLABE_CALLBACK_STRICT=1`` env var,
which flips warnings into exceptions for users who would rather fail
fast (typical for CI pipelines or production deployments).

The three failure modes:

1. **Connection failures** (Aim server unreachable, ``aim`` not
   installed) — logged at ``WARNING`` once, callback degrades to a
   no-op. ``strict`` mode: raises ``RuntimeError``.
2. **Per-metric track failures** (Aim disconnected mid-run, malformed
   metric value) — logged at ``DEBUG`` once per metric name. Training
   continues. ``strict`` mode: raises whatever Aim raised.
3. **Close failures** (Aim server gone at end of training) — silent. By
   the time we close the run, the relevant data has been streamed; a
   close failure is cosmetic.

Users who want to know whether logging is actually happening should
either set ``ASTROLABE_CALLBACK_STRICT=1`` (fail fast on any issue) or
read the ``WARNING`` log line emitted at callback init.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

__all__ = [
    "EVAL_METRIC_PREFIX",
    "DEFAULT_AIM_URL",
    "RunConfig",
    "WallTimeTracker",
    "close_run",
    "is_strict",
    "open_aim_run",
    "parse_aim_run_tags",
    "resolve_run_config",
    "track_safely",
]


# Single point of truth for the during-training validation namespace.
# v0.2.0 ships with "eval" for back-compat with existing ProjectOrion
# runs. v1.0.0 flips this to "val" alongside astrolabe v1.7's eval-runs
# schema, which disambiguates `val/` (during-training metrics) from
# `Eval` (post-training eval_results.json files). Flip = one-line change.
EVAL_METRIC_PREFIX = "eval"

# Default Aim tracking URL. Astrolabe sets up an SSH reverse tunnel
# from the GPU instance back to the NUC's Aim server on port 43800.
# Standalone users override via ``ASTROLABE_AIM_URL`` env or constructor.
DEFAULT_AIM_URL = "aim://localhost:43800"

_STRICT_ENV = "ASTROLABE_CALLBACK_STRICT"


def is_strict() -> bool:
    """Return ``True`` if strict-mode is enabled via env var.

    Strict mode flips graceful-degradation warnings into raised
    exceptions so users can fail fast on any logging issue. Read on
    every call rather than cached so tests can ``monkeypatch.setenv``
    without restarting the process.
    """
    return os.environ.get(_STRICT_ENV, "").lower() in ("1", "true", "yes")


def parse_aim_run_tags(raw: str | None) -> dict[str, str]:
    """Parse the ``AIM_RUN_TAGS`` env var into a tag dict.

    Format: ``key1=val1,key2=val2``. Whitespace around keys/values is
    stripped. Entries without ``=``, with empty keys, or duplicate keys
    (last wins) are tolerated rather than raising — this is reading
    something a researcher pasted into a shell, not a strict format.

    Parameters
    ----------
    raw : str | None
        The raw env var value, or ``None``/empty.

    Returns
    -------
    dict[str, str]
        Parsed tags. Empty if ``raw`` is empty or unparseable.
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


@dataclass(frozen=True)
class RunConfig:
    """Resolved configuration for an Aim run.

    Constructed by ``resolve_run_config`` after reading env vars and
    applying precedence rules. Frozen so the same config object can be
    safely passed across rank-zero/non-rank-zero processes.

    Attributes
    ----------
    experiment_name : str | None
        Aim experiment name. ``ASTROLABE_EXPERIMENT_NAME`` env wins
        over any constructor argument; falls through to ``None`` when
        neither is set (Aim assigns a default).
    aim_url : str
        Aim tracking URL. ``ASTROLABE_AIM_URL`` env wins over
        constructor argument; defaults to ``DEFAULT_AIM_URL``.
    tags : dict[str, str]
        Tags applied to the run on init. ``AIM_RUN_TAGS`` env wins over
        constructor argument when set (astrolabe is the orchestrator;
        its tags are authoritative).
    """

    experiment_name: str | None
    aim_url: str
    tags: dict[str, str] = field(default_factory=dict)


def resolve_run_config(
    *,
    experiment_name: str | None = None,
    aim_url: str | None = None,
    tags: dict[str, str] | None = None,
) -> RunConfig:
    """Read env vars + apply precedence rules into a ``RunConfig``.

    Precedence: astrolabe env vars win over constructor arguments.
    Astrolabe is the orchestrator; its identity is authoritative when
    it's the one driving the run. Constructor arguments are the
    standalone fallback for users running the callback without
    astrolabe.

    Parameters
    ----------
    experiment_name : str | None
        Constructor-supplied experiment name. Overridden by
        ``ASTROLABE_EXPERIMENT_NAME`` when that env var is set.
    aim_url : str | None
        Constructor-supplied Aim URL. Overridden by
        ``ASTROLABE_AIM_URL``; falls back to ``DEFAULT_AIM_URL``.
    tags : dict[str, str] | None
        Constructor-supplied tags. Overridden by ``AIM_RUN_TAGS`` when
        that env var is set; pass an empty dict to explicitly disable
        env fallback.

    Returns
    -------
    RunConfig
        Frozen config dataclass.
    """
    env_exp = os.environ.get("ASTROLABE_EXPERIMENT_NAME") or None
    resolved_exp = env_exp or experiment_name or None

    env_url = os.environ.get("ASTROLABE_AIM_URL")
    resolved_url = env_url or aim_url or DEFAULT_AIM_URL

    env_tags = parse_aim_run_tags(os.environ.get("AIM_RUN_TAGS"))
    if env_tags:
        resolved_tags = env_tags
    elif tags is not None:
        resolved_tags = dict(tags)
    else:
        resolved_tags = {}

    return RunConfig(
        experiment_name=resolved_exp,
        aim_url=resolved_url,
        tags=resolved_tags,
    )


def open_aim_run(cfg: RunConfig, *, run_name: str | None = None) -> Any:
    """Open an Aim run and apply astrolabe-flavored tags.

    Returns the live ``aim.Run`` on success. On failure (Aim not
    installed, server unreachable), logs a ``WARNING`` once and
    returns ``None`` — the calling callback then no-ops every Aim
    write. In strict mode (``ASTROLABE_CALLBACK_STRICT=1``), failures
    raise ``RuntimeError`` instead.

    Parameters
    ----------
    cfg : RunConfig
        Resolved configuration.
    run_name : str | None
        Optional human-readable name for the run (e.g. Composer's
        ``state.run_name`` or Lightning's logger name). Set as the
        ``Run.name`` property if non-empty.

    Returns
    -------
    aim.Run | None
        The opened run, or ``None`` if connection failed and strict
        mode is off.

    Raises
    ------
    RuntimeError
        If ``ASTROLABE_CALLBACK_STRICT=1`` and the Aim run could not be
        opened (import error or connection error).
    """
    try:
        from aim import Run
    except ImportError as exc:
        msg = (
            "aim not installed — astrolabe-callbacks logging disabled. "
            "Install with: pip install aim"
        )
        if is_strict():
            raise RuntimeError(msg) from exc
        logger.warning(msg)
        return None

    try:
        run = Run(repo=cfg.aim_url, experiment=cfg.experiment_name)
    except Exception as exc:
        msg = f"Aim connection to {cfg.aim_url} failed: {exc}"
        if is_strict():
            raise RuntimeError(msg) from exc
        logger.warning(msg + " — callback degrading to no-op.")
        return None

    # Run.name carries through to the dashboard so researchers see the
    # meaningful name (e.g. "bert-tiny") instead of the auto-generated
    # "Run: <hash>" placeholder. Best-effort; older Aim versions may
    # treat name as read-only.
    if run_name:
        try:
            run.name = run_name
        except Exception as exc:
            logger.debug("Failed to set Aim run name {}: {}", run_name, exc)

    # Apply astrolabe.* tags. Each tag write is independent — one
    # failing key shouldn't drop the others. Aim's Run is dict-like;
    # ``run[key] = value`` writes a tag/param visible in the dashboard.
    for key, value in cfg.tags.items():
        try:
            run[key] = value
        except Exception as exc:
            logger.debug("Failed to set Aim tag {}={}: {}", key, value, exc)

    tag_summary = (
        f" (tags: {', '.join(f'{k}={v}' for k, v in cfg.tags.items())})"
        if cfg.tags
        else ""
    )
    logger.info("Aim run opened at {}{}", cfg.aim_url, tag_summary)
    return run


def close_run(run: Any, *, status: str = "completed") -> None:
    """Finalize an Aim run, idempotent.

    Writes ``astrolabe.status`` (``completed`` / ``failed`` /
    ``interrupted``) before closing so the dashboard can show the run's
    final disposition. Idempotent so callers can invoke from both
    ``train_end`` and ``exception`` hooks without worrying about
    double-close.

    Failures are silent — by the time close runs, the relevant data
    has been streamed and a close failure is cosmetic.

    Parameters
    ----------
    run : aim.Run | None
        The run to close. ``None`` is a no-op (handles the case where
        ``open_aim_run`` returned ``None``).
    status : str
        Final status tag. Standard values: ``"completed"``, ``"failed"``,
        ``"interrupted"``. Custom values allowed for callback authors
        with extra states.
    """
    if run is None:
        return
    try:
        run["astrolabe.status"] = status
    except Exception as exc:
        logger.debug("Failed to set astrolabe.status={}: {}", status, exc)
    try:
        run.close()
        logger.info("Aim run finalized (status={})", status)
    except Exception as exc:
        logger.debug("Aim run close failed: {}", exc)


def track_safely(
    run: Any,
    *,
    name: str,
    value: float,
    step: int | None = None,
    context: dict | None = None,
) -> None:
    """Call ``run.track`` with graceful degradation on failure.

    Logs at ``WARNING`` once per metric name on first failure (rate-limit
    via the per-name set on the run object). Subsequent failures for the
    same metric are silenced so we don't flood with one warning per
    batch. In strict mode, re-raises whatever Aim raised.

    Parameters
    ----------
    run : aim.Run | None
        The Aim run (``None`` is a no-op).
    name : str
        Metric name (e.g. ``"train/loss"``, ``"eval/accuracy"``).
    value : float
        Metric value.
    step : int | None
        Optional step counter. Passed through to ``run.track``.
    context : dict | None
        Optional Aim context dict (e.g. ``{"subset": "val"}``).
    """
    if run is None:
        return
    try:
        run.track(value, name=name, step=step, context=context or {})
    except Exception as exc:
        if is_strict():
            raise
        # Rate-limit per metric name — one DEBUG line, not one per batch.
        # Track which names we've already complained about on the run
        # object itself so the rate-limit is per-run, not global.
        seen = getattr(run, "_astrolabe_track_failures", None)
        if seen is None:
            seen = set()
            try:
                run._astrolabe_track_failures = seen
            except Exception:
                # Some Aim versions disallow attribute writes on Run.
                # Fall back to per-call DEBUG (noisier but still capped
                # by user's log level filter).
                logger.warning("Aim track failed for {}: {!r}", name, exc)
                return
        if name not in seen:
            seen.add(name)
            logger.warning(
                "Aim track failed for {} (suppressing further for this metric): {!r}",
                name, exc,
            )


class WallTimeTracker:
    """Tracks elapsed training time, excluding setup and eval pauses.

    ``wall_time`` as a metric is anchored to the first **training**
    batch (not callback init), so setup time — model loading,
    dataloader build, tokenizer warmup, weight init — is excluded.
    During eval, time is paused; resumed when training continues. The
    result is *training-only elapsed* time: two runs with the same
    training cost but different eval cadences land at the same
    ``wall_time`` for the same step. Comparison stays apples-to-apples.

    Usage::

        tracker = WallTimeTracker()
        # in batch_end:
        tracker.mark_first_batch()    # idempotent
        elapsed = tracker.elapsed()   # seconds since first batch, minus eval pauses
        # in eval_start:
        tracker.pause_for_eval()
        # in eval_end:
        tracker.resume()
    """

    def __init__(self) -> None:
        # ``_start_time`` stays 0 until ``mark_first_batch`` fires for
        # the first time. Lets ``elapsed`` distinguish "training has
        # not started" (returns 0.0) from "training just started"
        # (returns ~0).
        self._start_time: float = 0.0
        self._eval_start: float = 0.0
        self._total_eval_time: float = 0.0

    def mark_first_batch(self) -> None:
        """Anchor wall-time to the current moment if not already anchored.

        Idempotent — only the first call has effect. Safe to call from
        every ``batch_end`` hook without checking.
        """
        if self._start_time == 0.0:
            self._start_time = time.monotonic()

    def pause_for_eval(self) -> None:
        """Mark the start of an eval pause."""
        self._eval_start = time.monotonic()

    def resume(self) -> None:
        """Mark the end of an eval pause and accumulate its duration."""
        if self._eval_start > 0:
            self._total_eval_time += time.monotonic() - self._eval_start
            self._eval_start = 0.0

    def elapsed(self) -> float:
        """Return seconds since first batch, minus accumulated eval time.

        Returns 0.0 if ``mark_first_batch`` has not fired yet.
        """
        if self._start_time == 0.0:
            return 0.0
        return (time.monotonic() - self._start_time) - self._total_eval_time
