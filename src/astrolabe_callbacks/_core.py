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
import queue
import threading
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


# Buffer sizing — defaults are generous for the common case (multi-hour
# training runs at per-batch logging cadence). At ~200 bytes per queued
# tuple, 100k items is ~20 MB of memory; sufficient to ride out a 5-10
# minute network partition at typical scalar-logging rates without ever
# touching the drop-oldest path. Override via env if you need bigger
# (rare) or smaller (memory-constrained training).
_DEFAULT_BUFFER_SIZE = 100_000
_DEFAULT_RETRY_INITIAL_S = 0.5
_DEFAULT_RETRY_MAX_S = 30.0
_DEFAULT_DRAIN_TIMEOUT_S = 30.0
# How often the drainer thread emits a snapshot of buffer counters at
# INFO level — purely observational, so an operator tailing logs can
# see "buffer is healthy / actively retrying / accumulating drops"
# mid-run instead of waiting until close to find out. 5 min is
# infrequent enough to not clutter logs, frequent enough to give a
# usable timeline post-mortem.
_DEFAULT_HEARTBEAT_INTERVAL_S = 300.0


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


class _MetricBuffer:
    """Bounded queue + background drainer thread per Aim Run.

    Owns write-side reliability for ``track_safely``:

    * Queues every metric write in-process (microseconds, no I/O).
    * A daemon thread pops items and calls ``run.track()``, retrying
      with exponential backoff on any exception (gRPC blip, transient
      Aim server unavailability, SSH-tunnel jitter).
    * Bounded — 100k items by default, ~20 MB at typical scalar-tuple
      sizes. If the queue ever fills (network partitioned for >10
      minutes at per-batch cadence), oldest is dropped to make room
      and the drop is counted.
    * ``close()`` blocks until the queue drains or a timeout elapses,
      so a clean training exit doesn't truncate in-flight writes.

    Attached to each opened Aim Run as ``run._astrolabe_buffer`` by
    ``open_aim_run``. Strict mode (``ASTROLABE_CALLBACK_STRICT=1``)
    bypasses the buffer entirely — strict semantics ("I want to know
    immediately if anything fails") map cleanly to synchronous +
    raise, so no queue.

    Why we own this layer rather than trusting Aim's client queue:
    Aim's gRPC client is designed for local/LAN deployment and uses
    aggressive drop-on-overflow with a small queue. Under astrolabe's
    SSH-tunneled writer-to-NUC path, sustained per-batch logging can
    silently lose 20%+ of writes (observed on real ProjectOrion runs).
    This layer makes the drop-on-overflow case observable and the
    transient-failure case invisible.
    """

    # Sentinel item that tells the drainer to stop after popping.
    _SENTINEL: Any = object()

    def __init__(
        self,
        run: Any,
        *,
        max_size: int = _DEFAULT_BUFFER_SIZE,
        retry_initial_backoff_s: float = _DEFAULT_RETRY_INITIAL_S,
        retry_max_backoff_s: float = _DEFAULT_RETRY_MAX_S,
        heartbeat_interval_s: float = _DEFAULT_HEARTBEAT_INTERVAL_S,
    ) -> None:
        self._run = run
        self._max_size = max_size
        self._retry_initial = retry_initial_backoff_s
        self._retry_max = retry_max_backoff_s
        self._heartbeat_interval_s = heartbeat_interval_s
        self._queue: queue.Queue = queue.Queue(maxsize=max_size)
        self._stop = threading.Event()
        # Counters surfaced at close time for visibility — first time
        # you see retried > 0 you know the network had trouble even
        # though your training didn't.
        self._submitted = 0
        self._drained = 0
        self._retried = 0
        self._dropped_oldest = 0
        # Heartbeat state — the drainer emits a periodic INFO snapshot
        # so operators tailing logs can see how buffer health evolves
        # over the lifetime of a run rather than waiting for the close
        # summary. Skip emission when nothing has changed since the
        # last heartbeat (otherwise a quiet run would still produce a
        # log line every interval). Initialized so the first heartbeat
        # can fire only after the configured interval elapses.
        self._last_heartbeat_t = time.monotonic()
        self._last_heartbeat_snapshot: tuple[int, int, int, int] = (0, 0, 0, 0)
        # Per-metric warning rate-limit — without this a persistent
        # failure would emit a WARNING per batch.
        self._warned: set[str] = set()
        self._drainer = threading.Thread(
            target=self._drain_loop,
            name="astrolabe-aim-buffer",
            daemon=True,
        )
        self._drainer.start()

    def _maybe_heartbeat(self) -> None:
        """Emit a snapshot of counters at INFO if the interval elapsed
        and any counter advanced since the last heartbeat.

        Called from the drainer's main loop, so absence of heartbeats
        also signals the drainer thread itself is alive (or hung — a
        long silence with no end-of-run summary either is a tell).

        Also persists each snapshot to ``$ASTROLABE_CALLBACK_STATS_PATH``
        when that env var is set. astrolabe's engine sets it on Lambda
        so the file rsyncs back at step end and survives instance
        termination — without that file, mid-run diagnostics are lost
        when the VM is reaped, which is exactly the moment we'd want
        to know whether the buffer was overflowing.
        """
        now = time.monotonic()
        if now - self._last_heartbeat_t < self._heartbeat_interval_s:
            return
        self._last_heartbeat_t = now
        snapshot = (
            self._submitted,
            self._drained,
            self._retried,
            self._dropped_oldest,
        )
        if snapshot == self._last_heartbeat_snapshot:
            # Quiet period — no submits, no drains, no failures since
            # last check. Skip the log line; the absence of activity
            # is signal enough.
            return
        self._last_heartbeat_snapshot = snapshot
        logger.info(
            "Aim buffer (heartbeat): {} submitted, {} drained, "
            "{} retried, {} dropped, queue depth {}",
            *snapshot,
            self._queue.qsize(),
        )
        _append_stats_line(
            kind="heartbeat",
            submitted=snapshot[0],
            drained=snapshot[1],
            retried=snapshot[2],
            dropped=snapshot[3],
            queue_depth=self._queue.qsize(),
        )

    def submit(
        self,
        name: str,
        value: float,
        step: int | None,
        context: dict | None,
    ) -> None:
        """Non-blocking enqueue. Drops oldest + WARN if full."""
        item = (name, value, step, context)
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
            return
        except queue.Full:
            pass

        # Queue full → drop oldest to make room. Single-producer (the
        # training thread) means the get/put pair below is effectively
        # atomic w.r.t. our own writes.
        try:
            self._queue.get_nowait()
            self._dropped_oldest += 1
            self._queue.task_done()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
            self._submitted += 1
        except queue.Full:
            # Concurrent drainer beat us back to full. Concede.
            self._dropped_oldest += 1

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Block until all currently-queued items have been processed.

        Unlike ``close``, leaves the drainer running. Used by tests
        that need synchronous semantics after a ``submit`` call;
        production code typically only calls ``close``.

        Returns ``True`` if drained within the timeout, ``False`` if
        the drainer fell behind enough that some items remain.
        """
        deadline = time.monotonic() + timeout_s
        # ``unfinished_tasks`` is decremented by ``task_done()`` which
        # the drainer calls after every item (success, retry-give-up,
        # or sentinel). Polling cadence is fine for test use.
        while (
            self._queue.unfinished_tasks > 0
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        return self._queue.unfinished_tasks == 0

    def close(self, timeout_s: float = _DEFAULT_DRAIN_TIMEOUT_S) -> int:
        """Block until the queue drains or ``timeout_s`` elapses.

        Returns the number of items still queued at timeout (zero if
        drain completed cleanly).
        """
        # Push the sentinel to unblock the drainer's blocking get when
        # the queue is already empty. The drainer treats it as "stop
        # after this".
        self._stop.set()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:
            # If the queue is full, the drainer is making progress —
            # the stop event will catch it next iteration.
            pass
        self._drainer.join(timeout=timeout_s)
        # Whatever's still in the queue at this point didn't drain.
        # qsize() doesn't include the sentinel reliably across Python
        # versions, but a stuck-shut queue with non-zero size means we
        # lost data — surfaced in close_run's WARNING line.
        return max(0, self._queue.qsize() - 1)  # subtract sentinel

    def stats(self) -> dict[str, int]:
        return {
            "submitted": self._submitted,
            "drained": self._drained,
            "retried": self._retried,
            "dropped_oldest": self._dropped_oldest,
            "queue_depth": self._queue.qsize(),
        }

    def _drain_loop(self) -> None:
        """Pop items and call run.track with retry until stop + drained."""
        while True:
            # Heartbeat is checked once per iteration. ~10 calls/sec
            # at the queue.get timeout cadence; the time-check is
            # nanoseconds so the loop overhead is negligible.
            self._maybe_heartbeat()
            try:
                # Short timeout so the loop can notice _stop even when
                # the queue is empty. Long enough that idle CPU is
                # negligible (~0.1% of one core at 100ms cadence).
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    return
                continue

            if item is self._SENTINEL:
                self._queue.task_done()
                # If stop was set, exit. Otherwise it was a stray
                # sentinel; loop and try again.
                if self._stop.is_set():
                    return
                continue

            name, value, step, context = item
            backoff = self._retry_initial
            while True:
                try:
                    self._run.track(
                        value, name=name, step=step, context=context or {}
                    )
                    self._drained += 1
                    break
                except Exception as exc:
                    self._retried += 1
                    if name not in self._warned:
                        self._warned.add(name)
                        logger.warning(
                            "Aim track failed for {} — buffer will retry "
                            "(suppressing further warnings for this metric): {!r}",
                            name,
                            exc,
                        )
                    if self._stop.is_set():
                        # Shutting down; give up on this item rather
                        # than blocking the drainer's exit indefinitely.
                        break
                    time.sleep(backoff)
                    backoff = min(backoff * 2, self._retry_max)
            self._queue.task_done()


def _append_stats_line(**fields) -> None:
    """Append one JSONL record to ``$ASTROLABE_CALLBACK_STATS_PATH``.

    No-op when the env var is unset (the standard out-of-band-of-
    astrolabe case). All failures are silenced — this is a
    diagnostic side-channel; if writing it breaks, the training run
    must NOT fail.

    Each record carries a wall-clock ``ts`` (Unix seconds, float)
    plus the caller-provided fields. JSONL means each diagnostic
    event is a single self-contained line — easy to ``tail``, easy to
    grep, easy to parse line-by-line without loading the whole file.
    """
    path = os.environ.get("ASTROLABE_CALLBACK_STATS_PATH")
    if not path:
        return
    try:
        import json
        # ~ expansion so callers can use $HOME-relative paths without
        # depending on whatever shell expansion their launcher does.
        expanded = os.path.expanduser(path)
        record = {"ts": time.time(), **fields}
        with open(expanded, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as exc:  # noqa: BLE001 — diagnostic must not raise
        logger.debug("Failed to append callback stats line: {}", exc)


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

    # Attach the write-buffer + drainer thread that owns reliability
    # for ``track_safely``. Default-on; strict mode bypasses (see the
    # _MetricBuffer docstring for rationale). Best-effort: if the Run
    # doesn't accept attribute writes (rare, older Aim versions),
    # track_safely falls back to synchronous + rate-limited WARN.
    try:
        run._astrolabe_buffer = _MetricBuffer(run)
    except Exception as exc:
        logger.debug(
            "Could not attach metric buffer (writes will be synchronous): {}", exc
        )

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

    # Drain the write-buffer before closing the Run. Without this, any
    # in-flight queued metrics would be lost when run.close()
    # tears down the gRPC connection.
    buffer = getattr(run, "_astrolabe_buffer", None)
    if buffer is not None:
        try:
            unflushed = buffer.close(timeout_s=_DEFAULT_DRAIN_TIMEOUT_S)
            stats = buffer.stats()
            summary = (
                f"Aim buffer: {stats['submitted']} submitted, "
                f"{stats['drained']} drained, "
                f"{stats['retried']} retried, "
                f"{stats['dropped_oldest']} dropped"
            )
            if unflushed:
                logger.warning(
                    "{}, {} UNFLUSHED at close (drain timed out — Aim server "
                    "may have been unreachable for an extended period)",
                    summary,
                    unflushed,
                )
            elif stats["retried"] > 0 or stats["dropped_oldest"] > 0:
                # Non-trivial buffer activity — surface it so the
                # operator notices network instability even though
                # training succeeded.
                logger.warning("{}, 0 unflushed.", summary)
            else:
                logger.info("{}, 0 unflushed.", summary)
            # Persist the close summary to the same stats file as the
            # heartbeats so post-mortem diagnosis has the FINAL numbers
            # even when the training stdout is gone with the instance.
            _append_stats_line(
                kind="close",
                status=status,
                submitted=stats["submitted"],
                drained=stats["drained"],
                retried=stats["retried"],
                dropped=stats["dropped_oldest"],
                unflushed=unflushed,
            )
        except Exception as exc:
            logger.debug("Buffer drain failed: {}", exc)

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

    # Strict mode: synchronous + raise. Strict semantics ("crash on any
    # failure, I want CI to fail fast") map cleanly to bypassing the
    # buffer — buffered failures happen on a background thread and
    # can't be raised to the caller.
    if is_strict():
        run.track(value, name=name, step=step, context=context or {})
        return

    # Default mode: route through the per-Run write buffer that
    # ``open_aim_run`` attached. Microsecond enqueue; the drainer
    # thread retries on transient failures and surfaces stats at
    # close. See ``_MetricBuffer`` docstring for the full contract.
    buffer = getattr(run, "_astrolabe_buffer", None)
    if buffer is not None:
        buffer.submit(name, value, step, context)
        return

    # Fallback: buffer wasn't attached (older Aim version that rejects
    # setattr on Run). Use the legacy synchronous + rate-limited-WARN
    # path so we degrade gracefully rather than dropping metrics
    # silently.
    try:
        run.track(value, name=name, step=step, context=context or {})
    except Exception as exc:
        seen = getattr(run, "_astrolabe_track_failures", None)
        if seen is None:
            seen = set()
            try:
                run._astrolabe_track_failures = seen
            except Exception:
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
