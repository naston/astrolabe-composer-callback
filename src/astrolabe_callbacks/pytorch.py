"""Raw-PyTorch (and any custom-loop) Aim logging via context manager.

For frameworks that expose callback hooks (Composer, Lightning, HF
Trainer), use the matching ``Astrolabe<Framework>Logger`` class. This
module is for everyone else: hand-written PyTorch loops, JAX/Flax,
HF Accelerate's manual-loop pattern, custom training frameworks. There
are no callback hooks to plug into, so we ship a context manager
instead.

Usage::

    from astrolabe_callbacks import Run

    with Run() as run:                              # reads env vars
        for batch_idx, batch in enumerate(dataloader):
            loss = model(batch)
            loss.backward()
            optimizer.step()
            run.log_train(loss=loss.item(), step=batch_idx)

        for val_batch in val_loader:
            val_loss = model(val_batch).item()
            run.log_eval(loss=val_loss, step=batch_idx)

The same env-var contract as the framework callbacks applies:
``ASTROLABE_EXPERIMENT_NAME``, ``AIM_RUN_TAGS``, ``ASTROLABE_AIM_URL``
all win over constructor arguments.

Failure handling
----------------

Same contract as the framework callbacks: connection failures log a
single ``WARNING`` and downgrade to no-ops; per-track failures log
``DEBUG`` once per metric. Set ``ASTROLABE_CALLBACK_STRICT=1`` to flip
warnings into raised exceptions for fail-fast CI behavior. See
``_core.py`` for the full failure-handling contract.

Distributed training
--------------------

``AstrolabeRun`` gates all writes on rank-zero (detected via
``torch.distributed`` if initialized, ``RANK``/``LOCAL_RANK`` env vars
otherwise). Non-rank-zero processes still need to enter and exit the
context manager — they just no-op every method. Single-process training
is treated as rank-zero.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from astrolabe_callbacks import _core
from astrolabe_callbacks._distributed import is_rank_zero

__all__ = ["AstrolabeRun", "Run"]


class AstrolabeRun:
    """Context manager that opens an Aim run + provides logging methods.

    Parameters
    ----------
    aim_url : str | None
        Aim tracking URL (e.g. ``aim://localhost:43800``). Overridden
        by ``ASTROLABE_AIM_URL`` env; defaults to the standard
        astrolabe SSH-tunneled URL.
    experiment_name : str | None
        Aim experiment name. Overridden by
        ``ASTROLABE_EXPERIMENT_NAME`` env.
    tags : dict[str, str] | None
        Tags applied to the Aim run on enter. Overridden by
        ``AIM_RUN_TAGS`` env when set.
    run_name : str | None
        Optional human-readable name for the run, displayed in the
        dashboard. Useful for distinguishing runs of the same
        experiment.

    Attributes
    ----------
    is_active : bool
        ``True`` if the underlying Aim run is open. ``False`` for
        non-rank-zero processes or after a connection failure (when
        not in strict mode).
    """

    def __init__(
        self,
        aim_url: str | None = None,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
        run_name: str | None = None,
    ):
        self._cfg = _core.resolve_run_config(
            experiment_name=experiment_name,
            aim_url=aim_url,
            tags=tags,
        )
        self._run_name = run_name
        self._run: Any = None
        self._wall_time = _core.WallTimeTracker()
        self._rank_zero = is_rank_zero()
        self._closed = False

    @property
    def is_active(self) -> bool:
        """Whether the run is open and accepting writes."""
        return self._run is not None and not self._closed

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "AstrolabeRun":
        if not self._rank_zero:
            return self
        self._run = _core.open_aim_run(self._cfg, run_name=self._run_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Only rank-zero closes (non-rank-zero never opened). The
        # ``status`` reflects whether the user's loop completed
        # cleanly: an exception propagating through ``__exit__`` marks
        # the run as failed so the dashboard shows the disposition.
        if not self._rank_zero:
            return
        status = "failed" if exc_type is not None else "completed"
        _core.close_run(self._run, status=status)
        self._run = None
        self._closed = True

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def log_train(self, *, step: int | None = None, **metrics: float) -> None:
        """Log training-side metrics under the ``train/`` namespace.

        Each kwarg becomes a metric named ``train/<kwarg>``. Anchors
        the wall-time clock to the first call and writes ``wall_time``
        alongside the user's metrics so the dashboard's elapsed-time
        x-axis works without extra setup.

        Parameters
        ----------
        step : int | None
            Optional step counter. ``None`` lets Aim auto-increment its
            internal step; pass ``batch_idx`` explicitly when you want
            metric step alignment with your training loop.
        **metrics : float
            Each kwarg is logged as ``train/<name>``. Example:
            ``run.log_train(loss=0.5, accuracy=0.9)``.
        """
        if not self.is_active:
            return
        self._wall_time.mark_first_batch()
        for name, value in metrics.items():
            _core.track_safely(
                self._run, name=f"train/{name}", value=value, step=step
            )
        _core.track_safely(
            self._run,
            name="wall_time",
            value=self._wall_time.elapsed(),
            step=step,
        )

    def log_eval(self, *, step: int | None = None, **metrics: float) -> None:
        """Log eval-side metrics under the during-training eval namespace.

        Each kwarg becomes a metric named ``<EVAL_PREFIX>/<kwarg>``,
        where ``EVAL_PREFIX`` is ``"val"`` as of v1.0.0 (aligning with
        astrolabe v1.7's eval-runs schema — the ``eval/`` prefix is
        reserved for post-training benchmark suites on dedicated eval
        Aim runs). Single point of truth — the same constant feeds
        every framework callback so the rename happens once.

        Parameters
        ----------
        step : int | None
            Optional step counter (typically the training-side step at
            which eval was triggered).
        **metrics : float
            Each kwarg is logged as ``<EVAL_PREFIX>/<name>``.
        """
        if not self.is_active:
            return
        for name, value in metrics.items():
            _core.track_safely(
                self._run,
                name=f"{_core.EVAL_METRIC_PREFIX}/{name}",
                value=value,
                step=step,
            )

    def log(
        self,
        name: str,
        value: float,
        *,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        """Escape hatch for arbitrary metric names.

        ``log_train`` / ``log_eval`` cover the common cases with the
        right namespace prefixes. Use ``log`` when you need a custom
        name (e.g. ``custom/throughput``, ``gpu/memory_gb``) or to pass
        an Aim ``context`` dict for advanced grouping.

        Parameters
        ----------
        name : str
            Full metric name including any namespace prefix you want
            (e.g. ``"throughput/samples_per_sec"``). No prefix is added.
        value : float
            Metric value.
        step : int | None
            Step counter.
        context : dict | None
            Aim context dict (e.g. ``{"subset": "val"}``).
        """
        if not self.is_active:
            return
        _core.track_safely(
            self._run, name=name, value=value, step=step, context=context
        )

    def set_tag(self, name: str, value: str) -> None:
        """Apply an additional tag to the run mid-flight.

        Useful for tags that aren't known until after training starts
        (e.g. final config resolved from a hyperparameter sweep). For
        tags known at construction, prefer the ``tags`` constructor
        argument or ``AIM_RUN_TAGS`` env var.
        """
        if not self.is_active:
            return
        try:
            self._run[name] = value
        except Exception as exc:
            if _core.is_strict():
                raise
            logger.debug("Failed to set tag {}={}: {}", name, value, exc)

    # ------------------------------------------------------------------
    # Wall-time control
    # ------------------------------------------------------------------

    def pause_eval(self) -> None:
        """Pause wall-time accounting (call before running eval).

        Eval time is excluded from ``wall_time`` so the metric reflects
        pure training compute. Two runs with the same training cost
        but different eval cadences land on the same ``wall_time``
        for the same step — comparison stays apples-to-apples.
        """
        if not self.is_active:
            return
        self._wall_time.pause_for_eval()

    def resume(self) -> None:
        """Resume wall-time accounting (call after eval finishes)."""
        if not self.is_active:
            return
        self._wall_time.resume()


# Convenience alias for the common case — `from astrolabe_callbacks
# import Run` is shorter and reads as "open a run", which matches the
# context manager idiom users already know from `wandb.init()` etc.
Run = AstrolabeRun
