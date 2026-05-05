"""PyTorch Lightning callback wired to astrolabe-callbacks ``_core``.

Same env-var contract as the other framework callbacks:
``ASTROLABE_EXPERIMENT_NAME``, ``AIM_RUN_TAGS``, ``ASTROLABE_AIM_URL``
all win over constructor arguments.

Pass-through philosophy
-----------------------

This callback streams **every** metric the user logs via
``self.log(...)`` to Aim, not just training loss. Names land in Aim
exactly as the user wrote them — no synthetic prefixes, no hidden
defaults. The only metric we synthesize is ``wall_time``.

To get a metric into Aim, call ``self.log("name", value)`` inside your
LightningModule (or pass ``on_step=True`` if you want it per-batch
rather than per-epoch). Anything that lands in ``trainer.callback_metrics``
flows through.

Convention: keys prefixed with ``val_`` or ``val/`` are treated as
eval-side and re-namespaced under the canonical eval prefix
(``eval/`` in v0.2.0, ``val/`` from v1.0.0). All other keys pass
through unchanged.

Usage::

    from astrolabe_callbacks import AstrolabeLightningLogger
    from lightning.pytorch import Trainer

    trainer = Trainer(
        ...,
        callbacks=[AstrolabeLightningLogger()],
    )
    trainer.fit(model, train_loader, val_loader)

Implementation note — why a Callback, not a Logger
--------------------------------------------------

Lightning has both a ``Callback`` base class and a ``Logger`` base
class. We deliberately implement this as a Callback because:

1. Callbacks have a tighter scope (just hooks; no metric machinery to
   maintain on our side).
2. Users typically already have a TB or W&B Logger and we don't want
   to compete with their primary logger.
3. Aim already ships ``aim.pytorch_lightning.AimLogger`` for the
   logger-side integration; our value-add is the astrolabe tag
   conventions, which are orthogonal to whichever logger the user has.

So this callback runs alongside whatever logger setup the user has,
adding astrolabe tags to a parallel Aim run.

Failure handling
----------------

Same as the other framework callbacks: connection failures log a single
``WARNING`` and downgrade to no-ops; per-track failures log ``DEBUG``
once per metric. Set ``ASTROLABE_CALLBACK_STRICT=1`` for fail-fast
behavior. See ``_core.py`` for the full contract.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from astrolabe_callbacks import _core
from astrolabe_callbacks._distributed import is_rank_zero

try:
    # Lightning 2.x unified package. ``pytorch-lightning`` (pre-2.x) is
    # not supported — users on the legacy package can ask later.
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover — Lightning is an optional extra
    Callback = object  # type: ignore[misc,assignment]

__all__ = ["AstrolabeLightningLogger"]


class AstrolabeLightningLogger(Callback):
    """Lightning callback that streams every user-logged metric to Aim.

    Reads ``trainer.callback_metrics`` on each batch- and validation-end
    hook; passes through everything the user logged via ``self.log()``.
    Plus ``wall_time``.

    Parameters
    ----------
    aim_url : str | None
        Aim tracking URL. Overridden by ``ASTROLABE_AIM_URL``.
    experiment_name : str | None
        Aim experiment name. Overridden by ``ASTROLABE_EXPERIMENT_NAME``.
    tags : dict[str, str] | None
        Tags applied to the Aim run on init. Overridden by
        ``AIM_RUN_TAGS`` env var.
    run_name : str | None
        Optional explicit run name. When ``None``, the callback uses
        ``trainer.logger.name`` if a Lightning logger is configured,
        falling back to the LightningModule's class name.
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
        self._explicit_run_name = run_name
        self._run: Any = None
        self._step = 0
        self._wall_time = _core.WallTimeTracker()
        self._rank_zero = is_rank_zero()

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
        """Open the Aim run when training starts.

        Lightning fires ``setup`` once per stage (``fit`` / ``validate``
        / ``test`` / ``predict``). We only open the run for ``fit`` —
        eval-only and predict stages are out of scope.
        """
        if stage != "fit":
            return
        if not self._rank_zero:
            return
        if self._run is not None:
            return  # double-open guard

        # Run name precedence: explicit constructor arg → trainer's
        # logger name → LightningModule class name.
        run_name = self._explicit_run_name
        if not run_name and getattr(trainer, "logger", None) is not None:
            run_name = getattr(trainer.logger, "name", None)
        if not run_name:
            run_name = pl_module.__class__.__name__

        self._run = _core.open_aim_run(self._cfg, run_name=run_name)

    def on_train_batch_end(
        self,
        trainer: Any,
        pl_module: Any,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Pass through every user-logged training metric + log wall_time.

        Reads ``trainer.callback_metrics`` and writes each non-val key
        as-is. Users should call ``self.log("my_metric", value,
        on_step=True)`` in their training_step to get per-batch metric
        flow; ``on_epoch=True``-only metrics will land in
        ``callback_metrics`` only at epoch end (and that's fine — we
        write what's available).
        """
        if not self._rank_zero or self._run is None:
            return

        self._wall_time.mark_first_batch()
        self._step += 1

        callback_metrics = getattr(trainer, "callback_metrics", None) or {}
        for raw_name, value in callback_metrics.items():
            if _is_val_metric(raw_name):
                continue  # eval-side; handled in on_validation_end
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            _core.track_safely(
                self._run,
                name=raw_name,
                value=scalar,
                step=self._step,
            )

        _core.track_safely(
            self._run,
            name="wall_time",
            value=self._wall_time.elapsed(),
            step=self._step,
        )

    def on_validation_start(self, trainer: Any, pl_module: Any) -> None:
        """Pause wall-time accounting during eval."""
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.pause_for_eval()

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        """Pass through every val metric in ``trainer.callback_metrics``.

        Lightning aggregates user-logged eval metrics via
        ``self.log(..., on_epoch=True)`` (the default for
        ``validation_step``) into ``trainer.callback_metrics`` by the
        time this hook fires. Keys prefixed with ``val_`` or ``val/``
        are stripped and re-namespaced under the canonical eval prefix
        so dashboards can group them.
        """
        # Resume wall-time first — accounting must be right even if
        # logging fails.
        self._wall_time.resume()

        if not self._rank_zero or self._run is None:
            return

        callback_metrics = getattr(trainer, "callback_metrics", None) or {}
        for raw_name, value in callback_metrics.items():
            clean = _normalize_val_metric_name(raw_name)
            if clean is None:
                continue
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            _core.track_safely(
                self._run,
                name=f"{_core.EVAL_METRIC_PREFIX}/{clean}",
                value=scalar,
                step=self._step,
            )

    def on_train_end(self, trainer: Any, pl_module: Any) -> None:
        """Close the Aim run cleanly at end of training."""
        if not self._rank_zero:
            return
        _core.close_run(self._run, status="completed")
        self._run = None

    def on_exception(
        self, trainer: Any, pl_module: Any, exception: BaseException
    ) -> None:
        """Mark the run as failed when training raises."""
        if not self._rank_zero:
            return
        _core.close_run(self._run, status="failed")
        self._run = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _is_val_metric(name: str) -> bool:
    """Whether a metric name belongs to the eval-side namespace."""
    return name.startswith("val/") or name.startswith("val_")


def _normalize_val_metric_name(name: str) -> str | None:
    """Strip val_/val/ prefix from a Lightning metric name.

    Returns the suffix (e.g. ``"loss"`` from ``"val_loss"``) or
    ``None`` if the name isn't val-side or the suffix is empty.
    """
    if name.startswith("val/"):
        return name[len("val/") :] or None
    if name.startswith("val_"):
        return name[len("val_") :] or None
    return None


def _to_scalar(value: Any) -> float | None:
    """Coerce a value to ``float``; return ``None`` if not numeric.

    Handles tensors (anything with ``.item()``), Python scalars, and
    rejects non-numeric types Aim would refuse.
    """
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
