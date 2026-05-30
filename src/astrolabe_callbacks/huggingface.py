"""HuggingFace ``Trainer`` callback wired to astrolabe-callbacks ``_core``.

Same env-var contract as the other framework callbacks:
``ASTROLABE_EXPERIMENT_NAME``, ``AIM_RUN_TAGS``, ``ASTROLABE_AIM_URL``
all win over constructor arguments.

Usage::

    from astrolabe_callbacks import AstrolabeHFTrainerCallback
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.add_callback(AstrolabeHFTrainerCallback())
    trainer.train()

Works with any ``Trainer`` subclass — including TRL's ``SFTTrainer``,
``DPOTrainer``, ``RewardTrainer``, etc., which all inherit from the
base HuggingFace ``Trainer``.

Wall-time precision
-------------------

HuggingFace ``Trainer`` doesn't expose a clean per-batch hook with
metadata, only ``on_step_end`` (no metric data) and ``on_log`` (fires
at ``logging_steps`` intervals). We anchor wall-time at the first
``on_step_end`` and report it on every ``on_log``. The downside:
when eval falls between two ``on_log`` events, that eval time is
included in the next ``wall_time`` reading. For most users this is
acceptable; for fine-grained timing, use ``AstrolabeRun`` with a
hand-written loop.

Failure handling
----------------

Same as the other framework callbacks: connection failures log a
single ``WARNING`` and downgrade to no-ops; per-track failures log
``DEBUG`` once per metric. Set ``ASTROLABE_CALLBACK_STRICT=1`` to
flip warnings into raised exceptions for fail-fast CI behavior.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from astrolabe_callbacks import _core
from astrolabe_callbacks._distributed import is_rank_zero

try:
    from transformers import TrainerCallback
except ImportError:  # pragma: no cover — transformers is an optional extra
    TrainerCallback = object  # type: ignore[misc,assignment]

__all__ = ["AstrolabeHFTrainerCallback"]


# Keys in HF Trainer's `logs` dict that we treat as training-side metrics.
# Anything else with the `eval_` prefix lands as an eval metric. The HF
# convention is `loss` (train loss, smoothed) and `learning_rate`; some
# users add `grad_norm`, `epoch`, etc.
_TRAIN_LOG_KEYS_TO_PREFIX = {
    "loss": "train/loss",
    "learning_rate": "train/lr",
    "grad_norm": "train/grad_norm",
    "epoch": "train/epoch",
}


class AstrolabeHFTrainerCallback(TrainerCallback):
    """HuggingFace Trainer callback that streams training metrics to Aim.

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
        ``args.run_name`` if set on Trainer's ``TrainingArguments``,
        falling back to the output directory's basename.
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
        self._wall_time = _core.WallTimeTracker()
        self._rank_zero = is_rank_zero()

    # ------------------------------------------------------------------
    # HF Trainer hooks
    # ------------------------------------------------------------------

    def on_train_begin(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Open the Aim run when training starts."""
        if not self._rank_zero:
            return
        if self._run is not None:
            return  # double-open guard

        # Run name precedence: explicit constructor arg → args.run_name
        # → args.output_dir basename → None.
        run_name = self._explicit_run_name
        if not run_name:
            run_name = getattr(args, "run_name", None)
        if not run_name:
            output_dir = getattr(args, "output_dir", None)
            if output_dir:
                from pathlib import Path
                run_name = Path(output_dir).name or None

        self._run = _core.open_aim_run(self._cfg, run_name=run_name)

    def on_step_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Anchor wall-time at the first training step.

        ``on_step_end`` fires every step which is too noisy for actual
        logging; we use it only to ensure ``mark_first_batch`` runs
        before the first ``on_log`` event so ``wall_time`` is anchored.
        """
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.mark_first_batch()

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Stream metrics from HF's ``logs`` dict to Aim.

        ``logs`` is HF Trainer's pre-aggregated metric dict. Training
        keys (``loss``, ``learning_rate``, etc.) get re-namespaced
        under ``train/``; ``eval_*`` keys get re-namespaced under
        the canonical eval prefix. Anything we don't recognize is
        passed through unchanged so user-added custom metrics still
        land in the run.
        """
        if not self._rank_zero or self._run is None or not logs:
            return

        step = getattr(state, "global_step", None)

        for key, value in logs.items():
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            name = _normalize_log_key(key)
            if name is None:
                continue
            _core.track_safely(
                self._run, name=name, value=value_float, step=step
            )

        # wall_time is part of the `train/` namespace conceptually but
        # gets its own write per log call.
        _core.track_safely(
            self._run,
            name="wall_time",
            value=self._wall_time.elapsed(),
            step=step,
        )

    def on_evaluate(
        self,
        args: Any,
        state: Any,
        control: Any,
        metrics: dict | None = None,
        **kwargs: Any,
    ) -> None:
        """Stream eval metrics from HF's ``metrics`` dict to Aim.

        ``on_evaluate`` fires after each eval pass. The metrics dict
        has ``eval_`` prefixes (``eval_loss``, ``eval_accuracy``); we
        strip and re-namespace under the canonical eval prefix.
        Overlaps with ``on_log`` when ``logging_steps == eval_steps``;
        the duplicate writes are harmless (Aim deduplicates same-step
        same-name writes).
        """
        if not self._rank_zero or self._run is None or not metrics:
            return

        step = getattr(state, "global_step", None)

        for key, value in metrics.items():
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            name = _normalize_log_key(key)
            if name is None:
                continue
            _core.track_safely(
                self._run, name=name, value=value_float, step=step
            )

    def on_train_end(
        self, args: Any, state: Any, control: Any, **kwargs: Any
    ) -> None:
        """Close the Aim run cleanly at end of training."""
        if not self._rank_zero:
            return
        _core.close_run(self._run, status="completed")
        self._run = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _normalize_log_key(key: str) -> str | None:
    """Map an HF log key to an astrolabe-namespaced metric name.

    Returns ``None`` for keys we deliberately skip (currently none —
    every key passes through).

    Mapping rules:

    - ``"loss"`` → ``"train/loss"``
    - ``"learning_rate"`` → ``"train/lr"``
    - ``"grad_norm"``, ``"epoch"`` → ``"train/<key>"``
    - ``"eval_<name>"`` → ``"<EVAL_PREFIX>/<name>"``
      (``"val/<name>"`` as of v1.0.0; pre-v1.0.0 was ``"eval/<name>"``).
    - Anything else → passes through unchanged so user-added custom
      metrics still land in the run.
    """
    if key in _TRAIN_LOG_KEYS_TO_PREFIX:
        return _TRAIN_LOG_KEYS_TO_PREFIX[key]
    if key.startswith("eval_"):
        suffix = key[len("eval_") :]
        if not suffix:
            return None
        # ``"eval_loss"`` → currently ``"eval/loss"``, future ``"val/loss"``
        return f"{_core.EVAL_METRIC_PREFIX}/{suffix}"
    return key
