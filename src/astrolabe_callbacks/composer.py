"""MosaicML Composer logger wired to astrolabe-callbacks ``_core``.

Designed to pair with astrolabe's experiment orchestration: astrolabe
sets ``ASTROLABE_EXPERIMENT_NAME`` and ``AIM_RUN_TAGS`` in the
per-experiment env, and this logger reads them on init to tag the Aim
run with version, submit-id, and any other astrolabe conventions.

This is a Composer ``LoggerDestination`` (not a plain ``Callback``) so
**every** metric the user logs via ``logger.log_metrics(...)`` flows
through to Aim — not just ``train/loss``. The user picks what to track
in their training YAML; we pass it through. The only metric we synthesize
is ``wall_time``.

Usage
-----

::

    from astrolabe_callbacks import AstrolabeComposerLogger
    from composer import Trainer

    trainer = Trainer(
        ...,
        loggers=[AstrolabeComposerLogger()],   # NOTE: loggers=, not callbacks=
    )

The ``loggers=`` slot is required because Composer's ``Logger`` only
broadcasts user metrics to ``LoggerDestination`` subclasses registered
there. Attaching via ``callbacks=`` will still get lifecycle hooks
(``init`` / ``batch_end`` / ``fit_end``) but will silently drop every
``log_metrics`` call — losing all the user-named metrics. Composer
typically rejects this attachment with a clear error since v0.20+.

Standalone usage (no astrolabe)::

    AstrolabeComposerLogger(
        aim_url="aim://localhost:43800",
        experiment_name="my-exp",
        tags={"thesis": "scale-laws"},
    )

Metric names
------------

User-logged metrics flow through unchanged. The only renames we apply
are to Composer's automatic emissions, for cleaner display:

- ``loss/train/total`` → ``train/loss`` (Composer's automatic train loss)
- ``metrics/train/<name>`` → ``train/<name>`` (Composer's per-batch
  training metrics, e.g. accuracy)
- ``metrics/eval/<name>`` → ``<EVAL_PREFIX>/<name>`` (Composer's eval
  metrics; ``EVAL_PREFIX`` flips from ``eval`` to ``val`` in v1.0.0)
- Anything else: pass through as-is. Custom user metrics
  (``my_thing/foo``, ``throughput``) land under their original names.

Failure handling
----------------

This logger never crashes training. If Aim is unreachable or
misconfigured, a single ``WARNING`` is logged at init and every
subsequent operation no-ops. Set ``ASTROLABE_CALLBACK_STRICT=1`` to
flip warnings into raised exceptions for fail-fast CI behavior. See
``_core.py`` for the full failure-handling contract.
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from astrolabe_callbacks import _core
from astrolabe_callbacks._distributed import is_rank_zero

try:
    # ``LoggerDestination`` itself inherits from ``Callback``, so we get
    # both the lifecycle hooks (``init``, ``batch_end``, ``fit_end``)
    # and the metric-broadcast hook (``log_metrics``) on a single base
    # class. The user attaches via ``loggers=`` rather than
    # ``callbacks=``; see the module docstring.
    from composer.loggers import LoggerDestination
except ImportError:  # pragma: no cover — Composer is an optional extra
    LoggerDestination = object  # type: ignore[misc,assignment]

# Re-export from _core for callers reaching for the parser directly.
parse_aim_run_tags = _core.parse_aim_run_tags

__all__ = ["AstrolabeComposerLogger", "parse_aim_run_tags"]


class AstrolabeComposerLogger(LoggerDestination):
    """Composer LoggerDestination that streams all logged metrics to Aim.

    Reads astrolabe env vars (``ASTROLABE_EXPERIMENT_NAME``,
    ``AIM_RUN_TAGS``, ``ASTROLABE_AIM_URL``) on init; constructor
    arguments are the standalone fallback. Attach via ``Trainer(...,
    loggers=[AstrolabeComposerLogger()])``.

    Parameters
    ----------
    aim_url : str | None
        Aim tracking URL. Overridden by ``ASTROLABE_AIM_URL`` env;
        defaults to the astrolabe SSH-tunneled URL when neither is set.
    experiment_name : str | None
        Aim experiment name. Overridden by ``ASTROLABE_EXPERIMENT_NAME``
        env.
    tags : dict[str, str] | None
        Tags applied to the Aim run on init. Overridden by
        ``AIM_RUN_TAGS`` env when set.
    """

    def __init__(
        self,
        aim_url: str | None = None,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
    ):
        self._cfg = _core.resolve_run_config(
            experiment_name=experiment_name,
            aim_url=aim_url,
            tags=tags,
        )
        self._run: Any = None
        self._wall_time = _core.WallTimeTracker()
        self._rank_zero = is_rank_zero()

    # ------------------------------------------------------------------
    # LoggerDestination metric hooks — pass through everything
    # ------------------------------------------------------------------

    def log_metrics(
        self, metrics: dict[str, Any], step: int | None = None
    ) -> None:
        """Pass every user-logged metric through to Aim.

        Composer's ``Logger`` calls this for every
        ``logger.log_metrics(...)`` invocation in the training YAML
        and any user callbacks. The result: any metric the user logs
        — ``MaskedLanguagePerplexity``, custom throughput counters,
        gradient norms — lands in Aim under the same name (with
        cosmetic renames for Composer's automatic emissions; see
        ``_normalize_composer_metric_name``).

        Non-numeric values are silently skipped — Aim only accepts
        scalars. Tensors with ``.item()`` are unwrapped.
        """
        if not self._rank_zero or self._run is None or not metrics:
            return
        for raw_name, value in metrics.items():
            scalar = _to_scalar(value)
            if scalar is None:
                continue
            name = _normalize_composer_metric_name(raw_name)
            _core.track_safely(
                self._run, name=name, value=scalar, step=step
            )

    def log_hyperparameters(self, hyperparameters: dict[str, Any]) -> None:
        """Apply hyperparameters as Aim run params."""
        if not self._rank_zero or self._run is None or not hyperparameters:
            return
        try:
            self._run["hparams"] = hyperparameters
        except Exception as exc:
            if _core.is_strict():
                raise
            logger.debug("Failed to set hparams: {}", exc)

    # ------------------------------------------------------------------
    # Composer Callback lifecycle (LoggerDestination inherits Callback)
    # ------------------------------------------------------------------

    def init(self, state: Any, logger_obj: Any) -> None:
        """Open the Aim run and apply astrolabe tags."""
        if not self._rank_zero:
            return
        if self._run is not None:
            return  # double-open guard

        run_name = getattr(state, "run_name", None)
        self._run = _core.open_aim_run(self._cfg, run_name=run_name)

    def eval_start(self, state: Any, logger_obj: Any) -> None:
        """Pause wall-time accounting during eval."""
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.pause_for_eval()

    def batch_end(self, state: Any, logger_obj: Any) -> None:
        """Anchor wall-time at first training batch + log it.

        Composer's auto-logged train metrics flow through ``log_metrics``
        (called by Composer's Logger before this hook on the same
        batch), so this hook only handles the synthesized ``wall_time``
        metric. The first call also anchors the wall-time clock.
        """
        if not self._rank_zero or self._run is None:
            return
        self._wall_time.mark_first_batch()

        try:
            step = int(state.timestamp.batch)
        except Exception:
            step = None
        _core.track_safely(
            self._run,
            name="wall_time",
            value=self._wall_time.elapsed(),
            step=step,
        )

    def eval_end(self, state: Any, logger_obj: Any) -> None:
        """Resume wall-time accounting after eval."""
        # Resume even if logging was disabled — accounting must be
        # right for the next batch_end.
        self._wall_time.resume()

    def fit_end(self, state: Any, logger_obj: Any) -> None:
        """Close the Aim run cleanly."""
        if not self._rank_zero:
            return
        _core.close_run(self._run, status="completed")
        self._run = None

    def post_close(self) -> None:
        """LoggerDestination hook fired during Trainer cleanup.

        Composer calls this even on training failures, so it's our
        last-ditch chance to mark a run as failed if ``fit_end``
        didn't fire. Idempotent; if ``fit_end`` already closed the run
        cleanly, ``self._run`` is ``None`` and this no-ops.
        """
        if not self._rank_zero or self._run is None:
            return
        _core.close_run(self._run, status="failed")
        self._run = None


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _to_scalar(value: Any) -> float | None:
    """Coerce a value to ``float``, or return ``None`` if not numeric.

    Handles tensors (anything with ``.item()``), Python scalars, and
    rejects strings, dicts, None, NaN-shaped Aim-rejected types.
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


def _normalize_composer_metric_name(name: str) -> str:
    """Re-namespace Composer's automatic emissions to astrolabe convention.

    Cosmetic only — user-named metrics pass through unchanged. The
    full mapping:

    - ``loss/train/total`` → ``train/loss``
    - ``metrics/train/<name>`` → ``train/<name>``
    - ``metrics/eval/<name>`` → ``<EVAL_PREFIX>/<name>``
    - everything else → unchanged
    """
    if name == "loss/train/total":
        return "train/loss"
    if name.startswith("metrics/train/"):
        return f"train/{name[len('metrics/train/') :]}"
    if name.startswith("metrics/eval/"):
        suffix = name[len("metrics/eval/") :]
        return f"{_core.EVAL_METRIC_PREFIX}/{suffix}"
    return name
