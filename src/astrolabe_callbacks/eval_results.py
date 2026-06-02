"""Producer-side helpers for logging benchmark eval results.

Researchers call :func:`log_eval_table` (primary, one-shot) or
:func:`start_eval_run` (lower-level escape hatch for streams / custom
metric names) from a post-training eval script. Both emit an Aim run
tagged with the three-tag contract that astrolabe's dashboard
discovers from the model-run page:

* ``astrolabe.kind = "eval"`` — discriminator alongside ``"metadata"``
  (engine-written cost runs) and the implicit training runs.
* ``astrolabe.task_set = "glue"`` — human label that groups sections
  on the dashboard's Eval tab.
* ``astrolabe.model_run_hash = "<training_run_hash>"`` — the join key.
  One eval run scores exactly one training run.

Metric path convention: ``eval/<task>/<metric>`` — the dashboard parses
this to populate the table's row (task) and metric column.

This lives in ``astrolabe-callbacks`` rather than the main ``astrolabe``
package so training/eval repos depend on **one** lightweight library
for all Aim instrumentation — they never pull in the orchestration
framework. It uses the same ``aim_url`` / ``ASTROLABE_AIM_URL``
connection convention as the framework callbacks and the raw-PyTorch
``Run`` context manager.

The ``eval/`` namespace here is distinct from ``val/`` (during-training
validation metrics emitted by the framework callbacks). ``val/`` lives
on the training run and the dashboard's Training tab; ``eval/`` lives on
a separate eval run and the Eval tab. See the package README for the
full namespace split.
"""

from __future__ import annotations

import os
from typing import Any

from ._core import DEFAULT_AIM_URL

__all__ = [
    "EvalInputError",
    "log_eval_table",
    "start_eval_run",
]


class EvalInputError(ValueError):
    """Raised when an eval helper receives malformed input.

    Surfaces in the researcher's eval script with a clear message —
    we'd rather fail loudly at the call site than write a half-formed
    Aim run that confuses the dashboard later.
    """


def _resolve_aim_url(aim_url: str | None) -> str:
    """Resolve the Aim connection URL with the lib's standard precedence.

    ``ASTROLABE_AIM_URL`` env wins over the constructor argument, which
    wins over :data:`DEFAULT_AIM_URL`. Matches ``resolve_run_config`` so
    an eval script run on the same instance as training connects the
    same way without extra configuration.

    Note this deliberately does NOT reuse ``resolve_run_config``: that
    helper also reads ``ASTROLABE_EXPERIMENT_NAME`` and ``AIM_RUN_TAGS``,
    which carry the *training* run's identity. An eval run must be filed
    under ``eval/<task_set>`` with its own three tags, so only the URL
    resolution is shared.
    """
    return os.environ.get("ASTROLABE_AIM_URL") or aim_url or DEFAULT_AIM_URL


def _validate_identity(model_run_hash: str, task_set: str) -> None:
    if not isinstance(model_run_hash, str) or not model_run_hash:
        raise EvalInputError("model_run_hash must be a non-empty string")
    if not isinstance(task_set, str) or not task_set:
        raise EvalInputError("task_set must be a non-empty string")


def _validate_rows(rows: dict[str, tuple[str, float]]) -> None:
    if not isinstance(rows, dict):
        raise EvalInputError(
            f"rows must be a dict, got {type(rows).__name__}"
        )
    if not rows:
        raise EvalInputError("rows must contain at least one task")
    for task, value in rows.items():
        if not isinstance(task, str) or not task:
            raise EvalInputError(
                f"task name must be a non-empty string, got {task!r}"
            )
        if "/" in task:
            # The dashboard parses ``eval/<task>/<metric>`` — embedding
            # a slash in the task name silently scrambles which segment
            # is which.
            raise EvalInputError(
                f"task name {task!r} must not contain '/'; "
                f"use a flat label per task"
            )
        if not isinstance(value, tuple) or len(value) != 2:
            raise EvalInputError(
                f"row {task!r} must be a (metric, score) tuple, got {value!r}"
            )
        metric, score = value
        if not isinstance(metric, str) or not metric:
            raise EvalInputError(
                f"metric label for task {task!r} must be a non-empty string, "
                f"got {metric!r}"
            )
        if "/" in metric:
            raise EvalInputError(
                f"metric label {metric!r} for task {task!r} must not contain '/'"
            )
        # bool is a subclass of int in Python — exclude it explicitly so
        # ``log_eval_table(rows={"cola": ("accuracy", True)})`` fails
        # loudly instead of logging 1.0.
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise EvalInputError(
                f"score for task {task!r} must be a number, "
                f"got {type(score).__name__}"
            )


def start_eval_run(
    *,
    model_run_hash: str,
    task_set: str,
    aim_url: str | None = None,
) -> Any:
    """Open an Aim run tagged for eval discovery.

    Lower-level helper for mid-training rolling evals, custom metric
    names, or anywhere the researcher needs full control over the Aim
    run's lifecycle. For the common case (one-shot post-training table)
    use :func:`log_eval_table` instead.

    The caller owns the returned ``aim.Run`` — they call ``.track(...)``
    to log values and ``.close()`` when finished. Forgetting to close
    leaves the run's ``end_time`` as zero; the dashboard will still
    display the run, but cost / duration views may treat it as
    in-flight indefinitely.

    Parameters
    ----------
    model_run_hash : str
        Hash of the training Aim run this eval scores. Must be
        non-empty. Becomes ``astrolabe.model_run_hash`` on the tag set;
        the dashboard uses this to discover eval runs from the model's
        experiment page.
    task_set : str
        Human label for the benchmark suite (``"glue"``, ``"mmlu"``,
        ``"agent-rollouts-2026q2"``, etc.). Becomes
        ``astrolabe.task_set``. Groups sections in the dashboard.
    aim_url : str | None
        Aim tracking URL. ``ASTROLABE_AIM_URL`` env wins over this
        argument; defaults to ``aim://localhost:43800`` (the SSH
        reverse tunnel astrolabe opens on GPU instances). Accepts a
        filesystem path too — ``aim.Run`` resolves either.

    Returns
    -------
    aim.Run
        An open Aim run with the three identity tags already set.

    Raises
    ------
    EvalInputError
        If ``model_run_hash`` or ``task_set`` is empty or not a string.
    ImportError
        If the ``aim`` package isn't installed (re-raised, not
        swallowed — eval scripts that can't reach Aim should fail
        loudly).

    Examples
    --------
    >>> run = start_eval_run(
    ...     model_run_hash="abc123",
    ...     task_set="glue",
    ... )
    >>> for checkpoint in (10_000, 20_000, 30_000):
    ...     run.track(
    ...         score_at(checkpoint),
    ...         name="eval/cola/matthews",
    ...         step=checkpoint,
    ...     )
    >>> run.close()
    """
    _validate_identity(model_run_hash, task_set)

    from aim import Run

    # Filing eval runs under ``eval/<task_set>`` keeps them out of the
    # model experiment's Aim run list (which the dashboard already
    # interprets as training runs). Discovery uses the tags, not the
    # Aim experiment name, so this is purely an Aim-UI convenience.
    run = Run(experiment=f"eval/{task_set}", repo=_resolve_aim_url(aim_url))
    run["astrolabe.kind"] = "eval"
    run["astrolabe.task_set"] = task_set
    run["astrolabe.model_run_hash"] = model_run_hash
    return run


def log_eval_table(
    *,
    model_run_hash: str,
    task_set: str,
    rows: dict[str, tuple[str, float]],
    aim_url: str | None = None,
) -> str:
    """Log a one-shot benchmark table for a single training run.

    Primary surface — researchers hand a dict, the library handles the
    Aim mechanics. Each ``(task, (metric, score))`` entry becomes a
    metric tracked at ``step=0`` under the name ``eval/<task>/<metric>``.
    The dashboard's table block parses this convention to populate the
    leaderboard column for that task.

    The Aim run is opened, tagged, populated, and closed atomically.
    If validation fails, no Aim run is created.

    Parameters
    ----------
    model_run_hash : str
        Hash of the training Aim run this eval scores.
    task_set : str
        Human label for the benchmark suite (``"glue"``, ``"mmlu"``, …).
    rows : dict[str, tuple[str, float]]
        ``{task_name: (metric_label, score)}``. ``task_name`` and
        ``metric_label`` are flat strings (no slashes). ``score`` is a
        number (int or float, not bool). At least one row is required.

        For an averaged-across-tasks summary column, log it as one of
        the rows (conventionally ``"avg"``) — the dashboard renders
        ``"avg"`` as the last column. The library does not compute
        aggregates itself; that's the researcher's call (mean? harmonic
        mean? a paper-specific subset?).
    aim_url : str | None
        Aim tracking URL. ``ASTROLABE_AIM_URL`` env wins; defaults to
        ``aim://localhost:43800``.

    Returns
    -------
    str
        The Aim run hash of the newly-created eval run.

    Raises
    ------
    EvalInputError
        If any input is malformed. No Aim run is created in that case.
    ImportError
        If the ``aim`` package isn't installed.

    Examples
    --------
    >>> log_eval_table(
    ...     model_run_hash="abc123",
    ...     task_set="glue",
    ...     rows={
    ...         "cola": ("matthews", 0.822),
    ...         "sst2": ("accuracy", 0.943),
    ...         "mnli": ("accuracy_matched", 0.864),
    ...         "avg":  ("mean", 0.876),
    ...     },
    ... )
    'b73e9c8d4f6a...'
    """
    _validate_identity(model_run_hash, task_set)
    _validate_rows(rows)

    run = start_eval_run(
        model_run_hash=model_run_hash,
        task_set=task_set,
        aim_url=aim_url,
    )
    try:
        for task, (metric, score) in rows.items():
            run.track(float(score), name=f"eval/{task}/{metric}", step=0)
        run_hash = run.hash
    finally:
        run.close()
    return run_hash
