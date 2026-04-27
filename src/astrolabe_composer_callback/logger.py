"""Composer callback that logs metrics to a remote Aim server.

Designed to pair with astrolabe's experiment orchestration: astrolabe sets
``AIM_RUN_TAGS`` in the per-experiment env, and this callback reads it on
init to tag the Aim run with version, submit-id, and any other
astrolabe-set conventions. Without astrolabe (i.e. running Composer
standalone), the callback degrades gracefully — set ``tags`` explicitly
on the constructor or just leave it unset.

Usage with astrolabe::

    from astrolabe_composer_callback import AstrolabeLogger

    trainer = Trainer(
        ...
        callbacks=[AstrolabeLogger(repo="aim://localhost:43800")],
    )

The callback reads ``AIM_RUN_TAGS`` from env on init. Astrolabe writes
that env var at engine ``_setup`` time, so the resulting Aim run carries
``astrolabe.version=v3``, ``astrolabe.submit_id=<uuid>``, etc.

Standalone usage (no astrolabe)::

    AstrolabeLogger(
        repo="aim://localhost:43800",
        experiment_name="my-exp",
        tags={"thesis": "scale-laws", "model": "BERT"},
    )

Either set ``tags`` directly or set ``AIM_RUN_TAGS`` env var manually.
"""

from __future__ import annotations

import os
import time
from typing import Any

from loguru import logger

try:
    # Only the base Callback class needs to be a real type at runtime —
    # it provides `run_event`, which Composer's engine calls for every
    # lifecycle event. Logger / State are only used as type hints, so we
    # import them as `Any` and don't depend on their import paths
    # (which have shifted across Composer versions: Logger moved from
    # composer.core to composer.loggers around 0.21, and importing the
    # old path raises ImportError, which would silently drop us back
    # to inheriting from `object` and break run_event dispatch).
    from composer.core import Callback
except ImportError:  # pragma: no cover — Composer is a runtime dep
    Callback = object  # type: ignore[misc,assignment]

__all__ = ["AstrolabeLogger", "parse_aim_run_tags"]


def parse_aim_run_tags(raw: str | None) -> dict[str, str]:
    """Parse the ``AIM_RUN_TAGS`` env var into a tag dict.

    Format: ``key1=val1,key2=val2``. Whitespace around keys/values is
    stripped. Entries without ``=``, with empty keys, or with duplicate
    keys (last wins) are tolerated rather than raising — this is reading
    something a researcher pasted into a shell.

    Parameters
    ----------
    raw : str | None
        The raw env var value, or ``None`` / empty.

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


class AstrolabeLogger(Callback):
    """MosaicML Composer callback that streams training metrics to Aim.

    Parameters
    ----------
    repo : str
        Aim tracking URI (e.g. ``aim://localhost:43800``). When running
        under astrolabe, the SSH reverse tunnel makes ``localhost:43800``
        on the GPU instance route back to the NUC's Aim server.
    experiment_name : str
        Experiment name for the Aim run. Falls back to the
        ``ASTROLABE_EXPERIMENT_NAME`` env var when empty so astrolabe's
        engine-set name flows through automatically.
    log_interval : int
        Log batch metrics every N batches. Default 1.
    tags : dict[str, str] | None
        Tags applied to the Aim run on init (e.g.
        ``{"astrolabe.version": "v3"}``). Falls back to parsing the
        ``AIM_RUN_TAGS`` env var when ``None``. Pass an empty dict to
        explicitly disable env fallback.
    """

    def __init__(
        self,
        repo: str = "aim://localhost:43800",
        experiment_name: str = "",
        log_interval: int = 1,
        tags: dict[str, str] | None = None,
    ):
        self._repo = repo
        # Precedence: astrolabe env vars win when set. Astrolabe is the
        # orchestrator; its experiment name and tags are authoritative
        # over anything baked into a Composer training YAML. Researchers
        # hardcoding experiment_name in their training config get that
        # value when running standalone, but astrolabe overrides when
        # it's the one driving the run. Without this, hardcoded YAMLs
        # silently produce runs under the wrong Aim experiment and the
        # dashboard can't find them.
        env_exp = os.environ.get("ASTROLABE_EXPERIMENT_NAME", "")
        self._experiment_name = env_exp or experiment_name
        self._log_interval = log_interval
        env_tags = parse_aim_run_tags(os.environ.get("AIM_RUN_TAGS"))
        self._tags: dict[str, str] = env_tags if env_tags else (dict(tags) if tags is not None else {})
        self._run = None
        self._step = 0
        # `_start_time` is anchored to the FIRST training batch (not init),
        # so wall_time excludes setup: model loading, dataloader build,
        # tokenizer warmup, etc. A run that takes 30s to set up and 60s
        # to actually train should report wall_time = 60s, not 90s, so
        # researchers comparing two runs aren't seeing setup-cost differences.
        self._start_time: float = 0.0
        # `_total_eval_time` accumulates seconds spent inside eval; we
        # subtract it from wall_time on each batch_end so wall_time is
        # *training-only* elapsed time. Two runs with the same training
        # cost but different eval cadence/duration land on the same
        # wall_time for the same step — comparison stays apples-to-apples.
        self._eval_start: float = 0.0
        self._total_eval_time: float = 0.0

    def init(self, state: Any, logger_obj: Any) -> None:
        """Initialize the Aim run when training starts."""
        try:
            from aim import Run
            self._run = Run(
                repo=self._repo,
                experiment=self._experiment_name or None,
            )
            # Carry Composer's run_name through to Aim so the dashboard
            # shows the meaningful name (e.g. "bert-tiny", "bert-2layer")
            # set in the training YAML, not the auto-generated
            # "Run: <hash>" placeholder. Aim's Run object exposes
            # `name` as a writable property.
            composer_run_name = getattr(state, "run_name", None)
            if composer_run_name:
                try:
                    self._run.name = composer_run_name
                except Exception as e:
                    logger.debug("Failed to set Aim run name {}: {}", composer_run_name, e)
            for key, value in self._tags.items():
                # Aim's Run is a dict-like; assignment writes a tag/param
                # readable from the dashboard later. Tags written here
                # land alongside hyperparameters in the run's metadata.
                try:
                    self._run[key] = value
                except Exception as e:
                    logger.debug("Failed to set Aim tag {}={}: {}", key, value, e)
            # Hyperparameter logging from Composer state when available.
            if hasattr(state, "model") and hasattr(state.model, "config"):
                try:
                    self._run["hparams"] = state.model.config
                except Exception:
                    pass
            # Don't set _start_time here — anchor it to the first batch_end
            # in batch_end() so setup time (model load, dataloader build,
            # tokenizer warmup, weight init) is excluded from wall_time.
            tag_summary = (
                f" (tags: {', '.join(f'{k}={v}' for k, v in self._tags.items())})"
                if self._tags
                else ""
            )
            logger.info("Aim logger connected to {}{}", self._repo, tag_summary)
        except ImportError:
            logger.warning("aim not installed — AstrolabeLogger disabled. pip install aim")
        except Exception as e:
            logger.warning("Aim connection failed (non-fatal): {}", e)
            self._run = None

    def eval_start(self, state: Any, logger_obj: Any) -> None:
        """Mark eval start so we can subtract eval duration from wall_time."""
        self._eval_start = time.monotonic()

    def batch_end(self, state: Any, logger_obj: Any) -> None:
        """Log batch-level metrics + wall time."""
        if not self._run:
            return

        # Anchor wall_time to the first training batch so setup time is
        # excluded. _start_time stays 0 until we get here.
        if self._start_time == 0.0:
            self._start_time = time.monotonic()

        self._step += 1
        if self._step % self._log_interval != 0:
            return

        # Training-only elapsed: total time since first batch, minus
        # accumulated eval time. Excludes setup AND eval pauses, so the
        # wall_time x-axis reflects pure training compute. Two runs with
        # the same training cost but different eval cadence stay aligned.
        elapsed = (time.monotonic() - self._start_time) - self._total_eval_time

        # Track the canonical training loss when Composer exposes it. This
        # is the simplest path that works across model types — anything
        # exotic, the researcher logs from their own callback.
        try:
            if hasattr(state, "loss") and state.loss is not None:
                loss_val = (
                    state.loss.item()
                    if hasattr(state.loss, "item")
                    else float(state.loss)
                )
                self._run.track(loss_val, name="train/loss", step=self._step)
        except Exception:
            pass

        # Wall time as its own metric so the dashboard can use it as the
        # x-axis when researchers want elapsed-time charts.
        try:
            self._run.track(elapsed, name="wall_time", step=self._step)
        except Exception:
            pass

    def eval_end(self, state: Any, logger_obj: Any) -> None:
        """Log eval metrics with cleaned-up names + bank the eval duration.

        Composer namespaces eval metrics by the eval-suite name (e.g.
        ``glue_mnli/accuracy``); when the suite is just called "eval" we
        flatten it to ``eval/accuracy`` for cleaner display.

        Accumulates the eval duration into ``_total_eval_time`` so the
        next batch_end's wall_time excludes it.
        """
        # Accumulate eval duration first — even if logging fails, the
        # wall-time accounting needs to be right.
        if self._eval_start > 0:
            self._total_eval_time += time.monotonic() - self._eval_start
            self._eval_start = 0.0

        if not self._run:
            return
        try:
            step = int(state.timestamp.batch)
            for eval_name, metrics in state.eval_metrics.items():
                for metric_name, metric in metrics.items():
                    try:
                        value = metric.compute().item()
                    except Exception:
                        continue
                    if eval_name == "eval":
                        name = f"eval/{metric_name}"
                    else:
                        name = f"{eval_name}/{metric_name}"
                    self._run.track(value, name=name, step=step)
        except Exception:
            pass

    def fit_end(self, state: Any, logger_obj: Any) -> None:
        """Finalize and flush the Aim run."""
        if self._run:
            try:
                self._run.close()
                logger.info("Aim run finalized and flushed")
            except Exception:
                pass
