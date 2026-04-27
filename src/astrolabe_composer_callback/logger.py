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
        self._experiment_name = experiment_name or os.environ.get(
            "ASTROLABE_EXPERIMENT_NAME", ""
        )
        self._log_interval = log_interval
        self._tags: dict[str, str] = (
            dict(tags) if tags is not None else parse_aim_run_tags(os.environ.get("AIM_RUN_TAGS"))
        )
        self._run = None
        self._step = 0
        self._start_time: float = 0.0

    def init(self, state: Any, logger_obj: Any) -> None:
        """Initialize the Aim run when training starts."""
        try:
            from aim import Run
            self._run = Run(
                repo=self._repo,
                experiment=self._experiment_name or None,
            )
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
            self._start_time = time.monotonic()
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

    def batch_end(self, state: Any, logger_obj: Any) -> None:
        """Log batch-level metrics + wall time."""
        if not self._run:
            return

        self._step += 1
        if self._step % self._log_interval != 0:
            return

        elapsed = time.monotonic() - self._start_time

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
                self._run.track(
                    loss_val,
                    name="train/loss",
                    step=self._step,
                    context={"x_axis": "wall_time"},
                )
        except Exception:
            pass

        # Wall time as its own metric so the dashboard can use it as the
        # x-axis when researchers want elapsed-time charts.
        try:
            self._run.track(elapsed, name="wall_time", step=self._step)
        except Exception:
            pass

    def eval_end(self, state: Any, logger_obj: Any) -> None:
        """Log eval metrics with cleaned-up names.

        Composer namespaces eval metrics by the eval-suite name (e.g.
        ``glue_mnli/accuracy``); when the suite is just called "eval" we
        flatten it to ``eval/accuracy`` for cleaner display.
        """
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
