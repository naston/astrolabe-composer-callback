"""Tests for ``AstrolabeLightningLogger``.

Lightning-specific contract: every metric the user logs via
``self.log(...)`` lands in ``trainer.callback_metrics`` and flows
through to Aim. Keys prefixed with ``val_`` / ``val/`` are eval-side
and re-namespaced under the canonical eval prefix; everything else
is training-side and passes through unchanged.
"""

from __future__ import annotations

import pytest

from astrolabe_callbacks._core import EVAL_METRIC_PREFIX
from astrolabe_callbacks.lightning import (
    AstrolabeLightningLogger,
    _is_val_metric,
    _normalize_val_metric_name,
    _to_scalar,
)


# ----------------------------------------------------------------------
# Helper: tiny stand-ins for Lightning's Trainer / LightningModule
# ----------------------------------------------------------------------


class _FakeTrainer:
    """Minimal Trainer stub. Tests that need callback_metrics set."""

    def __init__(self, callback_metrics: dict | None = None, logger=None):
        self.callback_metrics = callback_metrics or {}
        self.logger = logger


class _FakeModule:
    """Minimal LightningModule stub — we only read __class__.__name__."""

    pass


class _FakeLogger:
    def __init__(self, name: str):
        self.name = name


# ----------------------------------------------------------------------
# val-metric helpers — edge cases
# ----------------------------------------------------------------------


class TestValMetricHelpers:
    @pytest.mark.parametrize(
        "name,expected",
        [
            ("val_loss", True),
            ("val/accuracy", True),
            ("val_", True),  # edge: prefix only
            ("val/", True),
            ("train_loss", False),
            ("validation_loss", False),  # not "val_" prefix
            ("validate", False),
            ("", False),
            ("Val_loss", False),  # case-sensitive
        ],
    )
    def test_is_val_metric(self, name, expected):
        assert _is_val_metric(name) is expected

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("val_loss", "loss"),
            ("val/accuracy", "accuracy"),
            ("val_", None),  # empty suffix → None
            ("val/", None),
            ("train_loss", None),
            ("val/glue/mnli", "glue/mnli"),  # nested slashes preserved
        ],
    )
    def test_normalize_val_metric_name(self, name, expected):
        assert _normalize_val_metric_name(name) == expected

    def test_to_scalar_handles_tensors(self):
        import torch
        assert _to_scalar(torch.tensor(0.5)) == 0.5

    def test_to_scalar_returns_none_for_non_numeric(self):
        assert _to_scalar("nope") is None
        assert _to_scalar(None) is None
        assert _to_scalar({"k": 1}) is None


# ----------------------------------------------------------------------
# Constructor + setup
# ----------------------------------------------------------------------


class TestSetup:
    def test_setup_only_runs_for_fit_stage(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        # validate / test / predict stages should not open a run.
        cb.setup(_FakeTrainer(), _FakeModule(), stage="validate")
        cb.setup(_FakeTrainer(), _FakeModule(), stage="test")
        cb.setup(_FakeTrainer(), _FakeModule(), stage="predict")
        assert fake_aim_run == []  # no runs opened
        assert cb._run is None

    def test_setup_opens_run_for_fit(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        assert cb._run is not None

    def test_double_setup_does_not_re_open(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        first_run = cb._run
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        assert cb._run is first_run
        assert len(fake_aim_run) == 1

    def test_run_name_explicit_wins(self, fake_aim_run):
        cb = AstrolabeLightningLogger(run_name="my-explicit-name")
        cb.setup(
            _FakeTrainer(logger=_FakeLogger("logger-name")),
            _FakeModule(),
            stage="fit",
        )
        assert fake_aim_run[-1].name == "my-explicit-name"

    def test_run_name_falls_back_to_trainer_logger(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(
            _FakeTrainer(logger=_FakeLogger("from-trainer-logger")),
            _FakeModule(),
            stage="fit",
        )
        assert fake_aim_run[-1].name == "from-trainer-logger"

    def test_run_name_falls_back_to_module_class(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(logger=None), _FakeModule(), stage="fit")
        assert fake_aim_run[-1].name == "_FakeModule"

    def test_rank_nonzero_no_setup(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "1")
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        assert fake_aim_run == []
        assert cb._run is None


# ----------------------------------------------------------------------
# on_train_batch_end — training pass-through
# ----------------------------------------------------------------------


class TestOnTrainBatchEnd:
    def test_passes_through_user_metrics(self, fake_aim_run):
        # Critical for the user contract: any metric the user
        # ``self.log()`` ed in their training_step lands in callback_metrics
        # and flows through unchanged.
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        import torch
        trainer = _FakeTrainer(
            callback_metrics={
                "loss": torch.tensor(0.5),
                "throughput": torch.tensor(120.0),
                "MyCustomMetric": torch.tensor(0.99),
            }
        )
        cb.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=0)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "loss" in names
        assert "throughput" in names
        assert "MyCustomMetric" in names

    def test_skips_val_metrics(self, fake_aim_run):
        # val_loss / val/accuracy are eval-side; on_train_batch_end
        # must not log them (would double-write with on_validation_end).
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        import torch
        trainer = _FakeTrainer(
            callback_metrics={
                "loss": torch.tensor(0.5),
                "val_loss": torch.tensor(0.4),
                "val/accuracy": torch.tensor(0.9),
            }
        )
        cb.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=0)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "loss" in names
        assert "val_loss" not in names
        assert "val/accuracy" not in names

    def test_logs_wall_time(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        cb.on_train_batch_end(_FakeTrainer(), _FakeModule(), outputs=None, batch=None, batch_idx=0)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "wall_time" in names

    def test_skips_non_numeric_values(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        trainer = _FakeTrainer(
            callback_metrics={"good": 1.0, "bad_str": "nope"}
        )
        cb.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=0)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "good" in names
        assert "bad_str" not in names

    def test_no_run_is_noop(self):
        cb = AstrolabeLightningLogger()
        # setup never called → _run stays None
        cb.on_train_batch_end(
            _FakeTrainer(), _FakeModule(), outputs=None, batch=None, batch_idx=0
        )  # must not raise

    def test_rank_nonzero_is_noop(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "1")
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        trainer = _FakeTrainer(callback_metrics={"loss": 0.5})
        cb.on_train_batch_end(trainer, _FakeModule(), outputs=None, batch=None, batch_idx=0)
        assert fake_aim_run == []


# ----------------------------------------------------------------------
# on_validation_end — eval pass-through
# ----------------------------------------------------------------------


class TestOnValidationEnd:
    def test_namespaces_val_metrics_under_eval_prefix(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        import torch
        trainer = _FakeTrainer(
            callback_metrics={
                "val_loss": torch.tensor(0.4),
                "val/accuracy": torch.tensor(0.95),
                "val_perplexity": torch.tensor(15.0),
            }
        )
        cb.on_validation_end(trainer, _FakeModule())
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert f"{EVAL_METRIC_PREFIX}/loss" in names
        assert f"{EVAL_METRIC_PREFIX}/accuracy" in names
        assert f"{EVAL_METRIC_PREFIX}/perplexity" in names

    def test_skips_train_metrics(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        import torch
        trainer = _FakeTrainer(
            callback_metrics={
                "loss": torch.tensor(0.5),
                "train_loss": torch.tensor(0.5),
                "val_loss": torch.tensor(0.4),
            }
        )
        cb.on_validation_end(trainer, _FakeModule())
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert f"{EVAL_METRIC_PREFIX}/loss" in names
        assert "loss" not in names
        assert "train_loss" not in names

    def test_resumes_wall_time_on_validation_end(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        cb.on_validation_start(_FakeTrainer(), _FakeModule())
        # _eval_start is now > 0
        assert cb._wall_time._eval_start > 0
        cb.on_validation_end(_FakeTrainer(), _FakeModule())
        # After resume, _eval_start cleared.
        assert cb._wall_time._eval_start == 0

    def test_no_run_is_noop(self):
        cb = AstrolabeLightningLogger()
        cb.on_validation_end(_FakeTrainer(), _FakeModule())  # must not raise


# ----------------------------------------------------------------------
# Lifecycle: train_end + on_exception
# ----------------------------------------------------------------------


class TestLifecycleClose:
    def test_train_end_closes_with_completed(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        run = cb._run
        cb.on_train_end(_FakeTrainer(), _FakeModule())
        assert run.tags["astrolabe.status"] == "completed"
        assert run.closed is True
        assert cb._run is None

    def test_on_exception_closes_with_failed(self, fake_aim_run):
        cb = AstrolabeLightningLogger()
        cb.setup(_FakeTrainer(), _FakeModule(), stage="fit")
        run = cb._run
        cb.on_exception(_FakeTrainer(), _FakeModule(), RuntimeError("OOM"))
        assert run.tags["astrolabe.status"] == "failed"
        assert run.closed is True
        assert cb._run is None
