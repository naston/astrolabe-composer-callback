"""Tests for ``AstrolabeHFTrainerCallback``.

HF Trainer's ``on_log`` hook receives a pre-aggregated metrics dict.
We pass through everything; specific keys (``loss``, ``learning_rate``,
``grad_norm``, ``epoch``) get re-namespaced under ``train/`` for
display ergonomics. ``eval_*`` keys get re-namespaced under the
canonical eval prefix.
"""

from __future__ import annotations

import pytest

from astrolabe_callbacks._core import EVAL_METRIC_PREFIX
from astrolabe_callbacks.huggingface import (
    AstrolabeHFTrainerCallback,
    _normalize_log_key,
)


# ----------------------------------------------------------------------
# Helpers: minimal HF stand-ins
# ----------------------------------------------------------------------


class _FakeTrainerArgs:
    def __init__(self, run_name: str | None = None, output_dir: str | None = None):
        self.run_name = run_name
        self.output_dir = output_dir


class _FakeState:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


class _FakeControl:
    pass


# ----------------------------------------------------------------------
# Key normalizer — edge cases
# ----------------------------------------------------------------------


class TestNormalizeLogKeyEdgeCases:
    def test_empty_eval_suffix(self):
        # "eval_" with empty suffix → None (skipped to avoid empty
        # metric names that Aim would refuse).
        assert _normalize_log_key("eval_") is None

    def test_passes_through_unknown_train_keys(self):
        # User-added custom training metric. Must NOT be auto-prefixed
        # — the user's name is theirs, not ours to rewrite.
        assert _normalize_log_key("my_custom_metric") == "my_custom_metric"

    def test_passes_through_namespaced_user_metrics(self):
        assert _normalize_log_key("custom/throughput") == "custom/throughput"

    def test_eval_with_underscored_suffix(self):
        # "eval_my_custom" → "eval/my_custom" (whole suffix preserved).
        assert _normalize_log_key("eval_my_custom") == f"{EVAL_METRIC_PREFIX}/my_custom"

    def test_uppercase_eval_not_recognized(self):
        # Case-sensitive. HF emits lowercase.
        assert _normalize_log_key("EVAL_loss") == "EVAL_loss"

    def test_just_eval_no_underscore(self):
        # "eval" alone is not "eval_" so passes through unchanged.
        assert _normalize_log_key("eval") == "eval"


class TestNormalizeLogKeyHappyPath:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("loss", "train/loss"),
            ("learning_rate", "train/lr"),
            ("grad_norm", "train/grad_norm"),
            ("epoch", "train/epoch"),
            ("eval_loss", f"{EVAL_METRIC_PREFIX}/loss"),
            ("eval_accuracy", f"{EVAL_METRIC_PREFIX}/accuracy"),
            ("eval_f1", f"{EVAL_METRIC_PREFIX}/f1"),
            ("custom", "custom"),  # user-defined passes through
        ],
    )
    def test_canonical_renames(self, raw, expected):
        assert _normalize_log_key(raw) == expected


# ----------------------------------------------------------------------
# Constructor + on_train_begin
# ----------------------------------------------------------------------


class TestOnTrainBegin:
    def test_opens_run_with_explicit_name(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback(run_name="my-run")
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert fake_aim_run[-1].name == "my-run"

    def test_run_name_from_args(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(
            _FakeTrainerArgs(run_name="hf-args-run-name"),
            _FakeState(),
            _FakeControl(),
        )
        assert fake_aim_run[-1].name == "hf-args-run-name"

    def test_run_name_from_output_dir(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(
            _FakeTrainerArgs(output_dir="/tmp/checkpoints/bert-v3"),
            _FakeState(),
            _FakeControl(),
        )
        # basename of the output_dir.
        assert fake_aim_run[-1].name == "bert-v3"

    def test_run_name_chain_explicit_wins(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback(run_name="explicit")
        cb.on_train_begin(
            _FakeTrainerArgs(run_name="from-args", output_dir="/tmp/from-dir"),
            _FakeState(),
            _FakeControl(),
        )
        assert fake_aim_run[-1].name == "explicit"

    def test_double_on_train_begin_no_op(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        first_run = cb._run
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert cb._run is first_run
        assert len(fake_aim_run) == 1

    def test_rank_nonzero_no_run(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "3")
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert fake_aim_run == []
        assert cb._run is None


# ----------------------------------------------------------------------
# on_log — primary metric flow
# ----------------------------------------------------------------------


class TestOnLogEdgeCases:
    def test_no_logs_no_op(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(_FakeTrainerArgs(), _FakeState(), _FakeControl(), logs=None)
        cb.on_log(_FakeTrainerArgs(), _FakeState(), _FakeControl(), logs={})
        # Only wall_time would be logged if logs were non-empty; here
        # nothing.
        assert fake_aim_run[-1].tracked == []

    def test_skips_non_numeric_values(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(global_step=10), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={"loss": 0.5, "tag": "experiment-name"},
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        # Strings can't coerce to float → silently skipped.
        assert "tag" not in names

    def test_no_run_is_noop(self):
        cb = AstrolabeHFTrainerCallback()
        # on_train_begin not called → _run stays None
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(),
            _FakeControl(),
            logs={"loss": 0.5},
        )  # must not raise

    def test_rank_nonzero_is_noop(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "1")
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={"loss": 0.5},
        )
        assert fake_aim_run == []


class TestOnLogHappyPath:
    def test_passes_through_user_metrics(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(global_step=0), _FakeControl())
        cb.on_step_end(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=10),
            _FakeControl(),
            logs={
                "loss": 0.5,
                "learning_rate": 1e-4,
                "my_custom_metric": 99.0,
            },
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        assert "train/lr" in names
        assert "my_custom_metric" in names  # user-named pass-through
        assert "wall_time" in names

    def test_step_carries_through(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_log(
            _FakeTrainerArgs(),
            _FakeState(global_step=42),
            _FakeControl(),
            logs={"loss": 0.5},
        )
        loss_entry = next(
            t for t in fake_aim_run[-1].tracked if t["name"] == "train/loss"
        )
        assert loss_entry["step"] == 42


# ----------------------------------------------------------------------
# on_evaluate — eval-side pass-through
# ----------------------------------------------------------------------


class TestOnEvaluate:
    def test_eval_keys_renamed(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_evaluate(
            _FakeTrainerArgs(),
            _FakeState(global_step=100),
            _FakeControl(),
            metrics={"eval_loss": 0.4, "eval_accuracy": 0.95},
        )
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert f"{EVAL_METRIC_PREFIX}/loss" in names
        assert f"{EVAL_METRIC_PREFIX}/accuracy" in names

    def test_no_metrics_is_noop(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        cb.on_evaluate(
            _FakeTrainerArgs(), _FakeState(), _FakeControl(), metrics=None
        )
        assert fake_aim_run[-1].tracked == []


# ----------------------------------------------------------------------
# on_train_end
# ----------------------------------------------------------------------


class TestOnTrainEnd:
    def test_closes_with_completed(self, fake_aim_run):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_begin(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        run = cb._run
        cb.on_train_end(_FakeTrainerArgs(), _FakeState(), _FakeControl())
        assert run.tags["astrolabe.status"] == "completed"
        assert run.closed is True
        assert cb._run is None

    def test_no_run_is_noop(self):
        cb = AstrolabeHFTrainerCallback()
        cb.on_train_end(
            _FakeTrainerArgs(), _FakeState(), _FakeControl()
        )  # must not raise
