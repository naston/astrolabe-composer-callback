"""Tests for ``AstrolabeComposerLogger``.

Composer-specific contract: this is a ``LoggerDestination``, so it
receives every metric the user logs via Composer's ``Logger`` (via
``log_metrics``) plus standard lifecycle hooks. The pass-through
contract is enforced here.
"""

from __future__ import annotations

import pytest

from astrolabe_callbacks.composer import (
    AstrolabeComposerLogger,
    parse_aim_run_tags,
    _normalize_composer_metric_name,
    _to_scalar,
)
from tests.conftest import FakeAimRun


# ----------------------------------------------------------------------
# parse_aim_run_tags re-export
# ----------------------------------------------------------------------


class TestParseAimRunTagsReExport:
    """Symbol still importable via ``composer`` module for back-compat."""

    def test_basic_parse(self):
        assert parse_aim_run_tags("k=v") == {"k": "v"}

    def test_re_export_matches_core(self):
        from astrolabe_callbacks._core import parse_aim_run_tags as core_parse
        assert parse_aim_run_tags is core_parse


# ----------------------------------------------------------------------
# Constructor — env-var precedence
# ----------------------------------------------------------------------


class TestConstructorPrecedence:
    """Astrolabe env vars beat constructor args (full contract tested
    in test_core.py; this verifies the wiring through to the callback)."""

    def test_env_experiment_name_wins(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "from-env")
        cb = AstrolabeComposerLogger(experiment_name="hardcoded-in-yaml")
        assert cb._cfg.experiment_name == "from-env"

    def test_explicit_arg_used_when_env_unset(self, monkeypatch):
        cb = AstrolabeComposerLogger(experiment_name="standalone-name")
        assert cb._cfg.experiment_name == "standalone-name"

    def test_env_aim_url_wins(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_AIM_URL", "aim://from-env:9999")
        cb = AstrolabeComposerLogger(aim_url="aim://from-arg:1111")
        assert cb._cfg.aim_url == "aim://from-env:9999"

    def test_aim_url_default_when_neither_set(self):
        cb = AstrolabeComposerLogger()
        assert cb._cfg.aim_url == "aim://localhost:43800"

    def test_env_tags_win(self, monkeypatch):
        monkeypatch.setenv("AIM_RUN_TAGS", "from=env")
        cb = AstrolabeComposerLogger(tags={"from": "arg"})
        assert cb._cfg.tags == {"from": "env"}


# ----------------------------------------------------------------------
# Composer name normalizer
# ----------------------------------------------------------------------


class TestNameNormalizerEdgeCases:
    """Cosmetic renames for Composer's automatic emissions; user-named
    metrics must always pass through unchanged. Edge cases first to
    catch off-by-one slicing or partial-match bugs."""

    def test_user_metric_passes_through(self):
        # Critical: any name we don't recognize must pass through
        # unchanged. The user's metric names are not ours to rewrite.
        assert _normalize_composer_metric_name("my_metric") == "my_metric"

    def test_partial_match_does_not_trigger_rename(self):
        # "metrics/train_other/x" must NOT match "metrics/train/" prefix
        # because the prefix-strip uses the full "metrics/train/" string.
        assert (
            _normalize_composer_metric_name("metrics/train_other/x")
            == "metrics/train_other/x"
        )

    def test_partial_eval_match_does_not_trigger_rename(self):
        assert (
            _normalize_composer_metric_name("metrics/evaluation/x")
            == "metrics/evaluation/x"
        )

    def test_loss_train_total_substring_does_not_trigger(self):
        # Only the exact "loss/train/total" string maps; a metric
        # called "my/loss/train/total" must not be rewritten.
        assert (
            _normalize_composer_metric_name("my/loss/train/total")
            == "my/loss/train/total"
        )

    def test_eval_metric_with_slash_in_suffix(self):
        # MaskedLanguagePerplexity with no slashes is the common case;
        # a metric like "metrics/eval/glue/mnli" must keep its
        # internal slashes after the prefix strip.
        assert (
            _normalize_composer_metric_name("metrics/eval/glue/mnli")
            == "eval/glue/mnli"
        )

    def test_train_metric_with_slash_in_suffix(self):
        assert (
            _normalize_composer_metric_name("metrics/train/glue/mnli")
            == "train/glue/mnli"
        )

    def test_empty_string(self):
        assert _normalize_composer_metric_name("") == ""

    def test_just_prefix_no_suffix(self):
        # "metrics/eval/" with empty suffix — current behavior preserves
        # the structure and produces "eval/" as result. Aim accepts
        # this; document the actual behavior.
        assert _normalize_composer_metric_name("metrics/eval/") == "eval/"


class TestNameNormalizerHappyPath:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("loss/train/total", "train/loss"),
            ("metrics/train/Accuracy", "train/Accuracy"),
            (
                "metrics/eval/MaskedLanguagePerplexity",
                "eval/MaskedLanguagePerplexity",
            ),
            ("metrics/eval/CrossEntropy", "eval/CrossEntropy"),
            ("lr-DecoupledAdamW", "lr-DecoupledAdamW"),
            (
                "throughput/samples_per_sec",
                "throughput/samples_per_sec",
            ),
        ],
    )
    def test_canonical_renames(self, raw, expected):
        assert _normalize_composer_metric_name(raw) == expected


# ----------------------------------------------------------------------
# _to_scalar — non-numeric inputs must skip cleanly
# ----------------------------------------------------------------------


class TestToScalarEdgeCases:
    def test_none_returns_none(self):
        assert _to_scalar(None) is None

    def test_str_returns_none(self):
        assert _to_scalar("not a number") is None

    def test_dict_returns_none(self):
        assert _to_scalar({"k": 1}) is None

    def test_list_returns_none(self):
        assert _to_scalar([1, 2, 3]) is None

    def test_tensor_with_failing_item(self):
        class BrokenTensor:
            def item(self):
                raise RuntimeError("not on cpu")

        assert _to_scalar(BrokenTensor()) is None

    def test_int_returns_float(self):
        result = _to_scalar(42)
        assert result == 42.0
        assert isinstance(result, float)

    def test_float_passthrough(self):
        assert _to_scalar(0.5) == 0.5

    def test_real_tensor(self):
        import torch
        assert _to_scalar(torch.tensor(0.5)) == 0.5


# ----------------------------------------------------------------------
# init() — opens Aim run and applies tags
# ----------------------------------------------------------------------


class TestInit:
    def test_writes_tags_to_run(self, fake_aim_run):
        cb = AstrolabeComposerLogger(
            aim_url="aim://test:1",
            experiment_name="exp",
            tags={"astrolabe.version": "v3", "astrolabe.submit_id": "abc"},
        )
        cb.init(state=None, logger_obj=None)
        assert fake_aim_run[-1].tags["astrolabe.version"] == "v3"
        assert fake_aim_run[-1].tags["astrolabe.submit_id"] == "abc"

    def test_handles_missing_aim_module(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "aim", None)
        cb = AstrolabeComposerLogger()
        # Must not raise.
        cb.init(state=None, logger_obj=None)
        assert cb._run is None

    def test_strict_mode_raises_when_aim_missing(self, monkeypatch):
        import sys
        monkeypatch.setitem(sys.modules, "aim", None)
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        cb = AstrolabeComposerLogger()
        with pytest.raises(RuntimeError, match="aim not installed"):
            cb.init(state=None, logger_obj=None)

    def test_double_init_does_not_re_open(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        first_run = cb._run
        cb.init(state=None, logger_obj=None)
        # Second init should not replace the existing run.
        assert cb._run is first_run
        # Only one FakeRun instance was constructed.
        assert len(fake_aim_run) == 1

    def test_run_name_pulled_from_state(self, fake_aim_run):
        class FakeState:
            run_name = "bert-tiny"

        cb = AstrolabeComposerLogger()
        cb.init(state=FakeState(), logger_obj=None)
        assert fake_aim_run[-1].name == "bert-tiny"


# ----------------------------------------------------------------------
# log_metrics — pass-through with name normalization
# ----------------------------------------------------------------------


class TestLogMetricsEdgeCases:
    """``log_metrics`` is the primary surface — every user-logged metric
    flows through. Edge cases first."""

    def test_empty_dict_is_noop(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({})
        assert fake_aim_run[-1].tracked == []

    def test_none_is_noop(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics(None)  # type: ignore[arg-type]
        assert fake_aim_run[-1].tracked == []

    def test_no_run_is_noop(self):
        cb = AstrolabeComposerLogger()
        # init never called → _run stays None
        cb.log_metrics({"x": 1.0})  # must not raise

    def test_non_numeric_value_skipped(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({"good": 1.0, "bad_str": "nope", "bad_dict": {}})
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "good" in names
        assert "bad_str" not in names
        assert "bad_dict" not in names

    def test_rank_nonzero_is_noop(self, monkeypatch, fake_aim_run):
        monkeypatch.setenv("RANK", "2")
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)  # no-op on non-rank-zero
        cb.log_metrics({"x": 1.0})
        assert fake_aim_run == []  # no run ever opened


class TestLogMetricsHappyPath:
    def test_user_metric_passes_through_unchanged(self, fake_aim_run):
        # Critical for user trust: a custom metric called
        # "MaskedLanguagePerplexity" lands in Aim under exactly that
        # name, not buried under our prefix.
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({"MaskedLanguagePerplexity": 4.2}, step=10)
        tracked = fake_aim_run[-1].tracked
        assert any(
            t["name"] == "MaskedLanguagePerplexity" and t["value"] == 4.2 and t["step"] == 10
            for t in tracked
        )

    def test_composer_train_loss_renamed(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({"loss/train/total": 0.5}, step=1)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "train/loss" in names
        assert "loss/train/total" not in names

    def test_composer_eval_metric_renamed(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({"metrics/eval/MaskedLanguagePerplexity": 4.2}, step=100)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "eval/MaskedLanguagePerplexity" in names

    def test_tensors_unwrapped(self, fake_aim_run):
        import torch
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_metrics({"loss": torch.tensor(0.7)}, step=5)
        tracked = fake_aim_run[-1].tracked
        assert tracked[0]["value"] == pytest.approx(0.7, abs=1e-5)


# ----------------------------------------------------------------------
# log_hyperparameters
# ----------------------------------------------------------------------


class TestLogHyperparameters:
    def test_writes_to_run(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_hyperparameters({"lr": 1e-4, "batch_size": 32})
        assert fake_aim_run[-1].tags["hparams"] == {"lr": 1e-4, "batch_size": 32}

    def test_empty_is_noop(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.log_hyperparameters({})
        assert "hparams" not in fake_aim_run[-1].tags

    def test_no_run_is_noop(self):
        cb = AstrolabeComposerLogger()
        cb.log_hyperparameters({"lr": 1e-4})  # must not raise


# ----------------------------------------------------------------------
# Lifecycle hooks
# ----------------------------------------------------------------------


class TestLifecycle:
    def test_batch_end_logs_only_wall_time(self, fake_aim_run):
        # The new contract: Composer's automatic train metrics flow
        # through log_metrics, NOT batch_end. batch_end synthesizes
        # only wall_time.
        class FakeTimestamp:
            batch = 5

        class FakeState:
            timestamp = FakeTimestamp()
            loss = 999  # would have been logged in old design; now ignored

        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.batch_end(state=FakeState(), logger_obj=None)
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert "wall_time" in names
        # state.loss is NOT auto-extracted anymore — only flows via
        # log_metrics from Composer's Logger.
        assert "train/loss" not in names

    def test_eval_start_pauses_wall_time(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        # First need to anchor wall-time via a batch_end
        class FakeState:
            class timestamp:
                batch = 1
            loss = None

        cb.batch_end(state=FakeState(), logger_obj=None)
        cb.eval_start(state=None, logger_obj=None)
        # _eval_start should now be set on the wall-time tracker
        assert cb._wall_time._eval_start > 0

    def test_eval_end_resumes_wall_time(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.eval_start(state=None, logger_obj=None)
        cb.eval_end(state=None, logger_obj=None)
        # After resume, _eval_start is back to 0
        assert cb._wall_time._eval_start == 0

    def test_fit_end_closes_with_completed_status(self, fake_aim_run):
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        run = cb._run
        cb.fit_end(state=None, logger_obj=None)
        assert run.tags["astrolabe.status"] == "completed"
        assert run.closed is True
        assert cb._run is None

    def test_post_close_marks_failed_when_fit_end_skipped(self, fake_aim_run):
        # If training crashes before fit_end fires, post_close still
        # gets called by Composer's Trainer cleanup. This is our last
        # chance to mark the run as failed.
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        run = cb._run
        cb.post_close()
        assert run.tags["astrolabe.status"] == "failed"
        assert run.closed is True

    def test_post_close_after_fit_end_is_noop(self, fake_aim_run):
        # Idempotent: clean exit calls fit_end → close → _run=None.
        # Composer then calls post_close; it should no-op.
        cb = AstrolabeComposerLogger()
        cb.init(state=None, logger_obj=None)
        cb.fit_end(state=None, logger_obj=None)
        # post_close after fit_end must not raise or re-close.
        cb.post_close()  # _run is None at this point; no error
