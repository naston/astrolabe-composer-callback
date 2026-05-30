"""Tests for ``astrolabe_callbacks._core``.

Order: unhappy + edge first, happy paths last (per project test guide).
The contract is enforced here once; framework callback tests trust
that env-var precedence, AIM_RUN_TAGS parsing, etc. work correctly.
"""

from __future__ import annotations

import time

import pytest

from astrolabe_callbacks._core import (
    DEFAULT_AIM_URL,
    EVAL_METRIC_PREFIX,
    RunConfig,
    WallTimeTracker,
    close_run,
    is_strict,
    open_aim_run,
    parse_aim_run_tags,
    resolve_run_config,
    track_safely,
)
from tests.conftest import FakeAimRun, make_run_config


# ----------------------------------------------------------------------
# parse_aim_run_tags — env-var → dict converter
# ----------------------------------------------------------------------


class TestParseAimRunTagsEdgeCases:
    """Researcher-pasted env vars are messy; parser tolerates instead of raising."""

    def test_none_returns_empty(self):
        assert parse_aim_run_tags(None) == {}

    def test_empty_string_returns_empty(self):
        assert parse_aim_run_tags("") == {}

    def test_whitespace_only_returns_empty(self):
        assert parse_aim_run_tags("   ") == {}

    def test_only_commas_returns_empty(self):
        assert parse_aim_run_tags(",,,") == {}

    def test_entry_without_equals_dropped(self):
        # "bare_token" silently dropped — masking it as a key with empty
        # value would hide researcher typos like "no_value_here".
        assert parse_aim_run_tags("good=1,bad,also=2") == {"good": "1", "also": "2"}

    def test_empty_key_dropped(self):
        # "=value" skipped rather than producing {"": "value"}, which
        # would produce illegal Aim tag names.
        assert parse_aim_run_tags("=orphan,real=1") == {"real": "1"}

    def test_duplicate_keys_last_wins(self):
        assert parse_aim_run_tags("k=1,k=2,k=3") == {"k": "3"}

    def test_value_can_be_empty(self):
        # "k=" is a valid tag with empty value; the user explicitly
        # said key=. Not the same as "k" with no =.
        assert parse_aim_run_tags("k=") == {"k": ""}

    def test_value_with_equals_sign(self):
        # str.partition takes the first =; remaining = stays in value.
        # This matters for serialized values that contain "=" naturally
        # (e.g. base64, query strings).
        assert parse_aim_run_tags("config=key=value") == {"config": "key=value"}

    def test_trailing_comma_tolerated(self):
        assert parse_aim_run_tags("k=v,") == {"k": "v"}

    def test_leading_comma_tolerated(self):
        assert parse_aim_run_tags(",k=v") == {"k": "v"}

    def test_unicode_keys_and_values(self):
        # No ASCII restrictions — Aim accepts unicode tag names.
        assert parse_aim_run_tags("名前=値,team=データ") == {
            "名前": "値",
            "team": "データ",
        }


class TestParseAimRunTagsHappyPath:
    def test_single_tag(self):
        assert parse_aim_run_tags("k=v") == {"k": "v"}

    def test_multiple_tags(self):
        assert parse_aim_run_tags("a=1,b=2,c=3") == {"a": "1", "b": "2", "c": "3"}

    def test_realistic_astrolabe_tags(self):
        # The actual shape astrolabe writes into AIM_RUN_TAGS at submit.
        raw = "astrolabe.experiment=foo,astrolabe.version=v3,astrolabe.submit_id=abc-123"
        out = parse_aim_run_tags(raw)
        assert out["astrolabe.experiment"] == "foo"
        assert out["astrolabe.version"] == "v3"
        assert out["astrolabe.submit_id"] == "abc-123"


# ----------------------------------------------------------------------
# resolve_run_config — env-var precedence
# ----------------------------------------------------------------------


class TestResolveRunConfigPrecedence:
    """Astrolabe env vars beat constructor args. Astrolabe is the
    orchestrator; its identity is authoritative when it's the one
    driving the run. Constructor args are the standalone fallback."""

    def test_env_experiment_name_wins_over_arg(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "from-env")
        cfg = resolve_run_config(experiment_name="from-arg")
        assert cfg.experiment_name == "from-env"

    def test_arg_used_when_env_unset(self, monkeypatch):
        # autouse fixture already cleared env; explicit guard for clarity.
        monkeypatch.delenv("ASTROLABE_EXPERIMENT_NAME", raising=False)
        cfg = resolve_run_config(experiment_name="from-arg")
        assert cfg.experiment_name == "from-arg"

    def test_empty_env_falls_through_to_arg(self, monkeypatch):
        # Empty string env is treated as "not set" — `or None` chains
        # the precedence forward. Without this, accidentally setting
        # ASTROLABE_EXPERIMENT_NAME="" in a shell would silently break
        # the constructor fallback.
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "")
        cfg = resolve_run_config(experiment_name="from-arg")
        assert cfg.experiment_name == "from-arg"

    def test_env_aim_url_wins(self, monkeypatch):
        monkeypatch.setenv("ASTROLABE_AIM_URL", "aim://from-env:9999")
        cfg = resolve_run_config(aim_url="aim://from-arg:1111")
        assert cfg.aim_url == "aim://from-env:9999"

    def test_aim_url_default_when_neither_set(self):
        cfg = resolve_run_config()
        assert cfg.aim_url == DEFAULT_AIM_URL
        assert cfg.aim_url == "aim://localhost:43800"

    def test_env_tags_win_over_arg(self, monkeypatch):
        monkeypatch.setenv("AIM_RUN_TAGS", "from=env")
        cfg = resolve_run_config(tags={"from": "arg"})
        assert cfg.tags == {"from": "env"}

    def test_arg_tags_used_when_env_empty(self, monkeypatch):
        monkeypatch.setenv("AIM_RUN_TAGS", "")
        cfg = resolve_run_config(tags={"from": "arg"})
        assert cfg.tags == {"from": "arg"}

    def test_no_tags_anywhere_yields_empty_dict(self):
        cfg = resolve_run_config()
        assert cfg.tags == {}

    def test_tags_arg_copied_not_referenced(self):
        # If the caller mutates the dict they passed in, our cfg
        # shouldn't change. This protects RunConfig's frozen contract.
        original = {"k": "v"}
        cfg = resolve_run_config(tags=original)
        original["k"] = "changed"
        assert cfg.tags == {"k": "v"}


class TestRunConfigImmutability:
    """RunConfig is frozen — direct field writes raise."""

    def test_cannot_assign_to_frozen_field(self):
        cfg = make_run_config()
        with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
            cfg.experiment_name = "new"  # type: ignore[misc]


# ----------------------------------------------------------------------
# is_strict — env-driven escape hatch
# ----------------------------------------------------------------------


class TestIsStrict:
    def test_default_is_false(self):
        assert is_strict() is False

    def test_unset_explicit(self, monkeypatch):
        monkeypatch.delenv("ASTROLABE_CALLBACK_STRICT", raising=False)
        assert is_strict() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "TRUE", "Yes", "True"])
    def test_truthy_values(self, monkeypatch, value):
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", value)
        assert is_strict() is True

    @pytest.mark.parametrize(
        "value", ["", "0", "false", "no", "off", "False", "anything-else"]
    )
    def test_falsy_values(self, monkeypatch, value):
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", value)
        assert is_strict() is False


# ----------------------------------------------------------------------
# open_aim_run — connection failure modes
# ----------------------------------------------------------------------


class TestOpenAimRunFailureModes:
    """When Aim can't connect, default is graceful no-op + WARNING.
    Strict mode raises so CI can fail fast on misconfig."""

    def test_returns_none_when_aim_not_installed(self, monkeypatch):
        # Force `from aim import Run` to fail.
        import sys

        monkeypatch.setitem(sys.modules, "aim", None)
        cfg = make_run_config()
        assert open_aim_run(cfg) is None

    def test_strict_raises_when_aim_not_installed(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "aim", None)
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        cfg = make_run_config()
        with pytest.raises(RuntimeError, match="aim not installed"):
            open_aim_run(cfg)

    def test_returns_none_on_connection_failure(self, monkeypatch):
        # Aim is importable but Run() raises — simulates server unreachable.
        class BrokenRun:
            def __init__(self, **kwargs):
                raise ConnectionError("server down")

        monkeypatch.setattr("aim.Run", BrokenRun)
        cfg = make_run_config()
        assert open_aim_run(cfg) is None

    def test_strict_raises_on_connection_failure(self, monkeypatch):
        class BrokenRun:
            def __init__(self, **kwargs):
                raise ConnectionError("server down")

        monkeypatch.setattr("aim.Run", BrokenRun)
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        cfg = make_run_config()
        with pytest.raises(RuntimeError, match="Aim connection"):
            open_aim_run(cfg)

    def test_tag_write_failure_does_not_break_open(self, monkeypatch):
        """A failing tag write logs DEBUG and continues — other tags still apply."""

        class FlakeyRun(FakeAimRun):
            def __setitem__(self, key, value):
                if key == "boom":
                    raise RuntimeError("write rejected")
                super().__setitem__(key, value)

        monkeypatch.setattr("aim.Run", FlakeyRun)
        cfg = make_run_config(tags={"good": "1", "boom": "2", "also_good": "3"})
        run = open_aim_run(cfg)
        assert run is not None
        assert run.tags["good"] == "1"
        assert run.tags["also_good"] == "3"
        assert "boom" not in run.tags

    def test_run_name_set_when_provided(self, fake_aim_run):
        cfg = make_run_config()
        run = open_aim_run(cfg, run_name="bert-tiny")
        assert run is not None
        assert run.name == "bert-tiny"

    def test_run_name_unset_when_none(self, fake_aim_run):
        cfg = make_run_config()
        run = open_aim_run(cfg, run_name=None)
        assert run is not None
        assert run.name is None

    def test_run_name_unset_when_empty_string(self, fake_aim_run):
        cfg = make_run_config()
        run = open_aim_run(cfg, run_name="")
        assert run is not None
        assert run.name is None


class TestOpenAimRunHappyPath:
    def test_opens_run_with_aim_url_and_experiment(self, fake_aim_run):
        cfg = make_run_config(
            experiment_name="my-exp", aim_url="aim://test-host:8080"
        )
        run = open_aim_run(cfg)
        assert run is not None
        assert run.kwargs["repo"] == "aim://test-host:8080"
        assert run.kwargs["experiment"] == "my-exp"

    def test_applies_all_tags(self, fake_aim_run):
        cfg = make_run_config(
            tags={"astrolabe.version": "v3", "astrolabe.submit_id": "abc"}
        )
        run = open_aim_run(cfg)
        assert run is not None
        assert run.tags == {"astrolabe.version": "v3", "astrolabe.submit_id": "abc"}


# ----------------------------------------------------------------------
# close_run — final-status writing + cleanup
# ----------------------------------------------------------------------


class TestCloseRun:
    def test_none_run_is_noop(self):
        # Defends against `close_run(self._run)` being called when
        # open_aim_run returned None.
        close_run(None)  # must not raise

    def test_writes_completed_status_by_default(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        close_run(run)
        assert run.tags["astrolabe.status"] == "completed"
        assert run.closed is True

    def test_writes_failed_status_when_specified(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        close_run(run, status="failed")
        assert run.tags["astrolabe.status"] == "failed"
        assert run.closed is True

    def test_custom_status_passes_through(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        close_run(run, status="interrupted")
        assert run.tags["astrolabe.status"] == "interrupted"

    def test_close_failure_is_silent(self, monkeypatch):
        """A failing run.close() must not raise — by the time we're
        closing, data has been streamed and a close failure is cosmetic."""

        class CloseFails(FakeAimRun):
            def close(self):
                raise RuntimeError("aim server died")

        monkeypatch.setattr("aim.Run", CloseFails)
        run = open_aim_run(make_run_config())
        close_run(run)  # must not raise

    def test_status_write_failure_does_not_block_close(self, monkeypatch):
        class StatusWriteFails(FakeAimRun):
            def __setitem__(self, key, value):
                if key == "astrolabe.status":
                    raise RuntimeError("write rejected")
                super().__setitem__(key, value)

        monkeypatch.setattr("aim.Run", StatusWriteFails)
        run = open_aim_run(make_run_config())
        close_run(run)
        # Status didn't write but the run still closed cleanly.
        assert run.closed is True


# ----------------------------------------------------------------------
# track_safely — graceful + strict failure modes
# ----------------------------------------------------------------------


class TestTrackSafelyFailureModes:
    def test_none_run_is_noop(self):
        track_safely(None, name="x", value=1.0)  # must not raise

    def test_track_failure_swallowed_by_default(self, monkeypatch):
        class TrackFails(FakeAimRun):
            def track(self, *a, **kw):
                raise RuntimeError("aim disconnected")

        monkeypatch.setattr("aim.Run", TrackFails)
        run = open_aim_run(make_run_config())
        # Must not raise.
        track_safely(run, name="train/loss", value=0.5, step=1)

    def test_track_failure_raises_in_strict_mode(self, monkeypatch):
        class TrackFails(FakeAimRun):
            def track(self, *a, **kw):
                raise RuntimeError("aim disconnected")

        monkeypatch.setattr("aim.Run", TrackFails)
        run = open_aim_run(make_run_config())
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        with pytest.raises(RuntimeError, match="aim disconnected"):
            track_safely(run, name="train/loss", value=0.5, step=1)

    def test_failures_rate_limited_per_metric_name(self, monkeypatch, caplog):
        """Same metric failing every batch shouldn't spam logs.

        Rate-limit state moved from a flat ``run._astrolabe_track_failures``
        attribute to ``run._astrolabe_buffer._warned`` when the buffer
        layer landed (v0.2.x). Functionally the same — first failure
        per metric name logs WARNING, subsequent failures silenced.
        """

        class TrackAlwaysFails(FakeAimRun):
            def track(self, *a, **kw):
                raise RuntimeError("nope")

        monkeypatch.setattr("aim.Run", TrackAlwaysFails)
        run = open_aim_run(make_run_config())

        for i in range(100):
            track_safely(run, name="train/loss", value=0.5, step=i)

        assert "train/loss" in run._astrolabe_buffer._warned

    def test_different_names_each_log_once(self, monkeypatch):
        class TrackAlwaysFails(FakeAimRun):
            def track(self, *a, **kw):
                raise RuntimeError("nope")

        monkeypatch.setattr("aim.Run", TrackAlwaysFails)
        run = open_aim_run(make_run_config())

        track_safely(run, name="a", value=1.0)
        track_safely(run, name="b", value=1.0)
        track_safely(run, name="c", value=1.0)

        assert run._astrolabe_buffer._warned == {"a", "b", "c"}


class TestTrackSafelyHappyPath:
    def test_basic_track_call(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        track_safely(run, name="train/loss", value=0.5, step=10)
        assert run.tracked == [
            {"name": "train/loss", "value": 0.5, "step": 10, "context": {}}
        ]

    def test_track_with_context(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        track_safely(
            run,
            name="metric",
            value=1.0,
            step=1,
            context={"subset": "val"},
        )
        assert run.tracked[0]["context"] == {"subset": "val"}

    def test_track_with_no_step(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        track_safely(run, name="x", value=1.0)
        assert run.tracked[0]["step"] is None


# ----------------------------------------------------------------------
# WallTimeTracker — clock arithmetic
# ----------------------------------------------------------------------


class TestWallTimeTrackerEdgeCases:
    def test_elapsed_zero_before_mark_first_batch(self):
        t = WallTimeTracker()
        assert t.elapsed() == 0.0

    def test_mark_first_batch_idempotent(self):
        """Second + subsequent calls must not move the anchor."""
        t = WallTimeTracker()
        t.mark_first_batch()
        first_anchor = t._start_time
        time.sleep(0.01)
        t.mark_first_batch()
        assert t._start_time == first_anchor

    def test_resume_without_pause_is_noop(self):
        # Catches a bug where calling resume before pause would
        # accumulate the entire wall-clock time as eval time
        # (subtracting now() - 0 = a huge number).
        t = WallTimeTracker()
        t.mark_first_batch()
        time.sleep(0.02)
        t.resume()
        elapsed = t.elapsed()
        assert elapsed > 0
        assert elapsed < 1.0  # didn't go negative or balloon

    def test_pause_without_subsequent_resume_loses_time(self):
        # Documenting actual behavior: an unmatched pause leaks.
        # Real usage always pairs them inside eval_start/eval_end hooks.
        t = WallTimeTracker()
        t.mark_first_batch()
        t.pause_for_eval()
        # Without resume, the eval pause is open-ended.
        # _total_eval_time stays 0 until resume is called.
        assert t._total_eval_time == 0.0

    def test_multiple_pause_resume_cycles(self):
        t = WallTimeTracker()
        t.mark_first_batch()
        time.sleep(0.01)
        # Eval cycle 1
        t.pause_for_eval()
        time.sleep(0.02)
        t.resume()
        eval_time_after_first = t._total_eval_time
        assert eval_time_after_first > 0
        # Eval cycle 2
        t.pause_for_eval()
        time.sleep(0.02)
        t.resume()
        # Second cycle accumulates onto the first.
        assert t._total_eval_time > eval_time_after_first


class TestWallTimeTrackerHappyPath:
    def test_elapsed_grows_after_mark(self):
        t = WallTimeTracker()
        t.mark_first_batch()
        time.sleep(0.05)
        elapsed = t.elapsed()
        assert elapsed >= 0.04  # tolerate scheduler jitter
        assert elapsed < 0.5

    def test_eval_pause_excluded_from_elapsed(self):
        """Eval time is subtracted; training-only elapsed stays small."""
        t = WallTimeTracker()
        t.mark_first_batch()
        t.pause_for_eval()
        time.sleep(0.1)  # spend 100ms "in eval"
        t.resume()
        # Now elapsed should be ~0 since we paused immediately and
        # the only real training time was setup.
        assert t.elapsed() < 0.05  # < 50ms of training time


# ----------------------------------------------------------------------
# EVAL_METRIC_PREFIX — single-source-of-truth for v1.0.0 flip
# ----------------------------------------------------------------------


class TestEvalMetricPrefix:
    def test_default_value_is_val(self):
        # v1.0.0 flipped this from "eval" to "val" alongside astrolabe
        # v1.7's eval-runs schema. ``val/`` is during-training
        # validation (Training tab); ``eval/`` is the prefix astrolabe
        # uses for post-training benchmark suites tracked on separate
        # eval Aim runs (Eval tab). Flipping back should be intentional
        # and break this test.
        assert EVAL_METRIC_PREFIX == "val"
