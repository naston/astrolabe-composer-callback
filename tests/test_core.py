"""Tests for ``astrolabe_callbacks._core``.

Order: unhappy + edge first, happy paths last (per project test guide).
The contract is enforced here once; framework callback tests trust
that env-var precedence, AIM_RUN_TAGS parsing, etc. work correctly.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from astrolabe_callbacks._core import (
    DEFAULT_AIM_URL,
    EVAL_METRIC_PREFIX,
    RunConfig,
    SchemaPhaseState,
    WallTimeTracker,
    close_run,
    is_strict,
    maybe_finalize_schema,
    observe_name,
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


# ----------------------------------------------------------------------
# Schema-phase machinery — observe_name + maybe_finalize_schema
# ----------------------------------------------------------------------
#
# Schema-phase finalize is what makes the producer's metric names visible
# to a separate-process reader on the NUC. The mechanism: drain the
# write buffer, close the Aim Run (forces RocksDB memtable flush to
# stable SST files), reopen the Run with force_resume=True so the
# next sync cycle's rsync can see the schema and the dashboard can
# enumerate metrics.
#
# These tests verify the state machine and graceful-degradation paths.
# They use FakeAimRun so we exercise the orchestration without standing
# up a real Aim transport server (which would defeat the unit-test
# purpose). A live-aim integration test in test_composer covers the
# end-to-end behavior.


class _HashableFakeAimRun(FakeAimRun):
    """FakeAimRun with a settable ``hash`` attribute.

    Real ``aim.Run`` instances expose ``.hash``; the schema-finalize
    code reads it to call ``Run(run_hash=...)`` on reopen. FakeAimRun
    doesn't set hash by default, so tests that exercise the finalize
    path use this subclass and set ``.hash`` explicitly.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hash = kwargs.get("run_hash", None) or "test-hash-abcdef"


@pytest.fixture
def fake_aim_run_with_hash(monkeypatch):
    """Like ``fake_aim_run`` but instances carry a ``.hash`` attribute."""
    instances: list[_HashableFakeAimRun] = []

    class _Recording(_HashableFakeAimRun):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    monkeypatch.setattr("aim.Run", _Recording)
    return instances


class TestObserveNameEdgeCases:
    """``observe_name`` is a thin set.add wrapper, but the trivial
    contract still has cases worth pinning so a future refactor that
    e.g. adds normalization or filtering can't silently change behavior."""

    def test_observe_empty_string_recorded(self):
        # Empty string is a valid set element; if a metric name is empty
        # that's a bug elsewhere — observe_name records what it's given.
        state = SchemaPhaseState()
        observe_name(state, "")
        assert state.observed_names == {""}

    def test_observe_idempotent_same_name(self):
        state = SchemaPhaseState()
        observe_name(state, "train/loss")
        observe_name(state, "train/loss")
        observe_name(state, "train/loss")
        assert state.observed_names == {"train/loss"}

    def test_observe_distinct_names_accumulate(self):
        state = SchemaPhaseState()
        for name in ("train/loss", "lr", "train/loss", "throughput"):
            observe_name(state, name)
        assert state.observed_names == {"train/loss", "lr", "throughput"}

    def test_observe_does_not_touch_registered_names(self):
        # observe should ONLY grow observed_names; registered_names is
        # updated only by maybe_finalize_schema. Test prevents a future
        # refactor that conflates them.
        state = SchemaPhaseState()
        observe_name(state, "train/loss")
        assert state.registered_names == set()


class TestMaybeFinalizeSchemaNoOps:
    """Cases where maybe_finalize_schema must NOT call run.close() or
    open a new Run — graceful no-ops the framework callback can call at
    every boundary hook without worrying about wasted work."""

    def test_none_run_returns_none(self):
        # Callback degraded to no-op mode (Aim unreachable, strict-off).
        # maybe_finalize_schema must not crash on None.
        state = SchemaPhaseState()
        observe_name(state, "train/loss")
        result = maybe_finalize_schema(None, state, cfg=make_run_config())
        assert result is None

    def test_no_observed_names_returns_same_run(self, fake_aim_run_with_hash):
        run = _HashableFakeAimRun()
        state = SchemaPhaseState()
        result = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert result is run
        assert not run.closed  # we did NOT close it
        assert state.finalize_count == 0

    def test_all_observed_already_registered_returns_same_run(self, fake_aim_run_with_hash):
        run = _HashableFakeAimRun()
        state = SchemaPhaseState()
        state.observed_names = {"train/loss", "lr"}
        state.registered_names = {"train/loss", "lr"}  # already up to date
        result = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert result is run
        assert not run.closed
        assert state.finalize_count == 0

    def test_run_without_hash_returns_same_run(self, fake_aim_run_with_hash):
        # Defensive: if run has no .hash attribute (couldn't happen
        # against real aim.Run, but FakeAimRun could be misconstructed),
        # we don't crash — we just decline to finalize.
        run = _HashableFakeAimRun()
        run.hash = None
        state = SchemaPhaseState()
        observe_name(state, "train/loss")
        result = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert result is run
        assert not run.closed


class TestMaybeFinalizeSchemaHappyPath:
    """Cases where maybe_finalize_schema does the close + reopen cycle."""

    def test_observe_then_finalize_registers_names(self, fake_aim_run_with_hash):
        run = _HashableFakeAimRun()
        run.hash = "run-123456789"
        fake_aim_run_with_hash.append(run)  # account for the original
        state = SchemaPhaseState()
        observe_name(state, "train/loss")
        observe_name(state, "lr")

        new_run = maybe_finalize_schema(run, state, cfg=make_run_config())

        # State after finalize
        assert state.registered_names == {"train/loss", "lr"}
        assert state.finalize_count == 1
        # observed_names continues to grow over the run's lifetime; not reset.
        assert state.observed_names == {"train/loss", "lr"}
        # Original run was closed
        assert run.closed
        # New Run object was created (different instance from the old one)
        assert new_run is not run

    def test_finalize_passes_run_hash_to_reopen(self, fake_aim_run_with_hash):
        run = _HashableFakeAimRun()
        run.hash = "specific-hash-xyz"
        state = SchemaPhaseState()
        observe_name(state, "metric_a")

        cfg = make_run_config(aim_url="aim://test:9999")
        new_run = maybe_finalize_schema(run, state, cfg=cfg)

        # The newly-constructed FakeAimRun (via the fake_aim_run_with_hash
        # patch) should have received run_hash, repo, and force_resume.
        assert new_run is not run
        assert new_run.kwargs.get("run_hash") == "specific-hash-xyz"
        assert new_run.kwargs.get("repo") == "aim://test:9999"
        assert new_run.kwargs.get("force_resume") is True

    def test_second_finalize_with_new_names_works(self, fake_aim_run_with_hash):
        # Simulates: batch_end finalizes for training metrics, then
        # eval_end fires and a new eval metric appears, triggering
        # another finalize.
        run = _HashableFakeAimRun()
        run.hash = "hash-1"
        state = SchemaPhaseState()

        observe_name(state, "train/loss")
        run_after_1 = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert state.finalize_count == 1
        assert state.registered_names == {"train/loss"}

        # New eval metric appears
        observe_name(state, "val/loss")
        run_after_2 = maybe_finalize_schema(run_after_1, state, cfg=make_run_config())
        assert state.finalize_count == 2
        assert state.registered_names == {"train/loss", "val/loss"}
        assert run_after_2 is not run_after_1
        assert run_after_1.closed

    def test_finalize_called_again_without_new_names_is_noop(self, fake_aim_run_with_hash):
        # batch_end fires every batch; after schema settles, repeated
        # calls must NOT churn close+reopen.
        run = _HashableFakeAimRun()
        run.hash = "hash-1"
        state = SchemaPhaseState()
        observe_name(state, "train/loss")

        run_v1 = maybe_finalize_schema(run, state, cfg=make_run_config())
        # 100 more batch_end calls with the same metric set
        for _ in range(100):
            same = maybe_finalize_schema(run_v1, state, cfg=make_run_config())
            assert same is run_v1
        assert state.finalize_count == 1  # only the original finalize counted


class TestMaybeFinalizeSchemaSafetyCap:
    """Pathological case: a callback that emits new metric names at
    every boundary causes finalize churn. The safety cap halts at N
    finalizes and degrades to "captured at close, not live" for
    everything after."""

    def test_hits_max_finalizes_then_noops(self, fake_aim_run_with_hash):
        run = _HashableFakeAimRun()
        run.hash = "h"
        state = SchemaPhaseState(max_finalizes=3)

        # Drive 3 successful finalizes
        for i in range(3):
            observe_name(state, f"m{i}")
            run = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert state.finalize_count == 3

        # 4th attempt: new name, but cap reached — must be no-op
        observe_name(state, "m_too_late")
        before = run
        result = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert result is before
        assert state.finalize_count == 3
        assert "m_too_late" not in state.registered_names

    def test_max_finalizes_warning_logged_once(self, fake_aim_run_with_hash, caplog):
        import logging
        run = _HashableFakeAimRun()
        run.hash = "h"
        state = SchemaPhaseState(max_finalizes=1)
        observe_name(state, "first")
        run = maybe_finalize_schema(run, state, cfg=make_run_config())

        # Cap-hit case: should log a warning, but only once.
        observe_name(state, "second")
        with caplog.at_level(logging.WARNING):
            maybe_finalize_schema(run, state, cfg=make_run_config())
            observe_name(state, "third")
            maybe_finalize_schema(run, state, cfg=make_run_config())
        # Loguru routes through stderr by default — we set a sentinel
        # on state to rate-limit, so the second cap-hit must not
        # re-trigger the warning path.
        assert getattr(state, "_max_finalizes_logged", False) is True


class TestMaybeFinalizeSchemaFailureModes:
    """When the close + reopen cycle hits an error, the function must
    degrade gracefully — training continues even if dashboard
    visibility is compromised."""

    def test_run_close_raises_returns_original_run(self, fake_aim_run_with_hash):
        # If run.close() raises, we do NOT proceed to reopen. The
        # original run is returned. ``state.finalize_count`` does NOT
        # increment; future calls will retry.
        run = _HashableFakeAimRun()
        run.hash = "h"
        def raising_close(self):
            self.closed = True
            raise RuntimeError("simulated close failure")
        # Bind unbound method-style so self is passed in
        run.close = lambda: raising_close(run)

        state = SchemaPhaseState()
        observe_name(state, "train/loss")

        result = maybe_finalize_schema(run, state, cfg=make_run_config())
        assert result is run
        assert state.finalize_count == 0
        assert state.registered_names == set()  # NOT marked as registered

    def test_reopen_raises_returns_closed_run(self, monkeypatch):
        # close() succeeds, but Run(run_hash=..., force_resume=True)
        # raises. We return the (now-closed) original; subsequent
        # writes will silently fail via existing track_safely
        # degradation. The state is NOT advanced — names stay observed
        # but unregistered.
        original = _HashableFakeAimRun()
        original.hash = "h"

        def raising_run(**kwargs):
            raise RuntimeError("simulated reopen failure")
        monkeypatch.setattr("aim.Run", raising_run)

        state = SchemaPhaseState()
        observe_name(state, "train/loss")

        result = maybe_finalize_schema(original, state, cfg=make_run_config())
        assert result is original
        assert original.closed  # we did close it
        assert state.finalize_count == 0
        assert state.registered_names == set()


class TestSchemaPhaseStateDefaults:
    """The dataclass defaults are the contract for fresh runs."""

    def test_fresh_state_is_empty(self):
        state = SchemaPhaseState()
        assert state.observed_names == set()
        assert state.registered_names == set()
        assert state.finalize_count == 0

    def test_max_finalizes_default_is_10(self):
        # Default cap matches what's in the plan doc. Bumping requires
        # a deliberate change here and a corresponding plan update.
        state = SchemaPhaseState()
        assert state.max_finalizes == 10

    def test_max_finalizes_overridable(self):
        state = SchemaPhaseState(max_finalizes=3)
        assert state.max_finalizes == 3


# ----------------------------------------------------------------------
# Local aim server lifecycle — _maybe_start_local_aim_server +
# _parse_aim_url_host_port + _stop_local_aim_server
# ----------------------------------------------------------------------
#
# When the engine sets ASTROLABE_AIM_REPO_PATH, the callback starts
# a local ``aim server`` subprocess so the hot path (metric writes)
# stays on localhost. The NUC-side sync sidecar pulls chunks via
# SSH+rsync. These tests exercise the subprocess orchestration
# without actually spawning aim — they monkey-patch subprocess.Popen.


from astrolabe_callbacks._core import (
    _maybe_start_local_aim_server,
    _parse_aim_url_host_port,
    _stop_local_aim_server,
)


class TestParseAimUrlHostPort:
    """Pin URL-parsing edge cases. A wrong parse silently misroutes the
    local server, so the failure mode is invisible."""

    def test_standard_url(self):
        assert _parse_aim_url_host_port("aim://localhost:43800") == ("localhost", 43800)

    def test_ip_url(self):
        assert _parse_aim_url_host_port("aim://10.0.0.5:9999") == ("10.0.0.5", 9999)

    def test_wrong_scheme_returns_none(self):
        assert _parse_aim_url_host_port("http://foo:80") is None
        assert _parse_aim_url_host_port("foo:80") is None

    def test_missing_port_returns_none(self):
        assert _parse_aim_url_host_port("aim://localhost") is None

    def test_non_integer_port_returns_none(self):
        assert _parse_aim_url_host_port("aim://localhost:notanint") is None

    def test_negative_port_returns_none(self):
        assert _parse_aim_url_host_port("aim://localhost:-1") is None

    def test_zero_port_returns_none(self):
        assert _parse_aim_url_host_port("aim://localhost:0") is None

    def test_empty_host_returns_none(self):
        assert _parse_aim_url_host_port("aim://:43800") is None

    def test_url_with_trailing_path_extracts_host_port(self):
        # The remote-aim transport URLs sometimes include a trailing
        # ``/`` or path; we strip and just take host:port.
        assert _parse_aim_url_host_port("aim://nuc:43800/some/path") == ("nuc", 43800)


class TestMaybeStartLocalAimServerNoOps:
    """Cases where _maybe_start_local_aim_server must return None
    without spawning a process. Each test asserts that subprocess.Popen
    was NOT called — invariant for "no env var means existing behavior"
    back-compat."""

    def test_env_var_unset_returns_none(self, monkeypatch):
        # Standalone user case: no ASTROLABE_AIM_REPO_PATH means use
        # existing remote-only path. Must not start a server.
        monkeypatch.delenv("ASTROLABE_AIM_REPO_PATH", raising=False)
        called = []
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **kw: called.append(1) or MagicMock()
        )
        result = _maybe_start_local_aim_server(make_run_config())
        assert result is None
        assert called == []

    def test_unparseable_url_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ASTROLABE_AIM_REPO_PATH", str(tmp_path))
        cfg = make_run_config(aim_url="http://wrong-scheme:1234")
        called = []
        monkeypatch.setattr(
            "subprocess.Popen", lambda *a, **kw: called.append(1) or MagicMock()
        )
        result = _maybe_start_local_aim_server(cfg)
        assert result is None
        assert called == []


class TestMaybeStartLocalAimServerHappyPath:
    """When env + URL are valid, we call subprocess.Popen with the
    right args, poll for listening, and return the Popen handle."""

    def test_spawns_subprocess_with_correct_args(self, monkeypatch, tmp_path):
        import subprocess
        monkeypatch.setenv("ASTROLABE_AIM_REPO_PATH", str(tmp_path))
        # Pre-create .aim so we don't trigger init path.
        (tmp_path / ".aim").mkdir()

        spawned_args = []

        class FakePopen:
            def __init__(self, args, **kwargs):
                spawned_args.append(args)
                self.pid = 12345
                self._exit_code = None

            def poll(self):
                return self._exit_code  # None = still running

            def terminate(self):
                self._exit_code = 0

            def wait(self, timeout=None):
                self._exit_code = 0
                return 0

            def kill(self):
                self._exit_code = -9

        monkeypatch.setattr("subprocess.Popen", FakePopen)

        # Fake socket.create_connection to simulate "listening" on first try
        class FakeSocket:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        monkeypatch.setattr(
            "socket.create_connection", lambda *a, **kw: FakeSocket()
        )

        cfg = make_run_config(aim_url="aim://127.0.0.1:43800")
        proc = _maybe_start_local_aim_server(cfg)

        assert proc is not None
        assert len(spawned_args) == 1
        args = spawned_args[0]
        assert args[0] == "aim"
        assert args[1] == "server"
        assert "--host" in args
        assert "127.0.0.1" in args
        assert "--port" in args
        assert "43800" in args
        assert "--repo" in args
        assert str(tmp_path) in args

    def test_creates_repo_path_if_missing(self, monkeypatch, tmp_path):
        # Repo path doesn't exist — open_aim_run must create the dir
        # AND run ``aim init`` before spawning the server.
        repo_path = tmp_path / "new-repo"
        assert not repo_path.exists()
        monkeypatch.setenv("ASTROLABE_AIM_REPO_PATH", str(repo_path))

        run_calls = []
        def fake_run(*args, **kwargs):
            run_calls.append(args[0] if args else None)
            return MagicMock(returncode=0)
        monkeypatch.setattr("subprocess.run", fake_run)

        class FakePopen:
            def __init__(self, args, **kwargs):
                self.pid = 1; self._exit_code = None
            def poll(self): return self._exit_code
            def terminate(self): self._exit_code = 0
            def wait(self, timeout=None): self._exit_code = 0; return 0
            def kill(self): self._exit_code = -9
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        class FakeSocket:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        monkeypatch.setattr("socket.create_connection", lambda *a, **kw: FakeSocket())

        cfg = make_run_config(aim_url="aim://localhost:43800")
        proc = _maybe_start_local_aim_server(cfg)

        assert proc is not None
        assert repo_path.is_dir()
        # aim init was called
        assert len(run_calls) == 1
        assert "init" in run_calls[0]
        assert str(repo_path) in run_calls[0]


class TestMaybeStartLocalAimServerFailures:
    """When the subprocess fails to come up, we must clean up the
    process handle and return None — never leak an orphan."""

    def test_subprocess_exits_before_listening(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ASTROLABE_AIM_REPO_PATH", str(tmp_path))
        (tmp_path / ".aim").mkdir()

        class DyingPopen:
            def __init__(self, args, **kwargs):
                self.pid = 1
                self.returncode = 1  # already exited

            def poll(self):
                return self.returncode  # non-None = exited

            def terminate(self):
                pass
            def wait(self, timeout=None):
                return self.returncode
            def kill(self):
                pass

        monkeypatch.setattr("subprocess.Popen", DyingPopen)

        cfg = make_run_config(aim_url="aim://localhost:43800")
        proc = _maybe_start_local_aim_server(cfg)
        assert proc is None  # never returns a dead handle


class TestStopLocalAimServer:
    """Cleanup must be idempotent and never raise."""

    def test_none_proc_is_noop(self):
        # Defensive: callers shouldn't pass None, but we don't crash.
        _stop_local_aim_server(None)  # should not raise

    def test_already_exited_is_noop(self):
        class ExitedProc:
            pid = 1
            def poll(self): return 0  # already done

        # Should not call terminate/kill.
        _stop_local_aim_server(ExitedProc())

    def test_terminate_called_on_running_proc(self):
        calls = []

        class RunningProc:
            pid = 1
            def poll(self):
                return None if not calls else 0  # still running on first poll, exited after terminate
            def terminate(self):
                calls.append("terminate")
            def wait(self, timeout=None):
                return 0
            def kill(self):
                calls.append("kill")

        _stop_local_aim_server(RunningProc())
        assert "terminate" in calls
        assert "kill" not in calls  # clean shutdown, no kill needed

    def test_kill_escalation_on_timeout(self):
        calls = []

        class HangingProc:
            pid = 1
            def poll(self):
                return None  # never exits
            def terminate(self):
                calls.append("terminate")
            def wait(self, timeout=None):
                if "terminate" in calls and "kill" not in calls:
                    # First wait (post-terminate) times out; raise to trigger kill
                    raise TimeoutError("simulated wait timeout")
                return -9
            def kill(self):
                calls.append("kill")

        _stop_local_aim_server(HangingProc())
        assert calls == ["terminate", "kill"]
