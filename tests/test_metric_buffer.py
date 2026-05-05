"""Tests for the async write-buffer + retry layer (``_MetricBuffer``).

The rest of the test suite uses an autouse fixture in ``conftest.py``
that makes ``_MetricBuffer.submit`` synchronous so test bodies can
assert on tracked values immediately after a track call. This file
overrides that fixture so each test exercises the *real* async +
retry behavior — which is the whole point of the buffer.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest

from astrolabe_callbacks._core import (
    _DEFAULT_DRAIN_TIMEOUT_S,
    _MetricBuffer,
    close_run,
    open_aim_run,
    track_safely,
)
from tests.conftest import FakeAimRun, make_run_config


@pytest.fixture(autouse=True)
def synchronous_metric_buffer():
    """Override conftest's autouse sync fixture — these tests need
    the real async drainer thread so we can verify retry, drop-on-
    overflow, and drain-on-close. Yielding without monkeypatching
    leaves ``_MetricBuffer.submit`` at its production implementation.
    """
    yield


# ----------------------------------------------------------------------
# Edge cases — the failure modes that justify owning this layer
# ----------------------------------------------------------------------


class TestBufferRetry:
    """Transient track failures retry on the drainer thread until
    they succeed. The caller never sees the failure."""

    def test_retry_succeeds_after_transient_failures(self, monkeypatch):
        # Fails the first 3 calls, succeeds on the 4th. Buffer's retry
        # loop should drive through to success — the eventual value
        # lands in the run's tracked list.
        attempts = {"count": 0}

        class FlakyTrack(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                attempts["count"] += 1
                if attempts["count"] < 4:
                    raise RuntimeError(f"transient {attempts['count']}")
                super().track(value, name=name, step=step, context=context)

        monkeypatch.setattr("aim.Run", FlakyTrack)
        run = open_aim_run(make_run_config())
        # Use a fast retry backoff so the test doesn't wait 30s.
        run._astrolabe_buffer._retry_initial = 0.01
        run._astrolabe_buffer._retry_max = 0.05

        track_safely(run, name="train/loss", value=0.5, step=1)
        # Wait for retries to converge. 5 attempts × 50ms max + slack.
        assert run._astrolabe_buffer.flush(timeout_s=2.0)

        assert attempts["count"] == 4
        # The final successful track landed.
        assert run.tracked == [
            {"name": "train/loss", "value": 0.5, "step": 1, "context": {}}
        ]

    def test_retry_count_recorded_in_stats(self, monkeypatch):
        attempts = {"count": 0}

        class FlakyTrack(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                attempts["count"] += 1
                if attempts["count"] < 3:
                    raise RuntimeError("fail")
                super().track(value, name=name, step=step, context=context)

        monkeypatch.setattr("aim.Run", FlakyTrack)
        run = open_aim_run(make_run_config())
        run._astrolabe_buffer._retry_initial = 0.01
        run._astrolabe_buffer._retry_max = 0.05

        track_safely(run, name="x", value=1.0, step=0)
        run._astrolabe_buffer.flush(timeout_s=2.0)

        stats = run._astrolabe_buffer.stats()
        # Two failed attempts before success → retried = 2.
        assert stats["retried"] == 2
        assert stats["drained"] == 1
        assert stats["submitted"] == 1


class TestBufferDropOnOverflow:
    """When the queue fills, oldest is dropped to make room for the
    newest. The newest is what tells you 'where the run is now',
    which is more useful than 'where it was an hour ago' if you have
    to choose."""

    def test_drops_oldest_when_full(self, monkeypatch):
        # Block the drainer entirely so the queue fills up.
        blocking_track = threading.Event()

        class BlockingTrack(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                blocking_track.wait(timeout=10)
                super().track(value, name=name, step=step, context=context)

        monkeypatch.setattr("aim.Run", BlockingTrack)
        run = open_aim_run(make_run_config())
        # Tiny queue so we can fill it without writing 100k items.
        # Replace the buffer entirely so the size sticks.
        run._astrolabe_buffer.close(timeout_s=0.5)
        run._astrolabe_buffer = _MetricBuffer(run, max_size=4)

        try:
            # The drainer pulls one item immediately and starts to
            # block on it (waiting on the Event). That leaves room for
            # 4 more in the queue. The 6th item triggers drop-oldest.
            for i in range(8):
                track_safely(run, name="x", value=float(i), step=i)
                # Tiny pause so each submit is observed in order.
                time.sleep(0.005)

            stats = run._astrolabe_buffer.stats()
            assert stats["dropped_oldest"] >= 1
            assert stats["submitted"] == 8
        finally:
            blocking_track.set()
            run._astrolabe_buffer.close(timeout_s=2.0)


class TestBufferClose:
    """``close`` drains queued writes before letting the run go
    away. Critical for end-of-training: without this, fit_end's
    close would race the drainer and lose in-flight metrics."""

    def test_close_drains_queue(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        for i in range(50):
            track_safely(run, name="x", value=float(i), step=i)

        unflushed = run._astrolabe_buffer.close(timeout_s=5.0)
        assert unflushed == 0
        # All 50 items landed.
        assert len(run.tracked) == 50

    def test_close_returns_unflushed_on_timeout(self, monkeypatch):
        # Drainer that blocks forever — close should hit timeout and
        # return non-zero unflushed count.
        forever = threading.Event()

        class BlockingTrack(FakeAimRun):
            def track(self, *args, **kwargs):
                forever.wait(timeout=60)

        monkeypatch.setattr("aim.Run", BlockingTrack)
        run = open_aim_run(make_run_config())
        try:
            for i in range(10):
                track_safely(run, name="x", value=float(i), step=i)
            unflushed = run._astrolabe_buffer.close(timeout_s=0.3)
            # Drainer is stuck on item 0; items 1..9 are still queued.
            assert unflushed >= 1
        finally:
            forever.set()

    def test_close_run_drains_then_finalizes(self, fake_aim_run):
        """``close_run`` must drain the buffer before writing the
        status tag and calling run.close(). Without that ordering the
        in-flight queue would be lost when the Run is finalized."""
        run = open_aim_run(make_run_config())
        for i in range(10):
            track_safely(run, name="x", value=float(i), step=i)
        close_run(run, status="completed")

        # All 10 items made it through before close fired.
        names = [t["name"] for t in fake_aim_run[-1].tracked]
        assert names.count("x") == 10
        # Status tag set + run closed cleanly.
        assert fake_aim_run[-1].tags["astrolabe.status"] == "completed"
        assert fake_aim_run[-1].closed is True


class TestStrictModeBypass:
    """Strict mode (``ASTROLABE_CALLBACK_STRICT=1``) bypasses the
    buffer entirely — synchronous + raise. Strict semantics ('crash
    on any failure') don't fit a background-thread retry model."""

    def test_strict_mode_skips_buffer(self, fake_aim_run, monkeypatch):
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        run = open_aim_run(make_run_config())
        # Even though a buffer was attached at open, strict bypasses it.
        track_safely(run, name="x", value=1.0, step=0)
        # Synchronous: tracked immediately, buffer's submit not called.
        assert run.tracked == [
            {"name": "x", "value": 1.0, "step": 0, "context": {}}
        ]
        # Buffer counters didn't move.
        assert run._astrolabe_buffer.stats()["submitted"] == 0

    def test_strict_mode_raises_on_track_failure(self, monkeypatch):
        class FailingTrack(FakeAimRun):
            def track(self, *args, **kwargs):
                raise RuntimeError("aim broke")

        monkeypatch.setattr("aim.Run", FailingTrack)
        monkeypatch.setenv("ASTROLABE_CALLBACK_STRICT", "1")
        run = open_aim_run(make_run_config())

        with pytest.raises(RuntimeError, match="aim broke"):
            track_safely(run, name="x", value=1.0, step=0)


# ----------------------------------------------------------------------
# Happy path — confirms the buffer is wired in and stats track right
# ----------------------------------------------------------------------


class TestBufferHappyPath:
    def test_submit_then_flush_lands_value(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        track_safely(run, name="train/loss", value=0.5, step=10)
        assert run._astrolabe_buffer.flush(timeout_s=2.0)
        assert run.tracked == [
            {"name": "train/loss", "value": 0.5, "step": 10, "context": {}}
        ]

    def test_buffer_attached_at_open(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        assert hasattr(run, "_astrolabe_buffer")
        assert isinstance(run._astrolabe_buffer, _MetricBuffer)

    def test_stats_match_traffic(self, fake_aim_run):
        run = open_aim_run(make_run_config())
        for i in range(20):
            track_safely(run, name="x", value=float(i), step=i)
        run._astrolabe_buffer.flush(timeout_s=2.0)
        stats = run._astrolabe_buffer.stats()
        assert stats["submitted"] == 20
        assert stats["drained"] == 20
        assert stats["retried"] == 0
        assert stats["dropped_oldest"] == 0
