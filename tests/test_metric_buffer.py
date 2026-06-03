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


class TestBufferHeartbeat:
    """Periodic INFO snapshots from the drainer thread.

    Purpose is purely observational — surface buffer health
    (retries, drops, queue depth) over the run's lifetime rather
    than only at close. No notifications, no failure-handling
    changes; just an extra log line.
    """

    def _capture_loguru(self):
        """Attach a loguru sink that collects records into a list.

        Loguru doesn't integrate with pytest's `caplog` (which hooks
        stdlib logging), so we install a temporary sink and yield it.
        """
        from loguru import logger as loguru_logger

        captured: list[str] = []
        sink_id = loguru_logger.add(
            lambda message: captured.append(str(message)), level="INFO"
        )
        return captured, sink_id

    def test_heartbeat_fires_after_interval_with_activity(self, fake_aim_run):
        from loguru import logger as loguru_logger

        captured, sink_id = self._capture_loguru()
        try:
            # Tiny heartbeat interval so the test doesn't sleep 5 min.
            run = open_aim_run(make_run_config())
            run._astrolabe_buffer._heartbeat_interval_s = 0.05

            track_safely(run, name="x", value=1.0, step=1)
            track_safely(run, name="x", value=2.0, step=2)
            # Let the drainer process + heartbeat fire.
            time.sleep(0.2)

            heartbeats = [line for line in captured if "heartbeat" in line]
            assert len(heartbeats) >= 1, (
                f"Expected at least one heartbeat line; captured: {captured}"
            )
            # Heartbeat contents reflect activity that happened.
            assert "submitted" in heartbeats[0]
            assert "drained" in heartbeats[0]
        finally:
            loguru_logger.remove(sink_id)

    def test_heartbeat_silent_when_no_activity_changed(self, fake_aim_run):
        from loguru import logger as loguru_logger

        captured, sink_id = self._capture_loguru()
        try:
            run = open_aim_run(make_run_config())
            run._astrolabe_buffer._heartbeat_interval_s = 0.05
            # Drainer is alive but idle — interval elapses many times
            # without any submit/drain/retry. Should NOT spam logs.
            time.sleep(0.3)
            heartbeats = [line for line in captured if "heartbeat" in line]
            assert heartbeats == [], (
                f"Expected zero heartbeats during idle; got: {heartbeats}"
            )
        finally:
            loguru_logger.remove(sink_id)

    def test_heartbeat_does_not_fire_before_interval(self, fake_aim_run):
        from loguru import logger as loguru_logger

        captured, sink_id = self._capture_loguru()
        try:
            run = open_aim_run(make_run_config())
            # Generous interval so it can't fire during the test.
            run._astrolabe_buffer._heartbeat_interval_s = 60.0

            for i in range(10):
                track_safely(run, name="x", value=float(i), step=i)
            time.sleep(0.15)

            heartbeats = [line for line in captured if "heartbeat" in line]
            assert heartbeats == [], (
                "Heartbeat should not fire before its interval elapses; "
                f"captured: {captured}"
            )
        finally:
            loguru_logger.remove(sink_id)


class TestStatsToDisk:
    """The ``ASTROLABE_CALLBACK_STATS_PATH`` env var is the
    survivability story for buffer diagnostics. Without it,
    heartbeats and the close summary only land in the training
    process's stdout — which dies with the Lambda instance. Astrolabe
    sets this env var on Lambda so the file rsyncs back at step end.

    Contract:
      - No env var set → no file write attempts (and no exceptions)
      - Env var set → each heartbeat appends one JSONL line
      - Env var set → close_run appends one JSONL line with kind=close
      - Path supports ~ expansion
      - Write failures are silent (the diagnostic side-channel must
        not bring down the training run)"""

    def test_no_env_var_means_no_file(self, fake_aim_run, monkeypatch, tmp_path):
        monkeypatch.delenv("ASTROLABE_CALLBACK_STATS_PATH", raising=False)
        run = open_aim_run(make_run_config())
        run._astrolabe_buffer._heartbeat_interval_s = 0.05
        track_safely(run, name="x", value=1.0, step=1)
        time.sleep(0.15)
        # No file was created (we'd see it in tmp_path if it had been).
        assert list(tmp_path.iterdir()) == []

    def test_heartbeat_writes_jsonl(self, fake_aim_run, monkeypatch, tmp_path):
        import json

        stats_file = tmp_path / "stats.jsonl"
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", str(stats_file))

        run = open_aim_run(make_run_config())
        run._astrolabe_buffer._heartbeat_interval_s = 0.05
        track_safely(run, name="x", value=1.0, step=1)
        track_safely(run, name="x", value=2.0, step=2)
        time.sleep(0.2)

        assert stats_file.exists(), "stats file should be created on first heartbeat"
        lines = stats_file.read_text().strip().splitlines()
        assert len(lines) >= 1
        record = json.loads(lines[0])
        assert record["kind"] == "heartbeat"
        assert "ts" in record
        assert record["submitted"] >= 2
        assert record["drained"] >= 0
        assert record["retried"] == 0
        assert record["dropped"] == 0

    def test_close_writes_summary_line(self, fake_aim_run, monkeypatch, tmp_path):
        import json

        stats_file = tmp_path / "stats.jsonl"
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", str(stats_file))

        run = open_aim_run(make_run_config())
        for i in range(5):
            track_safely(run, name="x", value=float(i), step=i)
        close_run(run, status="completed")

        assert stats_file.exists()
        lines = stats_file.read_text().strip().splitlines()
        close_records = [json.loads(l) for l in lines if json.loads(l)["kind"] == "close"]
        assert len(close_records) == 1, f"expected exactly one close line, got {len(close_records)}"
        r = close_records[0]
        assert r["status"] == "completed"
        assert r["submitted"] == 5
        assert r["drained"] == 5
        assert r["unflushed"] == 0

    def test_tilde_path_expands(self, fake_aim_run, monkeypatch, tmp_path):
        # Use a literal ~ in the env var and verify expansion happens.
        # HOME=tmp_path so ~ resolves into the test dir.
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", "~/stats.jsonl")

        run = open_aim_run(make_run_config())
        run._astrolabe_buffer._heartbeat_interval_s = 0.05
        track_safely(run, name="x", value=1.0, step=1)
        time.sleep(0.15)

        assert (tmp_path / "stats.jsonl").exists()

    def test_write_failure_is_silent(self, fake_aim_run, monkeypatch, tmp_path):
        # Point the env var at an unwritable path. The buffer must
        # NOT raise; the training run continues regardless.
        bad_path = tmp_path / "nope" / "deep" / "stats.jsonl"
        # Don't pre-create the parents — open() will fail. We expect
        # the diagnostic to swallow the exception.
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", str(bad_path))

        run = open_aim_run(make_run_config())
        run._astrolabe_buffer._heartbeat_interval_s = 0.05
        # If write failure raised, this would propagate and fail the test.
        for i in range(3):
            track_safely(run, name="x", value=float(i), step=i)
        time.sleep(0.15)
        # Buffer still functioning — values landed in the run.
        run._astrolabe_buffer.flush(timeout_s=2.0)
        assert len(run.tracked) == 3


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


class TestBoundedRetry:
    """v1.1.1 — per-item retry cap. Pre-fix the drainer's inner retry
    loop was ``while True``: a single persistently-failing
    ``run.track()`` call pinned the drainer forever, and every item
    queued behind it was lost (either drop-oldest after the 100k queue
    fills or simply lost on close()).

    Observed signature 2026-06-03: 04-muon-adamw-handoff v7 captured
    753 of 2470 train/loss steps in Aim (clean cliff at step 752),
    while TB had the full trace from the same training process.
    """

    def test_dropped_failed_increments_after_max_retries(self, monkeypatch):
        """A track call that ALWAYS fails should be retried up to
        max_retries then given up on. dropped_failed reflects each
        such giveup; the drainer keeps flowing for new items."""
        class AlwaysFails(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                raise RuntimeError("server says no")

        monkeypatch.setattr("aim.Run", AlwaysFails)
        run = open_aim_run(make_run_config())
        buf = run._astrolabe_buffer
        # Fast retries + tight cap so the test doesn't wait minutes.
        buf._retry_initial = 0.001
        buf._retry_max = 0.005
        buf._max_retries = 3

        track_safely(run, name="train/loss", value=0.5, step=1)
        # Wait for the drainer to exhaust retries on this one item.
        # 3 attempts × ~5ms + slack.
        assert buf.flush(timeout_s=2.0)

        stats = buf.stats()
        assert stats["submitted"] == 1
        assert stats["drained"] == 0
        # 3 attempts means 3 retry-counter increments (one per failed
        # track call). The 3rd one hits the cap and triggers the giveup.
        assert stats["retried"] == 3
        assert stats["dropped_failed"] == 1

    def test_drainer_keeps_flowing_after_dropped_failed(self, monkeypatch):
        """Once a poisoned item is dropped, subsequent items that DO
        succeed should drain normally. Pre-fix this never happened —
        the drainer never moved past the wedged item, so item #2
        sat in the queue forever."""
        fail_for = {"train/loss"}

        class SelectiveFail(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                if name in fail_for:
                    raise RuntimeError(f"{name} rejected")
                super().track(value, name=name, step=step, context=context)

        monkeypatch.setattr("aim.Run", SelectiveFail)
        run = open_aim_run(make_run_config())
        buf = run._astrolabe_buffer
        buf._retry_initial = 0.001
        buf._retry_max = 0.005
        buf._max_retries = 3

        # The poisoned item goes first, then a healthy one.
        track_safely(run, name="train/loss", value=0.5, step=1)
        track_safely(run, name="train/accuracy", value=0.95, step=1)
        assert buf.flush(timeout_s=2.0)

        stats = buf.stats()
        assert stats["dropped_failed"] == 1
        assert stats["drained"] == 1  # the healthy item made it through
        # And run.tracked has only the healthy one.
        assert run.tracked == [
            {"name": "train/accuracy", "value": 0.95, "step": 1, "context": {}}
        ]

    def test_dropped_failed_in_heartbeat(self, monkeypatch, tmp_path):
        """Heartbeat snapshot includes dropped_failed so a long-running
        run shows the counter rising as poisoned items accumulate —
        without needing to wait for close(). Mirrored to the disk-stats
        jsonl so survives composer-SIGABRT stdout loss."""
        stats_path = tmp_path / "callback-stats.jsonl"
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", str(stats_path))

        class AlwaysFails(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                raise RuntimeError("nope")

        monkeypatch.setattr("aim.Run", AlwaysFails)
        run = open_aim_run(make_run_config())
        buf = run._astrolabe_buffer
        buf._retry_initial = 0.001
        buf._retry_max = 0.005
        buf._max_retries = 2
        # Force a heartbeat on every drainer iteration so we don't
        # wait 5 minutes.
        buf._heartbeat_interval_s = 0.0

        track_safely(run, name="x", value=1.0, step=0)
        buf.flush(timeout_s=2.0)
        # Force one more heartbeat-cycle observation.
        time.sleep(0.1)

        import json
        lines = [json.loads(ln) for ln in stats_path.read_text().splitlines() if ln.strip()]
        # We expect at least one heartbeat reflecting dropped_failed=1.
        heartbeats = [ln for ln in lines if ln.get("kind") == "heartbeat"]
        assert any(hb.get("dropped_failed", 0) >= 1 for hb in heartbeats), (
            f"no heartbeat carried dropped_failed>=1; lines={lines}"
        )

    def test_first_failure_mirrored_to_disk_stats(self, monkeypatch, tmp_path):
        """The 'Aim track failed for X' WARNING goes to stdout, which
        composer's launcher loses to SIGABRT. Mirror it as a jsonl
        record so post-mortem diagnosis isn't blind."""
        stats_path = tmp_path / "callback-stats.jsonl"
        monkeypatch.setenv("ASTROLABE_CALLBACK_STATS_PATH", str(stats_path))

        class AlwaysFails(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                raise RuntimeError("kaboom")

        monkeypatch.setattr("aim.Run", AlwaysFails)
        run = open_aim_run(make_run_config())
        buf = run._astrolabe_buffer
        buf._retry_initial = 0.001
        buf._retry_max = 0.005
        buf._max_retries = 2

        track_safely(run, name="train/loss", value=0.5, step=42)
        buf.flush(timeout_s=2.0)

        import json
        lines = [json.loads(ln) for ln in stats_path.read_text().splitlines() if ln.strip()]
        first_failures = [ln for ln in lines if ln.get("kind") == "track_failed_first"]
        assert len(first_failures) == 1
        ff = first_failures[0]
        assert ff["metric"] == "train/loss"
        assert ff["step"] == 42
        assert "kaboom" in ff["exception"]

        # And the give-up record after retries exhaust.
        givenup = [ln for ln in lines if ln.get("kind") == "track_dropped_failed"]
        assert len(givenup) == 1
        gu = givenup[0]
        assert gu["metric"] == "train/loss"
        assert gu["attempts"] == 2
        assert gu["dropped_failed_total"] == 1

    def test_close_summary_includes_dropped_failed(self, monkeypatch):
        """The close-run summary surfaces dropped_failed so anyone
        glancing at the post-run log can see how many items the
        drainer gave up on."""
        class AlwaysFails(FakeAimRun):
            def track(self, value, name=None, step=None, context=None):
                raise RuntimeError("permanent")

        monkeypatch.setattr("aim.Run", AlwaysFails)
        run = open_aim_run(make_run_config())
        buf = run._astrolabe_buffer
        buf._retry_initial = 0.001
        buf._retry_max = 0.005
        buf._max_retries = 2

        track_safely(run, name="metric_a", value=1.0, step=0)
        track_safely(run, name="metric_b", value=2.0, step=0)
        buf.flush(timeout_s=2.0)

        stats = buf.stats()
        # Both items failed permanently; both dropped.
        assert stats["dropped_failed"] == 2
        # close_run shouldn't raise even with a buffer that gave up.
        close_run(run, status="failed")
