"""Tests for astrolabe_callbacks.eval_results — post-training eval helpers.

Contract being verified:

* :func:`log_eval_table` opens an Aim run, applies the three-tag identity
  contract, tracks each row under ``eval/<task>/<metric>`` at ``step=0``,
  and closes the run.
* :func:`start_eval_run` returns an *open* run with the three tags set;
  the caller owns ``close()``.
* Both reject malformed inputs at the call site BEFORE creating any
  Aim run — half-tagged runs would silently confuse astrolabe's
  dashboard.
* The metric path convention ``eval/<task>/<metric>`` must be exact —
  slashes in the task or metric label scramble the dashboard's column
  parsing.

Tests mock ``aim.Run`` at the SDK boundary. Real Aim has
indexing/commit timing quirks that bite read-back tests; mocking the
SDK boundary is the convention across this package's tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from astrolabe_callbacks.eval_results import (
    EvalInputError,
    log_eval_table,
    start_eval_run,
)


# ---------- helpers ----------------------------------------------------


def _make_run_mock(run_hash: str = "abc123") -> MagicMock:
    """Stand-in for ``aim.Run``. Tracks setitem + track + close calls."""
    run = MagicMock()
    run.hash = run_hash
    return run


# ---------- input validation: log_eval_table -------------------------


class TestLogEvalTableValidation:
    """All validation errors MUST fire BEFORE any Aim run is created.

    A half-tagged eval run (e.g., kind set but task_set missing) would
    appear in the dashboard's discovery query but render nothing — silent
    corruption is worse than a noisy crash at the call site.
    """

    @pytest.fixture(autouse=True)
    def _aim_patch(self):
        """Patch aim.Run for every test in this class; assert it wasn't called.

        The fixture is autouse so even tests that expect validation to
        raise can verify Run() was never reached.
        """
        self.aim_run_mock = MagicMock(return_value=_make_run_mock())
        self.aim_url = "aim://test"
        with patch("aim.Run", self.aim_run_mock):
            yield

    def _assert_no_run_created(self):
        assert not self.aim_run_mock.called, (
            "Validation must reject BEFORE creating an Aim run; "
            f"aim.Run was called with {self.aim_run_mock.call_args}"
        )

    def test_rejects_empty_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            log_eval_table(
                model_run_hash="",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_string_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            log_eval_table(
                model_run_hash=None,  # type: ignore[arg-type]
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_task_set(self):
        with pytest.raises(EvalInputError, match="task_set"):
            log_eval_table(
                model_run_hash="abc",
                task_set="",
                rows={"cola": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_rows(self):
        with pytest.raises(EvalInputError, match="at least one task"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_dict_rows(self):
        with pytest.raises(EvalInputError, match="must be a dict"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows=[("cola", "matthews", 0.5)],  # type: ignore[arg-type]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_task_name(self):
        with pytest.raises(EvalInputError, match="task name"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_slash_in_task_name(self):
        # Metric path is ``eval/<task>/<metric>``. A slash in the task
        # scrambles which segment the dashboard reads as which.
        with pytest.raises(EvalInputError, match="must not contain '/'"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola/sub": ("matthews", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_slash_in_metric_label(self):
        with pytest.raises(EvalInputError, match="must not contain '/'"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews/v2", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_empty_metric_label(self):
        with pytest.raises(EvalInputError, match="metric label"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("", 0.5)},
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_non_tuple_row(self):
        with pytest.raises(EvalInputError, match="must be a .* tuple"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": 0.5},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_three_element_row(self):
        with pytest.raises(EvalInputError, match="must be a .* tuple"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5, "extra")},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_string_score(self):
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", "0.5")},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_bool_score(self):
        # bool is an int subclass in Python — without the explicit reject,
        # ``rows={"cola": ("accuracy", True)}`` would silently log 1.0.
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("accuracy", True)},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()

    def test_rejects_none_score(self):
        with pytest.raises(EvalInputError, match="must be a number"):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", None)},  # type: ignore[dict-item]
                aim_url=self.aim_url,
            )
        self._assert_no_run_created()


# ---------- input validation: start_eval_run -------------------------


class TestStartEvalRunValidation:
    @pytest.fixture(autouse=True)
    def _aim_patch(self):
        self.aim_run_mock = MagicMock(return_value=_make_run_mock())
        self.aim_url = "aim://test"
        with patch("aim.Run", self.aim_run_mock):
            yield

    def test_rejects_empty_model_run_hash(self):
        with pytest.raises(EvalInputError, match="model_run_hash"):
            start_eval_run(
                model_run_hash="",
                task_set="glue",
                aim_url=self.aim_url,
            )
        assert not self.aim_run_mock.called

    def test_rejects_empty_task_set(self):
        with pytest.raises(EvalInputError, match="task_set"):
            start_eval_run(
                model_run_hash="abc",
                task_set="",
                aim_url=self.aim_url,
            )
        assert not self.aim_run_mock.called


# ---------- tag contract ---------------------------------------------


class TestTagContract:
    """The three identity tags are the dashboard's only way to discover
    eval runs from the model run page. Missing any of them = invisible run."""

    def test_log_eval_table_sets_all_three_tags(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="model-hash-123",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "glue")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "model-hash-123")

    def test_start_eval_run_sets_all_three_tags(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="xyz",
                task_set="mmlu",
                aim_url="aim://test",
            )
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "mmlu")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "xyz")

    def test_aim_run_filed_under_eval_task_set_experiment(self, tmp_path):
        # Keeps eval runs out of the model experiment's Aim run list —
        # a UI convenience for browsing the raw Aim repo. Discovery
        # doesn't depend on it (uses tags).
        run = _make_run_mock()
        mock_run_factory = MagicMock(return_value=run)
        with patch("aim.Run", mock_run_factory):
            start_eval_run(
                model_run_hash="abc",
                task_set="glue",
                aim_url="aim://test",
            )
        mock_run_factory.assert_called_once()
        kwargs = mock_run_factory.call_args.kwargs
        assert kwargs["experiment"] == "eval/glue"


# ---------- metric path convention ----------------------------------


class TestMetricPaths:
    """The dashboard's table block parses ``eval/<task>/<metric>``.
    Tracking under any other shape leaves the table un-populated."""

    def test_paths_use_eval_task_metric_convention(self, tmp_path):
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={
                    "cola": ("matthews", 0.822),
                    "sst2": ("accuracy", 0.943),
                },
                aim_url="aim://test",
            )
        # Verify both metrics tracked with the right paths.
        run.track.assert_any_call(0.822, name="eval/cola/matthews", step=0)
        run.track.assert_any_call(0.943, name="eval/sst2/accuracy", step=0)

    def test_one_track_call_per_row(self, tmp_path):
        run = _make_run_mock()
        rows = {
            "cola": ("matthews", 0.5),
            "sst2": ("accuracy", 0.6),
            "mnli": ("accuracy_matched", 0.7),
        }
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows=rows,
                aim_url="aim://test",
            )
        assert run.track.call_count == len(rows)

    def test_step_is_zero(self, tmp_path):
        # step=0 marks "post-training one-shot" — the dispatcher uses
        # this to choose the table block over the trace block.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        for call_args in run.track.call_args_list:
            assert call_args.kwargs["step"] == 0

    def test_score_is_cast_to_float(self, tmp_path):
        # Aim's track() expects a numeric — int inputs should reach it
        # as floats so all eval values share a type in storage.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 1)},  # int score
                aim_url="aim://test",
            )
        ((value,), _) = run.track.call_args
        assert isinstance(value, float)
        assert value == 1.0


# ---------- run lifecycle ----------------------------------------------


class TestRunLifecycle:
    def test_log_eval_table_closes_run(self, tmp_path):
        # Forgetting to close leaves end_time=0 — the dashboard would
        # render this as in-flight forever.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        run.close.assert_called_once()

    def test_log_eval_table_closes_run_even_when_track_raises(self, tmp_path):
        # If Aim's track() blows up mid-loop, we still need to close the
        # run so we don't leak an in-flight tag on the dashboard.
        run = _make_run_mock()
        run.track.side_effect = RuntimeError("aim hiccup")
        with patch("aim.Run", return_value=run):
            with pytest.raises(RuntimeError, match="aim hiccup"):
                log_eval_table(
                    model_run_hash="abc",
                    task_set="glue",
                    rows={"cola": ("matthews", 0.5)},
                    aim_url="aim://test",
                )
        run.close.assert_called_once()

    def test_start_eval_run_does_not_close(self, tmp_path):
        # The lower-level helper hands the caller an OPEN run. Closing
        # here would set end_time=0 immediately.
        run = _make_run_mock()
        with patch("aim.Run", return_value=run):
            start_eval_run(
                model_run_hash="abc",
                task_set="glue",
                aim_url="aim://test",
            )
        assert not run.close.called

    def test_log_eval_table_returns_run_hash(self, tmp_path):
        run = _make_run_mock(run_hash="b73e9c8d")
        with patch("aim.Run", return_value=run):
            got = log_eval_table(
                model_run_hash="abc",
                task_set="glue",
                rows={"cola": ("matthews", 0.5)},
                aim_url="aim://test",
            )
        assert got == "b73e9c8d"


# ---------- happy path summary ----------------------------------------


class TestHappyPath:
    def test_full_glue_table_round_trip(self, tmp_path):
        run = _make_run_mock(run_hash="eval-abc")
        with patch("aim.Run", return_value=run):
            got = log_eval_table(
                model_run_hash="model-hash-123",
                task_set="glue",
                rows={
                    "cola": ("matthews",          0.822),
                    "sst2": ("accuracy",          0.943),
                    "mnli": ("accuracy_matched",  0.864),
                    "avg":  ("mean",              0.876),
                },
                aim_url="aim://test",
            )
        assert got == "eval-abc"
        # All three identity tags set
        run.__setitem__.assert_any_call("astrolabe.kind", "eval")
        run.__setitem__.assert_any_call("astrolabe.task_set", "glue")
        run.__setitem__.assert_any_call("astrolabe.model_run_hash", "model-hash-123")
        # All four rows tracked under the convention path
        run.track.assert_any_call(0.822, name="eval/cola/matthews", step=0)
        run.track.assert_any_call(0.943, name="eval/sst2/accuracy", step=0)
        run.track.assert_any_call(0.864, name="eval/mnli/accuracy_matched", step=0)
        run.track.assert_any_call(0.876, name="eval/avg/mean", step=0)
        # Run closed
        run.close.assert_called_once()
