"""Tests for AstrolabeLogger.

Run real Aim/Composer code where it doesn't require GPUs; mock at the
SDK boundary for anything that does.
"""

from __future__ import annotations

import pytest

from astrolabe_composer_callback import AstrolabeLogger
from astrolabe_composer_callback.logger import parse_aim_run_tags


class TestParseAimRunTags:
    """``parse_aim_run_tags`` — the env-var → dict converter."""

    def test_empty_returns_empty_dict(self):
        assert parse_aim_run_tags("") == {}
        assert parse_aim_run_tags(None) == {}

    def test_single_tag(self):
        assert parse_aim_run_tags("astrolabe.version=v3") == {
            "astrolabe.version": "v3"
        }

    def test_multiple_tags(self):
        out = parse_aim_run_tags(
            "astrolabe.version=v3,astrolabe.submit_id=abc,astrolabe.experiment=foo"
        )
        assert out == {
            "astrolabe.version": "v3",
            "astrolabe.submit_id": "abc",
            "astrolabe.experiment": "foo",
        }

    def test_whitespace_around_keys_and_values(self):
        assert parse_aim_run_tags(" key = value , key2 = value2 ") == {
            "key": "value",
            "key2": "value2",
        }

    def test_skips_entries_without_equals(self):
        # Bare strings without = are dropped, not treated as keys with empty
        # values — that would mask researcher typos like "no_value_here".
        assert parse_aim_run_tags("good=1,bad,also=2") == {"good": "1", "also": "2"}

    def test_empty_key_dropped(self):
        # "=value" has no key; skip rather than crash.
        assert parse_aim_run_tags("=orphan,real=1") == {"real": "1"}

    def test_duplicate_keys_last_wins(self):
        assert parse_aim_run_tags("k=1,k=2") == {"k": "2"}

    def test_value_can_contain_dot_and_dash(self):
        # Real astrolabe tag values are version strings, UUIDs, slugs.
        assert parse_aim_run_tags("v=v1.2-beta,sid=abc-123-def") == {
            "v": "v1.2-beta",
            "sid": "abc-123-def",
        }


class TestAstrolabeLoggerConstructor:
    """Constructor wiring — the parts that don't need a live Aim server.

    Precedence: astrolabe env vars (ASTROLABE_EXPERIMENT_NAME,
    AIM_RUN_TAGS) win over constructor args. Astrolabe is the
    orchestrator; its identity is authoritative. Constructor args are
    the standalone fallback.
    """

    def test_explicit_tags_used_when_no_env(self, monkeypatch):
        monkeypatch.delenv("AIM_RUN_TAGS", raising=False)
        log = AstrolabeLogger(tags={"k": "v"})
        assert log._tags == {"k": "v"}

    def test_env_tags_win_over_constructor(self, monkeypatch):
        # Astrolabe sets AIM_RUN_TAGS at engine setup. Whatever a
        # standalone user passes via constructor is overridden — the
        # orchestrator's tagging is authoritative.
        monkeypatch.setenv("AIM_RUN_TAGS", "astrolabe.version=v5")
        log = AstrolabeLogger(tags={"k": "v"})
        assert log._tags == {"astrolabe.version": "v5"}

    def test_no_env_no_arg_yields_empty(self, monkeypatch):
        monkeypatch.delenv("AIM_RUN_TAGS", raising=False)
        log = AstrolabeLogger()
        assert log._tags == {}

    def test_experiment_name_from_env_wins(self, monkeypatch):
        # Same precedence — env wins. This is the case that broke
        # ProjectOrion: training YAMLs hardcode experiment_name and
        # used to override astrolabe.
        monkeypatch.setenv("ASTROLABE_EXPERIMENT_NAME", "from-env")
        log = AstrolabeLogger(experiment_name="hardcoded-in-yaml")
        assert log._experiment_name == "from-env"

    def test_explicit_experiment_name_used_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("ASTROLABE_EXPERIMENT_NAME", raising=False)
        log = AstrolabeLogger(experiment_name="standalone-name")
        assert log._experiment_name == "standalone-name"


class TestInitWithMockedAim:
    """init() should set tags on the Aim run before training starts."""

    def test_init_writes_tags_to_run(self, monkeypatch):
        # Build a fake Aim Run that records all key writes.
        class FakeRun:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.tags: dict = {}

            def __setitem__(self, key, value):
                self.tags[key] = value

            def track(self, *a, **kw):
                pass

            def close(self):
                pass

        fake_run = {}

        class FakeAim:
            class Run(FakeRun):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    fake_run["instance"] = self

        monkeypatch.setitem(__import__("sys").modules, "aim", FakeAim)
        log = AstrolabeLogger(
            repo="aim://test:1",
            experiment_name="exp",
            tags={"astrolabe.version": "v3", "astrolabe.submit_id": "abc"},
        )
        log.init(state=None, logger_obj=None)
        assert fake_run["instance"].tags["astrolabe.version"] == "v3"
        assert fake_run["instance"].tags["astrolabe.submit_id"] == "abc"

    def test_init_handles_missing_aim_module(self, monkeypatch):
        # Force the `from aim import Run` to fail. The callback should
        # disable itself silently rather than crash the trainer.
        import sys
        monkeypatch.setitem(sys.modules, "aim", None)
        log = AstrolabeLogger()
        # Should not raise; _run stays None.
        log.init(state=None, logger_obj=None)
        assert log._run is None
