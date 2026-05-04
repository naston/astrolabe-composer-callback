"""Shared test fixtures for astrolabe-callbacks.

Three jobs:

1. ``clean_astrolabe_env`` (autouse) — every test starts from a known
   env state so leftover env vars from a previous test don't leak into
   the next one. Tests that need a specific env var explicitly
   ``monkeypatch.setenv``.

2. ``fake_aim_run`` — installs a fake ``aim.Run`` class that records
   every tag write and ``track()`` call. Lets us exercise the full
   open_aim_run / close_run / track_safely paths without a live Aim
   server.

3. ``make_run_config`` — builder for ``RunConfig`` so tests don't have
   to spell out every field every time.
"""

from __future__ import annotations

from typing import Any

import pytest

from astrolabe_callbacks._core import RunConfig


@pytest.fixture(autouse=True)
def clean_astrolabe_env(monkeypatch):
    """Reset all astrolabe-related env vars before each test.

    Without this, a test that sets ``AIM_RUN_TAGS`` could leak into
    the next test's resolve_run_config and produce nondeterministic
    failures. Distributed-rank env vars are scrubbed too so the
    rank-zero default is consistent.
    """
    for name in (
        "ASTROLABE_EXPERIMENT_NAME",
        "ASTROLABE_AIM_URL",
        "AIM_RUN_TAGS",
        "ASTROLABE_CALLBACK_STRICT",
        "RANK",
        "LOCAL_RANK",
    ):
        monkeypatch.delenv(name, raising=False)


class FakeAimRun:
    """Stand-in for ``aim.Run`` that records every interaction.

    Used in tests to verify what got tagged, tracked, and closed
    without spinning up a real Aim server. Mirrors the subset of the
    Aim Run API that astrolabe-callbacks uses: ``__setitem__`` for
    tags, ``track`` for metrics, ``close`` for finalization, and a
    writable ``name`` property.
    """

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs
        self.tags: dict[str, Any] = {}
        self.tracked: list[dict[str, Any]] = []
        self.name: str | None = None
        self.closed: bool = False

    def __setitem__(self, key: str, value: Any) -> None:
        self.tags[key] = value

    def track(
        self,
        value: float,
        name: str | None = None,
        step: int | None = None,
        context: dict | None = None,
    ) -> None:
        self.tracked.append(
            {"name": name, "value": value, "step": step, "context": context}
        )

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_aim_run(monkeypatch):
    """Patch ``aim.Run`` so callbacks talk to ``FakeAimRun`` instances.

    Returns a list that fills with each ``FakeAimRun`` instance
    constructed during the test (most tests only construct one; framework
    callbacks construct one per ``open_aim_run`` call).
    """
    instances: list[FakeAimRun] = []

    class _Recording(FakeAimRun):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            instances.append(self)

    monkeypatch.setattr("aim.Run", _Recording)
    return instances


def make_run_config(**overrides: Any) -> RunConfig:
    """Builder for ``RunConfig`` with sensible test defaults.

    Keeps test bodies focused on the field that actually matters for
    each test rather than spelling out every attribute. Overrides win.
    """
    defaults = {
        "experiment_name": "test-exp",
        "aim_url": "aim://test:1",
        "tags": {},
    }
    defaults.update(overrides)
    return RunConfig(**defaults)
