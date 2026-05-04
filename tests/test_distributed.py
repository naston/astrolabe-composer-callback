"""Tests for ``astrolabe_callbacks._distributed``.

The rank-zero gate is small but load-bearing: every framework callback
relies on it to avoid duplicating Aim writes across ranks. Unhappy
paths (env var present but malformed) come first.
"""

from __future__ import annotations

import pytest

from astrolabe_callbacks._distributed import is_rank_zero


# ----------------------------------------------------------------------
# Edge cases — env var present but unexpected shape
# ----------------------------------------------------------------------


class TestRankEnvEdgeCases:
    def test_empty_rank_env_treated_as_rank_zero(self, monkeypatch):
        # ``RANK=""`` is not "0" → strictly speaking, not rank-zero. But
        # this is also an unusual env state. Document actual behavior:
        # only the literal string "0" is rank-zero.
        monkeypatch.setenv("RANK", "")
        assert is_rank_zero() is False

    def test_whitespace_rank_not_zero(self, monkeypatch):
        # "0 " (trailing space) is not "0". torchrun never produces this,
        # but if a user manually sets it, we don't silently treat it as
        # rank-zero.
        monkeypatch.setenv("RANK", "0 ")
        assert is_rank_zero() is False

    def test_zero_padded_rank_not_zero(self, monkeypatch):
        # "00" is not "0". torchrun emits unpadded integers.
        monkeypatch.setenv("RANK", "00")
        assert is_rank_zero() is False

    def test_rank_takes_priority_over_local_rank(self, monkeypatch):
        # If both are set and disagree, RANK wins (it's the global
        # rank; LOCAL_RANK is per-node).
        monkeypatch.setenv("RANK", "1")
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert is_rank_zero() is False

    def test_local_rank_used_when_only_local_set(self, monkeypatch):
        # Some launchers set LOCAL_RANK before global RANK. Falls
        # through to it.
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "0")
        assert is_rank_zero() is True

    def test_local_rank_nonzero(self, monkeypatch):
        monkeypatch.delenv("RANK", raising=False)
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert is_rank_zero() is False


# ----------------------------------------------------------------------
# torch.distributed branch
# ----------------------------------------------------------------------


class TestTorchDistributedBranch:
    """When torch.distributed is initialized, it's the source of truth.

    Env vars are a fallback for the case where the process group hasn't
    been initialized yet (callback init fires before dist.init).
    """

    def test_initialized_distributed_overrides_env(self, monkeypatch):
        # When torch.distributed reports initialized + rank=0, that
        # wins regardless of RANK env. Patch the real module's
        # attributes; module-replacement via sys.modules doesn't work
        # because ``import torch.distributed as dist`` resolves through
        # the already-imported torch package.
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 0)
        monkeypatch.setenv("RANK", "1")
        assert is_rank_zero() is True

    def test_initialized_nonzero_rank(self, monkeypatch):
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: True)
        monkeypatch.setattr(dist, "get_rank", lambda: 2)
        monkeypatch.setenv("RANK", "0")  # env says zero, dist says 2
        assert is_rank_zero() is False

    def test_uninitialized_falls_through_to_env(self, monkeypatch):
        # Process group exists but isn't initialized → use env vars.
        # Common case before torchrun's dist.init_process_group fires.
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: True)
        monkeypatch.setattr(dist, "is_initialized", lambda: False)
        # get_rank would fail if it were called; the test verifies
        # that we don't call it when uninitialized.
        monkeypatch.setattr(
            dist, "get_rank", lambda: pytest.fail("should not call get_rank")
        )
        monkeypatch.setenv("RANK", "0")
        assert is_rank_zero() is True

    def test_unavailable_falls_through_to_env(self, monkeypatch):
        # Builds without distributed support (e.g. CPU-only torch wheel).
        import torch.distributed as dist

        monkeypatch.setattr(dist, "is_available", lambda: False)
        # get_rank should not be called when unavailable.
        monkeypatch.setattr(
            dist, "is_initialized", lambda: pytest.fail("should not check init")
        )
        monkeypatch.setenv("RANK", "2")
        assert is_rank_zero() is False


# ----------------------------------------------------------------------
# Happy paths
# ----------------------------------------------------------------------


class TestHappyPaths:
    def test_no_env_no_torch_returns_true(self):
        # Single-process training with no RANK env: treated as
        # rank-zero. This is the common case for laptop dev runs.
        assert is_rank_zero() is True

    def test_rank_env_zero(self, monkeypatch):
        monkeypatch.setenv("RANK", "0")
        assert is_rank_zero() is True

    @pytest.mark.parametrize("rank", ["1", "2", "7", "127"])
    def test_rank_env_nonzero(self, monkeypatch, rank):
        monkeypatch.setenv("RANK", rank)
        assert is_rank_zero() is False
