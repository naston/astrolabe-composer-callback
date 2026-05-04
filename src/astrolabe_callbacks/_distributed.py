"""Rank-zero detection for distributed training.

Every astrolabe-callbacks callback gates Aim writes on rank-zero so a
distributed run with N processes produces one Aim run, not N. Without
this we get N copies of every metric and the run-close handshake races.

Detection order:

1. ``torch.distributed.is_initialized()`` — preferred when available.
   Most launchers (``torchrun``, Composer, Lightning) initialize the
   process group before any callback runs.
2. ``RANK`` env var — set by ``torchrun`` and most cluster launchers
   before Python even imports ``torch``.
3. ``LOCAL_RANK`` env var — set per-node; useful when a callback runs
   before the global process group exists.
4. Falls back to ``True`` (treat as rank-zero) for single-process
   training, which is the common case.
"""

from __future__ import annotations

import os

__all__ = ["is_rank_zero"]


def is_rank_zero() -> bool:
    """Return ``True`` if the current process should perform Aim writes.

    Returns
    -------
    bool
        ``True`` for the rank-zero process (or for single-process
        training where no rank information is available). ``False`` for
        any non-zero rank in a distributed run.
    """
    # torch.distributed first — it's the source of truth once initialized.
    # Wrapped in try/except because torch is not a hard dependency of
    # astrolabe-callbacks; users on JAX or non-distributed PyTorch don't
    # need it.
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank() == 0
    except ImportError:
        pass

    # Env var fallback — torchrun / cluster launchers set RANK before
    # Python starts, so this works even if the process group hasn't been
    # initialized yet (e.g. callback `init` fires before `dist.init`).
    rank = os.environ.get("RANK")
    if rank is not None:
        return rank == "0"

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        return local_rank == "0"

    return True
