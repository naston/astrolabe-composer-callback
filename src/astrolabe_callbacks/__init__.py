"""astrolabe-callbacks — framework-agnostic Aim callbacks for ML training.

Public API (pick the one that matches your framework)::

    from astrolabe_callbacks import AstrolabeComposerLogger    # MosaicML Composer
    from astrolabe_callbacks import AstrolabeLightningLogger   # PyTorch Lightning
    from astrolabe_callbacks import AstrolabeHFTrainerCallback # HuggingFace Trainer
    from astrolabe_callbacks import Run                        # raw PyTorch / JAX / custom loops

Each per-framework class is imported lazily — `import astrolabe_callbacks`
only needs `aim` and `loguru`. Framework dependencies are pulled in on
first reference, surfacing a clear `ImportError` if the matching extras
aren't installed::

    pip install astrolabe-callbacks[composer]
    pip install astrolabe-callbacks[lightning]
    pip install astrolabe-callbacks[hf]
    pip install astrolabe-callbacks[all]
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = [
    "AstrolabeComposerLogger",
    "AstrolabeLightningLogger",
    "AstrolabeHFTrainerCallback",
    "AstrolabeRun",
    "Run",
    "__version__",
]


# PEP 562 module-level __getattr__ defers framework imports until a class
# is actually referenced. Without this, `import astrolabe_callbacks` would
# pull in Composer/Lightning/Transformers eagerly and the base install
# (aim only) would fail.
def __getattr__(name: str):
    if name == "AstrolabeComposerLogger":
        from astrolabe_callbacks.composer import AstrolabeComposerLogger
        return AstrolabeComposerLogger
    if name == "AstrolabeLightningLogger":
        from astrolabe_callbacks.lightning import AstrolabeLightningLogger
        return AstrolabeLightningLogger
    if name == "AstrolabeHFTrainerCallback":
        from astrolabe_callbacks.huggingface import AstrolabeHFTrainerCallback
        return AstrolabeHFTrainerCallback
    if name in ("AstrolabeRun", "Run"):
        from astrolabe_callbacks.pytorch import AstrolabeRun
        return AstrolabeRun
    raise AttributeError(f"module 'astrolabe_callbacks' has no attribute {name!r}")
