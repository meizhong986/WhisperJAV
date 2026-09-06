"""Offline mode for Hugging Face model loading (#415).

Two things live here, both light (no ML imports at module level):

1. ``enable_offline_mode()`` — sets ``HF_HUB_OFFLINE=1`` so huggingface_hub and
   transformers use only the local cache and refuse every request before a socket
   is opened. huggingface_hub reads the variable when it is imported
   (``huggingface_hub/constants.py``), so the entry points call this from a raw
   ``sys.argv`` scan BEFORE any import that pulls the hub library in
   (``whisperjav/main.py``, ``whisperjav/cli.py``). Child processes — ensemble pass
   workers and the Balanced recogniser worker — are spawned with the parent's
   environment and therefore inherit it.

2. ``load_cached_first()`` — call a ``from_pretrained`` loader against the local
   cache first and contact the hub only when a file is genuinely missing. Without
   this, every ``from_pretrained`` performs a per-file freshness check against
   huggingface.co, and when the host is unreachable each file costs five retries
   with backoff (minutes per file). Used for the loaders on the ensemble path
   (WhisperSeg's feature extractor, anime-whisper's processor and model).

Scope, stated plainly: this covers loads that go through huggingface_hub.
Silero via torch.hub (#263), openai-whisper weights, ModelScope enhancers and
NeMo configs have their own download paths and are not affected.
"""
from __future__ import annotations

import os

OFFLINE_FLAG = "--offline"
_HF_OFFLINE_ENV = "HF_HUB_OFFLINE"
# huggingface_hub also honours TRANSFORMERS_OFFLINE (constants.py: HF_HUB_OFFLINE = _is_true(
# HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE)); read both so this module agrees with the hub.
_HF_OFFLINE_ENVS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def offline_requested(argv) -> bool:
    """True when the raw argument list carries ``--offline``.

    argparse accepts any unambiguous prefix of a long option (``--offl``), and it
    parses long after this scan runs, so a prefix of at least ``--off`` counts too.
    No other WhisperJAV option starts with ``--off``; an ambiguous prefix would make
    argparse exit with an error anyway.
    """
    for tok in argv:
        if tok == OFFLINE_FLAG:
            return True
        if tok.startswith("--") and len(tok) >= 5 and OFFLINE_FLAG.startswith(tok):
            return True
    return False


def enable_offline_mode() -> None:
    """Switch this process (and every child it spawns) to downloaded models only."""
    os.environ[_HF_OFFLINE_ENV] = "1"


def hub_offline() -> bool:
    """True when huggingface_hub is in offline mode, by either variable it honours."""
    return any(
        os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
        for name in _HF_OFFLINE_ENVS
    )


# One definition: offline mode for WhisperJAV *is* the hub's offline mode.
is_offline = hub_offline


def _hub_cache_dir() -> str:
    try:
        from huggingface_hub import constants
        return str(constants.HF_HUB_CACHE)
    except Exception:  # noqa: BLE001 - diagnostics only
        return "the Hugging Face cache"


def load_cached_first(loader, model_id, **kwargs):
    """Call ``loader.from_pretrained(model_id, **kwargs)`` from the local cache first.

    Only a cache miss (``OSError`` — huggingface_hub's ``LocalEntryNotFoundError`` is a
    ``FileNotFoundError``, and transformers wraps misses in ``OSError``) triggers the
    fallback to a normal load, which downloads. Any other failure (corrupt weights,
    memory, dtype) propagates from the first attempt unchanged. Under offline mode the
    fallback would fail the same way, so the miss is raised with a message that names
    the model and the cache instead of the library's "check your internet connection".
    """
    try:
        return loader.from_pretrained(model_id, local_files_only=True, **kwargs)
    except OSError as e:
        if hub_offline():
            raise OSError(
                f"Offline mode: '{model_id}' is not in the local Hugging Face cache "
                f"({_hub_cache_dir()}). Download it once while online, or run without "
                f"--offline / untick Offline mode."
            ) from e
        return loader.from_pretrained(model_id, **kwargs)
