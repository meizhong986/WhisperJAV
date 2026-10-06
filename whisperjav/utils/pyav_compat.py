"""Keep faster-whisper working with PyAV 19 and later (private report PS-001, 2026-10-04).

PyAV 19.0.0 (2026-09-29) removed the ``metadata_encoding`` and ``metadata_errors`` arguments of ``av.open``.
faster-whisper (1.2.x, faster_whisper/audio.py) still calls ``av.open(file, mode="r", metadata_errors="ignore")``,
so wherever it decodes a file itself a fresh install fails with
``TypeError: open() got an unexpected keyword argument 'metadata_errors'``.

1.9.4 changes no dependency (a version cap would not reach users who upgrade with --wheel-only), so this wraps
``av.open``: when the installed PyAV rejects those two arguments, the call is repeated without them. With an older
PyAV the first call succeeds and nothing changes. Safe to call more than once; does nothing when PyAV is missing.
"""
import functools
import logging

logger = logging.getLogger("whisperjav")

_REMOVED_IN_PYAV_19 = ("metadata_encoding", "metadata_errors")


def ensure_av_open_compat() -> bool:
    """Wrap av.open once. Returns True when the wrapper is in place."""
    try:
        import av
    except Exception:
        return False
    if getattr(av.open, "_whisperjav_compat", False):
        return True
    original = av.open

    @functools.wraps(original)
    def open_compat(*args, **kwargs):
        try:
            return original(*args, **kwargs)
        except TypeError as err:
            dropped = [k for k in _REMOVED_IN_PYAV_19 if k in kwargs and k in str(err)]
            if not dropped:
                raise
            for k in _REMOVED_IN_PYAV_19:
                kwargs.pop(k, None)
            logger.debug("PyAV %s does not take %s; opening without them",
                         getattr(av, "__version__", "?"), ", ".join(dropped))
            return original(*args, **kwargs)

    open_compat._whisperjav_compat = True
    av.open = open_compat
    return True
