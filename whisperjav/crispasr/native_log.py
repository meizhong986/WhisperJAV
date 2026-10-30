"""CrispASR's own log lines on the console (owner W349, W350).

CrispASR's GPU library (ggml) writes its messages to the console through its default logger. With an NVIDIA GPU
it writes two lines per model call ("ggml_backend_cuda_graph_compute: CUDA graph warmup complete" / "... reset"),
at its DEBUG level: in a four-file Ensemble run of cohere and parakeet, 35,520 of the console's 39,479 lines
(owner's GUI test, 2026-10-29). WhisperJAV registers its own handler through ggml's ``ggml_log_set`` (exported by
the ``ggml-base`` library in the crispasr package): without --debug it drops ggml's DEBUG messages and passes the
others (INFO, WARN, ERROR) to the console unchanged; with --debug it passes every message, as before.

Lines CrispASR writes straight to the error output do not pass through ggml's logger and are not touched: the
aligner's "crispasr[aligner]: WARNING: alignment collapse suspected ..." and "[wav2vec2] vocab=..." (probe of
2026-10-29). Warnings stay on the console (W350).

``quiet_debug_lines()`` is called after each ``import crispasr`` (engine, aligner, timing model). It registers the
handler once per process and never raises: when the library or the symbol is not found, ggml's own logger stays.
"""

import ctypes
import sys
from pathlib import Path
from typing import Optional

from whisperjav.utils.logger import logger

# ggml's log levels (ggml.h, enum ggml_log_level). CONT continues the message before it.
GGML_LOG_LEVEL_DEBUG = 1
GGML_LOG_LEVEL_CONT = 5

_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_handler = None          # kept here: ggml holds only a pointer, so the ctypes object must not be collected
_last_level = 0


def should_show(level: int, last_level: int, debug: bool) -> bool:
    """Whether a ggml message reaches the console: everything with --debug; otherwise not a DEBUG message, nor
    the continuation (CONT) of one."""
    if debug:
        return True
    if level == GGML_LOG_LEVEL_CONT:
        return last_level != GGML_LOG_LEVEL_DEBUG
    return level != GGML_LOG_LEVEL_DEBUG


def _ggml_base(package_dir: Path) -> Optional[Path]:
    for pattern in ("ggml-base.dll", "libggml-base.so*", "libggml-base.dylib", "*ggml-base*"):
        found = sorted(package_dir.glob(pattern))
        if found:
            return found[0]
    return None


def quiet_debug_lines(crispasr_module) -> bool:
    """Register WhisperJAV's ggml log handler once; True when it is registered."""
    global _handler
    if _handler is not None:
        return True
    try:
        package_dir = Path(crispasr_module.__file__).parent
        path = _ggml_base(package_dir)
        if path is None:
            logger.debug("CrispASR log: no ggml-base library in %s; CrispASR's own log lines are unchanged",
                         package_dir)
            return False
        lib = ctypes.CDLL(str(path))
        set_log = lib.ggml_log_set
    except (OSError, AttributeError, TypeError) as e:
        logger.debug("CrispASR log: the ggml log switch could not be reached (%s); CrispASR's own log lines "
                     "are unchanged", e)
        return False

    def handler(level, text, _user_data):
        global _last_level
        try:
            if should_show(level, _last_level, logger.isEnabledFor(10)):   # 10 = logging.DEBUG (--debug)
                sys.stderr.write((text or b"").decode("utf-8", "replace"))
                sys.stderr.flush()
            if level != GGML_LOG_LEVEL_CONT:
                _last_level = level
        except Exception:
            pass   # a log line must never break a run

    set_log.argtypes = [_CALLBACK, ctypes.c_void_p]
    set_log.restype = None
    _handler = _CALLBACK(handler)
    set_log(_handler, None)
    return True
