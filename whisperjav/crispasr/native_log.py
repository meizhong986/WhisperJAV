"""CrispASR's own log lines on the console (owner W349, W350; W509 console item 6).

CrispASR's library writes its messages to the console through its logger. With an NVIDIA GPU it writes two lines per
model call ("ggml_backend_cuda_graph_compute: CUDA graph warmup complete" / "... reset") at its DEBUG level: in a
four-file Ensemble run of cohere and parakeet, 35,520 of the console's 39,479 lines (owner's GUI test, 2026-10-29).
At its INFO level it writes start-up details: "ggml_cuda_init: found 1 CUDA devices ..." with the device line at each
model load, and 36 "whisper_init_* / whisper_model_load / whisper_backend_init_gpu / whisper_init_state" lines when
the timing model loads (owner's log of 2026-11-03).

WhisperJAV registers its own handler: through ``whisper_log_set`` of the binding's library (crispasr.dll /
libcrispasr), which sets whisper.cpp's logger and ggml's together, so that loading the timing model (whose
``whisper_backend_init_gpu`` calls ``ggml_log_set`` with whisper.cpp's logger, crispasr_v0.8.50.cpp:1806) keeps it;
else through ggml's ``ggml_log_set`` (the ``ggml-base`` library). Without --debug it keeps the DEBUG and INFO messages
off the console and passes the warnings and errors to it unchanged; with --debug it passes every message, as before.
Two INFO lines stay on the console in any case: "no GPU found" (the timing model then runs on the CPU) and "DECODER
FAILED" (review F5). The other INFO lines go to the log file when there is one (owner W509, way B); the DEBUG lines are
not recorded.

Lines CrispASR writes straight to the error output do not pass through either logger and are not touched: the
aligner's "crispasr[aligner]: WARNING: alignment collapse suspected ...", "[wav2vec2] vocab=..." and
"moss_diarize: loaded ..." (probe of 2026-10-29; W509 item 6). Warnings stay on the console (W350).

``quiet_debug_lines()`` is called after each ``import crispasr`` (engine, aligner, timing model). It registers the
handler once per process and never raises: when the library or the symbol is not found, CrispASR's own logger stays.
"""

import ctypes
import logging
import sys
from pathlib import Path
from typing import Optional

from whisperjav.utils.logger import console_shows_debug, logger

# ggml's log levels (ggml.h, enum ggml_log_level). CONT continues the message before it.
GGML_LOG_LEVEL_DEBUG = 1
GGML_LOG_LEVEL_INFO = 2
GGML_LOG_LEVEL_CONT = 5
_HIDDEN = (GGML_LOG_LEVEL_DEBUG, GGML_LOG_LEVEL_INFO)

_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p)
_handler = None          # kept here: the library holds only a pointer, so the ctypes object must not be collected
_last_level = 0
_last_shown = True
_pending = []            # the start of an INFO line whose end has not come yet (CrispASR writes some in parts)

# CrispASR INFO lines that stay on the console without --debug (review F5): the timing model found no GPU and runs on
# the CPU (whisper_backend_init_gpu, crispasr_v0.8.50.cpp:1841), and a failed Whisper decode (:10416).
ALWAYS_SHOWN = ("no GPU found", "DECODER FAILED")


def should_show(level: int, last_level: int, debug: bool) -> bool:
    """Whether a CrispASR message reaches the console: everything with --debug; otherwise a warning or an error
    (not a DEBUG or INFO message, nor the continuation (CONT) of one)."""
    if debug:
        return True
    if level == GGML_LOG_LEVEL_CONT:
        return last_level not in _HIDDEN
    return level not in _HIDDEN


def always_shown(message: str) -> bool:
    """An INFO line that stays on the console without --debug (ALWAYS_SHOWN)."""
    return any(part in message for part in ALWAYS_SHOWN)


def to_log_file(message: str) -> None:
    """A CrispASR INFO line goes to the log file, when there is one (review F5; owner W509, way B): one DEBUG record
    of WhisperJAV's logger per whole line, handed to the file handlers only -- the console has the line already
    (--debug, or one of ALWAYS_SHOWN) or must not get it. CrispASR's DEBUG lines (two per model call on an NVIDIA GPU)
    are not recorded."""
    handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    if not handlers:
        _pending.clear()
        return
    _pending.append(message)
    if "\n" not in message:
        return
    lines = "".join(_pending).split("\n")
    tail = lines.pop()   # what follows the last line end: the start of the next line, or ""
    _pending[:] = [tail] if tail else []
    for line in lines:
        if line.strip():
            record = logger.makeRecord(logger.name, logging.DEBUG, __file__, 0, "CrispASR: %s", (line.rstrip(),),
                                       None)
            for h in handlers:
                if record.levelno >= h.level:
                    h.handle(record)


def _ggml_base(package_dir: Path) -> Optional[Path]:
    for pattern in ("ggml-base.dll", "libggml-base.so*", "libggml-base.dylib", "*ggml-base*"):
        found = sorted(package_dir.glob(pattern))
        if found:
            return found[0]
    return None


def _set_log_function(crispasr_module):
    """The library function that takes the handler: whisper_log_set of the binding's own library first (it covers
    whisper.cpp's logger and ggml's), else ggml_log_set of ggml-base. None when neither is found."""
    try:
        find_lib = crispasr_module._binding._find_lib
        lib = ctypes.CDLL(str(find_lib()))
        return lib.whisper_log_set
    except (OSError, AttributeError, TypeError) as e:
        logger.debug("CrispASR log: whisper_log_set not reached (%s); trying ggml_log_set", e)
    package_dir = Path(crispasr_module.__file__).parent
    path = _ggml_base(package_dir)
    if path is None:
        logger.debug("CrispASR log: no ggml-base library in %s; CrispASR's own log lines are unchanged", package_dir)
        return None
    try:
        return ctypes.CDLL(str(path)).ggml_log_set
    except (OSError, AttributeError, TypeError) as e:
        logger.debug("CrispASR log: the ggml log switch could not be reached (%s); CrispASR's own log lines "
                     "are unchanged", e)
        return None


def quiet_debug_lines(crispasr_module) -> bool:
    """Register WhisperJAV's CrispASR log handler once; True when it is registered."""
    global _handler
    if _handler is not None:
        return True
    try:
        set_log = _set_log_function(crispasr_module)
    except Exception as e:   # a log switch must never stop a run
        logger.debug("CrispASR log: %s; CrispASR's own log lines are unchanged", e)
        return False
    if set_log is None:
        return False

    def handler(level, text, _user_data):
        global _last_level, _last_shown
        try:
            message = (text or b"").decode("utf-8", "replace")
            if level == GGML_LOG_LEVEL_CONT:
                shown = _last_shown or console_shows_debug()
            else:
                shown = (should_show(level, _last_level, console_shows_debug())   # --debug on the console
                         or always_shown(message))
                _last_level, _last_shown = level, shown
            if shown:
                sys.stderr.write(message)
                sys.stderr.flush()
            if _last_level == GGML_LOG_LEVEL_INFO:   # an INFO line or its continuation: to the log file too
                to_log_file(message)
        except Exception:
            pass   # a log line must never break a run

    set_log.argtypes = [_CALLBACK, ctypes.c_void_p]
    set_log.restype = None
    _handler = _CALLBACK(handler)
    set_log(_handler, None)
    return True
