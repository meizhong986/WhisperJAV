#!/usr/bin/env python3
"""Simple logging setup for WhisperJAV."""

import logging
import sys
from pathlib import Path
from typing import Optional

try:  # pragma: no cover - best-effort import for color support
    from colorama import Fore, Style, init as colorama_init

    colorama_init()  # ensure ANSI works on Windows terminals
    _USE_COLOR = True
    _COLOR_MAP = {
        "blue": Fore.BLUE,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "red": Fore.RED,
        "cyan": Fore.CYAN,
        "magenta": Fore.MAGENTA,
    }
    _COLOR_RESET = Style.RESET_ALL
except Exception:  # pragma: no cover - color fallback when colorama missing
    _USE_COLOR = sys.stdout.isatty()
    _COLOR_MAP = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
    }
    _COLOR_RESET = "\033[0m"


def color_text(message: str, color: str) -> str:
    """Return message wrapped with ANSI color codes when supported."""

    if not _USE_COLOR:
        return message

    prefix = _COLOR_MAP.get(color)
    if not prefix:
        return message

    return f"{prefix}{message}{_COLOR_RESET}"


class ColorFormatter(logging.Formatter):
    """Formatter that applies ANSI colors based on record metadata."""

    def __init__(self, *args, **kwargs):
        self.enable_color = kwargs.pop("enable_color", True)
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.enable_color:
            return formatted

        color = getattr(record, "color", None)
        if not color:
            return formatted

        return color_text(formatted, color)


class _KnownLinesFilter(logging.Filter):
    """Drops a library's known, harmless warning by the start of its text (owner W509, console item 4)."""

    def __init__(self, starts):
        super().__init__()
        self.starts = tuple(starts)

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            return not str(record.getMessage()).lstrip().startswith(self.starts)
        except Exception:
            return True


# Library warnings that say nothing a user can act on (owner W509, console item 4: filtered by their text, not by
# silencing the loggers; every other warning of these loggers still shows). A filter on a logger sees only the records
# made on that logger, so each line is keyed by the logger that writes it.
# - torch's and three transformers lines, on every Qwen3-ASR load; the generation-flags line only for the flag list
#   Qwen3-ASR's generation settings give ('temperature' without sampling), so the same warning for other flags shows.
# - FireRedVAD's "Unexpected short speech segment" (11 in his log of 2026-11-03), not its other warnings such as
#   "Too long input, split every N frames".
_KNOWN_LIBRARY_LINES = {
    "torch.distributed.elastic.multiprocessing.redirects": ("NOTE: Redirects are currently not supported",),
    "transformers.modeling_rope_utils": ("Unrecognized keys in `rope_parameters`",),
    "transformers.generation.configuration_utils": (
        "The following generation flags are not valid and may be ignored: ['temperature'].",),
    "transformers.processing_utils": ("`Qwen3ASRProcessor` defines `feature_extractor_class",),
    "fireredvad.core.vad_postprocessor": ("Unexpected short speech segment, check vad_postprocessor.py",),
}
_quiet_state = {"filters": {}}


def quiet_library_loggers(debug: bool) -> None:
    """Without --debug, libraries' INFO lines and their known harmless warnings stay off the console (owner W509,
    console item 4; with --debug every line shows, as W350 chose). Called by a transcription run (main.py) and by
    each Ensemble pass worker after setup_logger -- not at import, because other entry points (the translate
    command, the updater, bench) set up the root logger themselves with logging.basicConfig().

    - httpx and httpcore: WARNING (their "HTTP Request: ..." lines are INFO).
    - The known lines above: filtered by their text.
    The root logger is left as it is; the one library known to give it a handler at INFO (nagisa, imported with
    Qwen3-ASR) is undone where it is imported: ``keep_root_logger()``.
    """
    if debug:
        for name, flt in _quiet_state["filters"].items():
            logging.getLogger(name).removeFilter(flt)
        _quiet_state["filters"] = {}
        return
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    for name, starts in _KNOWN_LIBRARY_LINES.items():
        if name not in _quiet_state["filters"]:
            flt = _KnownLinesFilter(starts)
            logging.getLogger(name).addFilter(flt)
            _quiet_state["filters"][name] = flt


class keep_root_logger:
    """Undo a library's logging.basicConfig() made while importing it (owner W509, console item 4; review F3).

    nagisa, imported with Qwen3-ASR (qwen_asr/inference/qwen3_forced_aligner.py), runs
    logging.basicConfig(level=logging.INFO) at import (nagisa/train.py): the root logger then gets a handler at INFO,
    so every library's INFO lines print (httpx's "HTTP Request: ..." on each model load) and every warning of a
    library with its own handler (huggingface_hub) prints twice. Used around that import: without --debug the handlers
    the import added to the root logger are removed and its level is set back; with --debug nothing is changed (every
    line shows, W350).

        with keep_root_logger():
            from qwen_asr import Qwen3ASRModel
    """

    def __enter__(self):
        root = logging.getLogger()
        self._handlers = list(root.handlers)
        self._level = root.level
        return self

    def __exit__(self, *exc):
        try:
            if console_shows_debug():
                return False
            root = logging.getLogger()
            for handler in list(root.handlers):
                if handler not in self._handlers:
                    root.removeHandler(handler)
            root.setLevel(self._level)
        except Exception:   # a log switch must never stop a run
            pass
        return False


def _console_handler(name: str = "whisperjav") -> Optional[logging.Handler]:
    for handler in logging.getLogger(name).handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            return handler
    return None


def console_level_name(name: str = "whisperjav") -> str:
    """The console's level as a name ("INFO", "DEBUG", ...), for a spawned process that sets up its own console
    (review F2: with a log file the logger itself is at DEBUG, so its level is not the console's)."""
    lg = logging.getLogger(name)
    handler = _console_handler(name)
    level = (handler.level if handler is not None and handler.level else lg.getEffectiveLevel()) or logging.INFO
    return logging.getLevelName(level)


def console_shows_debug(name: str = "whisperjav") -> bool:
    """Whether the console shows DEBUG lines (--debug or --log-level DEBUG). A log file records them in any case
    (owner W509, way B), so the logger's own level no longer tells; code that shows more on the console with --debug
    asks this."""
    lg = logging.getLogger(name)
    handler = _console_handler(name)
    if handler is not None:
        return (handler.level or lg.getEffectiveLevel()) <= logging.DEBUG
    return lg.isEnabledFor(logging.DEBUG)


def has_log_file(name: str = "whisperjav") -> bool:
    """Whether this process writes a log file (--log-file, or an Ensemble pass's own file)."""
    return any(isinstance(h, logging.FileHandler) for h in logging.getLogger(name).handlers)


def hide_weight_loading_bar() -> None:
    """transformers' "Loading weights" bar only with --debug (owner W509, console item 3): the GUI shows each redraw
    of a bar as a line (32 per model load). Done through transformers' tqdm hook, so the download bars of a first
    use stay (transformers' disable_progress_bar() would also hide them). Call before loading a transformers model."""
    if console_shows_debug():
        return
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        return
    if not hasattr(hf_logging, "set_tqdm_hook") or not hasattr(hf_logging, "EmptyTqdm"):
        return

    def hook(factory, args, kwargs):
        if kwargs.get("desc") == "Loading weights":
            return hf_logging.EmptyTqdm(*args, **kwargs)
        return factory(*args, **kwargs)

    hf_logging.set_tqdm_hook(hook)


def pass_log_file(log_file: Optional[str], pass_number: int) -> Optional[str]:
    """The log file of an Ensemble pass, beside the run's own (owner W509, way B): run.log -> run.pass1.log."""
    if not log_file:
        return None
    path = Path(log_file)
    return str(path.with_name(f"{path.stem}.pass{pass_number}{path.suffix or '.log'}"))


def setup_logger(name: str = "whisperjav",
                log_level: str = "INFO",
                log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with console and optional file output.

    The console shows log_level and above. The log file records every line of WhisperJAV's logger, the DEBUG lines
    included, whatever log_level is (owner W509, way B), so a run without --debug keeps a full record on disk while
    the console stays short.
    """

    level = getattr(logging, log_level.upper())
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers and prevent propagation to root logger.
    # Without this, messages duplicate when any dependency triggers
    # logging.basicConfig() — the root handler re-emits our messages
    # through stderr, doubling subprocess pipe traffic and risking
    # pipe buffer deadlocks on Windows GUI subprocesses.
    logger.handlers = []
    logger.propagate = False

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = ColorFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        enable_color=_USE_COLOR,
    )
    console_handler.setFormatter(console_format)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (optional). A file that cannot be opened (a folder without write access, a bad path) no longer
    # stops the run before it starts: one warning, and the run goes on without a log file (the GUI passes one for
    # every run, owner W511 2).
    if log_file:
        log_path = Path(log_file)
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
        except OSError as e:
            logger.warning("The log file %s could not be opened (%s); this run keeps no log file.", log_path, e)
            return logger
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.setLevel(logging.DEBUG)   # the file records the DEBUG lines too (way B); the console keeps `level`

    return logger


# Global logger instance
logger = setup_logger()