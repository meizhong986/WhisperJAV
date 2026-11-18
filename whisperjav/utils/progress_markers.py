"""Stdout progress markers for multi-file transcription/ensemble runs.

Producers (main.py, ensemble orchestrator/worker) print one marker line per file;
the webview GUI consumer (api.py) parses it to drive the live progress indicator.
Kept in a single module so the producer string and the parser cannot drift.
"""

import os
import re
import sys
from typing import Any

# The GUI sets this to "1" in the environment of the runs it starts.
# Markers are printed only then, so command-line output stays as it was.
GUI_MARKERS_ENV = "WHISPERJAV_GUI_MARKERS"


def emit_marker(line: str) -> None:
    """Print a marker line to the original stderr, only for runs the GUI started.

    The original stderr bypasses any stdout/stderr capture in the pipelines.

    Args:
        line: The marker line to print.
    """
    if os.environ.get(GUI_MARKERS_ENV) != "1":
        return
    try:
        print(line, file=sys.__stderr__ or sys.stderr, flush=True)
    except (ValueError, OSError):
        pass

# Matches both:
#   "Transcribing [4/20]: name.mp4"
#   "Transcribing [pass 1/2] [4/20]: name.mp4"
_TRANSCRIBE_RE = re.compile(r"Transcribing(?:\s+\[pass\s+(\d+)/(\d+)\])?\s+\[(\d+)/(\d+)\]:\s+(.+)")


def transcribe_marker(
    name: str, current: int, total: int, pass_number: int | None = None, pass_total: int | None = None
) -> str:
    """Build the stdout marker line for a file about to be transcribed.

    A pass label is included only when ``pass_total`` is provided and > 1, so
    single-pass runs read as plain transcription.
    """
    if pass_total is not None and pass_total > 1:
        return f"Transcribing [pass {pass_number}/{pass_total}] [{current}/{total}]: {name}"
    return f"Transcribing [{current}/{total}]: {name}"


def parse_transcribe_marker(line: str) -> dict[str, Any] | None:
    """Parse a marker line; return None when the line is not a marker."""
    m = _TRANSCRIBE_RE.match(line.lstrip())
    if not m:
        return None
    return {
        "pass_number": int(m.group(1)) if m.group(1) else None,
        "pass_total": int(m.group(2)) if m.group(2) else None,
        "current": int(m.group(3)),
        "total": int(m.group(4)),
        "name": m.group(5).strip(),
    }


# Overall-progress marker: a 0-100 percent blending file index and scene fraction.
# It keeps the GUI bar moving during one long file.
#   "Transcribing-progress [67]"
_PROGRESS_RE = re.compile(r"Transcribing-progress \[(\d+)\]")


def progress_marker(percent: int) -> str:
    """Build the overall-progress marker line for the given 0-100 percent."""
    return f"Transcribing-progress [{percent}]"


def parse_progress_marker(line: str) -> int | None:
    """Parse an overall-progress marker; return the int percent, or None."""
    m = _PROGRESS_RE.match(line.lstrip())
    return int(m.group(1)) if m else None
