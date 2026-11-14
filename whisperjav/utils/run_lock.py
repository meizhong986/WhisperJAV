"""One run per working folder (1.10.2, owner W647 2).

A run keeps its extracted audio, its scene files, each pass's result file and its unfinished subtitles in the
working folder (--temp-dir; by default the folder "whisperjav" in the system's temporary folder), under names
made from the media file's name alone, and it empties the folder when a file or the run ends. Two runs in
one working folder therefore delete and overwrite each other's files: in the owner's test of 2026-11-14 two
runs of one film left a pass 2 that stopped at minute 32 of 118 behind a "done" run summary.

claim_working_folder() is called by main.py once the working folder is known and before anything is written
into it. The claim is a lock the operating system holds on a small file for as long as this process lives,
so it is given up when the run ends in any way (a normal end, an error, a kill); no leftover file can refuse
a later run. The lock file is kept in the system's temporary folder, outside the working folder, because the
working folder is emptied at the end of a run.
"""
import atexit
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple

# The text main.py logs when the working folder is in use ({folder}: the working folder; {pid_part}: who
# holds it, when that could be read). main.py then exits with status 1, as it does for the working folder's
# other start-up refusal (temp_dir_conflicts).
IN_USE_FACT = (
    "WhisperJAV has not started, because another WhisperJAV run{pid_part} is using the working folder "
    "{folder}. Two runs in one working folder delete each other's files."
)
IN_USE_ADVICE = "Wait for that run to end, or give this run its own working folder with --temp-dir."
IN_USE_MESSAGE = IN_USE_FACT + " " + IN_USE_ADVICE
# The same advice for the GUI's dialog, which names the GUI's control instead of the flag (owner W648 2).
IN_USE_ADVICE_GUI = ("Wait for that run to end, or give this run its own working folder in Temp dir "
                     "(Advanced Options tab).")
_IN_USE_START = IN_USE_FACT.split("{")[0]


def gui_refusal_text(output_lines) -> Optional[str]:
    """The refusal a run printed, reworded for the GUI's "The run did not start" dialog, or None when the run
    printed none. ``output_lines``: the last lines of the run's output."""
    for line in reversed(list(output_lines)):
        at = line.find(_IN_USE_START)
        if at < 0:
            continue
        text = line[at:].strip()
        cut = text.find(IN_USE_ADVICE)
        if cut >= 0:
            text = text[:cut].rstrip()
        return f"{text} {IN_USE_ADVICE_GUI}"
    return None

# The byte that is locked. It lies far past the end of the file, so that the holder's process id, written at
# the start of the file, stays readable to a run that is refused (Windows refuses reads of a locked byte).
_LOCK_OFFSET = 1 << 30

# The claims this process holds: lock file path -> open file. Kept open for the life of the process.
_held: Dict[str, object] = {}


class WorkingFolderInUse(Exception):
    """Another WhisperJAV run holds the working folder."""

    def __init__(self, folder: str, pid: Optional[int]):
        self.folder = folder
        self.pid = pid
        super().__init__(in_use_message(folder, pid))


def in_use_message(folder: str, pid: Optional[int]) -> str:
    return IN_USE_MESSAGE.format(folder=folder, pid_part=f" (process {pid})" if pid else "")


def lock_file_for(temp_dir: str) -> Path:
    """The lock file of a working folder: one name per folder, whatever way the folder's path is written."""
    canonical = os.path.normcase(os.path.realpath(str(temp_dir)))
    digest = hashlib.sha256(canonical.encode("utf-8", "surrogatepass")).hexdigest()[:16]
    return _lock_home() / f"whisperjav_run_{digest}.lock"


def _lock_home() -> Path:
    """The folder the lock files are kept in (the tests point it at a folder of their own)."""
    return Path(tempfile.gettempdir())


def _give_up(key: str) -> None:
    """At the normal end of the process: close the lock file and, on Windows, remove it. Windows cannot remove
    a file another process has open, so a run that has just claimed it keeps it; elsewhere the file is left in
    place, because removing it there could let two later runs each hold a different file of the same name."""
    handle = _held.pop(key, None)
    if handle is None:
        return
    try:
        handle.close()
        if sys.platform == "win32":
            os.unlink(key)
    except OSError:
        pass


def _try_lock(handle) -> bool:
    """Lock one byte of the open file without waiting; False when another process holds it."""
    if sys.platform == "win32":
        import msvcrt
        handle.seek(_LOCK_OFFSET)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl
    try:
        # flock, not lockf: it belongs to this open file, so it is not given up when some other part of the
        # process opens and closes the same file.
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _holder_pid(lock_file: Path) -> Optional[int]:
    try:
        text = lock_file.read_text(encoding="ascii", errors="replace").split()
        return int(text[0]) if text else None
    except (OSError, ValueError):
        return None


def claim_working_folder(temp_dir: str) -> Tuple[Path, bool]:
    """Claim the working folder for this process until it ends.

    Returns (lock file, True) when the folder is now claimed, and (lock file, False) when the lock file could
    not be opened at all (a read-only temporary folder): the run then goes on unclaimed, as before 1.10.2.
    Raises WorkingFolderInUse when another process holds the folder. A second call for the same folder from
    this process is answered at once.
    """
    lock_file = lock_file_for(temp_dir)
    key = str(lock_file)
    if key in _held:
        return lock_file, True
    try:
        handle = open(lock_file, "a+b")
    except OSError:
        return lock_file, False
    if not _try_lock(handle):
        handle.close()
        raise WorkingFolderInUse(str(temp_dir), _holder_pid(lock_file))
    try:
        handle.seek(0)
        handle.truncate(0)
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.flush()
    except OSError:
        pass  # the claim holds; only the "who" of a refusal is lost
    _held[key] = handle
    atexit.register(_give_up, key)
    return lock_file, True
