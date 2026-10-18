"""Starting CrispASR's program for one film and judging its result (master plan 4.1, 4.3).

Shared by every CrispASR pipeline (Q20 = A). The base is the second architecture study's blueprint
(5.3, 5.4); the changes are the plan's:
- the program runs under Popen with a reader thread per stream and ``-pp``, so its per-slice
  progress line ("crispasr: progress = N% (i/n slices)", stderr) can move WhisperJAV's progress
  display. The transcript itself reaches stdout only after every slice is done;
- it runs in the film's ``crispasr_out`` folder and is given only short ASCII names there
  (plan 1.7: its model loader opens files in the PC's code page);
- nothing typed by the user is passed through: the old Extra Args box is gone;
- the program is ended whenever WhisperJAV stops or fails, and on Windows it is tied to
  WhisperJAV's process, so it also ends if WhisperJAV itself is killed. There is no time limit:
  a CPU run on a long film can take hours;
- the program never downloads a model by itself (master plan 3.8 item 6; W130, W133 3, U19). It has
  no flag that forbids it: when the -m file cannot be opened, it uses a same-named file from its own
  cache folder, or downloads one if --auto-download is given or if its input is a terminal -- and on
  Windows it takes the "nothing" input (DEVNULL) for a terminal, asks "Download now? [Y/n]" and takes
  no answer as yes. So its input is a pipe closed at once, and --cache-dir names an empty folder in
  the run folder; WhisperJAV never passes --auto-download. A missing model then ends the run with
  exit code 13;
- ``run.log`` keeps the command line, the working folder, the exit code and the full stderr, so
  every run can be repeated outside WhisperJAV.

Imports no ASR module.
"""

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from whisperjav.crispasr.ojf import read_ojf
from whisperjav.utils.logger import logger

# Exit codes on the route WhisperJAV uses (--backend given, so CrispASR's unified dispatch,
# cli.cpp:2842-2845). Code 4 ("could not be decoded", docs/cli.md:331-340) belongs to the legacy
# whisper route only (cli.cpp:3354-3357).
EXIT_MEANING = {
    1: "bad --vad-import or response file, or bad arguments",
    2: "no input file left, or a config error",
    10: "--align-only error (no aligner, no text, no audio, or alignment failed)",
    11: "model could not be resolved",
    12: "back end not available in this build",
    13: "back end failed to initialise",
    14: "language not supported by this back end, or a worker failed to start",
    19: "-am auto could not be resolved",
    20: "input audio could not be read",
    30: "required VAD model failed (--strict-pipeline)",
    31: "required word timestamps missing (--strict-pipeline)",
    32: "required punctuation model failed (--strict-pipeline)",
}

# Windows reports a crash as an exception status (NTSTATUS >= 0xC0000000, e.g. 0xC0000409 in run A5);
# POSIX reports a signal as a negative return code.
_CRASH_STATUS_MIN = 0xC0000000
_KNOWN_CRASH_STATUS = {
    0xC0000005: "access violation",
    0xC000001D: "illegal instruction",
    0xC0000135: "a DLL the program needs was not found",
    0xC0000409: "stack buffer overrun (fail-fast)",
    0xC00000FD: "stack overflow",
}

# stderr lines collected for the end-of-run notes (master plan 2.6, 4.1).
WARN_RE = re.compile(r"keeping ASR timestamps|out-of-range dropped|failed to load|"
                     r"exported at --chunk-seconds|input file not found|warning:", re.IGNORECASE)
# The program fetching something by itself means WhisperJAV's command line missed a switch
# (plan 4.3, run P1). It is a note, and the command-line tests keep it from shipping. Only the lines
# that start a download count (crispasr_cache.cpp:515, 525, 679; "Auto-downloading"; the question
# "Download now? [Y/n]", which a terminal input answers yes): the lines a missing model prints
# otherwise ("Available for download: ...", "Use --auto-download ...") download nothing.
DOWNLOAD_RE = re.compile(r"downloading|download now\?", re.IGNORECASE)
# The program could not open the -m file and would have offered a download (U19, probes B and D).
NOT_FOUND_LOCALLY_RE = re.compile(r"not found locally", re.IGNORECASE)
PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)\s*%\s*\((\d+)\s*/\s*(\d+)\s*slices?\)", re.IGNORECASE)
NO_SPEECH_RE = re.compile(r"no speech detected", re.IGNORECASE)
# The program's own processor check, before any work (cli.cpp:2579-2613). Both lines end the run with
# exit code 1, which otherwise means bad arguments. --diagnostics exits before this check, so the
# start check cannot see it.
CPU_LACKS_RE = re.compile(r"this CPU lacks:\s*([^(\r\n]*)", re.IGNORECASE)
NO_CPU_BACKEND_RE = re.compile(r"no CPU ggml backend could be initiali[sz]ed", re.IGNORECASE)

_STDERR_TAIL_CHARS = 2000
_MAX_COLLECTED_LINES = 500


class CrispOutcome(str, Enum):
    OK = "ok"                # exit 0, complete JSON
    NO_SPEECH = "no_speech"  # exit 0, no JSON, "no speech detected" (run E1): empty, not a failure
    NO_OUTPUT = "no_output"  # exit 0, no JSON, no such line: a failure (the 1.9.x record)
    FAILED = "failed"        # a non-zero exit (EXIT_MEANING, or "undocumented exit code N")
    CRASHED = "crashed"      # an exception status (Windows) or a signal (POSIX)


class CrispRunCancelled(Exception):
    """The run was stopped from WhisperJAV's side; the program has been ended."""


class CrispASRRunError(RuntimeError):
    """The program ran but left nothing WhisperJAV may use: that file fails (plan 4.3)."""


# A pass error that starts with this is a plain-language reason -- the CrispASR program or its files
# failed -- and the end-of-run summary shows it rather than "ensemble pass failed" (plan 2.6). The Ensemble
# pass worker adds it; main.py looks for it.
FAILURE_PREFIX = "CrispASR: "


RUN_FOLDER = "crispasr_out"  # the name WhisperJAV's leftover handling knows (utils/media_leftovers.py)
# The program's --cache-dir: an empty folder in the film's run folder, so it finds no model of its own.
NO_CACHE_FOLDER = "no_cache"


def film_run_dir(temp_dir: Path, media_basename: str) -> Path:
    """The program's working folder for one film: <temp>/crispasr_out/<ASCII name>/.

    One folder per film, so --keep-temp keeps every film's run.log in a batch. The name holds only
    ASCII letters, digits, '-' and '_' (plan 1.7), plus a short hash so two films never share it.
    """
    import hashlib

    ascii_part = re.sub(r"[^A-Za-z0-9_-]+", "_", media_basename).strip("_")[:40]
    digest = hashlib.sha1(media_basename.encode("utf-8")).hexdigest()[:8]
    return Path(temp_dir) / RUN_FOLDER / (f"{ascii_part}_{digest}" if ascii_part else f"film_{digest}")


def default_threads() -> int:
    """``-t`` for the program: min(8, CPU count). 8 is the tested value; the program's own default is 4."""
    return max(1, min(8, os.cpu_count() or 4))


@dataclass
class CrispRunRequest:
    """One program run over one film (the second study's form 1A-file)."""
    exe: Path
    run_dir: Path      # the film's crispasr_out folder; the program's working folder (plan 1.7)
    backend: str       # "parakeet" | "fastconformer-ctc" | "cohere" | ...
    model: str         # -m: an all-ASCII absolute path, or a short name inside run_dir (plan 1.7)
    audio_name: str = "audio.wav"
    cuts_name: str = "cuts.json"
    out_name: str = "out"
    language: str = "ja"
    threads: int = field(default_factory=default_threads)
    extra_args: tuple = ()  # from the build (macOS --no-gpu) and the model (CTC --punc-model none)

    @property
    def json_path(self) -> Path:
        return Path(self.run_dir) / f"{self.out_name}.json"

    @property
    def log_path(self) -> Path:
        return Path(self.run_dir) / "run.log"

    def argv(self) -> list[str]:
        return [str(self.exe), "-m", self.model, "--backend", self.backend, "-t", str(self.threads),
                "-f", self.audio_name, "-l", self.language, "-ojf", "-pp", "-of", self.out_name,
                "--vad-import", self.cuts_name, "--cache-dir", NO_CACHE_FOLDER,
                *[str(a) for a in self.extra_args]]


@dataclass
class CrispRunResult:
    outcome: CrispOutcome
    exit_code: int
    json_path: Optional[Path]  # set only when the JSON is complete and may be used
    warnings: list[str]        # stderr lines matching WARN_RE (capped)
    warning_count: int         # all of them, including those past the cap
    download_lines: list[str]  # stderr lines that mention a download (should never happen)
    stderr_tail: str
    seconds: float
    log_path: Path
    # A plainer reason than the exit code gives, when the program's own output says more
    # (today: its processor check).
    detail: Optional[str] = None

    @property
    def meaning(self) -> str:
        return self.detail or exit_code_meaning(self.exit_code)


def exit_code_meaning(rc: int) -> str:
    """The documented meaning of an exit code, or what kind of crash it was."""
    if rc == 0:
        return "finished normally"
    if rc in EXIT_MEANING:
        return EXIT_MEANING[rc]
    if rc < 0:
        return f"ended by signal {-rc}"
    if rc >= _CRASH_STATUS_MIN:
        kind = _KNOWN_CRASH_STATUS.get(rc)
        return f"crashed with status 0x{rc:08X}" + (f" ({kind})" if kind else "")
    return f"undocumented exit code {rc}"


def is_crash(rc: int) -> bool:
    return rc < 0 or rc >= _CRASH_STATUS_MIN


# ---------------------------------------------------------------------------------------------------
# Tying the program to WhisperJAV's process (Windows)
# ---------------------------------------------------------------------------------------------------

class _KillOnCloseJob:
    """A Windows job object that ends its processes when its last handle closes.

    WhisperJAV holds the handle. If WhisperJAV's own process ends for any reason -- the GUI's Stop,
    a crash, Task Manager -- Windows closes the handle and ends the program with it, so it never
    keeps a GPU busy on its own. Best effort: any failure only means the program is not tied.
    """

    def __init__(self):
        self.handle = None
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

            class BASIC_LIMIT(ctypes.Structure):
                _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                            ("PerJobUserTimeLimit", ctypes.c_int64),
                            ("LimitFlags", wintypes.DWORD),
                            ("MinimumWorkingSetSize", ctypes.c_size_t),
                            ("MaximumWorkingSetSize", ctypes.c_size_t),
                            ("ActiveProcessLimit", wintypes.DWORD),
                            ("Affinity", ctypes.c_size_t),
                            ("PriorityClass", wintypes.DWORD),
                            ("SchedulingClass", wintypes.DWORD)]

            class EXTENDED_LIMIT(ctypes.Structure):
                _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                            ("IoInfo", IO_COUNTERS),
                            ("ProcessMemoryLimit", ctypes.c_size_t),
                            ("JobMemoryLimit", ctypes.c_size_t),
                            ("PeakProcessMemoryUsed", ctypes.c_size_t),
                            ("PeakJobMemoryUsed", ctypes.c_size_t)]

            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.CreateJobObjectW.restype = wintypes.HANDLE
            k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
            k32.SetInformationJobObject.restype = wintypes.BOOL
            k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                    wintypes.DWORD]
            k32.AssignProcessToJobObject.restype = wintypes.BOOL
            k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
            k32.CloseHandle.restype = wintypes.BOOL
            k32.CloseHandle.argtypes = [wintypes.HANDLE]
            self._k32 = k32

            handle = k32.CreateJobObjectW(None, None)
            if not handle:
                return
            info = EXTENDED_LIMIT()
            info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not k32.SetInformationJobObject(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
                k32.CloseHandle(handle)  # 9 = JobObjectExtendedLimitInformation
                return
            self.handle = handle
        except Exception as e:  # pragma: no cover - depends on the OS
            logger.debug(f"CrispASR: no job object ({e}); the program is not tied to WhisperJAV")
            self.handle = None

    def assign(self, proc: subprocess.Popen) -> bool:
        if not self.handle:
            return False
        try:
            return bool(self._k32.AssignProcessToJobObject(self.handle, int(proc._handle)))
        except Exception as e:  # pragma: no cover - depends on the OS
            logger.debug(f"CrispASR: could not tie the program to WhisperJAV ({e})")
            return False

    def close(self) -> None:
        """Close the handle. Call only after the program has ended: closing it ends the program."""
        if self.handle:
            try:
                self._k32.CloseHandle(self.handle)
            except Exception:  # pragma: no cover
                pass
            self.handle = None


# ---------------------------------------------------------------------------------------------------
# Reading the program's output while it runs
# ---------------------------------------------------------------------------------------------------

class _StreamReader(threading.Thread):
    """Reads one pipe to its end, splitting on both \\n and \\r (progress lines may use either).

    It only records: the latest progress count is picked up by the waiting thread, so a progress
    display is never touched from this thread.
    """

    def __init__(self, stream, watch_progress: bool = False):
        super().__init__(daemon=True)
        self.stream = stream
        self.watch_progress = watch_progress
        self.lines: list[str] = []
        self.progress: Optional[tuple[int, int]] = None  # (slices done, slices in all)

    def run(self) -> None:
        buf = b""
        after_cr = False
        try:
            while True:
                chunk = self.stream.read1(65536) if hasattr(self.stream, "read1") else self.stream.read(65536)
                if not chunk:
                    break
                # A line ending in "\r" is used at once, so a progress line reaches the display while the
                # program is still quiet. If the "\r" was the first half of a "\r\n" split across two
                # reads, the "\n" that opens the next read ends nothing and is dropped.
                if after_cr and chunk.startswith(b"\n"):
                    chunk = chunk[1:]
                after_cr = chunk.endswith(b"\r")
                buf += chunk
                parts = re.split(rb"\r\n|\n|\r", buf)
                buf = parts.pop()
                for raw in parts:
                    self._emit(raw)
            if buf:
                self._emit(buf)
        except (OSError, ValueError):
            pass  # the pipe closed under us: the program was ended

    def _emit(self, raw: bytes) -> None:
        line = raw.decode("utf-8", errors="replace")
        self.lines.append(line)
        if self.watch_progress:
            m = PROGRESS_RE.search(line)
            if m:
                self.progress = (int(m.group(2)), int(m.group(3)))


def _end_process(proc: subprocess.Popen, grace_s: float = 5.0) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=grace_s)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=grace_s)
        except Exception:
            pass


def _write_run_log(req: CrispRunRequest, rc: Optional[int], seconds: float, outcome: str,
                   stderr_lines: list[str], stdout_lines: list[str]) -> None:
    rc_text = "none (the program did not end by itself)" if rc is None else \
        f"{rc} (0x{rc & 0xFFFFFFFF:08X}): {exit_code_meaning(rc)}" if rc else "0"
    head = [
        f"command: {subprocess.list2cmdline(req.argv())}",
        f"working folder: {req.run_dir}",
        f"exit code: {rc_text}",
        f"outcome: {outcome}",
        f"seconds: {seconds:.1f}",
        "----- stderr",
    ]
    try:
        with open(req.log_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(head) + "\n")
            fh.write("\n".join(stderr_lines) + "\n")
            fh.write("----- stdout\n")
            fh.write("\n".join(stdout_lines) + "\n")
    except OSError as e:
        logger.warning(f"CrispASR: could not write {req.log_path}: {e}")


# ---------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------

def run_crispasr_json(req: CrispRunRequest,
                      on_progress: Optional[Callable[[int, int], None]] = None,
                      cancel_event: Optional[threading.Event] = None,
                      poll_s: float = 0.5) -> CrispRunResult:
    """Run the program once and judge what it left behind (master plan 4.3).

    ``on_progress(done, total)`` is called, from this thread, when the program's latest
    "progress = N% (i/n slices)" line changes.
    ``cancel_event``, when set, ends the program and raises CrispRunCancelled.
    Any exception while waiting (KeyboardInterrupt included) ends the program before it propagates.
    """
    run_dir = Path(req.run_dir)
    for name in (req.audio_name, req.cuts_name):
        if not (run_dir / name).is_file():
            raise FileNotFoundError(f"CrispASR input missing before the run: {run_dir / name}")
    if not Path(req.exe).is_file():
        # Gone since it was checked (deleted, or taken by an antivirus): the run's own error, in words.
        raise CrispASRRunError(f"the CrispASR program is missing: {req.exe}")
    # A stale file must never pass as this run's output.
    if req.json_path.exists():
        req.json_path.unlink()
    # The program's cache folder, empty, so it cannot use a model of its own in place of -m's.
    no_cache = run_dir / NO_CACHE_FOLDER
    if no_cache.exists():
        shutil.rmtree(no_cache)
    no_cache.mkdir()

    reported: Optional[tuple[int, int]] = None

    def report_progress() -> None:
        nonlocal reported
        latest = err_reader.progress
        if on_progress is None or latest is None or latest == reported:
            return
        reported = latest
        try:
            on_progress(*latest)
        except Exception as e:  # a progress display must never break the run
            logger.debug(f"CrispASR progress display failed: {e}")

    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    argv = req.argv()
    logger.debug(f"CrispASR command (in {run_dir}): {subprocess.list2cmdline(argv)}")
    started = time.monotonic()
    job = _KillOnCloseJob()
    try:
        # A pipe, closed at once: the program then sees no terminal and asks no download question.
        proc = subprocess.Popen(argv, cwd=str(run_dir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, creationflags=creationflags)
        try:
            proc.stdin.close()
        except OSError:
            pass  # the program has already ended
    except OSError as e:
        # A damaged file, or one an antivirus blocked (U17), cannot be started at all.
        job.close()
        _write_run_log(req, None, 0.0, f"could not be started: {e}", [], [])
        raise CrispASRRunError(f"the CrispASR program could not be started: {e}") from e
    except BaseException:
        job.close()
        raise
    job.assign(proc)
    out_reader = _StreamReader(proc.stdout)
    err_reader = _StreamReader(proc.stderr, watch_progress=True)
    out_reader.start()
    err_reader.start()

    rc: Optional[int] = None
    finished = False
    try:
        while True:
            try:
                rc = proc.wait(timeout=poll_s)
                break
            except subprocess.TimeoutExpired:
                report_progress()
                if cancel_event is not None and cancel_event.is_set():
                    raise CrispRunCancelled("the run was stopped; the CrispASR program was ended")
        # The pipes close when the program ends. A limit, so a process it may have left behind holding
        # them can never hang the run; the readers are daemon threads.
        out_reader.join(timeout=60)
        err_reader.join(timeout=60)
        if out_reader.is_alive() or err_reader.is_alive():
            logger.warning("CrispASR: the program ended but its output pipes stayed open; reading stopped")
        report_progress()
        finished = True
    finally:
        if not finished:
            # Whether the program was still running when WhisperJAV stopped: its exit code is then the
            # one TerminateProcess gave it, which says nothing about the program.
            was_running = proc.poll() is None
            _end_process(proc)
            out_reader.join(timeout=5)
            err_reader.join(timeout=5)
            _write_run_log(req, None if was_running else proc.returncode, time.monotonic() - started,
                           "ended by WhisperJAV", err_reader.lines, out_reader.lines)
        job.close()
        # Close each pipe once its reader has stopped (closing waits for a read in progress).
        for reader in (out_reader, err_reader):
            if not reader.is_alive():
                try:
                    reader.stream.close()
                except (OSError, ValueError):
                    pass

    seconds = time.monotonic() - started
    stderr_lines = err_reader.lines
    stderr_text = "\n".join(stderr_lines)
    all_warnings = [ln.strip() for ln in stderr_lines if WARN_RE.search(ln)]
    downloads = [ln.strip() for ln in stderr_lines if DOWNLOAD_RE.search(ln)]
    doc = read_ojf(req.json_path)

    if is_crash(rc):
        # The 1.9.x record holds a crash after the output was written (the 1.9.x runner,
        # pipelines/crispasr_runner.py, removed in 1.10.0; see git history):
        # a complete JSON is kept, and the crash is still reported.
        outcome, json_path = CrispOutcome.CRASHED, (req.json_path if doc is not None else None)
    elif rc != 0:
        outcome, json_path = CrispOutcome.FAILED, None  # no partial JSON is used
    elif doc is not None:
        outcome, json_path = CrispOutcome.OK, req.json_path
    elif NO_SPEECH_RE.search(stderr_text):
        outcome, json_path = CrispOutcome.NO_SPEECH, None
    else:
        outcome, json_path = CrispOutcome.NO_OUTPUT, None

    detail = None
    if rc != 0:
        lacks = CPU_LACKS_RE.search(stderr_text)
        if lacks:
            detail = (f"this processor lacks instructions the CrispASR program needs "
                      f"({lacks.group(1).strip() or 'AVX2'})")
        elif NO_CPU_BACKEND_RE.search(stderr_text):
            detail = "the CrispASR program found no processor support it can run on"
        elif NOT_FOUND_LOCALLY_RE.search(stderr_text):
            # DRAFT text, not yet approved by the owner (plan 3.8 item 6).
            detail = "the model file could not be opened; the CrispASR program may not download one by itself"

    _write_run_log(req, rc, seconds, outcome.value, stderr_lines, out_reader.lines)
    return CrispRunResult(outcome=outcome, exit_code=rc, json_path=json_path,
                          warnings=all_warnings[:_MAX_COLLECTED_LINES], warning_count=len(all_warnings),
                          download_lines=downloads[:_MAX_COLLECTED_LINES],
                          stderr_tail=stderr_text[-_STDERR_TAIL_CHARS:], seconds=seconds,
                          log_path=req.log_path, detail=detail)
