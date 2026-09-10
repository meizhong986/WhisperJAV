"""Per-file outcomes and the exit-status contract.

One vocabulary, one verdict function, one exit-status function, used by every
execution path (sync, async, ensemble). Nothing else in the code base may
decide the process exit status.

The vocabulary
--------------
``done``     an output file with at least one cue was written
``empty``    the run completed and produced no cues, and nothing contradicts
             that reading -- an observation, not a failure: silence, music, or
             speech the recogniser could not use all end here
``suspect``  something about the output does not add up: it spans less than
             ``--min-coverage`` of the media; the recogniser returned nothing for
             consecutive scenes while speech was still being detected; a health
             probe failed; or (ensemble) pass 2 failed and the output is pass 1
             alone. A zero-cue file with any of that evidence is ``suspect``,
             not ``empty``
``failed``   an error: an exception, a crash, a requested step (translation)
             that raised, or a subtitle file the pipeline reported writing that
             does not exist
``skipped``  nothing was attempted: the output already existed

Coverage is reported alongside the state as ``ok``, ``low`` or
``not assessed`` (unknown duration, media shorter than the assessable minimum,
or an execution path where the check cannot run). "Not assessed" is printed;
it is never left to mean "fine".

The contract
------------
The exit status is 1 if any file is ``failed``, or if the run did not complete
(interrupted, or stopped by an unhandled error), else 0. ``--fail-on empty``
and ``--fail-on suspect`` add those states to the set that fails the run.
Warnings never change the exit status. The same words appear in the console
table, the JSON manifest written next to the outputs, and the release notes.

Why this exists
---------------
Until v1.9.2 each execution path decided success on its own: the sync path
returned nothing, the async path hard-coded zero, the ensemble path kept its
own failure list, and the ctranslate2 fast exit was ``os._exit(0)``. The GUI,
which decides purely on the exit code, therefore printed ``[SUCCESS]`` over
0-byte subtitle files (#263, #394). Fixing that path by path produced a
release candidate on which every successful ensemble run exited 1. One
vocabulary and one mapping is the fix for both.
"""

from __future__ import annotations

import datetime as _dt
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Union

from whisperjav.utils.logger import logger
from whisperjav.utils.output_coverage import (
    DEFAULT_EMPTY_STREAK_THRESHOLD,
    DEFAULT_MIN_COVERAGE,
    assess_coverage,
)

# The five per-file states. Order is the order they print in.
STATES: Sequence[str] = ("done", "empty", "suspect", "failed", "skipped")

# States a user may promote to run failure with --fail-on. "failed" always
# fails and "done"/"skipped" never do, so neither is offered.
FAIL_ON_CHOICES: Sequence[str] = ("empty", "suspect")

# Name of the manifest written next to the outputs. Fixed rather than
# timestamped so a folder does not accumulate one file per run; a script reads
# it immediately after the run it started.
MANIFEST_NAME = "whisperjav_run.json"


@dataclass
class FileOutcome:
    """What happened to one input file, in the shared vocabulary."""

    path: str
    state: str
    detail: str = ""
    output: Optional[str] = None
    translated_output: Optional[str] = None
    subtitle_count: Optional[int] = None
    coverage: str = "not assessed"          # "ok" | "low" | "not assessed"
    coverage_ratio: Optional[float] = None
    coverage_detail: str = ""
    translation: str = "not requested"      # "not requested" | "done" | "failed" | "skipped"
    error: Optional[str] = None
    processing_time_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise ValueError(f"unknown file state {self.state!r}; expected one of {STATES}")

    @property
    def basename(self) -> str:
        return Path(self.path).name


# ---------------------------------------------------------------------------
# The verdict: one function, every path
# ---------------------------------------------------------------------------

def classify_output(
    path: str,
    output_path: Union[str, Path, None],
    media_duration_s: Optional[float],
    *,
    min_coverage: Optional[float] = None,
    speech_positive_empty_streak: int = 0,
    degraded: bool = False,
    degraded_reason: str = "",
    processing_time_s: Optional[float] = None,
) -> FileOutcome:
    """Turn a completed transcription into a FileOutcome.

    ``path`` is the input media; ``output_path`` the subtitle file the pipeline
    reports having written (None or "" when it reports none). ``degraded`` is
    the ensemble case where pass 2 failed and the output is pass 1 alone.
    Never raises: a classification problem must not turn a finished run into a
    crash, so anything unexpected degrades to "not assessed".
    """
    mc = DEFAULT_MIN_COVERAGE if min_coverage is None else min_coverage
    out = str(output_path) if output_path else None
    why_degraded = degraded_reason or "pass 2 failed; output is pass 1 alone"

    # A file the pipeline reported writing but which is not there is an error,
    # not an empty result: something between "written" and "here" went wrong.
    if out:
        try:
            missing = not Path(out).exists()
        except OSError:
            missing = True
        if missing:
            return failed_outcome(
                path, f"subtitle file was not created: {Path(out).name}",
                output=out, processing_time_s=processing_time_s,
            )

    try:
        cov = assess_coverage(
            out,
            media_duration_s,
            min_coverage=mc,
            speech_positive_empty_streak=speech_positive_empty_streak,
            empty_streak_threshold=DEFAULT_EMPTY_STREAK_THRESHOLD,
        )
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.debug("Could not assess %s: %s", out, exc)
        return FileOutcome(
            path=path, state="done", detail="output written; coverage not assessed",
            output=out, coverage="not assessed",
            coverage_detail=f"assessment error: {exc}",
            processing_time_s=processing_time_s,
        )

    if cov.verdict == "empty":
        # Zero cues. Plain silence is "empty"; zero cues *with* evidence that
        # the recogniser stopped, or after pass 2 failed, is "suspect" -- the
        # #394 cascade must not print the same word as a silent clip.
        if cov.corroborated:
            state, detail = "suspect", f"{cov.detail}; corroborated by {cov.corroboration_detail}"
        elif degraded:
            state, detail = "suspect", f"{why_degraded}; {cov.detail}"
        else:
            state, detail = "empty", cov.detail
        return FileOutcome(
            path=path, state=state, detail=detail, output=out,
            subtitle_count=0, coverage="not assessed",
            coverage_detail="no cues to measure", processing_time_s=processing_time_s,
        )

    if cov.verdict == "unknown":
        state, detail = "done", f"{cov.subtitle_count} cue(s); coverage not assessed"
        coverage = "not assessed"
    elif cov.verdict == "implausible":
        state, detail = "suspect", cov.detail
        coverage = "low"
    else:  # "ok"
        state, detail = "done", cov.detail
        coverage = "ok"

    if state == "done" and cov.corroborated:
        # Full-length output can still be corroborated as broken (#394: a
        # 10-minute file whose SRT stopped at ~6 minutes while dialogue went on).
        state = "suspect"
        detail = f"{cov.detail}; corroborated by {cov.corroboration_detail}"

    if degraded:
        state = "suspect"
        detail = f"{why_degraded}; {detail}"

    return FileOutcome(
        path=path, state=state, detail=detail, output=out,
        subtitle_count=cov.subtitle_count, coverage=coverage,
        coverage_ratio=cov.coverage_ratio, coverage_detail=cov.detail,
        processing_time_s=processing_time_s,
    )


def failed_outcome(path: str, error: str, *, output: Optional[str] = None,
                   processing_time_s: Optional[float] = None) -> FileOutcome:
    return FileOutcome(path=path, state="failed", detail=error, error=error,
                       output=output, processing_time_s=processing_time_s)


def skipped_outcome(path: str, reason: str = "output already exists") -> FileOutcome:
    return FileOutcome(path=path, state="skipped", detail=reason)


def mark_translation(outcome: FileOutcome, result: str, *,
                     translated_output: Optional[str] = None,
                     error: Optional[str] = None) -> FileOutcome:
    """Record the translation step. A failed translation is a failed file.

    The transcription may be intact, but a step the user asked for raised an
    error; the contract has no "half failed" state, and swallowing it is what
    the old ensemble path did (exit 1) while the sync path did not (exit 0).
    A translation *skipped* on purpose (too few cues, or a pass-1 fallback
    that is deliberately not translated) is not a failure.
    """
    outcome.translation = result
    if translated_output:
        outcome.translated_output = str(translated_output)
    if result == "failed":
        outcome.state = "failed"
        outcome.error = error or "translation failed"
        outcome.detail = f"transcribed; translation failed: {outcome.error}"
    elif result == "skipped" and error:
        outcome.detail = f"{outcome.detail}; translation skipped: {error}"
    return outcome


# ---------------------------------------------------------------------------
# The exit status: one function, every path
# ---------------------------------------------------------------------------

def parse_fail_on(values: Optional[Iterable[str]]) -> frozenset:
    """Parse repeated / comma-separated --fail-on values. Raises ValueError."""
    chosen = set()
    for raw in values or []:
        for token in str(raw).split(","):
            token = token.strip().lower()
            if not token:
                continue
            if token not in FAIL_ON_CHOICES:
                raise ValueError(
                    f"--fail-on: unknown state {token!r} (choose from "
                    f"{', '.join(FAIL_ON_CHOICES)})"
                )
            chosen.add(token)
    return frozenset(chosen)


def exit_status(outcomes: Iterable[FileOutcome], fail_on: Iterable[str] = ()) -> int:
    """0 unless any file is ``failed`` or in a state named by --fail-on.

    This is the per-file rule only. A run that did not complete (interrupt,
    unhandled error) is forced to 1 by the caller regardless of the files.
    """
    failing = {"failed"} | set(fail_on)
    return 1 if any(o.state in failing for o in outcomes) else 0


def count_states(outcomes: Iterable[FileOutcome]) -> dict[str, int]:
    counts = {s: 0 for s in STATES}
    for o in outcomes:
        counts[o.state] += 1
    return counts


# ---------------------------------------------------------------------------
# Reporting: the same words on the console and in the manifest
# ---------------------------------------------------------------------------

def log_outcome(outcome: FileOutcome) -> None:
    """One line per file, as it finishes, at a severity matching its state."""
    name = outcome.basename
    if outcome.state == "done":
        logger.info("done: %s -> %s (%s)", name,
                    Path(outcome.output).name if outcome.output else "-", outcome.detail)
    elif outcome.state == "empty":
        logger.warning(
            "empty: %s produced no subtitles (%s). This is reported, not treated "
            "as a failure; pass --fail-on empty to make it one.", name, outcome.detail)
    elif outcome.state == "suspect":
        logger.warning(
            "suspect: %s -> %s. %s. Reported, not treated as a failure; pass "
            "--fail-on suspect to make it one.", name,
            Path(outcome.output).name if outcome.output else "-", outcome.detail)
    elif outcome.state == "failed":
        logger.error("failed: %s - %s", name, outcome.detail)
    else:
        logger.info("skipped: %s (%s)", name, outcome.detail)


def _coverage_cell(o: FileOutcome) -> str:
    if o.coverage == "not assessed":
        return "not assessed"
    pct = f"{o.coverage_ratio:.0%}" if isinstance(o.coverage_ratio, (int, float)) else ""
    return f"{o.coverage} {pct}".strip()


def format_summary(outcomes: Sequence[FileOutcome], fail_on: Iterable[str],
                   status: int, manifest_path: Optional[Path] = None,
                   note: str = "") -> str:
    fail_set = sorted({"failed"} | set(fail_on))
    lines = []
    bar = "=" * 60
    lines.append(bar)
    lines.append(f"RUN SUMMARY  (exit status {status}; a run fails on: {', '.join(fail_set)})")
    lines.append(bar)
    if note:
        lines.append(note)
    lines.append(f"{'STATE':<8} {'COVERAGE':<14} FILE")
    for o in outcomes:
        target = ""
        if o.output and o.state != "skipped":
            target = f" -> {Path(o.output).name}"
        cues = f" ({o.subtitle_count} cue(s))" if o.subtitle_count else ""
        extra = f"; {o.detail}" if o.detail and o.state in ("suspect", "failed", "skipped", "empty") else ""
        if o.translated_output:
            extra += f"; translated -> {Path(o.translated_output).name}"
        lines.append(f"{o.state:<8} {_coverage_cell(o):<14} {o.basename}{target}{cues}{extra}")
    lines.append("-" * 60)
    counts = count_states(outcomes)
    lines.append("  ".join(f"{s} {counts[s]}" for s in STATES) + f"  total {len(outcomes)}")
    if manifest_path:
        lines.append(f"Manifest: {manifest_path}")
    lines.append(bar)
    return "\n".join(lines)


def print_summary(outcomes: Sequence[FileOutcome], fail_on: Iterable[str],
                  status: int, manifest_path: Optional[Path] = None,
                  note: str = "") -> None:
    """Print the table. A console that cannot encode a file name (cp932 and
    friends) must not turn a finished run into a crash, so fall back to a
    replaced-character rendering rather than raise."""
    text = "\n" + format_summary(outcomes, fail_on, status, manifest_path, note)
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"))


def default_manifest_path(output_dir_arg: str, inputs: Sequence[str]) -> Optional[Path]:
    """Where the manifest goes: the output directory, or for
    ``--output-dir source`` beside the first input *as the user gave it* --
    inside that folder when it is a folder, next to it when it is a file.

    ``inputs`` is the raw input list (the CLI's positional arguments, the
    GUI's selected files and folders), not the expanded media list. The GUI
    computes this path with the same call and the same values, so the two
    sides agree by construction; deriving it from the discovered media would
    put the manifest inside a sub-folder the GUI never sees.
    """
    try:
        if str(output_dir_arg).lower().strip() == "source":
            if not inputs:
                return None
            first = Path(str(inputs[0]))
            base = first if first.is_dir() else first.parent
            return base / MANIFEST_NAME
        return Path(output_dir_arg) / MANIFEST_NAME
    except Exception:  # noqa: BLE001 - a manifest path must never abort a run
        return None


def write_manifest(
    outcomes: Sequence[FileOutcome],
    path: Optional[Path],
    *,
    mode: str,
    fail_on: Iterable[str],
    status: int,
    started_at: Optional[_dt.datetime] = None,
    version: str = "",
    note: str = "",
) -> Optional[Path]:
    """Write the JSON manifest. Returns the path, or None if it could not be
    written; a manifest problem never changes the run's outcome."""
    if path is None:
        return None
    payload = {
        "whisperjav_version": version,
        "mode": mode,
        "started_at": started_at.isoformat(timespec="seconds") if started_at else None,
        "finished_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "exit_status": status,
        "fails_on": sorted({"failed"} | set(fail_on)),
        "note": note or None,
        "counts": count_states(outcomes),
        "files": [asdict(o) for o in outcomes],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not write run manifest to %s: %s", path, exc)
        return None
