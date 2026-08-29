"""Output coverage assessment (#394).

Why this exists
---------------
WhisperJAV could finish a run, report success, and hand the user a subtitle file
covering a small fraction of the input.  Reported instances:

* an 8,766 s input returning **PASS** with an SRT whose last cue ended at 376.9 s
  (4.299% of the file);
* a 118-minute input whose pass 2 produced 82 cues ending at 00:06:44;
* 314 consecutive failed scenes producing a 0-byte SRT while the console printed
  ``[SUCCESS] Process completed successfully``.

In every case the exit status was 0, so nothing downstream — least of all the
GUI, which declares success on ``exit_code == 0`` alone — had any way to tell a
good run from a destroyed one.

This module does not diagnose *why* output is missing; the underlying cause is
still under investigation.  It answers a narrower and fully decidable question:
**does the subtitle file span a plausible portion of the media it came from?**

Deliberate limits
-----------------
Span is not speech coverage.  A file whose dialogue genuinely stops early will
show a low span, and that is not a defect.  The thresholds are therefore set
conservatively, the assessment is advisory unless a caller opts into enforcement,
and the numbers are always reported so a human can judge.  A false failure would
be worse than the problem being solved.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from whisperjav.utils.logger import logger

# Below this ratio of (last cue end / media duration) the output is treated as
# implausible.  Chosen well under the lowest legitimate value we have seen and
# well above the observed failures (0.043, ~0.057).
DEFAULT_MIN_COVERAGE = 0.25

# Between DEFAULT_MIN_COVERAGE and this, report but do not fail.
SUSPICIOUS_COVERAGE = 0.60

# Media shorter than this are not assessed: a short clip can legitimately hold a
# single line near the start, making the ratio meaningless.
MIN_ASSESSABLE_DURATION_S = 120.0


@dataclass(frozen=True)
class CoverageReport:
    """Outcome of assessing one subtitle file against its source media."""

    verdict: str  # "ok" | "suspicious" | "implausible" | "empty" | "unknown"
    coverage_ratio: Optional[float]  # last cue end / media duration
    last_cue_end_s: Optional[float]
    media_duration_s: Optional[float]
    subtitle_count: int
    detail: str

    @property
    def is_failure(self) -> bool:
        """True when the output is bad enough that a run should not be called successful."""
        return self.verdict in ("implausible", "empty")

    @property
    def is_noteworthy(self) -> bool:
        """True when the user should see this, whether or not it fails the run."""
        return self.verdict in ("implausible", "empty", "suspicious")


def _parse_last_cue_end(srt_path: Path) -> tuple[int, Optional[float]]:
    """Return (cue_count, last_cue_end_seconds) for an SRT file.

    Parsed defensively: a malformed or partially written file yields whatever
    could be read rather than raising, because this runs on the failure path.
    """
    try:
        import srt as srt_lib
    except ImportError:  # pragma: no cover - srt is a hard dependency in practice
        logger.debug("Coverage check skipped: the 'srt' package is unavailable")
        return 0, None

    try:
        text = srt_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        logger.debug("Coverage check could not read %s: %s", srt_path, exc)
        return 0, None

    if not text.strip():
        return 0, None

    try:
        cues = list(srt_lib.parse(text))
    except Exception as exc:  # noqa: BLE001 - any parse error means "cannot assess"
        logger.debug("Coverage check could not parse %s: %s", srt_path, exc)
        return 0, None

    if not cues:
        return 0, None

    last_end = max(c.end.total_seconds() for c in cues)
    return len(cues), last_end


def assess_coverage(
    srt_path: Union[str, Path, None],
    media_duration_s: Optional[float],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
) -> CoverageReport:
    """Assess whether *srt_path* plausibly covers *media_duration_s*.

    Args:
        srt_path: the final subtitle file. A missing path yields "empty".
        media_duration_s: source duration in seconds, from media discovery.
        min_coverage: ratio below which the result is "implausible".
            Pass 0 to disable the ratio check entirely (zero-cue output is
            still reported as "empty").

    Returns:
        A :class:`CoverageReport`. Never raises.
    """
    if not srt_path:
        return CoverageReport("empty", None, None, media_duration_s, 0,
                              "no subtitle file was produced")

    path = Path(srt_path)
    if not path.exists():
        return CoverageReport("empty", None, None, media_duration_s, 0,
                              f"subtitle file was not created: {path.name}")

    count, last_end = _parse_last_cue_end(path)

    if count == 0 or last_end is None:
        return CoverageReport("empty", None, None, media_duration_s, 0,
                              f"{path.name} contains no subtitles")

    if not media_duration_s or media_duration_s <= 0:
        return CoverageReport("unknown", None, last_end, media_duration_s, count,
                              "media duration unknown, coverage not assessed")

    if media_duration_s < MIN_ASSESSABLE_DURATION_S:
        return CoverageReport("unknown", None, last_end, media_duration_s, count,
                              f"media shorter than {MIN_ASSESSABLE_DURATION_S:.0f}s, "
                              "coverage not assessed")

    ratio = last_end / media_duration_s

    if min_coverage > 0 and ratio < min_coverage:
        verdict = "implausible"
        detail = (
            f"subtitles stop at {last_end:.0f}s of {media_duration_s:.0f}s "
            f"({ratio:.1%} of the file); {count} cue(s) produced"
        )
    elif ratio < SUSPICIOUS_COVERAGE:
        verdict = "suspicious"
        detail = (
            f"subtitles stop at {last_end:.0f}s of {media_duration_s:.0f}s "
            f"({ratio:.1%} of the file); {count} cue(s) produced"
        )
    else:
        verdict = "ok"
        detail = f"{count} cue(s) spanning {ratio:.1%} of the file"

    return CoverageReport(verdict, ratio, last_end, media_duration_s, count, detail)


def report_coverage(report: CoverageReport, file_label: str) -> None:
    """Surface *report* to the user at a severity matching its verdict."""
    if report.verdict == "ok":
        logger.debug("Coverage OK for %s: %s", file_label, report.detail)
        return

    if report.verdict == "unknown":
        logger.debug("Coverage not assessed for %s: %s", file_label, report.detail)
        return

    if report.verdict == "suspicious":
        logger.warning(
            "Output looks short for %s: %s. This can be normal if speech genuinely "
            "stops early; if not, please re-run with --log-level DEBUG and report it.",
            file_label, report.detail,
        )
        return

    # implausible / empty
    logger.error(
        "Incomplete output for %s: %s. The run finished without raising an error, "
        "but the result does not plausibly cover the input. This is the failure "
        "tracked in issue #394.",
        file_label, report.detail,
    )
