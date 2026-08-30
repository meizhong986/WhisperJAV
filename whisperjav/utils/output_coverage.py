"""Output coverage assessment (#394).

Why this exists
---------------
WhisperJAV could finish a run, report success, and hand the user a subtitle file
covering a small fraction of the input. Reported instances:

* an 8,766 s input returning **PASS** with an SRT whose last cue ended at 376.9 s
  (4.299% of the file);
* a 118-minute input whose pass 2 produced 82 cues ending at 00:06:44;
* 314 consecutive failed scenes producing a 0-byte SRT while the console printed
  ``[SUCCESS] Process completed successfully``.

In every case the exit status was 0, so nothing downstream — least of all the
GUI, which declares success on ``exit_code == 0`` alone — had any way to tell a
good run from a destroyed one.

What decides a failure
----------------------
The first draft of this module treated low temporal span as sufficient grounds
to fail a run. Both reporters on #394 rejected that, and they were right:

* **@daoran9**: *"I would avoid using media-duration span as the only hard-failure
  signal. A long intro, outro, credits section or non-dialogue tail could make
  that metric ambiguous. Also, a transcription stopping at 60% would not trigger
  the 25% threshold."*
* **@13e5t** supplied a case the threshold would have missed entirely — a
  10-minute file whose SRT stopped at roughly 6 minutes while dialogue continued.
  At 60% span, no span-based threshold set low enough to be safe would catch it.

So span alone never fails a run; it warns. A run is failed only when the output
is unusable beyond argument (no cues at all), or when low span is **corroborated**
by evidence of the recogniser having actually stopped working — consecutive empty
results while the voice detector was still reporting speech, or a failed
same-instance health probe. That is the shape both reporters asked for, and it
catches the real cases through the corroborating signal rather than through a
threshold that has to be guessed.

This module does not diagnose *why* output is missing; the underlying cause is
still open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from whisperjav.utils.logger import logger

# Span below this fraction of the media duration is reported as implausible.
# On its own it only warns; see the module docstring.
DEFAULT_MIN_COVERAGE = 0.25

# Between DEFAULT_MIN_COVERAGE and this, the output is merely short.
SUSPICIOUS_COVERAGE = 0.60

# Consecutive empty ASR results, while the voice detector still reported speech,
# that constitute corroboration. Chosen well above ordinary quiet passages.
DEFAULT_EMPTY_STREAK_THRESHOLD = 5

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
    corroborated: bool = False

    @property
    def is_failure(self) -> bool:
        """True when the run should not be reported as successful.

        Deliberately narrow: no cues at all, or short output *plus* independent
        evidence that the recogniser stopped working. Short output on its own is
        not enough — speech can genuinely stop early.
        """
        return self.verdict == "empty" or (
            self.verdict == "implausible" and self.corroborated
        )

    @property
    def is_noteworthy(self) -> bool:
        """True when the user should see this, whether or not the run fails."""
        return self.verdict in ("implausible", "empty", "suspicious")


def _parse_last_cue_end(srt_path: Path) -> tuple:
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

    return len(cues), max(c.end.total_seconds() for c in cues)


def assess_coverage(
    srt_path: Union[str, Path, None],
    media_duration_s: Optional[float],
    *,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    speech_positive_empty_streak: int = 0,
    probe_failed: bool = False,
    empty_streak_threshold: int = DEFAULT_EMPTY_STREAK_THRESHOLD,
) -> CoverageReport:
    """Assess whether *srt_path* plausibly covers *media_duration_s*.

    Args:
        srt_path: the final subtitle file. A missing path yields "empty".
        media_duration_s: source duration in seconds, from media discovery.
        min_coverage: span below which the result is "implausible". Pass 0 to
            disable the ratio check; zero-cue output is still reported as empty.
        speech_positive_empty_streak: longest run of consecutive empty ASR
            results observed *while an external voice detector still reported
            speech*. This is the corroborating signal both #394 reporters asked
            for.

            **Only pass a non-zero value when an external segmenter is in use.**
            Under faster-whisper's native VAD — the balanced default since
            v1.9.0 — segmentation is bypassed by ``NullSpeechSegmenter``, which
            returns the whole scene as one segment unconditionally. That is a
            passthrough, not a speech detection, so counting it as
            speech-positive would invent corroboration where none exists.
            Issue #324 is the cautionary case: 33 consecutive empty scenes on a
            music performance where the audio genuinely held no dialogue. Under
            native VAD the honest corroboration is ``probe_failed``.
        probe_failed: True if a same-instance health probe failed — the signal
            @AlanZ-Git identified, where a known-good clip returns nothing from
            the already-loaded model but transcribes correctly in a fresh
            process.
        empty_streak_threshold: streak length that counts as corroboration.

    Returns:
        A :class:`CoverageReport`. Never raises.
    """
    corroborated = bool(probe_failed) or (
        speech_positive_empty_streak >= empty_streak_threshold
    )

    def _report(verdict, ratio, last_end, count, detail):
        return CoverageReport(
            verdict, ratio, last_end, media_duration_s, count, detail, corroborated
        )

    if not srt_path:
        return _report("empty", None, None, 0, "no subtitle file was produced")

    path = Path(srt_path)
    if not path.exists():
        return _report("empty", None, None, 0,
                       f"subtitle file was not created: {path.name}")

    count, last_end = _parse_last_cue_end(path)

    if count == 0 or last_end is None:
        return _report("empty", None, None, 0, f"{path.name} contains no subtitles")

    if not media_duration_s or media_duration_s <= 0:
        return _report("unknown", None, last_end, count,
                       "media duration unknown, coverage not assessed")

    if media_duration_s < MIN_ASSESSABLE_DURATION_S:
        return _report("unknown", None, last_end, count,
                       f"media shorter than {MIN_ASSESSABLE_DURATION_S:.0f}s, "
                       "coverage not assessed")

    ratio = last_end / media_duration_s
    span = (f"subtitles stop at {last_end:.0f}s of {media_duration_s:.0f}s "
            f"({ratio:.1%} of the file); {count} cue(s) produced")

    if min_coverage > 0 and ratio < min_coverage:
        detail = span
        if corroborated:
            why = ("a health probe failed" if probe_failed
                   else f"{speech_positive_empty_streak} consecutive empty results "
                        "while speech was still being detected")
            detail = f"{span}; corroborated by {why}"
        return _report("implausible", ratio, last_end, count, detail)

    if ratio < SUSPICIOUS_COVERAGE:
        return _report("suspicious", ratio, last_end, count, span)

    # Full-length output can still be corroborated as broken — @13e5t's case
    # stopped at ~60% of a 10-minute file, which no safe span threshold catches.
    if corroborated:
        why = ("a health probe failed" if probe_failed
               else f"{speech_positive_empty_streak} consecutive empty results "
                    "while speech was still being detected")
        return _report("suspicious", ratio, last_end, count,
                       f"{count} cue(s) spanning {ratio:.1%} of the file, but {why}")

    return _report("ok", ratio, last_end, count,
                   f"{count} cue(s) spanning {ratio:.1%} of the file")


def report_coverage(report: CoverageReport, file_label: str) -> None:
    """Surface *report* to the user at a severity matching its verdict."""
    if report.verdict == "ok":
        logger.debug("Coverage OK for %s: %s", file_label, report.detail)
        return

    if report.verdict == "unknown":
        logger.debug("Coverage not assessed for %s: %s", file_label, report.detail)
        return

    if report.is_failure:
        logger.error(
            "Incomplete output for %s: %s. The run finished without raising an "
            "error, but the result does not plausibly cover the input. This is "
            "the failure tracked in issue #394.",
            file_label, report.detail,
        )
        return

    logger.warning(
        "Output looks short for %s: %s. This can be normal when speech genuinely "
        "stops early, so the run is not being failed on this alone. If the file "
        "does have dialogue past that point, please re-run with --log-level DEBUG "
        "and report it on issue #394.",
        file_label, report.detail,
    )


# Segmenter names that are passthroughs rather than genuine speech detection.
# Under faster-whisper's native VAD -- the balanced default since v1.9.0 -- the
# external segmenter is NullSpeechSegmenter, which returns the whole scene as a
# single segment unconditionally. Counting that as "speech was detected" would
# manufacture corroboration and fail runs whose audio genuinely holds no
# dialogue; #324 is the case that would have been wrongly failed.
PASSTHROUGH_SEGMENTERS = frozenset({"none", ""})


class SpeechPositiveEmptyStreak:
    """Counts consecutive scenes where speech was detected but nothing came back.

    This is the corroborating signal both #394 reporters asked for -- the thing
    that distinguishes "the recogniser stopped working" from "the speech stopped".
    Span alone cannot make that distinction, which is why it only warns.

    A scene is only counted when an external segmenter genuinely reported speech.
    Scenes where the detector found nothing are neutral: they neither extend the
    streak (silence is not a malfunction) nor reset it (a quiet gap between two
    broken stretches should not disguise them as two short ones).
    """

    def __init__(self, segmenter_name: Optional[str] = None):
        self._trustworthy = (segmenter_name or "").lower() not in PASSTHROUGH_SEGMENTERS
        self.current = 0
        self.longest = 0

    @property
    def is_meaningful(self) -> bool:
        """False when no external detector is running, so the signal is unusable."""
        return self._trustworthy

    def record(self, produced_output: bool, speech_detected: bool) -> None:
        """Record one scene's outcome."""
        if not self._trustworthy:
            return
        if produced_output:
            self.current = 0
        elif speech_detected:
            self.current += 1
            self.longest = max(self.longest, self.current)
        # else: detector found no speech and none was produced -- consistent, neutral

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (f"SpeechPositiveEmptyStreak(longest={self.longest}, "
                f"current={self.current}, meaningful={self._trustworthy})")
