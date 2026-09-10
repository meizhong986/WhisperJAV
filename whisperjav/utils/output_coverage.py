"""Output coverage measurement (#394).

Measures how much of the source media a subtitle file spans and whether an
independent signal corroborates a recogniser failure. It *observes*; it does
not decide. The per-file state and the exit status are decided in
``whisperjav.utils.run_outcome`` from these observations.

Why this exists
---------------
WhisperJAV could finish a run, report success, and hand the user a subtitle file
covering a small fraction of the input. Reported instances:

* an 8,766 s input returning **PASS** with an SRT whose last cue ended at 376.9 s
  (4.299% of the file);
* a 118-minute input whose pass 2 produced 82 cues ending at 00:06:44;
* 314 consecutive failed scenes producing a 0-byte SRT while the console printed
  ``[SUCCESS] Process completed successfully``.

What is measured
----------------
* ``verdict``: ``"empty"`` (no cues at all), ``"unknown"`` (duration unknown or
  media too short to assess), ``"implausible"`` (span below ``min_coverage``),
  or ``"ok"``.
* ``corroborated``: whether an independent signal says the recogniser stopped
  working -- consecutive empty results while a genuine voice detector still
  reported speech, or a failed same-instance health probe.

Span alone is ambiguous: a long intro, credits, a music performance or speech
that genuinely stops early all shorten it (both #394 reporters said so, and
#324 is a music show where three external segmenters produced one cue each).
That is why span and corroboration are reported separately and why neither
fails a run by itself; see ``run_outcome`` for the contract.

This module does not diagnose *why* output is missing; the underlying cause is
still open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from whisperjav.utils.logger import logger

# Span below this fraction of the media duration is reported as implausible.
DEFAULT_MIN_COVERAGE = 0.25

# Consecutive empty ASR results, while the voice detector still reported speech,
# that constitute corroboration. Chosen well above ordinary quiet passages.
DEFAULT_EMPTY_STREAK_THRESHOLD = 5

# Media shorter than this are not assessed: a short clip can legitimately hold a
# single line near the start, making the ratio meaningless.
MIN_ASSESSABLE_DURATION_S = 120.0


@dataclass(frozen=True)
class CoverageReport:
    """Measurements of one subtitle file against its source media."""

    verdict: str  # "ok" | "implausible" | "empty" | "unknown"
    coverage_ratio: Optional[float]  # last cue end / media duration
    last_cue_end_s: Optional[float]
    media_duration_s: Optional[float]
    subtitle_count: int
    detail: str
    corroborated: bool = False
    corroboration_detail: str = ""


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
    empty_streak_threshold: int = DEFAULT_EMPTY_STREAK_THRESHOLD,
) -> CoverageReport:
    """Measure whether *srt_path* plausibly covers *media_duration_s*.

    Args:
        srt_path: the final subtitle file. A missing path yields "empty".
        media_duration_s: source duration in seconds, from media discovery.
        min_coverage: span below which the result is "implausible". Pass 0 to
            disable the ratio check; zero-cue output is still reported as empty.
        speech_positive_empty_streak: longest run of consecutive empty ASR
            results observed *while an external voice detector still reported
            speech*.

            **Only pass a non-zero value when an external segmenter is in use.**
            Under faster-whisper's built-in VAD -- the balanced default -- no
            external detection happens at all, so there is no speech signal to
            count against. v1.9.2 removed the counter that used to feed this;
            nothing in production passes a non-zero value any more.
            Issue #324 is the cautionary case: 33 consecutive empty scenes on a
            music performance where the audio genuinely held no dialogue.
        empty_streak_threshold: streak length that counts as corroboration.

    Returns:
        A :class:`CoverageReport`. Never raises.
    """
    corroborated = speech_positive_empty_streak >= empty_streak_threshold
    # A noun phrase, so it reads after "corroborated by ..." wherever it is used.
    why = ""
    if corroborated:
        why = (f"{speech_positive_empty_streak} consecutive empty results "
               "while speech was still being detected")

    def _report(verdict, ratio, last_end, count, detail):
        return CoverageReport(
            verdict, ratio, last_end, media_duration_s, count, detail, corroborated, why
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
        detail = f"{span}; corroborated by {why}" if corroborated else span
        return _report("implausible", ratio, last_end, count, detail)

    return _report("ok", ratio, last_end, count,
                   f"{count} cue(s) spanning {ratio:.1%} of the file")


# Segmenter names that are passthroughs rather than genuine speech detection.
# Under faster-whisper's native VAD -- the balanced default since v1.9.0 -- the
# Segmenter names that are not real detections. Retained for callers that
# still need to ask; v1.9.2 removed its only production consumer.
PASSTHROUGH_SEGMENTERS = frozenset({"none", ""})


# v1.9.2 (owner V1/V3): SpeechPositiveEmptyStreak was removed. Its signal --
# "the detector reported speech and nothing came back" -- is produced identically
# by a recogniser that has stopped working and by a scene with no intelligible
# speech in it, and it cannot separate the two. It was also inert under the
# built-in VAD, recording a reassuring zero for the runs that actually failed.
# `assess_coverage`'s speech_positive_empty_streak parameter is kept (default 0)
# so the exit-status contract is unchanged, but nothing feeds it any more.
