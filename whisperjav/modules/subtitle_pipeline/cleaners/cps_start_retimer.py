"""CPS-anomaly start retimer for Qwen-family SRT output (v1.9.0).

Qwen output sometimes carries subtitles whose duration is far too long for
their text — e.g. a 10-second entry holding four characters. The characters-
per-second (CPS) rate is then a major anomaly: the end timestamp is right
(speech genuinely ends there) but the start was smeared backwards over
non-speech.

The legacy Whisper pipelines already fix this in ``TimingAdjuster``
(conditions (b) duration-hallucination and (d) abnormally-slow-CPS): keep the
END timestamp fixed, recompute the expected duration from text length at the
language reading speed, and move the START to ``end - expected_duration``.
The Qwen pipeline deliberately bypasses the whole legacy sanitizer stack, so
that rule never reached Qwen output. This module ports exactly those two
content-independent triggers (the other TimingAdjuster conditions compare
original-vs-sanitized text and have no meaning for an already-final SRT),
reusing the same constants from ``sanitization_constants``:

    trigger 1 (duration_hallucination): duration > MAX_SUBTITLE_DURATION (12s)
    trigger 2 (abnormally_slow_cps):    text is short in ABSOLUTE terms —
                                        text_len < MIN_SAFE_CPS *
                                        min(duration, _CPS_REF_DURATION_S)

    new_start = end - clamp(len(text) / CHARS_PER_SECOND[lang],
                            MIN_SUBTITLE_DURATION, MAX_SUBTITLE_DURATION)

    v1.9.0 duration-aware refinement of trigger 2 (owner field finding): the
    original rule (CPS < MIN_SAFE_CPS regardless of length) false-positived on
    legitimate slow JAV delivery — e.g. 8 characters drawn out over 9 seconds
    is real speech, not a smear. The actual smear pathology is TINY text over
    a long window. Capping the duration term at _CPS_REF_DURATION_S turns the
    check into an absolute-short-text test for long subtitles: at MIN_SAFE_CPS
    1.0 and reference 5s, a subtitle with >= 5 characters is never
    CPS-retimed, while 3 characters over 9 seconds still is. For durations
    <= 5s the rule is arithmetically identical to the original CPS check.
    Equivalent framing: the effective CPS floor decays as ref/duration beyond
    the reference — a slower allowed reading speed for longer dialogs.

Both triggers fire only when the current duration EXCEEDS the recomputed one,
so starts only ever move LATER. That guarantees no new overlap with the
previous subtitle is created, which is why this retimer is safe to run before
``SceneOverlapResolver`` in Phase 8 (and should — the resolver then sees the
final geometry).
"""

from pathlib import Path
from typing import Dict, Union

import pysrt

from whisperjav.config.sanitization_constants import (
    CrossSubtitleConstants,
    TimingConstants,
)
from whisperjav.utils.logger import logger

# Reference duration for the slow-CPS trigger. Below this the check equals the
# classic CPS < MIN_SAFE_CPS rule; above it the duration term stops growing, so
# only absolutely-short text (< MIN_SAFE_CPS * this many chars) is retimed.
_CPS_REF_DURATION_S = 5.0


class CpsStartRetimer:
    """Retimes anomalously long / slow-CPS subtitle starts, end fixed."""

    def __init__(self, language: str = "ja"):
        self.language = language
        self._timing = TimingConstants()
        self._cross = CrossSubtitleConstants()
        self.chars_per_second = self._cross.CHARS_PER_SECOND.get(
            language, self._cross.CHARS_PER_SECOND["default"]
        )

    def _ideal_duration_s(self, text_len: int) -> float:
        """Expected duration for text_len chars, clamped to legal bounds."""
        ideal = max(
            text_len / self.chars_per_second,
            self._timing.MIN_SUBTITLE_DURATION,
        )
        return min(ideal, self._timing.MAX_SUBTITLE_DURATION)

    def _classify(self, text_len: int, duration_s: float) -> str:
        """Return the trigger name, or '' if the entry needs no retiming.

        Mirrors TimingAdjuster's evaluation order: duration-hallucination
        (condition b) is checked before abnormally-slow-CPS (condition d).

        The slow-CPS check is duration-aware (v1.9.0): the duration term is
        capped at _CPS_REF_DURATION_S, so long subtitles are only retimed
        when their text is short in absolute terms (the smear pathology),
        never merely for slow delivery. See the module docstring.
        """
        if duration_s > self._timing.MAX_SUBTITLE_DURATION:
            return "duration_hallucination"
        if (
            text_len >= self._timing.MIN_TEXT_LENGTH_FOR_CPS_CHECK
            and duration_s > 0
            and text_len
            < self._timing.MIN_SAFE_CPS * min(duration_s, _CPS_REF_DURATION_S)
        ):
            return "abnormally_slow_cps"
        return ""

    def retime_srt_file(self, srt_path: Union[str, Path]) -> Dict[str, int]:
        """Retime anomalous entries in place (parse -> retime -> write back).

        Entry count and ordering never change (this retimer moves starts,
        it drops nothing), so no renumbering is needed.

        Returns:
            Stats dict: original_count, retimed_long_duration,
            retimed_slow_cps, retimed_total, final_count.
        """
        path = Path(srt_path)
        stats = {
            "original_count": 0,
            "retimed_long_duration": 0,
            "retimed_slow_cps": 0,
            "retimed_total": 0,
            "final_count": 0,
        }
        if not path.exists() or path.stat().st_size == 0:
            return stats

        try:
            subs = pysrt.open(str(path), encoding="utf-8")
        except Exception as e:  # pragma: no cover - defensive parse guard
            logger.warning(
                "[CpsStartRetimer] retime_srt_file failed to parse %s: %s", path, e
            )
            return stats

        stats["original_count"] = len(subs)

        for sub in subs:
            text_len = len((sub.text_without_tags or "").strip())
            duration_s = sub.duration.ordinal / 1000.0

            trigger = self._classify(text_len, duration_s)
            if not trigger:
                continue

            ideal_duration_s = self._ideal_duration_s(text_len)
            new_start_ordinal = sub.end.ordinal - int(ideal_duration_s * 1000)
            if new_start_ordinal <= sub.start.ordinal:
                # Recomputed duration is not shorter than the current one —
                # nothing to fix (can only happen for borderline entries at
                # the clamp bounds). Never move a start EARLIER.
                continue

            old_start = sub.start
            sub.start = pysrt.SubRipTime(milliseconds=new_start_ordinal)
            key = (
                "retimed_long_duration"
                if trigger == "duration_hallucination"
                else "retimed_slow_cps"
            )
            stats[key] += 1
            stats["retimed_total"] += 1
            logger.debug(
                "[CpsStartRetimer] sub %d (%s): start %s -> %s (%.1fs -> %.1fs, %d chars)",
                sub.index,
                trigger,
                old_start,
                sub.start,
                duration_s,
                ideal_duration_s,
                text_len,
            )

        if stats["retimed_total"]:
            subs.save(str(path), encoding="utf-8")

        stats["final_count"] = len(subs)
        if stats["retimed_total"]:
            logger.info(
                "[CpsStartRetimer] SRT retime %s: %d/%d entries retimed "
                "(%d duration>%.0fs, %d CPS<%.1f; ends unchanged)",
                path.name,
                stats["retimed_total"],
                stats["original_count"],
                stats["retimed_long_duration"],
                self._timing.MAX_SUBTITLE_DURATION,
                stats["retimed_slow_cps"],
                self._timing.MIN_SAFE_CPS,
            )
        return stats
