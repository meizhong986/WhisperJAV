"""Tests for output coverage measurement (#394).

This module *measures*; it does not decide. The per-file state and the exit
status are decided in ``whisperjav.utils.run_outcome`` and tested there. What
is guarded here is that the measurements stay honest: a 0-byte file is empty,
a 4% span is implausible, a short clip is not assessed, and corroboration is
reported separately from span.
"""

import datetime

import pytest

from whisperjav.utils.output_coverage import (
    DEFAULT_EMPTY_STREAK_THRESHOLD,
    DEFAULT_MIN_COVERAGE,
    MIN_ASSESSABLE_DURATION_S,
    SpeechPositiveEmptyStreak,
    assess_coverage,
)


def _write_srt(path, cues):
    """cues: list of (start_s, end_s, text)."""
    import srt as srt_lib

    subs = [
        srt_lib.Subtitle(
            index=i + 1,
            start=datetime.timedelta(seconds=s),
            end=datetime.timedelta(seconds=e),
            content=t,
        )
        for i, (s, e, t) in enumerate(cues)
    ]
    path.write_text(srt_lib.compose(subs), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Empty output is measured as empty, whatever the reason
# ---------------------------------------------------------------------------

def test_zero_byte_srt_is_empty(tmp_path):
    """#263: a 0-byte SRT was written while the console printed [SUCCESS]."""
    srt = tmp_path / "c.srt"
    srt.write_text("", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.subtitle_count == 0


def test_missing_file_is_empty(tmp_path):
    report = assess_coverage(tmp_path / "nope.srt", media_duration_s=3600.0)
    assert report.verdict == "empty"


def test_none_path_is_empty():
    assert assess_coverage(None, media_duration_s=3600.0).verdict == "empty"


def test_malformed_srt_does_not_raise(tmp_path):
    srt = tmp_path / "bad.srt"
    srt.write_text("this is not an srt file\n\n-->\n", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"


# ---------------------------------------------------------------------------
# Span is measured; corroboration is reported alongside, never folded in
# ---------------------------------------------------------------------------

def test_daoran9_case_is_implausible(tmp_path):
    """#394: 8,766 s input, last cue at 376.9 s, reported as PASS."""
    srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
    report = assess_coverage(srt, media_duration_s=8766.0)
    assert report.verdict == "implausible"
    assert report.coverage_ratio == pytest.approx(376.9 / 8766.0, rel=1e-3)
    assert not report.corroborated


def test_corroboration_is_reported_separately(tmp_path):
    srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
    report = assess_coverage(
        srt, media_duration_s=8766.0,
        speech_positive_empty_streak=DEFAULT_EMPTY_STREAK_THRESHOLD,
    )
    assert report.verdict == "implausible"
    assert report.corroborated
    assert "consecutive empty results" in report.corroboration_detail
    assert "corroborated" in report.detail


def test_probe_failure_corroborates_even_full_length_output(tmp_path):
    """@13e5t: SRT stopped at ~6 min of a 10 min file; span alone cannot see it."""
    srt = _write_srt(tmp_path / "ok.srt", [(1, 2, "a"), (3500, 3590, "b")])
    report = assess_coverage(srt, media_duration_s=3600.0, probe_failed=True)
    assert report.verdict == "ok"
    assert report.corroborated
    assert "health probe" in report.corroboration_detail


def test_short_empty_streak_is_not_corroboration(tmp_path):
    srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
    report = assess_coverage(
        srt, media_duration_s=8766.0,
        speech_positive_empty_streak=DEFAULT_EMPTY_STREAK_THRESHOLD - 1,
    )
    assert not report.corroborated


def test_full_length_output_is_ok(tmp_path):
    srt = _write_srt(tmp_path / "ok.srt", [(1, 2, "a"), (3500, 3590, "b")])
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "ok"
    assert report.coverage_ratio > 0.95


def test_short_media_is_not_assessed(tmp_path):
    srt = _write_srt(tmp_path / "clip.srt", [(1, 2, "a")])
    report = assess_coverage(srt, media_duration_s=MIN_ASSESSABLE_DURATION_S - 1)
    assert report.verdict == "unknown"
    assert report.subtitle_count == 1


def test_unknown_duration_is_not_assessed(tmp_path):
    srt = _write_srt(tmp_path / "clip.srt", [(1, 2, "a")])
    for duration in (None, 0, -5):
        assert assess_coverage(srt, media_duration_s=duration).verdict == "unknown"


def test_min_coverage_zero_disables_the_ratio_check(tmp_path):
    srt = _write_srt(tmp_path / "gqn.srt", [(10, 12, "a"), (370, 376.9, "b")])
    report = assess_coverage(srt, media_duration_s=8766.0, min_coverage=0)
    assert report.verdict == "ok"


def test_threshold_boundary_is_ok(tmp_path):
    duration = 1000.0
    srt = _write_srt(tmp_path / "b.srt", [(0, DEFAULT_MIN_COVERAGE * duration, "a")])
    assert assess_coverage(srt, media_duration_s=duration).verdict == "ok"


# ---------------------------------------------------------------------------
# The corroborating signal itself (#394)
# ---------------------------------------------------------------------------

class TestSpeechPositiveEmptyStreak:
    def test_counts_consecutive_speech_positive_empties(self):
        s = SpeechPositiveEmptyStreak("silero-v3.1")
        for _ in range(4):
            s.record(produced_output=False, speech_detected=True)
        assert s.longest == 4

    def test_output_resets_the_streak(self):
        s = SpeechPositiveEmptyStreak("silero-v3.1")
        for _ in range(3):
            s.record(produced_output=False, speech_detected=True)
        s.record(produced_output=True, speech_detected=True)
        s.record(produced_output=False, speech_detected=True)
        assert s.current == 1
        assert s.longest == 3

    def test_silence_is_neutral(self):
        s = SpeechPositiveEmptyStreak("silero-v3.1")
        s.record(produced_output=False, speech_detected=True)
        s.record(produced_output=False, speech_detected=False)  # quiet gap
        s.record(produced_output=False, speech_detected=True)
        assert s.current == 2

    def test_no_speech_and_no_output_never_counts(self):
        s = SpeechPositiveEmptyStreak("silero-v3.1")
        for _ in range(10):
            s.record(produced_output=False, speech_detected=False)
        assert s.longest == 0

    @pytest.mark.parametrize("name", ["none", "", None])
    def test_passthrough_segmenter_produces_no_signal(self, name):
        """NullSpeechSegmenter returns the whole scene as 'speech' unconditionally
        (#324): under native VAD there is no detector, so no corroboration."""
        s = SpeechPositiveEmptyStreak(name)
        assert not s.is_meaningful
        for _ in range(40):
            s.record(produced_output=False, speech_detected=True)
        assert s.longest == 0
