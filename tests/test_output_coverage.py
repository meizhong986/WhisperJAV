"""Tests for output coverage assessment (#394).

The failure being guarded against is a run that reports success while handing the
user a subtitle file covering a fraction of the input. The real-world cases from
the issue are encoded as regressions so a future change cannot quietly stop
detecting them.

The failure *policy* here was set by the two reporters rather than by us: span on
its own only warns, because speech can legitimately stop early, and a run is
failed only when there are no cues at all or when short output is corroborated by
evidence that the recogniser stopped working. See the module docstring.
"""

import datetime

import pytest

from whisperjav.utils.output_coverage import (
    DEFAULT_EMPTY_STREAK_THRESHOLD,
    DEFAULT_MIN_COVERAGE,
    MIN_ASSESSABLE_DURATION_S,
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
# Unusable output — failed without needing corroboration
# ---------------------------------------------------------------------------

def test_zero_byte_srt_is_a_failure(tmp_path):
    """#263: a 0-byte SRT was written while the console printed [SUCCESS]."""
    srt = tmp_path / "c.srt"
    srt.write_text("", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.is_failure
    assert report.subtitle_count == 0


def test_missing_file_is_a_failure(tmp_path):
    report = assess_coverage(tmp_path / "nope.srt", media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.is_failure


def test_none_path_is_a_failure():
    assert assess_coverage(None, media_duration_s=3600.0).is_failure


def test_malformed_srt_does_not_raise(tmp_path):
    """This runs on the failure path; it must never throw."""
    srt = tmp_path / "j.srt"
    srt.write_text("not an srt at all\n\n???", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.detail


# ---------------------------------------------------------------------------
# The reported failures — warn alone, fail when corroborated
# ---------------------------------------------------------------------------

def test_daoran9_case_warns_without_corroboration(tmp_path):
    """#394: 8,766s input, SRT ending at 376.9s = 4.299%.

    Flagged, but NOT failed on span alone — @daoran9 explicitly asked that span
    not be the sole hard-failure signal.
    """
    srt = _write_srt(tmp_path / "a.srt", [(370.0, 376.9, "last line")])
    report = assess_coverage(srt, media_duration_s=8766.0)
    assert report.verdict == "implausible"
    assert report.is_noteworthy
    assert not report.is_failure
    assert report.coverage_ratio == pytest.approx(0.043, abs=0.001)


def test_daoran9_case_fails_when_corroborated(tmp_path):
    """The same run, with the recogniser observed to have stopped working."""
    srt = _write_srt(tmp_path / "a.srt", [(370.0, 376.9, "last line")])
    report = assess_coverage(
        srt, media_duration_s=8766.0,
        speech_positive_empty_streak=DEFAULT_EMPTY_STREAK_THRESHOLD,
    )
    assert report.verdict == "implausible"
    assert report.corroborated
    assert report.is_failure
    assert "consecutive empty results" in report.detail


def test_probe_failure_alone_corroborates(tmp_path):
    """@AlanZ-Git's same-instance probe is the other accepted signal."""
    srt = _write_srt(tmp_path / "a.srt", [(370.0, 376.9, "x")])
    report = assess_coverage(srt, media_duration_s=8766.0, probe_failed=True)
    assert report.is_failure
    assert "health probe failed" in report.detail


def test_13e5t_60_percent_case_is_caught_by_corroboration(tmp_path):
    """#394: a 10-minute file whose SRT stopped at ~6 minutes.

    No span threshold safe enough to deploy would catch 60%. This is precisely
    why the empty-streak signal exists, and the reason span alone is insufficient.
    """
    srt = _write_srt(tmp_path / "e.srt", [(350.0, 360.0, "x")])
    duration = 600.0

    span_only = assess_coverage(srt, media_duration_s=duration)
    assert not span_only.is_failure, "60% must not fail on span alone"

    corroborated = assess_coverage(
        srt, media_duration_s=duration,
        speech_positive_empty_streak=DEFAULT_EMPTY_STREAK_THRESHOLD,
    )
    assert corroborated.is_noteworthy
    assert "consecutive empty results" in corroborated.detail


def test_full_length_output_with_probe_failure_is_still_flagged(tmp_path):
    """Corroboration matters even when the span looks healthy."""
    srt = _write_srt(tmp_path / "f.srt", [(10.0, 15.0, "a"), (3500.0, 3540.0, "b")])
    report = assess_coverage(srt, media_duration_s=3600.0, probe_failed=True)
    assert report.verdict == "suspicious"
    assert report.is_noteworthy


# ---------------------------------------------------------------------------
# Healthy output must not be flagged
# ---------------------------------------------------------------------------

def test_full_length_output_is_ok(tmp_path):
    srt = _write_srt(tmp_path / "d.srt", [(10.0, 15.0, "a"), (3500.0, 3540.0, "b")])
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "ok"
    assert not report.is_failure
    assert not report.is_noteworthy


def test_speech_ending_early_warns_but_never_fails(tmp_path):
    """The false positive both reporters were most concerned about."""
    srt = _write_srt(tmp_path / "e.srt", [(10.0, 1800.0, "a")])
    report = assess_coverage(srt, media_duration_s=3600.0)  # 50%
    assert report.verdict == "suspicious"
    assert not report.is_failure
    assert report.is_noteworthy


def test_short_empty_streak_is_not_corroboration(tmp_path):
    """A few quiet scenes are ordinary and must not fail a run."""
    srt = _write_srt(tmp_path / "g.srt", [(370.0, 376.9, "x")])
    report = assess_coverage(
        srt, media_duration_s=8766.0,
        speech_positive_empty_streak=DEFAULT_EMPTY_STREAK_THRESHOLD - 1,
    )
    assert not report.corroborated
    assert not report.is_failure


# ---------------------------------------------------------------------------
# Guards against false positives
# ---------------------------------------------------------------------------

def test_short_media_is_not_assessed(tmp_path):
    srt = _write_srt(tmp_path / "h.srt", [(1.0, 2.0, "hi")])
    report = assess_coverage(srt, media_duration_s=MIN_ASSESSABLE_DURATION_S - 1)
    assert report.verdict == "unknown"
    assert not report.is_failure


def test_unknown_duration_is_not_assessed(tmp_path):
    srt = _write_srt(tmp_path / "i.srt", [(1.0, 2.0, "hi")])
    for duration in (None, 0, -1):
        report = assess_coverage(srt, media_duration_s=duration)
        assert report.verdict == "unknown"
        assert not report.is_failure


def test_min_coverage_zero_disables_the_ratio_check(tmp_path):
    srt = _write_srt(tmp_path / "k.srt", [(370.0, 376.9, "x")])
    report = assess_coverage(srt, media_duration_s=8766.0, min_coverage=0)
    assert not report.is_failure
    assert report.verdict == "suspicious"


def test_threshold_boundary_is_not_a_failure(tmp_path):
    duration = 1000.0
    srt = _write_srt(tmp_path / "l.srt", [(0.0, duration * DEFAULT_MIN_COVERAGE, "x")])
    report = assess_coverage(srt, media_duration_s=duration)
    assert not report.is_failure
