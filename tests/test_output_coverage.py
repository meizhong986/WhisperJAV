"""Tests for output coverage assessment (#394).

The failure this guards against is a run that reports success while handing the
user a subtitle file covering a fraction of the input.  The two real-world cases
below (daoran9's 4.299% and 13e5t's ~5.7%) are encoded as regression tests so a
future change cannot quietly stop detecting them.
"""

import datetime

import pytest

from whisperjav.utils.output_coverage import (
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
# The reported failures
# ---------------------------------------------------------------------------

def test_daoran9_case_is_flagged(tmp_path):
    """#394: 8,766s input, SRT ending at 376.9s = 4.299% -> must not pass."""
    srt = _write_srt(tmp_path / "a.srt", [(370.0, 376.9, "last line")])
    report = assess_coverage(srt, media_duration_s=8766.0)
    assert report.verdict == "implausible"
    assert report.is_failure
    assert report.coverage_ratio == pytest.approx(0.043, abs=0.001)


def test_13e5t_case_is_flagged(tmp_path):
    """#394: 118-minute input whose pass 2 ended at 00:06:44."""
    srt = _write_srt(tmp_path / "b.srt", [(400.0, 404.0, "line")])
    report = assess_coverage(srt, media_duration_s=118 * 60)
    assert report.verdict == "implausible"
    assert report.is_failure


def test_zero_byte_srt_is_empty(tmp_path):
    """#263: a 0-byte SRT was written while the console printed [SUCCESS]."""
    srt = tmp_path / "c.srt"
    srt.write_text("", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.is_failure
    assert report.subtitle_count == 0


def test_missing_file_is_empty(tmp_path):
    report = assess_coverage(tmp_path / "nope.srt", media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.is_failure


def test_none_path_is_empty():
    report = assess_coverage(None, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert report.is_failure


# ---------------------------------------------------------------------------
# Healthy output must not be flagged
# ---------------------------------------------------------------------------

def test_full_length_output_is_ok(tmp_path):
    srt = _write_srt(tmp_path / "d.srt", [(10.0, 15.0, "a"), (3500.0, 3540.0, "b")])
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "ok"
    assert not report.is_failure
    assert not report.is_noteworthy


def test_speech_ending_early_is_suspicious_not_failure(tmp_path):
    """Dialogue genuinely stopping early must warn, never fail the run."""
    srt = _write_srt(tmp_path / "e.srt", [(10.0, 1800.0, "a")])
    report = assess_coverage(srt, media_duration_s=3600.0)  # 50%
    assert report.verdict == "suspicious"
    assert not report.is_failure
    assert report.is_noteworthy


# ---------------------------------------------------------------------------
# Guards against false positives
# ---------------------------------------------------------------------------

def test_short_media_is_not_assessed(tmp_path):
    """A short clip may hold one line near the start; the ratio is meaningless."""
    srt = _write_srt(tmp_path / "f.srt", [(1.0, 2.0, "hi")])
    report = assess_coverage(srt, media_duration_s=MIN_ASSESSABLE_DURATION_S - 1)
    assert report.verdict == "unknown"
    assert not report.is_failure


def test_unknown_duration_is_not_assessed(tmp_path):
    srt = _write_srt(tmp_path / "g.srt", [(1.0, 2.0, "hi")])
    for duration in (None, 0, -1):
        report = assess_coverage(srt, media_duration_s=duration)
        assert report.verdict == "unknown"
        assert not report.is_failure


def test_min_coverage_zero_disables_ratio_check(tmp_path):
    """Opting out must still report emptiness, but never fail on ratio."""
    srt = _write_srt(tmp_path / "h.srt", [(370.0, 376.9, "x")])
    report = assess_coverage(srt, media_duration_s=8766.0, min_coverage=0)
    assert not report.is_failure
    assert report.verdict == "suspicious"


def test_threshold_boundary_is_inclusive_above(tmp_path):
    """Exactly at the threshold is not a failure."""
    duration = 1000.0
    srt = _write_srt(tmp_path / "i.srt", [(0.0, duration * DEFAULT_MIN_COVERAGE, "x")])
    report = assess_coverage(srt, media_duration_s=duration)
    assert not report.is_failure


def test_malformed_srt_does_not_raise(tmp_path):
    """This runs on the failure path; it must never throw."""
    srt = tmp_path / "j.srt"
    srt.write_text("not an srt at all\n\n???", encoding="utf-8")
    report = assess_coverage(srt, media_duration_s=3600.0)
    assert report.verdict == "empty"
    assert not report.detail == ""
