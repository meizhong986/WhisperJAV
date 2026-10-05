"""Tests for the audio integrity check (1.9.4, REQ1): the pure checks only.

The packet lists are made up, in the shapes measured on real files:
AAC packets of 1024 samples at 48 kHz (0.02133 s), and the IPZZ-912 damage --
a packet that states a 2.027 s length while holding one packet of sound, so
the gap is hidden inside the packet and the next packet starts exactly where
the stated length ends.
"""

import pytest

from whisperjav.modules.audio_integrity import (
    IntegrityReport,
    assess,
    count_decode_errors,
    find_timeline_breaks,
    track_timeline_s,
)

AAC = 1024 / 48000


def clean_packets(seconds, start=0.0):
    n = int(seconds / AAC)
    return [start + i * AAC for i in range(n)], [AAC] * n


def with_hidden_holes(seconds, hole_at, hole_len):
    """Packets where the packet at each time in hole_at states AAC + hole_len."""
    pts, dur, t = [], [], 0.0
    pending = sorted(hole_at)
    while t < seconds:
        pts.append(t)
        if pending and t >= pending[0]:
            pending.pop(0)
            dur.append(AAC + hole_len)
        else:
            dur.append(AAC)
        t += dur[-1]
    return pts, dur


FACTS = {"n_audio": 1, "has_video": True, "audio_dur": None, "video_dur": None, "format_dur": None}


def test_clean_track_has_no_findings():
    pts, dur = clean_packets(600)
    assert find_timeline_breaks(pts, dur) == []
    assert assess(FACTS, pts, dur, extracted_s=track_timeline_s(pts, dur)) == []


def test_hidden_holes_are_found_where_the_gap_rule_sees_nothing():
    pts, dur = with_hidden_holes(3000, [700, 1400, 2100], 2.006)
    # The plain rule (next start minus this end) finds nothing here.
    assert all(pts[i + 1] - (pts[i] + dur[i]) < 1e-9 for i in range(len(pts) - 1))
    found = find_timeline_breaks(pts, dur)
    assert [f.kind for f in found] == ["hole"]
    assert found[0].size_s == pytest.approx(3 * 2.006, abs=1e-6)
    assert "3 gap(s)" in found[0].detail and "0:11:40" in found[0].detail


def test_open_gap_between_packets_is_found():
    pts, dur = clean_packets(100)
    later = [p + 1.5 for p in pts[2000:]]
    pts = pts[:2000] + later
    found = find_timeline_breaks(pts, dur)
    assert [f.kind for f in found] == ["hole"]
    assert abs(found[0].size_s - 1.5) < 1e-6


def test_small_gaps_and_ordinary_jitter_stay_silent():
    pts, dur = clean_packets(100)
    pts = pts[:1000] + [p + 0.3 for p in pts[1000:]]   # below the 0.5 s starting value
    assert find_timeline_breaks(pts, dur) == []


def test_backwards_time_is_found():
    pts, dur = clean_packets(100)
    pts = pts[:2000] + [p - 2.0 for p in pts[2000:]]
    found = find_timeline_breaks(pts, dur)
    assert [f.kind for f in found] == ["backwards"]


def test_extracted_audio_shorter_than_the_track_is_found():
    pts, dur = clean_packets(3000)
    span = track_timeline_s(pts, dur)
    assert [f.kind for f in assess(FACTS, pts, dur, extracted_s=span - 3.0)] == ["length"]


def test_shortfall_the_holes_explain_is_not_reported_twice():
    pts, dur = with_hidden_holes(3000, [700], 2.006)
    span = track_timeline_s(pts, dur)
    assert [f.kind for f in assess(FACTS, pts, dur, extracted_s=span - 2.006)] == ["hole"]
    # A shortfall well beyond the holes is still its own finding.
    assert [f.kind for f in assess(FACTS, pts, dur, extracted_s=span - 6.0)] == ["hole", "length"]


def test_length_check_falls_back_to_container_duration_without_packets():
    facts = dict(FACTS, format_dur=1000.0)
    assert [f.kind for f in assess(facts, extracted_s=990.0)] == ["length"]
    assert assess(facts, extracted_s=999.5) == []


def test_audio_ending_well_before_video_is_found():
    pts, dur = clean_packets(500)
    facts = dict(FACTS, video_dur=520.0)
    assert [f.kind for f in assess(facts, pts, dur)] == ["av_length"]
    # MKV: no stream durations; the container duration stands in for the video.
    facts = dict(FACTS, format_dur=520.0)
    assert [f.kind for f in assess(facts, pts, dur)] == ["av_length"]
    # A sound-only file has no video to compare with.
    facts = dict(FACTS, has_video=False, format_dur=520.0)
    assert [f.kind for f in assess(facts, pts, dur)] == []


def test_no_audio_track():
    found = assess(dict(FACTS, n_audio=0))
    assert [f.kind for f in found] == ["no_audio"]


def test_decode_errors_are_counted_from_tagged_ffmpeg_lines():
    stderr = "\n".join([
        "[info]   Duration: 00:01:00.01, start: 0.000000, bitrate: 69 kb/s",
        "[aac @ 0000022FC513F280] [error] invalid band type",
        "[aist#0:0/aac @ 0000022FC50B6DC0] [dec:aac @ 0000022FC513EB40] [error] Error submitting packet "
        "to decoder: Invalid data found when processing input",
        "[aac @ 0000022FC513F280] [warning] something recoverable",
    ])
    assert count_decode_errors(stderr) == 2
    assert count_decode_errors("") == 0
    assert [f.kind for f in assess(FACTS, decode_error_lines=2)] == ["decode_errors"]


def test_report_summary_names_every_finding():
    pts, dur = with_hidden_holes(3000, [700], 2.006)
    report = IntegrityReport(findings=assess(FACTS, pts, dur, extracted_s=track_timeline_s(pts, dur) - 2.006))
    assert report.suspect
    assert report.summary().startswith("audio integrity: 1 gap(s)")
    assert IntegrityReport().summary() == "" and not IntegrityReport().suspect


# ---------------------------------------------------------------------------
# What reaches the run: the reason, the stop under --fail-on suspect, the text
# ---------------------------------------------------------------------------

from whisperjav.modules.audio_integrity import (  # noqa: E402
    STOP_ENV,
    AudioIntegrityStop,
    apply_to_run,
    console_message,
)


def _damaged_report():
    pts, dur = with_hidden_holes(3000, [700], 2.006)
    return IntegrityReport(findings=assess(FACTS, pts, dur), seconds=3.6)


def test_damaged_audio_adds_its_reason_to_the_run_summary(monkeypatch):
    monkeypatch.delenv(STOP_ENV, raising=False)
    degradations = []
    apply_to_run(_damaged_report(), degradations)
    assert len(degradations) == 1 and degradations[0].startswith("audio integrity: 1 gap(s)")


def test_clean_or_unchecked_audio_adds_nothing(monkeypatch):
    monkeypatch.setenv(STOP_ENV, "1")
    degradations = []
    apply_to_run(IntegrityReport(), degradations)
    apply_to_run(IntegrityReport(checked=False), degradations)
    apply_to_run(None, degradations)
    assert degradations == []


def test_ensemble_pass_2_does_not_repeat_the_reason(monkeypatch):
    monkeypatch.delenv(STOP_ENV, raising=False)
    degradations = []
    apply_to_run(_damaged_report(), degradations, in_summary=False)
    assert degradations == []


def test_fail_on_suspect_stops_a_damaged_file(monkeypatch):
    monkeypatch.setenv(STOP_ENV, "1")
    with pytest.raises(AudioIntegrityStop, match="^stopped before transcription: audio integrity: "):
        apply_to_run(_damaged_report(), [], in_summary=False)


def test_console_wording():
    rep = _damaged_report()
    assert console_message(IntegrityReport(seconds=3.6), "a.mp4") == "Audio check: no problems found (3.6 s)."
    msg = console_message(rep, "IPZZ-912.mp4")
    assert msg.startswith("Audio check: the audio track of IPZZ-912.mp4 has 1 gap(s), 2.0 s in all (2.0 s at 0:11:40)")
    assert msg.endswith("WhisperJAV filled the gaps with silence, so subtitle times stay in step with the video; "
                        "any speech inside the gaps is missing from the file. Transcription continues.")
    assert console_message(rep, "x.mp4", stopping=True).endswith(
        "This file is stopped before transcription (--fail-on suspect).")
    unchecked = IntegrityReport(checked=False, facts={"not_checked": "ffprobe not found"})
    assert console_message(unchecked, "x.mp4") == "Audio check: not run (ffprobe not found)."


# ---------------------------------------------------------------------------
# Extraction command: an FFmpeg that rejects "-loglevel level+info"
# ---------------------------------------------------------------------------

def test_extraction_retries_without_loglevel_on_an_old_ffmpeg(monkeypatch):
    import subprocess
    from whisperjav.modules.audio_extraction import AudioExtractor
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if "-loglevel" in cmd:
            raise subprocess.CalledProcessError(1, cmd, stderr='Invalid loglevel "level+info". Possible levels ...')
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    AudioExtractor._run_ffmpeg(["ffmpeg", "-loglevel", "level+info", "-i", "in.mp4", "out.wav"])
    assert calls == [["ffmpeg", "-loglevel", "level+info", "-i", "in.mp4", "out.wav"],
                     ["ffmpeg", "-i", "in.mp4", "out.wav"]]


def test_extraction_does_not_retry_other_ffmpeg_errors(monkeypatch):
    import subprocess
    from whisperjav.modules.audio_extraction import AudioExtractor

    def fake_run(cmd, **kwargs):
        raise subprocess.CalledProcessError(1, cmd, stderr="in.mp4: Invalid data found when processing input")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        AudioExtractor._run_ffmpeg(["ffmpeg", "-loglevel", "level+info", "-i", "in.mp4", "out.wav"])


def test_extraction_fills_holes_without_first_pts():
    """first_pts=0 must not come back: with a mid-file sample-rate change FFmpeg rebuilds
    the filter and a rebuilt first_pts=0 inserts silence as long as the time already
    played (adversary review 2026-10-05)."""
    import inspect
    from whisperjav.modules import audio_extraction
    src = inspect.getsource(audio_extraction.AudioExtractor.extract)
    assert '"-af", "aresample=async=1",' in src
    assert 'aresample=async=1:first_pts' not in src
