#!/usr/bin/env python3
"""Tests for the v1.9.2 audio analytics (whisperjav/modules/analytics.py).

The module tells a user which scenes look likely to lose speech and which look
acoustically difficult. It must never be able to disturb a run, and its two
thresholds must keep reproducing the measurements they were set from.

Synthetic audio is used so the tests are fast and need no media files. The
threshold values themselves were calibrated on real material -- HODV-22019 and the
seven Netflix clips with human Japanese subtitles -- and that calibration is
recorded in the module docstring; these tests guard the behaviour around the
thresholds, not the calibration.

Run with: pytest tests/test_analytics_v192.py
"""

import math
import wave

import numpy as np
import pytest

from whisperjav.modules import analytics


# ---------------------------------------------------------------- helpers

def write_wav(path, samples, rate=16000):
    """Write mono 16 kHz pcm_s16le, the format AudioExtractor always produces."""
    data = np.clip(samples, -1.0, 1.0)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((data * 32767).astype("<i2").tobytes())
    return path


def tone_and_silence(seconds=10.0, rate=16000, amplitude=0.2):
    """Alternating half-second tone and silence -- not speech, but loud/quiet."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    wave_ = amplitude * np.sin(2 * math.pi * 220 * t)
    gate = ((t * 2).astype(int) % 2 == 0).astype(np.float32)
    return (wave_ * gate).astype(np.float32)


# ---------------------------------------------------- thresholds are the agreed ones

def test_thresholds_are_the_calibrated_values():
    """The two lines the owner approved, guarded against silent drift.

    2.0 is a WHOLE-FILE figure: HODV-22019, which loses two thirds of its dialogue,
    measures 2.54, and nine files that transcribe correctly measure 1.13 to 1.71.
    Per scene the same measure does not separate them at all, which is why the quiet
    finding is made once for the file.

    3 dB came from the seven scored Netflix clips, where it flags exactly the three
    worst-transcribed ones and nothing else.
    """
    assert analytics.QUIET_RATIO == 2.0
    assert analytics.DIFFICULT_SNR_DB == 3.0
    assert analytics.PROBE_THRESHOLD == 0.15
    assert analytics.REFERENCE_THRESHOLD == 0.40
    assert analytics.QUIET_MIN_SPEECH_S == 2.0


# ---------------------------------------------------------------- records

def test_finding_is_serialisable():
    """Findings must survive as data, so later features read records not English."""
    f = analytics.Finding("quiet", 3, 12.5, 40.0, {"ratio": 2.345678})
    d = f.to_dict()
    assert d["kind"] == "quiet"
    assert d["scene_index"] == 3
    assert d["measurements"]["ratio"] == 2.3457
    import json
    json.dumps(d)  # must not raise


def test_analytics_speech_share_handles_empty():
    r = analytics.AudioAnalytics(filename="x.wav")
    assert r.speech_share == 0.0
    assert r.of_kind("quiet") == []
    import json
    json.dumps(r.to_dict())


# ---------------------------------------------------------------- rendering

def _result_with(findings, scenes_measured=10, total=1500.0, speech=77.0):
    r = analytics.AudioAnalytics(filename="film.wav", scene_count=10,
                                 total_duration_sec=total, speech_duration_sec=speech,
                                 scenes_measured=scenes_measured)
    r.findings = findings
    return r


def test_render_prints_only_the_summary_when_nothing_is_wrong():
    text = "\n".join(analytics.render(_result_with([])))
    assert "10 scenes" in text
    assert "unusually quiet" not in text
    assert "acoustically difficult" not in text
    assert "--vad-threshold" not in text


def test_render_states_quiet_for_the_whole_file_with_advice():
    findings = [analytics.Finding("quiet", None, 0.0, 1500.0, {"ratio": 2.34})]
    text = "\n".join(analytics.render(_result_with(findings)))
    assert "This file appears to be unusually quiet" in text
    assert "--vad-threshold 0.15" in text
    # 2026-09-12: this used to assert `--speech-enhancement ffmpeg-dsp`, a flag that
    # does not exist — argparse rejects it with exit 2 — so the test was holding the
    # bug in place rather than catching it. Enhancement is only reachable on a
    # two-pass run or --mode qwen, and the advice now says so.
    assert "--speech-enhancement" not in text
    assert "--pass1-speech-enhancer ffmpeg-dsp" in text
    assert "--qwen-enhancer ffmpeg-dsp" in text


def test_render_never_lists_scenes_for_the_quiet_finding():
    """Quiet is judged for the whole file and must not be attributed to scenes.

    Per scene the measure does not work: healthy films contain single scenes
    scoring up to 4.25, higher than five of the seven scenes that actually lost
    dialogue, and the scene that lost the most lines scored only 1.83. Naming
    scenes here would point the user at the wrong places.
    """
    findings = [analytics.Finding("quiet", None, 0.0, 1500.0, {"ratio": 2.34})]
    text = "\n".join(analytics.render(_result_with(findings)))
    assert "None" not in text
    assert "scene " not in text.split("acoustically difficult")[0]


def test_render_difficult_alone_gives_no_vad_advice():
    """A Qwen run has no quiet finding, so it must not suggest a Silero setting."""
    findings = [analytics.Finding("difficult", 1, 10.0, 50.0, {"snr_db": 2.1})]
    text = "\n".join(analytics.render(_result_with(findings)))
    assert "acoustically difficult" in text
    assert "--vad-threshold" not in text


def test_render_uses_hedged_wording():
    """The owner asked for 'it appears that' so expectations stay managed."""
    findings = [analytics.Finding("quiet", None, 0.0, 90.0, {"ratio": 2.5}),
                analytics.Finding("difficult", 1, 90.0, 180.0, {"snr_db": 1.0})]
    text = "\n".join(analytics.render(_result_with(findings)))
    # both forms hedge: "appears to be" for one scene, "appear to be" for several
    assert "appear to be" in text or "appears to be" in text
    assert "may be missing" in text
    for absolute in ("will be missing", "are missing", "has failed"):
        assert absolute not in text


def test_render_never_uses_the_scene_classifier_labels():
    """Those labels are unusable: a clip tagged 100% ambient_noise was the second
    best in the set, 56 subtitles and the highest SNR measured anywhere."""
    findings = [analytics.Finding("quiet", None, 0.0, 90.0, {"ratio": 2.5}),
                analytics.Finding("difficult", 1, 90.0, 180.0, {"snr_db": 1.0})]
    text = "\n".join(analytics.render(_result_with(findings))).lower()
    for label in ("ambient_noise", "ambient noise", "high_energy", "music_dominant",
                  "quiet_dialogue", "noisy_dialogue"):
        assert label not in text


# ---------------------------------------------------------------- measuring

def test_analyse_reads_a_real_wav_and_counts_scenes(tmp_path):
    path = write_wav(tmp_path / "a.wav", tone_and_silence(12.0))
    r = analytics.analyse(path, [(0, 0.0, 6.0), (1, 6.0, 12.0)])
    assert r.scene_count == 2
    assert r.scenes_measured == 2
    assert r.total_duration_sec == pytest.approx(12.0, abs=0.1)
    assert r.filename == "a.wav"


def test_analyse_skips_scenes_under_a_second(tmp_path):
    path = write_wav(tmp_path / "b.wav", tone_and_silence(6.0))
    r = analytics.analyse(path, [(0, 0.0, 0.4), (1, 1.0, 5.0)])
    assert r.scenes_measured == 1


def test_vad_threshold_none_suppresses_the_quiet_check(tmp_path, monkeypatch):
    """Qwen and anime-whisper segment with WhisperSeg, so a Silero-threshold
    finding would recommend a setting those runs never use.

    Silero is stubbed to a ratio of 3.0 so the quiet condition WOULD fire; the
    point of the test is that passing None suppresses it anyway. Without the stub
    the synthetic audio yields no speech and the assertion proves nothing.
    """
    import numpy as np

    def fake_speech(audio, threshold):
        seconds = 30.0 if threshold == analytics.PROBE_THRESHOLD else 10.0
        return seconds, np.ones(len(audio), dtype=bool)

    monkeypatch.setattr(analytics, "_speech_seconds", fake_speech)
    path = write_wav(tmp_path / "c.wav", tone_and_silence(20.0))

    assert analytics.analyse(path, [(0, 0.0, 20.0)], vad_threshold=None).of_kind("quiet") == []
    # the same audio WITH a threshold does produce the finding, proving the stub fires
    assert analytics.analyse(path, [(0, 0.0, 20.0)], vad_threshold=0.40).of_kind("quiet")


def test_quiet_finding_is_whole_file_not_per_scene(tmp_path, monkeypatch):
    """Exactly ONE quiet finding, carrying no scene index, however many scenes.

    Silero VAD is stubbed so the quiet condition is guaranteed to fire -- three
    scenes each with 10 s of speech at the run's threshold and 30 s at the
    permissive one, a ratio of 3.0. Without the stub this test passed on synthetic
    audio that contains no speech at all, so it asserted nothing.
    """
    import numpy as np

    def fake_speech(audio, threshold):
        seconds = 30.0 if threshold == analytics.PROBE_THRESHOLD else 10.0
        return seconds, np.ones(len(audio), dtype=bool)

    monkeypatch.setattr(analytics, "_speech_seconds", fake_speech)
    path = write_wav(tmp_path / "h.wav", tone_and_silence(30.0))
    r = analytics.analyse(path, [(0, 0.0, 10.0), (1, 10.0, 20.0), (2, 20.0, 30.0)])

    quiet = r.of_kind("quiet")
    assert len(quiet) == 1, "quiet must be judged once for the file, not per scene"
    assert quiet[0].scene_index is None, "a file-level finding must not name a scene"
    assert quiet[0].to_dict()["scope"] == "file"
    assert quiet[0].measurements["ratio"] == pytest.approx(3.0)


# ---------------------------------------------------------------- isolation

def test_report_skips_non_semantic_scene_detection(tmp_path, caplog):
    path = write_wav(tmp_path / "d.wav", tone_and_silence(4.0))
    assert analytics.report(path, [(0, 0.0, 4.0)], scene_method="auditok") is None
    assert analytics.report(path, [(0, 0.0, 4.0)], scene_method="silero") is None


def test_report_returns_none_for_no_scenes(tmp_path):
    path = write_wav(tmp_path / "e.wav", tone_and_silence(4.0))
    assert analytics.report(path, [], scene_method="semantic") is None


def test_report_swallows_a_missing_file():
    assert analytics.report("/nonexistent/nope.wav", [(0, 0.0, 4.0)],
                            scene_method="semantic") is None


def test_report_swallows_any_internal_fault(tmp_path, monkeypatch):
    """The contract that matters: a fault in analytics costs the user a report,
    never their subtitles. Nothing may propagate to the pipeline."""
    path = write_wav(tmp_path / "f.wav", tone_and_silence(4.0))

    def boom(*a, **k):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(analytics, "analyse", boom)
    assert analytics.report(path, [(0, 0.0, 4.0)], scene_method="semantic") is None

    monkeypatch.setattr(analytics, "render", boom)
    assert analytics.report(path, [(0, 0.0, 4.0)], scene_method="semantic") is None


def test_report_prints_at_info_not_debug(tmp_path, caplog):
    """The owner asked for INFO: a user who never passes --debug is exactly the
    user who needs to know their audio is difficult."""
    import logging
    path = write_wav(tmp_path / "g.wav", tone_and_silence(8.0))
    logger = logging.getLogger("whisperjav")
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Capture()
    logger.addHandler(handler)
    previous, logger.propagate = logger.propagate, False
    try:
        analytics.report(path, [(0, 0.0, 8.0)], scene_method="semantic")
    finally:
        logger.removeHandler(handler)
        logger.propagate = previous

    info = [r for r in records if r.levelno == logging.INFO]
    assert info, "analytics printed nothing at INFO"
    assert any("Audio analytics" in r.getMessage() for r in info)


# ---------------------------------------------------------------- helpers used above

def test_mmss_formats_offsets_a_user_can_seek_to():
    assert analytics._mmss(0) == "0:00"
    assert analytics._mmss(59.9) == "0:59"
    assert analytics._mmss(230.9) == "3:50"
    assert analytics._mmss(1467.4) == "24:27"
    assert analytics._mmss(7382) == "2:03:02"   # a feature film needs the hour
    assert analytics._mmss(-5) == "0:00"


class TestTheAdviceNamesRealFlags:
    """The notice tells the user what to type. Until 2026-09-12 it told them to type
    `--speech-enhancement ffmpeg-dsp`, which argparse rejects with exit 2 — and on the
    single-pass Balanced and Fidelity runs where the notice appears, there is no
    speech-enhancement control at all, on the command line or in the GUI."""

    def _advice_lines(self):
        from whisperjav.modules.analytics import AudioAnalytics, Finding, render
        r = AudioAnalytics(filename="x.mp4", scene_count=4, total_duration_sec=1500.0,
                           speech_duration_sec=90.0, scenes_measured=4)
        r.findings.append(Finding("quiet", None, 0.0, 1500.0, {"ratio": 2.54}))
        return render(r)

    def test_every_flag_in_the_advice_is_a_real_flag(self):
        import re
        import subprocess
        import sys
        help_text = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--help"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=600,
        ).stdout or ""
        assert "--mode" in help_text, "could not read --help"

        flags = set()
        for line in self._advice_lines():
            flags.update(re.findall(r"--[a-z0-9][a-z0-9-]+", line))
        assert flags, "the advice named no flags at all"
        unknown = sorted(f for f in flags if f not in help_text)
        assert not unknown, f"advice names flags that do not exist: {unknown}"

    def test_it_does_not_offer_enhancement_as_a_single_pass_option(self):
        """There is no single-pass enhancer flag, so the advice must not imply one."""
        text = "\n".join(self._advice_lines())
        assert "--speech-enhancement" not in text
        assert "--vad-threshold" in text
        # If enhancement is mentioned it must be the reachable forms.
        if "enhancer" in text:
            assert "--pass1-speech-enhancer" in text or "--qwen-enhancer" in text
