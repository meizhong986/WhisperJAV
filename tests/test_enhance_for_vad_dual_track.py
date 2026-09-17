#!/usr/bin/env python3
"""
Tests for "Enhance for VAD only" doing what its name says in the fidelity pipeline.

The setting is meant to give the speech detector the cleaned-up audio while the
recogniser still hears the original, so that clean-up helps find the speech
without colouring the words. Qwen already did that. Fidelity read the setting,
said so in the log, and then sent the cleaned-up audio to both. Owner, 2026-09-17:
"please make the VAD Only Enhancement feature audio separation path to work for
balanced and fidelity."

This pins the fidelity half: the recogniser's transcribe() now takes a second
file to detect speech in, and the two tracks must be the same recording at the
same rate or it refuses rather than putting subtitles in the wrong place.

Balanced is NOT covered here: its speech detection happens inside faster-whisper's
own transcribe() call, on the same buffer it recognises, so the split cannot be
made without changing what the balanced pipeline is -- which is the owner's to
decide.

Run with: pytest tests/test_enhance_for_vad_dual_track.py -v
"""

import inspect

import numpy as np
import pytest
import soundfile as sf

from whisperjav.modules.whisper_pro_asr import WhisperProASR

SAMPLE_RATE = 16000


class _Stop(Exception):
    """Ends the run right after segmentation, which is all these tests need."""


@pytest.fixture
def asr(monkeypatch):
    """
    A recogniser that is never loaded: only the audio handling before
    segmentation is under test, and loading a Whisper model here would make
    these tests slow and machine-dependent.
    """
    instance = object.__new__(WhisperProASR)
    instance.task = "transcribe"
    instance._last_vad_segments = []
    instance.seen = {}

    def fake_segmentation(audio_data, sample_rate):
        instance.seen["audio"] = np.asarray(audio_data).copy()
        instance.seen["sample_rate"] = sample_rate
        raise _Stop()

    monkeypatch.setattr(instance, "_run_speech_segmentation", fake_segmentation)
    monkeypatch.setattr(instance, "get_segmenter_name", lambda: "silero-v3.1")
    return instance


def _write(path, seconds=1.0, value=0.25, sample_rate=SAMPLE_RATE):
    audio = np.full(int(seconds * sample_rate), value, dtype=np.float32)
    sf.write(str(path), audio, sample_rate)
    return audio


class TestTheSignature:
    def test_transcribe_takes_a_detection_track(self):
        assert "vad_audio_path" in inspect.signature(WhisperProASR.transcribe).parameters

    def test_transcribe_to_srt_passes_it_on(self):
        assert "vad_audio_path" in inspect.signature(
            WhisperProASR.transcribe_to_srt).parameters

    def test_it_is_optional(self):
        assert inspect.signature(
            WhisperProASR.transcribe).parameters["vad_audio_path"].default is None


class TestWhichTrackIsUsed:
    def test_without_it_the_one_file_is_used_for_both(self, asr, tmp_path):
        main = tmp_path / "scene.wav"
        expected = _write(main, value=0.25)
        with pytest.raises(_Stop):
            asr.transcribe(main)
        assert np.allclose(asr.seen["audio"], expected)

    def test_with_it_the_detector_gets_the_cleaned_up_copy(self, asr, tmp_path):
        main = tmp_path / "scene.wav"
        cleaned = tmp_path / "scene_enhanced.wav"
        _write(main, value=0.25)
        cleaned_audio = _write(cleaned, value=0.75)

        with pytest.raises(_Stop):
            asr.transcribe(main, vad_audio_path=cleaned)

        assert np.allclose(asr.seen["audio"], cleaned_audio), (
            "the detector should have been given the cleaned-up audio")
        assert asr.seen["sample_rate"] == SAMPLE_RATE


class TestItRefusesTracksThatDoNotMatch:
    def test_a_different_sample_rate_is_refused(self, asr, tmp_path):
        main = tmp_path / "scene.wav"
        cleaned = tmp_path / "scene_enhanced.wav"
        _write(main, sample_rate=16000)
        _write(cleaned, sample_rate=48000)

        with pytest.raises(ValueError) as excinfo:
            asr.transcribe(main, vad_audio_path=cleaned)
        message = str(excinfo.value)
        assert "sample rate" in message
        assert "16000" in message and "48000" in message

    def test_a_different_recording_is_refused(self, asr, tmp_path):
        main = tmp_path / "scene.wav"
        cleaned = tmp_path / "other_scene.wav"
        _write(main, seconds=1.0)
        _write(cleaned, seconds=3.0)

        with pytest.raises(ValueError) as excinfo:
            asr.transcribe(main, vad_audio_path=cleaned)
        assert "same recording" in str(excinfo.value)

    def test_a_rounding_difference_is_tolerated(self, asr, tmp_path):
        """Enhancement can return a handful of samples more or fewer."""
        main = tmp_path / "scene.wav"
        cleaned = tmp_path / "scene_enhanced.wav"
        _write(main, seconds=1.0)
        sf.write(str(cleaned),
                 np.full(SAMPLE_RATE + 40, 0.75, dtype=np.float32), SAMPLE_RATE)

        with pytest.raises(_Stop):
            asr.transcribe(main, vad_audio_path=cleaned)
        assert len(asr.seen["audio"]) == SAMPLE_RATE + 40

    def test_an_unreadable_detection_track_is_reported(self, asr, tmp_path):
        main = tmp_path / "scene.wav"
        cleaned = tmp_path / "broken.wav"
        _write(main)
        cleaned.write_bytes(b"not a wav file")

        with pytest.raises(Exception) as excinfo:
            asr.transcribe(main, vad_audio_path=cleaned)
        assert not isinstance(excinfo.value, _Stop)


class TestTheFidelityPipelineWiring:
    """Read from the pipeline source: it must pair the two tracks by position."""

    def test_it_builds_both_tracks_and_pairs_them(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
                  / "fidelity_pipeline.py").read_text(encoding="utf-8")
        assert "vad_scene_paths = enhanced_paths" in source
        assert "resample_scenes(scene_paths, self.temp_dir)" in source
        assert "vad_audio_path=vad_scene_path" in source
        # And it checks the pairing rather than assuming it.
        assert "len(vad_scene_paths) != len(scene_paths)" in source

    def test_the_single_track_path_is_unchanged(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
                  / "fidelity_pipeline.py").read_text(encoding="utf-8")
        # Without the setting, the enhanced scenes are what everything uses.
        assert "scene_paths = enhanced_paths" in source


class TestBalancedDoesNotOfferIt:
    """
    Owner, 2026-09-17: "balanced shall not have the VAD only enhancement."
    Balanced finds the speech inside faster-whisper's own call, on the audio it
    transcribes, so there is no second track to hand the cleaned-up audio to.
    Accepting the flag and enhancing both -- what happened up to v1.9.2 -- is the
    silent difference the other balanced rules exist to prevent.
    """

    @pytest.mark.parametrize("n", [1, 2])
    def test_the_flag_is_refused_on_a_balanced_pass(self, n, tmp_path):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4",
             "--dump-params", str(tmp_path / "params.json"), "--ensemble",
             f"--pass{n}-pipeline", "balanced",
             f"--pass{n}-speech-enhancer", "clearvoice",
             f"--pass{n}-enhance-for-vad"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 2
        combined = result.stdout + result.stderr
        assert f"--pass{n}-enhance-for-vad is not available" in combined
        assert "fidelity or qwen" in combined

    @pytest.mark.parametrize("pipeline", ["fidelity", "qwen"])
    def test_it_is_still_accepted_where_it_works(self, pipeline, tmp_path):
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "dummy.mp4",
             "--dump-params", str(tmp_path / "params.json"), "--ensemble",
             "--pass1-pipeline", pipeline,
             "--pass1-speech-enhancer", "clearvoice", "--pass1-enhance-for-vad",
             "--pass2-pipeline", "balanced"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=300)
        assert result.returncode == 0, (result.stdout + result.stderr)[-500:]

    def test_the_gui_hides_the_box_for_a_balanced_pass(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui"
                  / "assets" / "app.js").read_text(encoding="utf-8")
        assert "const isBalanced = this.state[passId]?.pipeline === 'balanced';" in source
        assert "&& !isXxl && !isBalanced) ? 'block' : 'none';" in source

    def test_the_gui_does_not_send_the_flag_for_a_balanced_pass(self):
        from pathlib import Path
        source = (Path(__file__).resolve().parents[1] / "whisperjav" / "webview_gui"
                  / "api.py").read_text(encoding="utf-8")
        assert source.count("get('pipeline') != 'balanced'") == 2
