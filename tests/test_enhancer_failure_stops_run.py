#!/usr/bin/env python3
"""
A clean-up the user chose that fails stops the run -- at any point, in every
pipeline -- and the scenes that do come back are all at 16kHz.

Owner, 2026-09-17: *"yes stop."* Until then, a scene whose clean-up failed was
replaced by the untouched original and the run carried on to exit 0. The user got
a subtitle file made partly from audio they had asked to have cleaned up, with
only a warning in the log to explain it. The rule already covered a clean-up that
could not start (a missing package); it now covers one that fails part-way --
weights that cannot be fetched, a card out of memory, a backend that errors.

Every pipeline enhances its scenes through ``enhance_scenes``, so this one
function carries the rule for all of them.

The sample-rate half of this file comes from the same review. Scenes are
extracted at 48kHz when an enhancer is configured, and ``resample_scenes`` puts
the untouched originals at 16kHz for the "enhance for VAD only" split. The two
lists are paired scene by scene and the recogniser refuses two tracks recorded at
different rates, so anything returned at the wrong rate fails the whole file with
a message about sample rates rather than about what actually went wrong.

Run with: pytest tests/test_enhancer_failure_stops_run.py -v
"""

import numpy as np
import pytest
import soundfile as sf

from whisperjav.modules.speech_enhancement.base import (
    EnhancementResult,
    SpeechEnhancerUnavailable,
    create_failed_result,
)
from whisperjav.modules.speech_enhancement.pipeline_helper import (
    TARGET_SAMPLE_RATE,
    enhance_scenes,
    resample_scenes,
)

EXTRACTION_RATE = 48000


class StubEnhancer:
    """Stands in for a real backend; no model, no download."""

    name = "stub"
    display_name = "Stub enhancer"

    def __init__(self, mode):
        # mode: "works", "returns_failure", or "raises"
        self.mode = mode

    def get_preferred_sample_rate(self):
        return EXTRACTION_RATE

    def enhance(self, audio, sample_rate):
        if self.mode == "raises":
            raise RuntimeError("the model fell over on this scene")
        if self.mode == "returns_failure":
            return create_failed_result(
                audio=audio, sample_rate=sample_rate, method=self.name,
                error_message="not enough memory for this scene")
        return EnhancementResult(audio=audio, sample_rate=sample_rate, method=self.name)

    def cleanup(self):
        pass


@pytest.fixture
def scene(tmp_path):
    """One two-second 48kHz scene, as the extractor would leave it."""
    path = tmp_path / "scene_001.wav"
    samples = np.zeros(EXTRACTION_RATE * 2, dtype=np.float32)
    samples[::100] = 0.25  # something to resample
    sf.write(str(path), samples, EXTRACTION_RATE)
    return [(path, 0.0, 2.0, 2.0)]


def rate_of(scene_entry):
    return sf.info(str(scene_entry[0])).samplerate


class TestAFailedCleanUpStopsTheRun:
    def test_a_backend_that_reports_failure_stops_it(self, scene, tmp_path):
        with pytest.raises(SpeechEnhancerUnavailable) as caught:
            enhance_scenes(scene, StubEnhancer("returns_failure"), tmp_path)
        assert "not enough memory for this scene" in str(caught.value)

    def test_a_backend_that_raises_stops_it(self, scene, tmp_path):
        with pytest.raises(SpeechEnhancerUnavailable) as caught:
            enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        assert "the model fell over on this scene" in str(caught.value)

    def test_the_message_says_what_happened_and_what_to_do(self, scene, tmp_path):
        with pytest.raises(SpeechEnhancerUnavailable) as caught:
            enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        message = str(caught.value)
        # Which clean-up, where it failed, what it means, and the way out.
        assert "Stub enhancer" in message
        assert "scene 1 of 1" in message
        assert "cleaned up" in message
        assert "Choose a different clean-up, or none." in message

    def test_nothing_untouched_is_handed_back(self, scene, tmp_path):
        # The whole point: no path returns the original audio and carries on.
        try:
            result = enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        except SpeechEnhancerUnavailable:
            return
        pytest.fail(f"enhancement failed but the run continued with {result}")


class TestWhatDoesComeBackIsAtTheRightRate:
    def test_a_clean_up_that_works(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("works"), tmp_path)
        assert len(out) == 1
        assert rate_of(out[0]) == TARGET_SAMPLE_RATE

    def test_the_timings_are_carried_through_unchanged(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("works"), tmp_path)
        assert out[0][1:] == scene[0][1:]

    def test_the_untouched_track_of_the_split_matches_it(self, scene, tmp_path):
        # The fidelity "enhance for VAD only" arrangement: cleaned-up scenes on
        # one side, the originals resampled on the other, paired by position.
        for_vad = enhance_scenes(scene, StubEnhancer("works"), tmp_path)
        for_asr = resample_scenes(scene, tmp_path)

        assert len(for_vad) == len(for_asr)
        assert rate_of(for_vad[0]) == rate_of(for_asr[0]) == TARGET_SAMPLE_RATE

    def test_the_original_scene_file_is_left_alone(self, scene, tmp_path):
        enhance_scenes(scene, StubEnhancer("works"), tmp_path)
        assert sf.info(str(scene[0][0])).samplerate == EXTRACTION_RATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
