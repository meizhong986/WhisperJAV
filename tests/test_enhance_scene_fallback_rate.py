#!/usr/bin/env python3
"""
Every scene that comes back from the clean-up step is at 16kHz -- including the
ones whose clean-up failed.

Why this matters (found by review, 2026-09-17). Scenes are extracted at 48kHz
when an enhancer is configured. ``enhance_scenes`` resamples what it enhances
down to 16kHz, but used to hand back the untouched 48kHz file for any scene that
failed. In the "enhance for VAD only" split that list is paired, scene by scene,
with originals that ``resample_scenes`` has put at 16kHz, and the recogniser
refuses two tracks recorded at different rates. So one transient failure on one
scene -- a model hiccup, a moment of CUDA pressure -- failed the whole file, with
a message about sample rates rather than about the clean-up that actually failed.

Note on scope: a scene whose clean-up fails does NOT stop the run. The owner's
stop rule is about a component the user selected that cannot install or cannot
start; it is not a rule that every error anywhere stops the run. How the other
kinds of failure should be handled is being settled separately, and this file
will follow that decision rather than anticipate it.

Run with: pytest tests/test_enhance_scene_fallback_rate.py -v
"""

import numpy as np
import pytest
import soundfile as sf

from whisperjav.modules.speech_enhancement.base import (
    EnhancementResult,
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


class TestAFailedSceneStillComesBackAtTheRightRate:
    def test_a_backend_that_reports_failure(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("returns_failure"), tmp_path)
        assert len(out) == 1
        assert rate_of(out[0]) == TARGET_SAMPLE_RATE

    def test_a_backend_that_raises_on_one_scene(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        assert len(out) == 1
        assert rate_of(out[0]) == TARGET_SAMPLE_RATE

    def test_a_backend_that_works(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("works"), tmp_path)
        assert rate_of(out[0]) == TARGET_SAMPLE_RATE

    def test_the_timings_are_carried_through_unchanged(self, scene, tmp_path):
        out = enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        assert out[0][1:] == scene[0][1:]


class TestTheTwoTracksOfTheSplitAgree:
    def test_a_failed_scene_does_not_break_the_pair(self, scene, tmp_path):
        # This is the fidelity "enhance for VAD only" arrangement: the cleaned-up
        # scenes on one side, the originals resampled on the other.
        for_vad = enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        for_asr = resample_scenes(scene, tmp_path)

        assert len(for_vad) == len(for_asr)
        assert rate_of(for_vad[0]) == rate_of(for_asr[0]) == TARGET_SAMPLE_RATE

    def test_the_original_scene_file_is_left_alone(self, scene, tmp_path):
        enhance_scenes(scene, StubEnhancer("raises"), tmp_path)
        assert sf.info(str(scene[0][0])).samplerate == EXTRACTION_RATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
