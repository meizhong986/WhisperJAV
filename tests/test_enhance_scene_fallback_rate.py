#!/usr/bin/env python3
"""
What happens when a clean-up fails part-way, and what the scenes look like after.

Two things are pinned here, both from 2026-09-17.

THE RULE (owner: "only all-fail is fatal, agreed"). A scene whose clean-up fails
does NOT stop the run: that scene goes through uncleaned, the user is told how
many did, and the transcription still happens. A clean-up that failed on EVERY
scene did in effect not run at all, so it is refused the same way as one that
could not start. The owner's stop rule is about a component the user selected
being unavailable -- not about every error anywhere stopping the run.

THE SAMPLE RATE (found by review). Scenes are extracted at 48kHz when an enhancer
is configured. ``enhance_scenes`` resamples what it enhances down to 16kHz, but
used to hand back the untouched 48kHz file for any scene that failed. In the
"enhance for VAD only" split that list is paired, scene by scene, with originals
that ``resample_scenes`` has put at 16kHz, and the recogniser refuses two tracks
recorded at different rates. So one failed scene broke the whole file with a
message about sample rates rather than about the clean-up that actually failed.

Run with: pytest tests/test_enhance_scene_fallback_rate.py -v
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
    """Stands in for a real backend; no model, no download.

    ``failing`` is the set of 1-based scene numbers to fail on; "all" fails on
    every scene. ``raises`` decides whether a failure is thrown or reported as a
    failed result -- real backends do both.
    """

    name = "stub"
    display_name = "Stub enhancer"

    def __init__(self, failing=(), raises=False):
        self.failing = failing
        self.raises = raises
        self.seen = 0

    def get_preferred_sample_rate(self):
        return EXTRACTION_RATE

    def enhance(self, audio, sample_rate):
        self.seen += 1
        if self.failing == "all" or self.seen in self.failing:
            if self.raises:
                raise RuntimeError(f"the model fell over on scene {self.seen}")
            return create_failed_result(
                audio=audio, sample_rate=sample_rate, method=self.name,
                error_message=f"not enough memory for scene {self.seen}")
        return EnhancementResult(audio=audio, sample_rate=sample_rate, method=self.name)

    def cleanup(self):
        pass


def make_scenes(tmp_path, count):
    """`count` one-second 48kHz scenes, as the extractor would leave them."""
    scenes = []
    for n in range(count):
        path = tmp_path / f"scene_{n:03d}.wav"
        samples = np.zeros(EXTRACTION_RATE, dtype=np.float32)
        samples[::100] = 0.25  # something to resample
        sf.write(str(path), samples, EXTRACTION_RATE)
        scenes.append((path, float(n), float(n + 1), 1.0))
    return scenes


@pytest.fixture
def scenes(tmp_path):
    return make_scenes(tmp_path, 3)


def rate_of(scene_entry):
    return sf.info(str(scene_entry[0])).samplerate


class TestOnlyAllFailIsFatal:
    def test_one_scene_failing_does_not_stop_the_run(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(failing={2}), tmp_path)
        assert len(out) == 3

    def test_one_scene_throwing_does_not_stop_the_run(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(failing={2}, raises=True), tmp_path)
        assert len(out) == 3

    def test_every_scene_failing_stops_the_run(self, scenes, tmp_path):
        with pytest.raises(SpeechEnhancerUnavailable) as caught:
            enhance_scenes(scenes, StubEnhancer(failing="all"), tmp_path)
        message = str(caught.value)
        assert "every one of the 3 pieces" in message
        assert "not enough memory for scene 1" in message
        assert "Choose a different clean-up, or none." in message

    def test_every_scene_throwing_also_stops_the_run(self, scenes, tmp_path):
        with pytest.raises(SpeechEnhancerUnavailable):
            enhance_scenes(scenes, StubEnhancer(failing="all", raises=True), tmp_path)

    def test_a_single_scene_file_that_fails_is_all_fail(self, tmp_path):
        # One scene is the whole audio, so failing it is failing everything.
        one = make_scenes(tmp_path, 1)
        with pytest.raises(SpeechEnhancerUnavailable):
            enhance_scenes(one, StubEnhancer(failing="all"), tmp_path)


class TestTheUserIsToldWhatWasSkipped:
    """Degrading is only acceptable because it does not stay hidden."""

    def test_the_count_is_reported(self, scenes, tmp_path):
        notes = []
        enhance_scenes(scenes, StubEnhancer(failing={1, 3}), tmp_path,
                       degradations=notes)
        assert len(notes) == 1
        assert "2 of 3 scenes" in notes[0]
        assert "Stub enhancer" in notes[0]

    def test_nothing_is_reported_when_every_scene_is_cleaned_up(self, scenes, tmp_path):
        notes = []
        enhance_scenes(scenes, StubEnhancer(), tmp_path, degradations=notes)
        assert notes == []

    def test_the_list_is_optional(self, scenes, tmp_path):
        # Every existing caller passes nothing; that must keep working.
        assert len(enhance_scenes(scenes, StubEnhancer(failing={1}), tmp_path)) == 3


class TestAFailedSceneStillComesBackAtTheRightRate:
    def test_a_backend_that_reports_failure(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(failing={2}), tmp_path)
        assert all(rate_of(entry) == TARGET_SAMPLE_RATE for entry in out)

    def test_a_backend_that_raises_on_one_scene(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(failing={2}, raises=True), tmp_path)
        assert all(rate_of(entry) == TARGET_SAMPLE_RATE for entry in out)

    def test_a_backend_that_works(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(), tmp_path)
        assert all(rate_of(entry) == TARGET_SAMPLE_RATE for entry in out)

    def test_the_timings_are_carried_through_unchanged(self, scenes, tmp_path):
        out = enhance_scenes(scenes, StubEnhancer(failing={2}, raises=True), tmp_path)
        assert [entry[1:] for entry in out] == [entry[1:] for entry in scenes]


class TestTheTwoTracksOfTheSplitAgree:
    def test_a_failed_scene_does_not_break_the_pair(self, scenes, tmp_path):
        # The fidelity "enhance for VAD only" arrangement: the cleaned-up scenes
        # on one side, the originals resampled on the other, paired by position.
        for_vad = enhance_scenes(scenes, StubEnhancer(failing={2}, raises=True), tmp_path)
        for_asr = resample_scenes(scenes, tmp_path)

        assert len(for_vad) == len(for_asr)
        for vad_entry, asr_entry in zip(for_vad, for_asr):
            assert rate_of(vad_entry) == rate_of(asr_entry) == TARGET_SAMPLE_RATE

    def test_the_original_scene_files_are_left_alone(self, scenes, tmp_path):
        enhance_scenes(scenes, StubEnhancer(failing={2}, raises=True), tmp_path)
        for entry in scenes:
            assert sf.info(str(entry[0])).samplerate == EXTRACTION_RATE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
