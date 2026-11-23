"""ChronosJAV and the decoupled pipeline keep scenes under 12 s with the auditok and silero scene detectors (1.10.3,
owner W751 C9; a known fault of the 1.10.2 notes).

Their 12 s minimum (safe chunking) is meant for the semantic detector, which joins a shorter scene to its neighbours.
auditok and silero keep only scenes between their minimum and their maximum, so a 12 s minimum dropped every shorter
scene, and its speech. Since 1.10.3 the minimum is given only to a merging detector (MERGING_SCENE_DETECTORS) or, in
ChronosJAV, when the user set one.

Run with: python -m pytest tests/test_short_scenes_kept.py -q
"""
import wave
from pathlib import Path

import numpy as np
import pytest

from whisperjav.modules.scene_detection_backends import MERGING_SCENE_DETECTORS, SceneDetectorFactory

REPO = Path(__file__).resolve().parents[1]
SR = 16000


def _write_wav(path: Path) -> Path:
    """30 s: 1 s silence, 15 s of loud noise (a long scene), 4 s silence, 5 s of noise (a short scene), 5 s silence.
    (With no scene kept at all, SafeSceneDetector falls back to the whole audio; the speech is lost only when a
    long scene is kept and the short ones are dropped.)"""
    rng = np.random.default_rng(0)
    audio = np.zeros(30 * SR, dtype=np.float32)
    audio[1 * SR:16 * SR] = rng.uniform(-0.4, 0.4, 15 * SR).astype(np.float32)
    audio[20 * SR:25 * SR] = rng.uniform(-0.4, 0.4, 5 * SR).astype(np.float32)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    return path


def _scenes(tmp_path, **kwargs):
    wav = _write_wav(tmp_path / "short.wav")
    out = tmp_path / ("scenes_" + "_".join(f"{k}{v}" for k, v in sorted(kwargs.items())))
    out.mkdir()
    detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(method="auditok", **kwargs)
    return detector.detect_scenes(wav, out, "short").to_legacy_tuples()


def test_only_semantic_merges():
    assert MERGING_SCENE_DETECTORS == frozenset({"semantic"})


def _covers(scenes, t):
    return any(start <= t <= end for _, start, end, _ in scenes)


def test_auditok_drops_a_short_scene_under_a_12s_minimum_and_keeps_it_with_its_own(tmp_path):
    with_12s = _scenes(tmp_path, max_duration=48, min_duration=12)        # the 1.10.2 fault
    assert _covers(with_12s, 8.0) and not _covers(with_12s, 22.5)
    own = _scenes(tmp_path, max_duration=48)                              # auditok's own minimum (1.10.3)
    assert _covers(own, 8.0) and _covers(own, 22.5), own


@pytest.mark.parametrize("path,minimum", [
    ("whisperjav/pipelines/qwen_pipeline.py", 'scene_detector_kwargs["min_duration"] = min_dur'),
    ("whisperjav/pipelines/decoupled_pipeline.py", 'scene_detector_kwargs["min_duration"] = self.scene_min_duration'),
])
def test_the_pipelines_give_the_minimum_only_to_a_merging_detector(path, minimum):
    src = (REPO / path).read_text(encoding="utf-8")
    assert src.count(minimum) == 1
    i = src.index(minimum)
    before = src[max(0, i - 400):i]
    assert "self.scene_method in MERGING_SCENE_DETECTORS" in before
    assert 'scene_detector_kwargs["max_duration"]' in src[max(0, i - 700):i]   # the maximum stays for every detector
