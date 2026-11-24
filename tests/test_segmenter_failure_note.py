"""A scene whose speech segmenter failed is counted, so ChronosJAV and decoupled can name it in the run summary
(1.10.3, owner W751 B1 3; the GGUF pipelines' words, W764 2 texts sheet item 2). No model is loaded: the framer is a
stand-in that reports a failure for one scene, as VadGroupedFramer does when the segmenter caught its own error."""
from pathlib import Path

import numpy as np
import pytest

sf = pytest.importorskip("soundfile")

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import FramingResult, HardeningConfig, TemporalFrame, TimestampMode


class _FramerFailingOnScene:
    def __init__(self, failing: int):
        self.failing = failing
        self.calls = 0

    def frame(self, audio, sample_rate, **kwargs):
        index, self.calls = self.calls, self.calls + 1
        if index == self.failing:
            return FramingResult(frames=[], metadata={"segmenter_error": "model missing"})
        return FramingResult(frames=[TemporalFrame(0.0, 0.9)], metadata={})

    def reframe(self, audio, sample_rate, max_group_duration_s=None):
        return FramingResult(frames=[], metadata={"segmenter_error": "model missing"})


class _Generator:
    accepts_array = False


def _scenes(tmp_path, n):
    paths = []
    for i in range(n):
        wav = tmp_path / f"scene_{i:04d}.wav"
        sf.write(str(wav), (0.1 * np.sin(np.linspace(0, 2000, 16000))).astype(np.float32), 16000)
        paths.append(wav)
    return paths


def _orchestrator(framer, tmp_path):
    orch = DecoupledSubtitlePipeline(
        framer=framer, generator=_Generator(), cleaner=object(), aligner=None,
        hardening_config=HardeningConfig(timestamp_mode=TimestampMode.VAD_ONLY),
    )
    orch.segmenter_failed_scenes = 0
    orch.artifacts_dir = tmp_path / "frames"   # the frame WAVs go here, not to the user's temporary folder
    return orch


def test_a_scene_the_segmenter_failed_on_is_counted(tmp_path):
    orch = _orchestrator(_FramerFailingOnScene(failing=1), tmp_path)
    frames, *_ = orch._step1_frame_and_slice(_scenes(tmp_path, 3), [1.0, 1.0, 1.0])
    assert [len(f) for f in frames] == [1, 0, 1]
    assert orch.segmenter_failed_scenes == 1


def test_a_step_down_retry_does_not_count_again(tmp_path):
    orch = _orchestrator(_FramerFailingOnScene(failing=-1), tmp_path)
    orch._step1_frame_and_slice(_scenes(tmp_path, 2), [1.0, 1.0], framer_override_max_group=2.0)
    assert orch.segmenter_failed_scenes == 0


def test_the_count_starts_again_for_each_file():
    import inspect
    assert "self.segmenter_failed_scenes = 0" in inspect.getsource(DecoupledSubtitlePipeline.process_scenes)


def test_the_pipelines_use_the_gguf_words():
    """ChronosJAV, decoupled and fidelity write the note with the GGUF pipelines' words (read from the source, so that
    no ASR module is imported here)."""
    root = Path(__file__).resolve().parents[1] / "whisperjav" / "pipelines"
    for name, label in (("qwen_pipeline.py", "ChronosJAV"), ("decoupled_pipeline.py", "Decoupled"),
                        ("fidelity_pipeline.py", "Fidelity")):
        source = (root / name).read_text(encoding="utf-8")
        assert f'{label}: the speech segmenter failed on {{' in source, name
        assert "their speech was not " in source and 'transcribed."' in source, name
