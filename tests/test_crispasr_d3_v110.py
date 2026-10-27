"""D3 for the GGUF passes (design 22.9.1; owner W212 D3, W214 2, W217 19, W232 8, W232 14):
the scene detector per pass, the speech enhancer (one track or Enhance for VAD only), the
Customize values, and the model opened just in time when an enhancer runs.

The scene detector, the enhancer and CrispASR are stand-ins; nothing is downloaded or loaded.
"""

import sys
import types
from pathlib import Path

import pytest

from whisperjav.pipelines.cohere_crispasr_pipeline import CohereCrispASRPipeline
from whisperjav.pipelines.parakeet_crispasr_pipeline import ParakeetCrispASRPipeline
from whisperjav.pipelines import crispasr_pipeline as cp


def make(cls, tmp_path, **kwargs):
    return cls(output_dir=str(tmp_path / "out"), temp_dir=str(tmp_path / "temp"), **kwargs)


@pytest.mark.parametrize("detector,expected", [
    (None, {"method": "semantic", "min_duration": 12, "max_duration": 48}),   # owner W214 2: the default
    ("semantic", {"method": "semantic", "min_duration": 12, "max_duration": 48}),
    ("auditok", {"method": "auditok", "max_duration": 48}),                    # its own minimum (22.9.1)
    ("silero", {"method": "silero", "max_duration": 48}),
    ("none", {"method": "none"}),                                              # one scene per file
])
def test_the_scene_detector_settings_per_detector(tmp_path, detector, expected):
    pipe = make(ParakeetCrispASRPipeline, tmp_path, scene_detector=detector)
    assert pipe.scene_detector_kwargs() == expected


def test_customize_scene_values_win(tmp_path):
    pipe = make(CohereCrispASRPipeline, tmp_path, scene_detector="auditok",
                scene_params={"max_duration_s": 30, "pass1_energy_threshold": 40})
    kwargs = pipe.scene_detector_kwargs()
    assert kwargs["max_duration_s"] == 30 and kwargs["pass1_energy_threshold"] == 40


def test_customize_segmenter_values_win(tmp_path):
    base = make(ParakeetCrispASRPipeline, tmp_path)
    pipe = make(ParakeetCrispASRPipeline, tmp_path,
                segmenter_overrides={"threshold": 0.6, "max_group_duration_s": 4.0})
    assert pipe.segmenter.config["threshold"] == 0.6
    assert pipe.segmenter.max_group_duration_s == 4.0
    assert pipe.segmenter.chunk_threshold_s == base.segmenter.chunk_threshold_s


class _FakeDetector:
    def __init__(self, scenes_dir_files):
        self.files = scenes_dir_files

    def detect_scenes(self, wav, scenes_dir, basename):
        return types.SimpleNamespace(to_legacy_tuples=lambda: [(str(f), 0.0, 5.0, 5.0) for f in self.files],
                                     method="auditok")

    def cleanup(self):
        pass


def test_analytics_only_with_semantic_scenes(tmp_path, monkeypatch, caplog):
    from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
    seen = []
    monkeypatch.setattr(SceneDetectorFactory, "safe_create_from_legacy_kwargs",
                        staticmethod(lambda **kw: seen.append(kw) or _FakeDetector([tmp_path / "s0.wav"])))
    import whisperjav.modules.analytics as analytics
    monkeypatch.setattr(analytics, "report", lambda *a, **k: pytest.fail("no analytics with auditok scenes"))
    pipe = make(ParakeetCrispASRPipeline, tmp_path, scene_detector="auditok")
    with caplog.at_level("INFO", logger="whisperjav"):
        scenes = pipe._detect_scenes(tmp_path / "x.wav", "x", {"stages": {}})
    assert len(scenes) == 1 and seen == [{"method": "auditok", "max_duration": 48}]
    assert ("Parakeet-CrispASR: audio analytics need the semantic scene detector; this pass uses auditok, "
            "so they are not made.") in caplog.text


class _FakeEnhancer:
    def __init__(self, log):
        self.log = log

    def cleanup(self):
        self.log.append("enhancer released")


def _stub_enhancement(monkeypatch, log):
    import whisperjav.modules.speech_enhancement as se
    monkeypatch.setattr(se, "create_enhancer_direct",
                        lambda backend, model=None: log.append(f"enhancer {backend}") or _FakeEnhancer(log))
    monkeypatch.setattr(se, "enhance_scenes",
                        lambda tuples, enhancer, temp_dir, **k: [(Path(str(t[0]) + ".enh"),) + tuple(t[1:]) for t in tuples])
    monkeypatch.setattr(se, "resample_scenes",
                        lambda tuples, temp_dir: [(Path(str(t[0]) + ".16k"),) + tuple(t[1:]) for t in tuples])
    import whisperjav.utils.gpu_utils as gu
    monkeypatch.setattr(gu, "safe_cuda_cleanup", lambda: log.append("cuda cleanup"))


def _scenes(tmp_path):
    return [cp._Scene(0, tmp_path / "s0.wav", 0.0, 5.0), cp._Scene(1, tmp_path / "s1.wav", 5.0, 9.0)]


def test_no_enhancer_keeps_the_scenes(tmp_path):
    pipe = make(ParakeetCrispASRPipeline, tmp_path)
    scenes = _scenes(tmp_path)
    assert pipe._enhance(scenes, {"stages": {}}) == (scenes, scenes)


def test_one_track_the_enhanced_scenes_for_both(tmp_path, monkeypatch):
    log = []
    _stub_enhancement(monkeypatch, log)
    pipe = make(ParakeetCrispASRPipeline, tmp_path, speech_enhancer="zipenhancer")
    md = {"stages": {}}
    asr, vad = pipe._enhance(_scenes(tmp_path), md)
    assert [s.path.name for s in asr] == ["s0.wav.enh", "s1.wav.enh"] and asr == vad
    assert log == ["enhancer zipenhancer", "enhancer released", "cuda cleanup"]   # released before the ASR
    assert md["stages"]["enhancement"]["dual_track"] is False


def test_enhance_for_vad_the_enhanced_to_the_segmenter_and_the_original_to_crispasr(tmp_path, monkeypatch):
    log = []
    _stub_enhancement(monkeypatch, log)
    pipe = make(CohereCrispASRPipeline, tmp_path, speech_enhancer="bs-roformer", enhance_for_vad=True)
    md = {"stages": {}}
    asr, vad = pipe._enhance(_scenes(tmp_path), md)
    assert [s.path.name for s in vad] == ["s0.wav.enh", "s1.wav.enh"]
    assert [s.path.name for s in asr] == ["s0.wav.16k", "s1.wav.16k"]
    assert [(s.start, s.end) for s in asr] == [(0.0, 5.0), (5.0, 9.0)]
    assert md["stages"]["enhancement"]["dual_track"] is True


def test_with_an_enhancer_the_extraction_is_at_48_khz(tmp_path, monkeypatch):
    import whisperjav.modules.audio_extraction as ae
    rates = []

    class FakeExtractor:
        def __init__(self, sample_rate):
            rates.append(sample_rate)
            self.last_integrity = None

        def extract(self, input_file, out):
            return out, 10.0
    monkeypatch.setattr(ae, "AudioExtractor", FakeExtractor)
    import whisperjav.modules.audio_integrity as ai
    monkeypatch.setattr(ai, "apply_to_run", lambda *a, **k: None)
    make(ParakeetCrispASRPipeline, tmp_path, speech_enhancer="zipenhancer")._extract_audio(tmp_path / "a.mp4", "a")
    make(ParakeetCrispASRPipeline, tmp_path)._extract_audio(tmp_path / "a.mp4", "a")
    assert rates == [48000, 16000]


def test_with_an_enhancer_the_model_is_closed_after_each_file(tmp_path):
    src = Path(cp.__file__).read_text(encoding="utf-8")
    step = src[src.index('self.progress.set_current_step("Transcribing with CrispASR"'):
               src.index('self.progress.set_current_step("Building subtitles"')]
    assert 'self.enhancer_backend not in ("", "none")' in step and "self.asr.close()" in step


def test_the_pass_worker_and_the_gui_hand_these_settings_on():
    repo = Path(cp.__file__).resolve().parents[2]
    worker = (repo / "whisperjav" / "ensemble" / "pass_worker.py").read_text(encoding="utf-8")
    assert '"scene_detector": pass_config.get("scene_detector") or None,' in worker
    assert 'crispasr_kwargs["segmenter_overrides"]' in worker and 'crispasr_kwargs["scene_params"]' in worker
    app = (repo / "whisperjav" / "webview_gui" / "assets" / "app.js").read_text(encoding="utf-8")
    block = app[app.index("if (passState.isCrispasr) {\n            sensitivitySelect.disabled"):]
    block = block[:block.index("return;")]
    assert "disabled = true" not in block      # nothing greyed but the "none" segmenter (22.9.1)
    assert "async openCrispasrCustomize(passKey)" in app
