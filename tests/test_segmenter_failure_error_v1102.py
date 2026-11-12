"""A speech segmenter that catches its own failure says so (1.10.2, still-open item 4 of the 1.10.1 notes, GGUF part).

Silero v6.2, whisper-vad and NeMo returned an empty result without ``error`` when they failed, so the GGUF pipelines
took the scene as silence: no failed scene in the file's summary. Now each failure carries its reason in
``SegmentationResult.error``, which the VAD-grouped framer hands on as ``segmenter_error`` and the GGUF pipelines count
as a failed scene, as for TEN, FireRedVAD and WhisperSeg. No model is loaded: the failing step is replaced.
"""
from pathlib import Path

import numpy as np
import pytest


def _silence():
    return np.zeros(16000, dtype=np.float32), 16000


def test_silero_v6_failure_carries_its_reason(monkeypatch):
    silero_vad = pytest.importorskip("silero_vad")
    from whisperjav.modules.speech_segmentation.backends.silero_v6 import SileroV6SpeechSegmenter

    seg = SileroV6SpeechSegmenter()
    monkeypatch.setattr(seg, "_ensure_model", lambda: None)
    monkeypatch.setattr(seg, "_load_audio", lambda audio, sr: _silence())

    def broken(*args, **kwargs):
        raise RuntimeError("stand-in failure")

    monkeypatch.setattr(silero_vad, "get_speech_timestamps", broken)
    result = seg.segment(np.zeros(16000, dtype=np.float32), 16000)
    assert result.segments == [] and result.error == "stand-in failure"


def test_whisper_vad_unreadable_audio_carries_a_reason(monkeypatch):
    from whisperjav.modules.speech_segmentation.backends.whisper_vad import WhisperVadSpeechSegmenter

    seg = WhisperVadSpeechSegmenter()
    monkeypatch.setattr(seg, "_resolve_audio_path", lambda audio, sr: None)
    result = seg.segment(np.zeros(16000, dtype=np.float32), 16000)
    assert result.segments == [] and result.error == "the scene's audio could not be read"


def test_whisper_vad_failure_carries_its_reason(monkeypatch):
    from whisperjav.modules.speech_segmentation.backends.whisper_vad import WhisperVadSpeechSegmenter

    seg = WhisperVadSpeechSegmenter()
    monkeypatch.setattr(seg, "_resolve_audio_path", lambda audio, sr: Path("scene.wav"))

    def broken(path):
        raise OSError("stand-in failure")

    monkeypatch.setattr(seg, "_get_audio_duration", broken)
    result = seg.segment(np.zeros(16000, dtype=np.float32), 16000)
    assert result.segments == [] and result.error == "stand-in failure"


@pytest.mark.parametrize("variant", ["nemo-lite", "nemo-diarization"])
@pytest.mark.parametrize("step", ["audio", "model"])
def test_nemo_failure_carries_its_reason(monkeypatch, variant, step):
    from whisperjav.modules.speech_segmentation.backends import nemo

    seg = nemo.NemoSpeechSegmenter(variant=variant)

    def broken(*args, **kwargs):
        raise nemo.NemoModelLoadError("stand-in failure")

    if step == "audio":
        monkeypatch.setattr(seg, "_load_audio", lambda audio, sr: (_ for _ in ()).throw(OSError("stand-in failure")))
    else:
        monkeypatch.setattr(seg, "_load_audio", lambda audio, sr: _silence())
        monkeypatch.setattr(seg, "_segment_with_frame_vad", broken)
        monkeypatch.setattr(seg, "_segment_with_diarizer", broken)
    result = seg.segment(np.zeros(16000, dtype=np.float32), 16000)
    assert result.segments == [] and result.error == "stand-in failure"


def test_the_framer_hands_the_error_on_as_segmenter_error(monkeypatch):
    from whisperjav.modules.speech_segmentation.base import SegmentationResult
    from whisperjav.modules.subtitle_pipeline.framers import vad_grouped

    failed = SegmentationResult(segments=[], groups=[], method="stand-in", audio_duration_sec=1.0, parameters={},
                                processing_time_sec=0.0, error="stand-in failure")

    class Segmenter:
        def segment(self, *args, **kwargs):
            return failed

        def cleanup(self):
            pass

    framer = vad_grouped.VadGroupedFramer(segmenter_backend="silero-v6.2")
    framer._segmenter = Segmenter()   # _ensure_segmenter() keeps a segmenter already set
    result = framer.frame(np.zeros(16000, dtype=np.float32), 16000)
    assert result.metadata.get("segmenter_error") == "stand-in failure"
