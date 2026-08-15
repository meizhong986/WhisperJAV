"""Pathless (in-memory frame audio) regression tests for the qwen subtitle pipeline.

Verifies v1.9.0 Option A: when the generator declares ``accepts_array = True`` and
no aligner is present (anime-whisper runs ``vad_only``), the orchestrator passes
in-memory numpy slices to the generator and writes ZERO per-VAD-group temp WAVs.
When the generator does NOT accept arrays (qwen3 / cohere), the legacy path-based
behavior (one temp WAV per frame) is preserved unchanged.

Lightweight by design: no torch / ASR model load — the framer and generator are
stubbed so the test exercises only the orchestrator's slicing branch.
"""
from pathlib import Path
import glob

import numpy as np
import pytest

pytest.importorskip("soundfile")

from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
from whisperjav.modules.subtitle_pipeline.types import (
    FramingResult,
    HardeningConfig,
    TemporalFrame,
)
from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
    AnimeWhisperGenerator,
)


class _StubFramer:
    """Returns two sliced frames (0.0-0.5s, 0.5-1.0s) — not a full-scene frame."""

    def frame(self, audio, sr, **kwargs):
        return FramingResult(
            frames=[TemporalFrame(0.0, 0.5), TemporalFrame(0.5, 1.0)],
            metadata={},
        )

    def cleanup(self):
        pass


class _StubGen:
    def __init__(self, accepts_array: bool):
        if accepts_array:
            self.accepts_array = True

    def load(self):
        pass

    def unload(self):
        pass

    def cleanup(self):
        pass


class _StubAligner:
    """Stub TextAligner; pathless gate only reads its ``accepts_array`` flag."""

    def __init__(self, accepts_array: bool):
        if accepts_array:
            self.accepts_array = True

    def load(self):
        pass

    def unload(self):
        pass

    def cleanup(self):
        pass

    def align_batch(self, audio_paths, texts, **kwargs):
        return []


def _make_scene_wav(tmp_path: Path) -> Path:
    import soundfile as sf

    wav = tmp_path / "scene_0000.wav"
    sf.write(str(wav), np.linspace(-0.2, 0.2, 16000).astype(np.float32), 16000)
    return wav


def _orchestrator(tmp_path: Path, accepts_array: bool, aligner=None) -> DecoupledSubtitlePipeline:
    return DecoupledSubtitlePipeline(
        framer=_StubFramer(),
        generator=_StubGen(accepts_array),
        cleaner=object(),
        aligner=aligner,
        hardening_config=HardeningConfig(),
        artifacts_dir=tmp_path,
    )


def test_pathless_passes_arrays_and_writes_no_wavs(tmp_path):
    wav = _make_scene_wav(tmp_path)
    orch = _orchestrator(tmp_path, accepts_array=True)

    _frames, frame_audio, _regions = orch._step1_frame_and_slice([wav], [1.0], None, None)

    entries = frame_audio[0]
    assert all(isinstance(e, np.ndarray) for e in entries)
    assert [len(e) for e in entries] == [8000, 8000]  # 0.5s @ 16kHz
    assert glob.glob(str(tmp_path / "dsp_*.wav")) == []  # zero temp WAVs


def test_fallback_writes_one_wav_per_frame(tmp_path):
    wav = _make_scene_wav(tmp_path)
    orch = _orchestrator(tmp_path, accepts_array=False)

    _frames, frame_audio, _regions = orch._step1_frame_and_slice([wav], [1.0], None, None)

    entries = frame_audio[0]
    assert all(isinstance(e, Path) for e in entries)
    assert len(glob.glob(str(tmp_path / "dsp_*.wav"))) == 2  # legacy behavior intact


def test_anime_generator_declares_array_capability_and_passthrough():
    assert AnimeWhisperGenerator.accepts_array is True

    arr = np.array([0.1, -0.1, 0.2], dtype=np.float32)
    out = AnimeWhisperGenerator._load_audio(arr)
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    assert np.allclose(out, arr)

    # stereo input is downmixed to mono
    stereo = np.stack([arr, arr], axis=1)
    assert AnimeWhisperGenerator._load_audio(stereo).ndim == 1


def test_pathless_when_aligner_is_array_capable(tmp_path):
    """qwen3 path: generator AND aligner both array-capable -> pathless, no WAVs."""
    wav = _make_scene_wav(tmp_path)
    orch = _orchestrator(tmp_path, accepts_array=True, aligner=_StubAligner(accepts_array=True))

    _frames, frame_audio, _regions = orch._step1_frame_and_slice([wav], [1.0], None, None)

    assert all(isinstance(e, np.ndarray) for e in frame_audio[0])
    assert glob.glob(str(tmp_path / "dsp_*.wav")) == []


def test_fallback_when_aligner_not_array_capable(tmp_path):
    """A non-array-capable aligner forces the legacy WAV path (fail-safe gate)."""
    wav = _make_scene_wav(tmp_path)
    orch = _orchestrator(tmp_path, accepts_array=True, aligner=_StubAligner(accepts_array=False))

    _frames, frame_audio, _regions = orch._step1_frame_and_slice([wav], [1.0], None, None)

    assert all(isinstance(e, Path) for e in frame_audio[0])
    assert len(glob.glob(str(tmp_path / "dsp_*.wav"))) == 2
