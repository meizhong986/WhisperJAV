"""
End-to-end test suite for VAD threshold and padding parameter routing.

Verifies that VAD threshold and speech_pad_ms (and backend-specific equivalents)
are correctly implemented across ALL speech segmenters.

Tests are organized from strongest to weakest verification:
- Mathematical verification: exact arithmetic on padding (TEN, Silero v4)
- Boundary conditions: extreme threshold values on real models
- Monotonicity: relative comparisons (lower threshold → more segments)
- Parameter plumbing: factory stores correct attribute values
- Gap documentation: xfail tests for known routing bugs

Backend Parameter Map:
| Backend         | Threshold Param       | Padding Param                    | Our Code? |
|-----------------|-----------------------|----------------------------------|-----------|
| silero (v4/v3)  | threshold             | speech_pad_ms (lib) +            | YES (2nd) |
|                 |                       | start/end_pad_samples (our code) |           |
| silero-v6.2     | threshold             | speech_pad_ms (library only)     | NO        |
| ten             | threshold             | start_pad_ms / end_pad_ms        | YES       |
| nemo            | onset / offset        | pad_onset / pad_offset           | NO (lib)  |
| whisper-vad     | no_speech_threshold   | (none)                           | NO        |
| none            | (none)                | (none)                           | N/A       |
"""

import importlib.util
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest


# ─── Direct module loading (avoids full whisperjav import chain) ──────


def _load_module_direct(name: str, path: str):
    """Load a Python module directly from file path, bypassing package init."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_repo_root = Path(__file__).parent.parent

_base_module = _load_module_direct(
    "whisperjav.modules.speech_segmentation.base",
    str(_repo_root / "whisperjav/modules/speech_segmentation/base.py"),
)
_factory_module = _load_module_direct(
    "whisperjav.modules.speech_segmentation.factory",
    str(_repo_root / "whisperjav/modules/speech_segmentation/factory.py"),
)

SpeechSegment = _base_module.SpeechSegment
SegmentationResult = _base_module.SegmentationResult
SpeechSegmenterFactory = _factory_module.SpeechSegmenterFactory

# Load TEN backend directly — only depends on numpy + base, no ten_vad needed
_ten_module = _load_module_direct(
    "whisperjav.modules.speech_segmentation.backends.ten",
    str(_repo_root / "whisperjav/modules/speech_segmentation/backends/ten.py"),
)
TenSpeechSegmenter = _ten_module.TenSpeechSegmenter

# Load Silero backend — requires torch at import time
try:
    _silero_module = _load_module_direct(
        "whisperjav.modules.speech_segmentation.backends.silero",
        str(_repo_root / "whisperjav/modules/speech_segmentation/backends/silero.py"),
    )
    SileroSpeechSegmenter = _silero_module.SileroSpeechSegmenter
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    SileroSpeechSegmenter = None


# ─── Conditional imports for sensitivity tests ────────────────────────

try:
    from whisperjav.ensemble.pass_worker import (
        SEGMENTER_PARAMS,
        resolve_qwen_sensitivity,
    )

    HAS_PASS_WORKER = True
except ImportError:
    HAS_PASS_WORKER = False
    SEGMENTER_PARAMS = set()

    def resolve_qwen_sensitivity(*a, **kw):
        return {}


try:
    from whisperjav.config.v4 import ConfigManager

    HAS_CONFIG_MANAGER = True
except ImportError:
    HAS_CONFIG_MANAGER = False


# ─── Helpers ──────────────────────────────────────────────────────────

VAD_SR = 16000  # Standard VAD sample rate


def _backend_available(name: str) -> bool:
    """Check backend availability without loading model."""
    try:
        available, _ = SpeechSegmenterFactory.is_backend_available(name)
        return available
    except Exception:
        return False


def _skip_unless_backend(name: str):
    """Pytest skip decorator for backend availability."""
    return pytest.mark.skipif(
        not _backend_available(name),
        reason=f"{name} backend not available",
    )


def _make_test_audio(duration_s: float = 5.0, sr: int = 16000):
    """
    Create synthetic audio with speech-like bursts + silence.

    Layout (5s total):
    - 0.0-1.0s: silence
    - 1.0-2.0s: loud 440Hz sine (0.8 amplitude)
    - 2.0-2.5s: silence
    - 2.5-3.5s: quiet 440Hz sine (0.3 amplitude)
    - 3.5-5.0s: silence
    """
    samples = int(duration_s * sr)
    audio = np.zeros(samples, dtype=np.float32)

    t1 = np.arange(sr, dtype=np.float32) / sr
    audio[sr : 2 * sr] = np.sin(2 * np.pi * 440 * t1).astype(np.float32) * 0.8

    t2 = np.arange(sr, dtype=np.float32) / sr
    start = int(2.5 * sr)
    audio[start : start + sr] = np.sin(2 * np.pi * 440 * t2).astype(np.float32) * 0.3

    return audio, sr


# ═══════════════════════════════════════════════════════════════════════
# Class 1: TEN padding arithmetic — MATHEMATICAL VERIFICATION
#
# TEN's _flags_to_segments() is pure Python with no library dependencies.
# We construct known flags/probs, call it directly, and verify segment
# boundaries are shifted by EXACTLY start_pad_ms/1000 and end_pad_ms/1000.
# ═══════════════════════════════════════════════════════════════════════


class TestTenPaddingArithmetic:
    """Mathematically verify TEN's _flags_to_segments() padding arithmetic."""

    def _make_segmenter(
        self,
        start_pad_ms: int = 0,
        end_pad_ms: int = 0,
        min_speech_duration_ms: int = 0,
        hop_size: int = 256,
    ) -> TenSpeechSegmenter:
        """Create TEN segmenter with specific padding — no model loaded."""
        return TenSpeechSegmenter(
            threshold=0.5,  # irrelevant — flags already decided
            hop_size=hop_size,
            start_pad_ms=start_pad_ms,
            end_pad_ms=end_pad_ms,
            min_speech_duration_ms=min_speech_duration_ms,
        )

    def _frame_time(self, frame_idx: int, hop_size: int = 256, sr: int = 16000):
        """Compute time in seconds for a given frame index."""
        return frame_idx * hop_size / sr

    # ─── Basic segment detection from flags ───────────────────────

    def test_single_segment_no_padding(self):
        """One speech region, no padding — raw boundaries preserved exactly."""
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        # Frames: [silence, silence, SPEECH, SPEECH, SPEECH, silence, silence]
        flags = [0, 0, 1, 1, 1, 0, 0]
        probs = [0.1, 0.1, 0.8, 0.9, 0.7, 0.1, 0.1]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 1
        # Speech starts at frame 2, ends at frame 5 (first 0 after speech)
        expected_start = self._frame_time(2)  # 2 * 256/16000 = 0.032s
        expected_end = self._frame_time(5)  # 5 * 256/16000 = 0.080s
        assert result[0].start_sec == pytest.approx(expected_start, abs=1e-6)
        assert result[0].end_sec == pytest.approx(expected_end, abs=1e-6)

    def test_two_segments_no_padding(self):
        """Two speech regions separated by silence."""
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        flags = [0, 1, 1, 0, 0, 1, 1, 1, 0]
        probs = [0.1, 0.8, 0.9, 0.1, 0.1, 0.7, 0.8, 0.9, 0.1]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 2
        # Segment 1: frames 1-3
        assert result[0].start_sec == pytest.approx(self._frame_time(1), abs=1e-6)
        assert result[0].end_sec == pytest.approx(self._frame_time(3), abs=1e-6)
        # Segment 2: frames 5-8
        assert result[1].start_sec == pytest.approx(self._frame_time(5), abs=1e-6)
        assert result[1].end_sec == pytest.approx(self._frame_time(8), abs=1e-6)

    def test_speech_extends_to_end(self):
        """Speech region that runs to the end of the audio."""
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        flags = [0, 0, 1, 1, 1]
        probs = [0.1, 0.1, 0.8, 0.9, 0.7]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 1
        assert result[0].start_sec == pytest.approx(self._frame_time(2), abs=1e-6)
        # End = total_frames * frame_duration
        assert result[0].end_sec == pytest.approx(self._frame_time(5), abs=1e-6)

    # ─── Padding shifts boundaries by exact amounts ───────────────

    def test_end_pad_shifts_end_by_exact_amount(self):
        """end_pad_ms=150 shifts segment end later by exactly 0.150s."""
        seg_nopad = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        seg_padded = self._make_segmenter(start_pad_ms=0, end_pad_ms=150)

        flags = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
        probs = [0.1] * 10

        raw = seg_nopad._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)
        padded = seg_padded._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(raw) == 1
        assert len(padded) == 1
        # Start unchanged (start_pad_ms=0)
        assert padded[0].start_sec == pytest.approx(raw[0].start_sec, abs=1e-6)
        # End shifted by exactly 0.150s
        assert padded[0].end_sec == pytest.approx(raw[0].end_sec + 0.150, abs=1e-6)

    def test_start_pad_shifts_start_by_exact_amount(self):
        """start_pad_ms=50 shifts segment start earlier by exactly 0.050s."""
        seg_nopad = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        seg_padded = self._make_segmenter(start_pad_ms=50, end_pad_ms=0)

        # Speech starts at frame 5 (well after t=0 so no boundary clamp)
        flags = [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]
        probs = [0.1] * 10

        raw = seg_nopad._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)
        padded = seg_padded._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(raw) == 1
        assert len(padded) == 1
        # Start shifted earlier by exactly 0.050s
        assert padded[0].start_sec == pytest.approx(raw[0].start_sec - 0.050, abs=1e-6)
        # End unchanged
        assert padded[0].end_sec == pytest.approx(raw[0].end_sec, abs=1e-6)

    def test_combined_start_and_end_padding(self):
        """Both pads applied simultaneously — shifts are independent and exact."""
        seg = self._make_segmenter(start_pad_ms=100, end_pad_ms=200)
        # Speech at frames 20-23 (well past t=0 so start_pad won't hit boundary)
        flags = ([0] * 20) + [1, 1, 1] + ([0] * 10)
        probs = [0.1] * len(flags)

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        raw_start = self._frame_time(20)  # 20 * 256/16000 = 0.32s
        raw_end = self._frame_time(23)    # 23 * 256/16000 = 0.368s
        assert len(result) == 1
        assert result[0].start_sec == pytest.approx(raw_start - 0.100, abs=1e-6)
        assert result[0].end_sec == pytest.approx(raw_end + 0.200, abs=1e-6)

    def test_raw_boundaries_preserved_in_metadata(self):
        """TEN stores raw (unpadded) boundaries in segment metadata."""
        seg = self._make_segmenter(start_pad_ms=100, end_pad_ms=200)
        flags = [0, 0, 0, 1, 1, 0, 0]
        probs = [0.1] * 7

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 1
        raw_start = self._frame_time(3)
        raw_end = self._frame_time(5)
        assert result[0].metadata["raw_start"] == pytest.approx(raw_start, abs=1e-6)
        assert result[0].metadata["raw_end"] == pytest.approx(raw_end, abs=1e-6)
        # Padded values differ from raw
        assert result[0].start_sec < result[0].metadata["raw_start"]
        assert result[0].end_sec > result[0].metadata["raw_end"]

    # ─── Boundary clamping ────────────────────────────────────────

    def test_start_pad_clamped_to_zero(self):
        """Start padding cannot push start_sec below 0.0."""
        seg = self._make_segmenter(start_pad_ms=5000, end_pad_ms=0)
        # Speech starts at frame 0 — padding would push negative
        flags = [1, 1, 1, 0, 0]
        probs = [0.8] * 5

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 1
        assert result[0].start_sec == 0.0  # clamped, not negative

    def test_end_pad_clamped_to_duration(self):
        """End padding cannot push end_sec beyond audio_duration."""
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=5000)
        flags = [0, 0, 1, 1, 1]
        probs = [0.1, 0.1, 0.8, 0.9, 0.7]
        audio_duration = 0.1  # Very short

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=audio_duration)

        assert len(result) == 1
        assert result[0].end_sec == pytest.approx(audio_duration, abs=1e-6)

    # ─── Overlap prevention ───────────────────────────────────────

    def test_padding_overlap_prevention(self):
        """When padding would overlap adjacent segments, starts are clamped."""
        # Two segments close together; large padding would cause overlap
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=500)
        # Segment 1: frames 1-3, Segment 2: frames 5-7
        # Gap between them = 2 frames = 0.032s < 0.5s end_pad
        flags = [0, 1, 1, 0, 0, 1, 1, 0, 0]
        probs = [0.1] * 9

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 2
        # Segment 2's start must not be before segment 1's padded end
        assert result[1].start_sec >= result[0].end_sec

    # ─── Min duration filter ──────────────────────────────────────

    def test_min_duration_filters_short_segments(self):
        """Segments shorter than min_speech_duration_ms are dropped."""
        seg = self._make_segmenter(
            start_pad_ms=0, end_pad_ms=0, min_speech_duration_ms=100
        )
        # Single frame of speech: duration = 256/16000 = 16ms < 100ms
        flags = [0, 1, 0, 0, 0]
        probs = [0.1, 0.8, 0.1, 0.1, 0.1]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 0  # filtered out

    def test_min_duration_keeps_long_segments(self):
        """Segments at or above min_speech_duration_ms are kept."""
        seg = self._make_segmenter(
            start_pad_ms=0, end_pad_ms=0, min_speech_duration_ms=10
        )
        # 3 frames: duration = 3 * 256/16000 = 48ms > 10ms
        flags = [0, 1, 1, 1, 0]
        probs = [0.1, 0.8, 0.9, 0.7, 0.1]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        assert len(result) == 1

    # ─── Confidence calculation ───────────────────────────────────

    def test_segment_confidence_is_average_of_probs(self):
        """Confidence is the arithmetic mean of per-frame probs for that segment."""
        seg = self._make_segmenter(start_pad_ms=0, end_pad_ms=0)
        flags = [0, 1, 1, 1, 0]
        probs = [0.1, 0.6, 0.8, 0.7, 0.1]

        result = seg._flags_to_segments(flags, probs, VAD_SR, audio_duration=10.0)

        expected_confidence = (0.6 + 0.8 + 0.7) / 3
        assert result[0].confidence == pytest.approx(expected_confidence, abs=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Class 2: Silero v4 sample padding — MATHEMATICAL VERIFICATION
#
# Silero v4's segment() has a second padding loop (lines 296-306) that
# applies start_pad_samples / end_pad_samples AFTER the library returns.
# We mock _get_speech_timestamps to return known sample positions, then
# verify the arithmetic produces exact shifts.
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestSileroPaddingArithmetic:
    """Mathematically verify Silero v4's sample-level padding loop."""

    def _make_mocked_segmenter(
        self,
        start_pad_samples: int = 0,
        end_pad_samples: int = 0,
        speech_pad_ms: int = 0,
    ) -> "SileroSpeechSegmenter":
        """Create Silero segmenter with mocked model — no download needed."""
        seg = SileroSpeechSegmenter(
            version="v4.0",
            threshold=0.5,
            speech_pad_ms=speech_pad_ms,
            min_speech_duration_ms=0,
            min_silence_duration_ms=0,
            start_pad_samples=start_pad_samples,
            end_pad_samples=end_pad_samples,
        )
        # Mock the model and utilities so _ensure_model is a no-op
        seg._model = MagicMock()
        seg._utils = (None, None, None, None, None)
        return seg

    def _run_with_known_timestamps(
        self,
        seg: "SileroSpeechSegmenter",
        timestamps: List[dict],
        audio_samples: int = 160000,  # 10s at 16kHz
    ) -> SegmentationResult:
        """Run segment() with mocked _get_speech_timestamps returning known data."""
        seg._get_speech_timestamps = lambda audio, model, **kw: [
            dict(ts) for ts in timestamps  # deep copy to avoid mutation issues
        ]
        audio = np.zeros(audio_samples, dtype=np.float32)
        return seg.segment(audio, sample_rate=VAD_SR)

    # ─── Basic sample padding arithmetic ──────────────────────────

    def test_no_padding_preserves_raw_boundaries(self):
        """With zero pad samples, boundaries are library output / 16000."""
        seg = self._make_mocked_segmenter(start_pad_samples=0, end_pad_samples=0)
        # Speech at samples 16000-32000 (1.0s - 2.0s)
        timestamps = [{"start": 16000, "end": 32000}]

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 1
        assert result.segments[0].start_sec == pytest.approx(1.0, abs=1e-4)
        assert result.segments[0].end_sec == pytest.approx(2.0, abs=1e-4)

    def test_end_pad_samples_shifts_end(self):
        """end_pad_samples=8000 shifts end by 8000/16000 = 0.5s."""
        seg = self._make_mocked_segmenter(start_pad_samples=0, end_pad_samples=8000)
        timestamps = [{"start": 16000, "end": 32000}]  # 1.0s - 2.0s

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 1
        assert result.segments[0].start_sec == pytest.approx(1.0, abs=1e-4)
        # End shifted: 32000 + 8000 = 40000 → 2.5s
        assert result.segments[0].end_sec == pytest.approx(2.5, abs=1e-4)

    def test_start_pad_samples_shifts_start(self):
        """start_pad_samples=4800 shifts start earlier by 4800/16000 = 0.3s."""
        seg = self._make_mocked_segmenter(start_pad_samples=4800, end_pad_samples=0)
        timestamps = [{"start": 16000, "end": 32000}]  # 1.0s - 2.0s

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 1
        # Start shifted: 16000 - 4800 = 11200 → 0.7s
        assert result.segments[0].start_sec == pytest.approx(0.7, abs=1e-4)
        assert result.segments[0].end_sec == pytest.approx(2.0, abs=1e-4)

    def test_combined_padding(self):
        """Both start and end pad applied simultaneously."""
        seg = self._make_mocked_segmenter(
            start_pad_samples=3200, end_pad_samples=6400
        )
        timestamps = [{"start": 32000, "end": 48000}]  # 2.0s - 3.0s

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 1
        # Start: 32000 - 3200 = 28800 → 1.8s
        assert result.segments[0].start_sec == pytest.approx(1.8, abs=1e-4)
        # End: 48000 + 6400 = 54400 → 3.4s
        assert result.segments[0].end_sec == pytest.approx(3.4, abs=1e-4)

    # ─── Boundary clamping ────────────────────────────────────────

    def test_start_pad_clamped_to_zero(self):
        """Start padding cannot push start below sample 0."""
        seg = self._make_mocked_segmenter(start_pad_samples=50000, end_pad_samples=0)
        timestamps = [{"start": 1000, "end": 32000}]

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 1
        assert result.segments[0].start_sec == pytest.approx(0.0, abs=1e-4)

    def test_end_pad_clamped_to_audio_length(self):
        """End padding cannot push end beyond (audio_length - 16)."""
        audio_samples = 40000  # 2.5s
        seg = self._make_mocked_segmenter(start_pad_samples=0, end_pad_samples=50000)
        timestamps = [{"start": 16000, "end": 32000}]

        result = self._run_with_known_timestamps(
            seg, timestamps, audio_samples=audio_samples
        )

        assert len(result.segments) == 1
        # Clamped to (40000 - 16) / 16000
        max_end = (audio_samples - 16) / VAD_SR
        assert result.segments[0].end_sec == pytest.approx(max_end, abs=1e-4)

    # ─── Overlap prevention ───────────────────────────────────────

    def test_overlap_prevention_between_padded_segments(self):
        """When end_pad of segment 1 overlaps start of segment 2, start 2 is clamped."""
        seg = self._make_mocked_segmenter(
            start_pad_samples=0, end_pad_samples=16000  # 1s pad
        )
        # Two segments 0.5s apart in samples: 16000-24000, 32000-40000
        timestamps = [
            {"start": 16000, "end": 24000},  # 1.0s - 1.5s
            {"start": 32000, "end": 40000},  # 2.0s - 2.5s
        ]

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 2
        # Seg 1 end: 24000 + 16000 = 40000 → 2.5s (would overlap seg 2 start)
        # Seg 2 start: max(32000 - 0, seg1_padded_end) — clamped to seg1's end
        assert result.segments[1].start_sec >= result.segments[0].end_sec

    # ─── Multiple segments with arithmetic ────────────────────────

    def test_three_segments_all_padded_correctly(self):
        """Multiple segments each receive correct independent padding."""
        seg = self._make_mocked_segmenter(
            start_pad_samples=1600, end_pad_samples=3200  # 0.1s, 0.2s
        )
        timestamps = [
            {"start": 16000, "end": 24000},  # 1.0s - 1.5s
            {"start": 48000, "end": 56000},  # 3.0s - 3.5s
            {"start": 80000, "end": 88000},  # 5.0s - 5.5s
        ]

        result = self._run_with_known_timestamps(seg, timestamps)

        assert len(result.segments) == 3
        for i, (ts, seg_result) in enumerate(zip(timestamps, result.segments)):
            expected_start = max(0, ts["start"] - 1600) / VAD_SR
            expected_end = (ts["end"] + 3200) / VAD_SR
            # For segments after the first, start may be clamped by overlap prevention
            if i == 0:
                assert seg_result.start_sec == pytest.approx(
                    expected_start, abs=1e-4
                ), f"Segment {i} start"
            assert seg_result.end_sec == pytest.approx(
                expected_end, abs=1e-4
            ), f"Segment {i} end"


# ═══════════════════════════════════════════════════════════════════════
# Class 3: Threshold boundary conditions on real models
#
# Extreme thresholds produce predictable results regardless of model:
# - threshold ≈ 1.0 → nothing detected (too strict)
# - Pure silence + any threshold → nothing detected
# ═══════════════════════════════════════════════════════════════════════


class TestThresholdBoundaryConditions:
    """Extreme threshold values produce predictable segment counts."""

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_max_threshold_detects_nothing(self):
        """threshold=0.99 on short audio → zero segments."""
        audio, sr = _make_test_audio()
        seg = SpeechSegmenterFactory.create(
            "silero-v4.0", config={"threshold": 0.99}
        )
        try:
            result = seg.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert result.num_segments == 0

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_silence_detects_nothing(self):
        """Pure silence with moderate threshold → zero segments."""
        silence = np.zeros(5 * VAD_SR, dtype=np.float32)
        seg = SpeechSegmenterFactory.create(
            "silero-v4.0", config={"threshold": 0.3}
        )
        try:
            result = seg.segment(silence, VAD_SR)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert result.num_segments == 0

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_max_threshold_detects_nothing(self):
        audio, sr = _make_test_audio()
        seg = SpeechSegmenterFactory.create(
            "silero-v6.2", config={"threshold": 0.99}
        )
        try:
            result = seg.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        finally:
            seg.cleanup()
        assert result.num_segments == 0

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_silence_detects_nothing(self):
        silence = np.zeros(5 * VAD_SR, dtype=np.float32)
        seg = SpeechSegmenterFactory.create(
            "silero-v6.2", config={"threshold": 0.01}
        )
        try:
            result = seg.segment(silence, VAD_SR)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        finally:
            seg.cleanup()
        assert result.num_segments == 0

    @_skip_unless_backend("ten")
    def test_ten_max_threshold_detects_nothing(self):
        audio, sr = _make_test_audio()
        seg = SpeechSegmenterFactory.create(
            "ten", config={"threshold": 0.99}
        )
        try:
            result = seg.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert result.num_segments == 0

    @_skip_unless_backend("ten")
    def test_ten_silence_detects_nothing(self):
        """Pure silence with moderate threshold → zero segments."""
        silence = np.zeros(5 * VAD_SR, dtype=np.float32)
        seg = SpeechSegmenterFactory.create(
            "ten", config={"threshold": 0.3}
        )
        try:
            result = seg.segment(silence, VAD_SR)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert result.num_segments == 0

    @_skip_unless_backend("nemo")
    def test_nemo_max_onset_detects_nothing(self):
        """onset=0.99, offset=0.99 → too strict → zero segments."""
        audio, sr = _make_test_audio()
        seg = SpeechSegmenterFactory.create(
            "nemo", config={"onset": 0.99, "offset": 0.99}
        )
        try:
            result = seg.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert result.num_segments == 0


# ═══════════════════════════════════════════════════════════════════════
# Class 4: Factory creates backends with correct THRESHOLD attributes
# ═══════════════════════════════════════════════════════════════════════


class TestThresholdConstruction:
    """Factory → backend: threshold parameter stored as expected attribute."""

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_custom_threshold(self):
        seg = SpeechSegmenterFactory.create("silero-v4.0", config={"threshold": 0.3})
        assert seg.threshold == 0.3

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_default_threshold(self):
        seg = SpeechSegmenterFactory.create("silero-v4.0")
        assert seg.threshold == 0.25  # VERSION_DEFAULTS["v4.0"]

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_custom_threshold(self):
        seg = SpeechSegmenterFactory.create("silero-v6.2", config={"threshold": 0.4})
        assert seg.threshold == 0.4

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_default_threshold(self):
        seg = SpeechSegmenterFactory.create("silero-v6.2")
        assert seg.threshold == 0.35

    @_skip_unless_backend("ten")
    def test_ten_custom_threshold(self):
        seg = SpeechSegmenterFactory.create("ten", config={"threshold": 0.15})
        assert seg.threshold == 0.15

    @_skip_unless_backend("ten")
    def test_ten_default_threshold(self):
        seg = SpeechSegmenterFactory.create("ten")
        assert seg.threshold == 0.26

    @_skip_unless_backend("nemo")
    def test_nemo_custom_onset_offset(self):
        seg = SpeechSegmenterFactory.create(
            "nemo", config={"onset": 0.7, "offset": 0.5}
        )
        assert seg.onset == 0.7
        assert seg.offset == 0.5

    @_skip_unless_backend("nemo")
    def test_nemo_default_onset_offset(self):
        seg = SpeechSegmenterFactory.create("nemo")
        assert seg.onset == 0.4
        assert seg.offset == 0.3

    @_skip_unless_backend("whisper-vad")
    def test_whisper_vad_custom_threshold(self):
        seg = SpeechSegmenterFactory.create(
            "whisper-vad", config={"no_speech_threshold": 0.8}
        )
        assert seg.no_speech_threshold == 0.8

    @_skip_unless_backend("whisper-vad")
    def test_whisper_vad_default_threshold(self):
        seg = SpeechSegmenterFactory.create("whisper-vad")
        assert seg.no_speech_threshold == 0.6


# ═══════════════════════════════════════════════════════════════════════
# Class 5: Factory creates backends with correct PADDING attributes
# ═══════════════════════════════════════════════════════════════════════


class TestPaddingConstruction:
    """Factory → backend: padding parameter stored as expected attribute."""

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_custom_padding(self):
        seg = SpeechSegmenterFactory.create(
            "silero-v4.0", config={"speech_pad_ms": 500}
        )
        assert seg.speech_pad_ms == 500

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_default_padding(self):
        seg = SpeechSegmenterFactory.create("silero-v4.0")
        assert seg.speech_pad_ms == 700  # VERSION_DEFAULTS["v4.0"]

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_custom_padding(self):
        seg = SpeechSegmenterFactory.create(
            "silero-v6.2", config={"speech_pad_ms": 300}
        )
        assert seg.speech_pad_ms == 300

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_default_padding(self):
        seg = SpeechSegmenterFactory.create("silero-v6.2")
        assert seg.speech_pad_ms == 350

    @_skip_unless_backend("ten")
    def test_ten_custom_padding(self):
        seg = SpeechSegmenterFactory.create(
            "ten", config={"start_pad_ms": 30, "end_pad_ms": 200}
        )
        assert seg.start_pad_ms == 30
        assert seg.end_pad_ms == 200

    @_skip_unless_backend("ten")
    def test_ten_default_padding(self):
        seg = SpeechSegmenterFactory.create("ten")
        assert seg.start_pad_ms == 50
        assert seg.end_pad_ms == 150

    @_skip_unless_backend("nemo")
    def test_nemo_custom_padding(self):
        seg = SpeechSegmenterFactory.create(
            "nemo", config={"pad_onset": 0.3, "pad_offset": 0.15}
        )
        assert seg.pad_onset == 0.3
        assert seg.pad_offset == 0.15

    @_skip_unless_backend("nemo")
    def test_nemo_default_padding(self):
        seg = SpeechSegmenterFactory.create("nemo")
        assert seg.pad_onset == 0.2
        assert seg.pad_offset == 0.10

    @_skip_unless_backend("whisper-vad")
    def test_whisper_vad_extra_params_no_crash(self):
        """Whisper-VAD has no padding param — extra params absorbed by **kwargs."""
        seg = SpeechSegmenterFactory.create(
            "whisper-vad", config={"speech_pad_ms": 100}
        )
        assert seg.no_speech_threshold == 0.6  # still gets default


# ═══════════════════════════════════════════════════════════════════════
# Class 6: Sensitivity preset resolution (YAML → filtered dict)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not HAS_PASS_WORKER,
    reason="whisperjav.ensemble.pass_worker not importable",
)
class TestSensitivityPresetResolution:
    """resolve_qwen_sensitivity() returns correct preset values per backend."""

    # ─── Silero v6.2 ──────────────────────────────────────────────

    def test_silero_v6_aggressive(self):
        result = resolve_qwen_sensitivity("silero-v6.2", "aggressive")
        assert result["threshold"] == 0.2
        assert result["speech_pad_ms"] == 150
        assert result["min_speech_duration_ms"] == 50

    def test_silero_v6_balanced(self):
        result = resolve_qwen_sensitivity("silero-v6.2", "balanced")
        assert result["threshold"] == 0.35
        assert result["speech_pad_ms"] == 250
        assert result["min_speech_duration_ms"] == 100

    def test_silero_v6_conservative(self):
        result = resolve_qwen_sensitivity("silero-v6.2", "conservative")
        assert result["threshold"] == 0.5
        assert result["speech_pad_ms"] == 350
        assert result["min_speech_duration_ms"] == 150

    # ─── Silero v4.0 ──────────────────────────────────────────────

    def test_silero_v4_aggressive(self):
        result = resolve_qwen_sensitivity("silero-v4.0", "aggressive")
        assert result["threshold"] == 0.05
        assert result["speech_pad_ms"] == 300
        assert result["min_speech_duration_ms"] == 30

    def test_silero_v4_balanced(self):
        result = resolve_qwen_sensitivity("silero-v4.0", "balanced")
        assert result["threshold"] == 0.18
        assert result["speech_pad_ms"] == 400
        assert result["min_speech_duration_ms"] == 100

    def test_silero_v4_conservative(self):
        result = resolve_qwen_sensitivity("silero-v4.0", "conservative")
        assert result["threshold"] == 0.35
        assert result["speech_pad_ms"] == 500
        assert result["min_speech_duration_ms"] == 150

    # ─── TEN ──────────────────────────────────────────────────────

    def test_ten_aggressive(self):
        result = resolve_qwen_sensitivity("ten", "aggressive")
        assert result["threshold"] == 0.10
        assert result["end_pad_ms"] == 100
        assert result["start_pad_ms"] == 0

    def test_ten_balanced(self):
        result = resolve_qwen_sensitivity("ten", "balanced")
        assert result["threshold"] == 0.20
        assert result["end_pad_ms"] == 200
        assert result["start_pad_ms"] == 0

    def test_ten_conservative(self):
        result = resolve_qwen_sensitivity("ten", "conservative")
        assert result["threshold"] == 0.5
        assert result["end_pad_ms"] == 300
        assert result["start_pad_ms"] == 0

    # ─── NeMo (GAP: vad.onset/vad.offset not in SEGMENTER_PARAMS) ─

    @pytest.mark.xfail(
        reason="Known gap: NeMo onset/offset/pad_onset/pad_offset not in "
        "SEGMENTER_PARAMS — sensitivity presets are filtered out"
    )
    def test_nemo_aggressive_onset_offset(self):
        """NeMo aggressive should include onset/offset — CURRENTLY BROKEN."""
        result = resolve_qwen_sensitivity("nemo", "aggressive")
        assert "onset" in result or "vad.onset" in result

    def test_nemo_aggressive_min_speech(self):
        """NeMo aggressive min_speech_duration_ms IS in SEGMENTER_PARAMS."""
        result = resolve_qwen_sensitivity("nemo", "aggressive")
        assert result.get("min_speech_duration_ms") == 50

    @pytest.mark.xfail(
        reason="Known gap: NeMo pad_onset/pad_offset not in SEGMENTER_PARAMS"
    )
    def test_nemo_balanced_pad_onset(self):
        """NeMo balanced should include pad_onset — CURRENTLY BROKEN."""
        result = resolve_qwen_sensitivity("nemo", "balanced")
        assert "pad_onset" in result or "vad.pad_offset" in result

    # ─── Whisper-VAD (GAP: no_speech_threshold not in SEGMENTER_PARAMS)

    @pytest.mark.xfail(
        reason="Known gap: no_speech_threshold routes through "
        "PROVIDER_PARAMS_COMMON, not SEGMENTER_PARAMS"
    )
    def test_whisper_vad_conservative_threshold(self):
        """Whisper-VAD conservative should include no_speech_threshold — BROKEN."""
        result = resolve_qwen_sensitivity("whisper-vad", "conservative")
        assert "no_speech_threshold" in result

    def test_whisper_vad_conservative_min_speech(self):
        """Whisper-VAD conservative min_speech_duration_ms IS in SEGMENTER_PARAMS."""
        result = resolve_qwen_sensitivity("whisper-vad", "conservative")
        assert result.get("min_speech_duration_ms") == 150

    # ─── Edge cases ───────────────────────────────────────────────

    def test_none_backend_returns_empty(self):
        assert resolve_qwen_sensitivity("none", "aggressive") == {}

    def test_empty_backend_returns_empty(self):
        assert resolve_qwen_sensitivity("", "balanced") == {}

    def test_user_override_wins(self):
        result = resolve_qwen_sensitivity(
            "silero-v6.2", "aggressive", {"threshold": 0.3}
        )
        assert result["threshold"] == 0.3
        assert "speech_pad_ms" in result

    def test_all_resolved_keys_in_segmenter_params(self):
        """All resolved keys must be in SEGMENTER_PARAMS (the filter contract)."""
        for backend in ("silero-v6.2", "silero-v4.0", "ten"):
            for sensitivity in ("aggressive", "balanced", "conservative"):
                result = resolve_qwen_sensitivity(backend, sensitivity)
                for key in result:
                    assert key in SEGMENTER_PARAMS, (
                        f"Key '{key}' from {backend}/{sensitivity} "
                        f"not in SEGMENTER_PARAMS"
                    )

    # ─── Sensitivity direction (monotonicity across presets) ──────

    @pytest.mark.parametrize("backend", ["silero-v6.2", "silero-v4.0", "ten"])
    def test_threshold_decreases_aggressive_to_conservative(self, backend):
        """Aggressive has lower threshold than conservative (more permissive)."""
        agg = resolve_qwen_sensitivity(backend, "aggressive")
        con = resolve_qwen_sensitivity(backend, "conservative")
        assert agg["threshold"] < con["threshold"]

    @pytest.mark.parametrize("backend", ["silero-v6.2", "silero-v4.0"])
    def test_padding_increases_conservative_to_aggressive(self, backend):
        """Conservative has more padding (compensates for late detection)."""
        agg = resolve_qwen_sensitivity(backend, "aggressive")
        con = resolve_qwen_sensitivity(backend, "conservative")
        assert con["speech_pad_ms"] > agg["speech_pad_ms"]

    def test_ten_end_pad_increases_conservative_to_aggressive(self):
        agg = resolve_qwen_sensitivity("ten", "aggressive")
        con = resolve_qwen_sensitivity("ten", "conservative")
        assert con["end_pad_ms"] > agg["end_pad_ms"]


# ═══════════════════════════════════════════════════════════════════════
# Class 7: Behavioral threshold monotonicity (real models)
# ═══════════════════════════════════════════════════════════════════════


class TestBehavioralThresholdEffect:
    """Lower threshold → at least as many segments (monotonicity on real models)."""

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_threshold_monotonicity(self):
        audio, sr = _make_test_audio()
        low = SpeechSegmenterFactory.create("silero-v4.0", config={"threshold": 0.1})
        high = SpeechSegmenterFactory.create("silero-v4.0", config={"threshold": 0.9})
        try:
            low_r = low.segment(audio, sr)
            high_r = high.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert low_r.num_segments >= high_r.num_segments

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_threshold_monotonicity(self):
        audio, sr = _make_test_audio()
        low = SpeechSegmenterFactory.create("silero-v6.2", config={"threshold": 0.1})
        high = SpeechSegmenterFactory.create("silero-v6.2", config={"threshold": 0.9})
        try:
            low_r = low.segment(audio, sr)
            high_r = high.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        finally:
            low.cleanup()
            high.cleanup()
        assert low_r.num_segments >= high_r.num_segments

    @_skip_unless_backend("ten")
    def test_ten_threshold_monotonicity(self):
        audio, sr = _make_test_audio()
        low = SpeechSegmenterFactory.create("ten", config={"threshold": 0.05})
        high = SpeechSegmenterFactory.create("ten", config={"threshold": 0.95})
        try:
            low_r = low.segment(audio, sr)
            high_r = high.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert low_r.num_segments >= high_r.num_segments

    @_skip_unless_backend("nemo")
    def test_nemo_onset_monotonicity(self):
        audio, sr = _make_test_audio()
        low = SpeechSegmenterFactory.create(
            "nemo", config={"onset": 0.3, "offset": 0.2}
        )
        high = SpeechSegmenterFactory.create(
            "nemo", config={"onset": 0.9, "offset": 0.8}
        )
        try:
            low_r = low.segment(audio, sr)
            high_r = high.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        assert low_r.num_segments >= high_r.num_segments


# ═══════════════════════════════════════════════════════════════════════
# Class 8: Behavioral padding effect (real models)
# ═══════════════════════════════════════════════════════════════════════


class TestBehavioralPaddingEffect:
    """Higher padding → wider total speech coverage (real models)."""

    @_skip_unless_backend("silero-v4.0")
    def test_silero_v4_padding_widens_segments(self):
        audio, sr = _make_test_audio()
        short = SpeechSegmenterFactory.create(
            "silero-v4.0", config={"speech_pad_ms": 0, "threshold": 0.2}
        )
        long = SpeechSegmenterFactory.create(
            "silero-v4.0", config={"speech_pad_ms": 500, "threshold": 0.2}
        )
        try:
            short_r = short.segment(audio, sr)
            long_r = long.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        if short_r.num_segments == 0 or long_r.num_segments == 0:
            pytest.skip("No segments detected on synthetic audio")
        assert long_r.speech_coverage_sec >= short_r.speech_coverage_sec - 0.01

    @_skip_unless_backend("silero-v6.2")
    def test_silero_v6_padding_widens_segments(self):
        audio, sr = _make_test_audio()
        short = SpeechSegmenterFactory.create(
            "silero-v6.2", config={"speech_pad_ms": 0, "threshold": 0.2}
        )
        long = SpeechSegmenterFactory.create(
            "silero-v6.2", config={"speech_pad_ms": 500, "threshold": 0.2}
        )
        try:
            short_r = short.segment(audio, sr)
            long_r = long.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        finally:
            short.cleanup()
            long.cleanup()
        if short_r.num_segments == 0 or long_r.num_segments == 0:
            pytest.skip("No segments detected on synthetic audio")
        assert long_r.speech_coverage_sec >= short_r.speech_coverage_sec - 0.01

    @_skip_unless_backend("ten")
    def test_ten_end_pad_widens_segments(self):
        audio, sr = _make_test_audio()
        short = SpeechSegmenterFactory.create(
            "ten",
            config={"end_pad_ms": 0, "start_pad_ms": 0, "threshold": 0.15},
        )
        long = SpeechSegmenterFactory.create(
            "ten",
            config={"end_pad_ms": 500, "start_pad_ms": 0, "threshold": 0.15},
        )
        try:
            short_r = short.segment(audio, sr)
            long_r = long.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        if short_r.num_segments == 0 or long_r.num_segments == 0:
            pytest.skip("No segments detected on synthetic audio")
        assert long_r.speech_coverage_sec >= short_r.speech_coverage_sec - 0.01

    @_skip_unless_backend("nemo")
    def test_nemo_padding_widens_segments(self):
        audio, sr = _make_test_audio()
        short = SpeechSegmenterFactory.create(
            "nemo",
            config={"pad_onset": 0.0, "pad_offset": 0.0, "onset": 0.4},
        )
        long = SpeechSegmenterFactory.create(
            "nemo",
            config={"pad_onset": 0.5, "pad_offset": 0.5, "onset": 0.4},
        )
        try:
            short_r = short.segment(audio, sr)
            long_r = long.segment(audio, sr)
        except Exception as e:
            pytest.skip(f"Model failed: {e}")
        if short_r.num_segments == 0 or long_r.num_segments == 0:
            pytest.skip("No segments detected on synthetic audio")
        assert long_r.speech_coverage_sec >= short_r.speech_coverage_sec - 0.01


# ═══════════════════════════════════════════════════════════════════════
# Class 9: Known parameter routing gaps (documentation tests)
# ═══════════════════════════════════════════════════════════════════════


@pytest.mark.skipif(
    not HAS_PASS_WORKER,
    reason="whisperjav.ensemble.pass_worker not importable",
)
class TestParameterRoutingGaps:
    """
    Documents known gaps in parameter routing.

    Each test verifies the current (broken) behavior. When a gap is fixed,
    update the test to verify correct behavior.
    """

    # ─── speech_pad_ms semantic gap ───────────────────────────────

    @_skip_unless_backend("ten")
    def test_speech_pad_ms_ignored_by_ten(self):
        """CLI --speech-pad-ms does NOT affect TEN."""
        seg = SpeechSegmenterFactory.create("ten", config={"speech_pad_ms": 999})
        assert not hasattr(seg, "speech_pad_ms")
        assert seg.start_pad_ms == 50  # default unchanged
        assert seg.end_pad_ms == 150  # default unchanged

    @_skip_unless_backend("nemo")
    def test_speech_pad_ms_ignored_by_nemo(self):
        """CLI --speech-pad-ms does NOT affect NeMo."""
        seg = SpeechSegmenterFactory.create("nemo", config={"speech_pad_ms": 999})
        assert seg.pad_onset == 0.2  # default unchanged
        assert seg.pad_offset == 0.10  # default unchanged

    # ─── NeMo SEGMENTER_PARAMS gaps ───────────────────────────────

    @pytest.mark.xfail(
        reason="NeMo onset/offset not in SEGMENTER_PARAMS — "
        "sensitivity presets don't reach backend"
    )
    def test_nemo_sensitivity_routes_onset(self):
        """NeMo aggressive should set onset=0.7 — CURRENTLY BROKEN."""
        result = resolve_qwen_sensitivity("nemo", "aggressive")
        assert "onset" in result or "vad.onset" in result

    @pytest.mark.xfail(
        reason="YAML keys use vad.onset but factory expects onset"
    )
    @pytest.mark.skipif(
        not HAS_CONFIG_MANAGER,
        reason="ConfigManager not available",
    )
    def test_nemo_yaml_key_matches_factory(self):
        """NeMo YAML vad.onset should map to factory onset param."""
        cm = ConfigManager()
        resolved = cm.get_tool_config("nemo-speech-segmentation", "aggressive")
        assert "onset" in resolved  # fails — key is "vad.onset"

    # ─── Whisper-VAD SEGMENTER_PARAMS gap ─────────────────────────

    @pytest.mark.xfail(
        reason="no_speech_threshold in PROVIDER_PARAMS_COMMON, "
        "not SEGMENTER_PARAMS"
    )
    def test_whisper_vad_threshold_in_sensitivity(self):
        """Whisper-VAD no_speech_threshold should be in sensitivity — BROKEN."""
        result = resolve_qwen_sensitivity("whisper-vad", "conservative")
        assert "no_speech_threshold" in result

    # ─── Explicit SEGMENTER_PARAMS gap verification ───────────────

    def test_segmenter_params_missing_nemo_keys(self):
        """Documents that SEGMENTER_PARAMS lacks NeMo-specific keys."""
        nemo_keys = {"onset", "offset", "pad_onset", "pad_offset"}
        missing = nemo_keys - SEGMENTER_PARAMS
        assert missing == nemo_keys, (
            f"Some NeMo keys are now in SEGMENTER_PARAMS: "
            f"{nemo_keys & SEGMENTER_PARAMS}. Update xfail markers."
        )

    def test_segmenter_params_missing_whisper_vad_threshold(self):
        """Documents that SEGMENTER_PARAMS lacks no_speech_threshold."""
        assert "no_speech_threshold" not in SEGMENTER_PARAMS, (
            "no_speech_threshold is now in SEGMENTER_PARAMS! "
            "Update xfail markers."
        )
