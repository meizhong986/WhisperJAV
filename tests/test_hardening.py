"""
Unit tests for subtitle pipeline hardening functions.

Tests cover the 4 timestamp gap fixes:
    G1 complement: vad_only produces frame-boundary timing (no-op hardening)
    G2: aligner_vad_fallback distributes null segments within speech regions
    G3: (tested in orchestrator integration, not here)
    G4: vad_only hardening is a no-op (timestamps unchanged)

Also includes regression guard for aligner_interpolation (unchanged behavior).
"""

import pytest
from types import SimpleNamespace
from typing import Optional


# ---------------------------------------------------------------------------
# Mock stable_whisper types (lightweight, no real dependency needed)
# ---------------------------------------------------------------------------


class MockWord:
    """Minimal mock of stable_whisper.Word."""

    def __init__(self, word: str, start: float, end: float):
        self.word = word
        self.start = start
        self.end = end

    def __repr__(self):
        return f"MockWord({self.word!r}, {self.start:.3f}, {self.end:.3f})"


class MockSegment:
    """Minimal mock of stable_whisper.Segment."""

    def __init__(self, text: str, start: float, end: float, words: Optional[list] = None):
        self.text = text
        self._start = start
        self._end = end
        self.words = words or []

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, value):
        self._start = value

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, value):
        self._end = value

    def __repr__(self):
        return f"MockSegment({self.text!r}, {self.start:.3f}-{self.end:.3f})"


class MockResult:
    """Minimal mock of stable_whisper.WhisperResult."""

    def __init__(self, segments: list):
        self.segments = segments


# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------

from whisperjav.modules.subtitle_pipeline.hardening import (
    _apply_vad_only_timestamps,
    _apply_vad_timestamp_fallback,
    _apply_timestamp_interpolation,
    _timeline_to_real,
    _clip_regions,
    harden_scene_result,
)
from whisperjav.modules.subtitle_pipeline.types import (
    HardeningConfig,
    TimestampMode,
)


# ===========================================================================
# G4: vad_only hardening is a no-op
# ===========================================================================


class TestVadOnlyNoOp:
    """G4: _apply_vad_only_timestamps must NOT modify timestamps."""

    def test_timestamps_unchanged(self):
        """Segments should retain their original timestamps after vad_only hardening."""
        seg1 = MockSegment("こんにちは", 1.0, 3.0)
        seg2 = MockSegment("お元気ですか", 5.0, 8.0)
        result = MockResult([seg1, seg2])

        _apply_vad_only_timestamps(result, group_duration=10.0)

        assert seg1.start == 1.0
        assert seg1.end == 3.0
        assert seg2.start == 5.0
        assert seg2.end == 8.0

    def test_word_level_timestamps_unchanged(self):
        """Word-level timestamps should also be preserved."""
        w1 = MockWord("こん", 1.0, 1.5)
        w2 = MockWord("にちは", 1.5, 3.0)
        seg = MockSegment("こんにちは", 1.0, 3.0, words=[w1, w2])
        result = MockResult([seg])

        _apply_vad_only_timestamps(result, group_duration=10.0)

        assert w1.start == 1.0
        assert w1.end == 1.5
        assert w2.start == 1.5
        assert w2.end == 3.0

    def test_empty_result_no_error(self):
        """Should handle empty results gracefully."""
        _apply_vad_only_timestamps(None, 10.0)
        _apply_vad_only_timestamps(MockResult([]), 10.0)

    def test_full_pipeline_vad_only(self):
        """Full harden_scene_result with VAD_ONLY should not modify timestamps."""
        seg = MockSegment("テスト", 2.0, 4.0)
        result = MockResult([seg])
        config = HardeningConfig(
            timestamp_mode=TimestampMode.VAD_ONLY,
            scene_duration_sec=10.0,
        )

        diag = harden_scene_result(result, config)

        assert seg.start == 2.0
        assert seg.end == 4.0
        assert diag.timestamp_mode == "vad_only"


# ===========================================================================
# G2: aligner_vad_fallback with speech_regions
# ===========================================================================


class TestVadFallbackWithSpeechRegions:
    """G2: _apply_vad_timestamp_fallback should use speech_regions."""

    def test_null_segments_land_in_speech(self):
        """Null-timestamp segments should be anchored to speech region positions."""
        # Speech at 2-4s and 7-9s. Silence at 0-2s, 4-7s, 9-10s.
        speech_regions = [(2.0, 4.0), (7.0, 9.0)]

        # All segments have null timestamps (end <= 0)
        seg1 = MockSegment("ああ", 0.0, 0.0)
        seg2 = MockSegment("うう", 0.0, 0.0)
        result = MockResult([seg1, seg2])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=speech_regions)

        assert count == 2

        # Total speech = 4s. Equal char count → each segment gets 2s of speech time.
        # seg1: speech timeline [0, 2) → real [2.0, 4.0] (first region)
        # seg2: speech timeline [2, 4) → real [4.0, 9.0] (boundary→end of region 2)
        # Key property: segment starts are at speech positions (not in silence gaps)
        assert seg1.start >= 2.0, f"seg1.start={seg1.start:.3f} before speech"
        assert seg1.end <= 4.05, f"seg1.end={seg1.end:.3f} past first speech region"
        # seg2 may span across the silence gap (boundary effect), but its end
        # is anchored to speech region 2
        assert seg2.end <= 9.05, f"seg2.end={seg2.end:.3f} past last speech region"
        # Non-overlapping
        assert seg1.end <= seg2.start + 0.001

    def test_anchored_segments_with_gaps_in_speech(self):
        """Gaps between anchors should be filled within speech regions."""
        speech_regions = [(0.0, 3.0), (6.0, 10.0)]

        # Anchor at index 0, null at index 1, anchor at index 2
        seg_anchor1 = MockSegment("はい", 0.5, 2.0)  # valid anchor
        seg_null = MockSegment("そうですね", 0.0, 0.0)  # null — needs fallback
        seg_anchor2 = MockSegment("わかった", 7.0, 9.0)  # valid anchor

        result = MockResult([seg_anchor1, seg_null, seg_anchor2])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=speech_regions)

        assert count == 1  # Only seg_null needed fallback
        # The null segment should be between anchor1.end (2.0) and anchor2.start (7.0)
        assert seg_null.start >= 2.0
        assert seg_null.end <= 7.0
        # And within the clipped speech region in that window: [2.0, 3.0] and [6.0, 7.0]
        assert (
            (2.0 <= seg_null.start <= 3.0) or (6.0 <= seg_null.start <= 7.0)
        ), f"seg_null.start={seg_null.start:.3f} not in speech"

    def test_no_speech_regions_proportional_fallback(self):
        """Without speech regions, should fall back to proportional distribution."""
        seg1 = MockSegment("ああ", 0.0, 0.0)
        seg2 = MockSegment("いい", 0.0, 0.0)
        result = MockResult([seg1, seg2])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=None)

        assert count == 2
        # Should be distributed across [0, 10]
        assert seg1.start >= 0.0
        assert seg2.end <= 10.0
        assert seg1.end <= seg2.start + 0.001  # non-overlapping

    def test_no_null_segments_returns_zero(self):
        """If all segments have valid timestamps, return 0."""
        seg1 = MockSegment("はい", 1.0, 3.0)
        seg2 = MockSegment("ええ", 4.0, 6.0)
        result = MockResult([seg1, seg2])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=[(0.0, 10.0)])

        assert count == 0
        # Timestamps unchanged
        assert seg1.start == 1.0
        assert seg2.end == 6.0

    def test_trailing_gap_in_speech(self):
        """Trailing null segments after last anchor should land in speech."""
        speech_regions = [(0.0, 5.0), (8.0, 10.0)]

        seg_anchor = MockSegment("はい", 1.0, 3.0)  # valid
        seg_null = MockSegment("そう", 0.0, 0.0)  # null trailing
        result = MockResult([seg_anchor, seg_null])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=speech_regions)

        assert count == 1
        # Should be between anchor.end (3.0) and group_duration (10.0)
        # Speech in that window: [3.0, 5.0] and [8.0, 10.0]
        assert seg_null.start >= 3.0
        assert seg_null.end <= 10.05

    def test_word_level_distribution(self):
        """When segments have words, timestamps should be set at word level."""
        speech_regions = [(2.0, 6.0)]

        w1 = MockWord("こん", 0.0, 0.0)
        w2 = MockWord("にちは", 0.0, 0.0)
        seg = MockSegment("こんにちは", 0.0, 0.0, words=[w1, w2])
        result = MockResult([seg])

        count = _apply_vad_timestamp_fallback(result, 10.0, speech_regions=speech_regions)

        assert count == 1
        # Words should be within speech region [2.0, 6.0]
        assert w1.start >= 2.0
        assert w2.end <= 6.05
        assert w1.end <= w2.start + 0.001  # ordered

    def test_full_pipeline_aligner_vad_fallback(self):
        """Full harden_scene_result with ALIGNER_WITH_VAD_FALLBACK passes speech_regions."""
        speech_regions = [(1.0, 4.0), (6.0, 9.0)]

        seg_ok = MockSegment("はい", 1.5, 3.0)  # valid
        seg_null = MockSegment("うん", 0.0, 0.0)  # null
        result = MockResult([seg_ok, seg_null])

        config = HardeningConfig(
            timestamp_mode=TimestampMode.ALIGNER_WITH_VAD_FALLBACK,
            scene_duration_sec=10.0,
            speech_regions=speech_regions,
        )

        diag = harden_scene_result(result, config)

        assert diag.fallback_count == 1
        assert seg_null.start > 0.0  # was redistributed


# ===========================================================================
# Interpolation regression guard (unchanged behavior)
# ===========================================================================


class TestInterpolationRegression:
    """Ensure _apply_timestamp_interpolation is unmodified."""

    def test_basic_interpolation(self):
        """Gaps between anchors should be interpolated proportionally."""
        seg1 = MockSegment("はい", 0.0, 2.0)  # anchor
        seg2 = MockSegment("そう", 0.0, 0.0)  # null
        seg3 = MockSegment("です", 5.0, 7.0)  # anchor
        result = MockResult([seg1, seg2, seg3])

        count = _apply_timestamp_interpolation(result, 10.0)

        assert count == 1
        # seg2 should be interpolated between seg1.end (2.0) and seg3.start (5.0)
        assert seg2.start >= 2.0
        assert seg2.end <= 5.0

    def test_no_nulls_returns_zero(self):
        """If all segments have valid timestamps, return 0."""
        seg1 = MockSegment("はい", 0.0, 2.0)
        seg2 = MockSegment("ええ", 3.0, 5.0)
        result = MockResult([seg1, seg2])

        count = _apply_timestamp_interpolation(result, 10.0)
        assert count == 0

    def test_all_nulls_returns_zero(self):
        """If ALL segments are null (no anchors), cannot interpolate."""
        seg1 = MockSegment("ああ", 0.0, 0.0)
        seg2 = MockSegment("いい", 0.0, 0.0)
        result = MockResult([seg1, seg2])

        count = _apply_timestamp_interpolation(result, 10.0)
        assert count == 0

    def test_leading_gap(self):
        """Leading null segments should be interpolated from 0 to first anchor."""
        seg1 = MockSegment("ああ", 0.0, 0.0)  # null
        seg2 = MockSegment("はい", 3.0, 5.0)  # anchor
        result = MockResult([seg1, seg2])

        count = _apply_timestamp_interpolation(result, 10.0)

        assert count == 1
        assert seg1.start == 0.0
        assert seg1.end <= 3.0


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestTimelineToReal:
    """Test _timeline_to_real mapping."""

    def test_single_region(self):
        regions = [(2.0, 5.0)]  # 3s of speech
        assert _timeline_to_real(0.0, regions) == 2.0
        assert _timeline_to_real(1.5, regions) == 3.5
        assert _timeline_to_real(3.0, regions) == 5.0

    def test_two_regions_with_gap(self):
        regions = [(1.0, 3.0), (6.0, 8.0)]  # 2s + 2s = 4s speech
        # First region: timeline [0, 2] → real [1, 3]
        assert _timeline_to_real(0.0, regions) == 1.0
        assert _timeline_to_real(1.0, regions) == 2.0
        # At the boundary (timeline=2.0), >= maps to end of first region (3.0)
        # This is correct: the closed interval includes the endpoint.
        assert _timeline_to_real(2.0, regions) == 3.0
        # Just past the boundary enters the second region
        assert _timeline_to_real(2.01, regions) == pytest.approx(6.01, abs=0.01)
        # Midpoint of second region
        assert _timeline_to_real(3.0, regions) == 7.0

    def test_past_end_clamps(self):
        regions = [(0.0, 2.0)]
        assert _timeline_to_real(5.0, regions) == 2.0

    def test_empty_regions(self):
        assert _timeline_to_real(1.0, []) == 0.0


class TestClipRegions:
    """Test _clip_regions helper."""

    def test_full_overlap(self):
        regions = [(2.0, 5.0)]
        assert _clip_regions(regions, 0.0, 10.0) == [(2.0, 5.0)]

    def test_partial_overlap(self):
        regions = [(2.0, 8.0)]
        assert _clip_regions(regions, 4.0, 6.0) == [(4.0, 6.0)]

    def test_no_overlap(self):
        regions = [(2.0, 4.0)]
        assert _clip_regions(regions, 5.0, 7.0) == []

    def test_multiple_regions(self):
        regions = [(1.0, 3.0), (5.0, 7.0), (9.0, 11.0)]
        clipped = _clip_regions(regions, 2.0, 6.0)
        assert clipped == [(2.0, 3.0), (5.0, 6.0)]
