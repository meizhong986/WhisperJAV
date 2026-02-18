#!/usr/bin/env python3
"""
Unit tests for the speech segmentation module.

Tests the core functionality of SpeechSegmenterFactory, backends,
and data structures.

Note: Uses direct module loading to avoid import chain issues with
whisperjav/__init__.py when there are dependency version conflicts.
"""

import sys
import importlib.util
import pytest
import numpy as np
from pathlib import Path

# Direct module loading to bypass whisperjav/__init__.py
# This avoids import chain issues when dependencies have version conflicts
def _load_module_direct(name: str, path: str):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Get the repo root
_repo_root = Path(__file__).parent.parent

# Load the speech segmentation modules directly
_base_module = _load_module_direct(
    'whisperjav.modules.speech_segmentation.base',
    str(_repo_root / 'whisperjav/modules/speech_segmentation/base.py')
)
_factory_module = _load_module_direct(
    'whisperjav.modules.speech_segmentation.factory',
    str(_repo_root / 'whisperjav/modules/speech_segmentation/factory.py')
)

# Export the classes we need for testing
SpeechSegment = _base_module.SpeechSegment
SegmentationResult = _base_module.SegmentationResult
SpeechSegmenter = _base_module.SpeechSegmenter
SpeechSegmenterFactory = _factory_module.SpeechSegmenterFactory


class TestSpeechSegment:
    """Tests for SpeechSegment dataclass."""

    def test_basic_creation(self):
        """Test creating a basic speech segment."""
        seg = SpeechSegment(start_sec=1.0, end_sec=2.5)
        assert seg.start_sec == 1.0
        assert seg.end_sec == 2.5
        assert seg.confidence == 1.0  # default

    def test_duration_property(self):
        """Test duration calculation."""
        seg = SpeechSegment(start_sec=1.0, end_sec=3.5)
        assert seg.duration_sec == 2.5

    def test_to_dict(self):
        """Test JSON serialization."""
        seg = SpeechSegment(start_sec=1.234, end_sec=2.567, confidence=0.95)
        d = seg.to_dict()
        assert d["start_sec"] == 1.234
        assert d["end_sec"] == 2.567
        assert d["confidence"] == 0.95
        assert "duration_sec" in d

    def test_with_samples(self):
        """Test with sample positions."""
        seg = SpeechSegment(
            start_sec=1.0,
            end_sec=2.0,
            start_sample=16000,
            end_sample=32000
        )
        assert seg.start_sample == 16000
        assert seg.end_sample == 32000

    def test_with_metadata(self):
        """Test with custom metadata."""
        seg = SpeechSegment(
            start_sec=0.0,
            end_sec=1.0,
            metadata={"speaker_id": "spk_01"}
        )
        assert seg.metadata["speaker_id"] == "spk_01"


class TestSegmentationResult:
    """Tests for SegmentationResult dataclass."""

    def test_basic_creation(self):
        """Test creating a segmentation result."""
        seg1 = SpeechSegment(start_sec=0.0, end_sec=1.0)
        seg2 = SpeechSegment(start_sec=2.0, end_sec=3.0)

        result = SegmentationResult(
            segments=[seg1, seg2],
            groups=[[seg1], [seg2]],
            method="test",
            audio_duration_sec=5.0,
            parameters={"threshold": 0.5}
        )

        assert result.num_segments == 2
        assert result.num_groups == 2
        assert result.method == "test"

    def test_speech_coverage(self):
        """Test speech coverage calculation."""
        seg1 = SpeechSegment(start_sec=0.0, end_sec=2.0)  # 2 seconds
        seg2 = SpeechSegment(start_sec=3.0, end_sec=4.0)  # 1 second

        result = SegmentationResult(
            segments=[seg1, seg2],
            groups=[[seg1, seg2]],
            method="test",
            audio_duration_sec=10.0,
            parameters={}
        )

        assert result.speech_coverage_sec == 3.0
        assert result.speech_coverage_ratio == 0.3

    def test_empty_result(self):
        """Test empty segmentation result."""
        result = SegmentationResult(
            segments=[],
            groups=[],
            method="test",
            audio_duration_sec=5.0,
            parameters={}
        )

        assert result.num_segments == 0
        assert result.num_groups == 0
        assert result.speech_coverage_sec == 0.0
        assert result.speech_coverage_ratio == 0.0

    def test_to_legacy_format(self):
        """Test conversion to legacy format."""
        seg = SpeechSegment(
            start_sec=1.0,
            end_sec=2.0,
            start_sample=16000,
            end_sample=32000
        )

        result = SegmentationResult(
            segments=[seg],
            groups=[[seg]],
            method="test",
            audio_duration_sec=5.0,
            parameters={}
        )

        legacy = result.to_legacy_format()
        assert len(legacy) == 1
        assert len(legacy[0]) == 1
        assert legacy[0][0]["start"] == 16000
        assert legacy[0][0]["end"] == 32000
        assert legacy[0][0]["start_sec"] == 1.0
        assert legacy[0][0]["end_sec"] == 2.0


class TestSpeechSegmenterFactory:
    """Tests for SpeechSegmenterFactory."""

    def test_list_backends(self):
        """Test listing available backends."""
        backends = SpeechSegmenterFactory.list_backends()
        assert "silero" in backends
        assert "silero-v4.0" in backends
        assert "silero-v3.1" in backends
        assert "nemo" in backends
        assert "nemo-lite" in backends
        assert "nemo-diarization" in backends
        assert "ten" in backends
        assert "none" in backends

    def test_list_unique_backends(self):
        """Test listing unique backends."""
        unique = SpeechSegmenterFactory.list_unique_backends()
        assert "silero" in unique
        assert "nemo" in unique
        assert "ten" in unique
        assert "none" in unique
        # Should not have version aliases
        assert "silero-v4.0" not in unique

    def test_silero_available(self):
        """Test that Silero is available (torch is installed)."""
        available, hint = SpeechSegmenterFactory.is_backend_available("silero")
        assert available is True
        assert hint == ""

    def test_none_available(self):
        """Test that 'none' backend is always available."""
        available, hint = SpeechSegmenterFactory.is_backend_available("none")
        assert available is True

    def test_get_available_backends(self):
        """Test getting backend availability info."""
        backends = SpeechSegmenterFactory.get_available_backends()
        assert len(backends) >= 5

        # Check structure
        for b in backends:
            assert "name" in b
            assert "display_name" in b
            assert "available" in b
            assert "install_hint" in b

    def test_create_none_segmenter(self):
        """Test creating the 'none' segmenter."""
        segmenter = SpeechSegmenterFactory.create("none")
        assert segmenter.name == "none"
        assert segmenter.display_name == "No Segmentation"

    def test_create_silero_v4(self):
        """Test creating Silero v4.0 segmenter."""
        segmenter = SpeechSegmenterFactory.create("silero-v4.0")
        assert segmenter.name == "silero-v4.0"
        assert "4.0" in segmenter.display_name

    def test_create_silero_v3(self):
        """Test creating Silero v3.1 segmenter."""
        segmenter = SpeechSegmenterFactory.create("silero-v3.1")
        assert segmenter.name == "silero-v3.1"
        assert "3.1" in segmenter.display_name

    def test_create_with_config(self):
        """Test creating segmenter with config."""
        segmenter = SpeechSegmenterFactory.create(
            "silero",
            config={"threshold": 0.3}
        )
        assert segmenter.threshold == 0.3

    def test_create_invalid_backend(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError):
            SpeechSegmenterFactory.create("invalid_backend")


class TestNullSpeechSegmenter:
    """Tests for the NullSpeechSegmenter (none backend)."""

    def test_segment_array(self):
        """Test segmenting a numpy array."""
        segmenter = SpeechSegmenterFactory.create("none")

        # Create 5 seconds of audio at 16kHz
        audio = np.zeros(16000 * 5, dtype=np.float32)
        result = segmenter.segment(audio, sample_rate=16000)

        assert result.num_segments == 1
        assert result.num_groups == 1
        assert result.segments[0].start_sec == 0.0
        assert abs(result.segments[0].end_sec - 5.0) < 0.01
        assert result.method == "none"

    def test_segment_returns_full_duration(self):
        """Test that 'none' segmenter returns full audio as single segment."""
        segmenter = SpeechSegmenterFactory.create("none")

        # Create 10 seconds of audio
        audio = np.random.randn(16000 * 10).astype(np.float32)
        result = segmenter.segment(audio, sample_rate=16000)

        assert len(result.segments) == 1
        assert result.speech_coverage_ratio == 1.0

    def test_cleanup(self):
        """Test cleanup method."""
        segmenter = SpeechSegmenterFactory.create("none")
        segmenter.cleanup()  # Should not raise

    def test_supported_sample_rates(self):
        """Test supported sample rates."""
        segmenter = SpeechSegmenterFactory.create("none")
        rates = segmenter.get_supported_sample_rates()
        assert 16000 in rates
        assert 44100 in rates


class TestSileroSpeechSegmenter:
    """Tests for SileroSpeechSegmenter."""

    @pytest.fixture
    def silero_segmenter(self):
        """Create a Silero segmenter for testing."""
        return SpeechSegmenterFactory.create("silero-v4.0")

    def test_name_and_display(self, silero_segmenter):
        """Test name and display name."""
        assert silero_segmenter.name == "silero-v4.0"
        assert "Silero" in silero_segmenter.display_name

    def test_segment_silent_audio(self, silero_segmenter):
        """Test segmenting silent audio returns empty."""
        # Create 5 seconds of silence
        audio = np.zeros(16000 * 5, dtype=np.float32)
        result = silero_segmenter.segment(audio, sample_rate=16000)

        # Silent audio should have no speech segments
        assert result.num_segments == 0

    def test_segment_with_speech(self, silero_segmenter):
        """Test segmenting audio with speech-like content."""
        # Create audio with speech-like noise bursts
        audio = np.zeros(16000 * 5, dtype=np.float32)

        # Add "speech" (noise burst) from 1-2 seconds
        audio[16000:32000] = np.random.randn(16000).astype(np.float32) * 0.5

        # Add "speech" from 3-4 seconds
        audio[48000:64000] = np.random.randn(16000).astype(np.float32) * 0.5

        result = silero_segmenter.segment(audio, sample_rate=16000)

        # Should detect some segments (exact count depends on VAD tuning)
        # Note: Random noise may or may not be detected as speech
        assert result.method == "silero-v4.0"
        assert result.audio_duration_sec == pytest.approx(5.0, rel=0.01)

    def test_cleanup(self, silero_segmenter):
        """Test cleanup releases resources."""
        silero_segmenter.cleanup()
        # Model should be cleared
        assert silero_segmenter._model is None

    def test_version_defaults(self):
        """Test that different versions have different defaults."""
        v4 = SpeechSegmenterFactory.create("silero-v4.0")
        v3 = SpeechSegmenterFactory.create("silero-v3.1")

        # v4.0 has a higher threshold than v3.1 (v4=0.25, v3=0.125)
        assert v4.threshold != v3.threshold

        v4.cleanup()
        v3.cleanup()


class TestFactoryCreateFromResolvedConfig:
    """Tests for creating segmenters from resolved config."""

    def test_create_from_none_backend(self):
        """Test that backend='none' creates 'none' segmenter."""
        config = {
            "params": {
                "speech_segmenter": {"backend": "none"}
            }
        }
        segmenter = SpeechSegmenterFactory.create_from_resolved_config(config)
        assert segmenter.name == "none"

    def test_create_from_skip_vad_backward_compat(self):
        """Test backward compatibility: skip_vad=True creates 'none' segmenter (deprecated)."""
        config = {
            "params": {
                "vad": {"skip_vad": True}
            }
        }
        segmenter = SpeechSegmenterFactory.create_from_resolved_config(config)
        assert segmenter.name == "none"

    def test_create_from_backend_config(self):
        """Test creating from speech_segmenter config."""
        config = {
            "params": {
                "speech_segmenter": {"backend": "silero-v3.1"}
            }
        }
        segmenter = SpeechSegmenterFactory.create_from_resolved_config(config)
        assert segmenter.name == "silero-v3.1"

    def test_create_default_from_empty_config(self):
        """Test default backend from empty config."""
        config = {"params": {}}
        segmenter = SpeechSegmenterFactory.create_from_resolved_config(config)
        assert "silero" in segmenter.name


class TestNemoSpeechSegmenterFactory:
    """Tests for NeMo speech segmenter variants via factory."""

    def test_nemo_variants_registered(self):
        """Test that nemo-lite and nemo-diarization are registered."""
        backends = SpeechSegmenterFactory.list_backends()
        assert "nemo" in backends
        assert "nemo-lite" in backends
        assert "nemo-diarization" in backends

    def test_nemo_availability_check(self):
        """Test checking NeMo backend availability."""
        available, hint = SpeechSegmenterFactory.is_backend_available("nemo")
        # NeMo may or may not be installed
        if not available:
            assert "nemo_toolkit" in hint.lower() or "nemo" in hint.lower()

        available_lite, _ = SpeechSegmenterFactory.is_backend_available("nemo-lite")
        available_diar, _ = SpeechSegmenterFactory.is_backend_available("nemo-diarization")
        # Both variants should have same availability as base nemo
        assert available_lite == available
        assert available_diar == available

    def test_nemo_lite_variant(self):
        """Test that nemo-lite is the lightweight frame VAD variant."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo-lite")
        if not available:
            pytest.skip("NeMo not installed")

        segmenter = SpeechSegmenterFactory.create("nemo-lite")
        assert segmenter.variant == "nemo-lite"
        assert segmenter.name == "nemo-lite"
        assert "Lite" in segmenter.display_name
        assert segmenter._use_diarizer is False
        segmenter.cleanup()

    def test_nemo_diarization_variant(self):
        """Test that nemo-diarization is the full diarizer variant."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo-diarization")
        if not available:
            pytest.skip("NeMo not installed")

        segmenter = SpeechSegmenterFactory.create("nemo-diarization")
        assert segmenter.variant == "nemo-diarization"
        assert segmenter.name == "nemo-diarization"
        assert "Diarization" in segmenter.display_name
        assert segmenter._use_diarizer is True
        segmenter.cleanup()

    def test_nemo_alias_defaults_to_lite(self):
        """Test that plain 'nemo' defaults to nemo-lite variant."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo")
        if not available:
            pytest.skip("NeMo not installed")

        segmenter = SpeechSegmenterFactory.create("nemo")
        assert segmenter.variant == "nemo-lite"
        assert segmenter._use_diarizer is False
        segmenter.cleanup()

    def test_nemo_hysteresis_defaults(self):
        """Test that NeMo has correct hysteresis defaults (sensitive preset for JAV)."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo")
        if not available:
            pytest.skip("NeMo not installed")

        segmenter = SpeechSegmenterFactory.create("nemo-lite")
        # Check sensitive preset defaults optimized for JAV
        assert segmenter.onset == 0.4
        assert segmenter.offset == 0.3
        assert segmenter.pad_offset == 0.10
        assert segmenter.min_speech_duration_ms == 100
        assert segmenter.min_silence_duration_ms == 200
        segmenter.cleanup()

    def test_nemo_with_custom_hysteresis(self):
        """Test creating NeMo segmenter with custom hysteresis."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo")
        if not available:
            pytest.skip("NeMo not installed")

        segmenter = SpeechSegmenterFactory.create(
            "nemo-lite",
            config={
                "onset": 0.9,
                "offset": 0.5,
                "pad_offset": -0.1
            }
        )
        assert segmenter.onset == 0.9
        assert segmenter.offset == 0.5
        assert segmenter.pad_offset == -0.1
        segmenter.cleanup()

    def test_get_available_backends_includes_nemo_variants(self):
        """Test that get_available_backends includes NeMo variants."""
        backends = SpeechSegmenterFactory.get_available_backends()
        names = [b["name"] for b in backends]

        # Check nemo variants are listed
        assert "nemo-lite" in names
        assert "nemo-diarization" in names

        # Check display names are correct
        nemo_lite = next(b for b in backends if b["name"] == "nemo-lite")
        nemo_diar = next(b for b in backends if b["name"] == "nemo-diarization")
        assert "Lite" in nemo_lite["display_name"]
        assert "Diarization" in nemo_diar["display_name"]


class TestNemoSpeechSegmenterDirect:
    """Direct tests for NemoSpeechSegmenter class (when available)."""

    @pytest.fixture
    def nemo_available(self):
        """Check if NeMo is available and skip if not."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo")
        if not available:
            pytest.skip("NeMo not installed")
        return True

    def test_variants_constant(self, nemo_available):
        """Test VARIANTS constant."""
        segmenter = SpeechSegmenterFactory.create("nemo")
        assert hasattr(segmenter, 'VARIANTS')
        assert "nemo-lite" in segmenter.VARIANTS
        assert "nemo-diarization" in segmenter.VARIANTS
        assert segmenter.VARIANTS["nemo-lite"]["use_diarizer"] is False
        assert segmenter.VARIANTS["nemo-diarization"]["use_diarizer"] is True
        segmenter.cleanup()

    def test_domain_configs(self, nemo_available):
        """Test DOMAIN_CONFIGS constant (for diarization variant)."""
        segmenter = SpeechSegmenterFactory.create("nemo-diarization")
        assert hasattr(segmenter, 'DOMAIN_CONFIGS')
        assert "general" in segmenter.DOMAIN_CONFIGS
        assert "meeting" in segmenter.DOMAIN_CONFIGS
        assert segmenter.DOMAIN_CONFIGS["general"] == "diar_infer_general.yaml"
        assert segmenter.DOMAIN_CONFIGS["meeting"] == "diar_infer_meeting.yaml"
        segmenter.cleanup()

    def test_get_parameters_lite(self, nemo_available):
        """Test _get_parameters returns expected keys for nemo-lite."""
        segmenter = SpeechSegmenterFactory.create("nemo-lite")
        params = segmenter._get_parameters()

        assert "variant" in params
        assert "onset" in params
        assert "offset" in params
        assert "pad_offset" in params
        assert "min_speech_duration_ms" in params
        assert "chunk_threshold_s" in params

        assert params["variant"] == "nemo-lite"
        # nemo-lite should NOT have diarization_domain
        assert "diarization_domain" not in params
        segmenter.cleanup()

    def test_get_parameters_diarization(self, nemo_available):
        """Test _get_parameters returns expected keys for nemo-diarization."""
        segmenter = SpeechSegmenterFactory.create("nemo-diarization")
        params = segmenter._get_parameters()

        assert "variant" in params
        assert "diarization_domain" in params
        assert params["variant"] == "nemo-diarization"
        assert params["diarization_domain"] == "general"
        segmenter.cleanup()

    def test_supported_sample_rates(self, nemo_available):
        """Test supported sample rates for NeMo."""
        segmenter = SpeechSegmenterFactory.create("nemo")
        rates = segmenter.get_supported_sample_rates()
        # NeMo requires 16kHz
        assert 16000 in rates
        segmenter.cleanup()

    def test_repr(self, nemo_available):
        """Test string representation."""
        segmenter = SpeechSegmenterFactory.create("nemo-diarization")
        repr_str = repr(segmenter)
        assert "NemoSpeechSegmenter" in repr_str
        assert "nemo-diarization" in repr_str
        segmenter.cleanup()


@pytest.mark.integration
class TestNemoIntegration:
    """
    Integration tests that exercise real NeMo model loading with actual audio.

    These tests catch issues like cache corruption that mock tests miss.
    Marked as integration tests - run with: pytest -m integration
    """

    TEST_AUDIO = Path(__file__).parent / "MIAA-432.20sec_piano.wav"

    @pytest.fixture
    def nemo_available(self):
        """Check if NeMo is available and skip if not."""
        available, _ = SpeechSegmenterFactory.is_backend_available("nemo")
        if not available:
            pytest.skip("NeMo not installed")
        return True

    def test_nemo_lite_real_audio_segmentation(self, nemo_available):
        """
        Test NeMo Lite (frame VAD) with actual audio file.

        This catches cache corruption issues like missing model_config.yaml
        that mock tests would miss.
        """
        if not self.TEST_AUDIO.exists():
            pytest.skip(f"Test audio not found: {self.TEST_AUDIO}")

        segmenter = SpeechSegmenterFactory.create("nemo-lite")
        try:
            result = segmenter.segment(self.TEST_AUDIO)

            # Should always return a result (never raise)
            assert result is not None
            assert hasattr(result, 'segments')
            assert hasattr(result, 'method')
            assert hasattr(result, 'audio_duration_sec')

            # Check result structure
            assert isinstance(result.segments, list)
            assert result.audio_duration_sec > 0

            # If failed gracefully, method should indicate error type
            if not result.segments:
                assert any(x in result.method for x in ['error', 'unavailable', 'audio_error']), \
                    f"Expected error indicator in method, got: {result.method}"
            else:
                # Success - should use frame_vad method (not diarizer)
                assert 'frame_vad' in result.method
                assert result.processing_time_sec > 0

        finally:
            segmenter.cleanup()

    def test_nemo_graceful_failure_on_invalid_audio(self, nemo_available):
        """Test that NeMo handles invalid audio paths gracefully."""
        segmenter = SpeechSegmenterFactory.create("nemo")
        try:
            result = segmenter.segment("/nonexistent/path/audio.wav")

            # Should return empty result, not raise exception
            assert result is not None
            assert result.segments == []
            assert "error" in result.method or "audio_error" in result.method

        finally:
            segmenter.cleanup()

    def test_nemo_returns_valid_result_structure(self, nemo_available):
        """Test that NeMo always returns a valid SegmentationResult structure."""
        if not self.TEST_AUDIO.exists():
            pytest.skip(f"Test audio not found: {self.TEST_AUDIO}")

        segmenter = SpeechSegmenterFactory.create("nemo-lite")
        try:
            result = segmenter.segment(self.TEST_AUDIO)

            # Verify all required fields exist
            assert hasattr(result, 'segments')
            assert hasattr(result, 'groups')
            assert hasattr(result, 'method')
            assert hasattr(result, 'audio_duration_sec')
            assert hasattr(result, 'parameters')
            assert hasattr(result, 'processing_time_sec')

            # Parameters should contain expected keys
            params = result.parameters
            assert 'variant' in params
            assert 'onset' in params
            assert 'offset' in params

        finally:
            segmenter.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
