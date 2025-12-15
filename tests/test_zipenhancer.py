#!/usr/bin/env python
"""
ZipEnhancer Backend Unit Tests

Tests the ZipEnhancer speech enhancement backend.

Usage:
    # Run with pytest
    pytest tests/test_zipenhancer.py -v

    # Run specific test
    pytest tests/test_zipenhancer.py::test_zipenhancer_factory_registration -v

    # Skip tests requiring ModelScope (for CI without GPU)
    pytest tests/test_zipenhancer.py -v -k "not integration"

Author: WhisperJAV Team
Version: 1.7.3
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure whisperjav is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestZipEnhancerFactory:
    """Tests for ZipEnhancer factory registration."""

    def test_zipenhancer_factory_registration(self):
        """Test that zipenhancer is registered in the factory."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        backends = SpeechEnhancerFactory.list_backends()
        assert "zipenhancer" in backends, "zipenhancer should be registered in factory"

    def test_zipenhancer_availability_check(self):
        """Test availability check returns correct format."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        available, hint = SpeechEnhancerFactory.is_backend_available("zipenhancer")

        # Should return a boolean and string
        assert isinstance(available, bool)
        assert isinstance(hint, str)

        # If not available, should have install hint
        if not available:
            assert "modelscope" in hint.lower()

    def test_zipenhancer_in_available_backends(self):
        """Test that zipenhancer appears in get_available_backends."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        backends = SpeechEnhancerFactory.get_available_backends()

        zipenhancer_info = None
        for b in backends:
            if b["name"] == "zipenhancer":
                zipenhancer_info = b
                break

        assert zipenhancer_info is not None, "zipenhancer should be in available backends"
        assert "display_name" in zipenhancer_info
        assert "available" in zipenhancer_info
        assert "description" in zipenhancer_info

    def test_zipenhancer_default_model(self):
        """Test that zipenhancer has correct default model."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        backends = SpeechEnhancerFactory.get_available_backends()
        zipenhancer_info = next((b for b in backends if b["name"] == "zipenhancer"), None)

        assert zipenhancer_info is not None
        assert zipenhancer_info.get("default_model") == "torch"


class TestZipEnhancerBackend:
    """Tests for ZipEnhancer backend implementation."""

    def test_backend_import(self):
        """Test that backend module can be imported."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
            ZIPENHANCER_SAMPLE_RATE,
        )

        assert ZIPENHANCER_SAMPLE_RATE == 16000, "ZipEnhancer should use 16kHz"

    def test_backend_instantiation_torch(self):
        """Test backend can be instantiated with torch model."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        # Should be able to create instance without loading model
        backend = ZipEnhancerBackend(model="torch")

        assert backend.name == "zipenhancer"
        assert backend.model_variant == "torch"
        assert "16kHz" in backend.display_name
        assert "ONNX" not in backend.display_name
        assert backend.get_preferred_sample_rate() == 16000
        assert backend.get_output_sample_rate() == 16000

    def test_backend_instantiation_onnx(self):
        """Test backend can be instantiated with onnx model."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend(model="onnx")

        assert backend.name == "zipenhancer"
        assert backend.model_variant == "onnx"
        assert "ONNX" in backend.display_name
        assert backend.get_preferred_sample_rate() == 16000
        assert backend.get_output_sample_rate() == 16000

    def test_backend_default_model(self):
        """Test backend defaults to torch model."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend()
        assert backend.model_variant == "torch"

    def test_backend_invalid_model_fallback(self):
        """Test backend falls back to torch for invalid model."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend(model="invalid_model")
        assert backend.model_variant == "torch"

    def test_backend_supported_models(self):
        """Test backend reports supported models."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend()
        models = backend.get_supported_models()

        assert "torch" in models
        assert "onnx" in models

    def test_backend_config_options(self):
        """Test backend accepts configuration options."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        # Test with custom options
        backend = ZipEnhancerBackend(
            model="onnx",
            device="cpu",
            chunk_duration=5.0,
        )

        assert backend.model_variant == "onnx"
        assert backend.device == "cpu"
        assert backend.chunk_duration == 5.0

    def test_backend_cleanup_without_init(self):
        """Test cleanup works even if model was never loaded."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend()

        # Should not raise even if never initialized
        backend.cleanup()


class TestZipEnhancerChunking:
    """Tests for ZipEnhancer chunked processing logic."""

    def test_crossfade_stitch_single_chunk(self):
        """Test crossfade with single chunk returns it unchanged."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend()

        # Single chunk should be returned as-is
        chunk = np.ones(16000, dtype=np.float32)
        result = backend._crossfade_stitch([chunk], [0], 800, 16000)

        assert len(result) == 16000
        np.testing.assert_array_equal(result, chunk)

    def test_crossfade_stitch_two_chunks(self):
        """Test crossfade with two overlapping chunks."""
        from whisperjav.modules.speech_enhancement.backends.zipenhancer import (
            ZipEnhancerBackend,
        )

        backend = ZipEnhancerBackend()

        # Create two chunks with overlap
        chunk1 = np.ones(10000, dtype=np.float32) * 1.0
        chunk2 = np.ones(10000, dtype=np.float32) * 2.0

        # Overlap of 2000 samples, total length 18000
        overlap = 2000
        total_len = 18000
        positions = [0, 8000]

        result = backend._crossfade_stitch([chunk1, chunk2], positions, overlap, total_len)

        assert len(result) == total_len, f"Expected {total_len}, got {len(result)}"

        # Non-overlap regions should have original values
        # First 6000 samples (before overlap) should be ~1.0
        assert np.mean(result[:6000]) > 0.9, "First section should be ~1.0"

        # Last 8000 samples (after overlap starts) should trend toward 2.0
        assert np.mean(result[10000:]) > 1.5, "Last section should be closer to 2.0"


@pytest.mark.integration
class TestZipEnhancerIntegration:
    """Integration tests requiring ModelScope (skip in CI)."""

    @pytest.fixture
    def test_audio(self):
        """Generate synthetic test audio."""
        duration = 2.0  # 2 seconds
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # Sine wave with some noise
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        audio += 0.1 * np.random.randn(len(audio)).astype(np.float32)  # Noise

        return audio, sr

    @pytest.mark.skipif(
        True,  # Skip by default - requires ModelScope installation
        reason="Requires ModelScope installation and model download"
    )
    def test_enhancement_torch_mode(self, test_audio):
        """Test full enhancement pipeline with torch mode."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        audio, sr = test_audio

        # Create through factory with torch model
        enhancer = SpeechEnhancerFactory.create("zipenhancer", model="torch")

        try:
            result = enhancer.enhance(audio, sr)

            assert result.success, f"Enhancement failed: {result.error_message}"
            assert result.sample_rate == 16000
            assert len(result.audio) == len(audio)
            assert result.method == "zipenhancer"
            assert result.parameters.get("model") == "torch"
            assert result.processing_time_sec > 0

        finally:
            enhancer.cleanup()

    @pytest.mark.skipif(
        True,  # Skip by default - requires ModelScope installation
        reason="Requires ModelScope installation and model download"
    )
    def test_enhancement_onnx_mode(self, test_audio):
        """Test full enhancement pipeline with onnx mode."""
        from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory

        audio, sr = test_audio

        # Create through factory with onnx model
        enhancer = SpeechEnhancerFactory.create("zipenhancer", model="onnx")

        try:
            result = enhancer.enhance(audio, sr)

            assert result.success, f"Enhancement failed: {result.error_message}"
            assert result.sample_rate == 16000
            assert len(result.audio) == len(audio)
            assert result.method == "zipenhancer"
            assert result.parameters.get("model") == "onnx"
            assert result.processing_time_sec > 0

        finally:
            enhancer.cleanup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
