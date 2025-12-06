#!/usr/bin/env python3
"""
Unit tests for Silero VAD scene detection implementation.
Tests the scene_detection.py module with Silero method.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import soundfile as sf

# Import the scene detector
from whisperjav.modules.scene_detection import DynamicSceneDetector


class TestSileroInitialization:
    """Test Silero VAD initialization"""

    def test_silero_method_parameter_accepted(self):
        """Test that method='silero' is accepted as parameter"""
        detector = DynamicSceneDetector(method="silero")
        assert detector.method == "silero"

    def test_auditok_method_is_default(self):
        """Test that auditok remains the default method"""
        detector = DynamicSceneDetector()
        assert detector.method == "auditok"

    def test_silero_vad_model_loads(self):
        """Test that Silero VAD model loads successfully"""
        detector = DynamicSceneDetector(
            method="silero",
            pass2_config={
                'threshold': 0.11,
                'min_silence_duration_ms': 1500,
                'min_speech_duration_ms': 100,
                'max_speech_duration_s': 600,
            }
        )

        # Check that Silero VAD components are initialized
        assert hasattr(detector, 'vad_model')
        assert hasattr(detector, 'get_speech_timestamps')
        assert hasattr(detector, 'silero_threshold')
        assert detector.silero_threshold == 0.08  # Optimized default for balanced detection

    def test_silero_parameters_stored(self):
        """Test that Silero parameters are correctly stored"""
        # Silero params are passed directly via kwargs, not via pass2_config
        detector = DynamicSceneDetector(
            method="silero",
            silero_threshold=0.15,
            silero_neg_threshold=0.10,
            silero_min_silence_ms=2000,
            silero_min_speech_ms=200,
            silero_max_speech_s=300,
            silero_min_silence_at_max=600,
            silero_speech_pad_ms=150,
        )

        assert detector.silero_threshold == 0.15
        assert detector.silero_neg_threshold == 0.10
        assert detector.silero_min_silence_ms == 2000
        assert detector.silero_min_speech_ms == 200
        assert detector.silero_max_speech_s == 300
        assert detector.silero_min_silence_at_max == 600
        assert detector.silero_speech_pad_ms == 150


class TestSileroDetectionOutputFormat:
    """Test Silero Pass 2 output format compatibility"""

    @pytest.fixture
    def test_audio(self):
        """Create test audio (1 second of sine wave)"""
        sample_rate = 16000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        return audio, sample_rate

    def test_detect_pass2_silero_returns_list(self, test_audio):
        """Test that _detect_pass2_silero returns a list"""
        audio, sr = test_audio

        detector = DynamicSceneDetector(
            method="silero",
            pass2_config={'threshold': 0.11}
        )

        regions = detector._detect_pass2_silero(audio, sr, 0.0)

        assert isinstance(regions, list)

    def test_silero_region_has_start_end_attributes(self, test_audio):
        """Test that returned regions have .start and .end attributes"""
        audio, sr = test_audio

        detector = DynamicSceneDetector(
            method="silero",
            pass2_config={'threshold': 0.11}
        )

        regions = detector._detect_pass2_silero(audio, sr, 0.0)

        # If regions found, check format
        if regions:
            region = regions[0]
            assert hasattr(region, 'start')
            assert hasattr(region, 'end')
            assert isinstance(region.start, (int, float))
            assert isinstance(region.end, (int, float))

    def test_silero_handles_resampling(self, test_audio):
        """Test that _detect_pass2_silero handles non-16kHz audio"""
        # Create 48kHz audio
        sample_rate = 48000
        duration = 1.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        detector = DynamicSceneDetector(
            method="silero",
            pass2_config={'threshold': 0.11}
        )

        # Should not raise error
        regions = detector._detect_pass2_silero(audio, sample_rate, 0.0)

        assert isinstance(regions, list)


class TestSileroIntegration:
    """Test full integration with detect_scenes()"""

    @pytest.fixture
    def temp_audio_file(self):
        """Create temporary audio file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = Path(f.name)

        # Create 5 seconds of test audio
        sample_rate = 16000
        duration = 5.0
        frequency = 440.0

        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Add some silence in the middle to create scene boundaries
        mid_point = len(audio) // 2
        silence_samples = int(sample_rate * 2.0)  # 2 seconds silence
        audio[mid_point:mid_point + silence_samples] = 0.0

        sf.write(str(temp_path), audio, sample_rate)

        yield temp_path

        # Cleanup
        temp_path.unlink(missing_ok=True)

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_detect_scenes_with_silero_method(self, temp_audio_file, temp_output_dir):
        """Test detect_scenes() with method='silero'"""
        detector = DynamicSceneDetector(
            method="silero",
            max_duration=29.0,
            min_duration=0.2,
            pass2_config={
                'threshold': 0.11,
                'min_silence_duration_ms': 1500,
            }
        )

        scenes = detector.detect_scenes(
            audio_path=temp_audio_file,
            output_dir=temp_output_dir,
            media_basename="test"
        )

        # Should return list of tuples
        assert isinstance(scenes, list)

        # Each tuple: (scene_path, start_time, end_time, duration)
        if scenes:
            scene = scenes[0]
            assert len(scene) == 4
            assert isinstance(scene[0], Path)
            assert isinstance(scene[1], (int, float))
            assert isinstance(scene[2], (int, float))
            assert isinstance(scene[3], (int, float))

    def test_silero_vs_auditok_both_work(self, temp_audio_file, temp_output_dir):
        """Test that both methods work on the same file"""
        # Test with Auditok
        detector_auditok = DynamicSceneDetector(method="auditok")
        scenes_auditok = detector_auditok.detect_scenes(
            audio_path=temp_audio_file,
            output_dir=temp_output_dir / "auditok",
            media_basename="test_auditok"
        )

        # Test with Silero
        detector_silero = DynamicSceneDetector(
            method="silero",
            pass2_config={'threshold': 0.11}
        )
        scenes_silero = detector_silero.detect_scenes(
            audio_path=temp_audio_file,
            output_dir=temp_output_dir / "silero",
            media_basename="test_silero"
        )

        # Both should return scenes
        assert len(scenes_auditok) > 0
        assert len(scenes_silero) > 0


class TestSileroErrorHandling:
    """Test error handling for Silero method"""

    def test_invalid_method_falls_back(self):
        """Test that invalid method doesn't crash (handled by config)"""
        # This should default to auditok in the actual workflow
        # but at the module level, it will just accept any string
        detector = DynamicSceneDetector(method="invalid_method")
        assert detector.method == "invalid_method"

    def test_silero_with_missing_pass2_config(self):
        """Test Silero initialization with minimal config"""
        # Should use defaults
        detector = DynamicSceneDetector(
            method="silero",
            pass2_config={}
        )

        # Check defaults are applied
        assert detector.silero_threshold == 0.08  # Optimized default for balanced detection
        assert detector.silero_min_silence_ms == 1500


class TestBackwardCompatibility:
    """Test that existing functionality is not broken"""

    def test_no_method_parameter_uses_auditok(self):
        """Test that omitting method parameter defaults to auditok"""
        detector = DynamicSceneDetector()
        assert detector.method == "auditok"

    def test_existing_auditok_parameters_still_work(self):
        """Test that existing Auditok parameters still work"""
        detector = DynamicSceneDetector(
            max_duration=30.0,
            min_duration=0.5,
            pass1_energy_threshold=35,
            pass2_energy_threshold=40
        )

        assert detector.max_duration == 30.0
        assert detector.min_duration == 0.5
        assert detector.pass1_energy_threshold == 35
        assert detector.pass2_energy_threshold == 40


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
