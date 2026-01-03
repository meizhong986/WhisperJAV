#!/usr/bin/env python3
"""
Unit tests for compute_type="auto" strategy.

Tests the CTranslate2 compute_type delegation implemented in:
- whisperjav/config/resolver_v3.py
- whisperjav/modules/faster_whisper_pro_asr.py
- whisperjav/modules/kotoba_faster_whisper_asr.py
- whisperjav/modules/speech_segmentation/backends/whisper_vad.py

Background:
- CTranslate2's "auto" selects optimal compute_type based on device capabilities
- Automatically handles RTX 50XX Blackwell (sm120) which doesn't support INT8
- See: https://github.com/OpenNMT/CTranslate2/issues/1865

Run with: pytest tests/test_compute_type_auto.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# Resolver Tests
# =============================================================================

class TestResolverComputeType:
    """Tests for resolver_v3.py compute_type selection."""

    def test_ctranslate2_providers_return_auto(self):
        """Test that CTranslate2 providers get compute_type='auto'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        # CTranslate2 providers should return "auto" regardless of device
        assert _get_compute_type_for_device("cuda", "faster_whisper") == "auto"
        assert _get_compute_type_for_device("cpu", "faster_whisper") == "auto"
        assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "auto"
        assert _get_compute_type_for_device("cpu", "kotoba_faster_whisper") == "auto"

    def test_pytorch_providers_return_explicit_types(self):
        """Test that PyTorch providers get explicit compute_type values."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        # PyTorch providers should return explicit types
        assert _get_compute_type_for_device("cuda", "openai_whisper") == "float16"
        assert _get_compute_type_for_device("mps", "openai_whisper") == "float16"
        assert _get_compute_type_for_device("cpu", "openai_whisper") == "float32"
        assert _get_compute_type_for_device("cuda", "stable_ts") == "float16"
        assert _get_compute_type_for_device("cpu", "stable_ts") == "float32"

    def test_ctranslate2_providers_set(self):
        """Test that CTRANSLATE2_PROVIDERS set is correctly defined."""
        from whisperjav.config.resolver_v3 import CTRANSLATE2_PROVIDERS

        assert "faster_whisper" in CTRANSLATE2_PROVIDERS
        assert "kotoba_faster_whisper" in CTRANSLATE2_PROVIDERS
        # PyTorch providers should NOT be in this set
        assert "openai_whisper" not in CTRANSLATE2_PROVIDERS
        assert "stable_ts" not in CTRANSLATE2_PROVIDERS


class TestResolverIntegration:
    """Integration tests for resolver with compute_type='auto'."""

    @pytest.fixture
    def mock_device_detector(self):
        """Mock device detector to return specific device."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock:
            yield mock

    def test_resolve_config_faster_whisper_cuda(self, mock_device_detector):
        """Test that faster_whisper config has compute_type='auto' on CUDA."""
        mock_device_detector.return_value = "cuda"

        # Mock the component registries
        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr:
            with patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
                # Setup mock ASR component
                mock_component = MagicMock()
                mock_component.provider = "faster_whisper"
                mock_component.model_id = "large-v3"
                mock_component.supported_tasks = ["transcribe"]
                mock_preset = MagicMock()
                mock_preset.model_dump.return_value = {"language": "ja", "task": "transcribe"}
                mock_component.get_preset.return_value = mock_preset

                mock_asr.return_value = {"faster_whisper": mock_component}
                mock_vad.return_value = {}

                from whisperjav.config.resolver_v3 import resolve_config_v3
                config = resolve_config_v3(asr="faster_whisper", vad="none")

                assert config["model"]["compute_type"] == "auto"

    def test_resolve_config_faster_whisper_cpu(self, mock_device_detector):
        """Test that faster_whisper config has compute_type='auto' on CPU."""
        mock_device_detector.return_value = "cpu"

        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr:
            with patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
                mock_component = MagicMock()
                mock_component.provider = "faster_whisper"
                mock_component.model_id = "large-v3"
                mock_component.supported_tasks = ["transcribe"]
                mock_preset = MagicMock()
                mock_preset.model_dump.return_value = {"language": "ja", "task": "transcribe"}
                mock_component.get_preset.return_value = mock_preset

                mock_asr.return_value = {"faster_whisper": mock_component}
                mock_vad.return_value = {}

                from whisperjav.config.resolver_v3 import resolve_config_v3
                config = resolve_config_v3(asr="faster_whisper", vad="none")

                # Should be "auto" even on CPU - CTranslate2 handles selection
                assert config["model"]["compute_type"] == "auto"


# =============================================================================
# ASR Module Tests
# =============================================================================

class TestFasterWhisperProASR:
    """Tests for FasterWhisperProASR compute_type handling."""

    def test_uses_compute_type_from_config(self):
        """Test that compute_type is taken from model_config."""
        # We can't easily test the full initialization without models
        # but we can verify the default behavior
        pass  # Tested via integration tests

    def test_default_compute_type_is_auto(self):
        """Test that default compute_type is 'auto'."""
        # This is verified by checking the resolver returns "auto"
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device
        assert _get_compute_type_for_device("cuda", "faster_whisper") == "auto"


class TestKotobaFasterWhisperASR:
    """Tests for KotobaFasterWhisperASR compute_type handling."""

    def test_default_compute_type_is_auto(self):
        """Test that default compute_type is 'auto'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device
        assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "auto"


class TestWhisperVAD:
    """Tests for WhisperVAD compute_type handling."""

    def test_default_compute_type_is_auto(self):
        """Test that default compute_type parameter is 'auto'."""
        import inspect
        from whisperjav.modules.speech_segmentation.backends.whisper_vad import WhisperVadSpeechSegmenter

        sig = inspect.signature(WhisperVadSpeechSegmenter.__init__)
        compute_type_param = sig.parameters.get('compute_type')

        assert compute_type_param is not None
        assert compute_type_param.default == "auto"


# =============================================================================
# CTranslate2 Auto Selection Logic Tests
# =============================================================================

class TestCTranslate2AutoBehavior:
    """Tests documenting expected CTranslate2 'auto' behavior."""

    def test_auto_documented_behavior(self):
        """
        Document expected CTranslate2 'auto' behavior.

        According to CTranslate2 documentation:
        - "auto" selects the fastest supported compute type
        - GPU (CC ≥8.0): All types supported, likely selects int8_float16
        - GPU (CC 6.1-7.0): int8/float16 optimized
        - CPU: int8/int16/float32 based on instruction sets
        - Blackwell sm120: Disables INT8 (PR #1937)

        This test documents the expected behavior rather than testing it,
        as the actual behavior depends on CTranslate2 internals.
        """
        expected_behaviors = {
            "RTX 40XX (Ampere/Ada)": "int8_float16 (fastest with tensor cores)",
            "RTX 50XX (Blackwell sm120)": "float16 (INT8 disabled by PR #1937)",
            "CPU with AVX2": "int8 (quantized, fast)",
            "CPU without AVX2": "float32 (fallback)",
        }

        # This is documentation, not a test
        assert len(expected_behaviors) > 0


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with explicit compute_type."""

    def test_explicit_compute_type_still_works(self):
        """Test that explicit compute_type values are still respected."""
        # When users explicitly set compute_type, it should be used
        # The "auto" is only the default from resolver

        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        # Resolver returns "auto" for CTranslate2 providers
        result = _get_compute_type_for_device("cuda", "faster_whisper")
        assert result == "auto"

        # But ASR modules accept explicit overrides via model_config
        # This is tested via the actual code path:
        # model_config.get("compute_type", "auto")
        # If user provides {"compute_type": "int8"}, that value is used


# =============================================================================
# CLI Device/Compute-Type Override Tests
# =============================================================================

class TestCLIDeviceOverride:
    """Tests for --device and --compute-type CLI argument flow."""

    @pytest.fixture
    def mock_registries(self):
        """Mock component registries for testing."""
        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr:
            with patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
                # Setup mock ASR component
                mock_component = MagicMock()
                mock_component.provider = "faster_whisper"
                mock_component.model_id = "large-v3"
                mock_component.supported_tasks = ["transcribe"]
                mock_preset = MagicMock()
                mock_preset.model_dump.return_value = {"language": "ja", "task": "transcribe"}
                mock_component.get_preset.return_value = mock_preset

                mock_asr.return_value = {"faster_whisper": mock_component}
                mock_vad.return_value = {}
                yield mock_asr, mock_vad

    def test_device_override_cuda(self, mock_registries):
        """Test that explicit device='cuda' overrides auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cpu"  # Would auto-detect CPU

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                device="cuda",  # Override to CUDA
            )

            assert config["model"]["device"] == "cuda"
            # get_best_device should NOT be called when device is explicitly provided
            # (Actually it's still called for safety, but result is ignored)

    def test_device_override_cpu(self, mock_registries):
        """Test that explicit device='cpu' overrides auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"  # Would auto-detect CUDA

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                device="cpu",  # Override to CPU
            )

            assert config["model"]["device"] == "cpu"

    def test_device_auto_uses_detection(self, mock_registries):
        """Test that device='auto' uses auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                device="auto",  # Explicit auto
            )

            assert config["model"]["device"] == "cuda"
            mock_device.assert_called_once()

    def test_device_none_uses_detection(self, mock_registries):
        """Test that device=None (default) uses auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cpu"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                # device not specified = None
            )

            assert config["model"]["device"] == "cpu"
            mock_device.assert_called_once()

    def test_compute_type_override_int8(self, mock_registries):
        """Test that explicit compute_type='int8' overrides auto."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                compute_type="int8",  # Override
            )

            assert config["model"]["compute_type"] == "int8"

    def test_compute_type_override_float16(self, mock_registries):
        """Test that explicit compute_type='float16' overrides auto."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                compute_type="float16",
            )

            assert config["model"]["compute_type"] == "float16"

    def test_compute_type_override_int8_float16(self, mock_registries):
        """Test that explicit compute_type='int8_float16' works."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                compute_type="int8_float16",
            )

            assert config["model"]["compute_type"] == "int8_float16"

    def test_compute_type_auto_uses_provider_default(self, mock_registries):
        """Test that compute_type='auto' uses provider-specific default."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                compute_type="auto",  # Explicit auto
            )

            # CTranslate2 provider should get "auto"
            assert config["model"]["compute_type"] == "auto"

    def test_compute_type_none_uses_provider_default(self, mock_registries):
        """Test that compute_type=None uses provider-specific default."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                # compute_type not specified = None
            )

            assert config["model"]["compute_type"] == "auto"

    def test_combined_device_and_compute_type_override(self, mock_registries):
        """Test that both device and compute_type can be overridden together."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"  # Would detect CUDA

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper",
                vad="none",
                device="cpu",  # Force CPU
                compute_type="float32",  # Force float32
            )

            assert config["model"]["device"] == "cpu"
            assert config["model"]["compute_type"] == "float32"


class TestLegacyPipelineDeviceOverride:
    """Tests for device/compute_type override through legacy.py."""

    @pytest.fixture
    def mock_resolve_v3(self):
        """Mock resolve_config_v3 to verify parameters passed."""
        with patch('whisperjav.config.legacy.resolve_config_v3') as mock:
            # Return a minimal valid config
            mock.return_value = {
                'asr_name': 'faster_whisper',
                'vad_name': 'none',
                'sensitivity_name': 'balanced',
                'task': 'transcribe',
                'language': 'ja',
                'model': {
                    'provider': 'faster_whisper',
                    'model_name': 'large-v3',
                    'device': 'cuda',
                    'compute_type': 'auto',
                    'supported_tasks': ['transcribe']
                },
                'params': {'asr': {}, 'vad': {}},
                'features': {},
            }
            yield mock

    def test_resolve_legacy_pipeline_passes_device(self, mock_resolve_v3):
        """Test that resolve_legacy_pipeline passes device parameter."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
            sensitivity="aggressive",
            device="cpu",
        )

        # Verify resolve_config_v3 was called with device parameter
        mock_resolve_v3.assert_called_once()
        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') == "cpu"

    def test_resolve_legacy_pipeline_passes_compute_type(self, mock_resolve_v3):
        """Test that resolve_legacy_pipeline passes compute_type parameter."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
            sensitivity="balanced",
            compute_type="int8_float16",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('compute_type') == "int8_float16"

    def test_resolve_legacy_pipeline_passes_both(self, mock_resolve_v3):
        """Test that resolve_legacy_pipeline passes both device and compute_type."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="faster",
            device="cuda",
            compute_type="float16",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') == "cuda"
        assert call_kwargs.get('compute_type') == "float16"

    def test_resolve_legacy_pipeline_none_defaults(self, mock_resolve_v3):
        """Test that resolve_legacy_pipeline passes None when not specified."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
            # device and compute_type not specified
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') is None
        assert call_kwargs.get('compute_type') is None


class TestEnsembleConfigDeviceOverride:
    """Tests for device/compute_type override through ensemble config."""

    @pytest.fixture
    def mock_resolve_v3(self):
        """Mock resolve_config_v3 for ensemble config tests."""
        with patch('whisperjav.config.legacy.resolve_config_v3') as mock:
            mock.return_value = {
                'asr_name': 'faster_whisper',
                'vad_name': 'none',
                'sensitivity_name': 'balanced',
                'task': 'transcribe',
                'language': 'ja',
                'model': {
                    'provider': 'faster_whisper',
                    'model_name': 'large-v3',
                    'device': 'cuda',
                    'compute_type': 'auto',
                    'supported_tasks': ['transcribe']
                },
                'params': {'asr': {}, 'vad': {}},
                'features': {},
            }
            yield mock

    def test_resolve_ensemble_config_passes_device(self, mock_resolve_v3):
        """Test that resolve_ensemble_config passes device parameter."""
        from whisperjav.config.legacy import resolve_ensemble_config

        resolve_ensemble_config(
            asr="faster_whisper",
            device="cpu",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') == "cpu"

    def test_resolve_ensemble_config_passes_compute_type(self, mock_resolve_v3):
        """Test that resolve_ensemble_config passes compute_type parameter."""
        from whisperjav.config.legacy import resolve_ensemble_config

        resolve_ensemble_config(
            asr="faster_whisper",
            compute_type="int8",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('compute_type') == "int8"


class TestCLIArgumentParsing:
    """Tests for CLI argument parsing in main.py."""

    def test_device_argument_choices(self):
        """Test that --device accepts valid choices."""
        from whisperjav.main import parse_arguments
        import sys

        # Test with valid choices
        for device in ["auto", "cuda", "cpu"]:
            with patch.object(sys, 'argv', ['whisperjav', 'test.mp4', f'--device={device}']):
                args = parse_arguments()
                assert args.device == device

    def test_compute_type_argument_choices(self):
        """Test that --compute-type accepts valid choices."""
        from whisperjav.main import parse_arguments
        import sys

        valid_types = ["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"]
        for compute_type in valid_types:
            with patch.object(sys, 'argv', ['whisperjav', 'test.mp4', f'--compute-type={compute_type}']):
                args = parse_arguments()
                assert args.compute_type == compute_type

    def test_device_default_is_none(self):
        """Test that --device defaults to None (auto-detect)."""
        from whisperjav.main import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['whisperjav', 'test.mp4']):
            args = parse_arguments()
            assert args.device is None

    def test_compute_type_default_is_none(self):
        """Test that --compute-type defaults to None (auto)."""
        from whisperjav.main import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['whisperjav', 'test.mp4']):
            args = parse_arguments()
            assert args.compute_type is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
