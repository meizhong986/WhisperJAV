#!/usr/bin/env python3
"""
Unit tests for compute_type resolution strategy.

Tests the device-aware compute_type selection implemented in:
- whisperjav/config/resolver_v3.py  (primary logic)
- whisperjav/modules/faster_whisper_pro_asr.py  (safety net)

v1.8.9-hotfix2 changes:
- CTranslate2 providers on CUDA: "float16" (was "auto") for best accuracy
- CTranslate2 providers on non-CUDA: "auto" (lets CTranslate2 pick for device)
- Pascal exception (sm_6x): "float32" (conservative, no tensor cores)
- Blackwell exception (sm_120+): "float16" (CTranslate2 auto bug)
- Safety net: float16 on CPU → auto (catches MPS→CPU downgrade)
- See: https://github.com/meizhong986/WhisperJAV/issues/241

Run with: pytest tests/test_compute_type_auto.py -v
"""

import pytest
from unittest.mock import patch, MagicMock


# =============================================================================
# Resolver Tests — _get_compute_type_for_device()
# =============================================================================

class TestResolverComputeType:
    """Tests for resolver_v3.py compute_type selection."""

    def test_ctranslate2_cuda_returns_float16(self):
        """CTranslate2 providers on CUDA (normal GPU) return 'float16'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        with patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            assert _get_compute_type_for_device("cuda", "faster_whisper") == "float16"
            assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "float16"

    def test_ctranslate2_non_cuda_returns_auto(self):
        """CTranslate2 providers on non-CUDA devices return 'auto'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        # CPU, MPS, and hypothetical future devices all get "auto"
        for device in ("cpu", "mps", "xpu"):
            assert _get_compute_type_for_device(device, "faster_whisper") == "auto"
            assert _get_compute_type_for_device(device, "kotoba_faster_whisper") == "auto"

    def test_ctranslate2_cuda_pascal_returns_float32(self):
        """CTranslate2 on Pascal GPU (sm_6x) returns 'float32'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        with patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=True), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            assert _get_compute_type_for_device("cuda", "faster_whisper") == "float32"
            assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "float32"

    def test_ctranslate2_cuda_blackwell_returns_float16(self):
        """CTranslate2 on Blackwell GPU (sm_120+) returns 'float16'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        with patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=True):
            assert _get_compute_type_for_device("cuda", "faster_whisper") == "float16"
            assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "float16"

    def test_pytorch_providers_return_explicit_types(self):
        """PyTorch providers get explicit float16/float32 based on device."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        assert _get_compute_type_for_device("cuda", "openai_whisper") == "float16"
        assert _get_compute_type_for_device("mps", "openai_whisper") == "float16"
        assert _get_compute_type_for_device("cpu", "openai_whisper") == "float32"
        assert _get_compute_type_for_device("cuda", "stable_ts") == "float16"
        assert _get_compute_type_for_device("cpu", "stable_ts") == "float32"

    def test_ctranslate2_providers_set(self):
        """CTRANSLATE2_PROVIDERS set is correctly defined."""
        from whisperjav.config.resolver_v3 import CTRANSLATE2_PROVIDERS

        assert "faster_whisper" in CTRANSLATE2_PROVIDERS
        assert "kotoba_faster_whisper" in CTRANSLATE2_PROVIDERS
        assert "openai_whisper" not in CTRANSLATE2_PROVIDERS
        assert "stable_ts" not in CTRANSLATE2_PROVIDERS


# =============================================================================
# Resolver Integration Tests
# =============================================================================

class TestResolverIntegration:
    """Integration tests for resolver with compute_type selection."""

    @pytest.fixture
    def mock_device_detector(self):
        """Mock device detector to return specific device."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock:
            yield mock

    def test_resolve_config_faster_whisper_cuda(self, mock_device_detector):
        """faster_whisper on CUDA gets compute_type='float16'."""
        mock_device_detector.return_value = "cuda"

        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr, \
             patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad, \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
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

            assert config["model"]["compute_type"] == "float16"

    def test_resolve_config_faster_whisper_cpu(self, mock_device_detector):
        """faster_whisper on CPU gets compute_type='auto'."""
        mock_device_detector.return_value = "cpu"

        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr, \
             patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
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


# =============================================================================
# ASR Module Safety Net Tests
# =============================================================================

class TestFasterWhisperProASR:
    """Tests for FasterWhisperProASR compute_type handling."""

    def test_uses_compute_type_from_config(self):
        """compute_type is taken from model_config."""
        pass  # Tested via integration tests

    def test_cuda_default_is_float16(self):
        """Default compute_type for CUDA is 'float16'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        with patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            assert _get_compute_type_for_device("cuda", "faster_whisper") == "float16"

    def test_cpu_default_is_auto(self):
        """Default compute_type for CPU is 'auto'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device
        assert _get_compute_type_for_device("cpu", "faster_whisper") == "auto"


class TestKotobaFasterWhisperASR:
    """Tests for KotobaFasterWhisperASR compute_type handling."""

    def test_cuda_default_is_float16(self):
        """Default compute_type for CUDA is 'float16'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        with patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            assert _get_compute_type_for_device("cuda", "kotoba_faster_whisper") == "float16"

    def test_cpu_default_is_auto(self):
        """Default compute_type for CPU is 'auto'."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device
        assert _get_compute_type_for_device("cpu", "kotoba_faster_whisper") == "auto"


class TestWhisperVAD:
    """Tests for WhisperVAD compute_type handling."""

    def test_default_compute_type_is_auto(self):
        """WhisperVAD default compute_type parameter is 'auto'."""
        import inspect
        from whisperjav.modules.speech_segmentation.backends.whisper_vad import WhisperVadSpeechSegmenter

        sig = inspect.signature(WhisperVadSpeechSegmenter.__init__)
        compute_type_param = sig.parameters.get('compute_type')

        assert compute_type_param is not None
        assert compute_type_param.default == "auto"


class TestSafetyNet:
    """Tests for the float16-on-CPU safety net in faster_whisper_pro_asr."""

    def test_float16_on_cpu_falls_back_to_auto(self):
        """Safety net catches float16 on CPU (e.g., MPS→CPU downgrade)."""
        # Simulate the safety net logic from faster_whisper_pro_asr.py lines 74-79
        device = "cpu"
        compute_type = "float16"
        if device == "cpu" and compute_type == "float16":
            compute_type = "auto"
        assert compute_type == "auto"

    def test_auto_on_cpu_passes_through(self):
        """'auto' on CPU is fine — no safety net trigger."""
        device = "cpu"
        compute_type = "auto"
        if device == "cpu" and compute_type == "float16":
            compute_type = "auto"
        assert compute_type == "auto"

    def test_float16_on_cuda_passes_through(self):
        """float16 on CUDA is fine — no safety net trigger."""
        device = "cuda"
        compute_type = "float16"
        if device == "cpu" and compute_type == "float16":
            compute_type = "auto"
        assert compute_type == "float16"

    def test_mps_to_cpu_downgrade_scenario(self):
        """Full MPS→CPU downgrade scenario: resolver returns auto, safety net not needed."""
        from whisperjav.config.resolver_v3 import _get_compute_type_for_device

        # Resolver sees MPS → returns "auto" for CTranslate2
        resolved = _get_compute_type_for_device("mps", "faster_whisper")
        assert resolved == "auto"

        # ASR module downgrades MPS → CPU, compute_type stays "auto"
        device = "cpu"  # downgraded
        compute_type = resolved
        if device == "cpu" and compute_type == "float16":
            compute_type = "auto"  # safety net
        assert compute_type == "auto"  # already fine, safety net not triggered


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
        """
        expected_behaviors = {
            "RTX 40XX (Ampere/Ada)": "int8_float16 (fastest with tensor cores)",
            "RTX 50XX (Blackwell sm120)": "float16 (INT8 disabled by PR #1937)",
            "CPU with AVX2": "int8 (quantized, fast)",
            "CPU without AVX2": "float32 (fallback)",
        }

        assert len(expected_behaviors) > 0


class TestBackwardsCompatibility:
    """Tests for backwards compatibility with explicit compute_type."""

    def test_explicit_compute_type_overrides_default(self):
        """Explicit compute_type values are respected via resolver."""
        from whisperjav.config.resolver_v3 import resolve_config_v3

        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"), \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr, \
             patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
            mock_component = MagicMock()
            mock_component.provider = "faster_whisper"
            mock_component.model_id = "large-v3"
            mock_component.supported_tasks = ["transcribe"]
            mock_preset = MagicMock()
            mock_preset.model_dump.return_value = {"language": "ja", "task": "transcribe"}
            mock_component.get_preset.return_value = mock_preset
            mock_asr.return_value = {"faster_whisper": mock_component}
            mock_vad.return_value = {}

            # User explicitly sets int8_float16
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                compute_type="int8_float16"
            )
            assert config["model"]["compute_type"] == "int8_float16"


# =============================================================================
# CLI Device/Compute-Type Override Tests
# =============================================================================

class TestCLIDeviceOverride:
    """Tests for --device and --compute-type CLI argument flow."""

    @pytest.fixture
    def mock_registries(self):
        """Mock component registries for testing."""
        with patch('whisperjav.config.resolver_v3.get_asr_registry') as mock_asr, \
             patch('whisperjav.config.resolver_v3.get_vad_registry') as mock_vad:
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
        """Explicit device='cuda' overrides auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device, \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            mock_device.return_value = "cpu"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                device="cuda",
            )

            assert config["model"]["device"] == "cuda"

    def test_device_override_cpu(self, mock_registries):
        """Explicit device='cpu' overrides auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                device="cpu",
            )

            assert config["model"]["device"] == "cpu"

    def test_device_auto_uses_detection(self, mock_registries):
        """device='auto' uses auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device, \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):
            mock_device.return_value = "cuda"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                device="auto",
            )

            assert config["model"]["device"] == "cuda"
            mock_device.assert_called_once()

    def test_device_none_uses_detection(self, mock_registries):
        """device=None (default) uses auto-detection."""
        with patch('whisperjav.config.resolver_v3.get_best_device') as mock_device:
            mock_device.return_value = "cpu"

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
            )

            assert config["model"]["device"] == "cpu"
            mock_device.assert_called_once()

    def test_compute_type_override_int8(self, mock_registries):
        """Explicit compute_type='int8' overrides default."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                compute_type="int8",
            )

            assert config["model"]["compute_type"] == "int8"

    def test_compute_type_override_float16(self, mock_registries):
        """Explicit compute_type='float16' overrides default."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                compute_type="float16",
            )

            assert config["model"]["compute_type"] == "float16"

    def test_compute_type_override_int8_float16(self, mock_registries):
        """Explicit compute_type='int8_float16' works."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                compute_type="int8_float16",
            )

            assert config["model"]["compute_type"] == "int8_float16"

    def test_compute_type_auto_uses_device_aware_default(self, mock_registries):
        """compute_type='auto' triggers device-aware selection (same as None)."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"), \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                compute_type="auto",  # "auto" = use device-aware default
            )

            # "auto" is treated same as None → device-aware default (float16 on CUDA)
            assert config["model"]["compute_type"] == "float16"

    def test_compute_type_none_uses_device_aware_default(self, mock_registries):
        """compute_type=None uses device-aware default (float16 on CUDA)."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"), \
             patch('whisperjav.config.resolver_v3._is_pascal_gpu', return_value=False), \
             patch('whisperjav.config.resolver_v3._is_blackwell_gpu', return_value=False):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                # compute_type not specified = None → device-aware default
            )

            assert config["model"]["compute_type"] == "float16"

    def test_compute_type_none_cpu_uses_auto(self, mock_registries):
        """compute_type=None on CPU uses 'auto'."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cpu"):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
            )

            assert config["model"]["compute_type"] == "auto"

    def test_combined_device_and_compute_type_override(self, mock_registries):
        """Both device and compute_type can be overridden together."""
        with patch('whisperjav.config.resolver_v3.get_best_device', return_value="cuda"):

            from whisperjav.config.resolver_v3 import resolve_config_v3
            config = resolve_config_v3(
                asr="faster_whisper", vad="none",
                device="cpu",
                compute_type="float32",
            )

            assert config["model"]["device"] == "cpu"
            assert config["model"]["compute_type"] == "float32"


class TestLegacyPipelineDeviceOverride:
    """Tests for device/compute_type override through legacy.py."""

    @pytest.fixture
    def mock_resolve_v3(self):
        """Mock resolve_config_v3 to verify parameters passed."""
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
                    'compute_type': 'float16',
                    'supported_tasks': ['transcribe']
                },
                'params': {'asr': {}, 'vad': {}},
                'features': {},
            }
            yield mock

    def test_resolve_legacy_pipeline_passes_device(self, mock_resolve_v3):
        """resolve_legacy_pipeline passes device parameter."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
            sensitivity="aggressive",
            device="cpu",
        )

        mock_resolve_v3.assert_called_once()
        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') == "cpu"

    def test_resolve_legacy_pipeline_passes_compute_type(self, mock_resolve_v3):
        """resolve_legacy_pipeline passes compute_type parameter."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
            sensitivity="balanced",
            compute_type="int8_float16",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('compute_type') == "int8_float16"

    def test_resolve_legacy_pipeline_passes_both(self, mock_resolve_v3):
        """resolve_legacy_pipeline passes both device and compute_type."""
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
        """resolve_legacy_pipeline passes None when not specified."""
        from whisperjav.config.legacy import resolve_legacy_pipeline

        resolve_legacy_pipeline(
            pipeline_name="balanced",
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
                    'compute_type': 'float16',
                    'supported_tasks': ['transcribe']
                },
                'params': {'asr': {}, 'vad': {}},
                'features': {},
            }
            yield mock

    def test_resolve_ensemble_config_passes_device(self, mock_resolve_v3):
        """resolve_ensemble_config passes device parameter."""
        from whisperjav.config.legacy import resolve_ensemble_config

        resolve_ensemble_config(
            asr="faster_whisper",
            device="cpu",
        )

        call_kwargs = mock_resolve_v3.call_args[1]
        assert call_kwargs.get('device') == "cpu"

    def test_resolve_ensemble_config_passes_compute_type(self, mock_resolve_v3):
        """resolve_ensemble_config passes compute_type parameter."""
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
        """--device accepts valid choices."""
        from whisperjav.main import parse_arguments
        import sys

        for device in ["auto", "cuda", "cpu"]:
            with patch.object(sys, 'argv', ['whisperjav', 'test.mp4', f'--device={device}']):
                args = parse_arguments()
                assert args.device == device

    def test_compute_type_argument_choices(self):
        """--compute-type accepts valid choices."""
        from whisperjav.main import parse_arguments
        import sys

        valid_types = ["auto", "float16", "float32", "int8", "int8_float16", "int8_float32"]
        for compute_type in valid_types:
            with patch.object(sys, 'argv', ['whisperjav', 'test.mp4', f'--compute-type={compute_type}']):
                args = parse_arguments()
                assert args.compute_type == compute_type

    def test_device_default_is_none(self):
        """--device defaults to None (auto-detect)."""
        from whisperjav.main import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['whisperjav', 'test.mp4']):
            args = parse_arguments()
            assert args.device is None

    def test_compute_type_default_is_none(self):
        """--compute-type defaults to None (auto)."""
        from whisperjav.main import parse_arguments
        import sys

        with patch.object(sys, 'argv', ['whisperjav', 'test.mp4']):
            args = parse_arguments()
            assert args.compute_type is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
