"""
Tests for Features, Pipeline, and UI configuration schemas.
"""

import pytest
from pydantic import ValidationError as PydanticValidationError

from whisperjav.config.schemas import (
    AuditokSceneDetectionConfig,
    Backend,
    PostProcessingConfig,
    ResolvedConfig,
    ResolvedParams,
    SileroSceneDetectionConfig,
    UIPreferences,
    WorkflowConfig,
)


class TestAuditokSceneDetectionConfig:
    """Test AuditokSceneDetectionConfig schema (25 parameters)."""

    def test_default_values(self):
        """Test default values are correct."""
        config = AuditokSceneDetectionConfig()
        assert config.max_duration_s == 29.0
        assert config.min_duration_s == 0.2
        assert config.target_sr == 16000
        assert config.force_mono is True
        assert config.preserve_original_sr is True
        assert config.pass1_energy_threshold == 32
        assert config.pass2_energy_threshold == 38
        assert config.brute_force_fallback is True

    def test_custom_values(self):
        """Test custom values can be set."""
        config = AuditokSceneDetectionConfig(
            max_duration_s=60.0,
            pass1_energy_threshold=50,
            pass2_energy_threshold=60
        )
        assert config.max_duration_s == 60.0
        assert config.pass1_energy_threshold == 50

    def test_model_dump_includes_all_fields(self):
        """Test all fields are included in dump."""
        config = AuditokSceneDetectionConfig()
        result = config.model_dump()

        # Check all major categories are present
        assert "max_duration_s" in result
        assert "pass1_min_duration_s" in result
        assert "pass2_min_duration_s" in result
        assert "bandpass_low_hz" in result
        assert "brute_force_fallback" in result


class TestSileroSceneDetectionConfig:
    """Test SileroSceneDetectionConfig schema (30 parameters)."""

    def test_default_values(self):
        """Test default values are correct."""
        config = SileroSceneDetectionConfig()
        assert config.method == "silero"
        assert config.max_duration_s == 29.0
        assert config.silero_threshold == 0.08  # Optimized from 0.02 for balanced detection
        assert config.silero_neg_threshold == 0.1
        assert config.silero_min_silence_ms == 7800
        assert config.silero_speech_pad_ms == 500

    def test_custom_silero_settings(self):
        """Test Silero-specific settings."""
        config = SileroSceneDetectionConfig(
            silero_threshold=0.05,
            silero_min_silence_ms=5000,
            silero_max_speech_s=120
        )
        assert config.silero_threshold == 0.05
        assert config.silero_min_silence_ms == 5000
        assert config.silero_max_speech_s == 120


class TestPostProcessingConfig:
    """Test PostProcessingConfig schema."""

    def test_default_values(self):
        """Test default values."""
        config = PostProcessingConfig()
        assert config.enabled is True
        assert config.sanitize is True

    def test_custom_values(self):
        """Test custom values."""
        config = PostProcessingConfig(enabled=False, sanitize=False)
        assert config.enabled is False
        assert config.sanitize is False


class TestWorkflowConfig:
    """Test WorkflowConfig schema."""

    def test_valid_workflow(self):
        """Test valid workflow config."""
        config = WorkflowConfig(
            model="faster-whisper-large-v2-int8",
            vad="silero-v3.1",
            backend=Backend.FASTER_WHISPER,
            features={"scene_detection": "default"}
        )
        assert config.model == "faster-whisper-large-v2-int8"
        assert config.vad == "silero-v3.1"
        assert config.backend == "faster-whisper"

    def test_no_vad(self):
        """Test workflow without VAD."""
        config = WorkflowConfig(
            model="whisper-turbo",
            vad="none",
            backend=Backend.WHISPER
        )
        assert config.vad == "none"

    def test_empty_features(self):
        """Test workflow with empty features."""
        config = WorkflowConfig(
            model="whisper-turbo",
            vad="none",
            backend=Backend.WHISPER
        )
        assert config.features == {}


class TestResolvedParams:
    """Test ResolvedParams schema."""

    def test_valid_params(self):
        """Test valid resolved params."""
        config = ResolvedParams(
            decoder={"task": "transcribe", "language": "ja", "beam_size": 2},
            provider={"temperature": [0.0, 0.1], "regroup": True},
            vad={"threshold": 0.18}
        )
        assert config.decoder["task"] == "transcribe"
        assert config.provider["regroup"] is True
        assert config.vad["threshold"] == 0.18

    def test_empty_vad(self):
        """Test resolved params with empty VAD."""
        config = ResolvedParams(
            decoder={"task": "transcribe"},
            provider={"temperature": 0.0}
        )
        assert config.vad == {}


class TestResolvedConfig:
    """Test ResolvedConfig schema."""

    def test_valid_resolved_config(self):
        """Test complete resolved config structure."""
        config = ResolvedConfig(
            pipeline_name="balanced",
            sensitivity_name="aggressive",
            workflow=WorkflowConfig(
                model="faster-whisper-large-v2-int8",
                vad="silero-v3.1",
                backend=Backend.FASTER_WHISPER
            ),
            model={
                "provider": "faster_whisper",
                "model_name": "large-v2",
                "compute_type": "int8_float16",
                "supported_tasks": ["transcribe", "translate"]
            },
            params=ResolvedParams(
                decoder={"task": "transcribe", "language": "ja"},
                provider={"temperature": [0.0, 0.3]}
            ),
            features={"scene_detection": {"method": "silero"}},
            task="transcribe",
            language="ja"
        )
        assert config.pipeline_name == "balanced"
        assert config.sensitivity_name == "aggressive"
        assert config.task == "transcribe"

    def test_invalid_task(self):
        """Test invalid task is rejected."""
        with pytest.raises(PydanticValidationError):
            ResolvedConfig(
                pipeline_name="balanced",
                sensitivity_name="aggressive",
                workflow=WorkflowConfig(
                    model="test",
                    vad="none",
                    backend=Backend.WHISPER
                ),
                model={},
                params=ResolvedParams(decoder={}, provider={}),
                features={},
                task="invalid"
            )

    def test_model_dump_structure(self):
        """Test exported structure matches expected format."""
        config = ResolvedConfig(
            pipeline_name="fast",
            sensitivity_name="balanced",
            workflow=WorkflowConfig(
                model="whisper-turbo",
                vad="none",
                backend=Backend.WHISPER
            ),
            model={"provider": "openai_whisper"},
            params=ResolvedParams(
                decoder={"beam_size": 1},
                provider={"fp16": True}
            ),
            features={},
            task="transcribe"
        )

        result = config.model_dump()
        assert "pipeline_name" in result
        assert "sensitivity_name" in result
        assert "workflow" in result
        assert "model" in result
        assert "params" in result
        assert "features" in result
        assert "task" in result
        assert "language" in result


class TestUIPreferences:
    """Test UIPreferences schema (14 parameters)."""

    def test_default_values(self):
        """Test default values are correct."""
        config = UIPreferences()
        assert config.console_verbosity == "summary"
        assert config.progress_batch_size == 10
        assert config.show_scene_details is False
        assert config.max_console_lines == 1000
        assert config.auto_scroll is True
        assert config.show_timestamps is False
        assert config.theme == "default"
        assert config.last_mode == "fidelity"
        assert config.last_sensitivity == "conservative"
        assert config.last_language == "japanese"
        assert config.show_console is True
        assert config.adaptive_classification is True
        assert config.adaptive_audio_enhancement is True
        assert config.smart_postprocessing is True

    def test_invalid_verbosity(self):
        """Test invalid verbosity is rejected."""
        with pytest.raises(PydanticValidationError):
            UIPreferences(console_verbosity="invalid")

    def test_progress_batch_size_must_be_positive(self):
        """Test progress_batch_size must be >= 1."""
        with pytest.raises(PydanticValidationError):
            UIPreferences(progress_batch_size=0)

    def test_custom_values(self):
        """Test custom values can be set."""
        config = UIPreferences(
            console_verbosity="verbose",
            theme="dark",
            last_mode="balanced",
            last_sensitivity="aggressive"
        )
        assert config.console_verbosity == "verbose"
        assert config.theme == "dark"
        assert config.last_mode == "balanced"
