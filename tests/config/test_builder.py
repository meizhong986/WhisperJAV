"""
Tests for PipelineBuilder.
"""

import pytest

from whisperjav.config.builder import PipelineBuilder, quick_config
from whisperjav.config.errors import ConfigValidationError


class TestPipelineBuilderBasic:
    """Test basic PipelineBuilder functionality."""

    def test_default_build(self):
        """Test building with defaults."""
        config = PipelineBuilder("balanced").build()

        assert config['pipeline_name'] == "balanced"
        assert config['sensitivity_name'] == "balanced"
        assert config['task'] == "transcribe"

    def test_all_pipelines(self):
        """Test all pipeline names work."""
        for pipeline in ["faster", "fast", "fidelity", "balanced"]:
            config = PipelineBuilder(pipeline).build()
            assert config['pipeline_name'] == pipeline

    def test_with_sensitivity(self):
        """Test setting sensitivity."""
        config = (
            PipelineBuilder("balanced")
            .with_sensitivity("aggressive")
            .build()
        )
        assert config['sensitivity_name'] == "aggressive"

    def test_with_task(self):
        """Test setting task."""
        config = (
            PipelineBuilder("balanced")
            .with_task("translate")
            .build()
        )
        assert config['task'] == "translate"
        assert config['params']['decoder']['task'] == "translate"


class TestPipelineBuilderOverrides:
    """Test parameter overrides."""

    def test_decoder_param_override(self):
        """Test overriding decoder parameter."""
        config = (
            PipelineBuilder("balanced")
            .with_decoder_param("beam_size", 10)
            .build()
        )
        assert config['params']['decoder']['beam_size'] == 10

    def test_provider_param_override(self):
        """Test overriding provider parameter."""
        config = (
            PipelineBuilder("balanced")
            .with_provider_param("temperature", 0.5)
            .build()
        )
        assert config['params']['provider']['temperature'] == 0.5

    def test_vad_param_override(self):
        """Test overriding VAD parameter."""
        config = (
            PipelineBuilder("balanced")
            .with_vad_param("threshold", 0.6)
            .build()
        )
        assert config['params']['vad']['threshold'] == 0.6

    def test_beam_size_shorthand(self):
        """Test beam_size shorthand method."""
        config = (
            PipelineBuilder("balanced")
            .with_beam_size(15)
            .build()
        )
        assert config['params']['decoder']['beam_size'] == 15

    def test_patience_shorthand(self):
        """Test patience shorthand method."""
        config = (
            PipelineBuilder("balanced")
            .with_patience(2.0)
            .build()
        )
        assert config['params']['decoder']['patience'] == 2.0

    def test_temperature_shorthand(self):
        """Test temperature shorthand method."""
        config = (
            PipelineBuilder("balanced")
            .with_temperature(0.3)
            .build()
        )
        assert config['params']['provider']['temperature'] == 0.3

    def test_no_speech_threshold_shorthand(self):
        """Test no_speech_threshold shorthand method."""
        config = (
            PipelineBuilder("balanced")
            .with_no_speech_threshold(0.7)
            .build()
        )
        assert config['params']['provider']['no_speech_threshold'] == 0.7

    def test_vad_threshold_shorthand(self):
        """Test vad_threshold shorthand method."""
        config = (
            PipelineBuilder("balanced")
            .with_vad_threshold(0.5)
            .build()
        )
        assert config['params']['vad']['threshold'] == 0.5

    def test_multiple_overrides(self):
        """Test multiple overrides together."""
        config = (
            PipelineBuilder("balanced")
            .with_sensitivity("aggressive")
            .with_task("translate")
            .with_beam_size(8)
            .with_patience(1.5)
            .with_temperature(0.2)
            .build()
        )

        assert config['sensitivity_name'] == "aggressive"
        assert config['task'] == "translate"
        assert config['params']['decoder']['beam_size'] == 8
        assert config['params']['decoder']['patience'] == 1.5
        assert config['params']['provider']['temperature'] == 0.2


class TestPipelineBuilderFeatures:
    """Test feature overrides."""

    def test_scene_detection_method(self):
        """Test setting scene detection method."""
        config = (
            PipelineBuilder("balanced")
            .with_scene_detection_method("silero")
            .build()
        )

        if 'scene_detection' in config['features']:
            assert config['features']['scene_detection']['method'] == "silero"

    def test_feature_override(self):
        """Test overriding feature configuration."""
        config = (
            PipelineBuilder("balanced")
            .with_feature("scene_detection", {"min_duration_s": 5.0})
            .build()
        )

        if 'scene_detection' in config['features']:
            assert config['features']['scene_detection']['min_duration_s'] == 5.0


class TestPipelineBuilderValidation:
    """Test validation errors."""

    def test_invalid_sensitivity_raises(self):
        """Test invalid sensitivity raises error."""
        with pytest.raises(ConfigValidationError):
            PipelineBuilder("balanced").with_sensitivity("invalid")

    def test_invalid_task_raises(self):
        """Test invalid task raises error."""
        with pytest.raises(ConfigValidationError):
            PipelineBuilder("balanced").with_task("invalid")


class TestPipelineBuilderCopy:
    """Test builder copy functionality."""

    def test_copy_creates_independent_builder(self):
        """Test copy creates independent instance."""
        original = (
            PipelineBuilder("balanced")
            .with_sensitivity("aggressive")
            .with_beam_size(10)
        )

        copied = original.copy()
        copied.with_beam_size(20)

        original_config = original.build()
        copied_config = copied.build()

        assert original_config['params']['decoder']['beam_size'] == 10
        assert copied_config['params']['decoder']['beam_size'] == 20

    def test_copy_preserves_all_settings(self):
        """Test copy preserves all builder settings."""
        original = (
            PipelineBuilder("fidelity")
            .with_sensitivity("conservative")
            .with_task("translate")
            .with_beam_size(5)
            .with_temperature(0.1)
        )

        copied = original.copy()
        config = copied.build()

        assert config['pipeline_name'] == "fidelity"
        assert config['sensitivity_name'] == "conservative"
        assert config['task'] == "translate"
        assert config['params']['decoder']['beam_size'] == 5
        assert config['params']['provider']['temperature'] == 0.1


class TestQuickConfig:
    """Test quick_config helper function."""

    def test_basic_quick_config(self):
        """Test basic quick_config usage."""
        config = quick_config("balanced", "balanced", "transcribe")

        assert config['pipeline_name'] == "balanced"
        assert config['sensitivity_name'] == "balanced"
        assert config['task'] == "transcribe"

    def test_quick_config_with_overrides(self):
        """Test quick_config with parameter overrides."""
        config = quick_config(
            "balanced",
            "aggressive",
            beam_size=10,
            temperature=0.3
        )

        assert config['sensitivity_name'] == "aggressive"
        assert config['params']['decoder']['beam_size'] == 10
        assert config['params']['provider']['temperature'] == 0.3

    def test_quick_config_defaults(self):
        """Test quick_config with all defaults."""
        config = quick_config()

        assert config['pipeline_name'] == "balanced"
        assert config['sensitivity_name'] == "balanced"
        assert config['task'] == "transcribe"

    def test_quick_config_vad_params(self):
        """Test quick_config with VAD parameters."""
        config = quick_config(
            "balanced",
            threshold=0.5,
            min_speech_duration_ms=300
        )

        assert config['params']['vad']['threshold'] == 0.5
        assert config['params']['vad']['min_speech_duration_ms'] == 300

    def test_quick_config_scene_detection(self):
        """Test quick_config with scene detection method."""
        config = quick_config(
            "balanced",
            scene_detection_method="silero"
        )

        if 'scene_detection' in config['features']:
            assert config['features']['scene_detection']['method'] == "silero"


class TestFluentChaining:
    """Test fluent API chaining works correctly."""

    def test_method_chaining_returns_builder(self):
        """Test all methods return builder for chaining."""
        builder = PipelineBuilder("balanced")

        # Each method should return the builder
        assert builder.with_sensitivity("aggressive") is builder
        assert builder.with_task("translate") is builder
        assert builder.with_decoder_param("beam_size", 5) is builder
        assert builder.with_provider_param("temperature", 0.1) is builder
        assert builder.with_vad_param("threshold", 0.5) is builder
        assert builder.with_beam_size(10) is builder
        assert builder.with_patience(1.5) is builder
        assert builder.with_temperature(0.2) is builder
        assert builder.with_no_speech_threshold(0.6) is builder
        assert builder.with_vad_threshold(0.4) is builder
        assert builder.with_scene_detection_method("auditok") is builder
        assert builder.with_feature("test", {}) is builder

    def test_long_chain(self):
        """Test long method chain works."""
        config = (
            PipelineBuilder("faster")
            .with_sensitivity("conservative")
            .with_task("translate")
            .with_beam_size(3)
            .with_patience(1.0)
            .with_temperature(0.0)
            .with_no_speech_threshold(0.8)
            .build()
        )

        assert config['pipeline_name'] == "faster"
        assert config['sensitivity_name'] == "conservative"
        assert config['task'] == "translate"
        assert config['params']['decoder']['beam_size'] == 3
        assert config['params']['decoder']['patience'] == 1.0
        assert config['params']['provider']['temperature'] == 0.0
        assert config['params']['provider']['no_speech_threshold'] == 0.8
