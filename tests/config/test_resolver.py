"""
Tests for Configuration Resolver.
"""

import pytest

from whisperjav.config.resolver import resolve_config
from whisperjav.config.errors import ConfigValidationError, UnknownComponentError


class TestResolveConfigBasic:
    """Test basic resolve_config functionality."""

    def test_balanced_pipeline_balanced_sensitivity(self):
        """Test resolving balanced pipeline with balanced sensitivity."""
        result = resolve_config("balanced", "balanced", "transcribe")

        assert result['pipeline_name'] == "balanced"
        assert result['sensitivity_name'] == "balanced"
        assert result['task'] == "transcribe"
        assert result['language'] == "ja"

    def test_output_structure(self):
        """Test output has all required keys."""
        result = resolve_config("balanced", "balanced", "transcribe")

        assert 'pipeline_name' in result
        assert 'sensitivity_name' in result
        assert 'workflow' in result
        assert 'model' in result
        assert 'params' in result
        assert 'features' in result
        assert 'task' in result
        assert 'language' in result

    def test_params_structure(self):
        """Test params has decoder, provider, vad."""
        result = resolve_config("balanced", "balanced", "transcribe")

        assert 'decoder' in result['params']
        assert 'provider' in result['params']
        assert 'vad' in result['params']

    def test_workflow_structure(self):
        """Test workflow has required keys."""
        result = resolve_config("balanced", "balanced", "transcribe")

        workflow = result['workflow']
        assert 'model' in workflow
        assert 'vad' in workflow
        assert 'backend' in workflow

    def test_model_structure(self):
        """Test model has required keys."""
        result = resolve_config("balanced", "balanced", "transcribe")

        model = result['model']
        assert 'provider' in model
        assert 'model_name' in model
        assert 'compute_type' in model
        assert 'supported_tasks' in model


class TestAllPipelines:
    """Test all pipeline configurations."""

    @pytest.mark.parametrize("pipeline", ["faster", "fast", "fidelity", "balanced"])
    def test_all_pipelines_resolve(self, pipeline):
        """Test all pipelines can be resolved."""
        result = resolve_config(pipeline, "balanced", "transcribe")
        assert result['pipeline_name'] == pipeline

    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_all_sensitivities_resolve(self, sensitivity):
        """Test all sensitivities can be resolved."""
        result = resolve_config("balanced", sensitivity, "transcribe")
        assert result['sensitivity_name'] == sensitivity

    @pytest.mark.parametrize("task", ["transcribe", "translate"])
    def test_both_tasks_resolve(self, task):
        """Test both tasks can be resolved."""
        result = resolve_config("balanced", "balanced", task)
        assert result['task'] == task
        assert result['params']['decoder']['task'] == task


class TestDecoderParams:
    """Test decoder parameter resolution."""

    def test_decoder_has_required_params(self):
        """Test decoder params have required keys."""
        result = resolve_config("balanced", "balanced", "transcribe")
        decoder = result['params']['decoder']

        assert 'task' in decoder
        assert 'language' in decoder
        assert 'beam_size' in decoder
        assert 'patience' in decoder

    def test_decoder_values_match_sensitivity(self):
        """Test decoder values change with sensitivity."""
        balanced = resolve_config("balanced", "balanced", "transcribe")
        aggressive = resolve_config("balanced", "aggressive", "transcribe")

        # Aggressive should have different patience
        assert balanced['params']['decoder']['patience'] != aggressive['params']['decoder']['patience']

    def test_task_override(self):
        """Test task is overridden in decoder params."""
        result = resolve_config("balanced", "balanced", "translate")
        assert result['params']['decoder']['task'] == "translate"


class TestProviderParams:
    """Test provider parameter resolution."""

    def test_provider_has_params(self):
        """Test provider params are populated."""
        result = resolve_config("balanced", "balanced", "transcribe")
        provider = result['params']['provider']

        # Should have transcriber options
        assert 'temperature' in provider
        assert 'compression_ratio_threshold' in provider

    def test_no_none_values(self):
        """Test provider params have no None values."""
        result = resolve_config("balanced", "balanced", "transcribe")
        provider = result['params']['provider']

        def check_no_none(d):
            for key, value in d.items():
                assert value is not None, f"Found None for key: {key}"
                if isinstance(value, dict):
                    check_no_none(value)

        check_no_none(provider)


class TestVADParams:
    """Test VAD parameter resolution."""

    def test_balanced_pipeline_has_vad_params(self):
        """Test balanced pipeline has separate VAD params."""
        result = resolve_config("balanced", "balanced", "transcribe")
        vad = result['params']['vad']

        # Balanced uses separate VAD handling
        assert 'threshold' in vad or len(vad) == 0

    def test_vad_sensitivity_values(self):
        """Test VAD values change with sensitivity."""
        conservative = resolve_config("balanced", "conservative", "transcribe")
        aggressive = resolve_config("balanced", "aggressive", "transcribe")

        # Should have different thresholds
        if conservative['params']['vad'] and aggressive['params']['vad']:
            assert conservative['params']['vad'].get('threshold', 0) != aggressive['params']['vad'].get('threshold', 0)


class TestFeatures:
    """Test feature configuration resolution."""

    def test_balanced_has_scene_detection(self):
        """Test balanced pipeline has scene detection feature."""
        result = resolve_config("balanced", "balanced", "transcribe")

        assert 'scene_detection' in result['features']

    def test_scene_detection_has_method(self):
        """Test scene detection config has method."""
        result = resolve_config("balanced", "balanced", "transcribe")

        if 'scene_detection' in result['features']:
            assert 'method' in result['features']['scene_detection']

    def test_scene_detection_method_override(self):
        """Test scene detection method can be overridden."""
        result = resolve_config(
            "balanced", "balanced", "transcribe",
            scene_detection_method="silero"
        )

        if 'scene_detection' in result['features']:
            assert result['features']['scene_detection']['method'] == "silero"


class TestTypeConversions:
    """Test backend-specific type conversions."""

    def test_stable_ts_integer_conversion(self):
        """Test integer params are converted for stable-ts."""
        result = resolve_config("faster", "balanced", "transcribe")

        # faster pipeline uses stable-ts backend
        decoder = result['params']['decoder']
        assert isinstance(decoder['beam_size'], int)
        assert isinstance(decoder['best_of'], int)

    def test_float_params_conversion(self):
        """Test float params are floats."""
        result = resolve_config("balanced", "balanced", "transcribe")

        decoder = result['params']['decoder']
        assert isinstance(decoder['patience'], float)


class TestErrorHandling:
    """Test error handling."""

    def test_unknown_pipeline_raises(self):
        """Test unknown pipeline raises error."""
        with pytest.raises(UnknownComponentError):
            resolve_config("nonexistent", "balanced", "transcribe")

    def test_invalid_sensitivity_raises(self):
        """Test invalid sensitivity raises error."""
        with pytest.raises(ConfigValidationError):
            resolve_config("balanced", "invalid", "transcribe")

    def test_invalid_task_raises(self):
        """Test invalid task raises error."""
        with pytest.raises(ConfigValidationError):
            resolve_config("balanced", "balanced", "invalid")


class TestModelResolution:
    """Test model resolution and overrides."""

    def test_balanced_model_resolution(self):
        """Test balanced pipeline resolves correct model."""
        result = resolve_config("balanced", "balanced", "transcribe")

        model = result['model']
        assert model['provider'] == "faster_whisper"
        assert model['model_name'] == "large-v2"

    def test_model_override_for_translate(self):
        """Test model override for translate task."""
        # balanced pipeline has model_overrides for translate
        result = resolve_config("balanced", "balanced", "translate")

        # Should still use the same model for balanced
        assert result['model']['model_name'] == "large-v2"
