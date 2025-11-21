"""
Tests for Legacy Pipeline Mappings.
"""

import pytest

from whisperjav.config.legacy import (
    resolve_legacy_pipeline,
    list_legacy_pipelines,
    get_legacy_pipeline_info,
    LEGACY_PIPELINES,
)


class TestResolveLegacyPipeline:
    """Test resolve_legacy_pipeline function."""

    @pytest.mark.parametrize("pipeline", ["balanced", "faster", "fast", "fidelity"])
    def test_all_pipelines_resolve(self, pipeline):
        """Test all legacy pipelines can be resolved."""
        config = resolve_legacy_pipeline(pipeline, "balanced", "transcribe")

        assert config['pipeline_name'] == pipeline
        assert config['sensitivity_name'] == "balanced"

    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_all_sensitivities(self, sensitivity):
        """Test all sensitivity levels work."""
        config = resolve_legacy_pipeline("balanced", sensitivity)

        assert config['sensitivity_name'] == sensitivity

    @pytest.mark.parametrize("task", ["transcribe", "translate"])
    def test_both_tasks(self, task):
        """Test both task types."""
        config = resolve_legacy_pipeline("balanced", "balanced", task)

        assert config['task'] == task
        assert config['params']['decoder']['task'] == task


class TestLegacyStructure:
    """Test legacy output structure."""

    def test_has_pipeline_name(self):
        """Test output has pipeline_name."""
        config = resolve_legacy_pipeline("balanced")
        assert config['pipeline_name'] == "balanced"

    def test_has_workflow(self):
        """Test output has workflow structure."""
        config = resolve_legacy_pipeline("balanced")

        assert 'workflow' in config
        assert 'model' in config['workflow']
        assert 'vad' in config['workflow']
        assert 'backend' in config['workflow']

    def test_has_params_decoder(self):
        """Test output has params.decoder."""
        config = resolve_legacy_pipeline("balanced")

        assert 'decoder' in config['params']
        assert 'task' in config['params']['decoder']
        assert 'language' in config['params']['decoder']
        assert 'beam_size' in config['params']['decoder']

    def test_has_params_provider(self):
        """Test output has params.provider."""
        config = resolve_legacy_pipeline("balanced")

        assert 'provider' in config['params']
        assert 'temperature' in config['params']['provider']
        assert 'no_speech_threshold' in config['params']['provider']

    def test_has_params_vad(self):
        """Test output has params.vad."""
        config = resolve_legacy_pipeline("balanced")

        assert 'vad' in config['params']
        # balanced has silero VAD
        assert 'threshold' in config['params']['vad']


class TestBalancedPipeline:
    """Test balanced pipeline specifically."""

    def test_uses_faster_whisper(self):
        """Test balanced uses faster_whisper."""
        config = resolve_legacy_pipeline("balanced")

        assert config['model']['provider'] == 'faster_whisper'
        assert config['workflow']['backend'] == 'faster-whisper'

    def test_has_silero_vad(self):
        """Test balanced has silero VAD."""
        config = resolve_legacy_pipeline("balanced")

        assert config['workflow']['vad'] == 'silero'
        assert config['params']['vad'] != {}

    def test_has_scene_detection(self):
        """Test balanced has scene detection."""
        config = resolve_legacy_pipeline("balanced")

        assert 'scene_detection' in config['features']


class TestFasterPipeline:
    """Test faster pipeline specifically."""

    def test_uses_stable_ts(self):
        """Test faster uses stable_ts."""
        config = resolve_legacy_pipeline("faster")

        assert config['workflow']['backend'] == 'stable-ts'

    def test_no_vad(self):
        """Test faster has no VAD."""
        config = resolve_legacy_pipeline("faster")

        assert config['workflow']['vad'] == 'none'
        assert config['params']['vad'] == {}

    def test_no_features(self):
        """Test faster has no features."""
        config = resolve_legacy_pipeline("faster")

        assert config['features'] == {}


class TestOverrides:
    """Test parameter overrides with legacy pipelines."""

    def test_asr_override(self):
        """Test overriding ASR parameter."""
        config = resolve_legacy_pipeline(
            "balanced", "balanced",
            overrides={'asr.beam_size': 10}
        )

        assert config['params']['decoder']['beam_size'] == 10

    def test_vad_override(self):
        """Test overriding VAD parameter."""
        config = resolve_legacy_pipeline(
            "balanced", "balanced",
            overrides={'vad.threshold': 0.7}
        )

        assert config['params']['vad']['threshold'] == 0.7


class TestErrorHandling:
    """Test error handling."""

    def test_unknown_pipeline_raises(self):
        """Test unknown pipeline raises error."""
        with pytest.raises(ValueError, match="Unknown pipeline"):
            resolve_legacy_pipeline("nonexistent")


class TestHelperFunctions:
    """Test helper functions."""

    def test_list_legacy_pipelines(self):
        """Test listing legacy pipelines."""
        pipelines = list_legacy_pipelines()

        assert 'balanced' in pipelines
        assert 'faster' in pipelines
        assert 'fast' in pipelines
        assert 'fidelity' in pipelines

    def test_get_legacy_pipeline_info(self):
        """Test getting pipeline info."""
        info = get_legacy_pipeline_info("balanced")

        assert info['name'] == 'balanced'
        assert info['asr'] == 'faster_whisper'
        assert info['vad'] == 'silero'
        assert 'description' in info


class TestBackwardCompatibility:
    """Test backward compatibility with old system."""

    @pytest.mark.parametrize("pipeline", ["balanced", "faster", "fast", "fidelity"])
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_all_combinations_have_required_keys(self, pipeline, sensitivity):
        """Test all combinations have required output keys."""
        config = resolve_legacy_pipeline(pipeline, sensitivity)

        # Required top-level keys
        assert 'pipeline_name' in config
        assert 'sensitivity_name' in config
        assert 'workflow' in config
        assert 'model' in config
        assert 'params' in config
        assert 'features' in config
        assert 'task' in config
        assert 'language' in config

        # Required params keys
        assert 'decoder' in config['params']
        assert 'provider' in config['params']
        assert 'vad' in config['params']
