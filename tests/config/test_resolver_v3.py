"""
Tests for Configuration Resolver v3.0 (Component-Based).
"""

import pytest

from whisperjav.config.resolver_v3 import (
    resolve_config_v3,
    get_asr_options,
    get_vad_options,
    get_compatible_vads,
    list_available_asr,
    list_available_vad,
    list_available_features,
)


class TestResolveConfigV3Basic:
    """Test basic resolve_config_v3 functionality."""

    def test_basic_resolution(self):
        """Test basic configuration resolution."""
        config = resolve_config_v3(
            asr='faster_whisper',
            vad='silero',
            sensitivity='balanced',
            task='transcribe'
        )

        assert config['asr_name'] == 'faster_whisper'
        assert config['vad_name'] == 'silero'
        assert config['sensitivity_name'] == 'balanced'
        assert config['task'] == 'transcribe'

    def test_output_structure(self):
        """Test output has all required keys."""
        config = resolve_config_v3('faster_whisper', 'silero', 'balanced')

        assert 'asr_name' in config
        assert 'vad_name' in config
        assert 'sensitivity_name' in config
        assert 'task' in config
        assert 'language' in config
        assert 'model' in config
        assert 'params' in config
        assert 'features' in config

    def test_params_structure(self):
        """Test params has asr and vad."""
        config = resolve_config_v3('faster_whisper', 'silero', 'balanced')

        assert 'asr' in config['params']
        assert 'vad' in config['params']

    def test_model_structure(self):
        """Test model has required keys."""
        config = resolve_config_v3('faster_whisper', 'silero', 'balanced')

        model = config['model']
        assert 'provider' in model
        assert 'model_name' in model
        assert 'device' in model
        assert 'compute_type' in model
        assert 'supported_tasks' in model


class TestAllCombinations:
    """Test all valid combinations."""

    @pytest.mark.parametrize("asr", ["faster_whisper", "stable_ts"])
    @pytest.mark.parametrize("sensitivity", ["conservative", "balanced", "aggressive"])
    def test_asr_sensitivity_combinations(self, asr, sensitivity):
        """Test all ASR × sensitivity combinations."""
        config = resolve_config_v3(asr, 'none', sensitivity)

        assert config['asr_name'] == asr
        assert config['sensitivity_name'] == sensitivity
        assert config['params']['asr'] is not None

    @pytest.mark.parametrize("task", ["transcribe", "translate"])
    def test_both_tasks(self, task):
        """Test both task types."""
        config = resolve_config_v3('faster_whisper', 'silero', 'balanced', task)

        assert config['task'] == task
        assert config['params']['asr']['task'] == task


class TestSensitivityPresets:
    """Test sensitivity presets work correctly with v1-accurate values."""

    def test_conservative_values(self):
        """Test conservative preset has expected v1 values."""
        config = resolve_config_v3('faster_whisper', 'silero', 'conservative')

        # Conservative: strict thresholds, smallest beam
        assert config['params']['vad']['threshold'] == 0.35
        assert config['params']['asr']['beam_size'] == 1
        assert config['params']['asr']['no_speech_threshold'] == 0.74

    def test_aggressive_values(self):
        """Test aggressive preset has expected v1 values."""
        config = resolve_config_v3('faster_whisper', 'silero', 'aggressive')

        # Aggressive: permissive thresholds, low VAD threshold
        assert config['params']['vad']['threshold'] == 0.05
        assert config['params']['asr']['beam_size'] == 2
        assert config['params']['asr']['no_speech_threshold'] == 0.22
        assert config['params']['asr']['patience'] == 2.9

    def test_presets_differ(self):
        """Test presets produce different values."""
        conservative = resolve_config_v3('faster_whisper', 'silero', 'conservative')
        aggressive = resolve_config_v3('faster_whisper', 'silero', 'aggressive')

        # beam_size: conservative=1, aggressive=2
        assert conservative['params']['asr']['beam_size'] != aggressive['params']['asr']['beam_size']
        # VAD threshold: conservative=0.35, aggressive=0.05
        assert conservative['params']['vad']['threshold'] != aggressive['params']['vad']['threshold']
        # no_speech_threshold: conservative=0.74, aggressive=0.22
        assert conservative['params']['asr']['no_speech_threshold'] != aggressive['params']['asr']['no_speech_threshold']


class TestOverrides:
    """Test parameter overrides."""

    def test_asr_override(self):
        """Test overriding ASR parameter."""
        config = resolve_config_v3(
            'faster_whisper', 'silero', 'balanced',
            overrides={'asr.beam_size': 10}
        )

        assert config['params']['asr']['beam_size'] == 10

    def test_vad_override(self):
        """Test overriding VAD parameter."""
        config = resolve_config_v3(
            'faster_whisper', 'silero', 'balanced',
            overrides={'vad.threshold': 0.7}
        )

        assert config['params']['vad']['threshold'] == 0.7

    def test_multiple_overrides(self):
        """Test multiple overrides."""
        config = resolve_config_v3(
            'faster_whisper', 'silero', 'balanced',
            overrides={
                'asr.beam_size': 12,
                'asr.temperature': 0.5,
                'vad.threshold': 0.8
            }
        )

        assert config['params']['asr']['beam_size'] == 12
        assert config['params']['asr']['temperature'] == 0.5
        assert config['params']['vad']['threshold'] == 0.8


class TestFeatures:
    """Test feature configuration."""

    def test_with_scene_detection(self):
        """Test configuration with scene detection feature."""
        config = resolve_config_v3(
            'faster_whisper', 'silero', 'balanced',
            features=['auditok_scene_detection']
        )

        assert 'scene_detection' in config['features']
        assert 'max_duration_s' in config['features']['scene_detection']

    def test_no_features(self):
        """Test configuration without features."""
        config = resolve_config_v3('faster_whisper', 'silero', 'balanced')

        assert config['features'] == {}


class TestNoVAD:
    """Test configuration without VAD."""

    def test_vad_none(self):
        """Test vad='none' produces empty vad params."""
        config = resolve_config_v3('faster_whisper', 'none', 'balanced')

        assert config['vad_name'] == 'none'
        assert config['params']['vad'] == {}


class TestErrorHandling:
    """Test error handling."""

    def test_unknown_asr_raises(self):
        """Test unknown ASR raises error."""
        with pytest.raises(ValueError, match="Unknown ASR"):
            resolve_config_v3('nonexistent', 'silero', 'balanced')

    def test_unknown_vad_raises(self):
        """Test unknown VAD raises error."""
        with pytest.raises(ValueError, match="Unknown VAD"):
            resolve_config_v3('faster_whisper', 'nonexistent', 'balanced')

    def test_unknown_feature_raises(self):
        """Test unknown feature raises error."""
        with pytest.raises(ValueError, match="Unknown feature"):
            resolve_config_v3(
                'faster_whisper', 'silero', 'balanced',
                features=['nonexistent_feature']
            )


class TestHelperFunctions:
    """Test helper functions."""

    def test_list_available_asr(self):
        """Test listing available ASR components."""
        asr_list = list_available_asr()

        names = [a['name'] for a in asr_list]
        assert 'faster_whisper' in names
        assert 'stable_ts' in names

    def test_list_available_vad(self):
        """Test listing available VAD components."""
        vad_list = list_available_vad()

        names = [v['name'] for v in vad_list]
        assert 'silero' in names

    def test_list_available_features(self):
        """Test listing available features."""
        feature_list = list_available_features()

        names = [f['name'] for f in feature_list]
        assert 'auditok_scene_detection' in names

    def test_get_asr_options(self):
        """Test getting ASR parameter options."""
        options = get_asr_options('faster_whisper')

        param_names = [p['name'] for p in options]
        assert 'beam_size' in param_names
        assert 'temperature' in param_names

    def test_get_vad_options(self):
        """Test getting VAD parameter options."""
        options = get_vad_options('silero')

        param_names = [p['name'] for p in options]
        assert 'threshold' in param_names
        assert 'min_speech_duration_ms' in param_names

    def test_get_compatible_vads(self):
        """Test getting compatible VADs for ASR."""
        compatible = get_compatible_vads('faster_whisper')

        assert 'silero' in compatible
