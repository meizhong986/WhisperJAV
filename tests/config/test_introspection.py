"""
Tests for Introspection API.
"""

import pytest

from whisperjav.config.introspection import (
    extract_parameter_schema,
    get_available_components,
    get_component_defaults,
    get_component_schema,
    list_pipelines,
    list_sensitivities,
)
from whisperjav.config.schemas import SileroVADOptions, StableTSEngineOptions


class TestGetAvailableComponents:
    """Test get_available_components function."""

    def test_returns_all_types(self):
        """Test that all component types are returned."""
        components = get_available_components()

        assert 'vad' in components
        assert 'asr' in components
        assert 'scene_detection' in components

    def test_vad_components_present(self):
        """Test all VAD components are listed."""
        components = get_available_components()
        vad_names = [v['name'] for v in components['vad']]

        assert 'silero_vad' in vad_names
        assert 'stable_ts_vad' in vad_names
        assert 'faster_whisper_vad' in vad_names

    def test_asr_components_present(self):
        """Test all ASR components are listed."""
        components = get_available_components()
        asr_names = [a['name'] for a in components['asr']]

        assert 'faster_whisper' in asr_names
        assert 'stable_ts' in asr_names
        assert 'openai_whisper' in asr_names

    def test_component_structure(self):
        """Test component dictionaries have required fields."""
        components = get_available_components()

        for vad in components['vad']:
            assert 'name' in vad
            assert 'display_name' in vad
            assert 'description' in vad
            assert 'tags' in vad


class TestGetComponentSchema:
    """Test get_component_schema function."""

    def test_vad_schema(self):
        """Test getting VAD component schema."""
        schema = get_component_schema('vad', 'silero_vad')

        assert schema['name'] == 'silero_vad'
        assert 'parameters' in schema
        assert len(schema['parameters']) > 0

    def test_asr_schema(self):
        """Test getting ASR component schema."""
        schema = get_component_schema('asr', 'stable_ts')

        assert schema['name'] == 'stable_ts'
        assert 'parameters' in schema
        # StableTSEngineOptions has many parameters
        assert len(schema['parameters']) >= 20

    def test_parameter_schema_structure(self):
        """Test parameter schema has required fields."""
        schema = get_component_schema('vad', 'silero_vad')

        for param in schema['parameters']:
            assert 'name' in param
            assert 'type' in param
            assert 'description' in param
            assert 'required' in param

    def test_parameter_constraints(self):
        """Test constraints are extracted."""
        schema = get_component_schema('vad', 'silero_vad')

        # Find threshold parameter (has ge=0.0, le=1.0)
        threshold_param = next(
            (p for p in schema['parameters'] if p['name'] == 'threshold'),
            None
        )
        assert threshold_param is not None
        assert 'constraints' in threshold_param
        assert threshold_param['constraints'].get('ge') == 0.0
        assert threshold_param['constraints'].get('le') == 1.0

    def test_unknown_component_raises(self):
        """Test unknown component raises error."""
        with pytest.raises(KeyError):
            get_component_schema('vad', 'nonexistent')

    def test_unknown_type_raises(self):
        """Test unknown component type raises error."""
        with pytest.raises(ValueError):
            get_component_schema('unknown_type', 'test')


class TestExtractParameterSchema:
    """Test extract_parameter_schema function."""

    def test_silero_vad_parameters(self):
        """Test extracting SileroVADOptions parameters."""
        params = extract_parameter_schema(SileroVADOptions)

        param_names = [p['name'] for p in params]
        assert 'threshold' in param_names
        assert 'min_speech_duration_ms' in param_names
        assert 'speech_pad_ms' in param_names

    def test_stable_ts_engine_parameters(self):
        """Test extracting StableTSEngineOptions parameters (29 params)."""
        params = extract_parameter_schema(StableTSEngineOptions)

        param_names = [p['name'] for p in params]
        assert 'regroup' in param_names
        assert 'suppress_silence' in param_names
        assert 'q_levels' in param_names
        assert len(params) >= 20

    def test_type_extraction(self):
        """Test type annotations are extracted correctly."""
        params = extract_parameter_schema(SileroVADOptions)

        # Find threshold (float)
        threshold = next(p for p in params if p['name'] == 'threshold')
        assert 'float' in threshold['type']

        # Find min_speech_duration_ms (int)
        min_speech = next(p for p in params if p['name'] == 'min_speech_duration_ms')
        assert 'int' in min_speech['type']

    def test_defaults_extracted(self):
        """Test default values are extracted."""
        params = extract_parameter_schema(StableTSEngineOptions)

        # Find regroup (default=True)
        regroup = next(p for p in params if p['name'] == 'regroup')
        assert regroup.get('default') is True

        # Find q_levels (default=20)
        q_levels = next(p for p in params if p['name'] == 'q_levels')
        assert q_levels.get('default') == 20


class TestGetComponentDefaults:
    """Test get_component_defaults function."""

    def test_vad_defaults(self):
        """Test getting VAD component defaults."""
        defaults = get_component_defaults('vad', 'stable_ts_vad')

        # StableTSVADOptions doesn't have defaults on required fields
        # but the function should return empty dict or only optional defaults
        assert isinstance(defaults, dict)

    def test_asr_defaults(self):
        """Test getting ASR component defaults."""
        defaults = get_component_defaults('asr', 'stable_ts')

        # StableTSEngineOptions has many defaults
        assert defaults.get('regroup') is True
        assert defaults.get('suppress_silence') is True
        assert defaults.get('q_levels') == 20
        assert defaults.get('k_size') == 5

    def test_scene_detection_defaults(self):
        """Test getting scene detection defaults."""
        defaults = get_component_defaults('scene_detection', 'auditok')

        assert defaults.get('max_duration_s') == 29.0
        assert defaults.get('target_sr') == 16000
        assert defaults.get('force_mono') is True


class TestListFunctions:
    """Test list helper functions."""

    def test_list_pipelines(self):
        """Test list_pipelines returns all pipelines."""
        pipelines = list_pipelines()

        assert 'faster' in pipelines
        assert 'fast' in pipelines
        assert 'fidelity' in pipelines
        assert 'balanced' in pipelines

    def test_list_sensitivities(self):
        """Test list_sensitivities returns all levels."""
        sensitivities = list_sensitivities()

        assert 'conservative' in sensitivities
        assert 'balanced' in sensitivities
        assert 'aggressive' in sensitivities
