"""
Integration Tests - Compare resolve_config() with TranscriptionTuner.

These tests verify backward compatibility by comparing output from the
new v2.0 resolver with the existing TranscriptionTuner.
"""

import pytest

from whisperjav.config.resolver import resolve_config
from whisperjav.config.transcription_tuner import TranscriptionTuner


# All test combinations
PIPELINES = ["faster", "fast", "fidelity", "balanced"]
SENSITIVITIES = ["conservative", "balanced", "aggressive"]
TASKS = ["transcribe", "translate"]


@pytest.fixture
def tuner():
    """Get TranscriptionTuner instance."""
    return TranscriptionTuner()


class TestOutputStructure:
    """Test output structure matches between old and new systems."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_top_level_keys_match(self, tuner, pipeline, sensitivity):
        """Test top-level keys are identical."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        old_keys = set(old.keys())
        new_keys = set(new.keys())

        assert old_keys == new_keys, f"Keys differ: old={old_keys}, new={new_keys}"

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_params_keys_match(self, tuner, pipeline):
        """Test params sub-keys match."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        old_param_keys = set(old['params'].keys())
        new_param_keys = set(new['params'].keys())

        assert old_param_keys == new_param_keys


class TestDecoderParamsMatch:
    """Test decoder parameters match between systems."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    @pytest.mark.parametrize("task", TASKS)
    def test_decoder_task_matches(self, tuner, pipeline, sensitivity, task):
        """Test decoder task is set correctly."""
        old = tuner.resolve_params(pipeline, sensitivity, task)
        new = resolve_config(pipeline, sensitivity, task)

        assert old['params']['decoder']['task'] == new['params']['decoder']['task']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_decoder_language_matches(self, tuner, pipeline, sensitivity):
        """Test decoder language is set correctly."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        assert old['params']['decoder']['language'] == new['params']['decoder']['language']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_decoder_beam_size_matches(self, tuner, pipeline, sensitivity):
        """Test decoder beam_size matches."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        assert old['params']['decoder']['beam_size'] == new['params']['decoder']['beam_size']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_decoder_patience_matches(self, tuner, pipeline, sensitivity):
        """Test decoder patience matches."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        old_patience = old['params']['decoder'].get('patience')
        new_patience = new['params']['decoder'].get('patience')

        if old_patience is not None and new_patience is not None:
            assert abs(old_patience - new_patience) < 0.01


class TestProviderParamsMatch:
    """Test provider parameters match between systems."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_temperature_matches(self, tuner, pipeline, sensitivity):
        """Test temperature matches."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        old_temp = old['params']['provider'].get('temperature')
        new_temp = new['params']['provider'].get('temperature')

        assert old_temp == new_temp

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    def test_no_speech_threshold_matches(self, tuner, pipeline, sensitivity):
        """Test no_speech_threshold matches."""
        old = tuner.resolve_params(pipeline, sensitivity, "transcribe")
        new = resolve_config(pipeline, sensitivity, "transcribe")

        old_val = old['params']['provider'].get('no_speech_threshold')
        new_val = new['params']['provider'].get('no_speech_threshold')

        if old_val is not None and new_val is not None:
            assert abs(old_val - new_val) < 0.01


class TestWorkflowMatch:
    """Test workflow configuration matches."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_workflow_model_matches(self, tuner, pipeline):
        """Test workflow model matches."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        assert old['workflow']['model'] == new['workflow']['model']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_workflow_backend_matches(self, tuner, pipeline):
        """Test workflow backend matches."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        assert old['workflow']['backend'] == new['workflow']['backend']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_workflow_vad_matches(self, tuner, pipeline):
        """Test workflow vad matches."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        assert old['workflow']['vad'] == new['workflow']['vad']


class TestModelMatch:
    """Test model configuration matches."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_model_provider_matches(self, tuner, pipeline):
        """Test model provider matches."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        assert old['model']['provider'] == new['model']['provider']

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_model_name_matches(self, tuner, pipeline):
        """Test model name matches."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        assert old['model']['model_name'] == new['model']['model_name']


class TestFeaturesMatch:
    """Test feature configurations match."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_features_keys_match(self, tuner, pipeline):
        """Test feature keys match."""
        old = tuner.resolve_params(pipeline, "balanced", "transcribe")
        new = resolve_config(pipeline, "balanced", "transcribe")

        old_features = set(old.get('features', {}).keys())
        new_features = set(new.get('features', {}).keys())

        assert old_features == new_features

    def test_scene_detection_method_matches(self, tuner):
        """Test scene detection method matches."""
        old = tuner.resolve_params("balanced", "balanced", "transcribe")
        new = resolve_config("balanced", "balanced", "transcribe")

        if 'scene_detection' in old.get('features', {}):
            old_method = old['features']['scene_detection'].get('method')
            new_method = new['features']['scene_detection'].get('method')
            assert old_method == new_method


class TestTypeConversionsMatch:
    """Test type conversions are applied correctly."""

    @pytest.mark.parametrize("pipeline", ["faster", "fast"])
    def test_stable_ts_beam_size_is_int(self, pipeline):
        """Test beam_size is int for stable-ts pipelines."""
        result = resolve_config(pipeline, "balanced", "transcribe")
        beam_size = result['params']['decoder']['beam_size']
        assert isinstance(beam_size, int)

    @pytest.mark.parametrize("pipeline", PIPELINES)
    def test_patience_is_float(self, pipeline):
        """Test patience is float."""
        result = resolve_config(pipeline, "balanced", "transcribe")
        patience = result['params']['decoder'].get('patience')
        if patience is not None:
            assert isinstance(patience, float)


class TestAllCombinations:
    """Test all 24 combinations work without errors."""

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    @pytest.mark.parametrize("task", TASKS)
    def test_combination_resolves(self, tuner, pipeline, sensitivity, task):
        """Test each combination can be resolved."""
        old = tuner.resolve_params(pipeline, sensitivity, task)
        new = resolve_config(pipeline, sensitivity, task)

        # Both should return valid results
        assert old is not None
        assert new is not None

        # Both should have params
        assert 'params' in old
        assert 'params' in new

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    @pytest.mark.parametrize("task", TASKS)
    def test_pipeline_name_matches(self, tuner, pipeline, sensitivity, task):
        """Test pipeline_name matches in output."""
        old = tuner.resolve_params(pipeline, sensitivity, task)
        new = resolve_config(pipeline, sensitivity, task)

        assert old['pipeline_name'] == new['pipeline_name'] == pipeline

    @pytest.mark.parametrize("pipeline", PIPELINES)
    @pytest.mark.parametrize("sensitivity", SENSITIVITIES)
    @pytest.mark.parametrize("task", TASKS)
    def test_sensitivity_name_matches(self, tuner, pipeline, sensitivity, task):
        """Test sensitivity_name matches in output."""
        old = tuner.resolve_params(pipeline, sensitivity, task)
        new = resolve_config(pipeline, sensitivity, task)

        assert old['sensitivity_name'] == new['sensitivity_name'] == sensitivity
