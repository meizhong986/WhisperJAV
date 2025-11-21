"""
Tests for Component Base Classes and Registration.
"""

import pytest
from pydantic import BaseModel, Field

from whisperjav.config.components.base import (
    ASRComponent,
    VADComponent,
    FeatureComponent,
    register_asr,
    register_vad,
    register_feature,
    get_asr_registry,
    get_vad_registry,
    get_feature_registry,
)


class TestASRComponent:
    """Test ASRComponent base class."""

    def test_basic_definition(self):
        """Test defining a basic ASR component."""

        class TestOptions(BaseModel):
            beam_size: int = Field(5, ge=1, le=20, description="Beam size")
            temperature: float = Field(0.0, ge=0.0, le=1.0)

        @register_asr
        class TestASR(ASRComponent):
            name = "test_asr"
            display_name = "Test ASR"
            description = "Test ASR component"
            provider = "test_provider"
            model_id = "test-model"
            supported_tasks = ["transcribe", "translate"]
            compatible_vad = ["silero", "ten"]
            Options = TestOptions
            presets = {
                "conservative": TestOptions(beam_size=3, temperature=0.0),
                "balanced": TestOptions(beam_size=5, temperature=0.0),
                "aggressive": TestOptions(beam_size=8, temperature=0.0),
            }

        # Verify registration
        assert "test_asr" in get_asr_registry()
        assert get_asr_registry()["test_asr"] == TestASR

    def test_to_dict(self):
        """Test converting component to dictionary."""

        @register_asr
        class DictTestASR(ASRComponent):
            name = "dict_test_asr"
            display_name = "Dict Test ASR"
            description = "For testing to_dict"
            provider = "test"
            model_id = "test"
            supported_tasks = ["transcribe"]
            compatible_vad = ["silero"]

        result = DictTestASR.to_dict()

        assert result["name"] == "dict_test_asr"
        assert result["display_name"] == "Dict Test ASR"
        assert result["provider"] == "test"
        assert result["supported_tasks"] == ["transcribe"]
        assert result["compatible_vad"] == ["silero"]

    def test_get_schema(self):
        """Test getting parameter schema."""

        class SchemaOptions(BaseModel):
            threshold: float = Field(0.5, ge=0.0, le=1.0, description="Threshold value")
            count: int = Field(10, ge=1, description="Count value")

        @register_asr
        class SchemaTestASR(ASRComponent):
            name = "schema_test_asr"
            display_name = "Schema Test"
            description = "Test"
            provider = "test"
            Options = SchemaOptions

        schema = SchemaTestASR.get_schema()

        assert "parameters" in schema
        params = schema["parameters"]
        assert len(params) == 2

        threshold_param = next(p for p in params if p["name"] == "threshold")
        assert threshold_param["type"] == "float"
        assert threshold_param["default"] == 0.5
        assert threshold_param["description"] == "Threshold value"
        assert threshold_param["constraints"]["ge"] == 0.0
        assert threshold_param["constraints"]["le"] == 1.0

    def test_get_preset(self):
        """Test getting presets."""

        class PresetOptions(BaseModel):
            value: int = 5

        @register_asr
        class PresetTestASR(ASRComponent):
            name = "preset_test_asr"
            display_name = "Preset Test"
            description = "Test"
            provider = "test"
            Options = PresetOptions
            presets = {
                "conservative": PresetOptions(value=3),
                "balanced": PresetOptions(value=5),
                "aggressive": PresetOptions(value=8),
            }

        assert PresetTestASR.get_preset("conservative").value == 3
        assert PresetTestASR.get_preset("balanced").value == 5
        assert PresetTestASR.get_preset("aggressive").value == 8
        assert PresetTestASR.get_preset("unknown") is None


class TestVADComponent:
    """Test VADComponent base class."""

    def test_basic_definition(self):
        """Test defining a basic VAD component."""

        class VADOptions(BaseModel):
            threshold: float = Field(0.5, ge=0.0, le=1.0)

        @register_vad
        class TestVAD(VADComponent):
            name = "test_vad"
            display_name = "Test VAD"
            description = "Test VAD component"
            compatible_asr = ["whisper", "kandi"]
            Options = VADOptions
            presets = {
                "balanced": VADOptions(threshold=0.5),
            }

        assert "test_vad" in get_vad_registry()

    def test_to_dict(self):
        """Test VAD to_dict includes compatible_asr."""

        @register_vad
        class DictTestVAD(VADComponent):
            name = "dict_test_vad"
            display_name = "Dict Test"
            description = "Test"
            compatible_asr = ["whisper", "kandi"]

        result = DictTestVAD.to_dict()
        assert result["compatible_asr"] == ["whisper", "kandi"]


class TestFeatureComponent:
    """Test FeatureComponent base class."""

    def test_basic_definition(self):
        """Test defining a feature component."""

        class FeatureOptions(BaseModel):
            enabled: bool = True

        @register_feature
        class TestFeature(FeatureComponent):
            name = "test_feature"
            display_name = "Test Feature"
            description = "Test feature component"
            feature_type = "scene_detection"
            Options = FeatureOptions

        assert "test_feature" in get_feature_registry()

    def test_to_dict(self):
        """Test Feature to_dict includes feature_type."""

        @register_feature
        class DictTestFeature(FeatureComponent):
            name = "dict_test_feature"
            display_name = "Dict Test"
            description = "Test"
            feature_type = "post_processing"

        result = DictTestFeature.to_dict()
        assert result["feature_type"] == "post_processing"


class TestRegistration:
    """Test component registration system."""

    def test_registration_without_name_raises(self):
        """Test that registration without name raises error."""
        with pytest.raises(ValueError, match="must have a 'name' attribute"):
            @register_asr
            class NoNameASR(ASRComponent):
                display_name = "No Name"

    def test_duplicate_registration_overwrites(self):
        """Test that duplicate registration overwrites."""

        @register_vad
        class DupeVAD1(VADComponent):
            name = "dupe_vad"
            display_name = "First"
            description = "First"

        @register_vad
        class DupeVAD2(VADComponent):
            name = "dupe_vad"
            display_name = "Second"
            description = "Second"

        # Second registration wins
        assert get_vad_registry()["dupe_vad"].display_name == "Second"


class TestGetDefaults:
    """Test getting default values."""

    def test_get_defaults(self):
        """Test getting default values from Options."""

        class DefaultOptions(BaseModel):
            a: int = 1
            b: float = 2.5
            c: str = "default"

        @register_asr
        class DefaultTestASR(ASRComponent):
            name = "default_test_asr"
            display_name = "Default Test"
            description = "Test"
            provider = "test"
            Options = DefaultOptions

        defaults = DefaultTestASR.get_defaults()

        assert defaults["a"] == 1
        assert defaults["b"] == 2.5
        assert defaults["c"] == "default"

    def test_no_options_returns_empty(self):
        """Test component without Options returns empty defaults."""

        @register_asr
        class NoOptionsASR(ASRComponent):
            name = "no_options_asr"
            display_name = "No Options"
            description = "Test"
            provider = "test"

        defaults = NoOptionsASR.get_defaults()
        assert defaults == {}
