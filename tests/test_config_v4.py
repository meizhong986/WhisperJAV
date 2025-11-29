"""
Comprehensive Tests for WhisperJAV Configuration System v4.

Tests cover:
- Schema validation (Pydantic models)
- YAML loading and parsing
- Config merging and inheritance
- Registry operations
- GUI API
- Error handling
"""

import pytest
from pathlib import Path
from typing import Dict, Any

# Import v4 modules
from whisperjav.config.v4 import (
    ConfigManager,
    V4ConfigError,
    SchemaValidationError,
    ModelNotFoundError,
    EcosystemNotFoundError,
    ToolNotFoundError,
    YAMLParseError,
    IncompatibleVersionError,
)
from whisperjav.config.v4.schemas import (
    ConfigBase,
    MetadataBlock,
    ModelConfig,
    EcosystemConfig,
    ToolConfig,
    PresetConfig,
    GUIWidget,
    GUIHint,
)
from whisperjav.config.v4.loaders import (
    YAMLLoader,
    ConfigMerger,
    deep_merge,
    load_yaml_string,
)
from whisperjav.config.v4.loaders.merger import (
    apply_overrides,
    flatten_dict,
    unflatten_dict,
    MergeStrategy,
)
from whisperjav.config.v4.gui_api import GUIAPI


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_model_yaml() -> str:
    """Sample model YAML for testing."""
    return """
schemaVersion: v1
kind: Model

metadata:
  name: test-model
  ecosystem: test-ecosystem
  displayName: "Test Model"
  description: "A test model for unit tests"
  version: "1.0.0"
  tags:
    - test
    - unit-test

spec:
  model.id: "test/model-id"
  model.device: auto
  decode.beam_size: 5
  decode.temperature: 0.0
  quality.no_speech: 0.6

presets:
  conservative:
    decode.beam_size: 3
  balanced: {}
  aggressive:
    decode.beam_size: 7

gui:
  model.device:
    widget: dropdown
    options:
      - auto
      - cuda
      - cpu
"""


@pytest.fixture
def sample_tool_yaml() -> str:
    """Sample tool YAML for testing."""
    return """
schemaVersion: v1
kind: Tool

metadata:
  name: test-tool
  displayName: "Test Tool"
  version: "1.0.0"
  tags:
    - test

tool_type: test_type

contract:
  input:
    audio_path: path
  output:
    segments: list

spec:
  threshold: 0.5
  min_duration: 1.0

presets:
  conservative:
    threshold: 0.7
  balanced: {}
  aggressive:
    threshold: 0.3
"""


@pytest.fixture
def sample_ecosystem_yaml() -> str:
    """Sample ecosystem YAML for testing."""
    return """
schemaVersion: v1
kind: Ecosystem

metadata:
  name: test-ecosystem
  displayName: "Test Ecosystem"
  version: "1.0.0"
  tags:
    - test

defaults:
  model.device: cuda
  decode.language: ja

provider:
  module: whisperjav.modules.test_module
  class: TestASR

compatible_tools:
  - test-tool
"""


@pytest.fixture
def v4_ecosystems_path() -> Path:
    """Path to v4 ecosystems directory."""
    return Path(__file__).parent.parent / "whisperjav" / "config" / "v4" / "ecosystems"


# =============================================================================
# Schema Tests
# =============================================================================


class TestMetadataBlock:
    """Tests for MetadataBlock schema."""

    def test_valid_metadata(self):
        """Test valid metadata creation."""
        metadata = MetadataBlock(
            name="test-model",
            displayName="Test Model",
            description="A test model",
            version="1.0.0",
            tags=["test", "unit"],
        )
        assert metadata.name == "test-model"
        assert metadata.displayName == "Test Model"

    def test_name_validation(self):
        """Test name must be valid identifier."""
        with pytest.raises(ValueError, match="must start with letter"):
            MetadataBlock(
                name="123-invalid",
                displayName="Invalid",
            )

    def test_tags_lowercase(self):
        """Test tags are lowercased."""
        metadata = MetadataBlock(
            name="test",
            displayName="Test",
            tags=["TEST", "UnitTest"],
        )
        assert metadata.tags == ["test", "unittest"]


class TestModelConfig:
    """Tests for ModelConfig schema."""

    def test_load_from_yaml_string(self, sample_model_yaml):
        """Test loading model from YAML string."""
        model = load_yaml_string(sample_model_yaml, ModelConfig)

        assert model.metadata.name == "test-model"
        assert model.spec["model.id"] == "test/model-id"
        assert "balanced" in model.presets

    def test_get_resolved_config(self, sample_model_yaml):
        """Test config resolution with presets."""
        model = load_yaml_string(sample_model_yaml, ModelConfig)

        # Balanced uses defaults
        balanced = model.get_resolved_config("balanced")
        assert balanced["decode.beam_size"] == 5

        # Conservative overrides
        conservative = model.get_resolved_config("conservative")
        assert conservative["decode.beam_size"] == 3

        # Aggressive overrides
        aggressive = model.get_resolved_config("aggressive")
        assert aggressive["decode.beam_size"] == 7

    def test_get_resolved_config_with_overrides(self, sample_model_yaml):
        """Test config resolution with user overrides."""
        model = load_yaml_string(sample_model_yaml, ModelConfig)

        config = model.get_resolved_config(
            "balanced",
            overrides={"decode.beam_size": 10}
        )
        assert config["decode.beam_size"] == 10

    def test_get_gui_schema(self, sample_model_yaml):
        """Test GUI schema generation."""
        model = load_yaml_string(sample_model_yaml, ModelConfig)
        schema = model.get_gui_schema()

        assert schema["model_name"] == "test-model"
        assert "parameters" in schema
        assert "groups" in schema
        assert "presets" in schema

    def test_spec_key_validation(self):
        """Test spec keys must be valid."""
        with pytest.raises(ValueError, match="Invalid spec key"):
            ModelConfig(
                kind="Model",
                metadata=MetadataBlock(name="test", displayName="Test"),
                spec={"invalid..key": "value"},
            )


class TestToolConfig:
    """Tests for ToolConfig schema."""

    def test_load_from_yaml_string(self, sample_tool_yaml):
        """Test loading tool from YAML string."""
        tool = load_yaml_string(sample_tool_yaml, ToolConfig)

        assert tool.metadata.name == "test-tool"
        assert tool.tool_type == "test_type"
        assert "input" in tool.contract
        assert "output" in tool.contract

    def test_get_resolved_config(self, sample_tool_yaml):
        """Test tool config resolution."""
        tool = load_yaml_string(sample_tool_yaml, ToolConfig)

        conservative = tool.get_resolved_config("conservative")
        assert conservative["threshold"] == 0.7


class TestEcosystemConfig:
    """Tests for EcosystemConfig schema."""

    def test_load_from_yaml_string(self, sample_ecosystem_yaml):
        """Test loading ecosystem from YAML string."""
        ecosystem = load_yaml_string(sample_ecosystem_yaml, EcosystemConfig)

        assert ecosystem.metadata.name == "test-ecosystem"
        assert ecosystem.defaults["model.device"] == "cuda"
        assert "test-tool" in ecosystem.compatible_tools


# =============================================================================
# Loader Tests
# =============================================================================


class TestYAMLLoader:
    """Tests for YAML loader."""

    def test_load_model_file(self, v4_ecosystems_path):
        """Test loading actual model file."""
        loader = YAMLLoader(v4_ecosystems_path)
        model_path = v4_ecosystems_path / "transformers" / "models" / "kotoba-whisper-v2.yaml"

        if model_path.exists():
            model = loader.load_model(model_path)
            assert model.metadata.name == "kotoba-whisper-v2"

    def test_parse_error_handling(self):
        """Test YAML parse error handling."""
        invalid_yaml = """
schemaVersion: v1
kind: Model
metadata:
  name: test
  displayName: Test
spec:
  invalid: {unclosed
"""
        with pytest.raises(YAMLParseError):
            load_yaml_string(invalid_yaml, ModelConfig)

    def test_schema_validation_error(self):
        """Test schema validation error."""
        invalid_yaml = """
schemaVersion: v1
kind: Model
metadata:
  name: "123-invalid"
  displayName: "Test"
"""
        with pytest.raises(SchemaValidationError):
            load_yaml_string(invalid_yaml, ModelConfig)

    def test_incompatible_version(self):
        """Test incompatible schema version."""
        future_yaml = """
schemaVersion: v99
kind: Model
metadata:
  name: test
  displayName: Test
"""
        with pytest.raises((SchemaValidationError, IncompatibleVersionError)):
            load_yaml_string(future_yaml, ModelConfig)


class TestConfigMerger:
    """Tests for config merging."""

    def test_strategic_merge(self):
        """Test strategic merge (default)."""
        base = {
            "a": 1,
            "nested": {"x": 10, "y": 20},
            "list": [1, 2, 3],
        }
        override = {
            "a": 2,
            "nested": {"x": 100},
            "list": [4, 5],
        }

        result = deep_merge(base, override)

        assert result["a"] == 2
        assert result["nested"]["x"] == 100
        assert result["nested"]["y"] == 20  # Preserved from base
        assert result["list"] == [4, 5]  # List replaced

    def test_apply_overrides(self):
        """Test applying dot-notation overrides."""
        config = {
            "model": {"device": "cpu"},
            "decode": {"beam_size": 5},
        }
        overrides = {
            "model.device": "cuda",
            "decode.beam_size": 10,
        }

        result = apply_overrides(config, overrides)

        assert result["model"]["device"] == "cuda"
        assert result["decode"]["beam_size"] == 10

    def test_flatten_unflatten(self):
        """Test dict flattening and unflattening."""
        nested = {
            "model": {"id": "test", "device": "cuda"},
            "decode": {"beam_size": 5},
        }

        flat = flatten_dict(nested)
        assert flat["model.id"] == "test"
        assert flat["decode.beam_size"] == 5

        restored = unflatten_dict(flat)
        assert restored == nested


# =============================================================================
# Manager Tests
# =============================================================================


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_initialization(self, v4_ecosystems_path):
        """Test manager initialization."""
        if v4_ecosystems_path.exists():
            manager = ConfigManager(v4_ecosystems_path)
            assert manager is not None

    def test_list_ecosystems(self, v4_ecosystems_path):
        """Test listing ecosystems."""
        if v4_ecosystems_path.exists():
            manager = ConfigManager(v4_ecosystems_path)
            ecosystems = manager.list_ecosystems()
            # Should have at least transformers
            assert isinstance(ecosystems, list)

    def test_list_models(self, v4_ecosystems_path):
        """Test listing models."""
        if v4_ecosystems_path.exists():
            manager = ConfigManager(v4_ecosystems_path)
            models = manager.list_models()
            assert isinstance(models, list)

    def test_get_model_config(self, v4_ecosystems_path):
        """Test getting resolved model config."""
        if v4_ecosystems_path.exists():
            manager = ConfigManager(v4_ecosystems_path)
            models = manager.list_models()

            if models:
                config = manager.get_model_config(models[0])
                assert isinstance(config, dict)


# =============================================================================
# GUI API Tests
# =============================================================================


class TestGUIAPI:
    """Tests for GUI API."""

    def test_initialization(self, v4_ecosystems_path):
        """Test GUI API initialization."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            assert api is not None

    def test_get_ecosystems_summary(self, v4_ecosystems_path):
        """Test getting ecosystems summary."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            summary = api.get_ecosystems_summary()

            assert "ecosystems" in summary
            assert "totalModels" in summary
            assert "totalEcosystems" in summary

    def test_get_models_list(self, v4_ecosystems_path):
        """Test getting models list."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            result = api.get_models_list()

            assert "models" in result
            assert isinstance(result["models"], list)

    def test_get_presets_list(self, v4_ecosystems_path):
        """Test getting presets list."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            result = api.get_presets_list()

            assert "presets" in result

    def test_get_model_schema(self, v4_ecosystems_path):
        """Test getting model GUI schema."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            models = api.get_models_list()["models"]

            if models:
                schema = api.get_model_schema(models[0]["name"])
                # Either valid schema or error dict
                assert "error" in schema or "parameters" in schema

    def test_model_not_found_error(self, v4_ecosystems_path):
        """Test model not found returns error dict."""
        if v4_ecosystems_path.exists():
            api = GUIAPI(v4_ecosystems_path)
            result = api.get_model_schema("non-existent-model")

            assert result.get("error") == True
            assert "errorType" in result


# =============================================================================
# Error Tests
# =============================================================================


class TestErrors:
    """Tests for custom error classes."""

    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError(
            model_name="test-model",
            available_models=["model-a", "model-b"],
        )
        assert "test-model" in str(error)
        assert error.model_name == "test-model"

    def test_schema_validation_error(self):
        """Test SchemaValidationError."""
        error = SchemaValidationError(
            message="Invalid value",
            field_path="model.device",
            actual_value="invalid",
            expected_type="str",
        )
        assert "model.device" in str(error)
        assert error.field_path == "model.device"

    def test_error_to_dict(self):
        """Test error serialization."""
        error = V4ConfigError(
            message="Test error",
            context={"key": "value"},
            suggestion="Fix it",
        )
        error_dict = error.to_dict()

        assert error_dict["message"] == "Test error"
        assert error_dict["context"] == {"key": "value"}
        assert error_dict["suggestion"] == "Fix it"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests using actual config files."""

    def test_full_workflow(self, v4_ecosystems_path):
        """Test complete workflow from discovery to config resolution."""
        if not v4_ecosystems_path.exists():
            pytest.skip("v4 ecosystems not found")

        # Initialize
        manager = ConfigManager(v4_ecosystems_path)

        # Discover
        ecosystems = manager.list_ecosystems()
        if not ecosystems:
            pytest.skip("No ecosystems found")

        models = manager.list_models()
        if not models:
            pytest.skip("No models found")

        # Get config
        model_name = models[0]
        config = manager.get_model_config(model_name, "balanced")
        assert isinstance(config, dict)

        # Apply overrides
        config_with_override = manager.get_model_config(
            model_name,
            "balanced",
            overrides={"decode.beam_size": 99}
        )
        assert config_with_override.get("decode.beam_size") == 99

    def test_gui_api_full_workflow(self, v4_ecosystems_path):
        """Test complete GUI API workflow."""
        if not v4_ecosystems_path.exists():
            pytest.skip("v4 ecosystems not found")

        api = GUIAPI(v4_ecosystems_path)

        # Get summary
        summary = api.get_ecosystems_summary()
        assert "ecosystems" in summary

        # Get models
        models = api.get_models_list()
        if not models["models"]:
            pytest.skip("No models found")

        # Get schema
        model_name = models["models"][0]["name"]
        schema = api.get_model_schema(model_name)

        if "error" not in schema:
            assert "parameters" in schema
            assert "groups" in schema

        # Get resolved config
        result = api.get_resolved_config(model_name, "balanced")
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
