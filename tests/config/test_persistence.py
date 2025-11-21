"""
Tests for Configuration Persistence.
"""

import pytest
import tempfile
from pathlib import Path

from whisperjav.config.persistence import (
    save_config,
    load_config,
    list_configs,
    delete_config,
    config_exists,
    get_config_dir,
)
from whisperjav.config.builder import PipelineBuilder
from whisperjav.config.errors import ConfigValidationError


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample configuration."""
    return PipelineBuilder("balanced").with_sensitivity("aggressive").build()


class TestGetConfigDir:
    """Test get_config_dir function."""

    def test_returns_path(self):
        """Test returns a Path object."""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)

    def test_creates_directory(self):
        """Test creates directory if not exists."""
        config_dir = get_config_dir()
        assert config_dir.exists()
        assert config_dir.is_dir()


class TestSaveConfig:
    """Test save_config function."""

    def test_saves_config(self, temp_dir, sample_config):
        """Test saving a configuration."""
        path = save_config(sample_config, "test_config", config_dir=temp_dir)

        assert path.exists()
        assert path.name == "test_config.json"

    def test_creates_valid_json(self, temp_dir, sample_config):
        """Test saved file is valid JSON."""
        save_config(sample_config, "test_config", config_dir=temp_dir)
        loaded = load_config("test_config", config_dir=temp_dir)

        assert loaded['pipeline_name'] == sample_config['pipeline_name']
        assert loaded['sensitivity_name'] == sample_config['sensitivity_name']

    def test_prevents_overwrite_by_default(self, temp_dir, sample_config):
        """Test prevents overwriting without flag."""
        save_config(sample_config, "test_config", config_dir=temp_dir)

        with pytest.raises(ConfigValidationError):
            save_config(sample_config, "test_config", config_dir=temp_dir)

    def test_allows_overwrite_with_flag(self, temp_dir, sample_config):
        """Test allows overwriting with flag."""
        save_config(sample_config, "test_config", config_dir=temp_dir)

        # Modify and save again
        sample_config['test_key'] = 'test_value'
        save_config(sample_config, "test_config", config_dir=temp_dir, overwrite=True)

        loaded = load_config("test_config", config_dir=temp_dir)
        assert loaded.get('test_key') == 'test_value'

    def test_invalid_name_raises(self, temp_dir, sample_config):
        """Test invalid name raises error."""
        with pytest.raises(ConfigValidationError):
            save_config(sample_config, "", config_dir=temp_dir)

        with pytest.raises(ConfigValidationError):
            save_config(sample_config, "invalid name!", config_dir=temp_dir)

    def test_valid_names(self, temp_dir, sample_config):
        """Test valid names are accepted."""
        # Alphanumeric
        save_config(sample_config, "config1", config_dir=temp_dir)
        # With underscores
        save_config(sample_config, "my_config", config_dir=temp_dir)
        # With hyphens
        save_config(sample_config, "my-config", config_dir=temp_dir)

        assert len(list_configs(config_dir=temp_dir)) == 3


class TestLoadConfig:
    """Test load_config function."""

    def test_loads_saved_config(self, temp_dir, sample_config):
        """Test loading a saved configuration."""
        save_config(sample_config, "test_config", config_dir=temp_dir)
        loaded = load_config("test_config", config_dir=temp_dir)

        assert loaded == sample_config

    def test_missing_config_raises(self, temp_dir):
        """Test missing configuration raises error."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent", config_dir=temp_dir)

    def test_preserves_all_data(self, temp_dir):
        """Test all configuration data is preserved."""
        config = (
            PipelineBuilder("fidelity")
            .with_sensitivity("conservative")
            .with_task("translate")
            .with_beam_size(3)
            .with_temperature(0.1)
            .build()
        )

        save_config(config, "full_config", config_dir=temp_dir)
        loaded = load_config("full_config", config_dir=temp_dir)

        assert loaded['pipeline_name'] == "fidelity"
        assert loaded['sensitivity_name'] == "conservative"
        assert loaded['task'] == "translate"
        assert loaded['params']['decoder']['beam_size'] == 3
        assert loaded['params']['provider']['temperature'] == 0.1


class TestListConfigs:
    """Test list_configs function."""

    def test_empty_directory(self, temp_dir):
        """Test empty directory returns empty list."""
        configs = list_configs(config_dir=temp_dir)
        assert configs == []

    def test_lists_all_configs(self, temp_dir, sample_config):
        """Test lists all saved configurations."""
        save_config(sample_config, "config1", config_dir=temp_dir)
        save_config(sample_config, "config2", config_dir=temp_dir)
        save_config(sample_config, "config3", config_dir=temp_dir)

        configs = list_configs(config_dir=temp_dir)
        assert len(configs) == 3
        assert "config1" in configs
        assert "config2" in configs
        assert "config3" in configs

    def test_returns_sorted_list(self, temp_dir, sample_config):
        """Test returns sorted list."""
        save_config(sample_config, "zebra", config_dir=temp_dir)
        save_config(sample_config, "alpha", config_dir=temp_dir)
        save_config(sample_config, "beta", config_dir=temp_dir)

        configs = list_configs(config_dir=temp_dir)
        assert configs == ["alpha", "beta", "zebra"]

    def test_ignores_non_json_files(self, temp_dir, sample_config):
        """Test ignores non-JSON files."""
        save_config(sample_config, "valid", config_dir=temp_dir)

        # Create non-JSON file
        (temp_dir / "readme.txt").write_text("test")

        configs = list_configs(config_dir=temp_dir)
        assert configs == ["valid"]


class TestDeleteConfig:
    """Test delete_config function."""

    def test_deletes_existing(self, temp_dir, sample_config):
        """Test deletes existing configuration."""
        save_config(sample_config, "test_config", config_dir=temp_dir)
        assert config_exists("test_config", config_dir=temp_dir)

        result = delete_config("test_config", config_dir=temp_dir)
        assert result is True
        assert not config_exists("test_config", config_dir=temp_dir)

    def test_returns_false_for_missing(self, temp_dir):
        """Test returns False for missing configuration."""
        result = delete_config("nonexistent", config_dir=temp_dir)
        assert result is False


class TestConfigExists:
    """Test config_exists function."""

    def test_returns_true_for_existing(self, temp_dir, sample_config):
        """Test returns True for existing configuration."""
        save_config(sample_config, "test_config", config_dir=temp_dir)
        assert config_exists("test_config", config_dir=temp_dir) is True

    def test_returns_false_for_missing(self, temp_dir):
        """Test returns False for missing configuration."""
        assert config_exists("nonexistent", config_dir=temp_dir) is False


class TestRoundTrip:
    """Test complete round-trip operations."""

    def test_save_load_delete_cycle(self, temp_dir):
        """Test complete save-load-delete cycle."""
        # Create config
        config = (
            PipelineBuilder("balanced")
            .with_sensitivity("aggressive")
            .with_beam_size(10)
            .build()
        )

        # Save
        save_config(config, "round_trip", config_dir=temp_dir)
        assert "round_trip" in list_configs(config_dir=temp_dir)

        # Load
        loaded = load_config("round_trip", config_dir=temp_dir)
        assert loaded['params']['decoder']['beam_size'] == 10

        # Delete
        delete_config("round_trip", config_dir=temp_dir)
        assert "round_trip" not in list_configs(config_dir=temp_dir)

    def test_multiple_configs(self, temp_dir):
        """Test managing multiple configurations."""
        configs = {
            "speed": PipelineBuilder("faster").with_sensitivity("aggressive").build(),
            "quality": PipelineBuilder("fidelity").with_sensitivity("conservative").build(),
            "balanced": PipelineBuilder("balanced").with_sensitivity("balanced").build(),
        }

        # Save all
        for name, config in configs.items():
            save_config(config, name, config_dir=temp_dir)

        # Verify all exist
        saved = list_configs(config_dir=temp_dir)
        assert len(saved) == 3

        # Load and verify each
        for name, original in configs.items():
            loaded = load_config(name, config_dir=temp_dir)
            assert loaded['pipeline_name'] == original['pipeline_name']
            assert loaded['sensitivity_name'] == original['sensitivity_name']
