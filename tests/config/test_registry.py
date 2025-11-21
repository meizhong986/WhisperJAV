"""
Tests for Component Registry.
"""

import pytest

from whisperjav.config.registry import (
    ComponentMeta,
    ComponentRegistry,
    get_registry,
    register_default_components,
)
from whisperjav.config.schemas import (
    FasterWhisperEngineOptions,
    SileroVADOptions,
    StableTSVADOptions,
)


@pytest.fixture
def fresh_registry():
    """Get a fresh registry for testing."""
    registry = get_registry()
    registry.reset()
    return registry


class TestComponentMeta:
    """Test ComponentMeta dataclass."""

    def test_basic_creation(self):
        """Test creating a component metadata."""
        meta = ComponentMeta(
            name="test_vad",
            display_name="Test VAD",
            description="A test VAD component",
            config_class=SileroVADOptions
        )
        assert meta.name == "test_vad"
        assert meta.display_name == "Test VAD"
        assert meta.config_class == SileroVADOptions
        assert meta.deprecated is False

    def test_with_optional_fields(self):
        """Test creating with all optional fields."""
        meta = ComponentMeta(
            name="old_vad",
            display_name="Old VAD",
            description="Deprecated VAD",
            config_class=SileroVADOptions,
            compatible_with=["whisper", "faster-whisper"],
            tags=["deprecated", "legacy"],
            version="0.9",
            deprecated=True,
            replacement="new_vad"
        )
        assert meta.deprecated is True
        assert meta.replacement == "new_vad"
        assert "whisper" in meta.compatible_with


class TestComponentRegistry:
    """Test ComponentRegistry singleton and operations."""

    def test_singleton_pattern(self):
        """Test that get_registry returns same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_register_vad(self, fresh_registry):
        """Test registering a VAD component."""
        meta = ComponentMeta(
            name="test_vad",
            display_name="Test VAD",
            description="Test",
            config_class=SileroVADOptions,
            compatible_with=["whisper"]
        )
        fresh_registry.register_vad(meta)

        assert len(fresh_registry.list_vads()) == 1
        assert fresh_registry.get_vad("test_vad") == meta

    def test_duplicate_registration_raises(self, fresh_registry):
        """Test that duplicate registration raises error."""
        meta = ComponentMeta(
            name="test_vad",
            display_name="Test VAD",
            description="Test",
            config_class=SileroVADOptions
        )
        fresh_registry.register_vad(meta)

        with pytest.raises(ValueError, match="already registered"):
            fresh_registry.register_vad(meta)

    def test_get_unknown_raises(self, fresh_registry):
        """Test that getting unknown component raises KeyError."""
        with pytest.raises(KeyError, match="Unknown VAD"):
            fresh_registry.get_vad("nonexistent")

    def test_list_excludes_deprecated(self, fresh_registry):
        """Test that list methods exclude deprecated by default."""
        active_meta = ComponentMeta(
            name="active_vad",
            display_name="Active VAD",
            description="Active",
            config_class=SileroVADOptions
        )
        deprecated_meta = ComponentMeta(
            name="old_vad",
            display_name="Old VAD",
            description="Old",
            config_class=SileroVADOptions,
            deprecated=True
        )

        fresh_registry.register_vad(active_meta)
        fresh_registry.register_vad(deprecated_meta)

        # Default excludes deprecated
        vads = fresh_registry.list_vads()
        assert len(vads) == 1
        assert vads[0].name == "active_vad"

        # Include deprecated
        all_vads = fresh_registry.list_vads(include_deprecated=True)
        assert len(all_vads) == 2

    def test_get_config_class(self, fresh_registry):
        """Test getting config class from registered component."""
        meta = ComponentMeta(
            name="test_vad",
            display_name="Test VAD",
            description="Test",
            config_class=SileroVADOptions
        )
        fresh_registry.register_vad(meta)

        config_class = fresh_registry.get_vad_config_class("test_vad")
        assert config_class == SileroVADOptions

    def test_compatibility_check(self, fresh_registry):
        """Test VAD-ASR compatibility checking."""
        # Register VAD compatible with whisper only
        vad_meta = ComponentMeta(
            name="whisper_only_vad",
            display_name="Whisper Only VAD",
            description="Test",
            config_class=SileroVADOptions,
            compatible_with=["whisper"]
        )
        fresh_registry.register_vad(vad_meta)

        # Check compatibility
        assert fresh_registry.is_compatible("whisper_only_vad", "whisper")
        assert not fresh_registry.is_compatible("whisper_only_vad", "faster-whisper")

    def test_get_compatible_vads(self, fresh_registry):
        """Test getting VADs compatible with an ASR."""
        # Register multiple VADs
        silero = ComponentMeta(
            name="silero",
            display_name="Silero",
            description="Test",
            config_class=SileroVADOptions,
            compatible_with=["whisper", "faster-whisper"]
        )
        stable_ts = ComponentMeta(
            name="stable_ts",
            display_name="Stable-TS VAD",
            description="Test",
            config_class=StableTSVADOptions,
            compatible_with=["stable-ts"]
        )
        fresh_registry.register_vad(silero)
        fresh_registry.register_vad(stable_ts)

        # Get compatible VADs for whisper
        compatible = fresh_registry.get_compatible_vads("whisper")
        assert len(compatible) == 1
        assert compatible[0].name == "silero"

        # Get compatible VADs for stable-ts
        compatible = fresh_registry.get_compatible_vads("stable-ts")
        assert len(compatible) == 1
        assert compatible[0].name == "stable_ts"

    def test_find_by_tag(self, fresh_registry):
        """Test finding components by tag."""
        meta1 = ComponentMeta(
            name="fast_vad",
            display_name="Fast VAD",
            description="Test",
            config_class=SileroVADOptions,
            tags=["fast", "simple"]
        )
        meta2 = ComponentMeta(
            name="accurate_vad",
            display_name="Accurate VAD",
            description="Test",
            config_class=SileroVADOptions,
            tags=["accurate", "complex"]
        )
        fresh_registry.register_vad(meta1)
        fresh_registry.register_vad(meta2)

        # Find by "fast" tag
        results = fresh_registry.find_by_tag("fast", "vad")
        assert len(results) == 1
        assert results[0].name == "fast_vad"


class TestDefaultRegistration:
    """Test default component registration."""

    def test_default_vads_registered(self):
        """Test all default VADs are registered."""
        registry = get_registry()

        # Re-register if needed
        if not registry._initialized:
            register_default_components()

        vads = registry.list_vads()
        vad_names = [v.name for v in vads]

        assert "silero_vad" in vad_names
        assert "stable_ts_vad" in vad_names
        assert "faster_whisper_vad" in vad_names

    def test_default_asr_registered(self):
        """Test all default ASR engines are registered."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        asr_list = registry.list_asr()
        asr_names = [a.name for a in asr_list]

        assert "faster_whisper" in asr_names
        assert "stable_ts" in asr_names
        assert "openai_whisper" in asr_names

    def test_default_scene_detection_registered(self):
        """Test all default scene detection methods are registered."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        sd_list = registry.list_scene_detection()
        sd_names = [s.name for s in sd_list]

        assert "auditok" in sd_names
        assert "silero" in sd_names

    def test_silero_vad_compatibility(self):
        """Test Silero VAD is compatible with whisper and faster-whisper."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        silero = registry.get_vad("silero_vad")
        assert "faster-whisper" in silero.compatible_with
        assert "whisper" in silero.compatible_with

    def test_stable_ts_vad_compatibility(self):
        """Test Stable-TS VAD is only compatible with stable-ts."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        stable_ts = registry.get_vad("stable_ts_vad")
        assert "stable-ts" in stable_ts.compatible_with
        assert "whisper" not in stable_ts.compatible_with

    def test_recommended_tags(self):
        """Test components have appropriate tags."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        # Silero VAD should be recommended for JAV
        silero = registry.get_vad("silero_vad")
        assert "recommended" in silero.tags
        assert "jav-optimized" in silero.tags

        # Stable-TS should be recommended for Japanese
        stable_ts = registry.get_asr("stable_ts")
        assert "japanese" in stable_ts.tags
        assert "recommended" in stable_ts.tags

    def test_config_classes_are_correct(self):
        """Test registered config classes are correct Pydantic models."""
        registry = get_registry()

        if not registry._initialized:
            register_default_components()

        # VAD config classes
        silero_class = registry.get_vad_config_class("silero_vad")
        assert silero_class == SileroVADOptions

        stable_ts_class = registry.get_vad_config_class("stable_ts_vad")
        assert stable_ts_class == StableTSVADOptions

        # ASR config classes
        fw_class = registry.get_asr_config_class("faster_whisper")
        assert fw_class == FasterWhisperEngineOptions
