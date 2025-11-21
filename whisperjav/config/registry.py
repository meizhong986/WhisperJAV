"""
Component Registry for WhisperJAV Configuration System v2.0.

Provides a central registry for all configurable components (VAD, ASR, scene detection).
Enables component discovery, compatibility checking, and future GUI integration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type

from pydantic import BaseModel


@dataclass
class ComponentMeta:
    """Complete metadata for a registered component."""

    name: str                              # Unique ID: "silero_vad"
    display_name: str                      # UI name: "Silero VAD"
    description: str                       # Full description
    config_class: Type[BaseModel]          # Pydantic model
    compatible_with: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    version: str = "1.0"
    deprecated: bool = False
    replacement: Optional[str] = None      # If deprecated, what to use instead


class ComponentRegistry:
    """
    Central registry for all configurable components.

    Singleton pattern ensures consistent state across the application.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize empty registries."""
        self._vad: Dict[str, ComponentMeta] = {}
        self._asr: Dict[str, ComponentMeta] = {}
        self._scene_detection: Dict[str, ComponentMeta] = {}
        self._post_processor: Dict[str, ComponentMeta] = {}
        self._models: Dict[str, ComponentMeta] = {}
        self._initialized = False

    def reset(self):
        """Reset registry to empty state. Useful for testing."""
        self._initialize()

    # Registration methods
    def register_vad(self, meta: ComponentMeta) -> None:
        """Register a VAD component."""
        if meta.name in self._vad:
            raise ValueError(f"VAD '{meta.name}' already registered")
        self._vad[meta.name] = meta

    def register_asr(self, meta: ComponentMeta) -> None:
        """Register an ASR engine component."""
        if meta.name in self._asr:
            raise ValueError(f"ASR '{meta.name}' already registered")
        self._asr[meta.name] = meta

    def register_scene_detection(self, meta: ComponentMeta) -> None:
        """Register a scene detection component."""
        if meta.name in self._scene_detection:
            raise ValueError(f"Scene detection '{meta.name}' already registered")
        self._scene_detection[meta.name] = meta

    def register_post_processor(self, meta: ComponentMeta) -> None:
        """Register a post-processing component."""
        if meta.name in self._post_processor:
            raise ValueError(f"Post-processor '{meta.name}' already registered")
        self._post_processor[meta.name] = meta

    # Query methods - list
    def list_vads(self, include_deprecated: bool = False) -> List[ComponentMeta]:
        """List all registered VAD components."""
        return [m for m in self._vad.values()
                if include_deprecated or not m.deprecated]

    def list_asr(self, include_deprecated: bool = False) -> List[ComponentMeta]:
        """List all registered ASR components."""
        return [m for m in self._asr.values()
                if include_deprecated or not m.deprecated]

    def list_scene_detection(self, include_deprecated: bool = False) -> List[ComponentMeta]:
        """List all registered scene detection components."""
        return [m for m in self._scene_detection.values()
                if include_deprecated or not m.deprecated]

    # Query methods - get single
    def get_vad(self, name: str) -> ComponentMeta:
        """Get a VAD component by name."""
        if name not in self._vad:
            available = list(self._vad.keys())
            raise KeyError(f"Unknown VAD: '{name}'. Available: {available}")
        return self._vad[name]

    def get_asr(self, name: str) -> ComponentMeta:
        """Get an ASR component by name."""
        if name not in self._asr:
            available = list(self._asr.keys())
            raise KeyError(f"Unknown ASR: '{name}'. Available: {available}")
        return self._asr[name]

    def get_scene_detection(self, name: str) -> ComponentMeta:
        """Get a scene detection component by name."""
        if name not in self._scene_detection:
            available = list(self._scene_detection.keys())
            raise KeyError(f"Unknown scene detection: '{name}'. Available: {available}")
        return self._scene_detection[name]

    # Config class helpers
    def get_vad_config_class(self, name: str) -> Type[BaseModel]:
        """Get the Pydantic config class for a VAD."""
        return self.get_vad(name).config_class

    def get_asr_config_class(self, name: str) -> Type[BaseModel]:
        """Get the Pydantic config class for an ASR engine."""
        return self.get_asr(name).config_class

    def get_scene_detection_config_class(self, name: str) -> Type[BaseModel]:
        """Get the Pydantic config class for scene detection."""
        return self.get_scene_detection(name).config_class

    # Compatibility checks
    def get_compatible_vads(self, asr_name: str) -> List[ComponentMeta]:
        """Return VADs compatible with the given ASR."""
        return [m for m in self._vad.values()
                if asr_name in m.compatible_with and not m.deprecated]

    def is_compatible(self, vad_name: str, asr_name: str) -> bool:
        """Check if a VAD is compatible with an ASR."""
        if vad_name not in self._vad:
            return False
        vad = self._vad[vad_name]
        return asr_name in vad.compatible_with

    # Search by tags
    def find_by_tag(self, tag: str, component_type: str = "all") -> List[ComponentMeta]:
        """Find components by tag."""
        results = []

        if component_type in ("all", "vad"):
            results.extend([m for m in self._vad.values() if tag in m.tags])
        if component_type in ("all", "asr"):
            results.extend([m for m in self._asr.values() if tag in m.tags])
        if component_type in ("all", "scene_detection"):
            results.extend([m for m in self._scene_detection.values() if tag in m.tags])

        return results


def get_registry() -> ComponentRegistry:
    """Get the singleton registry instance."""
    return ComponentRegistry()


def register_default_components():
    """Register all built-in components."""

    registry = get_registry()

    # Skip if already initialized
    if registry._initialized:
        return

    # Import schema classes
    from .schemas import (
        AuditokSceneDetectionConfig,
        FasterWhisperEngineOptions,
        FasterWhisperVADOptions,
        OpenAIWhisperEngineOptions,
        SileroSceneDetectionConfig,
        SileroVADOptions,
        StableTSEngineOptions,
        StableTSVADOptions,
    )

    # Register VAD components
    registry.register_vad(ComponentMeta(
        name="silero_vad",
        display_name="Silero VAD",
        description="Neural network VAD with excellent speech/music separation. "
                    "Best choice for JAV content with complex audio.",
        config_class=SileroVADOptions,
        compatible_with=["faster-whisper", "whisper"],
        tags=["neural", "accurate", "recommended", "jav-optimized"]
    ))

    registry.register_vad(ComponentMeta(
        name="stable_ts_vad",
        display_name="Stable-TS VAD",
        description="Integrated VAD for stable-ts backend. "
                    "Uses vad_threshold parameter.",
        config_class=StableTSVADOptions,
        compatible_with=["stable-ts"],
        tags=["integrated", "simple"]
    ))

    registry.register_vad(ComponentMeta(
        name="faster_whisper_vad",
        display_name="Faster Whisper VAD",
        description="Built-in VAD for faster-whisper. Minimal configuration.",
        config_class=FasterWhisperVADOptions,
        compatible_with=["faster-whisper"],
        tags=["simple", "fast"]
    ))

    # Register ASR engines
    registry.register_asr(ComponentMeta(
        name="faster_whisper",
        display_name="Faster Whisper",
        description="CTranslate2-optimized Whisper with fast inference. "
                    "Good balance of speed and accuracy.",
        config_class=FasterWhisperEngineOptions,
        tags=["fast", "efficient", "ctranslate2"]
    ))

    registry.register_asr(ComponentMeta(
        name="stable_ts",
        display_name="Stable-TS",
        description="Enhanced Whisper with Japanese regrouping and advanced "
                    "timestamp alignment. Best accuracy for dialogue.",
        config_class=StableTSEngineOptions,
        tags=["accurate", "japanese", "timestamps", "recommended"]
    ))

    registry.register_asr(ComponentMeta(
        name="openai_whisper",
        display_name="OpenAI Whisper",
        description="Original OpenAI Whisper implementation. "
                    "Maximum compatibility.",
        config_class=OpenAIWhisperEngineOptions,
        tags=["original", "compatible"]
    ))

    # Register scene detection methods
    registry.register_scene_detection(ComponentMeta(
        name="auditok",
        display_name="Auditok",
        description="Energy-based audio segmentation with two-pass detection. "
                    "Reliable for most content.",
        config_class=AuditokSceneDetectionConfig,
        tags=["energy-based", "reliable", "default"]
    ))

    registry.register_scene_detection(ComponentMeta(
        name="silero",
        display_name="Silero Scene Detection",
        description="VAD-based scene detection using Silero model. "
                    "Better for speech-heavy content.",
        config_class=SileroSceneDetectionConfig,
        tags=["vad-based", "speech-optimized"]
    ))

    registry._initialized = True


# Auto-register on module import
register_default_components()
