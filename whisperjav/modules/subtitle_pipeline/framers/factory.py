"""
TemporalFramer factory with lazy-import backend registry.

Follows the same factory pattern as SpeechSegmenterFactory — backends
are loaded on demand to avoid importing heavy dependencies at startup.
"""

import importlib
from typing import Any

from whisperjav.modules.subtitle_pipeline.protocols import TemporalFramer

# Registry: name → "module.path.ClassName"
_REGISTRY: dict[str, str] = {
    "full-scene": "whisperjav.modules.subtitle_pipeline.framers.full_scene.FullSceneFramer",
    "vad-grouped": "whisperjav.modules.subtitle_pipeline.framers.vad_grouped.VadGroupedFramer",
    "srt-source": "whisperjav.modules.subtitle_pipeline.framers.srt_source.SrtSourceFramer",
    "manual": "whisperjav.modules.subtitle_pipeline.framers.manual.ManualFramer",
}

# Cache for loaded classes
_CLASS_CACHE: dict[str, type] = {}


class TemporalFramerFactory:
    """Factory for creating TemporalFramer instances."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TemporalFramer:
        """
        Create a TemporalFramer by backend name.

        Args:
            name: Backend name (e.g., "full-scene", "vad-grouped", "srt-source", "manual").
            **kwargs: Passed to the backend constructor.

        Returns:
            A TemporalFramer instance.

        Raises:
            ValueError: If the backend name is unknown.
            ImportError: If the backend's dependencies are not installed.
        """
        backend_class = cls._load_class(name)
        return backend_class(**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Return list of registered backend names."""
        return list(_REGISTRY.keys())

    @classmethod
    def _load_class(cls, name: str) -> type:
        """Lazy-load and cache a backend class."""
        if name in _CLASS_CACHE:
            return _CLASS_CACHE[name]

        if name not in _REGISTRY:
            raise ValueError(f"Unknown temporal framer: '{name}'. Available: {cls.available()}")

        module_path = _REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            backend_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load temporal framer '{name}' from {module_path}: {e}") from e

        _CLASS_CACHE[name] = backend_class
        return backend_class
