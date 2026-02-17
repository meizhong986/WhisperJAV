"""
TextAligner factory with lazy-import backend registry.

Follows the same factory pattern as SpeechSegmenterFactory — backends
are loaded on demand to avoid importing heavy dependencies at startup.
"""

import importlib
from typing import Any

from whisperjav.modules.subtitle_pipeline.protocols import TextAligner

# Registry: name → "module.path.ClassName"
_REGISTRY: dict[str, str] = {
    "qwen3": "whisperjav.modules.subtitle_pipeline.aligners.qwen3.Qwen3ForcedAlignerAdapter",
    "none": "whisperjav.modules.subtitle_pipeline.aligners.none.NoneAligner",
}

# Cache for loaded classes
_CLASS_CACHE: dict[str, type] = {}


class TextAlignerFactory:
    """Factory for creating TextAligner instances."""

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> TextAligner:
        """
        Create a TextAligner by backend name.

        Args:
            name: Backend name (e.g., "qwen3", "none").
            **kwargs: Passed to the backend constructor.

        Returns:
            A TextAligner instance.

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
            raise ValueError(f"Unknown text aligner: '{name}'. Available: {cls.available()}")

        module_path = _REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            backend_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load text aligner '{name}' from {module_path}: {e}") from e

        _CLASS_CACHE[name] = backend_class
        return backend_class
