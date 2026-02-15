"""
Factory for creating scene detector instances.

Supports lazy loading to avoid import overhead for unused backends.
Mirrors the architecture of whisperjav/modules/speech_segmentation/factory.py
for cognitive uniformity across the codebase.

See docs/sprint3_scene_detection_refactor.md for the full design.
"""

import importlib
import importlib.util
import logging
from typing import Any, Dict, List, Tuple, Type

from .base import SceneDetector, SceneDetectionError

logger = logging.getLogger("whisperjav")

# Registry of available backends: name -> "module.path.ClassName"
# Backend modules are created in Phase 2. Until then, create() calls
# will raise ImportError for backends that don't yet exist.
_BACKEND_REGISTRY: Dict[str, str] = {
    "auditok": "whisperjav.modules.scene_detection_backends.auditok_backend.AuditokSceneDetector",
    "silero": "whisperjav.modules.scene_detection_backends.silero_backend.SileroSceneDetector",
    "semantic": "whisperjav.modules.scene_detection_backends.semantic_backend.SemanticSceneDetector",
    "none": "whisperjav.modules.scene_detection_backends.none_backend.NullSceneDetector",
}

# Cache for loaded backend classes (avoid repeated imports)
_BACKEND_CACHE: Dict[str, Type] = {}

# Dependency information for each backend
_BACKEND_DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    "auditok": {
        "packages": ["auditok"],
        "install_hint": "pip install auditok",
        "always_available": True,  # Core dependency of WhisperJAV
    },
    "silero": {
        "packages": ["torch"],
        "install_hint": "torch is already required by WhisperJAV",
        "always_available": True,
    },
    "semantic": {
        "packages": ["sklearn"],
        "install_hint": "pip install scikit-learn",
        "always_available": False,  # Optional dependency
    },
    "none": {
        "packages": [],
        "install_hint": "",
        "always_available": True,
    },
}


class SceneDetectorFactory:
    """
    Factory for creating scene detector instances.

    Supports lazy loading to avoid import overhead for unused backends.

    Example:
        # Create an auditok detector
        detector = SceneDetectorFactory.create("auditok", max_duration=29.0)

        # Create from legacy kwargs (migration bridge)
        detector = SceneDetectorFactory.create_from_legacy_kwargs(
            method="auditok", max_duration=29.0, min_duration=0.3
        )

        # Check availability
        available, hint = SceneDetectorFactory.is_backend_available("semantic")
    """

    @staticmethod
    def list_backends() -> List[str]:
        """Return list of all registered backend names."""
        return list(_BACKEND_REGISTRY.keys())

    @staticmethod
    def is_backend_available(name: str) -> Tuple[bool, str]:
        """
        Check if a backend's dependencies are installed.

        Args:
            name: Backend name (e.g., "auditok", "silero", "semantic")

        Returns:
            Tuple of (is_available, install_hint)
            - is_available: True if backend can be used
            - install_hint: Installation instructions if not available
        """
        if name not in _BACKEND_DEPENDENCIES:
            return False, f"Unknown backend: {name}"

        dep_info = _BACKEND_DEPENDENCIES[name]

        if dep_info["always_available"]:
            return True, ""

        # Check if required packages are importable WITHOUT actually importing them.
        # This avoids triggering heavy initialization at startup.
        for package in dep_info["packages"]:
            spec = importlib.util.find_spec(package)
            if spec is None:
                return False, dep_info["install_hint"]

        return True, ""

    @staticmethod
    def get_available_backends() -> List[Dict[str, Any]]:
        """
        Get information about all backends with availability status.

        Returns:
            List of dicts with name, display_name, available, install_hint
        """
        display_names = {
            "auditok": "Auditok (Silence-Based)",
            "silero": "Silero VAD (Two-Pass)",
            "semantic": "Semantic Audio Clustering",
            "none": "None (Skip Detection)",
        }

        backends = []
        for name in _BACKEND_REGISTRY:
            available, hint = SceneDetectorFactory.is_backend_available(name)
            backends.append({
                "name": name,
                "display_name": display_names.get(name, name),
                "available": available,
                "install_hint": hint,
            })

        return backends

    @staticmethod
    def _load_backend_class(name: str) -> Type:
        """
        Lazy load a backend class.

        Args:
            name: Backend name from registry

        Returns:
            Backend class

        Raises:
            ValueError: If backend name is unknown
            ImportError: If backend module not found or dependencies not installed
        """
        # Check cache first
        if name in _BACKEND_CACHE:
            return _BACKEND_CACHE[name]

        if name not in _BACKEND_REGISTRY:
            available = SceneDetectorFactory.list_backends()
            raise ValueError(
                f"Unknown scene detector: '{name}'. "
                f"Available backends: {available}"
            )

        # Check dependency availability
        available, hint = SceneDetectorFactory.is_backend_available(name)
        if not available:
            raise ImportError(
                f"Scene detector '{name}' is not available. {hint}"
            )

        # Import module and get class
        module_path = _BACKEND_REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            backend_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load scene detector '{name}': {e}"
            )

        # Cache for future use
        _BACKEND_CACHE[name] = backend_class
        return backend_class

    @staticmethod
    def create(name: str, **kwargs) -> SceneDetector:
        """
        Create a scene detector instance.

        Args:
            name: Backend name ('auditok', 'silero', 'semantic', 'none')
            **kwargs: Backend-specific parameters

        Returns:
            Configured SceneDetector instance

        Raises:
            ValueError: If backend name is unknown
            ImportError: If backend dependencies not installed
        """
        backend_class = SceneDetectorFactory._load_backend_class(name)

        logger.debug(f"Creating scene detector: {name} with params: {kwargs}")
        return backend_class(**kwargs)

    @staticmethod
    def create_from_legacy_kwargs(**kwargs) -> SceneDetector:
        """
        Create a scene detector from DynamicSceneDetector-style kwargs.

        This is the zero-risk migration bridge for Phase 3. Pipelines switch from:
            DynamicSceneDetector(method="auditok", **scene_opts)
        to:
            SceneDetectorFactory.create_from_legacy_kwargs(method="auditok", **scene_opts)

        The method pops the routing key ('method') and passes the remaining
        kwargs to the appropriate backend constructor. Each backend is
        responsible for accepting or ignoring kwargs it doesn't recognize.

        Args:
            **kwargs: All kwargs that would be passed to DynamicSceneDetector.
                      'method' key selects the backend (default: 'auditok').

        Returns:
            Configured SceneDetector instance
        """
        # Pop routing key
        method = kwargs.pop("method", "auditok")

        logger.debug(
            f"Creating scene detector from legacy kwargs: method={method}, "
            f"kwargs_keys={list(kwargs.keys())}"
        )

        return SceneDetectorFactory.create(method, **kwargs)
