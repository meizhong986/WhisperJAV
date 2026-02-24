"""
Factory for creating speech segmenter instances.

Supports lazy loading to avoid import overhead for unused backends.
"""

from typing import Dict, Type, Optional, Any, List, Tuple
import importlib
import importlib.util
import logging

from .base import SpeechSegmenter, SegmentationResult

logger = logging.getLogger("whisperjav")

# Registry of available backends: name -> module path
_BACKEND_REGISTRY: Dict[str, str] = {
    "silero": "whisperjav.modules.speech_segmentation.backends.silero.SileroSpeechSegmenter",
    "silero-v4.0": "whisperjav.modules.speech_segmentation.backends.silero.SileroSpeechSegmenter",
    "silero-v3.1": "whisperjav.modules.speech_segmentation.backends.silero.SileroSpeechSegmenter",
    "nemo": "whisperjav.modules.speech_segmentation.backends.nemo.NemoSpeechSegmenter",
    "nemo-lite": "whisperjav.modules.speech_segmentation.backends.nemo.NemoSpeechSegmenter",
    "nemo-diarization": "whisperjav.modules.speech_segmentation.backends.nemo.NemoSpeechSegmenter",
    "whisper-vad": "whisperjav.modules.speech_segmentation.backends.whisper_vad.WhisperVadSpeechSegmenter",
    "whisper-vad-tiny": "whisperjav.modules.speech_segmentation.backends.whisper_vad.WhisperVadSpeechSegmenter",
    "whisper-vad-base": "whisperjav.modules.speech_segmentation.backends.whisper_vad.WhisperVadSpeechSegmenter",
    "whisper-vad-small": "whisperjav.modules.speech_segmentation.backends.whisper_vad.WhisperVadSpeechSegmenter",
    "whisper-vad-medium": "whisperjav.modules.speech_segmentation.backends.whisper_vad.WhisperVadSpeechSegmenter",
    "ten": "whisperjav.modules.speech_segmentation.backends.ten.TenSpeechSegmenter",
    "silero-v6.2": "whisperjav.modules.speech_segmentation.backends.silero_v6.SileroV6SpeechSegmenter",
    "none": "whisperjav.modules.speech_segmentation.backends.none.NullSpeechSegmenter",
}

# Cache for loaded backend classes (avoid repeated imports)
_BACKEND_CACHE: Dict[str, Type] = {}

# Dependency information for each backend
_BACKEND_DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    "silero": {
        "packages": ["torch"],
        "install_hint": "torch is already required by WhisperJAV",
        "always_available": True,
    },
    "nemo": {
        # NOTE: Use top-level "nemo" for availability check to avoid triggering heavy initialization
        # The actual nemo.collections.asr import happens lazily when the backend is created
        "packages": ["nemo"],
        "install_hint": "pip install nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main",
        "always_available": False,
    },
    "ten": {
        "packages": ["ten_vad"],
        "install_hint": "See https://github.com/ten-framework/ten-vad",
        "always_available": False,
    },
    "silero-v6.2": {
        "packages": ["silero_vad"],
        "install_hint": "pip install silero-vad>=6.2",
        "always_available": False,
    },
    "whisper": {
        "packages": ["faster_whisper"],
        "install_hint": "pip install faster-whisper",
        "always_available": True,  # faster-whisper already required by WhisperJAV
    },
    "none": {
        "packages": [],
        "install_hint": "",
        "always_available": True,
    },
}


class SpeechSegmenterFactory:
    """
    Factory for creating speech segmenter instances.

    Supports lazy loading to avoid import overhead for unused backends.

    Example:
        # Create default Silero segmenter
        segmenter = SpeechSegmenterFactory.create("silero")

        # Create with custom parameters
        segmenter = SpeechSegmenterFactory.create(
            "silero-v4.0",
            threshold=0.3,
            min_speech_duration_ms=100
        )

        # Check availability before creating
        available, hint = SpeechSegmenterFactory.is_backend_available("nemo")
        if available:
            segmenter = SpeechSegmenterFactory.create("nemo")
    """

    @staticmethod
    def list_backends() -> List[str]:
        """Return list of all registered backend names."""
        return list(_BACKEND_REGISTRY.keys())

    @staticmethod
    def list_unique_backends() -> List[str]:
        """Return list of unique backend names (without version aliases)."""
        return ["silero", "nemo", "whisper", "ten", "none"]

    @staticmethod
    def is_backend_available(name: str) -> Tuple[bool, str]:
        """
        Check if a backend's dependencies are installed.

        Args:
            name: Backend name (e.g., "silero", "nemo")

        Returns:
            Tuple of (is_available, install_hint)
            - is_available: True if backend can be used
            - install_hint: Installation instructions if not available
        """
        # Check exact name first (e.g., "silero-v6.2" has its own entry),
        # then fall back to base name (e.g., "silero-v4.0" -> "silero")
        if name in _BACKEND_DEPENDENCIES:
            dep_info = _BACKEND_DEPENDENCIES[name]
        else:
            base_name = name.split("-")[0] if "-" in name else name
            if base_name not in _BACKEND_DEPENDENCIES:
                return False, f"Unknown backend: {name}"
            dep_info = _BACKEND_DEPENDENCIES[base_name]

        if dep_info["always_available"]:
            return True, ""

        # Check if required packages are importable WITHOUT actually importing them
        # This avoids triggering heavy initialization (like NeMo's Megatron init) at startup
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
        backends = []
        display_names = {
            "silero": "Silero VAD v4.0",
            "silero-v3.1": "Silero VAD v3.1",
            "nemo-lite": "NeMo Lite",
            "nemo-diarization": "NeMo Diarization",
            "whisper-vad": "Whisper VAD (small)",
            "whisper-vad-tiny": "Whisper VAD (tiny)",
            "whisper-vad-base": "Whisper VAD (base)",
            "whisper-vad-medium": "Whisper VAD (medium)",
            "silero-v6.2": "Silero VAD v6.2",
            "ten": "TEN VAD",
            "none": "None (Skip)",
        }

        for name in ["silero", "silero-v3.1", "silero-v6.2", "nemo-lite", "nemo-diarization", "whisper-vad", "whisper-vad-tiny", "whisper-vad-medium", "ten", "none"]:
            available, hint = SpeechSegmenterFactory.is_backend_available(name)
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
            ImportError: If backend dependencies not installed
        """
        # Check cache first
        if name in _BACKEND_CACHE:
            return _BACKEND_CACHE[name]

        # Normalize name for registry lookup
        registry_name = name
        if name.startswith("silero") and name not in _BACKEND_REGISTRY:
            registry_name = "silero"
        elif name.startswith("nemo") and name not in _BACKEND_REGISTRY:
            registry_name = "nemo"
        elif name.startswith("whisper-vad") and name not in _BACKEND_REGISTRY:
            registry_name = "whisper-vad"

        if registry_name not in _BACKEND_REGISTRY:
            available = SpeechSegmenterFactory.list_backends()
            raise ValueError(
                f"Unknown speech segmenter: '{name}'. "
                f"Available backends: {available}"
            )

        # Check availability
        available, hint = SpeechSegmenterFactory.is_backend_available(name)
        if not available:
            raise ImportError(
                f"Speech segmenter '{name}' is not available. {hint}"
            )

        # Import module and get class
        module_path = _BACKEND_REGISTRY[registry_name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            backend_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load speech segmenter '{name}': {e}"
            )

        # Cache for future use
        _BACKEND_CACHE[name] = backend_class
        return backend_class

    @staticmethod
    def create(
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SpeechSegmenter:
        """
        Create a speech segmenter instance.

        Args:
            name: Backend name ('silero', 'silero-v4.0', 'silero-v3.1',
                  'nemo', 'ten', 'none')
            config: Configuration dict (from v4 YAML or resolved config)
            **kwargs: Additional backend-specific parameters (override config)

        Returns:
            Configured SpeechSegmenter instance

        Raises:
            ValueError: If backend name is unknown
            ImportError: If backend dependencies not installed

        Example:
            # Simple creation
            segmenter = SpeechSegmenterFactory.create("silero")

            # With version
            segmenter = SpeechSegmenterFactory.create("silero-v3.1")

            # With parameters
            segmenter = SpeechSegmenterFactory.create(
                "silero",
                threshold=0.3,
                min_speech_duration_ms=100
            )
        """
        # Merge config with kwargs (kwargs take precedence)
        params = dict(config or {})
        params.update(kwargs)

        # Handle Silero version suffix
        if name.startswith("silero"):
            if "-" in name:
                version = name.split("-", 1)[1]
                params["version"] = version
            elif "version" not in params:
                params["version"] = "v4.0"

        # Handle NeMo variant suffix (nemo-lite, nemo-diarization)
        if name.startswith("nemo"):
            if "-" in name:
                variant = name  # Pass full name as variant (e.g., "nemo-lite")
            else:
                variant = "nemo-lite"  # Default to nemo-lite
            params["variant"] = variant

        # Handle Whisper VAD variant suffix (whisper-vad, whisper-vad-tiny, etc.)
        if name.startswith("whisper-vad"):
            params["variant"] = name  # Pass full name as variant

        # Load and instantiate
        backend_class = SpeechSegmenterFactory._load_backend_class(name)

        logger.debug(f"Creating speech segmenter: {name} with params: {params}")
        return backend_class(**params)

    @staticmethod
    def create_from_resolved_config(resolved_config: Dict[str, Any]) -> SpeechSegmenter:
        """
        Create segmenter from resolved pipeline configuration.

        This is the integration point for pipelines using the legacy
        config system.

        Args:
            resolved_config: Resolved configuration dict from main.py

        Returns:
            Configured SpeechSegmenter instance
        """
        params = resolved_config.get("params", {})
        seg_config = params.get("speech_segmenter", {})

        # Check for skip_vad flag (backward compatibility)
        vad_params = params.get("vad", {})
        if vad_params.get("skip_vad", False):
            logger.info("VAD skip requested, using 'none' segmenter")
            return SpeechSegmenterFactory.create("none")

        # Get backend from config
        backend = seg_config.get("backend", "silero")

        # Merge VAD params with segmenter config for backward compatibility
        merged_config = {**vad_params, **seg_config}

        return SpeechSegmenterFactory.create(backend, config=merged_config)
