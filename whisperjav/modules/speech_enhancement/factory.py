"""
Factory for creating speech enhancer instances.

Supports lazy loading to avoid import overhead for unused backends.

Design Decisions (v1.7.3):
- Single backend entry with model parameter (not separate entries per sample rate)
- Default model is 48kHz variant for best quality
- Graceful degradation handled at backend level
- Consistent with SpeechSegmenterFactory pattern
"""

from typing import Dict, Type, Optional, Any, List, Tuple
import importlib
import logging

from .base import SpeechEnhancer, EnhancementResult

logger = logging.getLogger("whisperjav")

# Registry of available backends: name -> module path
_BACKEND_REGISTRY: Dict[str, str] = {
    "none": "whisperjav.modules.speech_enhancement.backends.none.NullSpeechEnhancer",
    "ffmpeg-dsp": "whisperjav.modules.speech_enhancement.backends.ffmpeg_dsp.FFmpegDSPBackend",
    "zipenhancer": "whisperjav.modules.speech_enhancement.backends.zipenhancer.ZipEnhancerBackend",
    "clearvoice": "whisperjav.modules.speech_enhancement.backends.clearvoice.ClearVoiceSpeechEnhancer",
    "bs-roformer": "whisperjav.modules.speech_enhancement.backends.bs_roformer.BSRoformerSpeechEnhancer",
}

# Cache for loaded backend classes (avoid repeated imports)
_BACKEND_CACHE: Dict[str, Type] = {}

# Dependency information for each backend
_BACKEND_DEPENDENCIES: Dict[str, Dict[str, Any]] = {
    "none": {
        "packages": [],
        "install_hint": "",
        "always_available": True,
        "description": "No enhancement (passthrough)",
    },
    "ffmpeg-dsp": {
        "packages": [],
        "install_hint": "",
        "always_available": True,  # FFmpeg is bundled/required
        "description": "FFmpeg audio filters (loudnorm, compress, denoise)",
    },
    "zipenhancer": {
        "packages": ["modelscope"],
        "install_hint": "pip install modelscope>=1.20",
        "always_available": False,
        "description": "ZipEnhancer 16kHz (lightweight, SOTA quality)",
    },
    "clearvoice": {
        "packages": ["clearvoice"],
        "install_hint": "pip install clearvoice",
        "always_available": False,
        "description": "ClearerVoice speech enhancement (denoising)",
    },
    "bs-roformer": {
        "packages": ["bs_roformer"],
        "install_hint": "pip install bs-roformer-infer",
        "always_available": False,
        "description": "BS-RoFormer vocal isolation",
    },
}

# Default models for each backend
_DEFAULT_MODELS: Dict[str, str] = {
    "ffmpeg-dsp": "loudnorm",  # Default effect (can be comma-separated for combos)
    "zipenhancer": "torch",  # torch (GPU) or onnx (CPU/GPU)
    "clearvoice": "MossFormer2_SE_48K",  # 48kHz for best quality
    "bs-roformer": "vocals",
}


class SpeechEnhancerFactory:
    """
    Factory for creating speech enhancer instances.

    Supports lazy loading to avoid import overhead for unused backends.

    Example:
        # Create default ClearVoice enhancer (48kHz model)
        enhancer = SpeechEnhancerFactory.create("clearvoice")

        # Create with specific model
        enhancer = SpeechEnhancerFactory.create(
            "clearvoice",
            model="FRCRN_SE_16K"
        )

        # Check availability before creating
        available, hint = SpeechEnhancerFactory.is_backend_available("clearvoice")
        if available:
            enhancer = SpeechEnhancerFactory.create("clearvoice")

        # Get preferred sample rate for extraction
        if enhancer:
            extraction_sr = enhancer.get_preferred_sample_rate()
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
            name: Backend name (e.g., "clearvoice", "bs-roformer")

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

        # Try importing required packages
        for package in dep_info["packages"]:
            try:
                importlib.import_module(package)
            except ImportError:
                return False, dep_info["install_hint"]

        return True, ""

    @staticmethod
    def get_available_backends() -> List[Dict[str, Any]]:
        """
        Get information about all backends with availability status.

        Returns:
            List of dicts with name, display_name, available, install_hint, description
        """
        backends = []

        display_names = {
            "none": "None (Skip Enhancement)",
            "ffmpeg-dsp": "FFmpeg DSP (Audio Filters)",
            "zipenhancer": "ZipEnhancer 16kHz (Recommended)",
            "clearvoice": "ClearerVoice (48kHz)",
            "bs-roformer": "BS-RoFormer Vocal Isolation",
        }

        for name in _BACKEND_REGISTRY.keys():
            available, hint = SpeechEnhancerFactory.is_backend_available(name)
            dep_info = _BACKEND_DEPENDENCIES.get(name, {})

            backends.append({
                "name": name,
                "display_name": display_names.get(name, name),
                "available": available,
                "install_hint": hint,
                "description": dep_info.get("description", ""),
                "default_model": _DEFAULT_MODELS.get(name),
            })

        return backends

    @staticmethod
    def get_backend_models(name: str) -> List[str]:
        """
        Get available models for a backend.

        Args:
            name: Backend name

        Returns:
            List of model identifiers, or empty list if backend not available
        """
        # Try to load the backend class and get models
        try:
            backend_class = SpeechEnhancerFactory._load_backend_class(name)
            # Create temporary instance to query models
            instance = backend_class()
            models = instance.get_supported_models()
            instance.cleanup()
            return models
        except Exception:
            # Return known models as fallback
            if name == "ffmpeg-dsp":
                return ["loudnorm", "normalize", "compress", "denoise", "highpass", "lowpass", "deess", "amplify"]
            elif name == "zipenhancer":
                return ["zipenhancer"]  # No model variants
            elif name == "clearvoice":
                return ["FRCRN_SE_16K", "MossFormer2_SE_48K", "MossFormerGAN_SE_16K", "MossFormer2_SS_16K"]
            elif name == "bs-roformer":
                return ["vocals", "other"]
            return []

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

        if name not in _BACKEND_REGISTRY:
            available = SpeechEnhancerFactory.list_backends()
            raise ValueError(
                f"Unknown speech enhancer: '{name}'. "
                f"Available backends: {available}"
            )

        # Check availability
        available, hint = SpeechEnhancerFactory.is_backend_available(name)
        if not available:
            raise ImportError(
                f"Speech enhancer '{name}' is not available. {hint}"
            )

        # Import module and get class
        module_path = _BACKEND_REGISTRY[name]
        module_name, class_name = module_path.rsplit(".", 1)

        try:
            module = importlib.import_module(module_name)
            backend_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to load speech enhancer '{name}': {e}"
            )

        # Cache for future use
        _BACKEND_CACHE[name] = backend_class
        return backend_class

    @staticmethod
    def create(
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SpeechEnhancer:
        """
        Create a speech enhancer instance.

        Args:
            name: Backend name ('none', 'clearvoice', 'bs-roformer')
            config: Configuration dict (from resolved config or GUI)
            **kwargs: Additional backend-specific parameters (override config)
                - model: Model variant to use (e.g., "MossFormer2_SE_48K")
                - device: Device to use ("cuda", "cpu", "auto")

        Returns:
            Configured SpeechEnhancer instance

        Raises:
            ValueError: If backend name is unknown
            ImportError: If backend dependencies not installed

        Example:
            # Simple creation (uses default 48kHz model)
            enhancer = SpeechEnhancerFactory.create("clearvoice")

            # With specific model
            enhancer = SpeechEnhancerFactory.create(
                "clearvoice",
                model="FRCRN_SE_16K"
            )

            # Passthrough (no enhancement)
            enhancer = SpeechEnhancerFactory.create("none")
        """
        # Merge config with kwargs (kwargs take precedence)
        params = dict(config or {})
        params.update(kwargs)

        # Set default model if not specified
        if "model" not in params and name in _DEFAULT_MODELS:
            params["model"] = _DEFAULT_MODELS[name]

        # Load and instantiate
        backend_class = SpeechEnhancerFactory._load_backend_class(name)

        logger.debug(f"Creating speech enhancer: {name} with params: {params}")
        return backend_class(**params)

    @staticmethod
    def create_from_resolved_config(resolved_config: Dict[str, Any]) -> Optional[SpeechEnhancer]:
        """
        Create enhancer from resolved pipeline configuration.

        This is the integration point for pipelines using the
        config system.

        Args:
            resolved_config: Resolved configuration dict from main.py

        Returns:
            Configured SpeechEnhancer instance, or None if enhancement disabled
        """
        params = resolved_config.get("params", {})
        enhancer_config = params.get("speech_enhancer", {})

        # Get backend from config (default: none = skip enhancement)
        backend = enhancer_config.get("backend", "none")

        # If backend is "none" or empty, return None to indicate no enhancement
        if not backend or backend == "none":
            logger.debug("Speech enhancement disabled (backend='none')")
            return None

        return SpeechEnhancerFactory.create(backend, config=enhancer_config)

    @staticmethod
    def get_extraction_sample_rate(enhancer: Optional[SpeechEnhancer]) -> int:
        """
        Get the sample rate to use for audio extraction.

        If an enhancer is provided, returns its preferred rate.
        Otherwise, returns the default 16kHz.

        Args:
            enhancer: SpeechEnhancer instance or None

        Returns:
            Sample rate in Hz for audio extraction
        """
        if enhancer is None:
            return 16000  # Default for VAD/ASR
        return enhancer.get_preferred_sample_rate()
