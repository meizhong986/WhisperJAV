"""
Factory for creating speech enhancer instances.

Supports lazy loading to avoid import overhead for unused backends.

Design Decisions (v1.7.3, extended v1.8.x):
- Single backend entry with model parameter (not separate entries per sample rate)
- Default model varies by backend (e.g., "torch" for ZipEnhancer, "vocals" for BS-RoFormer)
- Graceful degradation handled at backend level
- Consistent with SpeechSegmenterFactory pattern
- Five backends: none, ffmpeg-dsp, zipenhancer, bs-roformer, htdemucs
  (ClearVoice was removed in v1.10.0; see REMOVED_BACKENDS)
"""

from typing import Dict, Type, Optional, Any, List, Tuple
import importlib
import importlib.util
import logging

from .base import SpeechEnhancer, EnhancementResult, SpeechEnhancerUnavailable

logger = logging.getLogger("whisperjav")

# Registry of available backends: name -> module path
_BACKEND_REGISTRY: Dict[str, str] = {
    "none": "whisperjav.modules.speech_enhancement.backends.none.NullSpeechEnhancer",
    "ffmpeg-dsp": "whisperjav.modules.speech_enhancement.backends.ffmpeg_dsp.FFmpegDSPBackend",
    "zipenhancer": "whisperjav.modules.speech_enhancement.backends.zipenhancer.ZipEnhancerBackend",
    "bs-roformer": "whisperjav.modules.speech_enhancement.backends.bs_roformer.BSRoformerSpeechEnhancer",
    "htdemucs": "whisperjav.modules.speech_enhancement.backends.htdemucs.HtDemucsSpeechEnhancer",
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
        "fatal_when_unavailable": True,
    },
    "bs-roformer": {
        "packages": ["bs_roformer"],
        "install_hint": "pip install bs-roformer-infer",
        "always_available": False,
        "description": "BS-RoFormer vocal isolation",
        "fatal_when_unavailable": True,
    },
    "htdemucs": {
        "packages": ["demucs"],
        "install_hint": "pip install demucs",
        "always_available": False,
        "description": "htdemucs vocal isolation (Demucs v4)",
        "fatal_when_unavailable": True,
    },
}

# Backends whose absence must STOP a run instead of falling back to "none".
#
# Owner, 2026-09-17: "if user selected any but they cannot run then it is a
# failure and the process shall stop with helpful communication." Choosing a
# clean-up is a deliberate act; quietly giving the user the untouched audio
# hands them a poor subtitle file from a run that exited 0, with only a warning
# in the log to explain it.
#
# This is every backend that has to be installed. "none" and "ffmpeg-dsp" are
# not in it because they cannot be missing: one does nothing and the other uses
# the FFmpeg that WhisperJAV already requires.
#
# Read from _BACKEND_DEPENDENCIES so there is one source of truth.
FATAL_WHEN_UNAVAILABLE = frozenset(
    name for name, info in _BACKEND_DEPENDENCIES.items()
    if info.get("fatal_when_unavailable")
)

# Default models for each backend
_DEFAULT_MODELS: Dict[str, str] = {
    "ffmpeg-dsp": "loudnorm",  # Default effect (can be comma-separated for combos)
    "zipenhancer": "torch",  # torch (GPU) or onnx (CPU/GPU)
    "bs-roformer": "vocals",
    "htdemucs": "htdemucs",
}

# Backends WhisperJAV no longer includes, with the name users knew them by.
#
# ClearVoice left in v1.10.0 (owner, W73). A value that still names it -- a saved
# GUI setting, an Ensemble preset, a script's command line or pass parameters --
# stops the run with removed_backend_message (owner, W75; text approved W79).
# Without this, an unknown name reaching the enhancer helpers would fall back to
# no clean-up with only a warning in the log.
REMOVED_BACKENDS: Dict[str, str] = {
    "clearvoice": "ClearVoice",
}


def removed_backend_message(value: Any, where: str) -> Optional[str]:
    """
    The message for a speech-enhancer value that names a removed backend, else None.

    value may be "backend" or "backend:model". where names what carried it -- a
    flag such as "--pass1-speech-enhancer" or a pass parameter -- and is put into
    the message as the place to choose another enhancer.
    """
    if not isinstance(value, str):
        return None
    label = REMOVED_BACKENDS.get(value.split(":", 1)[0].strip().lower())
    if label is None:
        return None
    return (f"{label} is no longer included in WhisperJAV 1.10. Choose another speech "
            f"enhancer for {where} (none, zipenhancer, bs-roformer, ffmpeg-dsp or "
            f"htdemucs) and run again.")


def removed_backend_in_pass(pass_config: Dict[str, Any], pass_number: int) -> Optional[str]:
    """
    removed_backend_message for an ensemble pass configuration, else None.

    Looks where a pass can carry a speech enhancer: its speech_enhancer, and the
    qwen_enhancer / hf_speech_enhancer keys of its Qwen3-ASR and Transformers
    parameters, which are passed through to the pipeline unchecked. The pass
    worker calls this before the pass reads any audio.
    """
    candidates = [(pass_config.get("speech_enhancer"), f"--pass{pass_number}-speech-enhancer")]
    for params_key, key, flag in (("qwen_params", "qwen_enhancer", "qwen-params"),
                                  ("hf_params", "hf_speech_enhancer", "hf-params")):
        params = pass_config.get(params_key)
        if isinstance(params, dict):
            candidates.append((params.get(key), f"{key} in --pass{pass_number}-{flag}"))
    for value, where in candidates:
        message = removed_backend_message(value, where)
        if message:
            return message
    return None


class SpeechEnhancerFactory:
    """
    Factory for creating speech enhancer instances.

    Supports lazy loading to avoid import overhead for unused backends.

    Example:
        # Create default ZipEnhancer enhancer (torch model)
        enhancer = SpeechEnhancerFactory.create("zipenhancer")

        # Create with specific model
        enhancer = SpeechEnhancerFactory.create(
            "zipenhancer",
            model="onnx"
        )

        # Check availability before creating
        available, hint = SpeechEnhancerFactory.is_backend_available("zipenhancer")
        if available:
            enhancer = SpeechEnhancerFactory.create("zipenhancer")

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
            name: Backend name (e.g., "zipenhancer", "bs-roformer")

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

        # Check if required packages are importable WITHOUT actually importing them
        # This avoids triggering heavy initialization at startup
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
            List of dicts with name, display_name, available, install_hint, description
        """
        backends = []

        display_names = {
            "none": "None (Skip Enhancement)",
            "ffmpeg-dsp": "FFmpeg DSP (Audio Filters)",
            "zipenhancer": "ZipEnhancer 16kHz (Recommended)",
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
            name: Backend name ('none', 'zipenhancer', 'bs-roformer')
            config: Configuration dict (from resolved config or GUI)
            **kwargs: Additional backend-specific parameters (override config)
                - model: Model variant to use (e.g., "vocals")
                - device: Device to use ("cuda", "cpu", "auto")

        Returns:
            Configured SpeechEnhancer instance

        Raises:
            ValueError: If backend name is unknown
            ImportError: If backend dependencies not installed

        Example:
            # Simple creation (uses the default model)
            enhancer = SpeechEnhancerFactory.create("zipenhancer")

            # With specific model
            enhancer = SpeechEnhancerFactory.create(
                "bs-roformer",
                model="vocals"
            )

            # Passthrough (no enhancement)
            enhancer = SpeechEnhancerFactory.create("none")
        """
        removed = removed_backend_message(name, "this run")
        if removed:
            raise SpeechEnhancerUnavailable(removed)

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
