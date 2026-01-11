"""
Configuration Resolver v3.0 - Component-Based.

.. deprecated:: 1.7.0
    This module is part of the LEGACY configuration system (v1-v3).
    For new development, use the v4 YAML-driven configuration system:

        from whisperjav.config.v4 import ConfigManager
        manager = ConfigManager()
        config = manager.get_model_config("kotoba-whisper-v2", "balanced")

    See: whisperjav/config/v4/README.md
    ADR: docs/adr/ADR-001-yaml-config-architecture.md

Resolves configuration from component selections instead of JSON.
"""

import logging
from typing import Any, Dict, List, Optional

from .components import (
    get_asr_registry,
    get_vad_registry,
    get_feature_registry,
)
from whisperjav.utils.device_detector import get_best_device

logger = logging.getLogger("whisperjav")

# CTranslate2-based providers support "auto" compute_type selection
CTRANSLATE2_PROVIDERS = {"faster_whisper", "kotoba_faster_whisper"}




def _is_pascal_gpu() -> bool:
    """
    Detect if GPU is NVIDIA Pascal architecture (GTX 10xx, Quadro Pxxx).

    Pascal GPUs (sm_60, sm_61, sm_62) don't support efficient float16 computation.
    When CTranslate2's "auto" compute_type selects float16, it fails with:
    "target device or backend do not support efficient float16 computation"

    Pascal GPUs require cu118 (the last PyTorch build to support them), but if
    the user has a newer CUDA version installed (cu121+), we need to fallback
    to float32 compute_type to avoid the error.

    See: https://github.com/meizhong986/WhisperJAV/issues/123

    Returns:
        True if Pascal GPU detected, False otherwise
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        # Get compute capability (major.minor)
        capability = torch.cuda.get_device_capability(0)
        major, minor = capability

        # Pascal is compute capability 6.x (sm_60, sm_61, sm_62)
        # - sm_60: GP100 (Tesla P100, Quadro GP100)
        # - sm_61: GP102, GP104, GP106, GP107 (GTX 1080 Ti, 1080, 1070, 1060, Quadro Pxxx)
        # - sm_62: GP10B (Jetson TX2, DRIVE PX2)
        if major == 6:
            gpu_name = torch.cuda.get_device_name(0)
            logger.debug(f"Pascal GPU detected: {gpu_name} (sm_{major}{minor})")
            return True

        # Also check by name for known Pascal GPUs (fallback)
        gpu_name = torch.cuda.get_device_name(0)
        pascal_names = [
            # GeForce GTX 10xx series
            "GTX 1080", "GTX 1070", "GTX 1060", "GTX 1050",
            # GeForce TITAN X Pascal
            "TITAN X", "TITAN XP",
            # Quadro Pascal series
            "QUADRO P", "QUADRO GP100",
            # Tesla Pascal
            "TESLA P",
        ]
        gpu_upper = gpu_name.upper()
        if any(x in gpu_upper for x in pascal_names):
            logger.debug(f"Pascal GPU detected by name: {gpu_name}")
            return True

        return False
    except Exception as e:
        logger.debug(f"Pascal detection failed: {e}")
        return False


def _is_blackwell_gpu() -> bool:
    """
    Detect if GPU is NVIDIA Blackwell architecture (RTX 50 series).

    Blackwell GPUs (sm_120, compute capability 12.x) have a bug in CTranslate2
    where 'auto' compute_type incorrectly selects int8_float16, which then fails
    with "target device or backend do not support efficient int8_float16 computation".

    See: https://github.com/meizhong986/WhisperJAV/issues/113
    See: https://github.com/OpenNMT/CTranslate2/issues/1865

    Returns:
        True if Blackwell GPU detected, False otherwise
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False

        # Get compute capability (major.minor)
        capability = torch.cuda.get_device_capability(0)
        major, minor = capability

        # Blackwell is compute capability 12.x (sm_120)
        if major >= 12:
            gpu_name = torch.cuda.get_device_name(0)
            logger.debug(f"Blackwell GPU detected: {gpu_name} (sm_{major}{minor})")
            return True

        # Also check by name for RTX 50 series
        gpu_name = torch.cuda.get_device_name(0)
        if any(x in gpu_name.upper() for x in ["5090", "5080", "5070", "5060", "5050"]):
            logger.debug(f"RTX 50 series detected by name: {gpu_name}")
            return True

        return False
    except Exception as e:
        logger.debug(f"Blackwell detection failed: {e}")
        return False

def _get_compute_type_for_device(device: str, provider: str) -> str:
    """
    Select optimal compute_type based on device and provider.

    CTranslate2 providers (faster_whisper, kotoba_faster_whisper):
    - Pascal GPUs (sm_6x): Force "float32" (float16 not efficiently supported)
    - RTX 50 Blackwell (sm_120): Force "float16" (int8_float16 buggy in CTranslate2)
    - Other GPUs: Returns "auto" to delegate selection to CTranslate2
    - See: https://github.com/OpenNMT/CTranslate2/issues/1865

    PyTorch providers (openai_whisper, stable_ts):
    - CUDA/MPS: float16
    - CPU: float32

    Args:
        device: Detected device ("cuda", "mps", "cpu")
        provider: ASR provider name

    Returns:
        Optimal compute_type for the device/provider combination
    """
    if provider in CTRANSLATE2_PROVIDERS:
        # Pascal workaround: Pascal GPUs (GTX 10xx, Quadro Pxxx) don't support
        # efficient float16 computation. CTranslate2's "auto" may select float16
        # which fails with "target device or backend do not support efficient float16"
        # Force float32 for Pascal GPUs as the safe fallback.
        # See: https://github.com/meizhong986/WhisperJAV/issues/123
        if device == "cuda" and _is_pascal_gpu():
            logger.info(
                "Pascal GPU detected (GTX 10xx/Quadro Pxxx). Using float32 compute_type "
                "because Pascal GPUs don't support efficient float16 computation."
            )
            return "float32"

        # Blackwell workaround: CTranslate2's "auto" incorrectly selects int8_float16
        # which fails with "target device or backend do not support efficient int8_float16"
        # Force float16 for Blackwell until CTranslate2 properly supports sm_120
        # See: https://github.com/meizhong986/WhisperJAV/issues/113
        if device == "cuda" and _is_blackwell_gpu():
            logger.info(
                "Blackwell GPU detected (RTX 50 series). Using float16 compute_type "
                "due to CTranslate2 int8_float16 compatibility issue."
            )
            return "float16"

        # Delegate to CTranslate2's internal "auto" selection logic
        # This automatically handles:
        # - GPU capability detection (int8_float16 for RTX 20/30/40)
        # - CPU instruction sets (AVX2, AVX512)
        return "auto"
    else:
        # PyTorch-based providers can use float16 on GPU/MPS, float32 on CPU
        return "float16" if device in ("cuda", "mps") else "float32"


def resolve_config_v3(
    asr: str,
    vad: str = "none",
    sensitivity: str = "balanced",
    task: str = "transcribe",
    features: Optional[List[str]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Resolve configuration from component selections.

    Args:
        asr: ASR component name (e.g., 'faster_whisper', 'stable_ts')
        vad: VAD component name (e.g., 'silero', 'none')
        sensitivity: Sensitivity level ('conservative', 'balanced', 'aggressive')
        task: Task type ('transcribe', 'translate')
        features: List of feature names (e.g., ['auditok_scene_detection'])
        overrides: Parameter overrides (e.g., {'asr.beam_size': 10})
        device: Device override (None/'auto' = auto-detect, 'cuda'/'cpu' = explicit)
        compute_type: Compute type override (None/'auto' = provider-specific default,
                      'float16'/'float32'/'int8'/'int8_float16'/'int8_float32' = explicit)

    Returns:
        Resolved configuration dictionary.

    Example:
        >>> config = resolve_config_v3(
        ...     asr='faster_whisper',
        ...     vad='silero',
        ...     sensitivity='aggressive',
        ...     task='transcribe',
        ...     features=['auditok_scene_detection'],
        ...     overrides={'asr.beam_size': 10}
        ... )
        >>> # With explicit hardware configuration:
        >>> config = resolve_config_v3(
        ...     asr='faster_whisper',
        ...     device='cuda',
        ...     compute_type='int8_float16'
        ... )
    """
    if features is None:
        features = []
    if overrides is None:
        overrides = {}

    # Get ASR component
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        available = list(asr_registry.keys())
        raise ValueError(f"Unknown ASR component: {asr}. Available: {available}")

    asr_component = asr_registry[asr]
    asr_preset = asr_component.get_preset(sensitivity)
    if asr_preset is None:
        raise ValueError(f"ASR '{asr}' has no preset for sensitivity '{sensitivity}'")

    asr_params = asr_preset.model_dump()

    # Override task
    if 'task' in asr_params:
        asr_params['task'] = task

    # Get VAD component
    vad_params = {}
    if vad and vad != "none":
        vad_registry = get_vad_registry()
        if vad not in vad_registry:
            available = list(vad_registry.keys())
            raise ValueError(f"Unknown VAD component: {vad}. Available: {available}")

        vad_component = vad_registry[vad]
        vad_preset = vad_component.get_preset(sensitivity)
        if vad_preset is None:
            raise ValueError(f"VAD '{vad}' has no preset for sensitivity '{sensitivity}'")

        vad_params = vad_preset.model_dump()

        # Check compatibility
        if asr not in vad_component.compatible_asr:
            logger.warning(
                f"VAD '{vad}' may not be compatible with ASR '{asr}'. "
                f"Compatible ASR: {vad_component.compatible_asr}"
            )

    # Get feature configurations
    feature_configs = {}
    feature_registry = get_feature_registry()
    for feature_name in features:
        if feature_name not in feature_registry:
            available = list(feature_registry.keys())
            raise ValueError(f"Unknown feature: {feature_name}. Available: {available}")

        feature_component = feature_registry[feature_name]
        feature_preset = feature_component.get_preset(sensitivity)
        if feature_preset:
            feature_configs[feature_component.feature_type] = feature_preset.model_dump()
        else:
            # Use defaults if no preset
            feature_configs[feature_component.feature_type] = feature_component.get_defaults()

    # Apply overrides
    _apply_overrides(asr_params, overrides, 'asr')
    _apply_overrides(vad_params, overrides, 'vad')
    for feature_type, feature_config in feature_configs.items():
        _apply_overrides(feature_config, overrides, feature_type)

    # Build model configuration with device/compute_type selection
    # Priority: 1) User-provided value, 2) Auto-detection
    #
    # Device selection:
    # - None or "auto" → auto-detect using get_best_device()
    # - "cuda"/"cpu" → use explicit value (MPS fallback handled by ASR modules)
    #
    # Compute type selection:
    # - None or "auto" → provider-specific default (CTranslate2 "auto" or PyTorch float16/32)
    # - Explicit value → pass through to ASR module
    if device and device != "auto":
        selected_device = device
        logger.debug(f"v3.0 resolver: Using user-specified device='{selected_device}'")
    else:
        selected_device = get_best_device()
        logger.debug(f"v3.0 resolver: Auto-detected device='{selected_device}'")

    if compute_type and compute_type != "auto":
        selected_compute_type = compute_type
        logger.debug(f"v3.0 resolver: Using user-specified compute_type='{selected_compute_type}'")
    else:
        selected_compute_type = _get_compute_type_for_device(selected_device, asr_component.provider)
        logger.debug(
            f"v3.0 resolver: Selected compute_type='{selected_compute_type}' "
            f"for provider='{asr_component.provider}'"
        )

    model_config = {
        'provider': asr_component.provider,
        'model_name': asr_component.model_id,
        'device': selected_device,
        'compute_type': selected_compute_type,
        'supported_tasks': asr_component.supported_tasks,
    }

    # Build result
    result = {
        'asr_name': asr,
        'vad_name': vad,
        'sensitivity_name': sensitivity,
        'task': task,
        'language': asr_params.get('language', 'ja'),
        'model': model_config,
        'params': {
            'asr': asr_params,
            'vad': vad_params,
        },
        'features': feature_configs,
    }

    logger.debug(f"v3.0 resolver: Resolved config for asr='{asr}', vad='{vad}', sensitivity='{sensitivity}'")

    return result


def _apply_overrides(params: Dict[str, Any], overrides: Dict[str, Any], prefix: str) -> None:
    """Apply parameter overrides."""
    for key, value in overrides.items():
        parts = key.split('.')
        if len(parts) == 2 and parts[0] == prefix:
            param_name = parts[1]
            if param_name in params:
                params[param_name] = value
            else:
                logger.warning(f"Override key '{key}' not found in {prefix} parameters")


def get_asr_options(asr: str) -> List[Dict[str, Any]]:
    """
    Get parameter schema for an ASR component.

    Useful for GUI to build parameter panel.
    """
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        raise ValueError(f"Unknown ASR component: {asr}")

    return asr_registry[asr].get_schema()['parameters']


def get_vad_options(vad: str) -> List[Dict[str, Any]]:
    """
    Get parameter schema for a VAD component.

    Useful for GUI to build parameter panel.
    """
    vad_registry = get_vad_registry()
    if vad not in vad_registry:
        raise ValueError(f"Unknown VAD component: {vad}")

    return vad_registry[vad].get_schema()['parameters']


def get_compatible_vads(asr: str) -> List[str]:
    """
    Get list of VAD components compatible with an ASR.

    Useful for GUI to filter VAD dropdown.
    """
    asr_registry = get_asr_registry()
    if asr not in asr_registry:
        raise ValueError(f"Unknown ASR component: {asr}")

    return asr_registry[asr].compatible_vad


def list_available_asr() -> List[Dict[str, Any]]:
    """List all available ASR components with metadata."""
    return [cls.to_dict() for cls in get_asr_registry().values()]


def list_available_vad() -> List[Dict[str, Any]]:
    """List all available VAD components with metadata."""
    return [cls.to_dict() for cls in get_vad_registry().values()]


def list_available_features() -> List[Dict[str, Any]]:
    """List all available feature components with metadata."""
    return [cls.to_dict() for cls in get_feature_registry().values()]
