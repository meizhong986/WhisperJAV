"""
V3 Architecture WhisperJAV configuration module.

This module provides configuration management for the WhisperJAV transcription system.
As of V3, the primary configuration system uses TranscriptionTunerV3, which implements
a clean, modular architecture with single source of truth configuration management.
"""

# V3 Architecture - Primary tuner
from .transcription_tuner_v3 import TranscriptionTunerV3

# Legacy tuners for backward compatibility (deprecated)
# These are maintained for existing code but should not be used for new development
from .transcription_tuner import TranscriptionTuner

# Optional V2 tuner (may not exist in all installations)
try:
    from .transcription_tuner_v2 import TranscriptionTunerV2
except ImportError:
    TranscriptionTunerV2 = None

# Export all available tuners
__all__ = ['TranscriptionTunerV3', 'TranscriptionTuner', 'TranscriptionTunerV2']

# For convenience and future compatibility, alias V3 as the default
# This allows future code to use 'from whisperjav.config import Tuner'
Tuner = TranscriptionTunerV3

# Version information for the configuration system
CONFIG_VERSION = "3.0"

# Deprecation notice function for legacy usage
def _deprecation_notice():
    """Print deprecation notice for legacy tuner usage."""
    import warnings
    warnings.warn(
        "TranscriptionTuner and TranscriptionTunerV2 are deprecated. "
        "Please use TranscriptionTunerV3 for all new development. "
        "Legacy tuners will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )