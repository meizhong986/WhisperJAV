"""
WhisperJAV Translation Module

Provides AI-powered subtitle translation via PySubtrans.

For programmatic usage (e.g., from main pipeline):
    from whisperjav.translate import translate_with_config

    result = translate_with_config(
        input_path="subtitles.srt",
        provider="deepseek",
        target_lang="english"
    )

For CLI usage:
    whisperjav-translate -i subtitles.srt --provider deepseek -t english
"""

# NOTE: cli is NOT imported here to avoid triggering dependency checks
# when main.py imports translate_with_config. The CLI has its own entry point
# (whisperjav-translate) that handles dependency checking appropriately.
# Users who need cli can still do: from whisperjav.translate import cli
from . import core, providers, service

# Export high-level service API for direct usage
from .service import (
    translate_with_config,
    TranslationError,
    ConfigurationError,
)

__all__ = [
    # Submodules (cli omitted - use entry point or explicit import)
    'core',
    'providers',
    'service',
    # Service layer exports
    'translate_with_config',
    'TranslationError',
    'ConfigurationError',
]
