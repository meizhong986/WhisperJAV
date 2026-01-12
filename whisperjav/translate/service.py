"""
Translation Service Layer.

Provides a high-level API for subtitle translation that can be called
directly from the main pipeline without subprocess invocation.

This module encapsulates:
- Configuration resolution (settings + parameters)
- Provider setup and validation
- Instruction file fetching/caching
- Translation execution via core.py

Usage from main.py:
    from whisperjav.translate.service import translate_with_config

    result_path = translate_with_config(
        input_path=output_path,
        provider=args.translate_provider,
        target_lang=args.translate_target,
        tone=args.translate_tone,
        api_key=args.translate_api_key
    )
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Callable, Optional

from .providers import PROVIDER_CONFIGS, SUPPORTED_TARGETS
from .settings import load_settings, DEFAULT_SETTINGS
from .instructions import get_instruction_content
from .core import translate_subtitle

logger = logging.getLogger(__name__)


class TranslationError(Exception):
    """Raised when translation fails."""
    pass


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def _resolve_api_key(provider_config: dict, api_key: Optional[str] = None) -> str:
    """
    Resolve API key from parameter or environment variable.

    Args:
        provider_config: Provider configuration dict with 'env_var' key
        api_key: Explicit API key (highest priority)

    Returns:
        Resolved API key

    Raises:
        ConfigurationError: If no API key found
    """
    if api_key:
        return api_key

    env_var = provider_config.get('env_var')
    if env_var:
        key = os.getenv(env_var)
        if key:
            return key

    raise ConfigurationError(
        f"API key not found. Set {env_var} environment variable or provide api_key parameter."
    )


def _resolve_model(provider_config: dict, model: Optional[str] = None, settings: Optional[dict] = None) -> str:
    """
    Resolve model name with precedence: parameter > settings > provider default.

    Args:
        provider_config: Provider configuration dict with 'model' key
        model: Explicit model override
        settings: User settings dict

    Returns:
        Resolved model name
    """
    if model:
        return model

    if settings and settings.get('model'):
        return settings['model']

    return provider_config['model']


def _resolve_instruction_file(tone: str = "standard", refresh: bool = False) -> Optional[str]:
    """
    Resolve instruction file path by fetching content and caching to temp file.

    Args:
        tone: Translation tone ('standard' or 'pornify')
        refresh: Force refresh of cached content

    Returns:
        Path to instruction file, or None if unavailable
    """
    instruction_content = get_instruction_content(tone=tone, refresh=refresh)

    if not instruction_content:
        logger.debug(f"No instruction content available for tone: {tone}")
        return None

    # Save to temp file
    temp_dir = Path(tempfile.gettempdir()) / 'whisperjav_translate'
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f'instructions_{tone}.txt'

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(instruction_content)

    return str(temp_file)


def _build_provider_options(
    tone: str = "standard",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    settings_model_params: Optional[dict] = None
) -> dict:
    """
    Build provider options with tone-aware defaults.

    Precedence: explicit params > settings > tone-aware defaults

    Args:
        tone: Translation tone for default selection
        temperature: Explicit temperature override
        top_p: Explicit top_p override
        settings_model_params: Model params from settings file

    Returns:
        Dict of provider options
    """
    # Start from tone-aware defaults
    if tone == 'pornify':
        default_temperature = 1.2
        default_top_p = 0.9
    else:
        default_temperature = 0.5
        default_top_p = 0.9

    result_temp = default_temperature
    result_top_p = default_top_p

    # Apply settings overrides
    if settings_model_params:
        if settings_model_params.get('temperature') is not None:
            try:
                result_temp = float(settings_model_params['temperature'])
            except (ValueError, TypeError):
                pass
        if settings_model_params.get('top_p') is not None:
            try:
                result_top_p = float(settings_model_params['top_p'])
            except (ValueError, TypeError):
                pass

    # Apply explicit parameter overrides (highest priority)
    if temperature is not None:
        result_temp = temperature
    if top_p is not None:
        result_top_p = top_p

    # Clamp to valid ranges
    result_temp = max(0.0, min(2.0, result_temp))
    result_top_p = max(0.0, min(1.0, result_top_p))

    return {
        'temperature': result_temp,
        'top_p': result_top_p
    }


def translate_with_config(
    input_path: str,
    provider: str = "deepseek",
    target_lang: str = "english",
    tone: str = "standard",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    source_lang: str = "japanese",
    output_path: Optional[str] = None,
    instruction_file: Optional[str] = None,
    scene_threshold: Optional[float] = None,
    max_batch_size: Optional[int] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    stream: bool = False,
    debug: bool = False,
    extra_context: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    n_gpu_layers: int = -1
) -> Optional[Path]:
    """
    Translate subtitle file with full configuration resolution.

    This is the primary entry point for programmatic translation.
    Can be called directly from main.py without subprocess overhead.

    Configuration is resolved with the following precedence:
    1. Explicit parameters (highest priority)
    2. User settings file (~/.config/WhisperJAV/translate/settings.json)
    3. Built-in defaults (lowest priority)

    Args:
        input_path: Path to input SRT file
        provider: AI provider name ('deepseek', 'openrouter', 'gemini', 'claude', 'gpt')
        target_lang: Target language ('english', 'chinese', 'indonesian', 'spanish')
        tone: Translation tone ('standard' or 'pornify')
        api_key: API key (or set via environment variable)
        model: Model override (uses provider default if not specified)
        source_lang: Source language ('japanese', 'korean', 'chinese')
        output_path: Output file path (auto-generated if not specified)
        instruction_file: Custom instruction file path
        scene_threshold: Scene threshold in seconds
        max_batch_size: Maximum batch size for translation
        temperature: Model temperature (0.0-2.0)
        top_p: Model top_p (0.0-1.0)
        stream: Stream translation progress
        debug: Enable debug output
        extra_context: Additional context for translation (movie title, etc.)
        progress_callback: Optional callback for progress updates

    Returns:
        Path to translated file, or None on failure

    Raises:
        ConfigurationError: Invalid provider or missing API key
        TranslationError: Translation execution failed
        FileNotFoundError: Input file not found
    """
    # Validate input file exists
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Validate provider
    provider = provider.lower()
    provider_config = PROVIDER_CONFIGS.get(provider)
    if not provider_config:
        valid_providers = list(PROVIDER_CONFIGS.keys())
        raise ConfigurationError(f"Unknown provider: {provider}. Valid providers: {valid_providers}")

    # Validate target language
    if target_lang not in SUPPORTED_TARGETS:
        raise ConfigurationError(f"Unsupported target language: {target_lang}. Valid: {SUPPORTED_TARGETS}")

    # Load user settings
    settings = load_settings()

    # Resolve API key (not needed for local provider)
    if provider == 'local':
        resolved_api_key = None
    else:
        resolved_api_key = _resolve_api_key(provider_config, api_key)

    # Resolve model
    resolved_model = _resolve_model(provider_config, model, settings)

    # Resolve instruction file
    if instruction_file:
        if not Path(instruction_file).exists():
            raise FileNotFoundError(f"Instruction file not found: {instruction_file}")
        resolved_instruction_file = instruction_file
    else:
        resolved_instruction_file = _resolve_instruction_file(tone)

    # Resolve processing options
    resolved_scene_threshold = scene_threshold if scene_threshold is not None else settings.get('scene_threshold', 60.0)
    resolved_max_batch_size = max_batch_size if max_batch_size is not None else settings.get('max_batch_size', 30)

    # Build provider options
    provider_options = _build_provider_options(
        tone=tone,
        temperature=temperature,
        top_p=top_p,
        settings_model_params=settings.get('model_params')
    )

    # Generate output path if not specified
    if output_path:
        resolved_output_path = Path(output_path)
    else:
        stem = input_file.stem
        # Remove existing language suffix if present
        parts = stem.split('.')
        if len(parts) > 1 and parts[-1] in ['japanese', 'english', 'ja', 'en', 'jp', 'chinese', 'indonesian', 'spanish']:
            stem = '.'.join(parts[:-1])
        resolved_output_path = input_file.parent / f"{stem}.{target_lang}.srt"

    # Log configuration
    logger.info(f"Translation: {input_file.name} -> {target_lang}")
    logger.debug(f"Provider: {provider} ({resolved_model})")
    logger.debug(f"Tone: {tone}")

    # Report progress if callback provided
    if progress_callback:
        progress_callback(f"Translating {input_file.name} from {source_lang} to {target_lang}...")
        progress_callback(f"Provider: {provider} ({resolved_model})")

    # Execute translation
    try:
        # For local provider: start server and use PySubtrans with OpenAI-compatible API
        if provider == 'local':
            from .local_backend import start_local_server, stop_local_server

            # Start local LLM server
            try:
                api_base, _ = start_local_server(
                    model=resolved_model,
                    n_gpu_layers=n_gpu_layers
                )
            except Exception as e:
                raise TranslationError(f"Failed to start local server: {e}")

            # Use Custom Server provider - designed for local OpenAI-compatible servers
            # This uses /v1/chat/completions endpoint which llama-cpp-python supports
            local_provider_config = {
                'pysubtrans_name': 'Custom Server',
                'server_address': api_base.replace('/v1', ''),  # Custom Server adds endpoint itself
                'endpoint': '/v1/chat/completions',
                'supports_conversation': True,
                'supports_system_messages': True,
            }

            try:
                result_path = translate_subtitle(
                    input_path=str(input_file),
                    output_path=resolved_output_path,
                    provider_config=local_provider_config,
                    model='local',
                    api_key='',  # Local server doesn't require API key
                    source_lang=source_lang,
                    target_lang=target_lang,
                    instruction_file=resolved_instruction_file,
                    scene_threshold=resolved_scene_threshold,
                    max_batch_size=resolved_max_batch_size,
                    stream=stream,
                    debug=debug,
                    provider_options=provider_options,
                    extra_context=extra_context,
                    emit_raw_output=True
                )
            finally:
                # Always stop server when done
                stop_local_server()
        else:
            result_path = translate_subtitle(
                input_path=str(input_file),
                output_path=resolved_output_path,
                provider_config=provider_config,
                model=resolved_model,
                api_key=resolved_api_key,
                source_lang=source_lang,
                target_lang=target_lang,
                instruction_file=resolved_instruction_file,
                scene_threshold=resolved_scene_threshold,
                max_batch_size=resolved_max_batch_size,
                stream=stream,
                debug=debug,
                provider_options=provider_options,
                extra_context=extra_context,
                emit_raw_output=True  # Always emit progress to stderr (CLI parity)
            )

        if result_path:
            logger.info(f"Translation complete: {result_path}")
            if progress_callback:
                progress_callback(f"Translation complete: {Path(result_path).name}")
            return Path(result_path) if isinstance(result_path, str) else result_path
        else:
            raise TranslationError("Translation returned no result")

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        if debug:
            import traceback
            traceback.print_exc()
        raise TranslationError(f"Translation failed: {e}") from e
