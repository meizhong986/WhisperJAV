"""
whisperjav-translate: Thin wrapper around PySubtrans for subtitle translation.

Usage:
    whisperjav-translate -i input.srt [OPTIONS]

This is the entry point for the 'whisperjav-translate' command.
Requires the [translate] extra: pip install whisperjav[translate]
"""

# ===========================================================================
# UTF-8 MODE — must be the very first thing before any library imports.
# On Chinese Windows (GBK locale), PySubtrans crashes with 'gbk' codec
# errors when processing Chinese translations. Relaunch in UTF-8 mode so
# that open() defaults to UTF-8 across the entire process. See #190.
# ===========================================================================
import os, sys  # noqa: E401 — intentionally early, minimal imports
if os.name == 'nt' and not getattr(sys.flags, 'utf8_mode', False):
    from whisperjav.utils.console import relaunch_for_utf8
    relaunch_for_utf8('whisperjav.translate.cli')

# ===========================================================================
# EARLY SETUP - Must be before any library imports
# ===========================================================================
from whisperjav.utils.console import setup_console, print_missing_extra_error
setup_console()

# Check for translate extra dependencies before importing them
def _check_translate_dependencies():
    """Check if translation dependencies are installed."""
    missing = []

    try:
        import PySubtrans  # noqa: F401 - Note: package name is 'pysubtrans' but import is 'PySubtrans'
    except ImportError:
        missing.append("pysubtrans")

    try:
        import openai  # noqa: F401
    except ImportError:
        missing.append("openai")

    if missing:
        print_missing_extra_error(
            extra_name="translate",
            missing_packages=missing,
            feature_description="AI-powered subtitle translation using PySubtrans"
        )
        sys.exit(1)

_check_translate_dependencies()

# ===========================================================================
# Standard imports (after dependency check)
# ===========================================================================
import argparse
import logging
import os
from pathlib import Path
from typing import Optional

from .providers import PROVIDER_CONFIGS, SUPPORTED_SOURCES, SUPPORTED_TARGETS
from .core import translate_subtitle, _normalize_api_base, _api_base_to_custom_server, cap_batch_size_for_context, compute_max_output_tokens
from .instructions import get_instruction_content, get_cache_dir
from .settings import load_settings, create_default_settings, show_settings, get_settings_path, resolve_config
from .configure import configure_command

import tempfile


def generate_output_path(input_path: str, target_lang: str) -> str:
    """Generate output filename for translated subtitle."""
    input_path = Path(input_path)
    stem = input_path.stem

    # If stem has language code, replace it
    parts = stem.split('.')
    if len(parts) > 1:
        # Remove last part if it looks like a language code
        if parts[-1] in ['japanese', 'english', 'ja', 'en', 'jp']:
            stem = '.'.join(parts[:-1])

    output_name = f"{stem}.{target_lang}.srt"
    return str(input_path.parent / output_name)


def build_provider_options(args, settings_model_params: dict, effective_tone: str) -> dict:
    """Build provider options with correct precedence and tone-aware defaults.

    Precedence: CLI > settings > defaults (tone-aware).
    Defaults:
      - standard: temperature=0.5, top_p=0.9
      - pornify:  temperature=1.2, top_p=0.9
    """
    def _to_float(val):
        try:
            return float(val)
        except Exception:
            return None

    # 1) Start from tone-aware defaults
    if effective_tone == 'pornify':
        temperature = 1.2
        top_p = 0.9
    else:
        temperature = 0.5
        top_p = 0.9

    # 2) Apply settings overrides (if provided and not None)
    if settings_model_params:
        s_temp = settings_model_params.get('temperature', None)
        s_top_p = settings_model_params.get('top_p', None)
        if s_temp is not None:
            f = _to_float(s_temp)
            if f is not None:
                temperature = f
        if s_top_p is not None:
            f = _to_float(s_top_p)
            if f is not None:
                top_p = f

    # 3) Apply CLI overrides last (highest precedence)
    if hasattr(args, 'temperature') and args.temperature is not None:
        temperature = args.temperature
    if hasattr(args, 'top_p') and args.top_p is not None:
        top_p = args.top_p

    # 4) Clamp ranges and warn when out-of-range
    if temperature is not None and not (0.0 <= temperature <= 2.0):
        print(f"Warning: temperature {temperature} outside recommended range [0.0, 2.0]", file=sys.stderr)
    if top_p is not None and not (0.0 <= top_p <= 1.0):
        print(f"Warning: top_p {top_p} outside valid range [0.0, 1.0]", file=sys.stderr)
    temperature = max(0.0, min(2.0, float(temperature))) if temperature is not None else None
    top_p = max(0.0, min(1.0, float(top_p))) if top_p is not None else None

    provider_opts = {}
    if temperature is not None:
        provider_opts['temperature'] = temperature
    if top_p is not None:
        provider_opts['top_p'] = top_p

    # Add rate limiting options from CLI
    if hasattr(args, 'rate_limit') and args.rate_limit is not None:
        provider_opts['rate_limit'] = args.rate_limit
    if hasattr(args, 'max_retries') and args.max_retries is not None:
        provider_opts['max_retries'] = args.max_retries
    if hasattr(args, 'backoff_time') and args.backoff_time is not None:
        provider_opts['backoff_time'] = args.backoff_time

    return provider_opts


def build_extra_context(args) -> str:
    """Build extra context string from movie metadata."""
    context_parts = []

    if hasattr(args, 'movie_title') and getattr(args, 'movie_title'):
        context_parts.append(f"Movie title: {args.movie_title}")

    if hasattr(args, 'actress') and getattr(args, 'actress'):
        context_parts.append(f"Actress: {args.actress}")

    if hasattr(args, 'movie_plot') and getattr(args, 'movie_plot'):
        context_parts.append(f"Plot: {args.movie_plot}")

    return '\n'.join(context_parts) if context_parts else ""


def resolve_instruction_file_or_content(args, merged: dict) -> Optional[str]:
    """Resolve instruction file path or fetch content."""
    # Check if explicit instruction file provided
    if hasattr(args, 'instructions_file') and args.instructions_file:
        instr_path = Path(args.instructions_file)
        if not instr_path.exists():
            print(f"Error: Instruction file not found: {instr_path}", file=sys.stderr)
            return None
        return str(instr_path)

    # Otherwise, fetch instruction content based on tone
    source_lang = merged.get('source', 'japanese')
    tone = merged.get('tone', 'standard')

    # Get instruction content
    instruction_content = get_instruction_content(tone=tone, refresh=False)

    if instruction_content:
        # Save to temp file
        temp_dir = Path(tempfile.gettempdir()) / 'whisperjav_translate'
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f'instructions_{tone}.txt'

        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(instruction_content)

        return str(temp_file)

    return None


def main():
    """Main CLI entry point."""
    # Load user settings
    settings = load_settings()

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Translate subtitles using AI (PySubtrans wrapper)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('-i', '--input', nargs='+', required=True,
                        help='Input subtitle file(s), directory, or glob pattern (.srt). '
                             'Supports: -i file.srt, -i dir/, -i "*.srt", -i a.srt b.srt')
    parser.add_argument('--recursive', '-r', action='store_true',
                        help="Search directories recursively for .srt files")

    # Core translation options
    translation_group = parser.add_argument_group("Translation Options")
    translation_group.add_argument(
        '--provider',
        choices=list(PROVIDER_CONFIGS.keys()),
        default=None,
        help=f"AI provider (default: {settings.get('provider', 'deepseek')})"
    )
    translation_group.add_argument(
        '--model',
        help="Model override. For --provider local: llama-8b (6GB VRAM, default), "
             "gemma-9b (8GB VRAM, best), llama-3b (3GB VRAM), auto"
    )
    translation_group.add_argument(
        '-t', '--target',
        choices=list(SUPPORTED_TARGETS),
        default=None,
        help=f"Target language (default: {settings.get('target_language', 'english')})"
    )
    translation_group.add_argument(
        '--source',
        choices=list(SUPPORTED_SOURCES),
        default='japanese',
        help="Source language (default: japanese)"
    )
    translation_group.add_argument(
        '--tone',
        choices=['standard', 'pornify'],
        default=None,
        help=f"Translation tone/style (default: {settings.get('tone', 'standard')})"
    )

    # API configuration
    api_group = parser.add_argument_group("API Configuration")
    api_group.add_argument(
        '--api-key',
        help="API key (or set via environment variable)"
    )
    api_group.add_argument(
        '--endpoint',
        help="Custom API endpoint URL (for OpenAI-compatible APIs)"
    )
    api_group.add_argument(
        '--temperature',
        type=float,
        help="Model temperature (0.0-2.0)"
    )
    api_group.add_argument(
        '--top-p',
        type=float,
        help="Model top_p (0.0-1.0)"
    )
    api_group.add_argument(
        '--rate-limit',
        type=float,
        help="Max API requests per minute (e.g., 10 = 6 sec between requests)"
    )
    api_group.add_argument(
        '--max-retries',
        type=int,
        default=3,
        help="Max retries on API failure (default: 3)"
    )
    api_group.add_argument(
        '--backoff-time',
        type=float,
        default=5.0,
        help="Backoff time in seconds after failure (default: 5.0)"
    )

    # Processing options
    process_group = parser.add_argument_group("Processing Options")
    process_group.add_argument(
        '--scene-threshold',
        type=float,
        help=f"Scene threshold in seconds (default: {settings.get('scene_threshold', 60.0)})"
    )
    process_group.add_argument(
        '--max-batch-size',
        type=int,
        help=f"Max batch size (default: {settings.get('max_batch_size', 30)})"
    )
    process_group.add_argument(
        '--translate-gpu-layers',
        type=int,
        default=-1,
        help="GPU layers for local LLM: -1=all (default), 0=CPU only, N=specific count"
    )

    # Instructions
    instr_group = parser.add_argument_group("Instructions")
    instr_group.add_argument(
        '--instructions-file',
        help="Custom instruction file path"
    )

    # Metadata
    meta_group = parser.add_argument_group("Movie Metadata (Optional)")
    meta_group.add_argument('--movie-title', help="Movie title for context")
    meta_group.add_argument('--actress', help="Actress name for context")
    meta_group.add_argument('--movie-plot', help="Plot summary for context")

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        '-o', '--output',
        help="Output file path (auto-generated if not specified)"
    )
    output_group.add_argument(
        '--stream',
        action='store_true',
        help="Stream translation progress to stderr"
    )
    output_group.add_argument(
        '--no-progress',
        action='store_true',
        help="Disable all progress output"
    )

    # Ollama options
    ollama_group = parser.add_argument_group("Ollama Options")
    ollama_group.add_argument(
        '--ollama-url',
        help="Custom Ollama server URL (default: http://localhost:11434)"
    )
    ollama_group.add_argument(
        '--list-ollama-models',
        action='store_true',
        help="List locally available Ollama models and exit"
    )
    ollama_group.add_argument(
        '--ollama-max-tokens', type=int, default=None,
        help="Override auto-computed max output tokens for Ollama translation. "
             "By default, WhisperJAV computes this from context window and batch size. "
             "Use this to manually cap token budget (e.g., 2048, 4096)."
    )
    ollama_group.add_argument(
        '--yes', '-y',
        action='store_true',
        help="Auto-confirm prompts (model downloads, server starts)"
    )

    # Utility commands
    util_group = parser.add_argument_group("Utility Commands")
    util_group.add_argument(
        '--configure',
        action='store_true',
        help="Run interactive configuration wizard"
    )
    util_group.add_argument(
        '--show-settings',
        action='store_true',
        help="Show current settings"
    )
    util_group.add_argument(
        '--create-default-settings',
        action='store_true',
        help="Create default settings file"
    )

    # Debug options
    parser.add_argument('--debug', action='store_true', help="Enable debug output")

    args = parser.parse_args()

    # Setup logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    # Handle utility commands
    if args.configure:
        configure_command()
        return

    if args.show_settings:
        show_settings()
        return

    if args.create_default_settings:
        if create_default_settings():
            print(f"Default settings created at: {get_settings_path()}")
        return

    if args.list_ollama_models:
        from .ollama_manager import OllamaManager
        mgr = OllamaManager(base_url=getattr(args, 'ollama_url', None))
        if not mgr.detect_server():
            print("Ollama server not running.", file=sys.stderr)
            sys.exit(1)
        models = mgr.list_models()
        if not models:
            print("No models found. Pull one with: ollama pull gemma3:12b", file=sys.stderr)
        else:
            print(f"{'Model':<30s}  {'Size':>10s}")
            print(f"{'-'*30}  {'-'*10}")
            for m in models:
                size = m.get('details', {}).get('parameter_size', '?')
                print(f"  {m['name']:<30s}  {size:>8s}")
        return

    # Validate and collect input files (supports files, directories, and glob patterns)
    import glob as glob_module

    files_to_process = []
    seen_paths = set()  # Deduplication via resolved paths

    for input_item in args.input:
        input_str = str(input_item)

        # 1. Glob pattern detection (contains * or ?)
        if '*' in input_str or '?' in input_str:
            matched = sorted(glob_module.glob(input_str, recursive=True))
            srt_matched = [Path(f) for f in matched if f.lower().endswith('.srt')]
            if not srt_matched:
                print(f"Warning: No .srt files matched pattern: {input_item}", file=sys.stderr)
            else:
                for f in srt_matched:
                    resolved = f.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        files_to_process.append(f)
                print(f"Matched {len(srt_matched)} SRT files for: {input_item}", file=sys.stderr)
            continue

        # 2. Existing path-based handling
        input_arg = Path(input_item)
        if not input_arg.exists():
            print(f"Error: Input not found: {input_arg}", file=sys.stderr)
            sys.exit(1)

        if input_arg.is_dir():
            # Directory: glob for .srt files (shallow or recursive)
            glob_method = input_arg.rglob if getattr(args, 'recursive', False) else input_arg.glob
            srt_files = sorted(list(glob_method("*.srt")))
            if not srt_files:
                depth = "recursively in" if getattr(args, 'recursive', False) else "in"
                print(f"Warning: No .srt files found {depth}: {input_arg}", file=sys.stderr)
            else:
                for f in srt_files:
                    resolved = f.resolve()
                    if resolved not in seen_paths:
                        seen_paths.add(resolved)
                        files_to_process.append(f)
                depth_note = " (recursive)" if getattr(args, 'recursive', False) else ""
                print(f"Found {len(srt_files)} SRT files in {input_arg}{depth_note}", file=sys.stderr)
        else:
            # Single file
            resolved = input_arg.resolve()
            if resolved not in seen_paths:
                seen_paths.add(resolved)
                files_to_process.append(input_arg)

    if not files_to_process:
        print("Error: No valid .srt files to process", file=sys.stderr)
        sys.exit(1)

    # Merge configuration
    merged = resolve_config(args, settings)

    # Get provider config
    provider_name = merged.get('provider', 'deepseek')
    provider_config = PROVIDER_CONFIGS.get(provider_name)
    if not provider_config:
        print(f"Error: Unknown provider: {provider_name}", file=sys.stderr)
        sys.exit(1)

    # Override endpoint if --endpoint provided (create copy to avoid modifying original)
    if hasattr(args, 'endpoint') and args.endpoint:
        provider_config = dict(provider_config)  # Make a copy
        normalized = _normalize_api_base(args.endpoint)
        psn = provider_config.get('pysubtrans_name')
        if psn in ('OpenAI', 'DeepSeek'):
            # These providers handle api_base natively via their SDKs
            provider_config['api_base'] = normalized
        else:
            # Route through Custom Server for reliable /chat/completions
            # access without reasoning model misclassification (#178)
            server_addr, endpoint_path = _api_base_to_custom_server(args.endpoint)
            provider_config['pysubtrans_name'] = 'Custom Server'
            provider_config['server_address'] = server_addr
            provider_config['endpoint'] = endpoint_path
            provider_config.pop('api_base', None)

    # Get API key (not needed for local/custom/ollama providers)
    if provider_name in ('local', 'custom', 'ollama'):
        api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else ''
    else:
        env_var = provider_config.get('env_var')
        api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else (os.getenv(env_var) if env_var else None)
        if not api_key:
            print(f"Error: API key not found. Set {env_var} or use --api-key", file=sys.stderr)
            sys.exit(1)

    # Get model
    model = merged.get('model') or provider_config['model']

    # Get languages
    source_lang = merged.get('source', 'japanese')
    target_lang = merged.get('target_language', 'english')

    # Resolve instruction file
    instruction_file = resolve_instruction_file_or_content(args, merged)

    # Build provider options (tone-aware defaults, settings, then CLI)
    effective_tone = merged.get('tone') or 'standard'
    provider_options = build_provider_options(args, merged.get('model_params', {}), effective_tone)

    # Build extra context
    extra_context = build_extra_context(args)

    # =========================================================================
    # DIAGNOSTIC: CLI Configuration Summary
    # =========================================================================
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"  WHISPERJAV TRANSLATE - CLI Configuration", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    print(f"  Files to process: {len(files_to_process)}", file=sys.stderr)
    print(f"  Provider: {provider_name}", file=sys.stderr)
    print(f"  Model: {model}", file=sys.stderr)
    print(f"  Source: {source_lang} -> Target: {target_lang}", file=sys.stderr)
    print(f"  Tone: {effective_tone}", file=sys.stderr)
    print(f"  Scene threshold: {merged.get('scene_threshold', 60.0)}s", file=sys.stderr)
    print(f"  Max batch size: {merged.get('max_batch_size', 30)}", file=sys.stderr)
    if instruction_file:
        print(f"  Instructions: {instruction_file}", file=sys.stderr)
    print(f"  Provider options: {provider_options}", file=sys.stderr)
    print(f"  Debug: {getattr(args, 'debug', False)}", file=sys.stderr)
    if provider_name == 'local':
        print(
            "\n  [DEPRECATION WARNING] --provider local is deprecated as of v1.8.10.",
            file=sys.stderr,
        )
        print(
            "  Please migrate to --provider ollama for better stability and model support.",
            file=sys.stderr,
        )
        print(
            "  The local LLM server (llama-cpp-python) will be removed in v1.9.0.\n",
            file=sys.stderr,
        )
        n_gpu_layers = getattr(args, 'translate_gpu_layers', -1)
        print(f"  NOTE: Local LLM provider - server will be started", file=sys.stderr)
        print(f"  GPU layers: {n_gpu_layers} (-1=all, 0=CPU only)", file=sys.stderr)
    if provider_name == 'ollama':
        print(f"  NOTE: Ollama provider — smart detection + auto-start", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # =========================================================================
    # Ollama provider: OllamaManager orchestration
    # =========================================================================
    if provider_name == 'ollama':
        from .ollama_manager import (
            OllamaManager, OllamaNotInstalledError,
            OllamaNotRunningError, ModelNotAvailableError,
        )

        mgr = OllamaManager(base_url=getattr(args, 'ollama_url', None))
        interactive = sys.stdin.isatty() and not getattr(args, 'yes', False)
        auto_pull = getattr(args, 'yes', False)

        try:
            readiness = mgr.ensure_ready(
                model=model,
                auto_start=True,
                auto_pull=auto_pull,
                interactive=interactive,
            )
        except OllamaNotInstalledError as e:
            print(f"\n{e}", file=sys.stderr)
            sys.exit(1)
        except OllamaNotRunningError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)
        except ModelNotAvailableError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            sys.exit(1)

        # Dynamic config from model metadata
        model = readiness['model']
        ollama_n_ctx = readiness['num_ctx']
        _user_batch = merged.get('max_batch_size', readiness['batch_size'])
        ollama_batch_size = cap_batch_size_for_context(_user_batch, ollama_n_ctx)
        ollama_max_tokens = compute_max_output_tokens(ollama_batch_size, ollama_n_ctx)
        _user_max_tokens = getattr(args, 'ollama_max_tokens', None)
        if _user_max_tokens is not None:
            ollama_max_tokens = _user_max_tokens

        if 'num_ctx' not in provider_options:
            provider_options['num_ctx'] = ollama_n_ctx

        # Ollama temperature override for local LLMs.
        # build_provider_options() always sets temperature (0.5 standard / 1.2
        # pornify) — these are cloud-provider defaults. Local models need lower
        # temperature (0.3 best-fit across 10 tested models). Override the
        # generic default with the Ollama-optimized value, but ONLY when:
        #   - User did NOT set --temperature on CLI
        #   - User did NOT set temperature in settings file
        #   - Tone is NOT pornify (pornify's 1.2 is an intentional user choice)
        _user_set_temp_cli = hasattr(args, 'temperature') and args.temperature is not None
        _user_set_temp_settings = bool(merged.get('model_params', {}).get('temperature'))
        if not _user_set_temp_cli and not _user_set_temp_settings:
            if effective_tone != 'pornify':
                provider_options['temperature'] = readiness.get('temperature', 0.3)

        provider_config = dict(provider_config)
        provider_config['max_tokens'] = ollama_max_tokens
        if readiness.get('base_url'):
            server_addr, endpoint_path = _api_base_to_custom_server(readiness['base_url'])
            provider_config['server_address'] = server_addr
            provider_config['endpoint'] = endpoint_path

        # Override supports_system_messages based on actual model template.
        # Models with bare templates (e.g. {{ .Prompt }}) silently drop
        # system messages — instructions must go in the user message instead.
        if not readiness.get('supports_system_messages', True):
            provider_config['supports_system_messages'] = False

        # Qwen3-family thinking models: their output goes to the 'reasoning'
        # field with empty 'content'. PySubtrans only reads 'content', so all
        # translations are silently lost. The '_thinking_model' flag activates
        # a response-parsing patch in core.py that moves reasoning → content.
        if readiness.get('thinking_model'):
            provider_options['_thinking_model'] = True

        if ollama_batch_size < _user_batch:
            print(f"[OLLAMA] Batch size auto-reduced from {_user_batch} to "
                  f"{ollama_batch_size} to fit {ollama_n_ctx}-token context", file=sys.stderr)
        print(f"[OLLAMA] model={model}, num_ctx={ollama_n_ctx}, batch_size={ollama_batch_size}, "
              f"max_tokens={ollama_max_tokens}", file=sys.stderr)
        print(f"[OLLAMA] Final provider_options: temperature={provider_options.get('temperature')}, "
              f"top_p={provider_options.get('top_p')}",
              file=sys.stderr)

    # Batch translation loop
    success_count = 0
    fail_count = 0

    for i, input_path in enumerate(files_to_process):
        # Determine output path for this file
        if hasattr(args, 'output') and args.output:
            # If explicit output provided:
            # - If it's a directory (or looks like one): generate filename inside it
            # - If it's a file path: use exactly what user gave (legacy single-file)
            out_arg = Path(args.output)
            if out_arg.is_dir() or len(files_to_process) > 1:
                if not out_arg.exists():
                    out_arg.mkdir(parents=True, exist_ok=True)
                # Output filename is same as generated one but in the specified dir
                default_name = Path(generate_output_path(str(input_path), target_lang)).name
                output_path = out_arg / default_name
            else:
                # Single file mode with explicit file path
                output_path = out_arg
        else:
            # Auto-generate
            output_path = Path(generate_output_path(str(input_path), target_lang))

        try:
            if not args.no_progress:
                if i > 0 and len(files_to_process) > 1:
                    print(file=sys.stderr)  # Blank line between files in batch
                print(f"Translating [{i+1}/{len(files_to_process)}]: {input_path.name}", file=sys.stderr)
                if i == 0:  # Only print provider info once
                    print(f"Provider: {provider_name} ({model})", file=sys.stderr)

            # Dispatch to local backend (via server) or PySubtrans
            if provider_name == 'local':
                from .local_backend import start_local_server, stop_local_server

                # Start local LLM server (only once for batch)
                if i == 0:
                    n_gpu_layers = getattr(args, 'translate_gpu_layers', -1)
                    print(f"\n[CLI] Starting local LLM server...", file=sys.stderr)
                    print(f"[CLI]   Model: {model}", file=sys.stderr)
                    print(f"[CLI]   GPU layers: {n_gpu_layers} (-1=all, 0=CPU)", file=sys.stderr)
                    try:
                        api_base, server_port, server_diagnostics = start_local_server(model=model, n_gpu_layers=n_gpu_layers)
                        print(f"[CLI]   Server ready at: {api_base}", file=sys.stderr)
                        if server_diagnostics.inference_speed_tps > 0:
                            print(f"[CLI]   Inference speed: {server_diagnostics.inference_speed_tps:.1f} tokens/sec", file=sys.stderr)
                    except Exception as e:
                        print(f"[CLI] ERROR: Failed to start local server: {e}", file=sys.stderr)
                        sys.exit(1)

                    # Auto-cap batch size to fit within local LLM context window.
                    # See #183: default batch_size=30 exceeds 8K context → "Hit API
                    # token limit" + "No matches" from PySubtrans.
                    local_n_ctx = 8192  # matches start_local_server() default
                    _user_batch = merged.get('max_batch_size', 30)
                    local_batch_size = cap_batch_size_for_context(_user_batch, local_n_ctx)
                    if local_batch_size < _user_batch:
                        print(f"[CLI]   NOTE: Batch size auto-reduced from {_user_batch} to "
                              f"{local_batch_size} to fit {local_n_ctx}-token context window", file=sys.stderr)

                    # Compute max_tokens to prevent finish_reason='length' on the server.
                    # CJK tokenization: Japanese chars encode as ~3 BPE tokens/char in
                    # LLaMA/Gemma → long JAV narration lines consume up to 300 input
                    # tokens each. Without a max_tokens cap the server fills all remaining
                    # context with output, truncating mid-translation and breaking the
                    # PySubtrans parser ("No matches found"). See issue #196.
                    local_max_tokens = compute_max_output_tokens(local_batch_size, local_n_ctx)
                    print(f"[CLI]   Output token limit (max_tokens): {local_max_tokens} "
                          f"(JAV/CJK-tuned, prevents context overflow)", file=sys.stderr)

                # Use Custom Server provider - designed for local OpenAI-compatible servers
                # This uses /v1/chat/completions endpoint which llama-cpp-python supports
                server_address = api_base.replace('/v1', '')
                local_provider_config = {
                    'pysubtrans_name': 'Custom Server',
                    'server_address': server_address,
                    'endpoint': '/v1/chat/completions',
                    'supports_conversation': True,
                    'supports_system_messages': True,
                    'max_tokens': local_max_tokens,  # Prevents finish_reason='length' (#196)
                    'supports_streaming': True,       # Required: CustomClient.enable_streaming =
                                                      # stream_responses AND supports_streaming.
                                                      # Without this flag CustomClient silently
                                                      # disables streaming even when requested.
                }
                if i == 0:
                    print(f"[CLI]   PySubtrans config: Custom Server at {server_address}", file=sys.stderr)

                try:
                    result_path = translate_subtitle(
                        input_path=str(input_path),
                        output_path=output_path,
                        provider_config=local_provider_config,
                        model='local',
                        api_key='',
                        source_lang=source_lang,
                        target_lang=target_lang,
                        instruction_file=instruction_file,
                        scene_threshold=merged.get('scene_threshold', 60.0),
                        max_batch_size=local_batch_size,
                        stream=True,  # Always stream for local LLM: non-streaming blocks the
                                      # entire HTTP connection until the last token is generated.
                                      # On slow backends (MPS, CPU) this exceeds the read timeout
                                      # every batch. Streaming delivers tokens incrementally so
                                      # no read timeout can fire. Output is identical. (#196)
                        debug=args.debug,
                        provider_options=provider_options,
                        extra_context=extra_context if extra_context else None,
                        emit_raw_output=not getattr(args, 'no_progress', False)
                    )
                finally:
                    # Stop server after last file
                    if i == len(files_to_process) - 1:
                        print(f"\n[CLI] Stopping local LLM server...", file=sys.stderr)
                        stop_local_server()
                        print(f"[CLI]   Server stopped, GPU memory released", file=sys.stderr)
            elif provider_name == 'ollama':
                result_path = translate_subtitle(
                    input_path=str(input_path),
                    output_path=output_path,
                    provider_config=provider_config,
                    model=model,
                    api_key='',
                    source_lang=source_lang,
                    target_lang=target_lang,
                    instruction_file=instruction_file,
                    scene_threshold=merged.get('scene_threshold', 60.0),
                    max_batch_size=ollama_batch_size,
                    stream=True,  # Always stream for local models (prevents timeout)
                    debug=args.debug,
                    provider_options=provider_options,
                    extra_context=extra_context if extra_context else None,
                    emit_raw_output=not getattr(args, 'no_progress', False)
                )
            else:
                result_path = translate_subtitle(
                    input_path=str(input_path),
                    output_path=output_path,
                    provider_config=provider_config,
                    model=model,
                    api_key=api_key,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    instruction_file=instruction_file,
                    scene_threshold=merged.get('scene_threshold', 60.0),
                    max_batch_size=merged.get('max_batch_size', 30),
                    stream=args.stream if hasattr(args, 'stream') else False,
                    debug=args.debug,
                    provider_options=provider_options,
                    extra_context=extra_context if extra_context else None,
                    emit_raw_output=not getattr(args, 'no_progress', False)
                )

            if result_path:
                # Print output path to stdout (for capture by calling process)
                print(str(result_path))
                if not args.no_progress:
                    print(f"Complete: {result_path.name}", file=sys.stderr)
                success_count += 1
            else:
                print(f"Failed: {input_path.name}", file=sys.stderr)
                fail_count += 1

        except KeyboardInterrupt:
            print("\nBatch translation interrupted", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error translating {input_path.name}: {e}", file=sys.stderr)
            fail_count += 1
            if args.debug:
                import traceback
                traceback.print_exc()
            # Continue to next file for batch robustness
            continue

    if fail_count > 0:
        print(f"\nBatch processing finished with {fail_count} errors.", file=sys.stderr)
        sys.exit(1 if success_count == 0 else 0)


if __name__ == '__main__':
    main()
