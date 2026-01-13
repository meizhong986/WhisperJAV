"""
whisperjav-translate: Thin wrapper around PySubtrans for subtitle translation.

Usage:
    whisperjav-translate -i input.srt [OPTIONS]
"""

import argparse
import logging
import os
import sys
import io
from pathlib import Path
from typing import Optional

# Fix stdout/stderr encoding for Windows to handle unicode in file paths
# Critical for Japanese, Korean, Chinese file names
def _ensure_utf8_console():
    """Ensure stdout and stderr use UTF-8 encoding."""
    if sys.stdout is not None and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding.lower() != 'utf-8'):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else io.BufferedWriter(io.FileIO(1, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass

    if sys.stderr is not None and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding.lower() != 'utf-8'):
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer if hasattr(sys.stderr, 'buffer') else io.BufferedWriter(io.FileIO(2, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass

# Apply fix at module level
_ensure_utf8_console()

from .providers import PROVIDER_CONFIGS, SUPPORTED_SOURCES, SUPPORTED_TARGETS
from .core import translate_subtitle
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
                        help="Input subtitle file(s) or directory (.srt). Supports: -i file.srt, -i dir/, -i a.srt b.srt")

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

    # Validate and collect input files (supports multiple inputs, directories, and mixed)
    files_to_process = []
    for input_item in args.input:
        input_arg = Path(input_item)
        if not input_arg.exists():
            print(f"Error: Input not found: {input_arg}", file=sys.stderr)
            sys.exit(1)

        if input_arg.is_dir():
            # Directory: glob for .srt files
            srt_files = sorted(list(input_arg.glob("*.srt")))
            if not srt_files:
                print(f"Warning: No .srt files found in directory: {input_arg}", file=sys.stderr)
            else:
                files_to_process.extend(srt_files)
                print(f"Found {len(srt_files)} SRT files in {input_arg}", file=sys.stderr)
        else:
            # Single file
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

    # Override api_base if --endpoint provided (create copy to avoid modifying original)
    if hasattr(args, 'endpoint') and args.endpoint:
        provider_config = dict(provider_config)  # Make a copy
        provider_config['api_base'] = args.endpoint
        # When using custom endpoint, use OpenAI-compatible backend
        if provider_config.get('pysubtrans_name') not in ('OpenAI', 'DeepSeek'):
            provider_config['pysubtrans_name'] = 'OpenAI'

    # Get API key (not needed for local provider)
    if provider_name == 'local':
        api_key = None
    else:
        api_key = args.api_key if hasattr(args, 'api_key') and args.api_key else os.getenv(provider_config['env_var'])
        if not api_key:
            print(f"Error: API key not found. Set {provider_config['env_var']} or use --api-key", file=sys.stderr)
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

    # Batch translation loop
    success_count = 0
    fail_count = 0

    for i, input_path in enumerate(files_to_process):
        # Determine output path for this file
        if hasattr(args, 'output') and args.output:
            # If explicit output provided:
            # - If processing multiple files, treat args.output as directory
            # - If processing single file, treat args.output as file (legacy behavior)
            out_arg = Path(args.output)
            if len(files_to_process) > 1:
                if not out_arg.exists():
                    out_arg.mkdir(parents=True, exist_ok=True)
                # Output filename is same as generated one but in the specified dir
                default_name = Path(generate_output_path(str(input_path), target_lang)).name
                output_path = out_arg / default_name
            else:
                # Single file mode - use exactly what user gave
                output_path = out_arg
        else:
            # Auto-generate
            output_path = Path(generate_output_path(str(input_path), target_lang))

        try:
            if not args.no_progress:
                print(f"[{i+1}/{len(files_to_process)}] Translating {input_path.name} from {source_lang} to {target_lang}...", file=sys.stderr)
                if i == 0:  # Only print provider info once
                    print(f"Provider: {provider_name} ({model})", file=sys.stderr)

            # Dispatch to local backend (via server) or PySubtrans
            if provider_name == 'local':
                from .local_backend import start_local_server, stop_local_server

                # Start local LLM server (only once for batch)
                if i == 0:
                    try:
                        api_base, _ = start_local_server(model=model)
                    except Exception as e:
                        print(f"Error: Failed to start local server: {e}", file=sys.stderr)
                        sys.exit(1)

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
                        input_path=str(input_path),
                        output_path=output_path,
                        provider_config=local_provider_config,
                        model='local',
                        api_key='',
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
                finally:
                    # Stop server after last file
                    if i == len(files_to_process) - 1:
                        stop_local_server()
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
