"""
Core translation logic - PySubtrans wrapper.
"""

import sys
from pathlib import Path


def cap_batch_size_for_context(max_batch_size: int, n_ctx: int) -> int:
    """Cap translation batch size to fit within the LLM context window.

    Local LLMs (e.g., gemma-9b via llama-cpp) have limited context windows.
    PySubtrans formats each subtitle line with numbered headers, original text,
    and translation placeholders. Both the prompt (input) and the expected
    response must fit within n_ctx tokens.

    Japanese text is tokenized at ~2-3 tokens per character by byte-level
    tokenizers (LLaMA, Gemma). A typical subtitle line of 30-50 Japanese
    characters becomes 75-150 tokens. Combined with the response (translated
    text), PySubtrans formatting headers, and <summary>/<scene> tags, we
    budget for worst-case (long) lines:

    Per subtitle line (both directions):
    - Input: ~150 tokens (``#N\\nOriginal>\\n`` + Japanese text, long lines)
    - Output: ~100 tokens (``#N\\nTranslation>\\n`` + translated text)
    - Subtotal: ~250 tokens typical, 500 budget for worst-case long lines

    Fixed overhead:
    - System message: ~500 tokens
    - Translation instructions (standard.txt): ~700 tokens
    - Context from previous batches (<scene>/<summary>): ~500 tokens
    - Response summary/scene tags: ~200 tokens
    - Formatting, preamble, retry margin: ~600 tokens
    - Total: ~2500 tokens

    Revision history:
    - v1.8.6: overhead=2000, per_line=350 → 17 for 8K. Proved unsafe (#196).
    - v1.8.7: overhead=2500, per_line=500 → 11 for 8K. Accounts for long
      Japanese lines, PySubtrans context, and response summary tags.

    Results by context size:
    - 8K (8192):  11 lines per batch
    - 16K (16384): 27 lines per batch
    - 32K+: 30 (capped by max_batch_size default)

    Args:
        max_batch_size: Current batch size setting
        n_ctx: LLM context window in tokens

    Returns:
        Capped batch size (may be unchanged if already within limits)
    """
    overhead = 2500  # system message + instructions + context + response tags
    tokens_per_line = 500  # input + output per subtitle line (worst-case long lines)
    safe_max = max(5, (n_ctx - overhead) // tokens_per_line)
    return min(max_batch_size, safe_max)


def compute_max_output_tokens(batch_size: int, n_ctx: int) -> int:
    """Compute max_tokens for local LLM output to prevent context overflow (#196).

    Japanese/CJK tokenization in LLaMA/Gemma BPE tokenizers:
    - Each CJK character encodes as 3 UTF-8 bytes → typically 2-3 BPE tokens/char
    - A long JAV narration line (80-100 Japanese chars) = ~240-300 input tokens
    - We budget 300 tokens/line input as the worst-case (very long narration)

    English output per line:
    - PySubtrans markers (#N\\nTranslation>\\n): ~8 tokens
    - Translated English text for a typical subtitle: ~50-100 tokens
    - Budget: 120 tokens/line output (generous for long translated sentences)

    Fixed per-batch output overhead:
    - PySubtrans appends <summary>, <scene>, <synopsis> tags after translations
    - These can consume 200-400 tokens depending on model verbosity
    - Budget: 500 tokens fixed overhead

    Strategy: clamp to the smaller of (available context after worst-case input)
    and (2× expected output). This prevents finish_reason='length' while
    still allowing the model to be verbose when content is shorter.

    Results:
    - 8K  (batch=11): ~2392 tokens max output (~2× the ~1320 tokens expected)
    - 16K (batch=27): ~5784 tokens max output (~1.5× the ~3740 tokens expected)
    - 32K (batch=30): ~8200 tokens max output (capped by 2× expected)

    Args:
        batch_size: Number of subtitle lines in the batch
        n_ctx: LLM context window in tokens

    Returns:
        max_tokens value to pass to the local LLM server
    """
    overhead = 2500          # matches cap_batch_size_for_context() — same fixed overhead
    input_per_line_cjk = 300  # JAV worst-case: 80-100 JP chars × ~3 BPE tokens/char
    output_per_line_en = 120  # English translation per line incl. PySubtrans markers
    output_fixed_tags = 500   # <summary>/<scene>/<synopsis> tags emitted per batch

    available = n_ctx - overhead - (batch_size * input_per_line_cjk)
    expected = (batch_size * output_per_line_en) + output_fixed_tags
    return max(512, min(available, expected * 2))


def _normalize_api_base(url: str) -> str:
    """Strip API path suffixes — the OpenAI SDK appends them automatically.

    Users sometimes paste full endpoint URLs like
    ``https://api.example.com/v1/chat/completions``.  The ``openai`` SDK
    already appends ``/chat/completions`` to ``base_url``, so passing the
    full path results in a doubled suffix and a 404.
    """
    if not url:
        return url
    for suffix in ('/chat/completions', '/responses', '/completions'):
        if url.rstrip('/').endswith(suffix):
            url = url.rstrip('/')[:-len(suffix)]
    return url.rstrip('/')


def _api_base_to_custom_server(api_base: str) -> tuple:
    """Convert an api_base URL to Custom Server's (server_address, endpoint) pair.

    PySubtrans's CustomClient uses httpx.Client(base_url=server_address) and
    then client.post(endpoint, ...).  An absolute endpoint path (starting with /)
    replaces the base_url path, so server_address must be scheme+host only,
    and endpoint must be the full path including /chat/completions.
    """
    from urllib.parse import urlparse
    normalized = _normalize_api_base(api_base)
    parsed = urlparse(normalized)
    server_address = f"{parsed.scheme}://{parsed.netloc}"
    path = parsed.path.rstrip('/')
    endpoint = f"{path}/chat/completions" if path else "/v1/chat/completions"
    return server_address, endpoint


def translate_subtitle(
    input_path: str,
    output_path: Path,
    provider_config: dict,
    model: str,
    api_key: str,
    source_lang: str = "japanese",
    target_lang: str = "english",
    instruction_file: str = None,
    scene_threshold: float = 120.0,
    max_batch_size: int = 30,
    stream: bool = False,
    debug: bool = False,
    provider_options: dict = None,
    extra_context: str = None,
    emit_raw_output: bool = True
):
    """
    Translate subtitle file using PySubtrans.

    Returns:
        Path to translated file
    """
    try:
        from PySubtrans import (
            init_options,
            init_translator,
            init_translation_provider,
            init_project
        )

        # =========================================================================
        # DIAGNOSTIC: Translation Configuration
        # =========================================================================
        print(f"\n[TRANSLATE] PySubtrans Configuration:", file=sys.stderr)
        print(f"[TRANSLATE]   Input: {input_path}", file=sys.stderr)
        print(f"[TRANSLATE]   Output: {output_path}", file=sys.stderr)
        print(f"[TRANSLATE]   Provider: {provider_config.get('pysubtrans_name', 'unknown')}", file=sys.stderr)
        print(f"[TRANSLATE]   Model: {model}", file=sys.stderr)
        print(f"[TRANSLATE]   Source lang: {source_lang} -> Target lang: {target_lang}", file=sys.stderr)
        print(f"[TRANSLATE]   Max batch size: {max_batch_size}", file=sys.stderr)
        print(f"[TRANSLATE]   Scene threshold: {scene_threshold}s", file=sys.stderr)
        print(f"[TRANSLATE]   Streaming: {stream}", file=sys.stderr)

        # Log provider-specific settings
        if 'server_address' in provider_config:
            print(f"[TRANSLATE]   Server address: {provider_config['server_address']}", file=sys.stderr)
        if 'endpoint' in provider_config:
            print(f"[TRANSLATE]   Endpoint: {provider_config['endpoint']}", file=sys.stderr)
        if 'api_base' in provider_config:
            print(f"[TRANSLATE]   API base: {provider_config['api_base']}", file=sys.stderr)

        # Build prompt
        prompt = f"Translate these subtitles from {source_lang} into {target_lang}."
        if extra_context:
            prompt += "\n" + extra_context

        # Build provider options
        opt_kwargs = {
            'provider': provider_config['pysubtrans_name'],
            'model': model,
            'api_key': api_key,
            'target_language': target_lang.capitalize(),
            'prompt': prompt,
            'preprocess_subtitles': True,
            'scene_threshold': scene_threshold,
            'max_batch_size': max_batch_size,
            'postprocess_translation': True
        }

        if 'api_base' in provider_config:
            opt_kwargs['api_base'] = _normalize_api_base(provider_config['api_base'])
        if stream:
            opt_kwargs['stream_responses'] = True

        # Custom Server provider settings (for local LLM)
        if 'server_address' in provider_config:
            opt_kwargs['server_address'] = provider_config['server_address']
        if 'endpoint' in provider_config:
            opt_kwargs['endpoint'] = provider_config['endpoint']
        if 'supports_conversation' in provider_config:
            opt_kwargs['supports_conversation'] = provider_config['supports_conversation']
        if 'supports_system_messages' in provider_config:
            opt_kwargs['supports_system_messages'] = provider_config['supports_system_messages']
        if 'max_tokens' in provider_config:
            opt_kwargs['max_tokens'] = provider_config['max_tokens']
        if 'max_completion_tokens' in provider_config:
            opt_kwargs['max_completion_tokens'] = provider_config['max_completion_tokens']
        if 'supports_streaming' in provider_config:
            opt_kwargs['supports_streaming'] = provider_config['supports_streaming']

        # Merge provider-specific options
        if provider_options:
            opt_kwargs.update(provider_options)
            print(f"[TRANSLATE]   Provider options: temperature={provider_options.get('temperature')}, top_p={provider_options.get('top_p')}", file=sys.stderr)

        # =========================================================================
        # DIAGNOSTIC: Final opt_kwargs (what PySubtrans actually receives)
        # =========================================================================
        if debug:
            print(f"[TRANSLATE] Full opt_kwargs for PySubtrans:", file=sys.stderr)
            for k, v in opt_kwargs.items():
                # Don't log API key
                if k == 'api_key':
                    print(f"[TRANSLATE]   {k}: {'*' * 8 if v else '(empty)'}", file=sys.stderr)
                elif k == 'prompt':
                    print(f"[TRANSLATE]   {k}: {v[:50]}...", file=sys.stderr)
                else:
                    print(f"[TRANSLATE]   {k}: {v}", file=sys.stderr)

        # Initialize options and provider
        print(f"[TRANSLATE] Initializing PySubtrans options...", file=sys.stderr)
        options = init_options(**opt_kwargs)

        # Initialize provider
        print(f"[TRANSLATE] Initializing provider: {provider_config['pysubtrans_name']}...", file=sys.stderr)
        try:
            provider = init_translation_provider(provider_config['pysubtrans_name'], options)
        except Exception as init_err:
            if "Unknown translation provider" in str(init_err):
                # PySubtrans silently skips providers whose SDK isn't installed.
                # Surface the likely missing package so users know what to install.
                _PROVIDER_DEPS = {
                    'Gemini': 'google-genai',
                    'Claude': 'anthropic',
                    'OpenAI': 'openai',
                    'DeepSeek': 'openai',
                    'OpenRouter': 'openai',
                    'Custom Server': None,
                }
                pkg = _PROVIDER_DEPS.get(provider_config['pysubtrans_name'])
                hint = f"  Hint: install the provider SDK:  pip install {pkg}" if pkg else ""
                raise RuntimeError(
                    f"PySubtrans does not have the '{provider_config['pysubtrans_name']}' provider registered.\n"
                    f"  This usually means its SDK package is not installed.\n"
                    f"{hint}"
                ) from init_err
            raise

        # Validate provider settings
        if hasattr(provider, 'ValidateSettings') and not provider.ValidateSettings():
            msg = getattr(provider, 'validation_message', 'Invalid provider settings')
            print(f"[TRANSLATE] ERROR: Provider validation failed: {msg}", file=sys.stderr)
            return None
        print(f"[TRANSLATE]   Provider initialized and validated", file=sys.stderr)

        # Log provider internals for debugging (if available)
        if debug:
            if hasattr(provider, 'client') and provider.client:
                client = provider.client
                if hasattr(client, 'settings'):
                    settings = client.settings
                    print(f"[TRANSLATE]   Client timeout: {getattr(settings, 'timeout', 'unknown')}", file=sys.stderr)
                    print(f"[TRANSLATE]   Client server_address: {getattr(settings, 'server_address', 'unknown')}", file=sys.stderr)
                    print(f"[TRANSLATE]   Client endpoint: {getattr(settings, 'endpoint', 'unknown')}", file=sys.stderr)

        # Initialize project (PySubtrans 1.5.x expects subtitle path in 'filepath' kwarg)
        print(f"[TRANSLATE] Loading subtitle project...", file=sys.stderr)
        project = init_project(options, filepath=str(input_path), persistent=True)

        # Check if resuming from existing project
        if hasattr(project, 'existing_project') and project.existing_project:
            print("[TRANSLATE]   Resuming from existing project file (.subtrans)", file=sys.stderr)
            if hasattr(project, 'subtitles') and project.subtitles:
                if hasattr(project.subtitles, 'any_translated') and project.subtitles.any_translated:
                    print("[TRANSLATE]   Found previously translated content - will skip already-translated batches", file=sys.stderr)
        else:
            print(f"[TRANSLATE]   New project created", file=sys.stderr)

        # Log subtitle count
        if hasattr(project, 'subtitles') and project.subtitles:
            subtitle_count = len(project.subtitles) if hasattr(project.subtitles, '__len__') else 'unknown'
            print(f"[TRANSLATE]   Subtitle lines: {subtitle_count}", file=sys.stderr)

        # Set instructions if provided
        if instruction_file is not None and hasattr(project, 'SetInstructions'):
            try:
                project.SetInstructions(str(instruction_file))
                print(f"[TRANSLATE]   Instructions loaded from: {instruction_file}", file=sys.stderr)
            except Exception as e:
                print(f"[TRANSLATE]   Warning: Could not load instructions: {e}", file=sys.stderr)

        # Register auto-save handler to save project after each batch
        if hasattr(project, 'events') and hasattr(project.events, 'batch_translated'):
            def _auto_save_handler(sender, **kwargs):
                """Auto-save project state after each translated batch."""
                try:
                    if hasattr(project, 'SaveProject'):
                        project.SaveProject()
                except Exception:
                    pass  # Don't fail translation due to save errors

            project.events.batch_translated.connect(_auto_save_handler)

        # Initialize translator and translate
        print(f"[TRANSLATE] Initializing translator...", file=sys.stderr)
        translator = init_translator(options, translation_provider=provider)

        # Enable resume mode to skip already-translated batches when resuming
        # This is critical for interrupted translations - without it, the translator
        # will re-translate everything from the beginning even if a .subtrans file exists
        translator.resume = True
        print(f"[TRANSLATE]   Resume mode: enabled (will skip already-translated batches)", file=sys.stderr)

        if emit_raw_output:
            def _emit_raw(message):
                if message is None:
                    return
                try:
                    print(message, file=sys.stderr, flush=True)
                except Exception:
                    pass

            def _make_wrapper():
                def _wrapper(sender, message=None, **kwargs):
                    msg = message
                    if msg is None and kwargs:
                        msg = kwargs.get('message')
                        if msg is None and kwargs:
                            msg = " ".join(str(v) for v in kwargs.values())
                    _emit_raw(msg)
                return _wrapper

            translator.events._default_error_wrapper = _make_wrapper()
            translator.events._default_warning_wrapper = _make_wrapper()
            translator.events._default_info_wrapper = _make_wrapper()

            # Connect the wrappers to the Blinker signals - this was missing!
            # Without this, the wrappers are replaced but never actually receive events
            translator.events.connect_default_loggers()

        # =========================================================================
        # DIAGNOSTIC: Translation Start
        # =========================================================================
        import time as _time
        _translation_start = _time.time()
        _translation_success = False
        print(f"\n[TRANSLATE] Starting translation...", file=sys.stderr)
        print(f"[TRANSLATE]   This may take several minutes depending on batch size and model speed.", file=sys.stderr)
        print(f"[TRANSLATE]   If using local LLM, watch for timeout errors (consider smaller batch size).", file=sys.stderr)
        print(f"", file=sys.stderr)

        # Translate subtitles (with timing regardless of success/failure)
        try:
            project.TranslateSubtitles(translator)
            _translation_success = True
        finally:
            _translation_elapsed = _time.time() - _translation_start
            if _translation_success:
                print(f"\n[TRANSLATE] Translation completed in {_translation_elapsed:.1f}s", file=sys.stderr)
            else:
                print(f"\n[TRANSLATE] Translation FAILED after {_translation_elapsed:.1f}s", file=sys.stderr)

        # Save final project state after successful translation
        print(f"[TRANSLATE] Saving project state...", file=sys.stderr)
        if hasattr(project, 'SaveProject'):
            try:
                project.SaveProject()
                print(f"[TRANSLATE]   Project state saved (.subtrans file)", file=sys.stderr)
            except Exception as e:
                print(f"[TRANSLATE]   Warning: Could not save project state: {e}", file=sys.stderr)

        # Save translation
        print(f"[TRANSLATE] Saving translated subtitles...", file=sys.stderr)
        saved_path = None
        try:
            saved_path = project.SaveTranslation(str(output_path))
        except TypeError:
            saved_path = project.SaveTranslation()
        except Exception as e:
            print(f"[TRANSLATE]   Warning: SaveTranslation error: {e}", file=sys.stderr)
            saved_path = None

        # Convert to Path if needed
        if isinstance(saved_path, (str, Path)):
            output_path = Path(saved_path)

        # =========================================================================
        # DIAGNOSTIC: Final Summary
        # =========================================================================
        print(f"", file=sys.stderr)
        print(f"[TRANSLATE] " + "=" * 50, file=sys.stderr)
        print(f"[TRANSLATE]   TRANSLATION COMPLETE", file=sys.stderr)
        print(f"[TRANSLATE]   Output: {output_path}", file=sys.stderr)
        print(f"[TRANSLATE] " + "=" * 50, file=sys.stderr)
        print(f"", file=sys.stderr)

        return output_path

    except Exception as e:
        # Save project state before propagating error
        if 'project' in locals() and hasattr(project, 'SaveProject'):
            try:
                project.SaveProject()
                print("WARNING: Translation failed, project state saved to .subtrans file", file=sys.stderr)
            except Exception as save_err:
                if debug:
                    print(f"Warning: Failed to save project on error: {save_err}", file=sys.stderr)

        if debug:
            import traceback
            traceback.print_exc()
        raise
