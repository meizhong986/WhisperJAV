"""
Core translation logic - PySubtrans wrapper.
"""

import os
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
    and expected output. Previous versions used 2× expected, but this gave models
    too much room to produce verbose/garbled output (e.g., 2392 tokens of garbage
    instead of ~1200 tokens of translation). Tightening to 1× forces concise output
    and reduces "No matches found" failures caused by off-format model verbosity.

    Results:
    - 8K  (batch=11): ~1820 tokens max output (matches expected)
    - 16K (batch=27): ~3740 tokens max output (matches expected)
    - 32K (batch=30): ~4100 tokens max output (matches expected)

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
    return max(512, min(available, expected))


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
        print(f"[TRANSLATE]   Stream requested: {stream} (actual depends on provider)", file=sys.stderr)

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
            print(f"[TRANSLATE]   Extra context: {extra_context[:200]}", file=sys.stderr)

        # Qwen3-family thinking model flag: consumed later by the response
        # parsing patch (after provider init). Remove from provider_options
        # so it doesn't get passed to PySubtrans as an unknown option.
        _is_thinking_model = provider_options.pop('_thinking_model', False) if provider_options else False
        if _is_thinking_model:
            print(f"[TRANSLATE]   Thinking model: YES (will patch response parsing)",
                  file=sys.stderr)

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

        # Pass instruction file to PySubtrans so it can parse the
        # ### prompt / ### instructions / ### retry_instructions sections.
        # Previously this was handled via project.SetInstructions() which
        # does not exist in current PySubtrans — instructions were silently
        # dropped for ALL providers.
        if instruction_file:
            opt_kwargs['instruction_file'] = str(instruction_file)

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
            _sys_msg = provider_config['supports_system_messages']
            print(f"[TRANSLATE]   System messages: {'enabled' if _sys_msg else 'DISABLED (embedded in user msg)'}",
                  file=sys.stderr)
        if 'max_tokens' in provider_config:
            opt_kwargs['max_tokens'] = provider_config['max_tokens']
        if 'max_completion_tokens' in provider_config:
            opt_kwargs['max_completion_tokens'] = provider_config['max_completion_tokens']
        if 'supports_streaming' in provider_config:
            opt_kwargs['supports_streaming'] = provider_config['supports_streaming']

        # Merge provider-specific options
        if provider_options:
            opt_kwargs.update(provider_options)
            _po_display = {k: v for k, v in provider_options.items() if k != 'api_key'}
            print(f"[TRANSLATE]   Provider options: {_po_display}", file=sys.stderr)

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

        # =========================================================================
        # DIAGNOSTIC: Instruction Loading Verification (always-on)
        # =========================================================================
        # Verify that PySubtrans actually loaded and parsed the instructions.
        # The instruction_file dead-path bug (commit 56315f2) went undetected
        # for months because nothing inspected what init_options() produced.
        _loaded_inst_file = options.get('instruction_file')
        _has_instructions = bool(options.get('instructions'))
        _has_retry = bool(options.get('retry_instructions'))
        if instruction_file:
            # We asked for a specific instruction file — verify it was loaded
            if _loaded_inst_file:
                print(f"[TRANSLATE]   Instructions loaded: {_loaded_inst_file}", file=sys.stderr)
                print(f"[TRANSLATE]   Sections: instructions={'YES' if _has_instructions else 'MISSING'}, "
                      f"retry={'YES' if _has_retry else 'MISSING'}", file=sys.stderr)
            else:
                print(f"[TRANSLATE]   WARNING: instruction_file='{instruction_file}' was passed "
                      f"but PySubtrans did not load it!", file=sys.stderr)
        else:
            print(f"[TRANSLATE]   Instructions: (default — no instruction file specified)", file=sys.stderr)

        # Debug-only: instruction content preview
        if debug:
            _inst_text = options.get('instructions', '')
            if _inst_text:
                print(f"[TRANSLATE]   Instructions preview: {_inst_text[:150]}...", file=sys.stderr)
            _prompt_text = options.get('prompt', '')
            if _prompt_text:
                print(f"[TRANSLATE]   Prompt: {_prompt_text[:150]}", file=sys.stderr)

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

        # =========================================================================
        # A2 FIX: Delete stale .subtrans files from previous failed attempts
        # =========================================================================
        # .subtrans files store project state including batch_size and other
        # settings from previous runs. When settings change (e.g., user adjusts
        # batch_size via CLI), the stale .subtrans file overrides the new CLI
        # settings via options.update(project_settings). This is a known bug
        # where "batch size not taking effect" (#212, zhstark's report).
        #
        # Fix: Delete .subtrans if WhisperJAV version changed (indicates user
        # upgraded and old settings are likely stale/incompatible).
        _subtrans_path = Path(str(input_path) + '.subtrans')
        if _subtrans_path.exists():
            _delete_stale = False
            try:
                import json as _json
                with open(_subtrans_path, 'r', encoding='utf-8') as _sf:
                    _subtrans_data = _json.load(_sf)
                _old_version = _subtrans_data.get('whisperjav_version', '')
                try:
                    from whisperjav.__version__ import __version__ as _current_version
                except ImportError:
                    _current_version = ''
                if _old_version and _current_version and _old_version != _current_version:
                    _delete_stale = True
                    print(f"[TRANSLATE]   Deleting stale .subtrans file (version changed: "
                          f"{_old_version} -> {_current_version})", file=sys.stderr)
            except (ValueError, KeyError, OSError):
                # If we can't read the .subtrans file, it may be corrupt — delete it
                _delete_stale = True
                print(f"[TRANSLATE]   Deleting unreadable .subtrans file", file=sys.stderr)

            if _delete_stale:
                try:
                    _subtrans_path.unlink()
                except OSError as _e:
                    print(f"[TRANSLATE]   Warning: Could not delete .subtrans: {_e}", file=sys.stderr)

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
            subtitle_count = getattr(project.subtitles, 'linecount', None)
            if subtitle_count is None:
                subtitle_count = 'unknown'
            print(f"[TRANSLATE]   Subtitle lines: {subtitle_count}", file=sys.stderr)

        # Instructions are verified in the post-init_options() block above.
        # The old project.SetInstructions() path was dead code — that method
        # does not exist in current PySubtrans versions.

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

        # =====================================================================
        # Qwen3 thinking model workaround: patch response parsing
        # =====================================================================
        # Ollama returns thinking output in 'reasoning' field with empty
        # 'content'. PySubtrans checks for 'reasoning_content' (OpenAI
        # convention) — field name mismatch means translations are silently
        # lost. This patch intercepts the response parsing to move
        # 'reasoning' → 'content' when content is empty.
        # Upstream fix: PySubtrans should also check 'reasoning' field.
        if _is_thinking_model:
            _client = getattr(translator, 'client', None)
            if _client and hasattr(_client, '_process_api_response'):
                _original_process = _client._process_api_response

                def _patched_process_api_response(content, result, _orig=_original_process):
                    """Patched to extract thinking model output from 'reasoning' field."""
                    import copy
                    patched_content = copy.deepcopy(content)
                    choices = patched_content.get('choices', [])
                    for choice in choices:
                        msg = choice.get('message', {})
                        if not msg.get('content') and msg.get('reasoning'):
                            msg['content'] = msg['reasoning']
                            if debug:
                                print(f"[TRANSLATE]   [thinking-patch] Moved 'reasoning' → 'content' "
                                      f"({len(msg['content'])} chars)", file=sys.stderr)
                    return _orig(patched_content, result)

                _client._process_api_response = _patched_process_api_response
                print(f"[TRANSLATE]   Thinking model patch: ACTIVE (reasoning→content fallback)",
                      file=sys.stderr)
            else:
                print(f"[TRANSLATE]   WARNING: Could not patch thinking model — "
                      f"translator.client not found", file=sys.stderr)

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
        # A1: Diagnostic token logging — track batch results and failures
        # =========================================================================
        _batch_stats = {'total': 0, 'success': 0, 'no_matches': 0, 'errors': 0}

        def _diagnostic_batch_handler(sender, **kwargs):
            """Track batch translation results for diagnostic summary.

            batch_translated fires for every batch attempt, regardless of whether
            translations were extracted. We count it as 'total' only — success is
            determined by subtracting known failures.
            """
            _batch_stats['total'] += 1

        def _diagnostic_warning_handler(sender, message=None, **kwargs):
            """Capture translation warnings with context."""
            msg = message or kwargs.get('message', '')
            if msg is None:
                return
            msg_str = str(msg)
            if 'No matches' in msg_str or 'no matches' in msg_str:
                _batch_stats['no_matches'] += 1
                # Warning fires before batch_translated increments total, so +1
                print(f"\n[TRANSLATE] *** NO MATCHES DETECTED (batch #{_batch_stats['total'] + 1}) ***",
                      file=sys.stderr)
                print(f"[TRANSLATE]   The model's response didn't match PySubtrans's expected format.",
                      file=sys.stderr)
                print(f"[TRANSLATE]   Raw warning: {msg_str[:500]}", file=sys.stderr)
                if provider_config.get('max_tokens'):
                    print(f"[TRANSLATE]   max_tokens was set to: {provider_config['max_tokens']}",
                          file=sys.stderr)
                print(f"[TRANSLATE]   Consider: smaller --max-batch-size, different model, or cloud provider",
                      file=sys.stderr)

        def _diagnostic_error_handler(sender, message=None, **kwargs):
            """Track all translation errors — not just HTTP errors."""
            _batch_stats['errors'] += 1
            msg = message or kwargs.get('message', '')
            if msg:
                msg_str = str(msg)
                # Categorize known failure modes
                if any(code in msg_str for code in ('502', '500', '503', 'Server Error', 'timed out')):
                    print(f"\n[TRANSLATE] *** SERVER ERROR (batch #{_batch_stats['total']}) ***",
                          file=sys.stderr)
                    print(f"[TRANSLATE]   Error: {msg_str[:300]}", file=sys.stderr)
                    print(f"[TRANSLATE]   This may indicate context overflow crashing the LLM server.",
                          file=sys.stderr)
                elif 'No text returned' in msg_str or 'no text' in msg_str.lower():
                    print(f"\n[TRANSLATE] *** EMPTY RESPONSE (batch #{_batch_stats['total']}) ***",
                          file=sys.stderr)
                    print(f"[TRANSLATE]   The model returned no text. Possible causes:",
                          file=sys.stderr)
                    print(f"[TRANSLATE]     - Model failed to generate (out of memory, crashed)",
                          file=sys.stderr)
                    print(f"[TRANSLATE]     - Streaming response contained no content chunks",
                          file=sys.stderr)
                    print(f"[TRANSLATE]     - Model is too large for available VRAM",
                          file=sys.stderr)
                    print(f"[TRANSLATE]   Try: smaller model, reduce --max-batch-size, or check Ollama logs",
                          file=sys.stderr)
                else:
                    print(f"\n[TRANSLATE] *** ERROR (batch #{_batch_stats['total']}): {msg_str[:300]} ***",
                          file=sys.stderr)

        # Connect diagnostic handlers to translator events
        if hasattr(translator, 'events'):
            if hasattr(translator.events, 'batch_translated'):
                translator.events.batch_translated.connect(_diagnostic_batch_handler)
            # Also hook warning/error signals for "No matches" detection
            if hasattr(translator.events, 'warning'):
                translator.events.warning.connect(_diagnostic_warning_handler)
            if hasattr(translator.events, 'error'):
                translator.events.error.connect(_diagnostic_error_handler)

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

            # =====================================================================
            # P1/P2: Ground-truth success detection via PySubtrans project state
            # =====================================================================
            # The event-signal approach (counting batch_translated / error signals)
            # is unreliable: PySubtrans catches TranslationResponseError internally
            # and fires batch_translated anyway — errors counter stays at 0 even
            # when ALL batches fail. Instead, check what PySubtrans actually stored.
            _any_translated = False
            _all_translated = False
            if hasattr(project, 'subtitles') and project.subtitles:
                _any_translated = getattr(project.subtitles, 'any_translated', False)
                _all_translated = getattr(project.subtitles, 'all_translated', False)

            if _translation_success:
                print(f"\n[TRANSLATE] Translation completed in {_translation_elapsed:.1f}s", file=sys.stderr)
            else:
                print(f"\n[TRANSLATE] Translation FAILED after {_translation_elapsed:.1f}s", file=sys.stderr)

            # Batch event stats (advisory — may undercount failures)
            print(f"[TRANSLATE] Batch statistics:", file=sys.stderr)
            print(f"[TRANSLATE]   Batches processed: {_batch_stats['total']}", file=sys.stderr)
            if _batch_stats['errors'] > 0:
                print(f"[TRANSLATE]   Errors (signal-based): {_batch_stats['errors']}", file=sys.stderr)
            if _batch_stats['no_matches'] > 0:
                print(f"[TRANSLATE]   'No matches' failures: {_batch_stats['no_matches']}", file=sys.stderr)
                print(f"[TRANSLATE]   The LLM produced output PySubtrans couldn't parse.", file=sys.stderr)
                print(f"[TRANSLATE]   Try: smaller --max-batch-size, different model, or cloud provider",
                      file=sys.stderr)
            _successful = _batch_stats['total'] - _batch_stats['errors'] - _batch_stats['no_matches']
            if _successful > 0 and _batch_stats['total'] > 0:
                print(f"[TRANSLATE]   Successful batches: {_successful}/{_batch_stats['total']}", file=sys.stderr)

            # Ground-truth: what PySubtrans actually stored in the project
            print(f"[TRANSLATE] Translation result (ground truth):", file=sys.stderr)
            print(f"[TRANSLATE]   Any subtitles translated: {'YES' if _any_translated else 'NO'}",
                  file=sys.stderr)
            print(f"[TRANSLATE]   All subtitles translated: {'YES' if _all_translated else 'NO'}",
                  file=sys.stderr)

            # Override success based on ground truth — if TranslateSubtitles()
            # didn't raise but nothing was actually translated, it's a failure.
            if _translation_success and not _any_translated and _batch_stats['total'] > 0:
                _translation_success = False
                print(f"[TRANSLATE] *** ALL {_batch_stats['total']} BATCHES FAILED — "
                      f"no subtitles were translated ***", file=sys.stderr)
                print(f"[TRANSLATE]   Common causes: model returned empty content (thinking mode),",
                      file=sys.stderr)
                print(f"[TRANSLATE]   output format not parseable, or server errors.", file=sys.stderr)

        # Save final project state
        print(f"[TRANSLATE] Saving project state...", file=sys.stderr)
        if hasattr(project, 'SaveProject'):
            try:
                project.SaveProject()
                print(f"[TRANSLATE]   Project state saved (.subtrans file)", file=sys.stderr)
                # Stamp WhisperJAV version into .subtrans for A2 stale detection
                _subtrans_save_path = Path(str(input_path) + '.subtrans')
                if _subtrans_save_path.exists():
                    try:
                        import json as _json
                        from whisperjav.__version__ import __version__ as _ver
                        with open(_subtrans_save_path, 'r', encoding='utf-8') as _sf:
                            _data = _json.load(_sf)
                        _data['whisperjav_version'] = _ver
                        with open(_subtrans_save_path, 'w', encoding='utf-8') as _sf:
                            _json.dump(_data, _sf, indent=2, ensure_ascii=False)
                    except Exception:
                        pass  # Best-effort version stamping
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

        # Verify the save actually produced a file.
        # PySubtrans' SubtitleProject.SaveTranslation() catches exceptions
        # internally (logging.error) and returns None — we never see the error.
        # Check the filesystem to know if the save actually worked.
        _save_succeeded = output_path.is_file()
        if not _save_succeeded:
            print(f"[TRANSLATE]   WARNING: SaveTranslation did not produce output at: {output_path}",
                  file=sys.stderr)

        # =====================================================================
        # Issue 2: Clean up PySubtrans' default .translated.srt artifact
        # =====================================================================
        # PySubtrans creates a .translated.srt file next to the input during
        # SaveProject() (line 303-304 of SubtitleProject.py) using the default
        # self.outputpath. When we then call SaveTranslation(output_path) with
        # a different path (e.g., .english.srt or a user-specified directory),
        # we end up with TWO output files. Clean up the redundant artifact.
        # IMPORTANT: Only clean up if the intended save succeeded — otherwise
        # the artifact is the ONLY copy of the translation.
        if hasattr(project, 'subtitles') and project.subtitles:
            _default_outpath = getattr(project.subtitles, 'outputpath', None)
            if _default_outpath:
                _default_outpath = Path(_default_outpath)
                if _default_outpath.exists() and _default_outpath.resolve() != output_path.resolve():
                    if _save_succeeded:
                        try:
                            _default_outpath.unlink()
                            print(f"[TRANSLATE]   Cleaned up intermediate artifact: {_default_outpath.name}",
                                  file=sys.stderr)
                        except OSError as _e:
                            print(f"[TRANSLATE]   Warning: Could not remove artifact {_default_outpath.name}: {_e}",
                                  file=sys.stderr)
                    else:
                        # The intended save failed but the artifact exists —
                        # this IS the translation. Report it as the output.
                        output_path = _default_outpath
                        print(f"[TRANSLATE]   Using fallback output: {_default_outpath.name}",
                              file=sys.stderr)

        # =========================================================================
        # DIAGNOSTIC: Final Summary — truthful status
        # =========================================================================
        print(f"", file=sys.stderr)
        print(f"[TRANSLATE] " + "=" * 50, file=sys.stderr)
        if _translation_success:
            print(f"[TRANSLATE]   TRANSLATION COMPLETE", file=sys.stderr)
            print(f"[TRANSLATE]   Output: {output_path}", file=sys.stderr)
        else:
            print(f"[TRANSLATE]   TRANSLATION FAILED", file=sys.stderr)
            print(f"[TRANSLATE]   All batches returned errors — no subtitles translated.", file=sys.stderr)
            print(f"[TRANSLATE]   Output file may be empty or contain only originals.", file=sys.stderr)
        print(f"[TRANSLATE] " + "=" * 50, file=sys.stderr)
        print(f"", file=sys.stderr)

        # Return None on total failure so cli.py reports it as failed
        return output_path if _translation_success else None

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
