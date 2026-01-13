"""
Core translation logic - PySubtrans wrapper.
"""

import sys
from pathlib import Path


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
            opt_kwargs['api_base'] = provider_config['api_base']
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

        # Merge provider-specific options
        if provider_options:
            opt_kwargs.update(provider_options)

        # Initialize options and provider
        options = init_options(**opt_kwargs)

        # Initialize provider
        provider = init_translation_provider(provider_config['pysubtrans_name'], options)
        if hasattr(provider, 'ValidateSettings') and not provider.ValidateSettings():
            msg = getattr(provider, 'validation_message', 'Invalid provider settings')
            print(f"ERROR: Provider validation failed: {msg}", file=sys.stderr)
            return None

        # Initialize project (PySubtrans 1.5.x expects subtitle path in 'filepath' kwarg)
        project = init_project(options, filepath=str(input_path), persistent=True)

        # Check if resuming from existing project
        if hasattr(project, 'existing_project') and project.existing_project:
            print("Resuming translation from existing project file...", file=sys.stderr)
            if hasattr(project, 'subtitles') and project.subtitles:
                if hasattr(project.subtitles, 'any_translated') and project.subtitles.any_translated:
                    print("Found previously translated content - will skip already-translated batches", file=sys.stderr)

        # Set instructions if provided
        if instruction_file is not None and hasattr(project, 'SetInstructions'):
            try:
                project.SetInstructions(str(instruction_file))
            except Exception:
                pass

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
        translator = init_translator(options, translation_provider=provider)

        # Enable resume mode to skip already-translated batches when resuming
        # This is critical for interrupted translations - without it, the translator
        # will re-translate everything from the beginning even if a .subtrans file exists
        translator.resume = True

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

        # Translate subtitles
        project.TranslateSubtitles(translator)

        # Save final project state after successful translation
        if hasattr(project, 'SaveProject'):
            try:
                project.SaveProject()
            except Exception:
                pass  # Don't fail on final save

        # Save translation
        saved_path = None
        try:
            saved_path = project.SaveTranslation(str(output_path))
        except TypeError:
            saved_path = project.SaveTranslation()
        except Exception:
            saved_path = None

        # Convert to Path if needed
        if isinstance(saved_path, (str, Path)):
            output_path = Path(saved_path)

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
