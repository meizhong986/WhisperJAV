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
    extra_context: str = None
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

        # Initialize project
        project = init_project(options, file=str(input_path))

        # Set instructions if provided
        if instruction_file is not None and hasattr(project, 'SetInstructions'):
            try:
                project.SetInstructions(str(instruction_file))
            except Exception:
                pass

        # Save project if possible
        if hasattr(project, 'SaveProject'):
            try:
                project.SaveProject()
            except Exception:
                pass

        # Initialize translator and translate
        translator = init_translator(options, translation_provider=provider)

        # Translate subtitles
        project.TranslateSubtitles(translator)

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
        if debug:
            import traceback
            traceback.print_exc()
        raise
