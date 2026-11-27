#!/usr/bin/env python3
"""Naming of translated subtitle files, one implementation for both paths.

There are two places a translated file gets its name: `translate/cli.py`
(`whisperjav-translate`, which the GUI's Translator tab spawns) and
`translate/service.py` (the `--translate` hook on a transcription run). Both call in here,
and the name itself is built by `whisperjav/utils/output_naming.py`, beside every other
subtitle name WhisperJAV writes.
"""
from pathlib import Path
from typing import Union

from whisperjav.utils.output_naming import translated_path


def generate_output_path(input_path: Union[str, Path], target_lang: str) -> str:
    """Give where a translation of `input_path` into `target_lang` should be written.

    Args:
        input_path: The subtitle file being translated.
        target_lang: The translation's language, for example `english`.

    Raises:
        ValueError: If the translation would overwrite its own input.

    Returns:
        The output path, for example `film.whisperjav.ja.translated.en.srt`.
    """
    return str(translated_path(input_path, target_lang))
