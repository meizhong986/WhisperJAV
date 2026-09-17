#!/usr/bin/env python3
"""Naming of translated subtitle files — one implementation for both paths.

There are two places a translated file gets its name: `translate/cli.py`
(`whisperjav-translate`, which the GUI's Translator tab spawns) and
`translate/service.py` (the `--translate` hook on a transcription run). Each used to keep
its own hand-written list of language suffixes to strip, and both had drifted: cli.py knew
only japanese/english/ja/en/jp, so re-translating `x.french.srt` to English produced
`x.french.english.srt`, and service.py was missing portuguese and french.

Both now call in here. Adding a language to SUPPORTED_TARGETS is enough; no list needs
editing. Standard library plus `.providers`, so this is importable in a test without
running cli.py's argument parser or loading a provider.
"""
from pathlib import Path
from typing import Union

from .providers import SUPPORTED_TARGETS

# Suffixes WhisperJAV itself writes onto a subtitle stem: every supported target, plus the
# source-language spellings used by the transcription side.
SOURCE_LANGUAGE_SUFFIXES = {"japanese", "ja", "en", "jp"}


def language_suffixes() -> set:
    """Every stem suffix that names a language rather than part of the title."""
    return SUPPORTED_TARGETS | SOURCE_LANGUAGE_SUFFIXES


def strip_language_suffix(stem: str) -> str:
    """Drop a trailing `.<language>` from a subtitle stem, if there is one.

    `SONE-853.french` -> `SONE-853`; `SONE-853.v2` -> `SONE-853.v2` (not a language).
    """
    parts = stem.split(".")
    if len(parts) > 1 and parts[-1].lower() in language_suffixes():
        return ".".join(parts[:-1])
    return stem


def generate_output_path(input_path: Union[str, Path], target_lang: str) -> str:
    """Where a translation of `input_path` into `target_lang` should be written."""
    input_path = Path(input_path)
    stem = strip_language_suffix(input_path.stem)
    return str(input_path.parent / f"{stem}.{target_lang}.srt")
