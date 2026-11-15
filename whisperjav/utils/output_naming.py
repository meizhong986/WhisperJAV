"""Centralized construction of WhisperJAV subtitle output filenames.

Single source of truth for how transcription, ensemble-merge, and translation
SRT files are named, so the scheme stays consistent across the pipelines, the
ensemble orchestrator, the GUI, and the translate CLI/service.

Naming scheme:
    transcription:  {basename}.whisperjav.{lang}.srt
    ensemble merge: {basename}.merged.whisperjav.{lang}.srt
    translation:    {basename}.whisperjav.{target}.srt

`lang`/`target` are ISO 639-1 short codes (ja, en, zh, ...). This module
depends only on the standard library so it is safe to import from anywhere.
"""

from pathlib import Path
from typing import Union

# Full language name -> ISO 639-1 short code, used only for output filenames.
LANGUAGE_FILENAME_CODES = {
    "japanese": "ja",
    "english": "en",
    "chinese": "zh",
    "korean": "ko",
    "indonesian": "id",
    "portuguese": "pt",
    "spanish": "es",
    "french": "fr",
    "italian": "it",
    "thai": "th",
}

# Trailing dotted segments treated as an existing language tag when re-deriving
# an output name from an input filename's stem (so tags don't stack).
_LANGUAGE_TAGS = set(LANGUAGE_FILENAME_CODES) | set(LANGUAGE_FILENAME_CODES.values()) | {"jp"}


def language_filename_code(language: str) -> str:
    """Map a language name (or code) to its short filename code.

    Unknown values are returned lowercased/stripped unchanged, so callers that
    already pass a short code (e.g. `'ja'`) keep working.
    """
    if not language:
        return language
    key = language.strip().lower()
    return LANGUAGE_FILENAME_CODES.get(key, key)


def transcription_srt_name(media_basename: str, lang_code: str) -> str:
    """Filename for a transcription SRT: `{basename}.whisperjav.{lang}.srt`."""
    return f"{media_basename}.whisperjav.{language_filename_code(lang_code)}.srt"


def ensemble_srt_name(media_basename: str, lang_code: str) -> str:
    """Filename for a merged ensemble SRT: `{basename}.merged.whisperjav.{lang}.srt`."""
    return f"{media_basename}.merged.whisperjav.{language_filename_code(lang_code)}.srt"


def translated_srt_path(input_path: Union[str, Path], target_lang: str) -> Path:
    """Output path for a translated SRT, placed next to the input file.

    Strips a trailing source-language tag from the input stem so a
    transcription named `<base>.whisperjav.ja.srt` becomes
    `<base>.whisperjav.en.srt` rather than stacking language tags.
    """
    input_path = Path(input_path)
    stem = input_path.stem
    parts = stem.split(".")
    if len(parts) > 1 and parts[-1].lower() in _LANGUAGE_TAGS:
        stem = ".".join(parts[:-1])
    output_name = f"{stem}.{language_filename_code(target_lang)}.srt"
    return input_path.parent / output_name
