"""Names of the subtitle files WhisperJAV writes.

Every name ends in the language's two-letter code just before the extension, as in
`film.whisperjav.ja.srt`, so video players can detect the language. All the names are built
here, so the pipelines, the Ensemble passes, `--skip-existing` and translation cannot drift
apart. Standard library only, so it is safe to import from anywhere, pass workers included.

Up to 1.10.3 the code came before the tag (`film.ja.whisperjav.srt`). Those names are still
recognised, so `--skip-existing` keeps skipping old outputs and a translation of an old file
gets a new name.
"""

import re
from pathlib import Path
from typing import Union

# The two-letter (ISO 639-1) code of every source and target language. tests/test_output_naming.py
# checks that every language in SUPPORTED_SOURCES, SUPPORTED_TARGETS and main.py's --language has one.
LANGUAGE_CODES = {
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

TAG = "whisperjav"
MERGED_TAG = "merged"
TRANSLATED_TAG = "translated"

_CODE_SHAPE = re.compile(r"[a-z]{2,3}")


def language_code(language: str) -> str:
    """Give the two-letter code of a language.

    Args:
        language: A language name (`japanese`) or a code (`ja`).

    Raises:
        ValueError: If `language` is neither a known name nor shaped like a code.

    Returns:
        The code, for example `ja`. A code not in `LANGUAGE_CODES` is returned as it is.
    """
    key = (language or "").strip().lower()
    if key in LANGUAGE_CODES:
        return LANGUAGE_CODES[key]
    if _CODE_SHAPE.fullmatch(key):
        return key
    raise ValueError(f"No language code for {language!r}")


def transcription_name(base: str, language: str, ext: str = "srt") -> str:
    """Name the subtitle file of a transcription.

    Args:
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.
        ext: The extension, without the dot.

    Returns:
        `{base}.whisperjav.{code}.{ext}`.
    """
    return f"{base}.{TAG}.{language_code(language)}.{ext}"


def merged_name(base: str, language: str) -> str:
    """Name the merged subtitle file of an Ensemble run.

    Args:
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.

    Returns:
        `{base}.merged.whisperjav.{code}.srt`.
    """
    return f"{base}.{MERGED_TAG}.{TAG}.{language_code(language)}.srt"


def pass_name(base: str, language: str, pass_number: int, unfinished: bool = False) -> str:
    """Name the subtitle file of one Ensemble pass.

    Args:
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.
        pass_number: The pass, 1 or 2.
        unfinished: True for the copy a stopped pass leaves beside the video.

    Returns:
        `{base}.passN.{code}.srt`, or `{base}.passN.unfinished.{code}.srt`.
    """
    middle = f"pass{pass_number}.unfinished" if unfinished else f"pass{pass_number}"
    return f"{base}.{middle}.{language_code(language)}.srt"


def transcription_exists(folder: Union[str, Path], base: str, language: str) -> bool:
    """Tell whether a transcription's subtitle file is already in a folder.

    Used by `--skip-existing`. The 1.10.3 names (`{base}.{code}.whisperjav.srt`) count too, so a
    folder done by an older version is not transcribed again.

    Args:
        folder: The folder the subtitle file would be written to.
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.

    Returns:
        True if the `.srt` or `.vtt` exists under the new name or the 1.10.3 name.
    """
    code = language_code(language)
    names = [f"{base}.{middle}.{ext}" for middle in (f"{TAG}.{code}", f"{code}.{TAG}") for ext in ("srt", "vtt")]
    return any((Path(folder) / name).exists() for name in names)


def merged_exists(folder: Union[str, Path], base: str, language: str) -> bool:
    """Tell whether an Ensemble run's merged subtitle file is already in a folder.

    Used by `--skip-existing`. The 1.10.3 name (`{base}.{code}.merged.whisperjav.srt`) counts too.

    Args:
        folder: The folder the merged file would be written to.
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.

    Returns:
        True if the merged file exists under the new name or the 1.10.3 name.
    """
    legacy = f"{base}.{language_code(language)}.{MERGED_TAG}.{TAG}.srt"
    return any((Path(folder) / name).exists() for name in (merged_name(base, language), legacy))


def pass_side_file_name(name: str, base: str, language: str, pass_number: int) -> str | None:
    """Rename a pass pipeline's side file the way its pass file is named.

    A side file is named after the transcription, as in `{base}.whisperjav.{code}.analytics.json`.
    In an Ensemble pass it becomes `{base}.passN.{code}.analytics.json`, beside `{base}.passN.{code}.srt`.

    Args:
        name: The side file's name.
        base: The media file's name without its extension.
        language: The subtitle language, as a name or a code.
        pass_number: The pass, 1 or 2.

    Returns:
        The new name, or None when `name` is not a side file of `base` (the subtitle file itself
        included).
    """
    code = language_code(language)
    prefix = f"{base}.{TAG}.{code}."
    if not name.startswith(prefix) or name == transcription_name(base, code):
        return None
    return f"{base}.pass{pass_number}.{code}.{name[len(prefix):]}"


# A finished transcription or merged file, new name or 1.10.3 name. Not a pass file, not a translation.
_FINAL_NAME = re.compile(rf"\.{TAG}(\.[a-z]{{2,3}})?\.srt$")


def is_final_subtitle_name(name: str) -> bool:
    """Tell whether a file name is that of a finished transcription or Ensemble run.

    Args:
        name: The file name.

    Returns:
        True for `film.whisperjav.ja.srt`, `film.merged.whisperjav.ja.srt` and their 1.10.3 names.
    """
    return bool(_FINAL_NAME.search(name))


def _is_code(part: str) -> bool:
    return part.lower() in LANGUAGE_CODES.values()


def _source_stem(stem: str) -> str:
    """Turn a translation input's stem into the stem its translation builds on.

    A trailing `.translated.{code}` is dropped, so tags do not stack. So is the full language
    name a 1.10.3 translation ended in (`.english`). The 1.10.3 orders `.{code}.whisperjav` and
    `.{code}.merged.whisperjav` become the new order. Nothing else is touched, so a code inside a
    title (`Movie.ko.2020`) stays.
    """
    parts = stem.split(".")
    if len(parts) > 2 and parts[-2].lower() == TRANSLATED_TAG and _is_code(parts[-1]):
        parts = parts[:-2]
    elif len(parts) > 1 and parts[-1].lower() in LANGUAGE_CODES:
        parts = parts[:-1]

    if len(parts) > 2 and parts[-1] == TAG and _is_code(parts[-2]):
        parts = parts[:-2] + [TAG, parts[-2]]
    elif len(parts) > 3 and parts[-2:] == [MERGED_TAG, TAG] and _is_code(parts[-3]):
        parts = parts[:-3] + [MERGED_TAG, TAG, parts[-3]]
    return ".".join(parts)


def translated_path(input_path: Union[str, Path], target: str) -> Path:
    """Place the translation of a subtitle file next to it.

    The input's language code stays in the name, so the translations of a Japanese file and of a
    Direct-to-English file of the same film never clash.

    Args:
        input_path: The subtitle file being translated.
        target: The translation's language, as a name or a code.

    Raises:
        ValueError: If the translation would overwrite its own input.

    Returns:
        For example `film.whisperjav.ja.srt` into English gives `film.whisperjav.ja.translated.en.srt`.
    """
    input_path = Path(input_path)
    stem = _source_stem(input_path.stem)
    output = input_path.parent / f"{stem}.{TRANSLATED_TAG}.{language_code(target)}.srt"
    if output == input_path:
        raise ValueError(f"Translating {input_path.name} into {target} would overwrite it")
    return output
