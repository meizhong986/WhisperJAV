"""Subtitle file names put the language code last, and old names still count.

Covers whisperjav/utils/output_naming.py: the names every pipeline, the Ensemble and translation
write, `--skip-existing` finding 1.10.3 outputs, the one language table, and the translation names
that must never clash or overwrite their input.
"""
import ast
from pathlib import Path

import pytest

from whisperjav.translate.output_naming import generate_output_path
from whisperjav.translate.providers import SUPPORTED_SOURCES, SUPPORTED_TARGETS
from whisperjav.utils.output_naming import (
    LANGUAGE_CODES,
    language_code,
    merged_exists,
    merged_name,
    pass_name,
    transcription_exists,
    transcription_name,
    translated_path,
)

ROOT = Path(__file__).resolve().parents[1]


def _main_language_choices():
    """The choices of `--language` in main.py, read from source (importing main.py re-execs)."""
    tree = ast.parse((ROOT / "whisperjav" / "main.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
                and node.args and getattr(node.args[0], "value", None) == "--language"):
            for kw in node.keywords:
                if kw.arg == "choices":
                    return set(ast.literal_eval(kw.value))
    raise AssertionError("--language not found in main.py")


# --- one language table ---------------------------------------------------------------

def test_every_source_and_target_has_a_code():
    languages = SUPPORTED_SOURCES | SUPPORTED_TARGETS | _main_language_choices()
    assert languages - set(LANGUAGE_CODES) == set()


def test_codes_are_unique_two_letter_codes():
    codes = list(LANGUAGE_CODES.values())
    assert len(codes) == len(set(codes))
    assert all(len(c) == 2 and c.isalpha() and c.islower() for c in codes)


def test_language_code_takes_a_name_or_a_code():
    assert language_code("japanese") == "ja"
    assert language_code("English") == "en"
    assert language_code("ja") == "ja"
    assert language_code("yue") == "yue"


def test_an_unknown_language_name_is_an_error():
    with pytest.raises(ValueError):
        language_code("klingon")


# --- the names, code last ---------------------------------------------------------------

def test_names_end_in_the_language_code():
    assert transcription_name("film", "japanese") == "film.whisperjav.ja.srt"
    assert transcription_name("film", "ja", "vtt") == "film.whisperjav.ja.vtt"
    assert merged_name("film", "ja") == "film.merged.whisperjav.ja.srt"
    assert pass_name("film", "ja", 1) == "film.pass1.ja.srt"
    assert pass_name("film", "ja", 2, unfinished=True) == "film.pass2.unfinished.ja.srt"


# --- point 1: --skip-existing finds new and 1.10.3 names --------------------------------

@pytest.mark.parametrize("name", ["film.whisperjav.ja.srt", "film.whisperjav.ja.vtt",
                                  "film.ja.whisperjav.srt", "film.ja.whisperjav.vtt"])
def test_a_finished_transcription_is_found_by_either_name(tmp_path, name):
    (tmp_path / name).write_text("", encoding="utf-8")
    assert transcription_exists(tmp_path, "film", "ja")


def test_another_language_or_film_is_not_a_finished_transcription(tmp_path):
    (tmp_path / "film.whisperjav.en.srt").write_text("", encoding="utf-8")
    (tmp_path / "other.whisperjav.ja.srt").write_text("", encoding="utf-8")
    assert not transcription_exists(tmp_path, "film", "ja")


@pytest.mark.parametrize("name", ["film.merged.whisperjav.ja.srt", "film.ja.merged.whisperjav.srt"])
def test_a_finished_ensemble_run_is_found_by_either_name(tmp_path, name):
    (tmp_path / name).write_text("", encoding="utf-8")
    assert merged_exists(tmp_path, "film", "japanese")


def test_a_pass_file_is_not_a_finished_ensemble_run(tmp_path):
    (tmp_path / "film.pass1.ja.srt").write_text("", encoding="utf-8")
    assert not merged_exists(tmp_path, "film", "ja")


# --- point 2: translations keep the source code, so they never clash ---------------------

@pytest.mark.parametrize("source, target, expected", [
    ("film.whisperjav.ja.srt", "english", "film.whisperjav.ja.translated.en.srt"),
    ("film.merged.whisperjav.ja.srt", "english", "film.merged.whisperjav.ja.translated.en.srt"),
    ("film.whisperjav.en.srt", "french", "film.whisperjav.en.translated.fr.srt"),
    ("movie.srt", "english", "movie.translated.en.srt"),
    ("movie.ja.srt", "english", "movie.ja.translated.en.srt"),
    ("Movie.ko.2020.srt", "english", "Movie.ko.2020.translated.en.srt"),
    # 1.10.3 names are put in the new order first.
    ("film.ja.whisperjav.srt", "english", "film.whisperjav.ja.translated.en.srt"),
    ("film.ja.merged.whisperjav.srt", "english", "film.merged.whisperjav.ja.translated.en.srt"),
    ("film.ja.whisperjav.english.srt", "french", "film.whisperjav.ja.translated.fr.srt"),
    # A translation of a translation does not stack tags.
    ("film.whisperjav.ja.translated.en.srt", "french", "film.whisperjav.ja.translated.fr.srt"),
])
def test_translation_names(source, target, expected):
    assert translated_path(Path("/media") / source, target) == Path("/media") / expected
    assert Path(generate_output_path(Path("/media") / source, target)).name == expected


def test_direct_to_english_and_an_english_translation_do_not_clash():
    direct = transcription_name("film", "en")
    translation = translated_path(Path(transcription_name("film", "ja")), "english").name
    assert direct != translation


def test_translations_of_two_sources_into_one_language_do_not_clash():
    from_ja = translated_path(Path(transcription_name("film", "ja")), "french")
    from_en = translated_path(Path(transcription_name("film", "en")), "french")
    assert from_ja != from_en


def test_a_translation_never_overwrites_its_input():
    with pytest.raises(ValueError):
        translated_path("/media/film.whisperjav.ja.translated.en.srt", "english")


@pytest.mark.parametrize("target", sorted(SUPPORTED_TARGETS))
def test_every_target_round_trips_without_stacking(target):
    first = translated_path("/media/film.whisperjav.ja.srt", target)
    second = translated_path(first, "english" if target != "english" else "french")
    assert second.name.count(".translated.") == 1
    assert second.name.startswith("film.whisperjav.ja.translated.")
