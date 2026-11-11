"""Tests for the centralized subtitle output-naming scheme.

Naming scheme (v1.8.15+):
    transcription:  {basename}.whisperjav.{lang}.srt
    ensemble merge: {basename}.merged.whisperjav.{lang}.srt
    translation:    {basename}.whisperjav.{target}.srt

`lang`/`target` are ISO 639-1 short codes (ja, en, zh, ...). Translation
strips a trailing source-language tag from the input stem so language tags
do not stack.
"""

from pathlib import Path

from whisperjav.utils.output_naming import (
    ensemble_srt_name,
    language_filename_code,
    transcription_srt_name,
    translated_srt_path,
)


class TestLanguageFilenameCode:
    def test_full_word_japanese_maps_to_ja(self):
        assert language_filename_code("japanese") == "ja"

    def test_full_word_english_maps_to_en(self):
        assert language_filename_code("english") == "en"

    def test_chinese_maps_to_zh(self):
        assert language_filename_code("chinese") == "zh"

    def test_supported_targets_have_codes(self):
        assert language_filename_code("indonesian") == "id"
        assert language_filename_code("portuguese") == "pt"
        assert language_filename_code("spanish") == "es"
        assert language_filename_code("french") == "fr"

    def test_case_insensitive(self):
        assert language_filename_code("English") == "en"
        assert language_filename_code("  JAPANESE  ") == "ja"

    def test_short_code_passthrough(self):
        # Callers may already pass a short code; it must survive unchanged.
        assert language_filename_code("ja") == "ja"
        assert language_filename_code("en") == "en"

    def test_unknown_is_passed_through_lowercased(self):
        assert language_filename_code("klingon") == "klingon"


class TestTranscriptionSrtName:
    def test_short_code(self):
        assert transcription_srt_name("video", "ja") == "video.whisperjav.ja.srt"

    def test_full_language_word_is_shortened(self):
        assert transcription_srt_name("video", "japanese") == "video.whisperjav.ja.srt"

    def test_english_direct(self):
        assert transcription_srt_name("video", "en") == "video.whisperjav.en.srt"

    def test_basename_with_dots_and_unicode(self):
        base = "[2023-05-15] はじめまして (@c-toru_2434) - Twitcast"
        assert transcription_srt_name(base, "ja") == f"{base}.whisperjav.ja.srt"


class TestEnsembleSrtName:
    def test_short_code(self):
        assert ensemble_srt_name("video", "ja") == "video.merged.whisperjav.ja.srt"

    def test_full_word(self):
        assert ensemble_srt_name("video", "english") == "video.merged.whisperjav.en.srt"


class TestTranslatedSrtPath:
    def test_strips_source_tag_and_shortens_target(self):
        src = Path("/d/video.whisperjav.ja.srt")
        out = translated_srt_path(src, "english")
        assert out == Path("/d/video.whisperjav.en.srt")

    def test_target_chinese(self):
        src = Path("/d/video.whisperjav.ja.srt")
        out = translated_srt_path(src, "chinese")
        assert out == Path("/d/video.whisperjav.zh.srt")

    def test_ensemble_source(self):
        src = Path("/d/video.merged.whisperjav.ja.srt")
        out = translated_srt_path(src, "english")
        assert out == Path("/d/video.merged.whisperjav.en.srt")

    def test_no_language_tag_just_appends(self):
        src = Path("/d/video.srt")
        out = translated_srt_path(src, "english")
        assert out == Path("/d/video.en.srt")

    def test_preserves_complex_basename(self):
        base = "[2023-05-15] はじめまして (@c-toru_2434) - Twitcast"
        src = Path("/d") / f"{base}.whisperjav.ja.srt"
        out = translated_srt_path(src, "english")
        assert out == Path("/d") / f"{base}.whisperjav.en.srt"

    def test_output_is_next_to_input(self):
        src = Path("/some/nested/dir/clip.whisperjav.ja.srt")
        out = translated_srt_path(src, "english")
        assert out.parent == Path("/some/nested/dir")

    def test_accepts_string_input(self):
        out = translated_srt_path("/d/video.whisperjav.ja.srt", "english")
        assert out == Path("/d/video.whisperjav.en.srt")
