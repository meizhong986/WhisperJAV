"""Unit tests for Fix 2 — symbol-only subtitle purge in SubtitleSanitizer.

Plan reference: docs/plans/V1811_SANITIZER_FIX_PLAN.md §5.2

Fix 2 adds a drop-or-keep filter at step 1.1.d in _process_content_cleaning
that removes subtitles with no linguistic content (pure punctuation / emoji
/ whitespace). This is defense-in-depth against residue from partial-strip
sanitizer passes that the #287 class of bugs can produce.

HAS_CONTENT regex matches: hiragana | katakana | CJK unified | halfwidth
alnum | fullwidth digits/uppercase/lowercase.

Test strategy: build synthetic pysrt lists and call the helper directly
(unit level), plus end-to-end through process() for a few cases.
"""

import pytest
import re

from whisperjav.modules.subtitle_sanitizer import (
    SubtitleSanitizer,
    _HAS_LINGUISTIC_CONTENT,
)
from whisperjav.config.sanitization_config import SanitizationConfig


class TestHasLinguisticContentRegex:
    """Verify the HAS_CONTENT regex classifies inputs correctly."""

    # --- Should DROP (no linguistic content) ---
    @pytest.mark.parametrize("text", [
        "!!",
        "???",
        "!!!",
        "……",
        "...",
        "♪♫♩",
        "🐈🐈🐈",
        "！！！",              # fullwidth punct only
        "  !!  ",              # whitespace plus punct
        "。。。",              # Japanese period repeats
        "？？？",              # fullwidth question
        "〜〜〜",              # long vowel marks
        "・・・",              # dots (Japanese)
        " ",                   # whitespace
        "",                    # empty
    ])
    def test_drops_symbol_only(self, text):
        assert _HAS_LINGUISTIC_CONTENT.search(text) is None, (
            f"Expected no content match in {text!r}"
        )

    # --- Should KEEP (has linguistic content) ---
    @pytest.mark.parametrize("text", [
        "こんにちは",
        "今日はいい天気ですね",
        "はい",
        "家",                  # kanji
        "い",                  # single hiragana
        "ア",                  # single katakana
        "ｈｅｌｌｏ",          # fullwidth ASCII
        "abc",                 # halfwidth ASCII
        "123",                 # digits
        "こんにちは!!",        # mixed
        "!!こんにちは",        # mixed (leading punct)
        "笑",                  # single kanji
        "ＡＢＣ",              # fullwidth upper
        "１２３",              # fullwidth digits
    ])
    def test_keeps_linguistic_content(self, text):
        assert _HAS_LINGUISTIC_CONTENT.search(text) is not None, (
            f"Expected content match in {text!r}"
        )


class TestSymbolPurgeEndToEnd:
    """End-to-end via SubtitleSanitizer.process() using synthetic SRT fixtures."""

    def _run_sanitizer(self, tmp_path, srt_content):
        """Helper: write content to tmp path, run sanitizer, return sanitized text."""
        config = SanitizationConfig()
        config.primary_language = "ja"
        config.save_artifacts = True
        config.preserve_original_file = True
        sanitizer = SubtitleSanitizer(config=config)

        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")
        result = sanitizer.process(srt_path)
        if result.sanitized_path and result.sanitized_path.exists():
            return result.sanitized_path.read_text(encoding="utf-8")
        return ""

    def test_pure_symbol_residue_dropped(self, tmp_path):
        """A subtitle that is only ellipsis should be dropped by Fix 2."""
        srt = (
            "1\n00:00:00,000 --> 00:00:02,000\n……\n"
        )
        out = self._run_sanitizer(tmp_path, srt)
        # Expected: sub dropped → output is empty or has no sub blocks
        assert "……" not in out, f"symbol-only sub survived: {out!r}"

    def test_music_notes_only_dropped(self, tmp_path):
        """♪♫♩ should be dropped by Fix 2 (♪ also stripped by L268 emoji)."""
        srt = "1\n00:00:00,000 --> 00:00:02,000\n♪♫♩\n"
        out = self._run_sanitizer(tmp_path, srt)
        assert "♪" not in out and "♫" not in out and "♩" not in out, (
            f"music symbols survived: {out!r}"
        )

    def test_animal_emoji_dropped(self, tmp_path):
        """🐈🐈🐈 should be dropped."""
        srt = "1\n00:00:00,000 --> 00:00:02,000\n🐈🐈🐈\n"
        out = self._run_sanitizer(tmp_path, srt)
        assert "🐈" not in out, f"emoji sub survived: {out!r}"

    def test_kanji_only_kept(self, tmp_path):
        """A single kanji must not be dropped by Fix 2 (has linguistic content).

        Uses a short duration so the pre-existing slow-CPS filter doesn't
        drop the 1-char sub (that's separate from what Fix 2 tests).
        """
        srt = "1\n00:00:00,000 --> 00:00:00,500\n家\n"
        out = self._run_sanitizer(tmp_path, srt)
        assert "家" in out, f"kanji sub was incorrectly dropped: {out!r}"

    def test_fullwidth_alnum_kept(self, tmp_path):
        """Fullwidth Latin must not be dropped."""
        srt = "1\n00:00:00,000 --> 00:00:02,000\nｈｅｌｌｏ\n"
        out = self._run_sanitizer(tmp_path, srt)
        assert "ｈｅｌｌｏ" in out

    def test_mixed_kana_and_punct_kept(self, tmp_path):
        """Kana + punct must not be dropped by Fix 2 (has kana)."""
        srt = (
            "1\n00:00:00,000 --> 00:00:02,000\nわああうう\n"
        )
        out = self._run_sanitizer(tmp_path, srt)
        # わああうう is not in filter list and contains kana, must be kept
        assert "わああうう" in out, f"kana+content sub was dropped: {out!r}"

    def test_fix1_residue_is_kept(self, tmp_path):
        """#287 integration: Fix 1 turns a long kana run + '?' into 'いい?'.
        Fix 2 must NOT drop this (has kana content)."""
        srt = (
            "1\n00:00:00,000 --> 00:00:00,500\nいいいいいいいいいいいい?\n"
        )
        out = self._run_sanitizer(tmp_path, srt)
        # After Fix 1: 'いい?'. After Fix 2: 'いい?' kept (has kana).
        assert "いい?" in out, f"Fix 1 residue was dropped by Fix 2: {out!r}"


class TestSymbolPurgeArtifactRecord:
    """Verify Fix 2 drops are recorded as artifacts with the expected reason code."""

    def test_symbol_only_residue_reason_recorded(self, tmp_path):
        """Drop reason should be 'symbol_only_residue' in the artifact list."""
        config = SanitizationConfig()
        config.primary_language = "ja"
        config.save_artifacts = True
        config.preserve_original_file = True
        sanitizer = SubtitleSanitizer(config=config)

        srt_path = tmp_path / "test.srt"
        srt_path.write_text(
            "1\n00:00:00,000 --> 00:00:02,000\n……\n", encoding="utf-8"
        )
        sanitizer.process(srt_path)

        reasons = [e.reason for e in sanitizer.artifact_entries]
        assert "symbol_only_residue" in reasons, (
            f"expected 'symbol_only_residue' reason in artifacts; "
            f"got {reasons}"
        )
