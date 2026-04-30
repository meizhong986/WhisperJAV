"""Tests for AnimeWhisperCleaner — text-level and SRT-level rules.

v1.8.12.post2 adds the ellipsis-only drop rule (text-level) and the
filter_srt_file() method (SRT-level: drop + renumber).
"""

from pathlib import Path

import pytest

from whisperjav.modules.subtitle_pipeline.cleaners.anime_whisper import (
    AnimeWhisperCleaner,
)


@pytest.fixture
def cleaner():
    return AnimeWhisperCleaner()


# ===========================================================================
# Ellipsis-only drop rule (text-level)
# ===========================================================================

class TestIsEllipsisOnly:
    """Direct test of the static helper that decides which strings are dropped."""

    @pytest.mark.parametrize("text", [
        "…",                # single horizontal ellipsis
        "‥",                # two-dot leader
        "...",              # ASCII triple-dot
        "……",               # double horizontal ellipsis
        "．．．",            # fullwidth dots
        "  …  ",            # surrounded by whitespace
        "…?",               # ellipsis + ASCII question
        "…!",               # ellipsis + ASCII exclamation
        "…」",              # ellipsis + closing CJK quote
        "…』",              # ellipsis + closing white CJK quote
        "…？",              # ellipsis + fullwidth question
        "…！",              # ellipsis + fullwidth exclamation
        "…?!",              # ellipsis + multiple closing punct
        "…)",               # ellipsis + ASCII closing paren
        "…）",              # ellipsis + fullwidth closing paren
        "…」 ",             # trailing whitespace
        " ……」",             # leading whitespace + multi-ellipsis + quote
        ".....",            # multiple ASCII dots
        "…\t",              # tab
    ])
    def test_drops(self, text):
        assert AnimeWhisperCleaner._is_ellipsis_only(text), (
            f"Expected ellipsis-only verdict for {text!r}"
        )

    @pytest.mark.parametrize("text", [
        "あ",               # single hiragana
        "あ…",              # letter + ellipsis
        "…あ",              # ellipsis + letter
        "?",                # bare punct (no dot)
        "!",                # bare punct (no dot)
        "」",               # bare closing quote (no dot)
        "？",               # bare fullwidth question
        "これは…",          # legit dialogue with trailing ellipsis
        "…はい",            # legit dialogue with leading ellipsis
        "ありがとう",       # legit dialogue, no ellipsis at all
        "  あ  …  ",        # letters + whitespace + ellipsis
        "",                 # empty string — explicit guard
        "   ",              # whitespace only — explicit guard (no dot)
    ])
    def test_keeps(self, text):
        assert not AnimeWhisperCleaner._is_ellipsis_only(text), (
            f"Expected NOT ellipsis-only for {text!r}"
        )


# ===========================================================================
# clean() integration with the new rule
# ===========================================================================

class TestCleanReturnsEmptyForEllipsis:
    """clean() must return '' for ellipsis-only artifacts."""

    @pytest.mark.parametrize("text", [
        "…",
        "‥",
        "...",
        "……",
        "  …  ",
        "…?",
        "…!",
        "…」",
        "…？",
        "…！",
        "…?!",
        "…)",
    ])
    def test_drops_ellipsis_artifact(self, cleaner, text):
        assert cleaner.clean(text) == ""


class TestCleanPreservesRealContent:
    """clean() must not drop legitimate dialogue."""

    def test_keeps_letter_with_ellipsis(self, cleaner):
        # Letter + trailing ellipsis: kept, no 。 added (ellipsis is sentence-final).
        assert cleaner.clean("あ…") == "あ…"

    def test_keeps_dialogue_with_ellipsis(self, cleaner):
        assert cleaner.clean("これは…") == "これは…"

    def test_adds_period_to_bare_letter(self, cleaner):
        # Existing behavior: appends 。 if no sentence-final punct.
        assert cleaner.clean("あ") == "あ。"

    def test_existing_sentence_ending_unchanged(self, cleaner):
        assert cleaner.clean("こんにちは") == "こんにちは。"

    def test_existing_repetition_collapse_unchanged(self, cleaner):
        # Pre-existing rule: 3+ consecutive repeats collapse.
        result = cleaner.clean("ありがとうありがとうありがとう")
        # After collapse → "ありがとう" + sentence-final 。 added.
        assert "ありがとう" in result
        assert result.count("ありがとう") == 1
        assert result.endswith("。")

    def test_empty_input(self, cleaner):
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""

    def test_punct_only_kept(self, cleaner):
        # No dot in input → not an ellipsis artifact → cleaner keeps it.
        # _ensure_sentence_ending: "?" IS in _SENTENCE_FINAL_PUNCT, so no 。
        # added.
        assert cleaner.clean("?") == "?"


# ===========================================================================
# clean_batch() reporting
# ===========================================================================

class TestCleanBatchEllipsisReporting:
    """clean_batch() should count ellipsis drops separately."""

    def test_drops_propagate_to_output(self, cleaner):
        texts = ["これは。", "…", "そうです", "…?", "…」"]
        out = cleaner.clean_batch(texts)
        assert len(out) == len(texts)
        assert out[0] == "これは。"
        assert out[1] == ""             # ellipsis-only dropped
        assert out[2] == "そうです。"     # 。 appended
        assert out[3] == ""             # ellipsis-only dropped
        assert out[4] == ""             # ellipsis-only dropped

    def test_all_ellipsis_batch(self, cleaner):
        texts = ["…", "…?", "…」", "‥", "...", "……"]
        out = cleaner.clean_batch(texts)
        assert all(c == "" for c in out)


# ===========================================================================
# filter_srt_file() — SRT-level drop + renumber
# ===========================================================================

class TestFilterSrtFile:
    """SRT-level filter that drops ellipsis-only entries and renumbers."""

    def _write_srt(self, path: Path, content: str) -> None:
        path.write_text(content, encoding='utf-8')

    def test_drops_ellipsis_entries_and_renumbers(self, cleaner, tmp_path):
        # Reproduces the user's example case: lines 8-12 are ellipsis-only.
        srt_content = """1
00:00:00,500 --> 00:00:02,000
こんにちは。

2
00:00:34,448 --> 00:00:34,792
…

3
00:00:39,248 --> 00:00:39,576
…

4
00:00:40,176 --> 00:00:40,488
…」

5
00:00:41,120 --> 00:00:41,480
…?

6
00:00:42,496 --> 00:00:42,872
…!

7
00:01:00,000 --> 00:01:01,500
さようなら。
"""
        srt_path = tmp_path / "test.srt"
        self._write_srt(srt_path, srt_content)

        stats = cleaner.filter_srt_file(srt_path)

        assert stats["original_count"] == 7
        assert stats["dropped_ellipsis"] == 5
        assert stats["dropped_empty"] == 0
        assert stats["final_count"] == 2

        # Read back and confirm renumbering + content survived.
        import pysrt
        subs = pysrt.open(str(srt_path), encoding='utf-8')
        assert len(subs) == 2
        assert subs[0].index == 1
        assert subs[1].index == 2
        assert "こんにちは" in subs[0].text
        assert "さようなら" in subs[1].text

    def test_empty_file_returns_zero_stats(self, cleaner, tmp_path):
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("", encoding='utf-8')
        stats = cleaner.filter_srt_file(srt_path)
        assert stats == {
            "original_count": 0,
            "dropped_ellipsis": 0,
            "dropped_empty": 0,
            "final_count": 0,
        }

    def test_nonexistent_file_returns_zero_stats(self, cleaner, tmp_path):
        srt_path = tmp_path / "does_not_exist.srt"
        stats = cleaner.filter_srt_file(srt_path)
        assert stats["original_count"] == 0
        assert stats["final_count"] == 0

    def test_no_ellipsis_entries_unchanged_count(self, cleaner, tmp_path):
        srt_content = """1
00:00:00,500 --> 00:00:02,000
こんにちは。

2
00:00:03,000 --> 00:00:04,000
さようなら。
"""
        srt_path = tmp_path / "clean.srt"
        self._write_srt(srt_path, srt_content)

        stats = cleaner.filter_srt_file(srt_path)
        assert stats["original_count"] == 2
        assert stats["dropped_ellipsis"] == 0
        assert stats["dropped_empty"] == 0
        assert stats["final_count"] == 2

    def test_all_ellipsis_drops_to_empty(self, cleaner, tmp_path):
        srt_content = """1
00:00:00,500 --> 00:00:01,000
…

2
00:00:02,000 --> 00:00:02,500
…?

3
00:00:03,000 --> 00:00:03,500
…」
"""
        srt_path = tmp_path / "all_ellipsis.srt"
        self._write_srt(srt_path, srt_content)

        stats = cleaner.filter_srt_file(srt_path)
        assert stats["original_count"] == 3
        assert stats["dropped_ellipsis"] == 3
        assert stats["final_count"] == 0
