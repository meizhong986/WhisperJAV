"""Regression tests for the nonverbal lone-line filter (v1.9.0; はい/うん added v1.9.2).

Covers:
    - is_nonverbal_line() token matching (positives + safety negatives)
    - filter_srt_file() drop/keep/renumber behavior on a real SRT file
    - QwenPipeline exposes drop_nonverbal_lines defaulting to True
"""

import inspect

import pytest

from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import (
    NonverbalLineFilter,
    NONVERBAL_TOKENS,
)


class TestIsNonverbalLine:
    """Whole-line token matching — the safety-critical predicate."""

    @pytest.mark.parametrize("token", list(NONVERBAL_TOKENS))
    def test_bare_token_and_with_period(self, token):
        assert NonverbalLineFilter.is_nonverbal_line(token) is True
        assert NonverbalLineFilter.is_nonverbal_line(token + "。") is True

    def test_surrounding_whitespace_is_stripped(self):
        assert NonverbalLineFilter.is_nonverbal_line("  は。 ") is True
        assert NonverbalLineFilter.is_nonverbal_line("\nあ\n") is True

    @pytest.mark.parametrize("text", ["はい", "はい。", "うん", "うん。", " はい。 "])
    def test_lone_backchannel_is_dropped_since_v192(self, text):
        # Owner CFF5 (2026-09-05): lone はい。/うん。 are ~90% moans in JAV.
        assert NonverbalLineFilter.is_nonverbal_line(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "はいはい。",     # repeated backchannel — kept (exact token only)
            "うんうん。",     # repeated — kept
            "ううん。",       # negation — kept
            "あ、うん。",     # combo — kept
            "はい、そうです。",  # sentence — kept
            "あー。",        # long-vowel onset — deliberately kept
            "あ、あ。",       # multi-token stutter — kept
            "はは。",        # laughter — kept
            "気持ちいい。",   # real dialogue
            "ダメ…",         # real dialogue with ellipsis
            "ふん。",        # not in the curated set (ふ + ん) — kept
            "",              # empty
            "   ",           # whitespace only
        ],
    )
    def test_negatives_are_kept(self, text):
        assert NonverbalLineFilter.is_nonverbal_line(text) is False

    def test_multichar_trailing_punct_not_matched(self):
        # Only a single optional "。" is honoured; other/extra punct ⇒ keep.
        assert NonverbalLineFilter.is_nonverbal_line("あ。。") is False
        assert NonverbalLineFilter.is_nonverbal_line("あ、") is False
        assert NonverbalLineFilter.is_nonverbal_line("あ…") is False


class TestFilterSrtFile:
    """End-to-end SRT drop / keep / renumber."""

    def _write(self, path, entries):
        blocks = []
        for i, (start, end, text) in enumerate(entries, start=1):
            blocks.append(f"{i}\n{start} --> {end}\n{text}\n")
        path.write_text("\n".join(blocks), encoding="utf-8")

    def test_drops_tokens_keeps_dialogue_and_renumbers(self, tmp_path):
        import pysrt

        srt = tmp_path / "sample.srt"
        self._write(
            srt,
            [
                ("00:00:01,000", "00:00:02,000", "あ。"),          # drop
                ("00:00:02,000", "00:00:03,000", "はい。"),        # drop (v1.9.2)
                ("00:00:03,000", "00:00:04,000", "は。"),          # drop
                ("00:00:04,000", "00:00:05,000", "気持ちいい。"),  # keep
                ("00:00:05,000", "00:00:06,000", "切。"),          # drop
                ("00:00:06,000", "00:00:07,000", "うん。"),        # drop (v1.9.2)
                ("00:00:07,000", "00:00:08,000", "ふっ。"),        # drop
                ("00:00:08,000", "00:00:09,000", "はい、そうです。"),  # keep (sentence)
            ],
        )

        stats = NonverbalLineFilter().filter_srt_file(srt)

        assert stats["original_count"] == 8
        assert stats["dropped_nonverbal"] == 6
        assert stats["dropped_empty"] == 0
        assert stats["final_count"] == 2

        subs = pysrt.open(str(srt), encoding="utf-8")
        assert [s.text for s in subs] == ["気持ちいい。", "はい、そうです。"]
        assert [s.index for s in subs] == [1, 2]  # renumbered

    def test_missing_file_returns_zero_stats(self, tmp_path):
        stats = NonverbalLineFilter().filter_srt_file(tmp_path / "nope.srt")
        assert stats["final_count"] == 0 and stats["original_count"] == 0


class TestPipelineWiring:
    def test_qwen_pipeline_defaults_drop_nonverbal_lines_true(self):
        """QwenPipeline.__init__ exposes drop_nonverbal_lines defaulting to True."""
        from whisperjav.pipelines.qwen_pipeline import QwenPipeline

        sig = inspect.signature(QwenPipeline.__init__)
        assert sig.parameters["drop_nonverbal_lines"].default is True
