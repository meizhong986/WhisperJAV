"""Tests for NonlinguisticUtteranceFilter — sound-only subtitle removal."""

import pytest

from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
    NonlinguisticUtteranceFilter,
    _has_evidence_of_language,
    _is_japanese_sound_line,
    _should_remove_entry,
)


# ── Evidence detection ──────────────────────────────────────────────────


class TestHasEvidenceOfLanguage:
    def test_kanji_is_evidence(self):
        assert _has_evidence_of_language("精液")

    def test_particle_wo(self):
        assert _has_evidence_of_language("それを")

    def test_particle_ga(self):
        assert _has_evidence_of_language("あが")

    def test_verb_ending_masu(self):
        assert _has_evidence_of_language("あります")

    def test_verb_ending_desu(self):
        assert _has_evidence_of_language("です")

    def test_common_word_kimochiii(self):
        assert _has_evidence_of_language("気持ちいい")

    def test_common_word_motto(self):
        assert _has_evidence_of_language("もっと")

    def test_demonstrative_aa(self):
        assert _has_evidence_of_language("ああ")

    def test_interjection_ett(self):
        assert _has_evidence_of_language("えっ")

    def test_question_mark(self):
        assert _has_evidence_of_language("あぁ？")

    def test_sentence_ending_yo(self):
        assert _has_evidence_of_language("あぁよ")

    def test_sentence_ending_ne(self):
        assert _has_evidence_of_language("いいね")

    def test_ha_particle_after_noun(self):
        assert _has_evidence_of_language("それは")

    def test_ha_breathing_not_evidence(self):
        assert not _has_evidence_of_language("はぁ")

    def test_ha_repeated_breathing(self):
        assert not _has_evidence_of_language("はぁはぁ")

    def test_pure_sound_kana_no_evidence(self):
        assert not _has_evidence_of_language("あぁっ")

    def test_katakana_two_distinct_is_evidence(self):
        assert _has_evidence_of_language("ハァハァ")

    def test_katakana_one_distinct_no_evidence(self):
        assert not _has_evidence_of_language("アアア")

    def test_common_word_dame(self):
        assert _has_evidence_of_language("ダメ")

    def test_common_word_yamete(self):
        assert _has_evidence_of_language("やめて")


# ── Sound-line classification ───────────────────────────────────────────


class TestIsJapaneseSoundLine:
    def test_pure_moan(self):
        assert _is_japanese_sound_line("あぁっ")

    def test_breathing(self):
        assert _is_japanese_sound_line("はぁはぁ")

    def test_sucking(self):
        assert _is_japanese_sound_line("ちゅぱちゅぱ")

    def test_nasal(self):
        assert _is_japanese_sound_line("んっんっ")

    def test_prolonged_moan(self):
        assert _is_japanese_sound_line("あぁぁぁーっ")

    def test_slurping(self):
        assert _is_japanese_sound_line("ずずっ")

    def test_swallowing(self):
        assert _is_japanese_sound_line("ごくっ")

    def test_licking(self):
        assert _is_japanese_sound_line("ぺろぺろ")

    def test_kissing(self):
        assert _is_japanese_sound_line("ちゅっ")

    def test_puffing(self):
        assert _is_japanese_sound_line("ぷはぁ")

    def test_dialogue_kept(self):
        assert not _is_japanese_sound_line("気持ちいい")

    def test_verb_kept(self):
        assert not _is_japanese_sound_line("やめて")

    def test_mixed_kept(self):
        assert not _is_japanese_sound_line("あぁ、もっと")

    def test_non_japanese_kept(self):
        assert not _is_japanese_sound_line("Oh yeah")

    def test_empty_kept(self):
        assert not _is_japanese_sound_line("")

    def test_pure_punctuation_kept(self):
        assert not _is_japanese_sound_line("...")

    def test_zurui_false_positive_guard(self):
        """ずるい — all chars in SOUND_KANA but it's a real word."""
        assert not _is_japanese_sound_line("ずるい")

    def test_furui_false_positive_guard(self):
        """ふるい — all chars in SOUND_KANA but it's a real word."""
        assert not _is_japanese_sound_line("ふるい")

    def test_single_sound_char(self):
        assert _is_japanese_sound_line("あ")

    def test_sound_with_period(self):
        assert _is_japanese_sound_line("あぁっ。")

    def test_sound_with_ellipsis(self):
        assert _is_japanese_sound_line("はぁ…")

    def test_non_sound_kana_blocks_removal(self):
        """な is not in SOUND_KANA — its presence prevents removal."""
        assert not _is_japanese_sound_line("なぁ")


# ── Whitelist false-positive guards ─────────────────────────────────────


class TestWhitelistFalsePositiveGuards:
    """Words composed entirely of SOUND_KANA characters, protected by _COMMON_WORDS."""

    def test_all_sk_verbs_kept(self):
        for word in ["ふる", "ぬる", "ぬぐ", "つぐ", "ふく", "つく", "むく",
                      "くぐる", "ふれる", "つれる", "くれる", "ぐれる",
                      "ふるえる", "ふくれる", "くずれる", "つぶれる",
                      "うごめく", "くずす", "ふくむ", "ふるう", "つぶす"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"

    def test_all_sk_adjectives_kept(self):
        for word in ["ぬるい", "えぐい", "ずるい", "ふるい"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"

    def test_all_sk_nouns_kept(self):
        for word in ["ちち", "うち", "おく", "こえ", "ふじ", "おふろ",
                      "ふくろ", "ちず", "こぶ", "こめ", "くず", "ふち",
                      "つち", "いぬ", "ちえ", "すじ", "ふぐ", "めす"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"

    def test_all_sk_adverbs_kept(self):
        for word in ["すぐ", "ぐっ", "じっ"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"

    def test_all_sk_mimetics_kept(self):
        for word in ["めちゃめちゃ", "ごちゃごちゃ", "ぐちゃぐちゃ",
                      "ごろごろ", "ぐるぐる", "ぶつぶつ", "ぷちぷち",
                      "じめじめ", "ぬめぬめ", "ぷるぷる", "ぶるぶる"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"

    def test_sound_effects_not_broken_by_whitelist(self):
        """Whitelist must not protect sound effects via substring matching."""
        for sound in ["ごくっ", "ごくごく", "くちゅ", "くちゅくちゅ",
                       "ぐちゅ", "ぐちゅぐちゅ"]:
            assert _is_japanese_sound_line(sound), f"{sound} incorrectly kept"

    def test_original_feedback_words_all_safe(self):
        """All 16 words from the false-positive feedback are safe."""
        for word in ["すごい", "すき", "したい", "しい", "おしい", "かしい",
                      "くさい", "うまい", "ひどい", "こわい", "つらい",
                      "きつい", "まずい", "やばい", "すごく", "ほしい"]:
            assert not _is_japanese_sound_line(word), f"{word} incorrectly removed"


# ── Entry-level decision (bilingual support) ────────────────────────────


class TestShouldRemoveEntry:
    def test_single_line_sound(self):
        assert _should_remove_entry("あぁっ")

    def test_single_line_dialogue(self):
        assert not _should_remove_entry("もっと")

    def test_bilingual_sound_jp_removes(self):
        assert _should_remove_entry("あぁっ\nAhh")

    def test_bilingual_dialogue_jp_keeps(self):
        assert not _should_remove_entry("もっと\nMore")

    def test_empty(self):
        assert not _should_remove_entry("")

    def test_english_only_kept(self):
        assert not _should_remove_entry("Hello world")

    def test_whitespace_only_kept(self):
        assert not _should_remove_entry("   \n   ")


# ── SRT file integration ───────────────────────────────────────────────


class TestFilterSrtFile:
    def test_mixed_srt(self, tmp_path):
        srt_content = (
            "1\n00:00:01,000 --> 00:00:02,000\nあぁっ\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\n気持ちいい\n\n"
            "3\n00:00:05,000 --> 00:00:06,000\nはぁはぁ\n\n"
            "4\n00:00:07,000 --> 00:00:08,000\nもっと\n\n"
            "5\n00:00:09,000 --> 00:00:10,000\nんっんっ\n\n"
        )
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        stats = NonlinguisticUtteranceFilter().filter_srt_file(srt_path)

        assert stats["original_count"] == 5
        assert stats["dropped_nonlinguistic"] == 3
        assert stats["final_count"] == 2

        import pysrt
        subs = pysrt.open(str(srt_path), encoding="utf-8")
        assert len(subs) == 2
        assert "気持ちいい" in subs[0].text
        assert "もっと" in subs[1].text
        assert subs[0].index == 1
        assert subs[1].index == 2

    def test_all_dialogue_unchanged(self, tmp_path):
        srt_content = (
            "1\n00:00:01,000 --> 00:00:02,000\nやめて\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nダメ\n\n"
        )
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        stats = NonlinguisticUtteranceFilter().filter_srt_file(srt_path)

        assert stats["original_count"] == 2
        assert stats["dropped_nonlinguistic"] == 0
        assert stats["final_count"] == 2

    def test_empty_file(self, tmp_path):
        srt_path = tmp_path / "empty.srt"
        srt_path.write_text("", encoding="utf-8")

        stats = NonlinguisticUtteranceFilter().filter_srt_file(srt_path)
        assert stats["original_count"] == 0

    def test_nonexistent_file(self, tmp_path):
        stats = NonlinguisticUtteranceFilter().filter_srt_file(
            tmp_path / "nonexistent.srt"
        )
        assert stats["original_count"] == 0

    def test_bilingual_srt(self, tmp_path):
        srt_content = (
            "1\n00:00:01,000 --> 00:00:02,000\nあぁっ\nAhh\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nもっと\nMore\n\n"
        )
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        stats = NonlinguisticUtteranceFilter().filter_srt_file(srt_path)

        assert stats["dropped_nonlinguistic"] == 1
        assert stats["final_count"] == 1

        import pysrt
        subs = pysrt.open(str(srt_path), encoding="utf-8")
        assert "もっと" in subs[0].text

    def test_drops_empty_entries(self, tmp_path):
        srt_content = (
            "1\n00:00:01,000 --> 00:00:02,000\n\n\n"
            "2\n00:00:03,000 --> 00:00:04,000\nもっと\n\n"
        )
        srt_path = tmp_path / "test.srt"
        srt_path.write_text(srt_content, encoding="utf-8")

        stats = NonlinguisticUtteranceFilter().filter_srt_file(srt_path)

        assert stats["dropped_empty"] == 1
        assert stats["final_count"] == 1
