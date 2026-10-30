"""
Taking one VAD group's CrispASR result apart and building its subtitles (whisperjav/crispasr/lines.py;
route B design 7.2, 7.3; owner W175, W203, W204).

Parakeet (line rule "tokens"): cut only at natural cues -- a sentence-end mark, a silence of at least
0.6 s, a 、 or a space between Japanese characters followed by a silence of at least 0.4 s -- never by
characters or duration; every time from the tokens. Cohere (line rule "group"): one subtitle with the
VAD group's edges.

The token data of groups 3 and 4 are Parakeet's own, from the binding check of 2026-10-24
(D:\\WJ110_measure\\crisp_binding_check_2026-10-24\\out\\check_binding.json); the check saved the first 12
tokens of group 4, so the rest of that group below is made for the test. No ASR import.

Run with: python -m pytest tests/test_crispasr_lines.py -q
"""
import pytest

from whisperjav.crispasr import lines
from whisperjav.crispasr.engine import CrispSegment, CrispToken, CrispWord

G0 = 100.0  # the group's start in media-file time


def toks(*items):
    return tuple(CrispToken(text, start, end, 1.0) for text, start, end in items)


def seg(tokens, text=None, start=None, end=None, words=()):
    tokens = toks(*tokens) if tokens and not isinstance(tokens[0], CrispToken) else tuple(tokens)
    text = text if text is not None else "".join(t.text for t in tokens).strip()
    timed = [t for t in tokens if t.start is not None]
    start = start if start is not None else (timed[0].start if timed else 0.0)
    end = end if end is not None else (timed[-1].end if timed else 0.0)
    return CrispSegment(text, start, end, tuple(words), tokens)


def build(segments, duration, stats=None):
    return lines.build("tokens", G0, G0 + duration, segments, stats)


def rounded(subs):
    return [(t, round(s - G0, 3), round(e - G0, 3)) for t, s, e in subs]


# Parakeet, binding check of 2026-10-24
GROUP_3 = [(" そう", 0.0, 0.32), ("ですよね", 0.64, 0.96), ("監", 1.92, 2.24), ("督", 2.24, 2.56), ("。", 2.56, 2.88)]
GROUP_4 = [(" ", 0.0, 0.16), ("やっぱり", 0.16, 0.40), ("本", 0.40, 0.56), ("番", 0.56, 0.80), ("行", 0.80, 0.96),
           ("為", 0.96, 1.12), ("は", 1.12, 1.28), ("問題", 1.28, 1.52), ("に", 1.52, 1.68), ("なります", 1.68, 1.76),
           ("よ", 1.76, 1.92), ("。", 1.92, 2.00),
           # made for the test from here: the words "俺" 2.00-2.16 and "が" 2.16-2.32 of the check, then the rest
           ("俺", 2.00, 2.16), ("が", 2.16, 2.32), ("責任", 2.32, 2.64), ("取る", 2.64, 2.88), ("って", 2.88, 3.04),
           ("言って", 3.04, 3.36), ("る", 3.36, 3.44), ("だろ", 3.44, 3.76), ("。", 3.76, 3.84)]


class TestParakeetFromTheBindingCheck:
    def test_group_3_is_cut_at_its_pause_and_ends_where_speech_ends(self):
        subs = build([seg(GROUP_3, text="そうですよね監督。")], 3.12)
        assert rounded(subs) == [("そうですよね", 0.0, 0.96), ("監督。", 1.92, 2.56)]

    def test_group_4_is_cut_after_its_sentence_end_mark(self):
        subs = build([seg(GROUP_4)], 3.98)
        assert rounded(subs) == [("やっぱり本番行為は問題になりますよ。", 0.16, 1.92),
                                 ("俺が責任取るって言ってるだろ。", 2.0, 3.76)]

    def test_the_blank_token_and_a_leading_space_are_not_text(self):
        subs = build([seg(GROUP_3)], 3.12)
        assert all(not t.startswith(" ") for t, _s, _e in subs)


class TestTheCues:
    def test_a_silence_of_0_6_s_cuts(self):
        subs = build([seg([("あの", 0.0, 0.4), ("ね", 1.0, 1.2)])], 2.0)
        assert [t for t, _s, _e in subs] == ["あの", "ね"]

    def test_a_shorter_silence_without_a_mark_does_not_cut(self):
        subs = build([seg([("あの", 0.0, 0.4), ("ね", 0.96, 1.2)])], 2.0)
        assert [t for t, _s, _e in subs] == ["あのね"]

    def test_a_comma_with_a_silence_of_0_4_s_cuts_and_closes_the_first_subtitle(self):
        subs = build([seg([("ねえ", 0.0, 0.4), ("、", 0.4, 0.48), ("待って", 0.8, 1.2)])], 2.0)
        assert rounded(subs) == [("ねえ、", 0.0, 0.4), ("待って", 0.8, 1.2)]

    def test_a_comma_without_a_silence_stays_inside_the_subtitle(self):
        subs = build([seg([("ねえ", 0.0, 0.4), ("、", 0.4, 0.48), ("待って", 0.6, 1.2)])], 2.0)
        assert [t for t, _s, _e in subs] == ["ねえ、待って"]

    def test_a_sentence_end_mark_cuts_without_any_silence(self):
        subs = build([seg([("はい", 0.0, 0.3), ("。", 0.3, 0.38), ("行く", 0.38, 0.7)])], 1.0)
        assert [t for t, _s, _e in subs] == ["はい。", "行く"]

    @pytest.mark.parametrize("mark", ["？", "！", "…", "?", "!", "．", "."])
    def test_every_sentence_end_mark_cuts(self, mark):
        subs = build([seg([("本当", 0.0, 0.3), (mark, 0.3, 0.38), ("うそ", 0.38, 0.7)])], 1.0)
        assert len(subs) == 2

    def test_a_space_between_japanese_characters_with_a_silence_cuts(self):
        segment = seg([("そう", 0.0, 0.3), (" だね", 0.8, 1.1)], text="そう だね")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["そう", "だね"]

    def test_a_space_between_japanese_characters_without_a_silence_stays(self):
        segment = seg([("そう", 0.0, 0.3), (" だね", 0.4, 0.7)], text="そう だね")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["そう だね"]

    def test_a_word_start_marker_is_no_space_when_the_text_has_none(self):
        segment = seg([("そう", 0.0, 0.3), (" だね", 0.8, 1.1)], text="そうだね")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["そうだね"]

    def test_a_space_between_latin_words_is_kept_and_never_cuts(self):
        segment = seg([("OK", 0.0, 0.3), (" Google", 0.7, 1.1)], text="OK Google")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["OK Google"]

    def test_a_closing_bracket_after_the_end_mark_closes_and_an_opening_one_opens(self):
        subs = build([seg([("「", 0.0, 0.0), ("だろ", 0.0, 0.3), ("。", 0.3, 0.38), ("」", 0.38, 0.4),
                           ("「", 0.4, 0.4), ("ああ", 0.5, 0.8), ("」", 0.8, 0.85)])], 1.0)
        assert [t for t, _s, _e in subs] == ["「だろ。」", "「ああ」"]


class TestThePauseValuesAreParameters:
    """W208: "ideally the values are not hardcoded and can be adjusted as parameters"."""

    PAUSE = [("あの", 0.0, 0.4), ("ね", 1.0, 1.2)]                          # a 0.6 s silence, no mark
    COMMA = [("ねえ", 0.0, 0.4), ("、", 0.4, 0.48), ("待って", 0.75, 1.2)]  # a 、 with a 0.35 s silence

    def test_the_defaults_are_0_6_s_and_0_4_s(self):
        assert lines.DEFAULT_CUES == lines.CueSettings(0.6, 0.4)
        assert lines.DEFAULT_CUES.as_dict() == {"pause_cut_s": 0.6, "weak_mark_pause_s": 0.4}

    def test_a_longer_pause_value_keeps_a_0_6_s_silence_inside(self):
        cues = lines.CueSettings(pause_cut_s=0.7)
        subs = lines.build("tokens", G0, G0 + 2.0, [seg(self.PAUSE)], None, cues)
        assert [t for t, _s, _e in subs] == ["あのね"]

    def test_a_shorter_weak_mark_value_cuts_at_a_shorter_silence_after_a_comma(self):
        assert [t for t, _s, _e in build([seg(self.COMMA)], 2.0)] == ["ねえ、待って"]
        cues = lines.CueSettings(weak_mark_pause_s=0.3)
        subs = lines.build("tokens", G0, G0 + 2.0, [seg(self.COMMA)], None, cues)
        assert [t for t, _s, _e in subs] == ["ねえ、", "待って"]

    def test_the_weak_mark_value_never_cuts_without_a_mark(self):
        cues = lines.CueSettings(pause_cut_s=5.0, weak_mark_pause_s=0.1)
        subs = lines.build("tokens", G0, G0 + 2.0, [seg(self.PAUSE)], None, cues)
        assert [t for t, _s, _e in subs] == ["あのね"]

    @pytest.mark.parametrize("bad", [{"pause_cut_s": 0}, {"weak_mark_pause_s": -0.1}])
    def test_a_value_of_0_or_less_is_refused(self, bad):
        with pytest.raises(ValueError, match="must be more than 0 s"):
            lines.CueSettings(**bad)

    def test_the_group_rule_takes_no_cues(self):
        cues = lines.CueSettings(pause_cut_s=0.1)
        assert lines.build("group", G0, G0 + 2.0, [seg(self.PAUSE)], None, cues) == [("あのね", G0, G0 + 2.0)]


class TestNoBruteForce:
    def test_a_long_run_without_cues_stays_one_subtitle(self):
        pieces = [(c, i * 0.08, (i + 1) * 0.08) for i, c in enumerate("あいうえおかきくけこさしすせそたちつてと" * 3)]
        subs = build([seg(pieces)], 6.0)
        assert len(subs) == 1 and len(subs[0][0]) == 60
        assert round(subs[0][2] - subs[0][1], 2) == 4.8

    def test_a_subtitle_longer_than_the_netflix_limit_is_not_cut(self):
        subs = build([seg([("ずっと", 0.0, 3.0), ("話して", 3.0, 5.9)])], 6.0)
        assert len(subs) == 1


class TestTimes:
    def test_a_subtitle_ends_at_its_last_spoken_token_not_at_the_mark(self):
        (sub,) = build([seg([("督", 2.24, 2.56), ("。", 2.56, 2.88)])], 3.12)
        assert round(sub[2] - G0, 3) == 2.56

    def test_subtitles_of_one_group_never_overlap(self):
        subs = build([seg([("あ", 0.0, 0.5), ("。", 0.5, 0.5), ("い", 0.4, 0.9), ("。", 0.9, 1.0), ("う", 0.8, 1.2)])], 2.0)
        for (_t1, _s1, e1), (_t2, s2, _e2) in zip(subs, subs[1:]):
            assert e1 <= s2

    def test_times_outside_the_group_are_clamped_and_counted(self):
        stats = lines.LineStats()
        subs = build([seg([("あ", -0.3, 0.4), ("。", 0.4, 0.4), ("い", 0.6, 2.5)])], 2.0, stats)
        assert rounded(subs) == [("あ。", 0.0, 0.4), ("い", 0.6, 2.0)]
        assert stats.clamped_times == 2

    def test_zero_length_tokens_get_a_short_display_time(self):
        (sub,) = build([seg([("あ", 0.5, 0.5)])], 2.0)
        assert round(sub[2] - sub[1], 3) == lines.MIN_DISPLAY_S

    def test_a_zero_length_token_at_the_groups_very_end_moves_its_start(self):
        (sub,) = build([seg([("あ", 2.0, 2.0)])], 2.0)
        assert round(sub[1] - G0, 3) == 1.9 and round(sub[2] - G0, 3) == 2.0

    def test_starts_that_go_backwards_are_held(self):
        subs = build([seg([("あ", 0.5, 0.7), ("い", 0.3, 0.6)])], 1.0)
        assert subs[0][1] <= subs[-1][1]


class TestWhatIsNoCue:
    """Found by the review of 2026-10-25: only the marks of design 7.2 Step A.3 are cues."""

    @pytest.mark.parametrize("left, mark, right", [("ジョン", "・", "スミス"), ("ね", "〜", "そう"), ("A", "-", "B"),
                                                   ("そう", "'", "だ")])
    def test_a_mark_inside_words_never_cuts_even_with_a_pause(self, left, mark, right):
        subs = build([seg([(left, 0.0, 0.3), (mark, 0.3, 0.3), (right, 0.75, 1.0)])], 2.0)
        assert [t for t, _s, _e in subs] == [left + mark + right]

    def test_a_space_next_to_latin_text_is_kept_and_never_cuts(self):
        segment = seg([("これは", 0.0, 0.4), (" OK", 0.85, 1.0), (" です", 1.45, 1.7)], text="これは OK です")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["これは OK です"]

    def test_a_word_start_marker_is_told_from_a_real_space_by_its_position(self):
        segment = seg([(" そう", 0.0, 0.3), (" です", 0.75, 1.0), ("よね", 1.0, 1.2), (" 監督", 1.65, 2.0)],
                      text="そうですよね 監督")
        assert [t for t, _s, _e in build([segment], 3.0)] == ["そうですよね", "監督"]

    @pytest.mark.parametrize("space", [" ", "　"])
    def test_a_space_that_is_a_token_of_its_own_is_a_cue_with_a_pause(self, space):
        segment = seg([("そうだね", 0.0, 0.4), (space, 0.4, 0.5), ("行こう", 0.9, 1.3)], text=f"そうだね{space}行こう")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["そうだね", "行こう"]

    def test_a_space_that_is_a_token_of_its_own_without_a_pause_stays_in_the_text(self):
        segment = seg([("そうだね", 0.0, 0.4), (" ", 0.4, 0.45), ("行こう", 0.5, 0.9)], text="そうだね 行こう")
        assert [t for t, _s, _e in build([segment], 2.0)] == ["そうだね 行こう"]

    def test_the_display_repair_never_runs_into_the_next_subtitle(self):
        subs = build([seg([("あ", 1.0, 1.0), ("。", 1.0, 1.0), ("い", 1.05, 1.5)])], 2.0)
        assert len(subs) == 2 and subs[0][2] <= subs[1][1]

    def test_a_closing_mark_before_the_groups_first_word_is_dropped(self):
        subs = build([seg([("。", 0.0, 0.1), ("あいう", 0.2, 0.5)])], 1.0)
        assert [t for t, _s, _e in subs] == ["あいう"]


class TestFallbacks:
    def test_a_token_without_time_uses_the_words_with_the_mark_split_off(self):
        tokens = (CrispToken("監", 1.92, 2.24, 1.0), CrispToken("督", None, None, None))
        words = (CrispWord("監督。", 1.92, 2.88, 1.0), CrispWord("はい", 3.6, 3.8, 1.0))
        stats = lines.LineStats()
        subs = build([CrispSegment("監督。はい", 1.92, 3.8, words, tokens)], 4.0, stats)
        assert rounded(subs) == [("監督。", 1.92, 2.88), ("はい", 3.6, 3.8)]
        assert stats.fallback_segments == 1

    def test_no_times_at_all_give_one_subtitle_with_the_segments_times(self):
        tokens = (CrispToken("あ", None, None, None),)
        subs = build([CrispSegment("あいうえお。かきくけこ。", 0.2, 1.8, (), tokens)], 2.0)
        assert rounded(subs) == [("あいうえお。かきくけこ。", 0.2, 1.8)]

    def test_a_segment_with_no_text_gives_nothing(self):
        assert build([CrispSegment("", 0.0, 0.0, (), ())], 1.0) == []

    def test_marks_alone_give_no_subtitle(self):
        assert build([seg([("。", 0.0, 0.1)])], 1.0) == []


class TestCohere:
    def test_one_subtitle_with_the_groups_edges(self):
        segment = CrispSegment("本番をやらなくてもいいんだよ。", 0.0, 4.08,
                               (CrispWord("本番をやらなくてもいいんだよ。", 0.4, 4.08, 1.0),),
                               (CrispToken("本", 0.4, 0.4, 0.98),))
        assert lines.build("group", G0, G0 + 4.08, [segment]) == [("本番をやらなくてもいいんだよ。", G0, G0 + 4.08)]

    def test_empty_text_gives_no_subtitle(self):
        assert lines.build("group", G0, G0 + 1.0, [CrispSegment("  ", 0.0, 1.0)]) == []

    def test_several_segments_are_joined(self):
        assert lines.build("group", G0, G0 + 2.0, [CrispSegment("あ", 0, 1), CrispSegment("い。", 1, 2)])[0][0] == "あい。"


class TestOnsetTimes:
    """Cohere-CrispASR aligned (owner W259; design note 2026-10-28, section 6): the aligner's character times are
    onsets, so only a sentence-end mark cuts, and each subtitle of a group but the last ends where the next starts."""

    # wav2vec2-ja spikes for "…た。それで？" (probe P1); the last unit ends at the group's end.
    SPIKES = [("った", 2.60, 2.64), ("た。", 2.66, 2.68), ("そ", 3.10, 3.12), ("れ", 3.34, 3.36), ("で？", 3.76, 3.79)]

    def onsets(self, items, duration):
        return rounded(lines.build("tokens", G0, G0 + duration, [seg(items)], onsets=True))

    def test_inner_subtitles_end_where_the_next_starts(self):
        assert self.onsets(self.SPIKES, 3.79) == [("ったた。", 2.6, 3.1), ("それで？", 3.1, 3.79)]

    def test_without_onsets_the_ends_are_the_tokens(self):
        assert rounded(build([seg(self.SPIKES)], 3.79)) == [("ったた。", 2.6, 2.68), ("それで？", 3.1, 3.79)]

    def test_a_held_sound_is_no_pause_cut(self):
        # 激しかったの (step 4 run, 115.66 s): 0.6 s and more between the onsets of 激 and し, no mark.
        items = [("激", 0.60, 0.62), ("し", 2.08, 2.10), ("か", 2.30, 2.32), ("った", 2.50, 2.54), ("の。", 2.80, 3.14)]
        assert self.onsets(items, 3.14) == [("激しかったの。", 0.6, 3.14)]
        assert len(build([seg(items)], 3.14)) == 2  # Parakeet's pause cue would cut it

    def test_a_comma_with_a_pause_is_no_cut(self):
        items = [("竹", 0.0, 0.02), ("二、", 0.3, 0.32), ("竹", 1.1, 1.12), ("二", 1.3, 1.32), ("様", 2.5, 2.52),
                 ("って。", 2.9, 3.53)]
        assert self.onsets(items, 3.53) == [("竹二、竹二様って。", 0.0, 3.53)]

    @pytest.mark.parametrize("stop", [".", "．"])
    def test_a_full_stop_in_a_number_is_no_cut(self, stop):
        # Owner W266 3: with onset times the ASCII and full-width full stops do not cut.
        items = [("3" + stop, 0.0, 0.02), ("5", 0.3, 0.32), ("キ", 0.6, 0.62), ("ロ。", 0.9, 1.2)]
        assert self.onsets(items, 1.2) == [(f"3{stop}5キロ。", 0.0, 1.2)]

    def test_ascii_question_and_exclamation_marks_cut(self):
        items = [("何", 0.0, 0.02), ("?", 0.02, 0.02), ("う", 0.5, 0.52), ("ん!", 0.8, 1.0)]
        assert self.onsets(items, 1.0) == [("何?", 0.0, 0.5), ("うん!", 0.5, 1.0)]

    def test_parakeet_keeps_the_full_stop_cue(self):
        items = [("3.", 0.0, 0.02), ("5", 0.3, 0.32), ("キ", 0.6, 0.62), ("ロ。", 0.9, 1.2)]
        assert len(build([seg(items)], 1.2)) == 2


class TestTheRule:
    def test_an_unknown_rule_is_refused(self):
        with pytest.raises(ValueError):
            lines.build("words", G0, G0 + 1.0, [])

    @pytest.mark.parametrize("ch, closing", [("、", True), ("。", True), ("」", True), ("！", True),
                                             ("「", False), ("・", False), ("'", False), ("〜", False)])
    def test_is_closing_mark(self, ch, closing):
        assert lines.is_closing_mark(ch) is closing
