"""
The timing model's rules (whisperjav/modules/subtitle_pipeline/timing_rules.py; item 12 C3, design note
docs/plans/design/2026-10-28-chronosjav-timing-c3-c1.md section 5.2, rules T1-T8 as revised after W255).

The inputs are the probe figures of 2026-10-27 (design note section 14): group A of The.Naked.Director
S02E02 Scene 5 (6.40-10.19 s), the wav2vec2-ja aligner's character times, and Whisper small's DTW onsets.
Pure: no model, no file.

Run with: python -m pytest tests/test_timing_rules.py -q
"""
import pytest

from whisperjav.modules.subtitle_pipeline.timing_rules import (
    NO_SPEECH_THRESHOLD,
    TimedSegment,
    counted,
    pieces,
    speech_start,
    split_points,
)

FRAME_S = 3.79
# wav2vec2-ja on group A (probe P1); the last end is the frame's end (step 2's aligner).
GROUP_A = [("近", 0.26, 0.28), ("く", 0.38, 0.40), ("の", 0.48, 0.50), ("カ", 0.60, 0.62), ("ラ", 0.72, 0.74),
           ("オ", 0.80, 0.82), ("ケ", 0.90, 0.92), ("で", 1.00, 1.02), ("そ", 1.12, 1.14), ("の", 1.26, 1.28),
           ("ま", 1.52, 1.54), ("ま", 1.74, 1.76), ("酔", 1.78, 1.80), ("っ", 1.86, 1.88), ("ぱ", 1.90, 1.92),
           ("ら", 2.08, 2.10), ("っ", 2.32, 2.34), ("ち", 2.36, 2.38), ("ゃ", 2.40, 2.42), ("っ", 2.60, 2.64),
           ("た。", 2.66, 2.68), ("そ", 3.10, 3.12), ("れ", 3.34, 3.36), ("で？", 3.76, 3.79)]
# Whisper small with DTW on group A (probe P2b): three segments, onsets 0.32, 2.00 (inside a word) and 3.30.
WHISPER_A = [TimedSegment(0.32, 0.0, True), TimedSegment(2.00, 0.0, True), TimedSegment(3.30, 0.0, True)]


def words(spec):
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def text_and_times(parts):
    return [("".join(w["word"] for w in p), p[0]["start"], p[-1]["end"]) for p in parts]


class TestCounted:
    def test_a_segment_counts_with_text_an_onset_and_little_doubt(self):
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(None, 0.0, True), TimedSegment(0.5, 0.0, False),
                TimedSegment(0.7, NO_SPEECH_THRESHOLD, True), TimedSegment(0.9, 0.59, True),
                TimedSegment(1.1, -1.0, True)]
        assert [s.onset for s in counted(segs)] == [0.1, 0.9, 1.1]

    def test_the_threshold_is_whispers_own(self):
        assert NO_SPEECH_THRESHOLD == 0.6


class TestSpeechStart:
    def test_the_first_counted_onset(self):
        assert speech_start(WHISPER_A, FRAME_S) == 0.32

    def test_nothing_counted_changes_nothing(self):
        assert speech_start([], FRAME_S) is None
        assert speech_start([TimedSegment(0.5, 0.9, True)], FRAME_S) is None

    def test_never_outside_the_frame(self):
        assert speech_start([TimedSegment(-0.2, 0.0, True)], FRAME_S) == 0.0
        assert speech_start([TimedSegment(3.79, 0.0, True)], FRAME_S) is None


class TestGroupA:
    def test_the_sentence_change_becomes_two_subtitles(self):
        got = text_and_times(pieces(words(GROUP_A), WHISPER_A, FRAME_S))
        assert got == [("近くのカラオケでそのまま酔っぱらっちゃった。", 0.32, 3.30), ("それで？", 3.30, 3.79)]

    def test_the_boundary_inside_a_word_gives_no_split(self):
        # Whisper's 2.00 s boundary has no mark between its onsets 0.32 and 2.00.
        assert split_points(words(GROUP_A), WHISPER_A) == [(20, 3.30)]

    def test_the_input_words_are_not_changed(self):
        w = words(GROUP_A)
        pieces(w, WHISPER_A, FRAME_S)
        assert w == words(GROUP_A)

    def test_with_split_off_only_the_start_moves(self):
        got = text_and_times(pieces(words(GROUP_A), WHISPER_A, FRAME_S, allow_split=False))
        assert got == [("近くのカラオケでそのまま酔っぱらっちゃった。それで？", 0.32, 3.79)]

    def test_without_counted_segments_nothing_changes(self):
        assert pieces(words(GROUP_A), [], FRAME_S) == [words(GROUP_A)]


class TestSplits:
    def test_a_comma_is_a_cut_mark(self):
        w = words([("は", 0.1, 0.12), ("い、", 0.3, 0.32), ("そ", 1.0, 1.02), ("う", 1.2, 1.5)])
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(1.0, 0.0, True)]
        assert text_and_times(pieces(w, segs, 1.5)) == [("はい、", 0.1, 1.0), ("そう", 1.0, 1.5)]

    def test_a_space_between_japanese_characters_is_a_cut_mark(self):
        w = words([("あ", 0.1, 0.12), ("い ", 0.3, 0.32), ("う", 1.0, 1.5)])
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(1.0, 0.0, True)]
        assert len(pieces(w, segs, 1.5)) == 2

    def test_no_mark_no_split(self):
        w = words([("あ", 0.1, 0.12), ("い", 0.3, 0.32), ("う", 1.0, 1.5)])
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(1.0, 0.0, True)]
        assert len(pieces(w, segs, 1.5)) == 1

    def test_a_mark_outside_the_stretch_between_the_onsets_is_not_taken(self):
        w = words([("あ。", 0.1, 0.12), ("い", 0.5, 0.52), ("う", 1.0, 1.5)])
        segs = [TimedSegment(0.3, 0.0, True), TimedSegment(1.0, 0.0, True)]
        assert len(pieces(w, segs, 1.5)) == 1

    def test_the_mark_nearest_the_next_onset_is_taken(self):
        w = words([("あ。", 0.2, 0.22), ("い。", 0.8, 0.82), ("う", 1.0, 1.5)])
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(1.0, 0.0, True)]
        assert split_points(w, segs) == [(1, 1.0)]

    def test_never_after_the_last_word(self):
        w = words([("あ", 0.1, 0.12), ("い。", 0.3, 0.32)])
        segs = [TimedSegment(0.1, 0.0, True), TimedSegment(0.9, 0.0, True)]
        assert split_points(w, segs) == []

    def test_a_doubted_segment_makes_no_boundary(self):
        segs = [TimedSegment(0.32, 0.0, True), TimedSegment(2.00, 0.0, True), TimedSegment(3.30, 0.95, True)]
        assert len(pieces(words(GROUP_A), segs, FRAME_S)) == 1


class TestTimes:
    @pytest.mark.parametrize("allow_split", [True, False])
    def test_times_stay_in_the_frame_and_never_run_backwards(self, allow_split):
        flat = [w for p in pieces(words(GROUP_A), WHISPER_A, FRAME_S, allow_split) for w in p]
        assert all(0.0 <= w["start"] <= w["end"] <= FRAME_S for w in flat)
        assert all(a["start"] <= b["start"] for a, b in zip(flat, flat[1:]))

    def test_words_before_a_late_onset_move_to_it(self):
        w = words([("あ", 0.1, 0.12), ("い", 0.2, 0.22), ("う", 0.9, 1.5)])
        part = pieces(w, [TimedSegment(0.5, 0.0, True)], 1.5)[0]
        assert [(x["start"], x["end"]) for x in part] == [(0.5, 0.5), (0.5, 0.5), (0.9, 1.5)]

    def test_empty_words_give_nothing(self):
        assert pieces([], WHISPER_A, FRAME_S) == []
