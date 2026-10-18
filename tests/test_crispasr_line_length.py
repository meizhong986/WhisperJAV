"""The CrispASR pipelines' line-length rule (whisperjav/crispasr/line_length.py; master plan 3.8 item 3, W138 3).

Pure functions: no stable-ts, no ASR import. The pipelines' use of the rule, with the real stable-ts line building,
is covered in tests/test_crispasr_pipelines.py.
"""

from types import SimpleNamespace

import pytest

from whisperjav.crispasr.line_length import (OFF, LineLimits, is_closing_mark, is_closing_marks_only,
                                             lines_from_segments, split_line)


def approx_lines(actual, expected, tol=1e-9):
    assert [t for t, _, _ in actual] == [t for t, _, _ in expected]
    assert [v for _, a, b in actual for v in (a, b)] == pytest.approx([v for _, a, b in expected for v in (a, b)],
                                                                      abs=tol)


# A 6 s line of six pieces with marks after the 2nd and 4th
PIECES = [("今日", 0.0, 1.0), ("は、", 1.0, 2.0), ("明日", 2.0, 3.0), ("も。", 3.0, 4.0), ("それ", 4.0, 5.0),
          ("から", 5.0, 6.0)]
TEXT = "今日は、明日も。それから"


def test_off_is_the_default_and_changes_nothing():
    assert OFF == LineLimits() and not OFF.active
    assert OFF.as_dict() == {"max_s": None, "max_chars": None}
    assert split_line(TEXT, 0.0, 6.0, PIECES, OFF) == [(TEXT, 0.0, 6.0)]
    assert LineLimits(max_s=5.0).active and LineLimits(max_chars=20).active


@pytest.mark.parametrize("ch", list("、。！？!?…，,.」』)】”"))
def test_closing_and_separating_marks_end_a_line(ch):
    assert is_closing_mark(ch)


@pytest.mark.parametrize("ch", list("「『（(【") + ["“", "ー", "～", "あ", "a", " "])
def test_opening_brackets_letters_and_the_long_vowel_mark_do_not(ch):
    assert not is_closing_mark(ch)


@pytest.mark.parametrize("ch", ["・", "･", "'", '"', "＇", "＂", "’", "〜", "-", "‐", "_"])
def test_marks_inside_words_dashes_and_straight_quotes_do_not(ch):
    assert not is_closing_mark(ch)


def test_a_name_with_a_middle_dot_is_not_split_inside():
    pieces = [("ジョン", 0.0, 2.0), ("・", 2.0, 2.0), ("スミス", 2.0, 4.0), ("さん", 4.0, 5.0), ("です", 5.0, 7.0)]
    assert split_line("ジョン・スミスさんです", 0.0, 7.0, pieces, LineLimits(max_s=5.0)) == \
        [("ジョン・スミスさんです", 0.0, 7.0)]


def test_marks_only():
    assert is_closing_marks_only("、") and is_closing_marks_only("!?") and is_closing_marks_only(" 。 ")
    assert not is_closing_marks_only("「") and not is_closing_marks_only("す。") and not is_closing_marks_only("  ")
    assert not is_closing_marks_only("'") and not is_closing_marks_only('"') and not is_closing_marks_only("・")


def test_a_line_within_both_limits_comes_back_as_it_was_given():
    assert split_line(TEXT, 0.0, 6.0, PIECES, LineLimits(max_s=6.0, max_chars=12)) == [(TEXT, 0.0, 6.0)]


def test_a_line_over_the_time_limit_is_split_at_the_mark_nearest_its_middle_then_again():
    # middle 3.0 s: the mark after "は、" ends at 2.0 s, the one after "も。" at 4.0 s; the earlier wins a tie
    approx_lines(split_line(TEXT, 0.0, 6.0, PIECES, LineLimits(max_s=4.0)),
                 [("今日は、", 0.0, 2.0), ("明日も。それから", 2.0, 6.0)])
    approx_lines(split_line(TEXT, 0.0, 6.0, PIECES, LineLimits(max_s=3.0)),
                 [("今日は、", 0.0, 2.0), ("明日も。", 2.0, 4.0), ("それから", 4.0, 6.0)])


def test_a_line_over_the_character_limit_is_split_and_spaces_are_not_counted():
    approx_lines(split_line(TEXT, 0.0, 6.0, PIECES, LineLimits(max_chars=8)),
                 [("今日は、", 0.0, 2.0), ("明日も。それから", 2.0, 6.0)])
    spaced = [("OK, ", 0.0, 1.0), ("then", 1.0, 2.0)]
    assert split_line("OK, then", 0.0, 2.0, spaced, LineLimits(max_chars=7)) == [("OK, then", 0.0, 2.0)]


def test_a_part_with_no_mark_left_stays_over_the_limit():
    approx_lines(split_line(TEXT, 0.0, 6.0, PIECES, LineLimits(max_s=1.0)),
                 [("今日は、", 0.0, 2.0), ("明日も。", 2.0, 4.0), ("それから", 4.0, 6.0)])


def test_a_line_with_no_mark_inside_stays_whole():
    pieces = [("テスト", 0.0, 5.0), ("です", 5.0, 9.0), ("。", 9.0, 9.1)]  # the only mark ends the line
    assert split_line("テストです。", 0.0, 9.1, pieces, LineLimits(max_s=5.0)) == [("テストです。", 0.0, 9.1)]
    assert split_line("テスト", 0.0, 9.0, [], LineLimits(max_s=5.0)) == [("テスト", 0.0, 9.0)]


def test_a_mark_inside_a_piece_is_a_split_point_with_the_time_shared_out_by_characters():
    # one piece, no times inside it (plan 3.8 item 2 (iii)): 10 s over 10 characters (4 + 3 + 3)
    text = "あいう!えお!かきく"
    lines = split_line(text, 100.0, 110.0, [(text, 100.0, 110.0)], LineLimits(max_s=5.0))
    approx_lines(lines, [("あいう!", 100.0, 104.0), ("えお!", 104.0, 107.0), ("かきく", 107.0, 110.0)])


def test_a_run_of_marks_inside_a_piece_stays_together():
    text = "あいう!?えおか"  # 5 + 3 characters over 8 s
    lines = split_line(text, 0.0, 8.0, [(text, 0.0, 8.0)], LineLimits(max_s=5.0))
    approx_lines(lines, [("あいう!?", 0.0, 5.0), ("えおか", 5.0, 8.0)])


def test_an_opening_bracket_is_never_left_at_a_line_end():
    # "「" is no split point; "」と" is cut after its "」", its 1 s shared out by characters
    pieces = [("彼は", 0.0, 2.0), ("「", 2.0, 2.0), ("はい", 2.0, 4.0), ("」と", 4.0, 5.0), ("言った", 5.0, 7.0)]
    approx_lines(split_line("彼は「はい」と言った", 0.0, 7.0, pieces, LineLimits(max_s=4.6)),
                 [("彼は「はい」", 0.0, 4.5), ("と言った", 4.5, 7.0)])


def test_part_times_run_from_the_first_piece_to_the_last_and_never_backwards():
    pieces = [("あ、", 1.0, 3.0), ("い", 6.0, 5.5)]  # a backward end, as Cohere's estimated times can have
    approx_lines(split_line("あ、い", 1.0, 6.0, pieces, LineLimits(max_s=4.0)), [("あ、", 1.0, 3.0), ("い", 6.0, 6.0)])


def test_lines_from_segments_puts_every_segment_through_the_rule_and_skips_empty_ones():
    def word(w, s, e):
        return SimpleNamespace(word=w, start=s, end=e)

    segments = [SimpleNamespace(text=" " + TEXT, start=0.0, end=6.0, words=[word(*p) for p in PIECES]),
                SimpleNamespace(text="  ", start=6.0, end=7.0, words=[]),
                SimpleNamespace(text="短い。", start=7.0, end=8.0, words=[word("短い。", 7.0, 8.0)])]
    approx_lines(lines_from_segments(segments, LineLimits(max_s=4.0)),
                 [("今日は、", 0.0, 2.0), ("明日も。それから", 2.0, 6.0), ("短い。", 7.0, 8.0)])
    approx_lines(lines_from_segments(segments, OFF), [(TEXT, 0.0, 6.0), ("短い。", 7.0, 8.0)])
