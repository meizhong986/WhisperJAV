"""Hand-built cases for subeval: each test states a situation, the answer a person would give, and checks that
subeval gives it. Run: python -m pytest subeval/tests/test_subeval.py"""
import json
import random

import pytest

from subeval.align import _pair_table, align, levenshtein
from subeval.cli import clip_id, main, pair_folder
from subeval.compare import compare
from subeval.evaluate import evaluate_clip, summarize
from subeval.normalize import normalizer, unbalanced_brackets
from subeval.srt import Cue, parse_srt_text


def cues(*items):
    """cues((start, end, text), ...)"""
    return [Cue(s, e, t, k) for k, (s, e, t) in enumerate(items, 1)]


def ev(ref, test, **kw):
    return evaluate_clip(cues(*ref), cues(*test), "c", **kw)


# ---------------------------------------------------------------- reading and normalising

def test_srt_bom_crlf_dot_millis_no_index_and_damaged_blocks_reported():
    text = ("﻿1\r\n00:00:01,000 --> 00:00:02,500\r\nこんにちは\r\n二行目\r\n\r\n"
            "00:00:03.000 --> 00:00:04.000\r\nインデックスなし\r\n\r\n"
            "3\r\nこれは壊れたブロック\r\n\r\n"
            "4\r\n00:00:06,000 --> 00:00:05,000\r\n逆の時間\r\n")
    f = parse_srt_text(text)
    assert [(c.start, c.end, c.text) for c in f.cues] == [(1.0, 2.5, "こんにちは\n二行目"), (3.0, 4.0, "インデックスなし")]
    assert len(f.problems) == 2 and "no timing line" in f.problems[0] and "before start" in f.problems[1]


def test_normalisation_profile_ja_v1():
    n = normalizer("ja-v1")
    assert n("（村西）ちょっと待って！ ＡＢＣ１２３…") == "ちょっと待ってABC123"
    assert n("(笑)ハァ…") == "ハァ"
    assert n("♪〜") == ""
    assert unbalanced_brackets("の音）") and not unbalanced_brackets("（音）あ")


def test_srt_missing_blank_line_is_split_and_reported_and_file_numbers_kept():
    text = ("5\n00:00:01,000 --> 00:00:02,000\nあ\n6\n00:00:03,000 --> 00:00:04,000\nい\n\n"
            "7\nこわれた\n\n8\n00:00:05,000 --> 00:00:06,000\nう\n")
    f = parse_srt_text(text)
    assert [(c.number, c.text) for c in f.cues] == [(5, "あ"), (6, "い"), (8, "う")]
    assert any("no blank line" in p for p in f.problems) and any("no timing line" in p for p in f.problems)


def test_clip_id_rule():
    from pathlib import Path
    assert clip_id(Path("A.B.ja.srt")) == "A.B" and clip_id(Path("A.B.srt")) == "A.B"


# ---------------------------------------------------------------- alignment

def test_plain_levenshtein_matches_a_brute_force_reference():
    def slow(a, b):
        if not a or not b:
            return len(a) + len(b)
        return min(slow(a[1:], b) + 1, slow(a, b[1:]) + 1, slow(a[1:], b[1:]) + (a[0] != b[0]))
    rnd = random.Random(1)
    for _ in range(200):
        a = "".join(rnd.choice("あいう") for _ in range(rnd.randint(0, 6)))
        b = "".join(rnd.choice("あいう") for _ in range(rnd.randint(0, 6)))
        assert levenshtein(a, b) == slow(a, b)


def test_time_aware_alignment_with_huge_window_has_the_plain_edit_count():
    rnd = random.Random(2)
    for _ in range(100):
        a = "".join(rnd.choice("あいうえ") for _ in range(rnd.randint(1, 12)))
        b = "".join(rnd.choice("あいうえ") for _ in range(rnd.randint(1, 12)))
        ra = [(0.0, 1.0)] * 1
        _, edits = align(a, [0] * len(a), ra, b, [0] * len(b), [(5.0, 6.0)], window=1e6)
        assert edits == levenshtein(a, b)
        _, edits_m = align(a, [0] * len(a), ra, b, [0] * len(b), [(5.0, 6.0)], window=1e6, direction="mirror")
        assert edits_m == edits


def _all_alignments(n, m, ok):
    def rec(i, j):
        if i == n and j == m:
            yield []
            return
        if i < n and j < m and ok(i, j):
            for rest in rec(i + 1, j + 1):
                yield [("P", i, j)] + rest
        if i < n:
            for rest in rec(i + 1, j):
                yield [("missing", i, None)] + rest
        if j < m:
            for rest in rec(i, j + 1):
                yield [("extra", None, j)] + rest
    yield from rec(0, 0)


def test_alignment_is_the_exact_lexicographic_optimum_over_every_alignment():
    # several lines per stream, so the gap and midpoint tie levels are exercised (after the review's p1 probe)
    rnd = random.Random(1)
    checked = 0
    for _ in range(1500):
        def lines(k):
            out, t = [], 0.0
            for _ in range(k):
                t += rnd.choice([0, 0.3, 0.7, 1.5])
                d = rnd.choice([0.4, 1.0, 2.0])
                out.append((round(t, 3), round(t + d, 3), "".join(rnd.choice("あいう") for _ in range(rnd.randint(1, 3)))))
                t += d
            return out
        rl, tl = lines(rnd.randint(1, 3)), lines(rnd.randint(1, 3))
        rc = "".join(x[2] for x in rl)
        tc = "".join(x[2] for x in tl)
        if len(rc) > 6 or len(tc) > 6:
            continue
        ro = [k for k, x in enumerate(rl) for _ in x[2]]
        to = [k for k, x in enumerate(tl) for _ in x[2]]
        rs, ts = [x[:2] for x in rl], [x[:2] for x in tl]
        w = rnd.choice([0.5, 1.0])
        table = _pair_table(rs, ts, w)

        def key(ops):
            e = g = md = 0
            for kind, i, j in ops:
                if kind in ("P", "hit", "wrong"):
                    c = table[ro[i]][to[j]]
                    g, md, e = g + c[0], md + c[1], e + (rc[i] != tc[j])
                else:
                    e += 1
            return e, g, md
        best = min(key(a) for a in _all_alignments(len(rc), len(tc), lambda i, j: table[ro[i]][to[j]] is not None))
        for direction in ("forward", "mirror"):
            ops, edits = align(rc, ro, rs, tc, to, ts, w, direction)
            assert key(ops) == best and edits == best[0]
        checked += 1
    assert checked > 500


def test_window_forbids_pairing_far_apart_lines():
    # the same word said at 10 s in the reference but shown at 30 s in the test: missing + extra, not a hit
    r = ev([(10, 11, "ありがとう")], [(30, 31, "ありがとう")])
    assert r.edits == 10 and r.plain_distance == 0
    assert r.ref[0].missed and r.test[0].hallucination


def test_repeated_word_pairs_with_the_occurrence_at_the_same_time():
    # reference: うん at 10 s only; test: うん at 2 s (hallucinated) and at 10 s
    r = ev([(10, 11, "うん")], [(2, 3, "うん"), (10, 11, "うん")])
    assert r.ref[0].home == 1 and not r.ref[0].missed
    assert r.test[0].hallucination and not r.test[1].hallucination


def test_tie_break_prefers_the_overlapping_line_over_a_near_one():
    # both test lines are within the window; the alignment must pick the one that overlaps the reference line
    r = ev([(10, 12, "はい")], [(12.5, 13, "はい"), (10, 12, "はい")])
    assert r.test[r.ref[0].home].start == 10 and r.test[r.ref[0].home].number == 2


# ---------------------------------------------------------------- indicators

def test_identical_subtitles_score_perfectly():
    ref = [(1, 2, "こんにちは"), (3, 4.5, "元気ですか")]
    r = ev(ref, ref)
    s = summarize([r])
    assert s["cer"] == 0 and s["cer_plain"] == 0 and s["missed_lines"] == 0 and s["hallucination_lines"] == 0
    assert s["start_ok"] == 2 and s["end_ok"] == 2 and s["start_abs_median"] == 0 and s["end_abs_median"] == 0
    assert s["ambiguous_ref_lines"] == 0


def test_missed_line():
    r = ev([(1, 2, "こんにちは"), (3, 4, "さようなら")], [(1, 2, "こんにちは")])
    assert [x.missed for x in r.ref] == [False, True]
    assert summarize([r])["missed_lines"] == 1


def test_mishearing_is_a_missed_line_and_a_misheard_test_line_not_a_hallucination():
    r = ev([(1, 2, "どうも")], [(1, 2, "さま")])
    assert r.ref[0].hits == 0 and r.ref[0].wrong == 2 and r.ref[0].missed and r.ref[0].missed_kind == "wrong only"
    assert r.test[0].misheard and not r.test[0].hallucination
    s = summarize([r])
    assert s["missed_wrong_only"] == 1 and s["missed_nothing"] == 0 and s["hallucination_lines"] == 0
    assert s["misheard_test_lines"] == 1


def test_one_shared_character_decides_a_line_and_is_counted():
    r = ev([(1, 3, "そうなの")], [(1, 3, "ご視聴ありがとうございました")])
    assert not r.ref[0].missed and r.ref[0].hits == 1
    assert summarize([r])["single_hit_ref_lines"] == 1


def test_suspected_hallucination_line():
    r = ev([(1, 2, "こんにちは")], [(1, 2, "こんにちは"), (20, 22, "ご視聴ありがとうございました")])
    assert [t.hallucination for t in r.test] == [False, True]
    assert summarize([r])["hallucination_chars"] == len("ご視聴ありがとうございました")


def test_opening_cut_off():
    r = ev([(1, 3, "ねえちょっと待って")], [(1.4, 3, "ちょっと待って")])
    x = r.ref[0]
    assert x.start_class == "lost" and x.lead_lost == 2 and x.end_class == "ok"
    assert x.start_err == pytest.approx(0.4)


def test_ending_cut_off():
    r = ev([(1, 3, "ちょっと待ってよ")], [(1, 2.5, "ちょっと待って")])
    x = r.ref[0]
    assert x.end_class == "lost" and x.tail_lost == 1 and x.start_class == "ok"
    assert x.end_err == pytest.approx(-0.5)


def test_wrong_first_character():
    r = ev([(1, 2, "かえります")], [(1, 2, "がえります")])
    assert r.ref[0].start_class == "wrong" and r.ref[0].end_class == "ok"


def test_our_boundary_cuts_through_a_reference_line():
    # reference line split over two of our lines: the opening sits in another line than most of the text
    r = ev([(1, 4, "あのね今日はとても寒いね")], [(1, 1.8, "あのね"), (1.8, 4, "今日はとても寒いね")])
    x = r.ref[0]
    assert x.home == 1 and x.start_class == "misplaced" and x.end_class == "ok"
    # the reference line's edges are timed where they are: its opening at the first test line, its end at the
    # second; the two test-line edges inside the reference line (at 1.8 s) are not timed
    assert x.start_err == pytest.approx(0.0) and x.end_err == pytest.approx(0.0)
    assert summarize([r])["start_measured"] == 1 and summarize([r])["end_measured"] == 1


def test_one_test_line_holding_two_reference_lines_measures_each_boundary_once():
    ref = [(1, 2, "おはよう"), (2.2, 3.5, "元気ですか")]
    r = ev(ref, [(0.9, 3.8, "おはよう元気ですか")])
    a, b = r.ref
    assert a.start_err == pytest.approx(-0.1) and a.end_err is None
    assert b.start_err is None and b.end_err == pytest.approx(0.3)
    assert a.start_class == a.end_class == b.start_class == b.end_class == "ok"


def test_equal_cost_alternatives_are_marked_uncertain_not_lost():
    # reference ああ, test あ: whether the first or the second あ was dropped is undecidable
    r = ev([(1, 2, "ああそう")], [(1, 2, "あそう")])
    x = r.ref[0]
    assert x.start_class == "uncertain" and x.lead_lost == 0
    assert 0 in r.ambiguity["start"]


def test_empty_lines_after_normalisation_are_counted_not_scored():
    r = ev([(1, 2, "♪〜"), (3, 4, "はい")], [(3, 4, "はい"), (5, 6, "…")])
    s = summarize([r])
    assert s["ref_lines_empty"] == 1 and s["test_lines_empty"] == 1 and s["missed_lines"] == 0
    assert s["hallucination_lines"] == 0


def test_cer_parts_add_up():
    r = ev([(1, 2, "今日は晴れです"), (3, 4, "明日は雨")], [(1, 2, "今日は晴れでした"), (3, 4, "明日")])
    s = summarize([r])
    assert s["wrong_rate"] + s["missing_rate"] + s["extra_rate"] == pytest.approx(s["cer"])
    assert s["cer"] >= s["cer_plain"]


# ---------------------------------------------------------------- comparing two runs

def test_compare_identical_runs_and_a_later_end():
    ref = [(1, 2, "こんにちは"), (3, 4, "さようなら")]
    base = [evaluate_clip(cues(*ref), cues(*ref), "c")]
    same = compare(base, [evaluate_clip(cues(*ref), cues(*ref), "c")])
    assert same["d_edits"] == 0 and same["end_clips_better"] == 0 and same["end_clips_worse"] == 0
    late = compare(base, [evaluate_clip(cues(*ref), cues((1, 2.6, "こんにちは"), (3, 4.6, "さようなら")), "c")])
    assert late["end_med_cand"] == pytest.approx(0.6) and late["end_clips_worse"] == 1 and late["d_edits"] == 0


def test_compare_refuses_different_clip_sets():
    a = [evaluate_clip(cues((1, 2, "あ")), cues((1, 2, "あ")), "c1")]
    b = [evaluate_clip(cues((1, 2, "あ")), cues((1, 2, "あ")), "c2")]
    with pytest.raises(ValueError):
        compare(a, b)


def test_compare_refuses_different_references():
    a = [evaluate_clip(cues((1, 2, "あ")), cues((1, 2, "あ")), "c")]
    b = [evaluate_clip(cues((1, 2, "い")), cues((1, 2, "い")), "c")]
    with pytest.raises(ValueError):
        compare(a, b)


# ---------------------------------------------------------------- command line

def test_cli_score_and_compare_folders(tmp_path):
    ref, t1, t2 = tmp_path / "ref", tmp_path / "t1", tmp_path / "t2"
    for d in (ref, t1, t2):
        d.mkdir()
    (ref / "clip.ja.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n", encoding="utf-8")
    (t1 / "clip.ja.pass1.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nこんにちは\n", encoding="utf-8")
    (t2 / "clip.ja.pass1.srt").write_text("1\n00:00:01,000 --> 00:00:02,500\nこんにちわ\n", encoding="utf-8")
    assert pair_folder(ref, t1)[0][0] == "clip"
    assert main(["score", "--ref-dir", str(ref), "--test-dir", str(t2), "--out", str(tmp_path / "o")]) == 0
    md = (tmp_path / "o" / "report.md").read_text(encoding="utf-8")
    assert "## Definitions" in md and "Missed lines" in md and "Suspected hallucination lines" in md
    data = json.loads((tmp_path / "o" / "result.json").read_text(encoding="utf-8"))
    assert data["summary"]["edits"] == 1
    assert "R1 [" in (tmp_path / "o" / "pairs.txt").read_text(encoding="utf-8")
    assert main(["compare", "--ref-dir", str(ref), "--base-dir", str(t1), "--cand-dir", str(t2),
                 "--out", str(tmp_path / "c")]) == 0
    cmp = json.loads((tmp_path / "c" / "compare.json").read_text(encoding="utf-8"))
    assert cmp["d_edits"] == 1 and cmp["end_clips_worse"] == 1


def test_cli_help_exits_zero():
    for argv in (["--help"], ["score", "--help"], ["compare", "--help"]):
        with pytest.raises(SystemExit) as e:
            main(argv)
        assert e.value.code == 0
