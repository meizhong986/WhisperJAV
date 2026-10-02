"""Tests for whisperjav.bench.timing (start/end errors against a reference)."""

from pathlib import Path

import pytest

from whisperjav.bench.timing import compare_clip, match_errors, summarize
from whisperjav.bench.timing_cli import clip_id_of, find_hyp, main


def sub(start, end, text):
    return {"start": start, "end": end, "text": text}


REF = [
    sub(1.0, 3.0, "こんにちは"),
    sub(5.0, 7.0, "元気ですか"),
    sub(10.0, 12.0, "ありがとう"),
]


def test_errors_are_hyp_minus_ref():
    hyp = [sub(1.2, 3.5, "こんにちは"), sub(4.9, 6.8, "元気ですか")]
    errors = match_errors(REF, hyp)
    assert errors[(1.0, 3.0, "こんにちは")] == pytest.approx((0.2, 0.5))
    assert errors[(5.0, 7.0, "元気ですか")] == pytest.approx((-0.1, -0.2))
    assert (10.0, 12.0, "ありがとう") not in errors


def test_summarize_median_p90_and_signed_mean():
    errors = [(0.1 * i, -0.1 * i) for i in range(1, 11)]  # 0.1 .. 1.0
    m = summarize(errors)
    assert m.n_lines == 10
    assert m.start_abs_median == pytest.approx(0.55)
    assert m.start_abs_p90 == pytest.approx(0.9)
    assert m.start_signed_mean == pytest.approx(0.55)
    assert m.end_signed_mean == pytest.approx(-0.55)
    assert m.end_abs_p90 == pytest.approx(0.9)


def test_summarize_empty():
    m = summarize([])
    assert m.n_lines == 0 and m.start_abs_median == 0.0


def test_compare_uses_only_common_lines_and_counts_lost():
    base = [sub(1.5, 3.5, "こんにちは"), sub(5.5, 7.5, "元気ですか"), sub(10.5, 12.5, "ありがとう")]
    # candidate is perfect on line 1, drops line 3 (a "hard" line)
    cand = [sub(1.0, 3.0, "こんにちは"), sub(5.5, 7.5, "元気ですか")]
    c = compare_clip(REF, base, cand)
    assert c.n_common == 2
    assert c.n_lost == 1
    assert c.base.n_lines == c.cand.n_lines == 2
    assert c.base.start_abs_median == pytest.approx(0.5)
    assert c.cand.start_abs_median == pytest.approx(0.25)


def test_identical_runs_have_zero_difference():
    run = [sub(1.3, 3.1, "こんにちは"), sub(5.2, 7.4, "元気ですか")]
    c = compare_clip(REF, run, list(run))
    assert c.base.as_dict() == c.cand.as_dict()
    assert c.n_lost == 0


def test_clip_id_of():
    assert clip_id_of(Path("A.S01E03.Scene.3.ja.srt")) == "A.S01E03.Scene.3"
    assert clip_id_of(Path("B.Scene.5.Bubble.More.srt")) == "B.Scene.5.Bubble.More"


def _write_srt(path: Path, lines):
    blocks = []
    for i, (s, e, t) in enumerate(lines, 1):
        def ts(x):
            ms = int(round(x * 1000))
            return f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}"
        blocks.append(f"{i}\n{ts(s)} --> {ts(e)}\n{t}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def test_find_hyp_requires_exactly_one(tmp_path):
    _write_srt(tmp_path / "clip.ja.whisperjav.srt", [(1, 2, "a")])
    assert find_hyp(tmp_path, "clip").name == "clip.ja.whisperjav.srt"
    _write_srt(tmp_path / "clip.ja.pass1.srt", [(1, 2, "a")])
    with pytest.raises(SystemExit):
        find_hyp(tmp_path, "clip")


def test_cli_end_to_end(tmp_path, capsys):
    ref_dir, base_dir, cand_dir = (tmp_path / d for d in ("ref", "base", "cand"))
    for d in (ref_dir, base_dir, cand_dir):
        d.mkdir()
    _write_srt(ref_dir / "clip.ja.srt", [(r["start"], r["end"], r["text"]) for r in REF])
    _write_srt(base_dir / "clip.ja.whisperjav.srt", [(1.5, 3.5, "こんにちは"), (5.5, 7.5, "元気ですか")])
    _write_srt(cand_dir / "clip.ja.whisperjav.srt", [(1.1, 3.1, "こんにちは"), (5.1, 7.1, "元気ですか")])
    out = tmp_path / "t.json"
    assert main(["--ref-dir", str(ref_dir), "--base", str(base_dir), "--cand", str(cand_dir), "-o", str(out)]) == 0
    import json
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["totals"]["n_common"] == 2
    assert data["pooled_base"]["start_abs_median"] == pytest.approx(0.5)
    assert data["pooled_cand"]["start_abs_median"] == pytest.approx(0.1)
