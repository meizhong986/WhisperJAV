"""Post-processing of the CrispASR pipelines (whisperjav/crispasr/cleanup.py; 1.10.1, owner W418 5, W420-W422).

Pinned here: the stages run in fidelity's order and with fidelity's pieces (duplicate merge, hallucination filter,
repetition cleaner), then the three earlier filters; what is left out on purpose (the speed rules, the 200-character
cut, any timing change; W203, W411); the raw copy and the artifacts file with every removal and change (R-RAW), in
fidelity's names and format (W421 i5). The hallucination filter uses its own built-in list here, never the network.

Run with: python -m pytest tests/test_crispasr_cleanup.py -q
"""
from pathlib import Path

import pysrt
import pytest

from whisperjav.crispasr import cleanup

CLOSING = "ご視聴ありがとうございました"


@pytest.fixture(autouse=True)
def builtin_lists(monkeypatch):
    from whisperjav.modules.hallucination_remover import HallucinationRemover

    def no_lists(remover):
        raise RuntimeError("no lists in tests")

    monkeypatch.setattr(HallucinationRemover, "_load_patterns", no_lists)


def write_srt(path: Path, entries):
    subs = [pysrt.SubRipItem(index=n, start=pysrt.SubRipTime(milliseconds=int(s * 1000)),
                             end=pysrt.SubRipTime(milliseconds=int(e * 1000)), text=t)
            for n, (t, s, e) in enumerate(entries, 1)]
    pysrt.SubRipFile(subs).save(str(path), encoding="utf-8")


def read_srt(path: Path):
    return [(s.text, s.start.ordinal / 1000, s.end.ordinal / 1000) for s in pysrt.open(str(path), encoding="utf-8")]


def run(tmp_path, entries, stages=cleanup.SRT_STAGES, text_changes=()):
    stitched = tmp_path / "film_stitched.srt"
    write_srt(stitched, entries)
    stats, count = cleanup.SrtCleanup(stages).run(stitched, tmp_path / "raw_subs", "film.kotoba-crispasr_stitched",
                                                  len(entries), text_changes)
    return stitched, stats, count


def test_the_stage_lists():
    assert cleanup.SRT_STAGES == ("duplicates", "hallucination", "repetition", "nonverbal", "nonlinguistic",
                                  "scene_overlap")
    assert cleanup.EARLIER_STAGES == ("nonverbal", "nonlinguistic", "scene_overlap")
    with pytest.raises(ValueError, match="unknown clean-up stage"):
        cleanup.SrtCleanup(("cps",))
    assert cleanup.file_stem("film", "cohere-crispasr") == "film.cohere-crispasr_stitched"


def test_a_hallucination_is_removed_and_listed(tmp_path):
    stitched, stats, count = run(tmp_path, [(CLOSING, 1.0, 2.0), ("本番だよ", 3.0, 4.0)])
    assert read_srt(stitched) == [("本番だよ", 3.0, 4.0)] and count == 1
    assert stats["hallucinations_removed"] == 1
    art = (tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.artifacts.srt").read_text(encoding="utf-8")
    assert f"[REMOVED - Hallucination]\nOriginal: {CLOSING}" in art
    assert "Original subtitles: 2" in art and "Final subtitles: 1" in art


def test_the_raw_copy_is_written_first_and_unchanged(tmp_path):
    entries = [(CLOSING, 1.0, 2.0), ("本番だよ", 3.0, 4.0)]
    run(tmp_path, entries)
    assert read_srt(tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.original.srt") == entries


def test_three_identical_lines_close_together_become_one(tmp_path):
    entries = [("もう一回", 1.0, 1.5), ("もう一回", 1.8, 2.3), ("もう一回", 2.6, 3.1), ("監督です", 5.0, 6.0)]
    stitched, stats, _count = run(tmp_path, entries)
    assert read_srt(stitched) == [("もう一回", 1.0, 3.1), ("監督です", 5.0, 6.0)]
    assert stats["duplicates_merged"] == 1
    art = (tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.artifacts.srt").read_text(encoding="utf-8")
    assert "Sequential Duplicates Merged" in art


def test_two_identical_lines_or_a_wide_gap_stay(tmp_path):
    entries = [("もう一回", 1.0, 1.5), ("もう一回", 1.8, 2.3), ("監督です", 3.0, 4.0),
               ("いくよ", 5.0, 5.5), ("いくよ", 6.5, 7.0), ("いくよ", 8.0, 8.5)]   # gaps of 1 s: not merged
    stitched, _stats, _count = run(tmp_path, entries)
    assert read_srt(stitched) == entries


def test_a_repeated_phrase_is_shortened_not_dropped(tmp_path):
    flood = "あああああああああ本当に"
    stitched, stats, count = run(tmp_path, [(flood, 1.0, 3.0)])
    (text, _s, _e), = read_srt(stitched)
    assert count == 1 and text != flood and text.endswith("本当に") and len(text) < len(flood)
    assert stats["lines_changed"] == 1


def test_no_cut_at_200_characters(tmp_path):
    # fidelity cuts a line over 200 characters; the CrispASR clean-up does not (W203)
    long = "".join(chr(0x4E00 + 7 * n) for n in range(230))   # 230 different kanji: no repeat to clean
    assert len(long) > 200
    stitched, _stats, _count = run(tmp_path, [(long, 1.0, 20.0)])
    assert read_srt(stitched)[0][0] == long


def test_no_speed_rules_and_no_timing_change(tmp_path):
    # fidelity drops a line over 30 characters per second and a line of 4 characters or fewer under 1 per second,
    # and moves starts; the CrispASR clean-up does neither (W203, W411)
    entries = [("監督", 1.0, 7.0), ("今日はとても良い天気ですね本当にそう思いますよ私は本当に", 8.0, 8.5)]
    stitched, _stats, _count = run(tmp_path, entries)
    assert read_srt(stitched) == entries


def test_the_earlier_filters_still_run_and_their_drops_and_moved_starts_are_listed(tmp_path):
    # found by the review (2026-11-01, F1): the three filters' drops and moved starts reach the artifacts file too
    entries = [("あ。", 1.0, 1.3), ("本番だよ", 2.0, 6.0), ("監督です", 3.0, 4.0), ("いくよ", 5.5, 7.0)]
    stitched, stats, _count = run(tmp_path, entries)
    assert stats["nonverbal_dropped"] == 1 and stats["overlap_nested_dropped"] == 1 and stats["overlap_shifted"] == 1
    assert [t for t, _s, _e in read_srt(stitched)] == ["本番だよ", "いくよ"]
    art = (tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.artifacts.srt").read_text(encoding="utf-8")
    assert "[REMOVED - Nonverbal Line]\nOriginal: あ。" in art
    assert "[REMOVED - Scene Overlap]\nOriginal: 監督です" in art
    assert "[MODIFIED - Scene Overlap Start Moved]\nOriginal: いくよ\nModified: いくよ  (start 00:00:05,500 -> " in art
    assert "Total artifact entries: 3" in art and "Final subtitles: 2" in art


def test_the_earlier_stages_alone_leave_hallucinations_and_write_no_files(tmp_path):
    # MOSS-CrispASR's clean-up (W421 i3): as in 1.10.0, no raw copy and no artifacts file
    stitched, stats, _count = run(tmp_path, [(CLOSING, 1.0, 2.0), ("あ。", 3.0, 3.3)], stages=cleanup.EARLIER_STAGES)
    assert read_srt(stitched) == [(CLOSING, 1.0, 2.0)]
    assert not (tmp_path / "raw_subs").exists() and "artifacts_srt" not in stats


def test_an_earlier_runs_trace_is_removed_first(tmp_path):
    # found by the review (F3): the two files always describe this run
    run(tmp_path, [(CLOSING, 1.0, 2.0), ("本番だよ", 3.0, 4.0)])
    run(tmp_path, [("本番だよ", 3.0, 4.0)])                       # nothing removed this time
    raw = tmp_path / "raw_subs"
    assert read_srt(raw / "film.kotoba-crispasr_stitched.original.srt") == [("本番だよ", 3.0, 4.0)]
    assert not (raw / "film.kotoba-crispasr_stitched.artifacts.srt").exists()


def test_a_removed_closing_phrase_is_counted_as_a_hallucination_in_the_summary(tmp_path):
    # found by the review (F5): the filter's category "closing_phrase" was not counted by fidelity's summary
    run(tmp_path, [(CLOSING, 1.0, 2.0), ("本番だよ", 3.0, 4.0)])
    art = (tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.artifacts.srt").read_text(encoding="utf-8")
    assert "Hallucinations modified/removed: 1" in art


def test_the_names_with_the_pass():
    assert cleanup.file_stem("film", "cohere-crispasr", "pass1") == "film.pass1.cohere-crispasr_stitched"


def test_text_cleaner_changes_are_listed_too(tmp_path):
    change = cleanup.TextChange(1.0, 4.0, "行け行け行け行け行け。", "行け行け。")
    run(tmp_path, [("行け行け。", 1.2, 4.0)], text_changes=[change])
    art = (tmp_path / "raw_subs" / "film.kotoba-crispasr_stitched.artifacts.srt").read_text(encoding="utf-8")
    assert "[MODIFIED - Text Cleaner]\nOriginal: 行け行け行け行け行け。\nModified: 行け行け。" in art
    assert "00:00:01,000 --> 00:00:04,000" in art


def test_the_text_cleaner_is_chronosjavs():
    cleaner = cleanup.TextCleaner()
    assert cleaner.clean("行け行け行け行け行け。") == "行け行け。"
    assert cleaner.clean("今日は撮影です。") == "今日は撮影です。"


def test_nothing_to_do_on_an_empty_file(tmp_path):
    stitched = tmp_path / "film_stitched.srt"
    stitched.write_text("", encoding="utf-8")
    assert cleanup.SrtCleanup(cleanup.SRT_STAGES).run(stitched, tmp_path / "raw_subs", "x", 0) == ({}, 0)
    assert not (tmp_path / "raw_subs").exists()
