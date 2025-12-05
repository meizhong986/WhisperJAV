#!/usr/bin/env python3
"""Tests for the ensemble merge engine covering all strategies and edge cases."""

from pathlib import Path
import random
import textwrap

from whisperjav.ensemble.merge import MergeEngine, Subtitle


def _sub(start: float, end: float, text: str) -> Subtitle:
    return Subtitle(index=0, start_time=start, end_time=end, text=text)


def test_full_merge_includes_all_and_sorts_by_start():
    engine = MergeEngine()
    subs1 = [_sub(4.0, 5.0, "late"), _sub(0.0, 1.0, "early")]
    subs2 = [_sub(2.0, 3.0, "middle")]

    merged = engine._merge_full(subs1, subs2)

    assert [sub.text for sub in merged] == ["early", "middle", "late"]


def test_pass1_primary_skips_overlapping_secondary_segments():
    engine = MergeEngine()
    primary = [_sub(0.0, 5.0, "core"), _sub(10.0, 12.0, "ending")]
    secondary = [_sub(2.0, 4.0, "overlap"), _sub(6.0, 9.0, "gap fill"), _sub(10.5, 11.0, "tiny overlap")]

    merged = engine._merge_pass1_primary(primary, secondary)

    texts = [sub.text for sub in merged]
    assert "gap fill" in texts  # fills the real gap
    assert "overlap" not in texts
    assert "tiny overlap" not in texts


def test_pass1_overlap_allows_small_conflict_under_threshold():
    engine = MergeEngine()
    primary = [_sub(0.0, 10.0, "long window")]
    # overlap duration 2s, threshold = 10 * 0.3 = 3 -> secondary allowed
    secondary = [_sub(8.0, 12.0, "tail detail")]

    merged = engine._merge_pass1_overlap(primary, secondary)
    assert any(sub.text == "tail detail" for sub in merged)


def test_pass2_primary_respects_secondary_priority():
    engine = MergeEngine()
    subs1 = [_sub(0.0, 2.0, "p1"), _sub(5.0, 6.0, "p1 gap")]
    subs2 = [_sub(0.0, 2.0, "p2"), _sub(2.5, 4.0, "p2 unique")]

    merged = engine._merge_pass2_primary(subs1, subs2)
    texts = [sub.text for sub in merged]
    assert "p2" in texts and "p2 unique" in texts
    assert "p1" not in texts  # overlapping primary dropped


def test_smart_merge_prefers_precise_timing_over_text_length():
    engine = MergeEngine()

    subs1 = [_sub(0.0, 2.0, "short but accurate")]
    subs2 = [_sub(0.0, 8.0, "very long line that previously won due to text length")]

    merged = engine._merge_smart(subs1, subs2)

    assert len(merged) == 1
    assert merged[0].text == "short but accurate"
    assert merged[0].start_time == 0.0
    assert merged[0].end_time == 2.0


def test_smart_merge_prefers_segment_with_higher_coverage():
    engine = MergeEngine()

    subs1 = [_sub(0.0, 4.0, "pass1 wide window")]
    subs2 = [_sub(1.0, 3.0, "pass2 precise window")]

    merged = engine._merge_smart(subs1, subs2)

    assert len(merged) == 1
    assert merged[0].text == "pass2 precise window"
    assert merged[0].start_time == 1.0
    assert merged[0].end_time == 3.0


def test_smart_merge_tie_prefers_shorter_duration_then_earliest_start():
    engine = MergeEngine()

    subs1 = [_sub(0.0, 4.0, "compact window")]
    # Slightly longer duration but nearly identical coverage (diff < 5%)
    subs2 = [_sub(0.0, 4.2, "stretched window")]

    merged = engine._merge_smart(subs1, subs2)

    assert len(merged) == 1
    assert merged[0].text == "compact window"
    assert merged[0].end_time == 4.0


def test_smart_merge_keeps_unmatched_pass2_segments():
    engine = MergeEngine()

    subs1 = [_sub(0.0, 2.0, "shared"), _sub(3.0, 4.0, "unique1")]
    subs2 = [_sub(0.0, 2.0, "shared_other"), _sub(6.0, 7.0, "unique2")]

    merged = engine._merge_smart(subs1, subs2)

    assert len(merged) == 3
    assert merged[0].text == "shared"
    assert merged[1].text == "unique1"
    assert merged[2].text == "unique2"
    assert merged[2].start_time == 6.0
    assert merged[2].end_time == 7.0


def test_smart_merge_preserves_monotonic_timings():
    engine = MergeEngine()
    subs1 = [_sub(0.0, 1.0, "a"), _sub(2.0, 3.0, "b")]
    subs2 = [_sub(0.5, 1.5, "a2"), _sub(2.1, 2.9, "b2")]

    merged = engine._merge_smart(subs1, subs2)
    assert merged[0].end_time <= merged[1].start_time


def test_merge_method_writes_sorted_srt_output(tmp_path):
    engine = MergeEngine()
    srt1 = tmp_path / "pass1.srt"
    srt2 = tmp_path / "pass2.srt"
    out = tmp_path / "merged.srt"

    srt1.write_text(textwrap.dedent(
        """
        1
        00:00:00,000 --> 00:00:02,000
        hello

        2
        00:00:03,000 --> 00:00:04,000
        world
        """
    ).strip() + "\n", encoding="utf-8")

    srt2.write_text(textwrap.dedent(
        """
        1
        00:00:01,000 --> 00:00:03,500
        overlap

        2
        00:00:05,000 --> 00:00:06,000
        tail
        """
    ).strip() + "\n", encoding="utf-8")

    stats = engine.merge(srt1, srt2, out, strategy="smart_merge")

    assert stats["pass1_count"] == 2
    assert stats["pass2_count"] == 2
    assert stats["merged_count"] == 3

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "1"
    assert "tail" in lines[-1]


def _generate_random_subtitles(prefix: str, count: int, rng: random.Random) -> list[Subtitle]:
    subs = []
    for idx in range(count):
        start = rng.uniform(0, 60)
        duration = rng.uniform(0.5, 8.0)
        end = start + duration
        subs.append(_sub(round(start, 3), round(end, 3), f"{prefix}_{idx}"))
    return sorted(subs, key=lambda s: s.start_time)


def _overlap_duration(a: Subtitle, b: Subtitle) -> float:
    start = max(a.start_time, b.start_time)
    end = min(a.end_time, b.end_time)
    if end <= start:
        return 0.0
    return end - start


def test_pass1_primary_randomized_respects_overlaps():
    engine = MergeEngine()
    rng = random.Random(1337)

    for _ in range(30):
        primary = _generate_random_subtitles("p", 6, rng)
        secondary = _generate_random_subtitles("s", 6, rng)

        merged = engine._merge_pass1_primary(primary, secondary)
        merged_texts = {sub.text for sub in merged}

        for sec in secondary:
            if sec.text in merged_texts:
                assert all(
                    _overlap_duration(sec, pri) == 0.0 for pri in primary
                ), f"Secondary {sec.text} overlaps despite primary priority"


def test_pass1_overlap_randomized_allows_small_conflicts_only():
    engine = MergeEngine()
    rng = random.Random(4242)
    threshold = engine.OVERLAP_THRESHOLD

    for _ in range(30):
        primary = _generate_random_subtitles("p", 5, rng)
        secondary = _generate_random_subtitles("s", 5, rng)

        merged = engine._merge_pass1_overlap(primary, secondary)
        merged_texts = {sub.text for sub in merged}

        for sec in secondary:
            if sec.text in merged_texts:
                for pri in primary:
                    overlap = _overlap_duration(sec, pri)
                    if overlap == 0.0:
                        continue
                    assert overlap <= pri.duration * threshold + 1e-6, (
                        f"Secondary {sec.text} exceeded overlap threshold"
                    )


def test_pass2_primary_randomized_symmetry():
    engine = MergeEngine()
    rng = random.Random(5150)

    for _ in range(30):
        subs1 = _generate_random_subtitles("p1", 5, rng)
        subs2 = _generate_random_subtitles("p2", 5, rng)

        merged = engine._merge_pass2_primary(subs1, subs2)
        merged_texts = {sub.text for sub in merged}

        for pri in subs1:
            if pri.text in merged_texts:
                # Means this primary segment filled a gap in pass2; ensure no pass2 overlap
                assert all(
                    _overlap_duration(pri, sec) == 0.0 for sec in subs2
                )
