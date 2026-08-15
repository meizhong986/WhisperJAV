"""Tests for the v1.9.0 qwen scene-overlap timestamp resolver.

Cases mirror real ChronosJAV/anime-whisper output where semantic scene
detection's ±0.35s buffer produces ~0.7s scene overlaps: partial tail overlaps
and nested-duplicate onset fragments at scene boundaries.
"""

from pathlib import Path

import pysrt
import pytest

from whisperjav.modules.subtitle_pipeline.cleaners.scene_overlap_resolver import (
    SceneOverlapResolver,
)


def _ts(ms: int) -> str:
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _write_srt(path: Path, entries) -> None:
    """entries: list of (start_ms, end_ms, text)."""
    blocks = []
    for i, (start, end, text) in enumerate(entries, start=1):
        blocks.append(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n")
    path.write_text("\n".join(blocks), encoding="utf-8")


def _resolve(tmp_path, entries):
    p = tmp_path / "stitched.srt"
    _write_srt(p, entries)
    stats = SceneOverlapResolver().resolve_srt_file(p)
    subs = pysrt.open(str(p), encoding="utf-8")
    return stats, subs


class TestPartialOverlap:
    def test_tail_overlap_is_clipped(self, tmp_path):
        # #1/#2 real case: 0.700s tail overlap -> clip #1.end to #2.start - 1ms.
        stats, subs = _resolve(tmp_path, [
            (25_098, 28_062, "ごめん、本当にごめん。"),
            (27_362, 31_362, "…ごめん、大丈夫です…"),
        ])
        assert stats["shifted_starts"] == 1
        assert stats["dropped_nested"] == 0
        assert stats["final_count"] == 2
        # #1's end is KEPT (accurate scene-A tail); #2's start is pushed to just
        # after it. The earlier sub keeps its full duration.
        assert subs[0].end.ordinal == 28_062            # #1 end untouched
        assert subs[1].start.ordinal == 28_062 + 1      # #2 start pushed past #1
        assert subs[1].end.ordinal == 31_362            # #2 end untouched
        # No residual overlap anywhere.
        assert subs[0].end.ordinal < subs[1].start.ordinal

    def test_no_overlap_is_noop(self, tmp_path):
        stats, subs = _resolve(tmp_path, [
            (1_000, 4_000, "A"),
            (4_040, 8_040, "B"),
        ])
        assert stats["shifted_starts"] == 0
        assert stats["dropped_nested"] == 0
        assert stats["final_count"] == 2
        assert subs[0].end.ordinal == 4_000
        assert subs[1].start.ordinal == 4_040


class TestNestedDuplicate:
    def test_nested_fragment_dropped_full_kept(self, tmp_path):
        # #11/#12 real case: "少し" (0.516s) nested inside "少しだけだぞ…".
        stats, subs = _resolve(tmp_path, [
            (470_626, 472_926, "少しだけだぞ…"),
            (470_810, 471_326, "少し…"),
        ])
        assert stats["dropped_nested"] == 1
        assert stats["shifted_starts"] == 0
        assert stats["final_count"] == 1
        # The FULL sub survives intact (naive clip would have corrupted it).
        assert subs[0].text == "少しだけだぞ…"
        assert subs[0].start.ordinal == 470_626
        assert subs[0].end.ordinal == 472_926

    def test_nested_fragment_second_boundary(self, tmp_path):
        # #13/#14 real case: "…先生。" nested inside "…先生私で…". Exceptionally
        # long negative gap (-3.816s) in a start-sorted view.
        stats, subs = _resolve(tmp_path, [
            (494_242, 498_242, "…先生私で興奮して立ってくれたんですか?"),
            (494_426, 494_942, "…先生。"),
        ])
        assert stats["dropped_nested"] == 1
        assert stats["final_count"] == 1
        assert subs[0].text.startswith("…先生私で")
        assert subs[0].end.ordinal == 498_242

    def test_same_start_keeps_longer(self, tmp_path):
        # Equal start, differing lengths -> keep the longer/fuller entry.
        stats, subs = _resolve(tmp_path, [
            (100_000, 100_500, "短い"),
            (100_000, 104_000, "こちらが完全な文です"),
        ])
        assert stats["dropped_nested"] == 1
        assert stats["final_count"] == 1
        assert subs[0].text == "こちらが完全な文です"
        assert subs[0].end.ordinal == 104_000


class TestOrderingAndScale:
    def test_output_is_start_sorted_and_renumbered(self, tmp_path):
        # Stitcher may append scene subs out of global start order; resolver sorts.
        stats, subs = _resolve(tmp_path, [
            (10_000, 12_000, "second"),
            (1_000, 3_000, "first"),
            (20_000, 22_000, "third"),
        ])
        assert stats["final_count"] == 3
        assert [s.text for s in subs] == ["first", "second", "third"]
        assert [s.index for s in subs] == [1, 2, 3]

    def test_empty_and_single_are_safe(self, tmp_path):
        p = tmp_path / "s.srt"
        p.write_text("", encoding="utf-8")
        assert SceneOverlapResolver().resolve_srt_file(p)["final_count"] == 0

        stats, subs = _resolve(tmp_path, [(1_000, 2_000, "only")])
        assert stats["final_count"] == 1
        assert stats["shifted_starts"] == 0
        assert stats["dropped_nested"] == 0

    def test_chain_of_three_overlaps(self, tmp_path):
        # Three consecutive scene-boundary tail overlaps resolve independently:
        # each entry's END is kept; the next entry's START is pushed past it.
        stats, subs = _resolve(tmp_path, [
            (0, 3_000, "A"),
            (2_500, 5_500, "B"),
            (5_000, 8_000, "C"),
        ])
        assert stats["shifted_starts"] == 2
        assert stats["final_count"] == 3
        # ends all untouched
        assert [s.end.ordinal for s in subs] == [3_000, 5_500, 8_000]
        # starts pushed to just after the previous end
        assert subs[0].start.ordinal == 0
        assert subs[1].start.ordinal == 3_000 + 1
        assert subs[2].start.ordinal == 5_500 + 1
