"""Tests for CpsStartRetimer (v1.9.0) — Qwen Phase-8 CPS-anomaly start retiming.

The retimer ports the legacy TimingAdjuster rule (end fixed, start moved to
end - expected duration) to the Qwen pipeline's stitched SRT. See
whisperjav/modules/subtitle_pipeline/cleaners/cps_start_retimer.py.
"""

import pysrt
import pytest

from whisperjav.modules.subtitle_pipeline.cleaners.cps_start_retimer import (
    CpsStartRetimer,
)


def _make_srt(tmp_path, entries):
    """Write an SRT from (start_ms, end_ms, text) tuples; return its path."""
    subs = pysrt.SubRipFile(
        items=[
            pysrt.SubRipItem(
                index=i,
                start=pysrt.SubRipTime(milliseconds=start_ms),
                end=pysrt.SubRipTime(milliseconds=end_ms),
                text=text,
            )
            for i, (start_ms, end_ms, text) in enumerate(entries, start=1)
        ]
    )
    path = tmp_path / "test.srt"
    subs.save(str(path), encoding="utf-8")
    return path


class TestClassify:
    def setup_method(self):
        self.retimer = CpsStartRetimer(language="ja")

    def test_normal_entry_not_flagged(self):
        # 7 chars over 1s at ja 7.0 CPS — exactly nominal.
        assert self.retimer._classify(text_len=7, duration_s=1.0) == ""

    def test_duration_hallucination(self):
        assert (
            self.retimer._classify(text_len=20, duration_s=15.0)
            == "duration_hallucination"
        )

    def test_slow_cps(self):
        # 4 chars over 8s = 0.5 CPS < MIN_SAFE_CPS 1.0
        assert self.retimer._classify(text_len=4, duration_s=8.0) == "abnormally_slow_cps"

    def test_duration_checked_before_cps(self):
        # Both violated (2 chars over 15s) — duration trigger wins (parity
        # with TimingAdjuster's condition order).
        assert (
            self.retimer._classify(text_len=2, duration_s=15.0)
            == "duration_hallucination"
        )

    def test_empty_text_not_cps_checked_but_duration_still_applies(self):
        # text_len 0 < MIN_TEXT_LENGTH_FOR_CPS_CHECK (1) -> no CPS trigger...
        assert self.retimer._classify(text_len=0, duration_s=5.0) == ""
        # ...but a >12s duration still triggers independent of text.
        assert (
            self.retimer._classify(text_len=0, duration_s=13.0)
            == "duration_hallucination"
        )


class TestRetimeSrtFile:
    def test_slow_cps_start_moved_end_fixed(self, tmp_path):
        # 4 chars over 10s = 0.4 CPS. Expected duration = max(4/7.0, 0.3) ≈ 0.571s.
        path = _make_srt(tmp_path, [(0, 10_000, "うん。あ")])
        stats = CpsStartRetimer().retime_srt_file(path)

        assert stats["retimed_slow_cps"] == 1
        assert stats["retimed_total"] == 1
        subs = pysrt.open(str(path), encoding="utf-8")
        assert subs[0].end.ordinal == 10_000  # end untouched
        assert subs[0].start.ordinal == 10_000 - int((4 / 7.0) * 1000)

    def test_duration_hallucination_clamped_to_max(self, tmp_path):
        # 200 chars over 30s: 200/7 ≈ 28.6s exceeds MAX_SUBTITLE_DURATION,
        # so expected duration clamps to 12s.
        path = _make_srt(tmp_path, [(0, 30_000, "あ" * 200)])
        stats = CpsStartRetimer().retime_srt_file(path)

        assert stats["retimed_long_duration"] == 1
        subs = pysrt.open(str(path), encoding="utf-8")
        assert subs[0].end.ordinal == 30_000
        assert subs[0].start.ordinal == 30_000 - 12_000

    def test_normal_entries_untouched(self, tmp_path):
        entries = [
            (0, 2_000, "気持ちいいですか。"),  # 9 chars / 2s = 4.5 CPS
            (2_500, 4_000, "はい。"),  # 3 chars / 1.5s = 2.0 CPS
        ]
        path = _make_srt(tmp_path, entries)
        stats = CpsStartRetimer().retime_srt_file(path)

        assert stats["retimed_total"] == 0
        subs = pysrt.open(str(path), encoding="utf-8")
        assert [(s.start.ordinal, s.end.ordinal) for s in subs] == [
            (0, 2_000),
            (2_500, 4_000),
        ]

    def test_starts_only_move_later_no_overlap_created(self, tmp_path):
        # Anomalous sub follows a normal one; retiming must land its new
        # start AFTER the previous end (starts move later, never earlier).
        entries = [
            (0, 3_000, "こんにちは。"),
            (3_000, 12_500, "あ。"),  # 2 chars / 9.5s = 0.21 CPS
        ]
        path = _make_srt(tmp_path, entries)
        CpsStartRetimer().retime_srt_file(path)

        subs = pysrt.open(str(path), encoding="utf-8")
        assert subs[1].start.ordinal > subs[0].end.ordinal
        assert subs[1].start.ordinal > 3_000  # moved later than original

    def test_entry_count_and_order_preserved(self, tmp_path):
        entries = [
            (0, 2_000, "普通の字幕です。"),
            (2_000, 12_500, "あ。"),
            (13_000, 15_000, "また普通の字幕。"),
        ]
        path = _make_srt(tmp_path, entries)
        stats = CpsStartRetimer().retime_srt_file(path)

        assert stats["original_count"] == 3
        assert stats["final_count"] == 3
        subs = pysrt.open(str(path), encoding="utf-8")
        assert [s.index for s in subs] == [1, 2, 3]
        assert [s.text for s in subs] == [e[2] for e in entries]

    def test_missing_file_returns_zero_stats(self, tmp_path):
        stats = CpsStartRetimer().retime_srt_file(tmp_path / "nope.srt")
        assert stats == {
            "original_count": 0,
            "retimed_long_duration": 0,
            "retimed_slow_cps": 0,
            "retimed_total": 0,
            "final_count": 0,
        }

    def test_empty_file_returns_zero_stats(self, tmp_path):
        path = tmp_path / "empty.srt"
        path.write_text("", encoding="utf-8")
        stats = CpsStartRetimer().retime_srt_file(path)
        assert stats["original_count"] == 0
        assert stats["retimed_total"] == 0

    def test_no_write_when_nothing_retimed(self, tmp_path):
        path = _make_srt(tmp_path, [(0, 2_000, "普通の字幕です。")])
        before = path.read_bytes()
        mtime = path.stat().st_mtime_ns
        CpsStartRetimer().retime_srt_file(path)
        assert path.read_bytes() == before
        assert path.stat().st_mtime_ns == mtime

    def test_html_tags_excluded_from_cps_text_length(self, tmp_path):
        # Tag characters must not inflate text length: <i>あ。</i> is 2 visible
        # chars over 9.5s -> slow CPS despite the 9-char raw string.
        path = _make_srt(tmp_path, [(0, 9_500, "<i>あ。</i>")])
        stats = CpsStartRetimer().retime_srt_file(path)
        assert stats["retimed_slow_cps"] == 1

    def test_non_ja_language_uses_its_reading_speed(self, tmp_path):
        # en: 15 CPS. 30 chars over 25s -> duration hallucination; expected
        # duration = max(30/15, 0.3) = 2.0s.
        path = _make_srt(tmp_path, [(0, 25_000, "a" * 30)])
        stats = CpsStartRetimer(language="en").retime_srt_file(path)
        assert stats["retimed_long_duration"] == 1
        subs = pysrt.open(str(path), encoding="utf-8")
        assert subs[0].start.ordinal == 25_000 - 2_000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
