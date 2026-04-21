"""Unit tests for Fix 3 — exclude internal newlines from CPS char count.

Plan reference: docs/plans/V1811_SANITIZER_FIX_PLAN.md §5.3

Fix 3 changes _remove_abnormally_fast_subs to compute CPS based on the
visible character count (excluding internal '\n' newlines) rather than
the raw length including newlines. This prevents multi-line subtitles
from being treated as "longer" than they actually are when evaluating
reading speed.

The change is a one-line edit at L1044:
  text_len = len(sub.text_without_tags.strip())
  ->
  text_len = len(sub.text_without_tags.replace('\n', '').strip())

Most multi-line inputs don't change verdict (CPS differences are small),
but one meaningful case flips: 4 kana across 4 lines over 10 seconds
has pre-fix text_len=7 (kana+3 newlines) so the slow-CPS filter skips
it (len>4), whereas post-fix text_len=4 triggers slow-CPS removal.
"""

import pysrt
import pytest

from whisperjav.modules.subtitle_sanitizer import SubtitleSanitizer
from whisperjav.config.sanitization_config import SanitizationConfig


@pytest.fixture
def sanitizer():
    config = SanitizationConfig()
    config.primary_language = "ja"
    config.save_artifacts = True
    config.preserve_original_file = True
    return SubtitleSanitizer(config=config)


def _make_sub(index, start_ms, end_ms, text):
    """Build a pysrt.SubRipItem from milliseconds."""
    from pysrt import SubRipTime
    s = SubRipTime.from_ordinal(start_ms)
    e = SubRipTime.from_ordinal(end_ms)
    return pysrt.SubRipItem(index=index, start=s, end=e, text=text)


class TestCPSCharCountExcludesNewlines:
    """Unit-level tests on _remove_abnormally_fast_subs behavior."""

    def test_single_line_behavior_unchanged(self, sanitizer):
        """Single-line subs have no newlines to exclude — identical behavior."""
        sub = _make_sub(1, 0, 1000, "あいう")  # 3 chars, 1s = 3 CPS
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 1, "single-line sub in safe CPS range should be kept"

    def test_multiline_slow_short_now_dropped(self, sanitizer):
        """4 kana / 4 lines / 10s:
          pre-fix: text_len=7 (4 kana + 3 newlines), 7/10=0.7 CPS, len>4 → KEEP
          post-fix: text_len=4, 4/10=0.4 CPS, len<=4 AND cps<1.0 → REMOVE
        """
        sub = _make_sub(1, 0, 10_000, "あ\nい\nう\nえ")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 0, (
            "Fix 3: 4-kana multiline slow sub should now be removed as slow-CPS; "
            f"got {len(result)} kept"
        )

    def test_multiline_fast_still_dropped(self, sanitizer):
        """Fast multi-line: extreme CPS catches it regardless of newline handling."""
        # 6 chars across 3 lines in 100ms → CPS ~60 either way
        sub = _make_sub(1, 0, 100, "はい\nはい\nはい")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 0, "fast multi-line should remain dropped"

    def test_multiline_normal_cps_kept(self, sanitizer):
        """Multi-line normal reading speed — kept both pre and post fix."""
        # 'こんにちは\n今日は' at 2s: 8 visible chars / 2s = 4 CPS
        sub = _make_sub(1, 0, 2000, "こんにちは\n今日は")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 1, "normal-CPS multi-line should be kept"

    def test_edge_case_4_chars_4_lines_short_duration(self, sanitizer):
        """Same 4-kana-across-4-lines but at 3s — post-fix CPS = 4/3 ≈ 1.33.
        That's above MIN_SAFE_CPS so it should be KEPT even post-fix.
        """
        sub = _make_sub(1, 0, 3000, "あ\nい\nう\nえ")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 1, (
            "post-fix: 4 visible chars / 3s = 1.33 CPS > 1.0 → should be kept"
        )


class TestCPSNewlineHandlingBoundaries:
    """Boundary cases that previously relied on newline-inflated char counts."""

    def test_kanji_with_single_newline(self, sanitizer):
        """A short kanji pair with one internal newline — short enough for
        post-fix slow-CPS check but previously escaped via newline inflation."""
        # '家\n族' at 10s: pre-fix len=3, post-fix len=2 → CPS=0.2 → REMOVE
        sub = _make_sub(1, 0, 10_000, "家\n族")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 0, "post-fix: 2-char multiline slow sub should be dropped"

    def test_trailing_newline_stripped(self, sanitizer):
        """Trailing newlines are stripped by .strip(), matching pre-existing behavior."""
        sub = _make_sub(1, 0, 1000, "こんにちは\n")
        result = sanitizer._remove_abnormally_fast_subs([sub])
        assert len(result) == 1
