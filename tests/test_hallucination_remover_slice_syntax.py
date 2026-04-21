"""Unit tests for Fix 1 — ${N:0:M} slice syntax in _apply_regex_replacement_safe.

Plan reference: docs/plans/V1811_SANITIZER_FIX_PLAN.md §5.1

These tests verify that the replacement syntax `${N:0:M}` — which the JSON
regex library in regexp_v09.json uses for several patterns — is correctly
interpreted as "keep the first M characters of match.group(N)".

Before Fix 1: _apply_regex_replacement_safe treated `${...}` as a signal to
drop the match entirely (replace with empty string), breaking the patterns'
intent and producing the #287 "!!" residue symptom when hallucinated kana
runs were followed by punctuation.

After Fix 1: the literal slice syntax is honored.

Tests target:
- The four regexp_v09.json patterns that use ${N:0:M}:
  L142 vowel run      → ${1:0:2}
  L156 vocalization   → ${1:0:2}
  L162 small-tsu      → ${1:0:2}
  L170 long vowel     → ${1:0:3}
- Fallback paths for empty/malformed/unsupported replacements
"""

import pytest

from whisperjav.modules.hallucination_remover import HallucinationRemover
from whisperjav.config.sanitization_constants import HallucinationConstants


@pytest.fixture
def remover():
    """HallucinationRemover configured with Japanese + default constants."""
    return HallucinationRemover(HallucinationConstants(), primary_language="ja")


class TestSliceSyntaxBasic:
    """Direct tests of _apply_regex_replacement_safe with literal input."""

    def test_slice_group1_first_two_chars_on_vowel_run(self, remover):
        # L142 pattern: group 1 captures the whole kana run
        pattern = r'([あアぁァ]{10,}|[いイぃィ]{10,})'
        replacement = '${1:0:2}'
        text = 'いいいいいいいいいいいい'
        result = remover._apply_regex_replacement_safe(pattern, replacement, text)
        assert result == 'いい', f"expected 'いい', got {result!r}"

    def test_slice_group1_first_three_chars_on_long_vowel_mark(self, remover):
        # L170 pattern: keep first 3 of long vowel marks
        pattern = r'([～〜ー]{10,})'
        replacement = '${1:0:3}'
        text = '〜〜〜〜〜〜〜〜〜〜〜〜'
        result = remover._apply_regex_replacement_safe(pattern, replacement, text)
        assert result == '〜〜〜', f"expected '〜〜〜', got {result!r}"

    def test_slice_preserves_non_matching_text(self, remover):
        # Surrounding non-matching text must be preserved
        pattern = r'(いいい+)'
        replacement = '${1:0:2}'
        text = 'prefix いいいいいいいいい suffix'
        result = remover._apply_regex_replacement_safe(pattern, replacement, text)
        assert result == 'prefix いい suffix', f"got {result!r}"


class TestSliceSyntaxWithTrailingPunctuation:
    """#287 reproducer: kana run + trailing single punctuation."""

    def test_vowel_run_plus_question_mark(self, remover):
        # Pre-fix: L142 match stripped to empty → output '?'
        # Post-fix: L142 group(1)[:2] → 'いい?'
        pattern = r'([あアぁァ]{10,}|[いイぃィ]{10,})'
        replacement = '${1:0:2}'
        text = 'いいいいいいいいいいいい?'
        result = remover._apply_regex_replacement_safe(pattern, replacement, text)
        assert result == 'いい?', f"expected 'いい?', got {result!r}"

    def test_vowel_run_plus_period(self, remover):
        pattern = r'([うウぅゥ]{10,})'
        replacement = '${1:0:2}'
        text = 'うううううううううううう。'
        result = remover._apply_regex_replacement_safe(pattern, replacement, text)
        assert result == 'うう。', f"expected 'うう。', got {result!r}"


class TestSliceSyntaxFallbacks:
    """Verify non-slice replacements still work correctly."""

    def test_empty_replacement_still_strips(self, remover):
        """Empty string replacement removes the match entirely."""
        pattern = r'aaa+'
        result = remover._apply_regex_replacement_safe(pattern, '', 'xaaaaaay')
        assert result == 'xy'

    def test_literal_replacement_works(self, remover):
        """Plain string replacement still uses re.sub replacement."""
        pattern = r'aaa+'
        result = remover._apply_regex_replacement_safe(pattern, 'A', 'xaaaaaay')
        assert result == 'xAy'

    def test_null_treated_as_empty(self, remover):
        result = remover._apply_regex_replacement_safe(r'x+', 'null', 'axxxxb')
        assert result == 'ab'

    def test_none_treated_as_empty(self, remover):
        result = remover._apply_regex_replacement_safe(r'x+', 'None', 'axxxxb')
        assert result == 'ab'

    def test_malformed_slice_missing_close_brace_falls_to_empty(self, remover):
        """${1 without } drops match (existing safe behavior)."""
        result = remover._apply_regex_replacement_safe(r'x+', '${1', 'axxxxb')
        assert result == 'ab'

    def test_unsupported_slice_syntax_falls_to_empty(self, remover):
        """${1:0:2:extra} or similar unrecognized forms drop the match."""
        result = remover._apply_regex_replacement_safe(
            r'(x+)', '${1:5:10}', 'axxxxb'
        )
        # Start offset 5 is NOT supported by our fix (only 0 is). Fall through
        # to the existing "unsupported ${...}" safe-empty branch.
        assert result == 'ab'


class TestSliceSyntaxErrorPaths:
    """Edge cases that should not crash."""

    def test_invalid_regex_returns_original_text(self, remover):
        # Intentionally malformed regex
        text = 'いいいいいいいい'
        result = remover._apply_regex_replacement_safe(r'[unclosed', '${1:0:2}', text)
        assert result == text, "invalid regex should leave text unchanged"

    def test_missing_group_does_not_crash(self, remover):
        """${9:0:2} where group 9 doesn't exist — safe fallback to empty."""
        pattern = r'(x+)'
        text = 'axxxxb'
        result = remover._apply_regex_replacement_safe(pattern, '${9:0:2}', text)
        # Our _slice_replace catches IndexError/TypeError and returns ''
        assert result == 'ab'


class TestSliceSyntaxEndToEnd:
    """End-to-end via remove_hallucinations, exercising the regex-matching
    branch. These verify the fix integrates correctly with the full
    hallucination_remover flow."""

    def test_287_reproducer_end_to_end(self, remover):
        """The classic #287 shape: long kana run + trailing '?' should yield
        'いい?' after Fix 1, not '?' or empty."""
        text = 'いいいいいいいいいいいい?'
        result, mods = remover.remove_hallucinations(text)
        # Fix 1 should turn this into 'いい?' via regex_partial_strip
        assert result == 'いい?', f"expected 'いい?', got {result!r}"

    def test_hagh_repetition_residue(self, remover):
        """はぁ repeated 10+ times, group(1) captures last 'はぁ', [:2] = 'はぁ'."""
        text = 'はぁはぁはぁはぁはぁはぁはぁはぁはぁはぁ'
        result, mods = remover.remove_hallucinations(text)
        assert result == 'はぁ', f"expected 'はぁ', got {result!r}"

    def test_long_vowel_mark_run(self, remover):
        """Long vowel marks — Fix 1 produces 3-char residue (no kana content)."""
        text = '〜〜〜〜〜〜〜〜〜〜〜〜'
        result, mods = remover.remove_hallucinations(text)
        assert result == '〜〜〜', f"expected '〜〜〜', got {result!r}"

    def test_plain_text_unchanged(self, remover):
        """No hallucination match → text returned unchanged.

        Uses a specific phrase unlikely to be in filter_list_v08.json so
        the test verifies the no-match pass-through path, not the filter
        list lookup.
        """
        text = 'わああうう'  # non-trigger, not in filter list
        result, mods = remover.remove_hallucinations(text)
        assert result == 'わああうう'
        assert mods == []
