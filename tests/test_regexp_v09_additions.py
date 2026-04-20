"""Unit tests for Fix 5a — SDH regex pattern additions to regexp_v09.json.

Plan reference: docs/plans/V1811_SANITIZER_FIX_PLAN.md §5.4

Fix 5a adds three things to the bundled regex library:
1. Extended the existing emoji pattern (L268) to cover ♫ ♩ ♬ 🎵 🎶
2. New pattern for tortoise shell brackets: 〔 ... 〕
3. New pattern for SDH annotations inside any bracket type with keywords
   (歓声, 喘息, 呼吸, ノイズ, SE, BGM, SFX)

These tests validate the JSON file directly — its structure, regex
compilability, and match behavior on representative inputs. They don't
go through HallucinationRemover because its loading priority puts the
bundled file 4th (cache > gist > stale-cache > bundled), so changes
to the bundled file may not be observable without cache-clearing.

The bundled JSON is the source of truth for what the maintainer pushes
to the gist; these tests verify the source is correct.
"""

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLED_REGEXP_PATH = (
    REPO_ROOT / "whisperjav" / "data" / "hallucination_filters" / "regexp_v09.json"
)


@pytest.fixture(scope="module")
def regexp_data():
    """Load the bundled regex JSON once per module."""
    with BUNDLED_REGEXP_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


class TestJSONStructure:
    """Basic structural validity of regexp_v09.json."""

    def test_json_loads(self, regexp_data):
        assert "patterns" in regexp_data
        assert isinstance(regexp_data["patterns"], list)

    def test_all_patterns_compile(self, regexp_data):
        """Every regex in the library must compile without errors."""
        for i, entry in enumerate(regexp_data["patterns"]):
            pattern = entry.get("pattern", "")
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Pattern #{i} does not compile: {pattern!r} -> {e}")

    def test_all_patterns_have_required_fields(self, regexp_data):
        """Every entry needs pattern, category, confidence, replacement."""
        required = {"pattern", "category", "confidence", "replacement"}
        for i, entry in enumerate(regexp_data["patterns"]):
            missing = required - set(entry.keys())
            assert not missing, f"Pattern #{i} missing fields: {missing}"


class TestExtendedMusicSymbolPattern:
    """The emoji pattern at L268 should now match ♫ ♩ ♬ 🎵 🎶 in addition to ♪."""

    def test_new_music_symbols_in_emoji_pattern(self, regexp_data):
        emoji_entries = [
            p for p in regexp_data["patterns"]
            if p["category"] == "emoji" and p.get("pattern", "").startswith("[")
        ]
        assert len(emoji_entries) >= 1, "emoji character-class pattern not found"
        pattern_str = emoji_entries[0]["pattern"]
        for sym in ["♪", "♫", "♩", "♬", "🎵", "🎶"]:
            assert sym in pattern_str, (
                f"expected {sym!r} in emoji pattern, got {pattern_str!r}"
            )

    @pytest.mark.parametrize("sym", ["♫", "♩", "♬", "🎵", "🎶"])
    def test_new_music_symbols_match(self, regexp_data, sym):
        """Each new music symbol should match the emoji pattern."""
        emoji_entries = [
            p for p in regexp_data["patterns"]
            if p["category"] == "emoji" and p.get("pattern", "").startswith("[")
        ]
        pattern = re.compile(emoji_entries[0]["pattern"])
        assert pattern.search(sym), f"{sym!r} should match emoji pattern"

    @pytest.mark.parametrize("sym", ["♪", "★", "☆"])
    def test_existing_symbols_still_match(self, regexp_data, sym):
        """Pre-existing symbols must not regress."""
        emoji_entries = [
            p for p in regexp_data["patterns"]
            if p["category"] == "emoji" and p.get("pattern", "").startswith("[")
        ]
        pattern = re.compile(emoji_entries[0]["pattern"])
        assert pattern.search(sym), f"{sym!r} should still match emoji pattern"


class TestTortoiseShellPattern:
    """New pattern: 〔...〕"""

    @pytest.fixture
    def pattern(self, regexp_data):
        matching = [
            p for p in regexp_data["patterns"]
            if "〔" in p.get("pattern", "") and "〕" in p.get("pattern", "")
        ]
        assert matching, "tortoise shell pattern not found in regexp_v09.json"
        return re.compile(matching[0]["pattern"])

    @pytest.mark.parametrize("text,should_match", [
        ("〔効果音〕", True),
        ("〔拍手〕こんにちは", True),
        ("〔音楽〕", True),
        ("〔〕", False),            # empty brackets — pattern requires content
        ("効果音", False),           # no brackets
        ("【効果音】", False),       # different bracket type (caught by L86)
    ])
    def test_tortoise_shell_matches(self, pattern, text, should_match):
        match = pattern.search(text)
        assert bool(match) == should_match, (
            f"{text!r}: expected match={should_match}, got {bool(match)}"
        )


class TestHIKeywordPattern:
    """New pattern: [( or [ or （ or 【 ] with SDH keyword content."""

    @pytest.fixture
    def pattern(self, regexp_data):
        # Find the pattern that includes 'BGM' as a keyword
        matching = [
            p for p in regexp_data["patterns"]
            if "BGM" in p.get("pattern", "") and "歓声" in p.get("pattern", "")
        ]
        assert matching, "HI keyword pattern not found in regexp_v09.json"
        return re.compile(matching[0]["pattern"])

    @pytest.mark.parametrize("text,should_match", [
        ("[BGM]", True),
        ("[SE]", True),
        ("[SFX]", True),
        ("(歓声)", True),
        ("(喘息)", True),
        ("【呼吸】", True),
        ("（ノイズ）", True),
        ("[BGM]こんにちは", True),
        ("BGM", False),               # no brackets
        ("[音楽]", False),             # 音楽 is handled by different pattern L191/L198
        ("(laughter)", False),         # not in this keyword set
    ])
    def test_sdh_keyword_matches(self, pattern, text, should_match):
        match = pattern.search(text)
        assert bool(match) == should_match, (
            f"{text!r}: expected match={should_match}, got {bool(match)}"
        )


class TestReplacementFormat:
    """All new patterns use empty-string replacement (not ${...})."""

    def test_new_patterns_use_empty_replacement(self, regexp_data):
        """No new pattern should use the ${...} syntax — they all drop fully."""
        # Find the new patterns by their notes tags
        new_notes = [
            "(added v1.8.11)",
        ]
        for entry in regexp_data["patterns"]:
            notes = entry.get("notes", "")
            if any(tag in notes for tag in new_notes):
                assert entry.get("replacement", "") == "", (
                    f"New pattern with notes {notes!r} must use empty replacement, "
                    f"got {entry.get('replacement')!r}"
                )
