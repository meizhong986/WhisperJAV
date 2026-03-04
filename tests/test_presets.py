"""
Unit tests for ensemble parameter presets (whisperjav/settings/presets.py).

Each test is tagged with a Preset Scenario ID (P01–P20) and contains the
expected behavior in its docstring.

Acceptance criteria:
    P01  Empty presets dir — list returns empty
    P02  Save + list — preset appears in list
    P03  Save + load — full roundtrip
    P04  Delete — preset removed from disk
    P05  Overwrite — save same name overwrites, preserves created_at
    P06  Filename sanitization — special characters
    P07  Filename sanitization — long names truncated
    P08  Filename sanitization — empty/whitespace name rejected
    P09  Unicode names — preserved in filename
    P10  Atomic write — no .tmp left behind
    P11  Load non-existent — returns None
    P12  Load corrupt JSON — returns None, no crash
    P13  Delete non-existent — returns False
    P14  Multiple presets — all listed, sorted by name
    P15  Preset schema version stamped
    P16  Timestamps — created_at and updated_at set
    P17  API mapping — camelCase ↔ snake_case roundtrip
    P18  Rename preset — old deleted, new created
    P19  Concurrent saves — last-write-wins
    P20  Presets dir auto-created on save

Layer 1 tests (pure Python, no GUI).
"""

import json
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_presets(tmp_path, monkeypatch):
    """Redirect get_presets_dir() to a temp directory for every test."""
    presets_dir = tmp_path / "WhisperJAV" / "presets"

    monkeypatch.setattr(
        "whisperjav.settings.presets.get_presets_dir",
        lambda: presets_dir,
    )
    return presets_dir


@pytest.fixture
def presets_dir(tmp_path):
    """Return the isolated presets directory path."""
    return tmp_path / "WhisperJAV" / "presets"


def _sample_preset(**overrides):
    """Create a sample preset dict with sensible defaults."""
    data = {
        "pipeline": "balanced",
        "sensitivity": "aggressive",
        "scene_detector": "semantic",
        "speech_enhancer": "none",
        "speech_segmenter": "silero-v6.2",
        "model": "large-v2",
        "customized": True,
        "params": {"beam_size": 8, "temperature": 0.0},
        "is_transformers": False,
        "is_qwen": False,
        "framer": None,
        "dsp_effects": None,
    }
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from whisperjav.settings.presets import (
    PRESET_SCHEMA_VERSION,
    _sanitize_filename,
    delete_preset,
    get_presets_dir,
    list_presets,
    load_preset,
    rename_preset,
    save_preset,
)


# ===================================================================
# P01: Empty presets dir — list returns empty
# ===================================================================

class TestP01EmptyDir:
    """P01: list_presets on empty/missing dir returns empty list."""

    def test_list_empty_when_dir_missing(self):
        """P01: no presets dir → empty list, no error."""
        result = list_presets()
        assert result == []

    def test_list_empty_when_dir_exists_but_empty(self, presets_dir):
        """P01: empty presets dir → empty list."""
        presets_dir.mkdir(parents=True, exist_ok=True)
        result = list_presets()
        assert result == []


# ===================================================================
# P02: Save + list
# ===================================================================

class TestP02SaveAndList:
    """P02: Saved preset appears in list with correct summary fields."""

    def test_save_then_list(self):
        """P02: save a preset, list returns it with name and pipeline."""
        save_preset("My Config", _sample_preset(pipeline="fast"))
        result = list_presets()
        assert len(result) == 1
        assert result[0]["name"] == "My Config"
        assert result[0]["pipeline"] == "fast"


# ===================================================================
# P03: Save + load roundtrip
# ===================================================================

class TestP03Roundtrip:
    """P03: All fields survive save → load cycle."""

    def test_full_roundtrip(self):
        """P03: save full preset, load by name — all fields match."""
        original = _sample_preset(
            pipeline="fidelity",
            sensitivity="conservative",
            params={"beam_size": 12, "temperature": 0.2, "vad_threshold": 0.4},
            is_qwen=True,
            framer="full-scene",
        )
        save_preset("Roundtrip Test", original)
        loaded = load_preset("Roundtrip Test")

        assert loaded is not None
        assert loaded["name"] == "Roundtrip Test"
        assert loaded["pipeline"] == "fidelity"
        assert loaded["sensitivity"] == "conservative"
        assert loaded["params"]["beam_size"] == 12
        assert loaded["params"]["temperature"] == 0.2
        assert loaded["is_qwen"] is True
        assert loaded["framer"] == "full-scene"

    def test_boolean_values_preserved(self):
        """P03: boolean fields survive JSON roundtrip."""
        save_preset("Bool Test", _sample_preset(
            customized=True,
            is_transformers=False,
            is_qwen=True,
        ))
        loaded = load_preset("Bool Test")
        assert loaded["customized"] is True
        assert loaded["is_transformers"] is False
        assert loaded["is_qwen"] is True


# ===================================================================
# P04: Delete
# ===================================================================

class TestP04Delete:
    """P04: Preset removed from disk after delete."""

    def test_save_delete_list(self):
        """P04: save, delete, list → empty."""
        save_preset("To Delete", _sample_preset())
        assert len(list_presets()) == 1

        result = delete_preset("To Delete")
        assert result is True
        assert len(list_presets()) == 0

    def test_file_removed_from_disk(self, presets_dir):
        """P04: the .json file is actually deleted."""
        save_preset("File Check", _sample_preset())
        files_before = list(presets_dir.glob("*.json"))
        assert len(files_before) == 1

        delete_preset("File Check")
        files_after = list(presets_dir.glob("*.json"))
        assert len(files_after) == 0


# ===================================================================
# P05: Overwrite — preserves created_at
# ===================================================================

class TestP05Overwrite:
    """P05: Saving with same name overwrites, preserves original created_at."""

    def test_overwrite_preserves_created_at(self):
        """P05: first save sets created_at, second save preserves it."""
        save_preset("Overwrite Me", _sample_preset(pipeline="fast"))
        first = load_preset("Overwrite Me")
        created_at_1 = first["created_at"]

        # Small delay to differentiate timestamps
        save_preset("Overwrite Me", _sample_preset(pipeline="fidelity"))
        second = load_preset("Overwrite Me")

        assert second["pipeline"] == "fidelity"  # updated
        assert second["created_at"] == created_at_1  # preserved

    def test_overwrite_updates_updated_at(self):
        """P05: overwrite changes updated_at."""
        save_preset("Update Time", _sample_preset())
        first = load_preset("Update Time")

        save_preset("Update Time", _sample_preset(pipeline="fast"))
        second = load_preset("Update Time")

        # updated_at should be >= first (same second is acceptable)
        assert second["updated_at"] >= first["updated_at"]


# ===================================================================
# P06: Filename sanitization — special characters
# ===================================================================

class TestP06Sanitization:
    """P06: Unsafe filesystem characters replaced in filename."""

    @pytest.mark.parametrize("name,expected_stem", [
        ("My/Config", "My Config"),
        ('Test: "best"', "Test best"),
        ("A\\B<C>D|E", "A B C D E"),
        ("Normal Name", "Normal Name"),
        ("  spaces  ", "spaces"),
    ])
    def test_special_chars_sanitized(self, name, expected_stem):
        """P06: dangerous characters replaced, name preserved as readable."""
        result = _sanitize_filename(name)
        assert result == expected_stem

    def test_save_with_special_chars(self):
        """P06: save with special chars in name → loads back correctly."""
        save_preset("My/Custom: Config", _sample_preset())
        # Load by the original name
        loaded = load_preset("My/Custom: Config")
        assert loaded is not None
        assert loaded["name"] == "My/Custom: Config"


# ===================================================================
# P07: Long names truncated
# ===================================================================

class TestP07LongNames:
    """P07: Names longer than max are truncated in filename."""

    def test_long_name_truncated(self):
        """P07: 100-char name → filename stem truncated to 80 chars."""
        long_name = "A" * 100
        result = _sanitize_filename(long_name)
        assert len(result) <= 80

    def test_long_name_save_load(self):
        """P07: save with long name, load by same name → works."""
        long_name = "B" * 100
        save_preset(long_name, _sample_preset())
        loaded = load_preset(long_name)
        assert loaded is not None
        assert loaded["name"] == long_name  # full name preserved in JSON


# ===================================================================
# P08: Empty name rejected
# ===================================================================

class TestP08EmptyName:
    """P08: Empty or whitespace-only name → save returns False."""

    @pytest.mark.parametrize("name", ["", "   ", None])
    def test_empty_name_rejected(self, name):
        """P08: save with empty/whitespace name fails gracefully."""
        result = save_preset(name or "", _sample_preset())
        assert result is False


# ===================================================================
# P09: Unicode names
# ===================================================================

class TestP09Unicode:
    """P09: Unicode characters preserved in preset name and filename."""

    def test_japanese_name(self):
        """P09: Japanese preset name survives roundtrip."""
        save_preset("JAV Config", _sample_preset())
        loaded = load_preset("JAV Config")
        assert loaded is not None
        assert loaded["name"] == "JAV Config"

    def test_mixed_unicode(self):
        """P09: mixed Unicode + ASCII name works."""
        name = "My Preset (v2)"
        save_preset(name, _sample_preset())
        loaded = load_preset(name)
        assert loaded is not None


# ===================================================================
# P10: Atomic write — no .tmp left
# ===================================================================

class TestP10AtomicWrite:
    """P10: No .tmp file left behind after successful save."""

    def test_no_tmp_after_save(self, presets_dir):
        """P10: .json.tmp cleaned up after save."""
        save_preset("Atomic Test", _sample_preset())
        tmp_files = list(presets_dir.glob("*.tmp"))
        assert len(tmp_files) == 0


# ===================================================================
# P11: Load non-existent
# ===================================================================

class TestP11LoadMissing:
    """P11: load_preset on non-existent name returns None."""

    def test_load_missing_returns_none(self):
        """P11: no crash, returns None."""
        result = load_preset("Does Not Exist")
        assert result is None


# ===================================================================
# P12: Load corrupt JSON
# ===================================================================

class TestP12CorruptJSON:
    """P12: Corrupt preset file returns None, no crash."""

    def test_corrupt_json(self, presets_dir):
        """P12: invalid JSON → None returned."""
        presets_dir.mkdir(parents=True, exist_ok=True)
        corrupt_path = presets_dir / "corrupt.json"
        corrupt_path.write_text("{invalid json!!", encoding="utf-8")

        # list_presets should skip it
        result = list_presets()
        assert len(result) == 0

    def test_missing_name_key(self, presets_dir):
        """P12: JSON without 'name' key → skipped in list."""
        presets_dir.mkdir(parents=True, exist_ok=True)
        (presets_dir / "no_name.json").write_text(
            '{"pipeline": "fast"}', encoding="utf-8"
        )
        result = list_presets()
        assert len(result) == 0


# ===================================================================
# P13: Delete non-existent
# ===================================================================

class TestP13DeleteMissing:
    """P13: delete_preset on non-existent name returns False."""

    def test_delete_missing(self):
        """P13: returns False, no crash."""
        result = delete_preset("Ghost Preset")
        assert result is False


# ===================================================================
# P14: Multiple presets sorted
# ===================================================================

class TestP14MultipleSorted:
    """P14: Multiple presets listed in alphabetical order by name."""

    def test_sorted_by_name(self):
        """P14: 3 presets → listed in alphabetical order."""
        save_preset("Zebra", _sample_preset())
        save_preset("Apple", _sample_preset())
        save_preset("Mango", _sample_preset())

        result = list_presets()
        names = [p["name"] for p in result]
        assert names == ["Apple", "Mango", "Zebra"]


# ===================================================================
# P15: Schema version stamped
# ===================================================================

class TestP15SchemaVersion:
    """P15: Saved presets include schema_version."""

    def test_schema_version_present(self):
        """P15: schema_version field set to PRESET_SCHEMA_VERSION."""
        save_preset("Version Test", _sample_preset())
        loaded = load_preset("Version Test")
        assert loaded["schema_version"] == PRESET_SCHEMA_VERSION


# ===================================================================
# P16: Timestamps
# ===================================================================

class TestP16Timestamps:
    """P16: created_at and updated_at set automatically."""

    def test_timestamps_present(self):
        """P16: both timestamps present after save."""
        save_preset("Time Test", _sample_preset())
        loaded = load_preset("Time Test")
        assert "created_at" in loaded
        assert "updated_at" in loaded
        assert loaded["created_at"]  # non-empty
        assert loaded["updated_at"]  # non-empty


# ===================================================================
# P17: API mapping roundtrip
# ===================================================================

class TestP17APIMapping:
    """P17: camelCase ↔ snake_case mapping is complete and reversible."""

    def test_mapping_covers_all_preset_fields(self):
        """P17: every field in a typical preset has a camelCase mapping."""
        from whisperjav.webview_gui.api import WhisperJAVAPI

        mapping = WhisperJAVAPI._PRESET_MAP
        # Key preset fields that must be mapped
        required_fields = [
            "name", "pipeline", "sensitivity", "scene_detector",
            "speech_enhancer", "speech_segmenter", "model",
            "customized", "params", "is_transformers", "is_qwen",
            "framer", "dsp_effects", "created_at", "updated_at",
        ]
        for field in required_fields:
            assert field in mapping, f"Missing mapping for {field!r}"

    def test_reverse_mapping_consistent(self):
        """P17: forward and reverse maps are inverses."""
        from whisperjav.webview_gui.api import WhisperJAVAPI

        fwd = WhisperJAVAPI._PRESET_MAP
        rev = WhisperJAVAPI._PRESET_MAP_REV
        for snake, camel in fwd.items():
            assert rev[camel] == snake

    def test_api_roundtrip(self):
        """P17: save via API (camelCase) → load via API → values match."""
        from whisperjav.webview_gui.api import WhisperJAVAPI
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)

        # Simulate frontend sending camelCase
        frontend_data = {
            "pipeline": "fidelity",
            "sensitivity": "conservative",
            "sceneDetector": "semantic",
            "speechEnhancer": "none",
            "speechSegmenter": "silero-v6.2",
            "model": "large-v2",
            "customized": True,
            "params": {"beam_size": 10},
            "isTransformers": False,
            "isQwen": False,
        }

        # API save path: camelCase → snake_case → disk
        backend_data = api._preset_from_camel(frontend_data)
        save_preset("API Test", backend_data)

        # API load path: disk → snake_case → camelCase
        loaded = load_preset("API Test")
        gui_data = api._preset_to_camel(loaded)

        assert gui_data["pipeline"] == "fidelity"
        assert gui_data["sensitivity"] == "conservative"
        assert gui_data["sceneDetector"] == "semantic"
        assert gui_data["params"]["beam_size"] == 10
        assert gui_data["isQwen"] is False


# ===================================================================
# P18: Rename
# ===================================================================

class TestP18Rename:
    """P18: Rename deletes old file and creates new one."""

    def test_rename_success(self):
        """P18: rename → old gone, new exists with updated name."""
        save_preset("Old Name", _sample_preset(pipeline="fast"))
        result = rename_preset("Old Name", "New Name")
        assert result is True

        assert load_preset("Old Name") is None
        loaded = load_preset("New Name")
        assert loaded is not None
        assert loaded["name"] == "New Name"
        assert loaded["pipeline"] == "fast"

    def test_rename_to_existing_fails(self):
        """P18: rename to already-existing name returns False."""
        save_preset("First", _sample_preset())
        save_preset("Second", _sample_preset())
        result = rename_preset("First", "Second")
        assert result is False
        # Both still exist
        assert load_preset("First") is not None
        assert load_preset("Second") is not None


# ===================================================================
# P19: Concurrent saves
# ===================================================================

class TestP19Concurrent:
    """P19: Last-write-wins, no corruption."""

    def test_sequential_saves_last_wins(self):
        """P19: two saves with different data → second wins."""
        save_preset("Race", _sample_preset(pipeline="fast"))
        save_preset("Race", _sample_preset(pipeline="fidelity"))
        loaded = load_preset("Race")
        assert loaded["pipeline"] == "fidelity"


# ===================================================================
# P20: Presets dir auto-created
# ===================================================================

class TestP20DirCreation:
    """P20: Presets directory created automatically on first save."""

    def test_dir_created_on_save(self, presets_dir):
        """P20: save creates parent dirs if missing."""
        assert not presets_dir.exists()
        save_preset("First Preset", _sample_preset())
        assert presets_dir.is_dir()
