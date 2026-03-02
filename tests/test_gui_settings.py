"""
Unit tests for GUI settings persistence (whisperjav/settings/gui_settings.py).

Each test is tagged with a Scenario ID (S01–S28) from the acceptance criteria
table.  The docstring contains the exact expected behaviour from that table so
the test is self-documenting and traceable.

Acceptance criteria reference:
    S01  First launch, no settings file
    S02  Change one dropdown, close, reopen
    S03  Change multiple fields across Tab 1 + Tab 3, reopen
    S04  (frontend — debounce; not testable here)
    S05  (crash window — design tradeoff; not testable here)
    S06  App upgraded, new keys in schema
    S07  App downgraded, unknown keys in file
    S08  Schema version changes — backup + migration
    S09  Settings file has invalid JSON — corruption recovery
    S10  Settings file read-only — save fails gracefully
    S11  Settings directory missing — auto-created on save
    S12  User deletes settings file — defaults on next load
    S13  Disk full / save fails — no crash
    S19  Saved dropdown value no longer exists (frontend only)
    S20  Boolean checkbox roundtrip
    S21  Empty outputDir/tempDir preserved
    S22  Custom outputDir persisted (backend part)
    S23  Two app instances, last-write-wins
    S24  Windows path resolution
    S25  macOS path resolution
    S26  Linux path resolution
    S27  CLI doesn't read GUI settings
    S28  snake_case ↔ camelCase roundtrip

Layer 1 tests (pure Python, no GUI, no browser).
All use tmp_path + monkeypatch to isolate from the real settings file.
"""

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_settings(tmp_path, monkeypatch):
    """Redirect get_gui_settings_path() to a temp directory for every test."""
    settings_file = tmp_path / "WhisperJAV" / "gui_settings.json"

    monkeypatch.setattr(
        "whisperjav.settings.gui_settings.get_gui_settings_path",
        lambda: settings_file,
    )
    return settings_file


@pytest.fixture
def settings_file(tmp_path):
    """Return the isolated settings file path (matches _isolate_settings)."""
    return tmp_path / "WhisperJAV" / "gui_settings.json"


def _write_json(path: Path, data: dict):
    """Helper — write a JSON dict to *path*, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _read_json(path: Path) -> dict:
    """Helper — read JSON dict from *path*."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Imports (after fixture setup to avoid import-time side-effects)
# ---------------------------------------------------------------------------

from whisperjav.settings.gui_settings import (
    DEFAULT_GUI_SETTINGS,
    SETTINGS_SCHEMA_VERSION,
    _MAX_BACKUPS,
    _migrate_settings,
    _rotate_backups,
    get_gui_settings_path,  # module-level ref → original function (not monkeypatched)
    load_gui_settings,
    save_gui_settings,
)


# ===================================================================
# S01: First launch, no settings file
# ===================================================================

class TestS01FirstLaunch:
    """S01: Form shows HTML defaults. No file created until user acts."""

    def test_load_returns_defaults_when_no_file(self, settings_file):
        """S01: load_gui_settings returns DEFAULT_GUI_SETTINGS when no file exists."""
        assert not settings_file.exists()
        result = load_gui_settings()
        assert result == DEFAULT_GUI_SETTINGS

    def test_load_does_not_create_file(self, settings_file):
        """S01: load should be read-only — no file creation side-effect."""
        load_gui_settings()
        assert not settings_file.exists()

    def test_defaults_have_correct_version(self):
        """S01: DEFAULT_GUI_SETTINGS version matches SETTINGS_SCHEMA_VERSION."""
        assert DEFAULT_GUI_SETTINGS["version"] == SETTINGS_SCHEMA_VERSION

    def test_defaults_match_html_selected_attrs(self):
        """S01: DEFAULT_GUI_SETTINGS values MUST match index.html selected attrs.

        This is the root cause of S01 bug: if these drift, first-launch users
        see wrong defaults.  The expected values below are copied from the HTML.
        If index.html changes, update both the HTML and this test.
        """
        # HTML `selected` values (source of truth: index.html)
        html_defaults = {
            "mode": "balanced",
            "source_language": "japanese",
            "subs_language": "native",
            "sensitivity": "aggressive",
            "model_override_enabled": False,
            "model_override": "",
            "output_to_source": True,
            "output_dir": "",
            "debug_logging": False,
            "keep_temp": False,
            "temp_dir": "",
            "accept_cpu_mode": False,
            "async_processing": False,
            "pass1_pipeline": "balanced",
            "pass1_sensitivity": "aggressive",
            "pass1_scene_detector": "semantic",
            "pass1_speech_enhancer": "none",
            "pass1_speech_segmenter": "silero-v6.2",
            "pass1_model": "large-v2",
            "pass2_enabled": False,
            "pass2_pipeline": "qwen",
            "pass2_sensitivity": "balanced",
            "pass2_scene_detector": "semantic",
            "pass2_speech_enhancer": "none",
            "pass2_speech_segmenter": "silero-v6.2",
            "pass2_model": "Qwen/Qwen3-ASR-1.7B",
            "merge_strategy": "pass1_primary",
        }
        for key, expected in html_defaults.items():
            actual = DEFAULT_GUI_SETTINGS.get(key)
            assert actual == expected, (
                f"DEFAULT_GUI_SETTINGS[{key!r}] = {actual!r}, "
                f"but HTML selected = {expected!r}"
            )


# ===================================================================
# S02: Single setting change persists across restart
# ===================================================================

class TestS02SingleChangePersists:
    """S02: Changed value persists. All other fields unchanged."""

    def test_save_then_load_roundtrip(self, settings_file):
        """S02: save one field, load — that field changed, others are defaults."""
        save_gui_settings({"mode": "fidelity"})
        result = load_gui_settings()
        assert result["mode"] == "fidelity"
        # Other fields untouched
        assert result["sensitivity"] == DEFAULT_GUI_SETTINGS["sensitivity"]
        assert result["pass1_pipeline"] == DEFAULT_GUI_SETTINGS["pass1_pipeline"]

    def test_file_is_valid_json(self, settings_file):
        """S02: saved file is readable JSON."""
        save_gui_settings({"mode": "fast"})
        data = _read_json(settings_file)
        assert data["mode"] == "fast"
        assert data["version"] == SETTINGS_SCHEMA_VERSION


# ===================================================================
# S03: Multiple fields across tabs persist
# ===================================================================

class TestS03MultipleFieldsPersist:
    """S03: All changed values restored after save+load cycle."""

    def test_tab1_and_tab3_fields(self, settings_file):
        """S03: mix of Tab 1 and Tab 3 fields all survive roundtrip."""
        changes = {
            "mode": "fidelity",
            "sensitivity": "conservative",
            "pass1_pipeline": "fast",
            "pass2_enabled": True,
            "pass2_pipeline": "balanced",
            "merge_strategy": "smart_merge",
        }
        save_gui_settings(changes)
        result = load_gui_settings()
        for key, expected in changes.items():
            assert result[key] == expected, f"{key}: {result[key]!r} != {expected!r}"

    def test_untouched_fields_remain_default(self, settings_file):
        """S03: fields NOT in the save call keep their default values."""
        save_gui_settings({"mode": "fidelity"})
        result = load_gui_settings()
        assert result["source_language"] == DEFAULT_GUI_SETTINGS["source_language"]
        assert result["pass2_model"] == DEFAULT_GUI_SETTINGS["pass2_model"]


# ===================================================================
# S06: App upgraded, new keys added to schema
# ===================================================================

class TestS06ForwardCompatibility:
    """S06: Existing settings preserved. New keys get defaults."""

    def test_missing_key_gets_default(self, settings_file):
        """S06: file without a key that exists in defaults → key filled in."""
        # Simulate old file with fewer keys
        old_data = {
            "version": SETTINGS_SCHEMA_VERSION,
            "mode": "fidelity",
            "sensitivity": "conservative",
        }
        _write_json(settings_file, old_data)

        result = load_gui_settings()
        assert result["mode"] == "fidelity"          # preserved
        assert result["sensitivity"] == "conservative"  # preserved
        # Keys not in file get defaults
        assert result["pass1_pipeline"] == DEFAULT_GUI_SETTINGS["pass1_pipeline"]
        assert result["merge_strategy"] == DEFAULT_GUI_SETTINGS["merge_strategy"]


# ===================================================================
# S07: App downgraded, unknown keys in file
# ===================================================================

class TestS07BackwardCompatibility:
    """S07: Unknown keys silently ignored. Known keys preserved."""

    def test_extra_keys_ignored(self, settings_file):
        """S07: file with unknown keys → those keys don't appear in result."""
        data = DEFAULT_GUI_SETTINGS.copy()
        data["future_feature_xyz"] = "some_value"
        data["another_unknown"] = 42
        _write_json(settings_file, data)

        result = load_gui_settings()
        assert "future_feature_xyz" not in result
        assert "another_unknown" not in result
        assert result["mode"] == DEFAULT_GUI_SETTINGS["mode"]


# ===================================================================
# S08: Schema version mismatch — backup + migration
# ===================================================================

class TestS08VersionMismatchBackup:
    """S08: Old file backed up. User values migrated. Defaults for new keys."""

    def test_backup_created_on_version_mismatch(self, settings_file):
        """S08: version mismatch triggers backup before migration."""
        old_data = {
            "version": "0.9.0",
            "mode": "fidelity",
            "sensitivity": "conservative",
        }
        _write_json(settings_file, old_data)

        result = load_gui_settings()

        # Backup should exist
        backup = settings_file.with_suffix(".json.bak.1")
        assert backup.exists(), "Backup file not created on version mismatch"

        # Backup contains original data
        backup_data = _read_json(backup)
        assert backup_data["version"] == "0.9.0"
        assert backup_data["mode"] == "fidelity"

    def test_user_values_migrated_forward(self, settings_file):
        """S08: known keys from old version preserved in migrated result."""
        old_data = {
            "version": "0.5.0",
            "mode": "fast",
            "sensitivity": "balanced",
            "pass1_pipeline": "faster",
        }
        _write_json(settings_file, old_data)

        result = load_gui_settings()
        assert result["mode"] == "fast"             # migrated
        assert result["sensitivity"] == "balanced"   # migrated
        assert result["pass1_pipeline"] == "faster"  # migrated
        # New keys get defaults
        assert result["pass2_model"] == DEFAULT_GUI_SETTINGS["pass2_model"]
        # Version stamp updated
        assert result["version"] == SETTINGS_SCHEMA_VERSION

    def test_version_mismatch_does_not_destroy_data(self, settings_file):
        """S08: original file content recoverable from backup after mismatch."""
        original = {"version": "0.1.0", "mode": "fidelity", "custom_old_key": "val"}
        _write_json(settings_file, original)

        load_gui_settings()

        backup = settings_file.with_suffix(".json.bak.1")
        recovered = _read_json(backup)
        assert recovered == original


# ===================================================================
# S09: Corrupt JSON — corruption recovery
# ===================================================================

class TestS09CorruptionRecovery:
    """S09: Defaults used. Warning logged. No crash."""

    def test_corrupt_json_returns_defaults(self, settings_file):
        """S09: invalid JSON file → defaults returned, no exception raised."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("{corrupt json!!! not valid", encoding="utf-8")

        result = load_gui_settings()
        assert result == DEFAULT_GUI_SETTINGS

    def test_corrupt_json_creates_backup(self, settings_file):
        """S09: corrupt file is backed up before returning defaults."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        corrupt_content = "{{{{not json at all!!!}}}}"
        settings_file.write_text(corrupt_content, encoding="utf-8")

        load_gui_settings()

        backup = settings_file.with_suffix(".json.bak.1")
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == corrupt_content

    def test_empty_file_returns_defaults(self, settings_file):
        """S09: empty file is treated as corrupt JSON."""
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text("", encoding="utf-8")

        result = load_gui_settings()
        assert result == DEFAULT_GUI_SETTINGS


# ===================================================================
# S10: Settings file read-only — save fails gracefully
# ===================================================================

class TestS10ReadOnlyFile:
    """S10: Load succeeds. Save fails silently, logs error."""

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows permission model")
    def test_save_returns_false_on_readonly(self, settings_file):
        """S10: save to read-only file returns False, no exception."""
        save_gui_settings({"mode": "fast"})  # create file first
        # Make read-only
        os.chmod(settings_file, stat.S_IREAD)
        try:
            result = save_gui_settings({"mode": "fidelity"})
            assert result is False
        finally:
            os.chmod(settings_file, stat.S_IWRITE | stat.S_IREAD)


# ===================================================================
# S11: Settings directory missing — auto-created
# ===================================================================

class TestS11DirectoryCreation:
    """S11: Directory created automatically on first save."""

    def test_save_creates_parent_dirs(self, settings_file):
        """S11: save_gui_settings creates the full directory tree."""
        assert not settings_file.parent.exists()
        result = save_gui_settings({"mode": "fast"})
        assert result is True
        assert settings_file.exists()
        assert settings_file.parent.is_dir()


# ===================================================================
# S12: User deletes settings file
# ===================================================================

class TestS12DeletedFile:
    """S12: Defaults on next launch (same as S01)."""

    def test_deleted_file_returns_defaults(self, settings_file):
        """S12: save, delete file, load → defaults returned."""
        save_gui_settings({"mode": "fidelity"})
        assert settings_file.exists()

        settings_file.unlink()
        assert not settings_file.exists()

        result = load_gui_settings()
        assert result == DEFAULT_GUI_SETTINGS


# ===================================================================
# S13: Disk full / save failure
# ===================================================================

class TestS13SaveFailure:
    """S13: Form works. Error logged. No crash."""

    def test_save_returns_false_on_exception(self, settings_file, monkeypatch):
        """S13: if write raises, save returns False without crashing."""
        # Patch open to raise IOError
        original_open = open

        def failing_open(path, *args, **kwargs):
            if ".json.tmp" in str(path):
                raise IOError("Simulated disk full")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr("builtins.open", failing_open)
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        result = save_gui_settings({"mode": "fast"})
        assert result is False


# ===================================================================
# S20: Boolean checkbox roundtrip
# ===================================================================

class TestS20BooleanRoundtrip:
    """S20: true/false survives JSON serialization cycle."""

    @pytest.mark.parametrize("field,value", [
        ("model_override_enabled", True),
        ("model_override_enabled", False),
        ("output_to_source", True),
        ("output_to_source", False),
        ("debug_logging", True),
        ("keep_temp", True),
        ("accept_cpu_mode", True),
        ("async_processing", True),
        ("pass2_enabled", True),
        ("pass2_enabled", False),
    ])
    def test_boolean_roundtrip(self, settings_file, field, value):
        """S20: boolean values survive save → file → load cycle."""
        save_gui_settings({field: value})
        result = load_gui_settings()
        assert result[field] is value
        assert type(result[field]) is bool


# ===================================================================
# S21 + S22: outputDir / tempDir handling
# ===================================================================

class TestS21S22DirectoryFields:
    """S21: Empty strings preserved. S22: Custom paths persisted."""

    def test_empty_output_dir_preserved(self, settings_file):
        """S21: empty outputDir saved and loaded as empty string."""
        save_gui_settings({"output_dir": ""})
        result = load_gui_settings()
        assert result["output_dir"] == ""

    def test_custom_output_dir_persisted(self, settings_file):
        """S22: custom output path survives roundtrip."""
        custom = "C:\\Users\\Test\\Videos\\Output"
        save_gui_settings({"output_dir": custom})
        result = load_gui_settings()
        assert result["output_dir"] == custom

    def test_custom_temp_dir_persisted(self, settings_file):
        """S22: custom temp path survives roundtrip."""
        custom = "/tmp/whisperjav_custom"
        save_gui_settings({"temp_dir": custom})
        result = load_gui_settings()
        assert result["temp_dir"] == custom


# ===================================================================
# S23: Concurrent access — last-write-wins
# ===================================================================

class TestS23ConcurrentAccess:
    """S23: Last-write-wins. No crash or corruption."""

    def test_sequential_saves_last_wins(self, settings_file):
        """S23: two saves in sequence → second value wins."""
        save_gui_settings({"mode": "fast"})
        save_gui_settings({"mode": "fidelity"})
        result = load_gui_settings()
        assert result["mode"] == "fidelity"


# ===================================================================
# S24–S26: Platform path resolution
# ===================================================================

class TestS24S25S26PlatformPaths:
    """S24/S25/S26: platform-specific settings paths."""

    def test_windows_path(self, monkeypatch):
        """S24: Windows → %APPDATA%\\WhisperJAV\\gui_settings.json."""
        # Use module-level get_gui_settings_path (original, not monkeypatched)
        monkeypatch.setattr("whisperjav.settings.gui_settings.sys.platform", "win32")
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")

        path = get_gui_settings_path()
        assert "WhisperJAV" in str(path)
        assert str(path).endswith("gui_settings.json")
        assert "AppData" in str(path) or "Roaming" in str(path)

    def test_darwin_path(self, monkeypatch):
        """S25: macOS → ~/Library/Application Support/WhisperJAV/..."""
        monkeypatch.setattr("whisperjav.settings.gui_settings.sys.platform", "darwin")

        path = get_gui_settings_path()
        path_str = str(path)
        assert "WhisperJAV" in path_str
        assert "Application Support" in path_str or "Library" in path_str

    def test_linux_path(self, monkeypatch):
        """S26: Linux → ~/.config/WhisperJAV/gui_settings.json."""
        monkeypatch.setattr("whisperjav.settings.gui_settings.sys.platform", "linux")

        path = get_gui_settings_path()
        path_str = str(path)
        assert "WhisperJAV" in path_str
        assert ".config" in path_str


# ===================================================================
# S27: CLI isolation — GUI settings not imported by main.py
# ===================================================================

class TestS27CLIIsolation:
    """S27: CLI has own resolution chain. GUI settings are GUI-only."""

    def test_main_py_does_not_import_gui_settings(self):
        """S27: whisperjav/main.py must NOT import from settings.gui_settings.

        If it did, CLI runs would be influenced by GUI preferences.
        """
        main_py = Path(__file__).resolve().parents[1] / "whisperjav" / "main.py"
        content = main_py.read_text(encoding="utf-8")
        assert "gui_settings" not in content, (
            "main.py imports gui_settings — CLI would read GUI preferences"
        )
        assert "from whisperjav.settings" not in content, (
            "main.py imports from whisperjav.settings — CLI isolation violated"
        )


# ===================================================================
# S28: snake_case ↔ camelCase mapping completeness + roundtrip
# ===================================================================

class TestS28MappingFidelity:
    """S28: No key drift after backend→frontend→backend cycle."""

    def test_mapping_covers_all_settings_keys(self):
        """S28: every key in DEFAULT_GUI_SETTINGS (except metadata) has a
        camelCase mapping in the API layer."""
        # Import the mapping from api.py
        # We test the mapping dict directly to avoid needing pywebview
        from whisperjav.webview_gui.api import WhisperJAVAPI

        mapping = WhisperJAVAPI._GUI_SETTINGS_MAP
        settings_keys = {
            k for k in DEFAULT_GUI_SETTINGS if k not in ("version", "_comment")
        }
        mapped_keys = set(mapping.keys())

        missing = settings_keys - mapped_keys
        assert not missing, (
            f"DEFAULT_GUI_SETTINGS keys missing from _GUI_SETTINGS_MAP: {missing}"
        )

    def test_reverse_mapping_is_bijective(self):
        """S28: forward and reverse maps are consistent (no duplicates)."""
        from whisperjav.webview_gui.api import WhisperJAVAPI

        fwd = WhisperJAVAPI._GUI_SETTINGS_MAP
        rev = WhisperJAVAPI._GUI_SETTINGS_MAP_REV

        # Forward map: no duplicate camelCase values
        camel_values = list(fwd.values())
        assert len(camel_values) == len(set(camel_values)), (
            "Duplicate camelCase values in _GUI_SETTINGS_MAP"
        )

        # Reverse map roundtrip
        for snake, camel in fwd.items():
            assert rev[camel] == snake, (
                f"Reverse mapping broken: {camel!r} → {rev.get(camel)!r}, "
                f"expected {snake!r}"
            )

    def test_camel_to_snake_roundtrip_via_save_load(self, settings_file):
        """S28: simulate frontend→API→backend→API→frontend cycle."""
        from whisperjav.webview_gui.api import WhisperJAVAPI

        fwd = WhisperJAVAPI._GUI_SETTINGS_MAP
        rev = WhisperJAVAPI._GUI_SETTINGS_MAP_REV

        # Simulate frontend sending camelCase
        frontend_data = {
            "mode": "fidelity",
            "sourceLanguage": "korean",
            "pass1Pipeline": "faster",
            "pass2Enabled": True,
            "mergeStrategy": "smart_merge",
        }

        # API save: camelCase → snake_case → file
        backend_data = {}
        for camel_key, value in frontend_data.items():
            snake_key = rev.get(camel_key)
            if snake_key:
                backend_data[snake_key] = value
        save_gui_settings(backend_data)

        # API load: file → snake_case → camelCase
        loaded = load_gui_settings()
        gui_out = {}
        for snake, camel in fwd.items():
            if snake in loaded:
                gui_out[camel] = loaded[snake]

        # Roundtrip: original values must come back
        for key, expected in frontend_data.items():
            assert gui_out[key] == expected, (
                f"Roundtrip failed for {key!r}: {gui_out.get(key)!r} != {expected!r}"
            )


# ===================================================================
# Backup rotation tests (C1 best practices)
# ===================================================================

class TestBackupRotation:
    """Backup rotation: keeps last N, oldest deleted."""

    def test_single_backup_created(self, settings_file):
        """First backup creates .bak.1."""
        _write_json(settings_file, {"version": "0.1.0", "mode": "fast"})
        backup = _rotate_backups(settings_file)
        assert backup is not None
        assert backup.name.endswith(".bak.1")
        assert backup.exists()

    def test_rotation_shifts_numbers(self, settings_file):
        """Multiple backups: .bak.1 → .bak.2 → .bak.3."""
        _write_json(settings_file, {"version": "0.1.0", "data": "first"})
        _rotate_backups(settings_file)

        _write_json(settings_file, {"version": "0.2.0", "data": "second"})
        _rotate_backups(settings_file)

        _write_json(settings_file, {"version": "0.3.0", "data": "third"})
        _rotate_backups(settings_file)

        bak1 = settings_file.with_suffix(".json.bak.1")
        bak2 = settings_file.with_suffix(".json.bak.2")
        bak3 = settings_file.with_suffix(".json.bak.3")

        assert bak1.exists()
        assert bak2.exists()
        assert bak3.exists()

        # .bak.1 = newest (third), .bak.3 = oldest (first)
        assert _read_json(bak1)["data"] == "third"
        assert _read_json(bak2)["data"] == "second"
        assert _read_json(bak3)["data"] == "first"

    def test_oldest_backup_deleted_when_full(self, settings_file):
        """When max backups reached, oldest is deleted on next rotation."""
        # Create _MAX_BACKUPS + 1 rotations
        for i in range(_MAX_BACKUPS + 1):
            _write_json(settings_file, {"version": f"0.{i}.0", "data": f"v{i}"})
            _rotate_backups(settings_file)

        # Only _MAX_BACKUPS should exist
        for i in range(1, _MAX_BACKUPS + 1):
            bak = settings_file.with_suffix(f".json.bak.{i}")
            assert bak.exists(), f".bak.{i} should exist"

        # One beyond max should NOT exist
        overflow = settings_file.with_suffix(f".json.bak.{_MAX_BACKUPS + 1}")
        assert not overflow.exists(), f".bak.{_MAX_BACKUPS + 1} should not exist"

    def test_no_backup_when_file_missing(self, settings_file):
        """Rotate on non-existent file returns None."""
        assert _rotate_backups(settings_file) is None


# ===================================================================
# Schema migration tests
# ===================================================================

class TestSchemaMigration:
    """Migration preserves known keys, fills new keys with defaults."""

    def test_migrate_preserves_known_keys(self):
        """Known keys from old schema carried forward."""
        old = {
            "version": "0.5.0",
            "mode": "fast",
            "sensitivity": "conservative",
        }
        result = _migrate_settings(old)
        assert result["mode"] == "fast"
        assert result["sensitivity"] == "conservative"
        assert result["version"] == SETTINGS_SCHEMA_VERSION

    def test_migrate_fills_missing_with_defaults(self):
        """Keys absent from old settings get default values."""
        old = {"version": "0.5.0", "mode": "fast"}
        result = _migrate_settings(old)
        assert result["pass1_pipeline"] == DEFAULT_GUI_SETTINGS["pass1_pipeline"]
        assert result["merge_strategy"] == DEFAULT_GUI_SETTINGS["merge_strategy"]

    def test_migrate_drops_removed_keys(self):
        """Keys not in current schema are dropped."""
        old = {
            "version": "0.5.0",
            "mode": "fast",
            "removed_feature": "obsolete_value",
        }
        result = _migrate_settings(old)
        assert "removed_feature" not in result


# ===================================================================
# Atomic write tests
# ===================================================================

class TestAtomicWrite:
    """Atomic writes: .tmp file should not persist after successful save."""

    def test_no_tmp_file_after_save(self, settings_file):
        """After successful save, .json.tmp should not exist."""
        save_gui_settings({"mode": "fast"})
        tmp = settings_file.with_suffix(".json.tmp")
        assert not tmp.exists(), ".tmp file left behind after save"

    def test_save_creates_valid_json(self, settings_file):
        """Saved file must be valid, parseable JSON."""
        save_gui_settings({"mode": "fidelity", "pass2_enabled": True})
        data = _read_json(settings_file)
        assert isinstance(data, dict)
        assert data["mode"] == "fidelity"
        assert data["pass2_enabled"] is True


# ===================================================================
# Partial save (merge behaviour)
# ===================================================================

class TestPartialSave:
    """save_gui_settings merges partial updates, preserving untouched keys."""

    def test_partial_update_preserves_existing(self, settings_file):
        """Save 2 keys, then save 1 different key — all 3 preserved."""
        save_gui_settings({"mode": "fidelity", "sensitivity": "conservative"})
        save_gui_settings({"pass1_pipeline": "faster"})

        result = load_gui_settings()
        assert result["mode"] == "fidelity"
        assert result["sensitivity"] == "conservative"
        assert result["pass1_pipeline"] == "faster"

    def test_version_cannot_be_overridden_by_caller(self, settings_file):
        """Callers cannot change version or _comment via save."""
        save_gui_settings({"version": "99.0.0", "_comment": "hacked"})
        data = _read_json(settings_file)
        assert data["version"] == SETTINGS_SCHEMA_VERSION
        assert data["_comment"] == DEFAULT_GUI_SETTINGS["_comment"]


# ===================================================================
# Completeness checks
# ===================================================================

class TestCompleteness:
    """Meta-tests ensuring the schema is complete and consistent."""

    def test_all_settings_keys_are_strings(self):
        """Every key in DEFAULT_GUI_SETTINGS is a string."""
        for key in DEFAULT_GUI_SETTINGS:
            assert isinstance(key, str), f"Non-string key: {key!r}"

    def test_no_none_default_values(self):
        """No default value should be None — use empty string or False instead."""
        for key, value in DEFAULT_GUI_SETTINGS.items():
            assert value is not None, f"DEFAULT_GUI_SETTINGS[{key!r}] is None"

    def test_settings_count(self):
        """Guard against accidental key removal.

        If you add or remove a settings key, update this count.
        """
        data_keys = {
            k for k in DEFAULT_GUI_SETTINGS if k not in ("version", "_comment")
        }
        # 13 Tab 1 keys + 14 Tab 3 keys = 27
        assert len(data_keys) == 27, (
            f"Expected 27 settings keys, got {len(data_keys)}: {data_keys}"
        )
