"""
GUI settings persistence for WhisperJAV.

Stores user preferences so the GUI form state survives across sessions.
Follows the same pattern as whisperjav/translate/settings.py.

Safety features:
- Atomic writes via .tmp file (crash mid-write won't corrupt)
- Backup rotation before overwrite on version mismatch or corruption (keeps last 3)
- Schema migration: preserves user values across version changes when possible
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SETTINGS_SCHEMA_VERSION = "1.0.0"
_MAX_BACKUPS = 3

# Default settings matching the GUI form's initial values.
# Keys use snake_case (backend convention); the API layer converts to/from
# camelCase for the JavaScript frontend.
#
# IMPORTANT: These values MUST match the `selected` attributes in
# whisperjav/webview_gui/assets/index.html.  On first launch (no settings
# file), loadFromBackend() returns these defaults and applies them to the
# form — a mismatch silently overrides the HTML-designed defaults.
DEFAULT_GUI_SETTINGS = {
    "version": SETTINGS_SCHEMA_VERSION,
    "_comment": "WhisperJAV GUI user preferences",

    # Tab 1 — Transcription  (values MUST match index.html `selected` attrs)
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

    # Tab 3 — Ensemble  (values MUST match index.html `selected` attrs)
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
    "pass1_preset": "",
    "pass2_preset": "",
}


def get_gui_settings_path() -> Path:
    """Get platform-specific path for GUI settings file."""
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA", "~"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path.home() / ".config"

    return (base / "WhisperJAV" / "gui_settings.json").expanduser()


def _rotate_backups(settings_path: Path) -> Optional[Path]:
    """
    Create a numbered backup of *settings_path* and rotate old backups.

    Keeps the last ``_MAX_BACKUPS`` copies (.bak.1 = newest, .bak.N = oldest).
    Returns the backup path on success, None on failure.
    """
    if not settings_path.exists():
        return None

    try:
        # Rotate: .bak.3 → delete, .bak.2 → .bak.3, .bak.1 → .bak.2
        for i in range(_MAX_BACKUPS, 1, -1):
            older = settings_path.with_suffix(f".json.bak.{i}")
            newer = settings_path.with_suffix(f".json.bak.{i - 1}")
            if newer.exists():
                if older.exists():
                    older.unlink()
                newer.rename(older)

        # Current → .bak.1
        backup_path = settings_path.with_suffix(".json.bak.1")
        shutil.copy2(str(settings_path), str(backup_path))
        return backup_path

    except Exception as exc:
        logger.warning("Failed to create settings backup: %s", exc)
        return None


def _migrate_settings(settings: dict) -> dict:
    """
    Attempt to migrate settings from an older schema version.

    Instead of discarding all user values on version mismatch, we preserve
    every key that still exists in the current schema.  Only truly removed
    keys are dropped.  Returns a merged dict with the current version stamp.
    """
    merged = DEFAULT_GUI_SETTINGS.copy()
    preserved = 0
    for key, value in settings.items():
        if key in ("version", "_comment"):
            continue
        if key in merged:
            merged[key] = value
            preserved += 1

    logger.info(
        "Migrated GUI settings from v%s → v%s (%d/%d keys preserved)",
        settings.get("version", "?"),
        SETTINGS_SCHEMA_VERSION,
        preserved,
        len([k for k in merged if k not in ("version", "_comment")]),
    )
    return merged


def load_gui_settings() -> dict:
    """
    Load GUI settings from disk.

    Behaviour on error:
    - Missing file → returns defaults (no backup needed).
    - Version mismatch → backs up old file, migrates user values forward.
    - Corrupt JSON → backs up old file, returns defaults.
    - Other I/O error → returns defaults (file untouched).
    """
    settings_path = get_gui_settings_path()

    if not settings_path.exists():
        logger.debug("No GUI settings file found, using defaults")
        return DEFAULT_GUI_SETTINGS.copy()

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        if settings.get("version") != SETTINGS_SCHEMA_VERSION:
            backup = _rotate_backups(settings_path)
            if backup:
                logger.warning(
                    "GUI settings version mismatch (got %s, expected %s). "
                    "Backed up to: %s",
                    settings.get("version"),
                    SETTINGS_SCHEMA_VERSION,
                    backup,
                )
            else:
                logger.warning(
                    "GUI settings version mismatch (got %s, expected %s), "
                    "migrating with defaults for new keys",
                    settings.get("version"),
                    SETTINGS_SCHEMA_VERSION,
                )
            return _migrate_settings(settings)

        # Merge with defaults so newly added keys are present
        merged = DEFAULT_GUI_SETTINGS.copy()
        for key, value in settings.items():
            if key in merged:
                merged[key] = value
        return merged

    except json.JSONDecodeError as exc:
        backup = _rotate_backups(settings_path)
        if backup:
            logger.warning(
                "Corrupt GUI settings file (JSON error: %s). Backed up to: %s",
                exc, backup,
            )
        else:
            logger.warning(
                "Corrupt GUI settings file (JSON error: %s), using defaults", exc,
            )
        return DEFAULT_GUI_SETTINGS.copy()

    except Exception as exc:
        logger.warning("Failed to load GUI settings: %s, using defaults", exc)
        return DEFAULT_GUI_SETTINGS.copy()


def save_gui_settings(settings: dict) -> bool:
    """
    Save GUI settings to disk.

    Merges *settings* into the existing file so that callers don't need to
    supply every key.  Uses atomic write (write to .tmp, then rename) so a
    crash mid-write won't corrupt the settings file.  Returns True on success.
    """
    settings_path = get_gui_settings_path()

    try:
        # Load existing to preserve keys the caller didn't touch
        existing = load_gui_settings()
        for key, value in settings.items():
            if key not in ("version", "_comment"):
                existing[key] = value

        # Ensure metadata
        existing["version"] = SETTINGS_SCHEMA_VERSION
        existing["_comment"] = DEFAULT_GUI_SETTINGS["_comment"]

        settings_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write: .tmp → rename
        tmp_path = settings_path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # On Windows, rename fails if target exists — remove first
        if settings_path.exists():
            settings_path.unlink()
        tmp_path.rename(settings_path)

        logger.debug("GUI settings saved to %s", settings_path)
        return True

    except Exception as exc:
        logger.error("Failed to save GUI settings: %s", exc)
        # Clean up .tmp if it was left behind
        tmp_path = settings_path.with_suffix(".json.tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False
