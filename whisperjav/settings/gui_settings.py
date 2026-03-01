"""
GUI settings persistence for WhisperJAV.

Stores user preferences so the GUI form state survives across sessions.
Follows the same pattern as whisperjav/translate/settings.py.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SETTINGS_SCHEMA_VERSION = "1.0.0"

# Default settings matching the GUI form's initial values.
# Keys use snake_case (backend convention); the API layer converts to/from
# camelCase for the JavaScript frontend.
DEFAULT_GUI_SETTINGS = {
    "version": SETTINGS_SCHEMA_VERSION,
    "_comment": "WhisperJAV GUI user preferences",

    # Tab 1 — Transcription
    "mode": "faster",
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

    # Tab 3 — Ensemble
    "pass1_pipeline": "balanced",
    "pass1_sensitivity": "balanced",
    "pass1_scene_detector": "auditok",
    "pass1_speech_enhancer": "none",
    "pass1_speech_segmenter": "silero",
    "pass1_model": "",
    "pass2_enabled": False,
    "pass2_pipeline": "fast",
    "pass2_sensitivity": "balanced",
    "pass2_scene_detector": "auditok",
    "pass2_speech_enhancer": "none",
    "pass2_speech_segmenter": "silero",
    "pass2_model": "",
    "merge_strategy": "pass1_primary",
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


def load_gui_settings() -> dict:
    """
    Load GUI settings from disk.

    Returns defaults on missing file, version mismatch, or corruption.
    """
    settings_path = get_gui_settings_path()

    if not settings_path.exists():
        logger.debug("No GUI settings file found, using defaults")
        return DEFAULT_GUI_SETTINGS.copy()

    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)

        if settings.get("version") != SETTINGS_SCHEMA_VERSION:
            logger.warning(
                "GUI settings version mismatch (got %s, expected %s), using defaults",
                settings.get("version"),
                SETTINGS_SCHEMA_VERSION,
            )
            return DEFAULT_GUI_SETTINGS.copy()

        # Merge with defaults so newly added keys are present
        merged = DEFAULT_GUI_SETTINGS.copy()
        for key, value in settings.items():
            if key in merged:
                merged[key] = value
        return merged

    except json.JSONDecodeError as exc:
        logger.warning("Corrupt GUI settings file (JSON error: %s), using defaults", exc)
        return DEFAULT_GUI_SETTINGS.copy()
    except Exception as exc:
        logger.warning("Failed to load GUI settings: %s, using defaults", exc)
        return DEFAULT_GUI_SETTINGS.copy()


def save_gui_settings(settings: dict) -> bool:
    """
    Save GUI settings to disk.

    Merges *settings* into the existing file so that callers don't need to
    supply every key.  Returns True on success.
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
        with open(settings_path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

        logger.debug("GUI settings saved to %s", settings_path)
        return True

    except Exception as exc:
        logger.error("Failed to save GUI settings: %s", exc)
        return False
