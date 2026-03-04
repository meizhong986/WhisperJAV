"""
Ensemble parameter presets for WhisperJAV.

Stores named preset configurations as individual JSON files in a platform-
specific presets directory.  Each preset captures a complete pass configuration
(pipeline, sensitivity, custom params, etc.) that can be loaded into either
pass slot in the Ensemble tab.

Storage layout:
    %APPDATA%/WhisperJAV/presets/   (Windows)
    ~/Library/Application Support/WhisperJAV/presets/   (macOS)
    ~/.config/WhisperJAV/presets/    (Linux)

Each file:  <sanitized-name>.json
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from .gui_settings import get_gui_settings_path

logger = logging.getLogger(__name__)

# Maximum length for a preset name (before sanitization)
_MAX_NAME_LENGTH = 80

# Preset schema version — bump when the preset fields change
PRESET_SCHEMA_VERSION = "1.0.0"

# Required keys in a valid preset (everything else is optional)
_REQUIRED_KEYS = {"name", "pipeline"}


def get_presets_dir() -> Path:
    """Return the platform-specific presets directory.

    Derived from the GUI settings path — same parent, ``presets/`` subdirectory.
    """
    return get_gui_settings_path().parent / "presets"


def _sanitize_filename(name: str) -> str:
    """Convert a human-readable preset name into a safe filename stem.

    Rules:
    - Strip leading/trailing whitespace
    - Replace characters not in [a-zA-Z0-9 _()-] with underscore
    - Collapse multiple underscores / spaces
    - Truncate to _MAX_NAME_LENGTH characters
    - Ensure non-empty (fallback to 'preset')

    Unicode characters (e.g., Japanese) are preserved if they are word
    characters; only filesystem-unsafe symbols are replaced.
    """
    name = name.strip()
    # Replace filesystem-unsafe characters: \ / : * ? " < > |
    name = re.sub(r'[\\/:*?"<>|]', '_', name)
    # Collapse whitespace and underscores
    name = re.sub(r'[\s_]+', ' ', name).strip()
    # Truncate
    name = name[:_MAX_NAME_LENGTH]
    # Fallback
    if not name:
        name = "preset"
    return name


def _preset_path(name: str) -> Path:
    """Return the full path for a preset with the given name."""
    return get_presets_dir() / f"{_sanitize_filename(name)}.json"


def list_presets() -> List[Dict]:
    """List all saved presets.

    Returns a list of dicts with summary info (name, pipeline, sensitivity,
    created_at) sorted by name.  Returns an empty list if the directory
    doesn't exist or is empty.
    """
    presets_dir = get_presets_dir()
    if not presets_dir.exists():
        return []

    result = []
    for path in sorted(presets_dir.glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "name" not in data:
                logger.warning("Skipping invalid preset file: %s", path.name)
                continue
            result.append({
                "name": data["name"],
                "pipeline": data.get("pipeline", ""),
                "sensitivity": data.get("sensitivity", ""),
                "model": data.get("model", ""),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
            })
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to read preset %s: %s", path.name, exc)

    return result


def load_preset(name: str) -> Optional[Dict]:
    """Load a preset by name.

    Returns the full preset dict, or None if not found / corrupt.
    """
    path = _preset_path(name)
    if not path.exists():
        logger.warning("Preset not found: %s", path)
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "name" not in data:
            logger.warning("Invalid preset structure in %s", path)
            return None
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to load preset %s: %s", name, exc)
        return None


def save_preset(name: str, data: Dict) -> bool:
    """Save a preset.

    *data* should contain the pass configuration fields.  The ``name`` key
    is set/overridden to *name*.  Timestamps are managed automatically.

    Uses atomic write (.tmp → rename) for crash safety.
    Returns True on success.
    """
    if not name or not name.strip():
        logger.error("Cannot save preset with empty name")
        return False

    path = _preset_path(name)

    try:
        presets_dir = get_presets_dir()
        presets_dir.mkdir(parents=True, exist_ok=True)

        # Set metadata
        now = time.strftime("%Y-%m-%dT%H:%M:%S")
        data["name"] = name.strip()
        data["schema_version"] = PRESET_SCHEMA_VERSION
        data["updated_at"] = now
        if "created_at" not in data:
            # Preserve original creation time on update
            existing = load_preset(name)
            if existing and "created_at" in existing:
                data["created_at"] = existing["created_at"]
            else:
                data["created_at"] = now

        # Atomic write
        tmp_path = path.with_suffix(".json.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())

        # Windows: rename fails if target exists
        if path.exists():
            path.unlink()
        tmp_path.rename(path)

        logger.debug("Preset saved: %s → %s", name, path)
        return True

    except Exception as exc:
        logger.error("Failed to save preset %s: %s", name, exc)
        # Clean up tmp
        tmp_path = path.with_suffix(".json.tmp") if 'path' in dir() else None
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        return False


def delete_preset(name: str) -> bool:
    """Delete a preset by name.  Returns True if deleted, False otherwise."""
    path = _preset_path(name)
    if not path.exists():
        logger.warning("Cannot delete preset — not found: %s", name)
        return False

    try:
        path.unlink()
        logger.debug("Preset deleted: %s", name)
        return True
    except Exception as exc:
        logger.error("Failed to delete preset %s: %s", name, exc)
        return False


def rename_preset(old_name: str, new_name: str) -> bool:
    """Rename a preset.  Returns True on success."""
    if not new_name or not new_name.strip():
        logger.error("Cannot rename preset to empty name")
        return False

    old_path = _preset_path(old_name)
    new_path = _preset_path(new_name)

    if not old_path.exists():
        logger.warning("Cannot rename preset — not found: %s", old_name)
        return False

    if new_path.exists():
        logger.warning("Cannot rename preset — target exists: %s", new_name)
        return False

    try:
        data = load_preset(old_name)
        if data is None:
            return False
        data["name"] = new_name.strip()
        data["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        if save_preset(new_name, data):
            old_path.unlink()
            return True
        return False
    except Exception as exc:
        logger.error("Failed to rename preset %s → %s: %s", old_name, new_name, exc)
        return False
