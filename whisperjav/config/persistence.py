"""
Configuration Persistence for WhisperJAV Configuration System v2.0.

Save and load custom pipeline configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ConfigValidationError


def get_config_dir() -> Path:
    """
    Get the user configuration directory.

    Returns:
        Path to ~/.whisperjav/configs/
    """
    config_dir = Path.home() / ".whisperjav" / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def save_config(
    config: Dict[str, Any],
    name: str,
    config_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Path:
    """
    Save a configuration to disk.

    Args:
        config: Configuration dictionary to save
        name: Name for the configuration (used as filename)
        config_dir: Optional custom directory (defaults to user config dir)
        overwrite: Whether to overwrite existing file

    Returns:
        Path to saved configuration file

    Raises:
        ConfigValidationError: If name is invalid or file exists and overwrite=False
    """
    # Validate name
    if not name or not name.replace("_", "").replace("-", "").isalnum():
        raise ConfigValidationError(
            "Config name must be alphanumeric (with optional underscores/hyphens)",
            field="name",
            value=name
        )

    # Get directory
    if config_dir is None:
        config_dir = get_config_dir()
    else:
        config_dir = Path(config_dir)
        config_dir.mkdir(parents=True, exist_ok=True)

    # Build path
    file_path = config_dir / f"{name}.json"

    # Check for existing
    if file_path.exists() and not overwrite:
        raise ConfigValidationError(
            f"Configuration '{name}' already exists. Use overwrite=True to replace.",
            field="name",
            value=name
        )

    # Save
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    return file_path


def load_config(
    name: str,
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load a configuration from disk.

    Args:
        name: Name of the configuration to load
        config_dir: Optional custom directory (defaults to user config dir)

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If configuration doesn't exist
    """
    # Get directory
    if config_dir is None:
        config_dir = get_config_dir()
    else:
        config_dir = Path(config_dir)

    # Build path
    file_path = config_dir / f"{name}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"Configuration '{name}' not found at {file_path}")

    # Load
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_configs(config_dir: Optional[Path] = None) -> List[str]:
    """
    List available saved configurations.

    Args:
        config_dir: Optional custom directory (defaults to user config dir)

    Returns:
        List of configuration names (without .json extension)
    """
    if config_dir is None:
        config_dir = get_config_dir()
    else:
        config_dir = Path(config_dir)

    if not config_dir.exists():
        return []

    return sorted([
        f.stem for f in config_dir.glob("*.json")
        if f.is_file()
    ])


def delete_config(
    name: str,
    config_dir: Optional[Path] = None
) -> bool:
    """
    Delete a saved configuration.

    Args:
        name: Name of the configuration to delete
        config_dir: Optional custom directory (defaults to user config dir)

    Returns:
        True if deleted, False if didn't exist
    """
    if config_dir is None:
        config_dir = get_config_dir()
    else:
        config_dir = Path(config_dir)

    file_path = config_dir / f"{name}.json"

    if file_path.exists():
        file_path.unlink()
        return True

    return False


def config_exists(
    name: str,
    config_dir: Optional[Path] = None
) -> bool:
    """
    Check if a configuration exists.

    Args:
        name: Name of the configuration
        config_dir: Optional custom directory (defaults to user config dir)

    Returns:
        True if configuration exists
    """
    if config_dir is None:
        config_dir = get_config_dir()
    else:
        config_dir = Path(config_dir)

    file_path = config_dir / f"{name}.json"
    return file_path.exists()
