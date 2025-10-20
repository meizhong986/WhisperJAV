"""
Settings file management with precedence rules.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default settings with all configuration options
DEFAULT_SETTINGS = {
    'version': 'v1.0.0',
    '_comment': 'WhisperJAV-Translate user settings',
    'provider': 'deepseek',
    'model': None,
    'target_language': 'english',
    'tone': 'standard',
    'scene_threshold': 60.0,
    'max_batch_size': 30,
    'include_original': False,
    'preprocess_subtitles': True,
    'postprocess_translation': True,
    'save_preprocessed_subtitles': False,
    # instruction_urls removed; Gist-based defaults are handled in instructions.py
    'autosave': True,
    'movie_title': None,
    'movie_plot': None,
    'actress': None,
    'model_params': {
        'temperature': None,
        'top_p': None
    },
    'instructions_files': {}
}


def get_settings_path() -> Path:
    """Get platform-specific settings file path."""
    if sys.platform == 'win32':
        base = Path(os.getenv('APPDATA', '~'))
    elif sys.platform == 'darwin':
        base = Path.home() / 'Library' / 'Application Support'
    else:
        base = Path.home() / '.config'

    settings_path = base / 'WhisperJAV' / 'translate' / 'settings.json'
    return settings_path.expanduser()


def load_settings() -> dict:
    """Load settings from file, return defaults if not found."""
    settings_path = get_settings_path()

    if not settings_path.exists():
        logger.debug("No settings file found, using defaults")
        return DEFAULT_SETTINGS.copy()

    try:
        with open(settings_path, 'r', encoding='utf-8') as f:
            settings = json.load(f)

        # Version check
        if settings.get('version') != DEFAULT_SETTINGS['version']:
            logger.warning("Settings version mismatch, using defaults")
            return DEFAULT_SETTINGS.copy()

        logger.debug(f"Loaded settings from {settings_path}")
        return settings

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid settings file (JSON error): {e}")
        logger.warning("Using defaults")
        return DEFAULT_SETTINGS.copy()
    except Exception as e:
        logger.warning(f"Failed to load settings: {e}")
        logger.warning("Using defaults")
        return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> bool:
    """Save settings to file."""
    settings_path = get_settings_path()

    try:
        # Create directory if it doesn't exist
        settings_path.parent.mkdir(parents=True, exist_ok=True)

        # Save settings
        with open(settings_path, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Settings saved to {settings_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to save settings: {e}")
        return False


def create_default_settings() -> bool:
    """Create default settings file."""
    settings_path = get_settings_path()

    if settings_path.exists():
        logger.warning(f"Settings file already exists: {settings_path}")
        return False

    return save_settings(DEFAULT_SETTINGS.copy())


def resolve_config(cli_args, settings: Optional[dict] = None) -> dict:
    """
    Merge configuration from all sources with precedence.

    Precedence (highest to lowest):
    1. CLI flags (explicit user intent)
    2. Environment variables (session-specific)
    3. Settings file (user preferences)
    4. Built-in defaults (fallback)

    Args:
        cli_args: Parsed argparse Namespace
        settings: Loaded settings dict (optional)

    Returns:
        Merged configuration dict
    """
    # Start with defaults
    config = DEFAULT_SETTINGS.copy()

    # Apply settings file (if provided)
    if settings:
        for key, value in settings.items():
            if key not in ('version', '_comment'):
                config[key] = value

    # Apply CLI arguments (highest precedence)
    if cli_args:
        cli_dict = vars(cli_args)
        for key, value in cli_dict.items():
            # Skip None values from CLI (means flag wasn't used)
            if value is None:
                continue

            # Map CLI arg names to config keys
            if key == 'target':
                config['target_language'] = value
            elif key == 'scene_threshold':
                config['scene_threshold'] = value
            elif key == 'max_batch_size':
                config['max_batch_size'] = value
            elif key == 'provider':
                config['provider'] = value
            elif key == 'model':
                if value:  # Only set if model explicitly provided
                    config['model'] = value
            elif key == 'tone':
                config['tone'] = value
            elif key == 'temperature':
                config.setdefault('model_params', {}).update({'temperature': value})
            elif key == 'top_p':
                config.setdefault('model_params', {}).update({'top_p': value})
            elif key == 'movie_title':
                if value:
                    config['movie_title'] = value
            elif key == 'movie_plot':
                if value:
                    config['movie_plot'] = value
            elif key == 'actress':
                if value:
                    config['actress'] = value

    return config


def show_settings():
    """Display current settings to stdout."""
    settings_path = get_settings_path()
    settings = load_settings()

    print(f"Settings file: {settings_path}")
    print(f"Exists: {settings_path.exists()}")
    print("\nCurrent settings:")
    print(json.dumps(settings, indent=2))
