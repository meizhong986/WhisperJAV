"""
Sync Validation
===============

Validates that installation artifacts stay in sync with the registry:
- pyproject.toml optional-dependencies match registry
- requirements.txt matches registry (if present)

WHY THIS VALIDATION:
-------------------
Before this refactor, WhisperJAV had packages defined in 4+ places:
- pyproject.toml (optional-dependencies)
- install_windows.bat (pip install commands)
- install_linux.sh (pip install commands)
- post_install.py.template (requirements list)

These would drift apart over time, causing:
- Users getting different packages depending on install method
- Version conflicts between files
- Missing dependencies in some install paths

NOW:
- Registry (registry.py) is the SINGLE SOURCE OF TRUTH
- pyproject.toml MUST match what the registry defines
- This validation blocks merges that would cause drift

Usage:
    from whisperjav.installer.validation.sync import validate_pyproject_sync

    errors = validate_pyproject_sync()
    if errors:
        print("Validation failed!")
        for error in errors:
            print(f"  - {error}")

Author: Senior Architect
Date: 2026-01-26
"""

import sys
from pathlib import Path
from typing import Dict, List, Set

# Use tomllib (Python 3.11+) or tomli fallback
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None  # type: ignore


def _find_project_root() -> Path:
    """
    Find the project root directory.

    WHY: Validation may be run from different directories.
    We need to reliably find pyproject.toml.
    """
    # Start from this file's location and go up
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def validate_pyproject_sync(pyproject_path: Path = None) -> List[str]:
    """
    Validate pyproject.toml optional-dependencies match registry.

    WHY THIS CHECK:
    pyproject.toml defines what users get when they run:
        pip install whisperjav[cli]
        pip install whisperjav[gui]
        etc.

    If pyproject.toml doesn't match the registry, users get different
    packages than what the installer would give them.

    WHAT WE CHECK:
    1. All extras in registry exist in pyproject.toml
    2. All packages in each extra match

    Args:
        pyproject_path: Path to pyproject.toml (auto-detected if None)

    Returns:
        List of error messages (empty if valid)
    """
    # Import here to avoid circular imports
    from ..core.registry import generate_pyproject_extras

    if tomllib is None:
        return ["Cannot validate: tomllib/tomli not available (Python 3.11+ or pip install tomli)"]

    if pyproject_path is None:
        try:
            pyproject_path = _find_project_root() / "pyproject.toml"
        except FileNotFoundError as e:
            return [str(e)]

    if not pyproject_path.exists():
        return [f"pyproject.toml not found at {pyproject_path}"]

    # Load pyproject.toml
    try:
        with open(pyproject_path, "rb") as f:
            pyproject = tomllib.load(f)
    except Exception as e:
        return [f"Failed to parse pyproject.toml: {e}"]

    # Get actual extras from pyproject.toml
    actual = pyproject.get("project", {}).get("optional-dependencies", {})

    # Get expected extras from registry
    expected = generate_pyproject_extras()

    errors = []

    # Check that expected extras exist
    for extra_name, expected_packages in expected.items():
        if extra_name not in actual:
            errors.append(f"Missing extra in pyproject.toml: [{extra_name}]")
            continue

        actual_packages = set(actual[extra_name])
        expected_set = set(expected_packages)

        # Check for missing packages
        missing = expected_set - actual_packages
        if missing:
            for pkg in sorted(missing):
                errors.append(f"Extra [{extra_name}] missing package: {pkg}")

        # Check for unexpected packages (in pyproject but not registry)
        extra_pkgs = actual_packages - expected_set
        if extra_pkgs:
            for pkg in sorted(extra_pkgs):
                errors.append(f"Extra [{extra_name}] has undeclared package: {pkg}")

    # Check for extras in pyproject.toml that aren't in registry
    #
    # KNOWN COMPOSITE EXTRAS:
    # These are combinations of other extras, not duplicate package lists.
    # They're defined in pyproject.toml but not in the registry because
    # they reference other extras (e.g., "whisperjav[cli,gui,translate]")
    #
    KNOWN_COMPOSITE_EXTRAS = {
        "all",      # All extras combined
        "colab",    # Google Colab optimized
        "kaggle",   # Kaggle optimized
        "unix",     # Linux/macOS CLI focused
        "windows",  # Windows with GUI
        "dev",      # Development dependencies
        "test",     # Testing dependencies
        "docs",     # Documentation dependencies
    }

    actual_extras = set(actual.keys())
    expected_extras = set(expected.keys())
    unknown_extras = actual_extras - expected_extras - KNOWN_COMPOSITE_EXTRAS
    for extra in sorted(unknown_extras):
        errors.append(f"Unknown extra in pyproject.toml: [{extra}]")

    return errors


def validate_requirements_sync(requirements_path: Path = None) -> List[str]:
    """
    Validate requirements.txt matches registry (if present).

    WHY THIS CHECK:
    requirements.txt may be used by some deployment methods.
    It should match what the registry defines.

    NOTE: This is optional - not all projects have requirements.txt.

    Args:
        requirements_path: Path to requirements.txt (auto-detected if None)

    Returns:
        List of error messages (empty if valid or file doesn't exist)
    """
    # Import here to avoid circular imports
    from ..core.registry import generate_requirements_txt

    if requirements_path is None:
        try:
            requirements_path = _find_project_root() / "requirements.txt"
        except FileNotFoundError:
            return []  # Can't find project root, skip

    if not requirements_path.exists():
        return []  # No requirements.txt is fine

    try:
        actual_content = requirements_path.read_text(encoding="utf-8").strip()
        expected_content = generate_requirements_txt().strip()

        if actual_content != expected_content:
            return ["requirements.txt does not match registry (run: python -m whisperjav.installer.validation --update)"]
    except Exception as e:
        return [f"Failed to validate requirements.txt: {e}"]

    return []


__all__ = [
    "validate_pyproject_sync",
    "validate_requirements_sync",
]
