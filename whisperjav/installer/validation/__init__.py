"""
WhisperJAV Installer Validation System
======================================

This subpackage provides validation tools to prevent configuration drift.

WHY VALIDATION EXISTS:
---------------------
Before this refactor, the installation system had 4+ files defining packages:
- pyproject.toml (optional-dependencies)
- install_windows.bat (pip install commands)
- install_linux.sh (pip install commands)
- post_install.py.template (requirements list)

These files would drift apart over time:
- Developer adds package to pyproject.toml, forgets install scripts
- Version bumped in one file but not others
- "Ghost dependencies" - imports that work locally but fail for users

VALIDATION TOOLS:
----------------
1. sync.py - Validates pyproject.toml matches registry
2. imports.py - Scans code for imports not in registry

CI/CD INTEGRATION:
-----------------
Run validation as part of CI:
    python -m whisperjav.installer.validation

This blocks merges that would cause drift.

Author: Senior Architect
Date: 2026-01-26
"""

# Validation exports
from .__main__ import (
    validate_standalone_self_containment,
    run_all_validations,
)
from .sync import (
    validate_pyproject_sync,
    validate_requirements_sync,
)
from .imports import (
    scan_imports,
)

__all__ = [
    # Main runner
    "validate_standalone_self_containment",
    "run_all_validations",
    # Sync validation
    "validate_pyproject_sync",
    "validate_requirements_sync",
    # Import scanner
    "scan_imports",
]
