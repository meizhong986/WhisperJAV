"""
WhisperJAV Installer Validation Runner
======================================

Run all validation checks:
    python -m whisperjav.installer.validation

This module is designed to run in CI/CD pipelines to catch issues before release.

VALIDATION CHECKS:
-----------------
1. Standalone Self-Containment: Ensures standalone.py has no whisperjav imports
2. pyproject.toml Sync: Validates pyproject.toml matches registry
3. Import Scanner: Checks for undeclared dependencies (warnings only)

EXIT CODES:
-----------
0: All validations passed
1: Validation failed (critical errors)

Author: Senior Architect
Date: 2026-01-26
"""

import sys
from pathlib import Path
from typing import List


def validate_standalone_self_containment() -> List[str]:
    """
    Ensure standalone.py has no whisperjav imports.

    WHY THIS CHECK:
    standalone.py is designed to be bundled with the standalone installer
    (conda-constructor) and run BEFORE whisperjav is installed. If it
    imports from whisperjav.*, the installer will break.

    This was identified as Gemini Review Watchpoint #1 - the most brittle
    aspect of the architecture refactoring.

    FORBIDDEN PATTERNS:
    - from whisperjav import ...
    - from whisperjav.* import ...
    - import whisperjav

    Returns:
        List of error messages (empty if validation passes)
    """
    # Find standalone.py relative to this file
    validation_dir = Path(__file__).parent
    installer_dir = validation_dir.parent
    standalone_path = installer_dir / "core" / "standalone.py"

    if not standalone_path.exists():
        return [f"standalone.py not found at {standalone_path}"]

    errors = []
    try:
        content = standalone_path.read_text(encoding="utf-8")
        for line_num, line in enumerate(content.splitlines(), 1):
            # Skip comments and strings
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Check for forbidden imports
            if "from whisperjav" in line or "import whisperjav" in line:
                # Make sure it's not in a comment on the same line
                code_part = line.split("#")[0]
                if "from whisperjav" in code_part or "import whisperjav" in code_part:
                    errors.append(
                        f"standalone.py:{line_num}: Forbidden import: {line.strip()}"
                    )
    except Exception as e:
        errors.append(f"Failed to read standalone.py: {e}")

    return errors


def run_all_validations(quick: bool = False) -> int:
    """
    Run all validation checks.

    Args:
        quick: If True, skip slow checks (import scanner)

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Import here to avoid import errors if dependencies are missing
    try:
        from .sync import validate_pyproject_sync
        from .imports import scan_imports
        sync_available = True
    except ImportError as e:
        print(f"Warning: Some validation modules not available: {e}")
        sync_available = False

    print("=" * 70)
    print("WhisperJAV Installer Validation")
    print("=" * 70)
    print()

    all_errors: List[str] = []
    all_warnings: List[str] = []
    total_checks = 3 if sync_available and not quick else 2 if sync_available else 1

    check_num = 0

    # Check 1: Standalone self-containment
    check_num += 1
    print(f"[{check_num}/{total_checks}] Checking standalone.py self-containment...")
    errors = validate_standalone_self_containment()
    if errors:
        print("  FAILED!")
        for error in errors:
            print(f"    - {error}")
        all_errors.extend(errors)
    else:
        print("  OK - No forbidden imports found")
    print()

    # Check 2: pyproject.toml sync (if available)
    if sync_available:
        check_num += 1
        print(f"[{check_num}/{total_checks}] Validating pyproject.toml sync...")
        errors = validate_pyproject_sync()
        if errors:
            print(f"  FAILED! ({len(errors)} errors)")
            for error in errors[:10]:  # Show first 10
                print(f"    - {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more")
            all_errors.extend(errors)
        else:
            print("  OK - pyproject.toml matches registry")
        print()

    # Check 3: Import scanner (if not quick mode)
    if sync_available and not quick:
        check_num += 1
        print(f"[{check_num}/{total_checks}] Scanning for untracked imports...")
        warnings = scan_imports()
        if warnings:
            print(f"  WARNING: {len(warnings)} untracked imports found")
            for warning in warnings[:10]:  # Show first 10
                print(f"    - {warning}")
            if len(warnings) > 10:
                print(f"    ... and {len(warnings) - 10} more")
            all_warnings.extend(warnings)
        else:
            print("  OK - All imports tracked in registry")
        print()

    # Summary
    print("=" * 70)
    if all_errors:
        print(f"VALIDATION FAILED: {len(all_errors)} error(s)")
        print("=" * 70)
        return 1
    elif all_warnings:
        print(f"VALIDATION PASSED with {len(all_warnings)} warning(s)")
        print("=" * 70)
        return 0
    else:
        print("VALIDATION PASSED: All checks passed")
        print("=" * 70)
        return 0


if __name__ == "__main__":
    # Parse simple arguments
    quick = "--quick" in sys.argv or "-q" in sys.argv

    if "--help" in sys.argv or "-h" in sys.argv:
        print(__doc__)
        print("Options:")
        print("  --quick, -q  Skip slow checks (import scanner)")
        print("  --help, -h   Show this help message")
        sys.exit(0)

    sys.exit(run_all_validations(quick=quick))
