#!/usr/bin/env python
"""
WhisperJAV v1.5.1 Installer Validation Script
==============================================

This script performs comprehensive pre-build validation checks to ensure
all required files are present and correctly configured before running
the conda-constructor build.

Validation checks:
1. All required files exist
2. Version consistency across files
3. Module paths are correct (no old GUI references)
4. Requirements.txt encoding is valid (UTF-8)
5. Icon file exists and is valid
6. README and LICENSE present
7. YAML syntax is valid

Run this before building the installer to catch issues early.
"""

import os
import sys
import re
from pathlib import Path

# ANSI color codes for better readability
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Expected version
EXPECTED_VERSION = "1.5.1"

# Required files for v1.5.1 installer
REQUIRED_FILES = [
    "construct_v1.5.1.yaml",
    "post_install_v1.5.1.bat",
    "post_install_v1.5.1.py",
    "requirements_v1.5.1.txt",
    "WhisperJAV_Launcher_v1.5.1.py",
    "create_desktop_shortcut_v1.5.1.bat",
    "README_INSTALLER_v1.5.1.txt",
    "LICENSE.txt",
    "whisperjav_icon.ico",
]

# Expected module path (no old Tkinter GUI references)
CORRECT_MODULE_PATH = "whisperjav.webview_gui.main"
OLD_MODULE_PATHS = [
    "whisperjav.gui.whisperjav_gui",
    "whisperjav.gui",
]


def print_header(title):
    """Print a formatted section header"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_check(description, passed, details=""):
    """Print a check result with formatting"""
    status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
    print(f"  [{status}] {description}")
    if details:
        print(f"          {details}")


def check_file_exists(filepath):
    """Check if a file exists"""
    return Path(filepath).exists()


def validate_required_files():
    """Check that all required files exist"""
    print_header("Phase 1: Required Files Check")

    all_passed = True
    for filename in REQUIRED_FILES:
        exists = check_file_exists(filename)
        print_check(filename, exists)
        if not exists:
            all_passed = False

    return all_passed


def validate_version_consistency():
    """Check version consistency across files"""
    print_header("Phase 2: Version Consistency Check")

    all_passed = True
    version_files = [
        ("construct_v1.5.1.yaml", r"version:\s*['\"]?(\d+\.\d+\.\d+)"),
        ("post_install_v1.5.1.py", r"WhisperJAV v(\d+\.\d+\.\d+)"),
        ("WhisperJAV_Launcher_v1.5.1.py", r"v(\d+\.\d+\.\d+)"),
        ("README_INSTALLER_v1.5.1.txt", r"WhisperJAV v(\d+\.\d+\.\d+)"),
    ]

    for filename, pattern in version_files:
        if not check_file_exists(filename):
            print_check(f"Version in {filename}", False, "File not found")
            all_passed = False
            continue

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            match = re.search(pattern, content)
            if match:
                version = match.group(1)
                passed = (version == EXPECTED_VERSION)
                details = f"Found: {version}" if not passed else ""
                print_check(f"Version in {filename}: {version}", passed, details)
                if not passed:
                    all_passed = False
            else:
                print_check(f"Version in {filename}", False, "Version pattern not found")
                all_passed = False

        except Exception as e:
            print_check(f"Version in {filename}", False, f"Error: {e}")
            all_passed = False

    return all_passed


def validate_module_paths():
    """Check that module paths reference the new PyWebView GUI, not old Tkinter GUI"""
    print_header("Phase 3: Module Path Validation")

    all_passed = True
    files_to_check = [
        "WhisperJAV_Launcher_v1.5.1.py",
        "create_desktop_shortcut_v1.5.1.bat",
        "post_install_v1.5.1.py",
    ]

    for filename in files_to_check:
        if not check_file_exists(filename):
            print_check(f"Module paths in {filename}", False, "File not found")
            all_passed = False
            continue

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for correct module path
            has_correct_path = CORRECT_MODULE_PATH in content

            # Check for old module paths (should not exist)
            has_old_paths = any(old_path in content for old_path in OLD_MODULE_PATHS)

            passed = has_correct_path and not has_old_paths

            if passed:
                print_check(f"Module paths in {filename}", True)
            else:
                if not has_correct_path:
                    print_check(
                        f"Module paths in {filename}",
                        False,
                        f"Missing correct path: {CORRECT_MODULE_PATH}"
                    )
                if has_old_paths:
                    print_check(
                        f"Module paths in {filename}",
                        False,
                        "Contains old Tkinter GUI references!"
                    )
                all_passed = False

        except Exception as e:
            print_check(f"Module paths in {filename}", False, f"Error: {e}")
            all_passed = False

    return all_passed


def validate_requirements_encoding():
    """Check that requirements.txt is valid UTF-8 without encoding issues"""
    print_header("Phase 4: Requirements File Encoding Check")

    filename = "requirements_v1.5.1.txt"

    if not check_file_exists(filename):
        print_check(f"{filename} encoding", False, "File not found")
        return False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for weird spacing (sign of encoding issues)
        if '  ' in content[:100]:  # Check first 100 chars for double spaces
            print_check(
                f"{filename} encoding",
                False,
                "Suspicious spacing detected - possible encoding issue"
            )
            return False

        # Try to parse as requirements
        lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]

        if len(lines) < 5:
            print_check(
                f"{filename} encoding",
                False,
                f"Too few packages ({len(lines)}) - file may be corrupted"
            )
            return False

        print_check(f"{filename} encoding", True, f"{len(lines)} packages found")
        return True

    except UnicodeDecodeError as e:
        print_check(f"{filename} encoding", False, f"UTF-8 decode error: {e}")
        return False
    except Exception as e:
        print_check(f"{filename} encoding", False, f"Error: {e}")
        return False


def validate_assets():
    """Check that icon and other assets exist and are valid"""
    print_header("Phase 5: Asset File Validation")

    all_passed = True

    # Check icon file
    icon_file = "whisperjav_icon.ico"
    if check_file_exists(icon_file):
        size = Path(icon_file).stat().st_size
        if size > 1000:  # Should be at least 1KB for a valid icon
            print_check(f"{icon_file}", True, f"{size} bytes")
        else:
            print_check(f"{icon_file}", False, f"File too small ({size} bytes)")
            all_passed = False
    else:
        print_check(f"{icon_file}", False, "File not found")
        all_passed = False

    # Check LICENSE
    license_file = "LICENSE.txt"
    if check_file_exists(license_file):
        size = Path(license_file).stat().st_size
        if size > 100:  # Should be at least 100 bytes
            print_check(f"{license_file}", True, f"{size} bytes")
        else:
            print_check(f"{license_file}", False, f"File too small ({size} bytes)")
            all_passed = False
    else:
        print_check(f"{license_file}", False, "File not found")
        all_passed = False

    return all_passed


def validate_yaml_syntax():
    """Basic YAML syntax validation for construct file"""
    print_header("Phase 6: YAML Syntax Validation")

    filename = "construct_v1.5.1.yaml"

    if not check_file_exists(filename):
        print_check(f"{filename} syntax", False, "File not found")
        return False

    try:
        import yaml
        with open(filename, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Check required keys
        required_keys = ['name', 'version', 'channels', 'specs', 'post_install', 'extra_files']
        missing_keys = [key for key in required_keys if key not in data]

        if missing_keys:
            print_check(
                f"{filename} syntax",
                False,
                f"Missing required keys: {', '.join(missing_keys)}"
            )
            return False

        # Verify version matches
        if data['version'] != EXPECTED_VERSION:
            print_check(
                f"{filename} syntax",
                False,
                f"Version mismatch: {data['version']} != {EXPECTED_VERSION}"
            )
            return False

        # Check extra_files exist
        missing_files = [f for f in data['extra_files'] if not check_file_exists(f)]
        if missing_files:
            print_check(
                f"{filename} syntax",
                False,
                f"Extra files missing: {', '.join(missing_files)}"
            )
            return False

        print_check(f"{filename} syntax", True, "YAML structure valid")
        return True

    except ImportError:
        print_check(f"{filename} syntax", False, "PyYAML not installed (skipping)")
        return True  # Non-fatal if yaml module not available
    except Exception as e:
        print_check(f"{filename} syntax", False, f"Error: {e}")
        return False


def main():
    """Run all validation checks"""
    print()
    print("=" * 80)
    print(" " * 20 + "WhisperJAV v1.5.1 Installer Validation")
    print("=" * 80)
    print()
    print("Running pre-build validation checks...")

    # Run all validation phases
    results = []
    results.append(("Required Files", validate_required_files()))
    results.append(("Version Consistency", validate_version_consistency()))
    results.append(("Module Paths", validate_module_paths()))
    results.append(("Requirements Encoding", validate_requirements_encoding()))
    results.append(("Asset Files", validate_assets()))
    results.append(("YAML Syntax", validate_yaml_syntax()))

    # Print summary
    print()
    print("=" * 80)
    print("  VALIDATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for phase, passed in results:
        status = f"{GREEN}✓ PASS{RESET}" if passed else f"{RED}✗ FAIL{RESET}"
        print(f"  {phase:.<50} [{status}]")
        if not passed:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print()
        print(f"{GREEN}✓ All validation checks passed!{RESET}")
        print()
        print("The installer is ready to build.")
        print("Run: build_installer_v1.5.1.bat")
        print()
        return 0
    else:
        print()
        print(f"{RED}✗ Validation failed!{RESET}")
        print()
        print("Please fix the issues above before building the installer.")
        print()
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        print("Validation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print()
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
