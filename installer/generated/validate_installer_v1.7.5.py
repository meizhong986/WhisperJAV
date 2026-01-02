#!/usr/bin/env python
"""
WhisperJAV v1.7.5 Installer Validation Script
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

# Configure UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # Fall back to default encoding if reconfiguration fails

# ANSI color codes for better readability
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

# Expected version
EXPECTED_VERSION = "1.7.5"

# Required files for v1.7.5 installer
REQUIRED_FILES = [
    "construct_v1.7.5.yaml",
    "post_install_v1.7.5.bat",
    "post_install_v1.7.5.py",
    "requirements_v1.7.5.txt",
    "constraints_v1.7.5.txt",    # Version pinning for problematic packages
    "WhisperJAV_Launcher_v1.7.5.py",
    "README_INSTALLER_v1.7.5.txt",
    "LICENSE.txt",
    "whisperjav_icon.ico",
]

# Optional files (warnings only, not errors) - built during installer build process
OPTIONAL_FILES = [
    ("WhisperJAV.exe", "Frozen launcher (built by PyInstaller in Phase 3)"),
]

# Wheel file pattern - matches version with PEP 440 normalization
# e.g., "1.7.0-beta" becomes "1.7.0b0" in wheel filename
WHEEL_PATTERN = "whisperjav-1.7.5*.whl"
WHEEL_PATTERN_FALLBACK = "whisperjav-*.whl"

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

    # Check for wheel file using glob pattern (try version-specific first)
    import glob
    wheels = glob.glob(WHEEL_PATTERN)
    if not wheels:
        # Fall back to any wheel file
        wheels = glob.glob(WHEEL_PATTERN_FALLBACK)

    if wheels:
        # Prefer the one matching our version
        wheel_file = wheels[0]
        for w in wheels:
            if EXPECTED_VERSION.replace("-", "") in w or EXPECTED_VERSION in w:
                wheel_file = w
                break
        print_check(f"Wheel file", True, f"Found: {wheel_file}")
    else:
        print_check("Wheel file", False, "No wheel file found matching pattern")
        all_passed = False

    return all_passed


def validate_version_consistency():
    """Check version consistency across files"""
    print_header("Phase 2: Version Consistency Check")

    all_passed = True
    # Version regex supports both formats:
    # - PEP 440: 1.7.0b0, 1.7.0a1, 1.7.0rc1 (no hyphen)
    # - Display: 1.7.0-beta, 1.7.0-alpha (with hyphen)
    version_pattern = r"(\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:-[a-zA-Z0-9]+)?)"
    version_files = [
        ("construct_v1.7.5.yaml", rf"version:\s*['\"]?{version_pattern}"),
        ("post_install_v1.7.5.py", rf"WhisperJAV v{version_pattern}"),
        ("WhisperJAV_Launcher_v1.7.5.py", rf"v{version_pattern}"),
        ("README_INSTALLER_v1.7.5.txt", rf"WhisperJAV v{version_pattern}"),
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
        "WhisperJAV_Launcher_v1.7.5.py",
        "post_install_v1.7.5.py",
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

    filename = "requirements_v1.7.5.txt"

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


def validate_optional_files():
    """Check optional files (warnings only, not errors)"""
    print_header("Phase 5: Optional Files Check")

    # Optional files don't cause validation failure
    for filename, description in OPTIONAL_FILES:
        exists = check_file_exists(filename)
        if exists:
            size = Path(filename).stat().st_size
            size_kb = size // 1024
            print_check(f"{filename}", True, f"{size_kb} KB - {description}")
        else:
            # Print as warning (yellow), not error
            print(f"  [{YELLOW}! WARN{RESET}] {filename}")
            print(f"          Not found: {description}")
            print(f"          (Will be created during build)")

    # Always return True - optional files don't fail validation
    return True


def validate_assets():
    """Check that icon and other assets exist and are valid"""
    print_header("Phase 6: Asset File Validation")

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


def validate_constraints_file():
    """Check that constraints file contains critical version pins"""
    print_header("Phase 7: Constraints File Validation")

    filename = "constraints_v1.7.5.txt"

    if not check_file_exists(filename):
        print_check(f"{filename}", False, "File not found")
        return False

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for critical version constraints
        critical_constraints = [
            ("numpy>=2.0", "NumPy 2.x for librosa compatibility"),
            ("scipy>=1.10.1", "SciPy for audio processing"),
            ("librosa>=0.11.0", "Librosa NumPy 2.x support"),
            ("datasets>=2.14.0,<4.0", "Datasets cap for modelscope"),
            ("pydantic>=2.0,<3.0", "Pydantic version cap"),
        ]

        all_found = True
        for constraint, description in critical_constraints:
            if constraint in content:
                print_check(f"{constraint}", True, description)
            else:
                print_check(f"{constraint}", False, f"Missing: {description}")
                all_found = False

        return all_found

    except Exception as e:
        print_check(f"{filename}", False, f"Error: {e}")
        return False


def validate_yaml_syntax():
    """Basic YAML syntax validation for construct file"""
    print_header("Phase 8: YAML Syntax Validation")

    filename = "construct_v1.7.5.yaml"

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

        # Check extra_files exist (excluding optional files that are built during build process)
        optional_filenames = [f for f, _ in OPTIONAL_FILES]
        missing_files = [
            f for f in data['extra_files']
            if not check_file_exists(f) and f not in optional_filenames
        ]
        if missing_files:
            print_check(
                f"{filename} syntax",
                False,
                f"Extra files missing: {', '.join(missing_files)}"
            )
            return False

        # Warn about optional files listed in extra_files that don't exist yet
        optional_missing = [
            f for f in data['extra_files']
            if not check_file_exists(f) and f in optional_filenames
        ]
        if optional_missing:
            print(f"  [{YELLOW}! NOTE{RESET}] Optional files in extra_files (built during build):")
            for f in optional_missing:
                print(f"          - {f}")

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
    print(" " * 20 + "WhisperJAV v1.7.5 Installer Validation")
    print("=" * 80)
    print()
    print("Running pre-build validation checks...")

    # Run all validation phases
    results = []
    results.append(("Required Files", validate_required_files()))
    results.append(("Version Consistency", validate_version_consistency()))
    results.append(("Module Paths", validate_module_paths()))
    results.append(("Requirements Encoding", validate_requirements_encoding()))
    results.append(("Optional Files", validate_optional_files()))
    results.append(("Asset Files", validate_assets()))
    results.append(("Constraints File", validate_constraints_file()))
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
        print("Run: build_installer_v1.7.5.bat")
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
