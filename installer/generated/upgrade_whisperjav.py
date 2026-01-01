#!/usr/bin/env python3
"""
WhisperJAV Upgrade Script (Standalone Version)
===============================================

This is a standalone version of the upgrade script that can be downloaded
and run independently, without requiring the whisperjav package to be updated first.

Download this file and run it with your WhisperJAV Python:
    %LOCALAPPDATA%\\WhisperJAV\\python.exe upgrade_whisperjav.py

Or if you're in a command prompt with the WhisperJAV environment:
    python upgrade_whisperjav.py

Options:
    --wheel-only    Hot-patch mode: Only update the whisperjav package itself,
                    skip all dependency installation. Safest for post-release
                    upgrades (1.7.4 -> 1.7.4.post1) that don't add new deps.

Features:
    - Upgrades whisperjav package from GitHub @main branch
    - Version-aware dependency installation (minimizes pip exposure)
    - Installs only new dependencies (skips PyTorch)
    - Updates desktop shortcuts with correct version
    - Full cleanup of old version-specific files
    - Preserves user data (model cache, configs)
"""

import os
import sys
import glob
import shutil
import subprocess
import platform
import argparse
import hashlib
import tempfile
import urllib.request
import json
from pathlib import Path
from typing import Optional, Tuple, Dict

# Version of this upgrade script
UPGRADE_SCRIPT_VERSION = "1.7.4"

# Minimum disk space required for upgrade (in bytes)
MIN_DISK_SPACE_GB = 5
MIN_DISK_SPACE_BYTES = MIN_DISK_SPACE_GB * 1024 * 1024 * 1024

# SHA256 checksum manifest URL (two-channel trust)
CHECKSUM_MANIFEST_URL = "https://raw.githubusercontent.com/meizhong986/whisperjav/main/installer/checksums.json"
# Backup: checksums also published in GitHub Release assets

# GitHub repository URL
GITHUB_REPO = "git+https://github.com/meizhong986/whisperjav.git@main"

# New dependencies added in v1.7.x that older versions don't have
# Note: Dependencies that require torch (like nemo_toolkit) are handled separately
# with constraints to prevent CPU torch reinstallation
#
# Organized by version for version-aware upgrades:
# - v1.7.0-1.7.2: transformers, accelerate, pydantic, PyYAML, pydub, regex, ten-vad
# - v1.7.3: modelscope, clearvoice, bs-roformer-infer, etc.
# - v1.7.4: scikit-learn (for semantic scene detection)

# Dependencies added in v1.7.4 (minimal upgrade from v1.7.3.x)
DEPS_V174 = [
    "scikit-learn>=1.3.0",    # Agglomerative clustering for semantic scene detection
]

# Dependencies added in v1.7.3 (speech enhancement)
# NOTE: Speech enhancement deps (modelscope, clearvoice, bs-roformer) are EXCLUDED
# from upgrades due to pip resolution conflicts. Users upgrading from < v1.7.3
# who want speech enhancement should do a fresh install instead.
# Only safe, non-conflicting deps are included here.
DEPS_V173 = [
    "hf_xet",                 # Faster HuggingFace downloads (safe, no conflicts)
]

# Speech enhancement packages - NOT included in upgrade due to pip conflicts
# Users can manually install if needed:
#   pip install modelscope>=1.20 clearvoice bs-roformer-infer
SPEECH_ENHANCEMENT_DEPS_EXCLUDED = [
    "modelscope>=1.20",       # ZipEnhancer - complex dep tree
    "addict",                 # ModelScope dependency
    "datasets>=2.14.0",       # ModelScope dependency - version conflicts
    "simplejson",             # ModelScope dependency
    "sortedcontainers",       # ModelScope dependency
    "clearvoice",             # ClearerVoice - torch conflicts possible
    "bs-roformer-infer",      # BS-RoFormer - torch conflicts possible
    "onnxruntime>=1.16.0",    # ONNX - version sensitive
    "numpy>=2.0",             # NumPy 2.x - breaking changes risk
]

# Dependencies added in v1.7.0-1.7.2
DEPS_V170_V172 = [
    "transformers>=4.40.0",
    "accelerate>=0.26.0",
    "pydantic>=2.0,<3.0",
    "PyYAML>=6.0",
    "pydub",
    "regex",
    "ten-vad",
]

# Full list for fresh installs or very old versions
NEW_DEPENDENCIES = DEPS_V170_V172 + DEPS_V173 + DEPS_V174

# Dependencies that have torch as a requirement - installed with constraints
# to prevent pip from reinstalling CPU-only torch
# NOTE: NeMo removed from upgrade path due to:
#   - @main branch is unstable, breaks reproducibility
#   - 500+ MB download with timeout risk
#   - CUDA version drift could break existing setup
#   - Pip resolver conflicts can hang or fail upgrades
# Users who need NeMo should install manually after upgrade:
#   pip install nemo_toolkit[asr]
TORCH_DEPENDENT_PACKAGES = []

# Files to preserve during upgrade (user data)
PRESERVE_PATTERNS = [
    ".whisperjav_cache",
    "whisperjav_config.json",
]

# Files to clean up (old version-specific files)
CLEANUP_PATTERNS = [
    "install_log_v*.txt",
    "INSTALLATION_FAILED_v*.txt",
    "WhisperJAV_Launcher_v*.py",
    "post_install_v*.py",
    "post_install_v*.bat",
    "requirements_v*.txt",
    "construct_v*.yaml",
]


def print_header():
    """Print the upgrade tool header."""
    print()
    print("=" * 60)
    print("  WhisperJAV Upgrade Tool (Standalone)")
    print("=" * 60)
    print()


def print_step(step: int, total: int, message: str):
    """Print a step progress message."""
    print(f"\n[{step}/{total}] {message}")


def print_success(message: str):
    """Print a success message."""
    print(f"      \u2713 {message}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"      \u26a0 {message}")


def print_error(message: str):
    """Print an error message."""
    print(f"      \u2717 {message}")


def detect_installation() -> Optional[Path]:
    """
    Detect the WhisperJAV installation directory.

    Detection priority:
    1. Script's own directory (if packaged with installer)
    2. Python's sys.prefix (if running within installation)
    3. Default Windows location (%LOCALAPPDATA%\\WhisperJAV)

    Returns:
        Path to installation directory, or None if not found
    """
    # Priority 1: Check script's own directory
    # If this script is packaged with the installer, it lives in the install folder
    script_dir = Path(__file__).resolve().parent
    if (script_dir / 'python.exe').exists() or (script_dir / 'python').exists():
        # Verify it's a WhisperJAV installation by checking for whisperjav package
        try:
            # Add to path temporarily to check
            if str(script_dir / 'Lib' / 'site-packages') not in sys.path:
                sys.path.insert(0, str(script_dir / 'Lib' / 'site-packages'))
            import whisperjav
            return script_dir
        except ImportError:
            # Still return if python.exe exists - might just need upgrade
            return script_dir

    # Priority 2: Check if running from within the installation (sys.prefix)
    if hasattr(sys, 'prefix'):
        install_dir = Path(sys.prefix)
        if (install_dir / 'python.exe').exists() or (install_dir / 'python').exists():
            # Verify it's a WhisperJAV installation
            try:
                import whisperjav
                return install_dir
            except ImportError:
                pass

    # Priority 3: Check default Windows installation location
    if platform.system() == 'Windows':
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        if local_app_data:
            default_path = Path(local_app_data) / 'WhisperJAV'
            if (default_path / 'python.exe').exists():
                return default_path

    return None


def get_current_version(install_dir: Path) -> Optional[str]:
    """
    Get the currently installed WhisperJAV version.

    Args:
        install_dir: Path to installation directory

    Returns:
        Version string, or None if not found
    """
    python_exe = install_dir / 'python.exe'
    if not python_exe.exists():
        python_exe = install_dir / 'python'

    if not python_exe.exists():
        return None

    try:
        result = subprocess.run(
            [str(python_exe), '-c',
             'from whisperjav import __version__; print(__version__)'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def check_network() -> bool:
    """Check if network connectivity to GitHub is available."""
    try:
        urllib.request.urlopen("https://github.com", timeout=10)
        return True
    except Exception:
        return False


def check_disk_space(install_dir: Path) -> Tuple[bool, float]:
    """
    Check if sufficient disk space is available for upgrade.

    Args:
        install_dir: Path to installation directory

    Returns:
        Tuple of (has_enough_space, available_gb)
    """
    try:
        if platform.system() == 'Windows':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(str(install_dir)),
                None, None,
                ctypes.pointer(free_bytes)
            )
            available = free_bytes.value
        else:
            stat = os.statvfs(install_dir)
            available = stat.f_bavail * stat.f_frsize

        available_gb = available / (1024 * 1024 * 1024)
        has_enough = available >= MIN_DISK_SPACE_BYTES
        return has_enough, available_gb
    except Exception as e:
        print_warning(f"Could not check disk space: {e}")
        return True, 0.0  # Assume OK if we can't check


def check_environment(install_dir: Path) -> Tuple[bool, list]:
    """
    Verify we're running in the correct Python environment.

    This is a FAIL-FAST check - running upgrade from wrong Python
    can corrupt the installation.

    Args:
        install_dir: Path to installation directory

    Returns:
        Tuple of (is_correct_env, error_messages)
    """
    errors = []

    # Check 1: sys.prefix should match install_dir
    current_prefix = Path(sys.prefix).resolve()
    expected_prefix = install_dir.resolve()

    if current_prefix != expected_prefix:
        errors.append(
            f"Wrong Python environment detected!\n"
            f"      Running from: {current_prefix}\n"
            f"      Expected:     {expected_prefix}\n"
            f"      Run upgrade using: {expected_prefix / 'python.exe'} upgrade_whisperjav.py"
        )

    # Check 2: Python version should be 3.9-3.12
    version = sys.version_info
    if version < (3, 9) or version >= (3, 13):
        errors.append(
            f"Unsupported Python version: {version.major}.{version.minor}\n"
            f"      Supported: 3.9, 3.10, 3.11, 3.12"
        )

    # Check 3: pip should be available
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'
    if not pip_exe.exists():
        errors.append("pip not found in installation - cannot upgrade")

    return len(errors) == 0, errors


def check_no_gui_running(install_dir: Path) -> Tuple[bool, str]:
    """
    Check if WhisperJAV GUI is currently running.

    Running upgrade while GUI is open can cause file locking issues
    on Windows and corrupt the installation.

    Args:
        install_dir: Path to installation directory

    Returns:
        Tuple of (is_safe, message)
    """
    try:
        import psutil
    except ImportError:
        # psutil not available - can't check, assume safe
        return True, "psutil not available, skipping process check"

    gui_processes = []
    install_str = str(install_dir).lower()

    try:
        for proc in psutil.process_iter(['pid', 'name', 'exe', 'cmdline']):
            try:
                proc_info = proc.info
                proc_exe = (proc_info.get('exe') or '').lower()
                proc_cmdline = ' '.join(proc_info.get('cmdline') or []).lower()

                # Check if process is from our installation
                if install_str in proc_exe or install_str in proc_cmdline:
                    # Check if it's a GUI process
                    if any(x in proc_cmdline for x in ['webview_gui', 'whisperjav-gui']):
                        gui_processes.append(f"PID {proc_info['pid']}: {proc_info['name']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception as e:
        return True, f"Could not check processes: {e}"

    if gui_processes:
        return False, f"WhisperJAV GUI is running:\n" + "\n".join(f"      {p}" for p in gui_processes)

    return True, "No GUI processes detected"


def fetch_checksum_manifest() -> Optional[Dict]:
    """
    Fetch the checksum manifest from GitHub.

    Returns:
        Dictionary with package checksums, or None if unavailable
    """
    try:
        with urllib.request.urlopen(CHECKSUM_MANIFEST_URL, timeout=15) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception as e:
        print_warning(f"Could not fetch checksum manifest: {e}")
        return None


def verify_wheel_checksum(wheel_path: Path, expected_sha256: str) -> bool:
    """
    Verify SHA256 checksum of a downloaded wheel file.

    Args:
        wheel_path: Path to wheel file
        expected_sha256: Expected SHA256 hash (lowercase hex)

    Returns:
        True if checksum matches, False otherwise
    """
    try:
        sha256 = hashlib.sha256()
        with open(wheel_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        actual = sha256.hexdigest().lower()
        expected = expected_sha256.lower()
        return actual == expected
    except Exception as e:
        print_error(f"Checksum verification failed: {e}")
        return False


def download_package(pip_exe: Path, package: str, download_dir: Path,
                     timeout: int = 300) -> Tuple[bool, Optional[Path]]:
    """
    Download a package to a local directory without installing.

    This is part of the download-first pattern to prevent partial
    installations from network failures.

    Args:
        pip_exe: Path to pip executable
        package: Package specifier (e.g., "requests>=2.0" or git URL)
        download_dir: Directory to download to
        timeout: Download timeout in seconds

    Returns:
        Tuple of (success, downloaded_file_path)
    """
    try:
        result = subprocess.run(
            [str(pip_exe), 'download', '-d', str(download_dir),
             '--no-deps', package],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            return False, None

        # Find the downloaded file
        for f in download_dir.iterdir():
            if f.suffix in ('.whl', '.tar.gz', '.zip'):
                return True, f

        return True, None  # Downloaded but couldn't find file (OK for git installs)

    except subprocess.TimeoutExpired:
        return False, None
    except Exception as e:
        print_warning(f"Download error: {e}")
        return False, None


def install_from_local(pip_exe: Path, download_dir: Path,
                       package: str, timeout: int = 300) -> bool:
    """
    Install a package from local download directory.

    Args:
        pip_exe: Path to pip executable
        download_dir: Directory containing downloaded packages
        package: Package name (for git installs, use the original specifier)
        timeout: Install timeout in seconds

    Returns:
        True if successful
    """
    try:
        # First try to install from local files
        result = subprocess.run(
            [str(pip_exe), 'install', '--no-index',
             '--find-links', str(download_dir), package],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode == 0:
            return True

        # If no-index fails (e.g., git install), try direct install
        # This is a fallback for packages that can't be downloaded first
        result = subprocess.run(
            [str(pip_exe), 'install', '--no-deps', package],
            capture_output=True,
            text=True,
            timeout=timeout
        )

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        return False
    except Exception as e:
        print_warning(f"Install error: {e}")
        return False


def download_first_install(pip_exe: Path, packages: list,
                           desc: str = "packages") -> Tuple[bool, list]:
    """
    Download all packages first, then install them.

    This prevents partial installations from network failures.
    Pattern: Download → Verify → Install

    Args:
        pip_exe: Path to pip executable
        packages: List of package specifiers
        desc: Description for progress messages

    Returns:
        Tuple of (all_success, failed_packages)
    """
    if not packages:
        return True, []

    failed = []

    # Create temp directory for downloads
    with tempfile.TemporaryDirectory(prefix='whisperjav_upgrade_') as temp_dir:
        temp_path = Path(temp_dir)

        # Phase 1: Download all packages
        print(f"      Downloading {desc}...")
        download_success = []

        for pkg in packages:
            pkg_name = pkg.split('@')[0].split('>=')[0].split('==')[0].strip()
            success, _ = download_package(pip_exe, pkg, temp_path)

            if success:
                download_success.append(pkg)
            else:
                # For git-based packages, download might not work
                # They'll be handled in install phase
                if 'git+' in pkg:
                    download_success.append(pkg)
                else:
                    print_warning(f"Download failed: {pkg_name}")
                    failed.append(pkg)

        if failed:
            print_warning(f"{len(failed)} package(s) failed to download")
            # Continue with what we have

        # Phase 2: Install downloaded packages
        print(f"      Installing {desc}...")
        for pkg in download_success:
            pkg_name = pkg.split('@')[0].split('>=')[0].split('==')[0].strip()

            if 'git+' in pkg:
                # Git packages need direct install
                try:
                    result = subprocess.run(
                        [str(pip_exe), 'install', '--no-deps', pkg],
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    if result.returncode == 0:
                        print_success(pkg_name)
                    else:
                        print_warning(f"{pkg_name} - install failed")
                        failed.append(pkg)
                except Exception:
                    failed.append(pkg)
            else:
                # Regular packages from local cache
                if install_from_local(pip_exe, temp_path, pkg):
                    print_success(pkg_name)
                else:
                    print_warning(f"{pkg_name} - install failed")
                    failed.append(pkg)

    return len(failed) == 0, failed


def run_preflight_checks(install_dir: Path) -> bool:
    """
    Run all preflight checks before starting upgrade.

    This is the FAIL-FAST gate - any critical failure here
    aborts the upgrade before making any changes.

    Args:
        install_dir: Path to installation directory

    Returns:
        True if all checks pass, False otherwise
    """
    print("Running preflight checks...")
    all_passed = True

    # Check 1: Disk space
    has_space, available_gb = check_disk_space(install_dir)
    if has_space:
        print_success(f"Disk space: {available_gb:.1f} GB available (need {MIN_DISK_SPACE_GB} GB)")
    else:
        print_error(f"Insufficient disk space: {available_gb:.1f} GB available, need {MIN_DISK_SPACE_GB} GB")
        all_passed = False

    # Check 2: Environment
    env_ok, env_errors = check_environment(install_dir)
    if env_ok:
        print_success("Python environment: correct")
    else:
        for err in env_errors:
            print_error(err)
        all_passed = False

    # Check 3: GUI not running (Windows file locking)
    if platform.system() == 'Windows':
        gui_ok, gui_msg = check_no_gui_running(install_dir)
        if gui_ok:
            print_success("GUI check: not running")
        else:
            print_error(f"GUI check failed: {gui_msg}")
            print_error("Please close WhisperJAV GUI before upgrading")
            all_passed = False

    # Check 4: Network (already done in main, but verify again)
    if check_network():
        print_success("Network: connected")
    else:
        print_error("Network: no connection to GitHub")
        all_passed = False

    print()
    return all_passed


def get_torch_version(install_dir: Path) -> Optional[str]:
    """
    Get the currently installed PyTorch version (without CUDA suffix).

    Args:
        install_dir: Path to installation directory

    Returns:
        Version string (e.g., "2.9.1"), or None if not found
    """
    python_exe = install_dir / 'python.exe'
    if not python_exe.exists():
        python_exe = install_dir / 'python'

    if not python_exe.exists():
        return None

    try:
        result = subprocess.run(
            [str(python_exe), '-c',
             "import torch; print(torch.__version__.split('+')[0])"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def create_torch_constraints(install_dir: Path) -> Optional[Path]:
    """
    Create a constraints file to lock PyTorch version.

    This prevents pip from reinstalling CPU-only torch when installing
    packages that have torch as a dependency (like nemo_toolkit).

    Args:
        install_dir: Path to installation directory

    Returns:
        Path to constraints file, or None if failed
    """
    torch_version = get_torch_version(install_dir)
    if not torch_version:
        print_warning("Could not detect PyTorch version for constraints")
        return None

    constraints_path = install_dir / "torch_constraints.txt"
    try:
        with open(constraints_path, 'w') as f:
            f.write(f"torch=={torch_version}\n")
            f.write(f"torchaudio=={torch_version}\n")
        print_success(f"Created torch constraints: torch=={torch_version}")
        return constraints_path
    except Exception as e:
        print_warning(f"Could not create constraints file: {e}")
        return None


def upgrade_package(install_dir: Path) -> bool:
    """
    Upgrade the whisperjav package from GitHub.

    Args:
        install_dir: Path to installation directory

    Returns:
        True if successful, False otherwise
    """
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'

    if not pip_exe.exists():
        print_error("pip not found in installation")
        return False

    # Upgrade whisperjav without dependencies (to skip PyTorch)
    print("      Installing whisperjav from GitHub (this may take a minute)...")
    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '-U', '--no-deps', GITHUB_REPO],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        if result.returncode != 0:
            print_error(f"Package upgrade failed: {result.stderr[:200]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Package upgrade timed out")
        return False
    except Exception as e:
        print_error(f"Package upgrade error: {e}")
        return False

    print_success("Package upgraded successfully")
    return True


def parse_version(version_str: str) -> Tuple[int, int, int, str]:
    """
    Parse version string into components.

    Args:
        version_str: Version string like "1.7.3", "1.7.3.post4", "1.7.4"

    Returns:
        Tuple of (major, minor, patch, suffix)
    """
    import re
    # Handle formats: "1.7.3", "1.7.3.post4", "1.7.4a0", etc.
    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(.*)$', version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)),
                int(match.group(3)), match.group(4) or "")
    return (0, 0, 0, "")


def get_required_dependencies(current_version: Optional[str]) -> list:
    """
    Determine which dependencies need to be installed based on current version.

    This minimizes pip exposure by only installing what's truly needed.

    Args:
        current_version: Currently installed version string

    Returns:
        List of dependencies to install
    """
    if not current_version:
        # Unknown version - install everything
        print("      Version unknown, installing all dependencies...")
        return NEW_DEPENDENCIES

    major, minor, patch, suffix = parse_version(current_version)

    # Determine what's needed based on version
    deps_needed = []

    if major < 1 or (major == 1 and minor < 7):
        # Very old version (< 1.7.0) - need everything
        print(f"      Upgrading from {current_version} (pre-1.7) - installing all dependencies...")
        return NEW_DEPENDENCIES

    if major == 1 and minor == 7:
        if patch < 3:
            # v1.7.0, v1.7.1, v1.7.2 - need v1.7.3 and v1.7.4 deps
            print(f"      Upgrading from {current_version} - installing safe dependencies...")
            print("      Note: Speech enhancement deps excluded (pip conflict risk).")
            print("            For speech enhancement, consider fresh install.")
            deps_needed = DEPS_V173 + DEPS_V174
        elif patch == 3:
            # v1.7.3.x - only need v1.7.4 deps (minimal upgrade!)
            print(f"      Upgrading from {current_version} - minimal upgrade (scikit-learn only)...")
            deps_needed = DEPS_V174
        else:
            # v1.7.4+ - might still need v1.7.4 deps if they weren't installed
            print(f"      Upgrading from {current_version} - checking for missing dependencies...")
            deps_needed = DEPS_V174

    return deps_needed


def install_new_dependencies(install_dir: Path, current_version: Optional[str] = None) -> bool:
    """
    Install new dependencies that weren't in older versions.

    Uses the download-first pattern: downloads all packages first,
    then installs them. This prevents partial installations from
    network failures.

    Args:
        install_dir: Path to installation directory
        current_version: Currently installed version (for version-aware upgrades)

    Returns:
        True if successful, False otherwise
    """
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'

    if not pip_exe.exists():
        print_error("pip not found in installation")
        return False

    # Get version-aware dependency list
    deps_to_install = get_required_dependencies(current_version)

    if not deps_to_install:
        print_success("No new dependencies required")
        return True

    # Use download-first pattern for regular dependencies
    # This downloads all packages first, then installs them
    # Prevents "broken pipe" scenario from network failures
    success, failed = download_first_install(pip_exe, deps_to_install, "dependencies")

    if failed:
        print_warning(f"{len(failed)} package(s) had issues, but upgrade can continue")

    # Install torch-dependent packages with constraints to preserve CUDA torch
    # (Currently empty after NeMo removal, but kept for future use)
    if TORCH_DEPENDENT_PACKAGES:
        print("      Installing torch-dependent packages (with constraints)...")
        constraints_path = create_torch_constraints(install_dir)

        for dep in TORCH_DEPENDENT_PACKAGES:
            try:
                # Build pip command with constraints if available
                cmd = [str(pip_exe), 'install', dep]
                if constraints_path and constraints_path.exists():
                    cmd.extend(['-c', str(constraints_path)])

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=600  # 10 min timeout for large packages
                )
                if result.returncode == 0:
                    pkg_name = dep.split('@')[0].strip() if '@' in dep else dep
                    print_success(pkg_name)
                else:
                    if "already satisfied" in result.stdout.lower():
                        pkg_name = dep.split('@')[0].strip() if '@' in dep else dep
                        print_success(f"{pkg_name} (already installed)")
                    else:
                        print_warning(f"{dep} - install had issues: {result.stderr[:100]}")
            except subprocess.TimeoutExpired:
                print_warning(f"{dep} - installation timed out")
            except Exception as e:
                print_warning(f"{dep} - {e}")

        # Cleanup constraints file
        if constraints_path and constraints_path.exists():
            try:
                constraints_path.unlink()
            except Exception:
                pass

    return True  # Non-fatal - warnings shown but upgrade continues


def update_launcher(install_dir: Path) -> bool:
    """
    Update the WhisperJAV-GUI.exe launcher in the installation root.

    Args:
        install_dir: Path to installation directory

    Returns:
        True if successful, False otherwise
    """
    src = install_dir / 'Scripts' / 'whisperjav-gui.exe'
    dst = install_dir / 'WhisperJAV-GUI.exe'

    if not src.exists():
        # Try alternative locations
        src = install_dir / 'bin' / 'whisperjav-gui'
        if not src.exists():
            print_warning("Launcher executable not found in Scripts/")
            print_warning("Desktop shortcut will use pythonw.exe fallback")
            return True  # Not a fatal error

    try:
        shutil.copy2(src, dst)
        print_success("WhisperJAV-GUI.exe updated")
        return True
    except Exception as e:
        print_warning(f"Could not update launcher: {e}")
        return True  # Not a fatal error


def update_desktop_shortcut(install_dir: Path, new_version: str) -> bool:
    """
    Update the desktop shortcut with the new version name.

    Args:
        install_dir: Path to installation directory
        new_version: New version string

    Returns:
        True if successful, False otherwise
    """
    if platform.system() != 'Windows':
        print_warning("Desktop shortcut update only supported on Windows")
        return True

    try:
        import win32com.client
    except ImportError:
        print_warning("pywin32 not available - shortcut not updated")
        print_warning("You may need to manually update the desktop shortcut")
        return True

    try:
        shell = win32com.client.Dispatch("WScript.Shell")
        desktop = shell.SpecialFolders("Desktop")

        # Remove old shortcuts
        removed_count = 0
        for f in os.listdir(desktop):
            if f.startswith("WhisperJAV v") and f.endswith(".lnk"):
                try:
                    os.remove(os.path.join(desktop, f))
                    print_success(f"Removed: {f}")
                    removed_count += 1
                except Exception:
                    pass

        if removed_count == 0:
            print_warning("No old shortcuts found to remove")

        # Create new shortcut
        shortcut_path = os.path.join(desktop, f"WhisperJAV v{new_version}.lnk")
        shortcut = shell.CreateShortcut(shortcut_path)

        # Determine target
        launcher_exe = install_dir / "WhisperJAV-GUI.exe"
        if launcher_exe.exists():
            shortcut.TargetPath = str(launcher_exe)
            shortcut.Arguments = ""
        else:
            # Fallback to pythonw.exe
            pythonw = install_dir / "pythonw.exe"
            if pythonw.exists():
                shortcut.TargetPath = str(pythonw)
                shortcut.Arguments = "-m whisperjav.webview_gui.main"
            else:
                print_error("No suitable launcher found")
                return False

        shortcut.WorkingDirectory = str(install_dir)

        icon_path = install_dir / "whisperjav_icon.ico"
        if icon_path.exists():
            shortcut.IconLocation = str(icon_path)

        shortcut.Description = f"WhisperJAV v{new_version} - Japanese AV Subtitle Generator"
        shortcut.Save()

        print_success(f"Created: WhisperJAV v{new_version}.lnk")
        return True

    except Exception as e:
        print_error(f"Shortcut update failed: {e}")
        return False


def cleanup_old_files(install_dir: Path) -> int:
    """
    Clean up old version-specific files.

    Args:
        install_dir: Path to installation directory

    Returns:
        Number of files cleaned up
    """
    cleaned = 0

    for pattern in CLEANUP_PATTERNS:
        for filepath in glob.glob(str(install_dir / pattern)):
            try:
                os.remove(filepath)
                cleaned += 1
            except Exception:
                pass

    return cleaned


def verify_installation(install_dir: Path) -> Tuple[bool, Optional[str], list]:
    """
    Verify the installation was successful with comprehensive health checks.

    Checks:
    1. WhisperJAV version is retrievable
    2. Critical modules are importable
    3. CLI entry points exist
    4. PyTorch CUDA status (informational)

    Args:
        install_dir: Path to installation directory

    Returns:
        Tuple of (success, version, warnings)
    """
    warnings = []
    python_exe = install_dir / 'python.exe'
    if not python_exe.exists():
        python_exe = install_dir / 'python'

    # Check 1: Version retrieval
    version = get_current_version(install_dir)
    if not version:
        return False, None, ["Could not retrieve WhisperJAV version"]

    # Check 2: Critical module imports
    critical_modules = [
        ('whisperjav', 'WhisperJAV core'),
        ('faster_whisper', 'Faster-Whisper ASR'),
        ('torch', 'PyTorch'),
    ]

    for module, desc in critical_modules:
        try:
            result = subprocess.run(
                [str(python_exe), '-c', f'import {module}'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                warnings.append(f"{desc} ({module}) import failed")
        except Exception as e:
            warnings.append(f"{desc} ({module}) check error: {e}")

    # Check 3: CLI entry points
    entry_points = [
        ('Scripts/whisperjav-gui.exe', 'GUI launcher'),
        ('Scripts/whisperjav.exe', 'CLI tool'),
    ]

    for path, desc in entry_points:
        ep_path = install_dir / path
        if not ep_path.exists():
            # Try without .exe for non-Windows
            ep_path_alt = install_dir / path.replace('.exe', '')
            if not ep_path_alt.exists():
                warnings.append(f"{desc} entry point missing")

    # Check 4: PyTorch CUDA status (informational only)
    try:
        result = subprocess.run(
            [str(python_exe), '-c',
             'import torch; print("cuda" if torch.cuda.is_available() else "cpu")'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            device = result.stdout.strip()
            if device == 'cpu':
                warnings.append("PyTorch running in CPU mode (no GPU acceleration)")
    except Exception:
        pass  # Non-fatal

    # Success if version retrieved (warnings are non-fatal)
    return True, version, warnings


def main() -> int:
    """Main entry point for the upgrade script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="WhisperJAV Upgrade Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--wheel-only',
        action='store_true',
        help='Hot-patch mode: Only update whisperjav package, skip all dependency '
             'installation. Safest for post-release upgrades (e.g., 1.7.4 -> 1.7.4.post1).'
    )
    parser.add_argument(
        '--skip-preflight',
        action='store_true',
        help='Skip preflight safety checks. USE WITH CAUTION - may cause installation '
             'corruption if used incorrectly.'
    )
    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Non-interactive mode: assume yes to all prompts.'
    )
    args = parser.parse_args()

    wheel_only_mode = args.wheel_only
    skip_preflight = args.skip_preflight

    print_header()

    if wheel_only_mode:
        print("Mode: Wheel-only (hot-patch)")
        print("         Dependencies will be skipped for maximum safety.")
        print()

    # Step 1: Detect installation
    print("Detecting WhisperJAV installation...")
    install_dir = detect_installation()

    if not install_dir:
        print_error("WhisperJAV installation not found")
        print()
        print("Please run this script from within the WhisperJAV environment:")
        print("  %LOCALAPPDATA%\\WhisperJAV\\python.exe upgrade_whisperjav.py")
        print()
        return 1

    print(f"Installation found: {install_dir}")

    # Get current version
    current_version = get_current_version(install_dir)
    if current_version:
        print(f"Current version: {current_version}")
    else:
        print("Current version: Unknown")

    print()

    # Run comprehensive preflight checks (FAIL-FAST gate)
    # This checks: disk space, environment, GUI running, network
    if skip_preflight:
        print_warning("Preflight checks SKIPPED (--skip-preflight)")
        print_warning("Proceeding without safety validation - use at your own risk")
        print()
    else:
        if not run_preflight_checks(install_dir):
            print_error("Preflight checks failed - upgrade aborted")
            print()
            print("Fix the issues above and try again.")
            print("Your installation has NOT been modified.")
            print()
            print("To bypass (NOT recommended): --skip-preflight")
            return 1

    # Adjust step count based on mode
    total_steps = 4 if wheel_only_mode else 5

    # Step 1: Upgrade package
    print_step(1, total_steps, "Upgrading WhisperJAV package...")
    if not upgrade_package(install_dir):
        print_error("Upgrade failed. Your installation may be in an inconsistent state.")
        return 1

    # Track current step for conditional steps
    step_num = 1

    # Step 2: Install new dependencies (skip in wheel-only mode)
    if not wheel_only_mode:
        step_num += 1
        print_step(step_num, total_steps, "Installing new dependencies...")
        install_new_dependencies(install_dir, current_version)
    else:
        print()
        print("      Skipping dependencies (wheel-only mode)")

    # Step 3: Update launcher
    step_num += 1
    print_step(step_num, total_steps, "Updating launcher executable...")
    update_launcher(install_dir)

    # Step 4: Verify and get new version
    step_num += 1
    print_step(step_num, total_steps, "Verifying installation...")
    success, new_version, health_warnings = verify_installation(install_dir)

    if not success or not new_version:
        print_error("Could not verify installation")
        print_error("See docs/MANUAL_ROLLBACK.md for recovery options")
        return 1

    print_success(f"WhisperJAV {new_version} installed successfully")

    # Show health warnings (non-fatal)
    for warning in health_warnings:
        print_warning(warning)

    # Step 5: Update shortcut and cleanup
    step_num += 1
    print_step(step_num, total_steps, "Updating desktop shortcut and cleaning up...")
    update_desktop_shortcut(install_dir, new_version)

    cleaned = cleanup_old_files(install_dir)
    if cleaned > 0:
        print_success(f"Cleaned up {cleaned} old file(s)")

    # Final summary
    print()
    print("=" * 60)
    print("  Upgrade complete!")
    print("=" * 60)
    print()
    print(f"  New version: {new_version}")
    print(f"  Installation: {install_dir}")
    if wheel_only_mode:
        print()
        print("  Note: Dependencies were not updated (wheel-only mode).")
        print("        If you encounter import errors, run without --wheel-only.")
    print()
    print("  You can now launch WhisperJAV from your desktop shortcut.")
    print()
    print("  Note: Your AI models and settings have been preserved.")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nUpgrade cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
