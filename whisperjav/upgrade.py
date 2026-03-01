#!/usr/bin/env python3
"""
WhisperJAV Upgrade Script
=========================

Upgrades WhisperJAV installations to the latest version.

Usage:
    whisperjav-upgrade                    # Interactive upgrade
    whisperjav-upgrade --check            # Check for updates only
    whisperjav-upgrade --yes              # Non-interactive mode
    whisperjav-upgrade --wheel-only       # Hot-patch mode (package only)
    whisperjav-upgrade --extras cli,gui   # Upgrade specific extras only
    whisperjav-upgrade --list-snapshots   # List available rollback snapshots
    whisperjav-upgrade --rollback         # Rollback to previous version
    whisperjav-upgrade --version          # Show script version

Features:
    - Upgrades whisperjav package from GitHub @main branch with all dependencies
    - Supports modular extras (cli, gui, translate, llm, enhance, etc.)
    - Automatic snapshot creation for safe rollback
    - Pre-upgrade compatibility checking
    - Fixes numpy/librosa versions (clearvoice metadata workaround)
    - Updates desktop shortcuts with correct version
    - Full cleanup of old version-specific files
    - Preserves user data (model cache, configs)
    - PyTorch won't be reinstalled if already satisfied
"""

import os
import sys
import glob
import json
import shutil
import subprocess
import platform
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict

# Version of this upgrade script
UPGRADE_SCRIPT_VERSION = "2.0.0"

# =============================================================================
# Configurable Endpoints (for testing)
# =============================================================================
# These can be overridden via environment variables to point at test stubs:
#
#   WHISPERJAV_UPGRADE_REPO - pip-installable URL for the package
#
# Example (pointing at test stub repo):
#   set WHISPERJAV_UPGRADE_REPO=git+https://github.com/meizhong986/whisperjav-test.git@v1.7.5
#
# Example (local wheel file):
#   set WHISPERJAV_UPGRADE_REPO=/path/to/whisperjav-1.7.5-py3-none-any.whl
#
# Example (local git repo):
#   set WHISPERJAV_UPGRADE_REPO=git+file:///c:/repos/whisperjav-test@main
# =============================================================================

# GitHub repository URL
GITHUB_REPO = os.environ.get(
    'WHISPERJAV_UPGRADE_REPO',
    'git+https://github.com/meizhong986/whisperjav.git@main'
)

# Packages to fix AFTER main installation (ensure consistent versions)
FIX_PACKAGES = [
    "numpy>=1.26.0,<2.0",  # NumPy 1.26.x for pyvideotrans compatibility
    "librosa>=0.10.0",
]

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

# Snapshot directory for rollback support
SNAPSHOT_DIR_NAME = ".whisperjav_snapshots"

# Valid extras for modular installation
VALID_EXTRAS = [
    "cli", "gui", "translate", "llm", "enhance",
    "huggingface", "analysis", "compatibility", "all",
]


def print_header():
    """Print the upgrade tool header."""
    print()
    print("=" * 60)
    print("  WhisperJAV Upgrade Tool")
    print("=" * 60)
    print()


def print_step(step: int, total: int, message: str):
    """Print a step progress message."""
    print(f"\n[{step}/{total}] {message}")


def _safe_print(message: str):
    """Print message with fallback for encoding issues (Windows cp1252)."""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fall back to ASCII-safe version
        ascii_msg = message.encode('ascii', errors='replace').decode('ascii')
        print(ascii_msg)


def print_success(message: str):
    """Print a success message."""
    _safe_print(f"      [OK] {message}")


def print_warning(message: str):
    """Print a warning message."""
    _safe_print(f"      [!] {message}")


def print_error(message: str):
    """Print an error message."""
    _safe_print(f"      [X] {message}")


# =============================================================================
# Snapshot and Rollback Functions
# =============================================================================

def get_snapshot_dir(install_dir: Path) -> Path:
    """Get the snapshot directory for an installation."""
    return install_dir / SNAPSHOT_DIR_NAME


def create_upgrade_snapshot(install_dir: Path) -> Optional[Path]:
    """
    Create a snapshot of current installation for rollback.

    Args:
        install_dir: Path to installation directory.

    Returns:
        Path to snapshot directory, or None if failed.
    """
    snapshot_dir = get_snapshot_dir(install_dir)
    snapshot_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_version = get_current_version(install_dir) or "unknown"
    snapshot_name = f"snapshot_{timestamp}_v{current_version}"
    snapshot_path = snapshot_dir / snapshot_name

    try:
        snapshot_path.mkdir()

        # Get pip executable
        pip_exe = install_dir / 'Scripts' / 'pip.exe'
        if not pip_exe.exists():
            pip_exe = install_dir / 'Scripts' / 'pip'
        if not pip_exe.exists():
            pip_exe = install_dir / 'bin' / 'pip'

        if not pip_exe.exists():
            print_warning("pip not found, skipping snapshot creation")
            return None

        # Get current package versions via pip freeze
        result = subprocess.run(
            [str(pip_exe), 'freeze'],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            print_warning(f"Failed to get package list: {result.stderr[:200]}")
            return None

        # Save freeze output
        (snapshot_path / "requirements.txt").write_text(result.stdout, encoding='utf-8')

        # Save metadata
        metadata = {
            "timestamp": timestamp,
            "whisperjav_version": current_version,
            "python_version": sys.version,
            "platform": sys.platform,
            "created_at": datetime.now().isoformat(),
        }
        (snapshot_path / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding='utf-8'
        )

        print_success(f"Created snapshot: {snapshot_name}")
        return snapshot_path

    except Exception as e:
        print_warning(f"Failed to create snapshot: {e}")
        # Clean up partial snapshot
        if snapshot_path.exists():
            shutil.rmtree(snapshot_path, ignore_errors=True)
        return None


def list_snapshots(install_dir: Path) -> List[Dict]:
    """
    List available snapshots with metadata.

    Args:
        install_dir: Path to installation directory.

    Returns:
        List of snapshot info dicts, sorted newest first.
    """
    snapshot_dir = get_snapshot_dir(install_dir)
    if not snapshot_dir.exists():
        return []

    snapshots = []
    for path in sorted(snapshot_dir.iterdir(), reverse=True):
        if path.is_dir() and path.name.startswith("snapshot_"):
            metadata_file = path / "metadata.json"
            if metadata_file.exists():
                try:
                    metadata = json.loads(metadata_file.read_text(encoding='utf-8'))
                    metadata["path"] = path
                    metadata["name"] = path.name
                    snapshots.append(metadata)
                except (json.JSONDecodeError, OSError):
                    # Include snapshot even without valid metadata
                    snapshots.append({
                        "name": path.name,
                        "path": path,
                        "whisperjav_version": "unknown",
                    })

    return snapshots


def rollback_to_snapshot(install_dir: Path, snapshot_path: Path) -> bool:
    """
    Rollback to a previous snapshot.

    Args:
        install_dir: Path to installation directory.
        snapshot_path: Path to snapshot directory.

    Returns:
        True if rollback successful, False otherwise.
    """
    requirements_file = snapshot_path / "requirements.txt"
    if not requirements_file.exists():
        print_error(f"Snapshot requirements not found: {requirements_file}")
        return False

    # Get pip executable
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'

    if not pip_exe.exists():
        print_error("pip not found in installation")
        return False

    print(f"      Rolling back to: {snapshot_path.name}")
    print("      This may take several minutes...")

    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '-r', str(requirements_file)],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )

        if result.returncode != 0:
            print_error(f"Rollback failed: {result.stderr[:500]}")
            return False

        print_success("Rollback completed successfully")
        return True

    except subprocess.TimeoutExpired:
        print_error("Rollback timed out")
        return False
    except Exception as e:
        print_error(f"Rollback error: {e}")
        return False


def cleanup_old_snapshots(install_dir: Path, keep: int = 3) -> int:
    """
    Remove old snapshots, keeping the most recent ones.

    Args:
        install_dir: Path to installation directory.
        keep: Number of recent snapshots to keep.

    Returns:
        Number of snapshots removed.
    """
    snapshots = list_snapshots(install_dir)
    removed = 0

    for snapshot_info in snapshots[keep:]:
        try:
            shutil.rmtree(snapshot_info["path"])
            removed += 1
        except Exception:
            pass

    return removed


# =============================================================================
# Compatibility Check Functions
# =============================================================================

def check_upgrade_compatibility(install_dir: Path, extras: str = "all") -> Tuple[bool, List[str]]:
    """
    Check if upgrade would cause dependency conflicts.

    Args:
        install_dir: Path to installation directory.
        extras: Extras to check (comma-separated or "all").

    Returns:
        Tuple of (is_compatible, list of warnings).
    """
    warnings = []

    # Get pip executable
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'

    if not pip_exe.exists():
        return False, ["pip not found in installation"]

    # Build the install specifier
    if extras == "all":
        install_spec = "whisperjav[all]"
    else:
        extras_list = [e.strip() for e in extras.split(",") if e.strip()]
        install_spec = f"whisperjav[{','.join(extras_list)}]"

    # Use pip's dry-run to check for conflicts (PEP 508 syntax, not #egg=)
    print("      Checking dependency compatibility...")
    try:
        result = subprocess.run(
            [
                str(pip_exe), 'install', '--dry-run', '--ignore-installed',
                f"{install_spec} @ {GITHUB_REPO}"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )

        if result.returncode != 0:
            # Check for specific error patterns
            stderr = result.stderr
            if "ResolutionImpossible" in stderr or "conflict" in stderr.lower():
                warnings.append("Dependency conflicts detected")
                # Extract relevant error lines
                for line in stderr.split('\n'):
                    if 'conflict' in line.lower() or 'requires' in line.lower():
                        warnings.append(f"  {line.strip()}")
                return False, warnings
            else:
                warnings.append(f"pip check failed: {stderr[:300]}")

        # Check for known problematic packages in the output
        output = result.stdout + result.stderr
        if "numpy" in output.lower() and "2.0" in output:
            warnings.append("NumPy 2.0 may cause compatibility issues with some dependencies")

    except subprocess.TimeoutExpired:
        warnings.append("Compatibility check timed out")
    except Exception as e:
        warnings.append(f"Compatibility check error: {e}")

    # Return compatible if no blocking issues found
    has_blocking = any("conflict" in w.lower() for w in warnings)
    return not has_blocking, warnings


def validate_extras(extras: str) -> Tuple[bool, List[str]]:
    """
    Validate that extras are recognized.

    Args:
        extras: Comma-separated extras or "all".

    Returns:
        Tuple of (is_valid, list of invalid extras).
    """
    if extras == "all":
        return True, []

    extras_list = [e.strip().lower() for e in extras.split(",") if e.strip()]
    invalid = [e for e in extras_list if e not in VALID_EXTRAS]

    return len(invalid) == 0, invalid


def detect_installation() -> Optional[Path]:
    """
    Detect the WhisperJAV installation directory.

    Returns:
        Path to installation directory, or None if not found
    """
    # Check if running from within the installation
    if hasattr(sys, 'prefix'):
        install_dir = Path(sys.prefix)
        if (install_dir / 'python.exe').exists() or (install_dir / 'python').exists():
            # Verify it's a WhisperJAV installation
            try:
                import whisperjav
                return install_dir
            except ImportError:
                pass

    # Check default Windows installation location
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
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=10)
        return True
    except Exception:
        return False


def upgrade_package(install_dir: Path) -> bool:
    """
    Upgrade the whisperjav package from GitHub with all dependencies.

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

    # Upgrade whisperjav WITH dependencies (pip resolves versions naturally)
    # PyTorch won't be reinstalled if already satisfied
    print("      Installing whisperjav from GitHub (this may take several minutes)...")
    print("      (New dependencies will be downloaded as needed)")
    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '-U', GITHUB_REPO],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout for large downloads
        )
        if result.returncode != 0:
            print_error(f"Package upgrade failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Package upgrade timed out")
        return False
    except Exception as e:
        print_error(f"Package upgrade error: {e}")
        return False

    print_success("Package upgraded successfully")
    return True


def fix_package_versions(install_dir: Path) -> bool:
    """
    Fix package versions after main installation.

    Ensures numpy and librosa are at the correct versions for pyvideotrans
    compatibility (NumPy 1.26.x).

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

    print("      Upgrading numpy and librosa to latest versions...")
    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '--upgrade'] + FIX_PACKAGES,
            capture_output=True,
            text=True,
            timeout=300
        )
        if result.returncode == 0:
            print_success("numpy and librosa upgraded")
            return True
        else:
            print_warning("Package upgrade had issues, but continuing...")
            return True
    except Exception as e:
        print_warning(f"Package upgrade error: {e}")
        return True  # Non-fatal


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


def verify_installation(install_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Verify the installation was successful.

    Args:
        install_dir: Path to installation directory

    Returns:
        Tuple of (success, version)
    """
    version = get_current_version(install_dir)
    if version:
        return True, version
    return False, None


def check_for_updates_cli(force: bool = False) -> int:
    """
    Check for updates and display result in CLI format.

    Args:
        force: If True, bypass cache

    Returns:
        0 if up to date, 1 if update available, 2 on error
    """
    try:
        from whisperjav.version_checker import (
            check_for_updates,
            get_update_notification_level,
            CURRENT_VERSION,
        )

        print(f"Current version: {CURRENT_VERSION}")
        print("Checking for updates...")

        result = check_for_updates(force=force)

        if result.error:
            print(f"Error: {result.error}")
            return 2

        if result.update_available:
            level = get_update_notification_level(result)
            level_labels = {
                'critical': 'CRITICAL',
                'major': 'Major',
                'minor': 'Minor',
                'patch': 'Patch',
            }
            level_label = level_labels.get(level, 'New')

            print()
            print(f"  {level_label} update available!")
            print(f"  Latest version: {result.latest_version}")
            print()

            if result.version_info:
                print(f"  Release URL: {result.version_info.release_url}")
                if result.version_info.release_notes:
                    # Show first 200 chars of release notes
                    notes = result.version_info.release_notes[:200]
                    if len(result.version_info.release_notes) > 200:
                        notes += "..."
                    print()
                    print("  Release notes:")
                    for line in notes.split('\n')[:5]:
                        print(f"    {line}")

            print()
            print("  Run 'whisperjav-upgrade' to update.")
            print()
            return 1
        else:
            print()
            print("  You have the latest version!")
            print()
            return 0

    except ImportError as e:
        print(f"Error: Could not import version checker: {e}")
        return 2
    except Exception as e:
        print(f"Error checking for updates: {e}")
        return 2


def upgrade_package_wheel_only(install_dir: Path) -> bool:
    """
    Hot-patch mode: Only upgrade the whisperjav package, skip dependencies.

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

    # Upgrade whisperjav WITHOUT dependencies (--no-deps)
    print("      Installing whisperjav from GitHub (wheel only, no deps)...")
    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '-U', '--no-deps', GITHUB_REPO],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        if result.returncode != 0:
            print_error(f"Package upgrade failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Package upgrade timed out")
        return False
    except Exception as e:
        print_error(f"Package upgrade error: {e}")
        return False

    print_success("Package upgraded successfully (wheel only)")
    return True


def upgrade_package_with_extras(install_dir: Path, extras: str) -> bool:
    """
    Upgrade the whisperjav package with specific extras.

    Args:
        install_dir: Path to installation directory.
        extras: Comma-separated extras or "all".

    Returns:
        True if successful, False otherwise.
    """
    pip_exe = install_dir / 'Scripts' / 'pip.exe'
    if not pip_exe.exists():
        pip_exe = install_dir / 'Scripts' / 'pip'
    if not pip_exe.exists():
        pip_exe = install_dir / 'bin' / 'pip'

    if not pip_exe.exists():
        print_error("pip not found in installation")
        return False

    # Build the install specifier using PEP 508 syntax.
    # The legacy #egg= fragment does NOT support extras (brackets) and
    # modern pip rejects it with "egg fragment is invalid" (#192).
    if extras == "all":
        install_spec = f"whisperjav[all] @ {GITHUB_REPO}"
    else:
        extras_list = [e.strip() for e in extras.split(",") if e.strip()]
        extras_str = ",".join(extras_list)
        install_spec = f"whisperjav[{extras_str}] @ {GITHUB_REPO}"

    print(f"      Installing whisperjav[{extras}] from GitHub...")
    print("      (This may take several minutes)")

    try:
        result = subprocess.run(
            [str(pip_exe), 'install', '-U', install_spec],
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        if result.returncode != 0:
            print_error(f"Package upgrade failed: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("Package upgrade timed out")
        return False
    except Exception as e:
        print_error(f"Package upgrade error: {e}")
        return False

    print_success(f"Package upgraded successfully with [{extras}] extras")
    return True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV Upgrade Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisperjav-upgrade                    # Interactive upgrade (all extras)
  whisperjav-upgrade --check            # Check for updates only
  whisperjav-upgrade --yes              # Non-interactive upgrade
  whisperjav-upgrade --extras cli,gui   # Upgrade specific extras only
  whisperjav-upgrade --wheel-only       # Hot-patch mode (package only)
  whisperjav-upgrade --list-snapshots   # List available rollback snapshots
  whisperjav-upgrade --rollback         # Rollback to previous version

Available extras:
  cli         - Audio processing, VAD, scene detection
  gui         - PyWebView GUI interface
  translate   - AI-powered subtitle translation
  llm         - Local LLM server for offline translation
  enhance     - Speech enhancement (denoising, vocal isolation)
  huggingface - HuggingFace Transformers integration
  analysis    - Scientific analysis and visualization
  compatibility - pyvideotrans compatibility layer
  all         - Everything (default)
"""
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Check for updates only, do not upgrade'
    )

    parser.add_argument(
        '--yes', '-y',
        action='store_true',
        help='Non-interactive mode, auto-confirm all prompts'
    )

    parser.add_argument(
        '--wheel-only', '-w',
        action='store_true',
        help='Hot-patch mode: only update whisperjav package, skip dependencies'
    )

    parser.add_argument(
        '--extras', '-e',
        type=str,
        default='all',
        metavar='EXTRAS',
        help='Extras to upgrade (comma-separated). Default: all'
    )

    parser.add_argument(
        '--list-snapshots',
        action='store_true',
        help='List available rollback snapshots'
    )

    parser.add_argument(
        '--rollback',
        nargs='?',
        const='latest',
        metavar='SNAPSHOT',
        help='Rollback to a snapshot. Use --list-snapshots to see available.'
    )

    parser.add_argument(
        '--no-snapshot',
        action='store_true',
        help='Skip creating a snapshot before upgrade (faster but no rollback)'
    )

    parser.add_argument(
        '--skip-compat-check',
        action='store_true',
        help='Skip pre-upgrade compatibility check'
    )

    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show upgrade script version'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help='Force update check (bypass cache)'
    )

    return parser.parse_args()


def main() -> int:
    """Main entry point for the upgrade script."""
    args = parse_args()

    # Handle --version
    if args.version:
        print(f"WhisperJAV Upgrade Script v{UPGRADE_SCRIPT_VERSION}")
        return 0

    # Handle --check
    if args.check:
        return check_for_updates_cli(force=args.force)

    print_header()

    # Step 1: Detect installation
    print("Detecting WhisperJAV installation...")
    install_dir = detect_installation()

    if not install_dir:
        print_error("WhisperJAV installation not found")
        print()
        print("Please run this script from within the WhisperJAV environment:")
        print("  %LOCALAPPDATA%\\WhisperJAV\\python.exe -m whisperjav.upgrade")
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

    # Handle --list-snapshots
    if args.list_snapshots:
        snapshots = list_snapshots(install_dir)
        if not snapshots:
            print("No snapshots available.")
            return 0

        print("Available snapshots:")
        print()
        for i, snap in enumerate(snapshots, 1):
            version = snap.get('whisperjav_version', 'unknown')
            timestamp = snap.get('timestamp', 'unknown')
            name = snap.get('name', 'unknown')
            print(f"  {i}. {name}")
            print(f"     Version: {version}")
            print(f"     Created: {timestamp}")
            print()

        print(f"To rollback: whisperjav-upgrade --rollback <snapshot_name>")
        return 0

    # Handle --rollback
    if args.rollback:
        snapshots = list_snapshots(install_dir)
        if not snapshots:
            print_error("No snapshots available for rollback.")
            return 1

        snapshot_path = None
        if args.rollback == 'latest':
            # Use the most recent snapshot
            snapshot_path = snapshots[0]['path']
            print(f"Rolling back to latest snapshot: {snapshots[0]['name']}")
        else:
            # Find snapshot by name
            for snap in snapshots:
                if snap['name'] == args.rollback:
                    snapshot_path = snap['path']
                    break

            if not snapshot_path:
                print_error(f"Snapshot not found: {args.rollback}")
                print("Use --list-snapshots to see available snapshots.")
                return 1

        if not args.yes:
            print(f"This will rollback to: {snapshot_path.name}")
            print("Your current installation will be reverted.")
            print()
            try:
                response = input("Continue? [y/N]: ").strip().lower()
                if response not in ('y', 'yes'):
                    print("Rollback cancelled.")
                    return 0
            except (EOFError, KeyboardInterrupt):
                print("\nRollback cancelled.")
                return 0

        if rollback_to_snapshot(install_dir, snapshot_path):
            # Update shortcut with rolled-back version
            success, version = verify_installation(install_dir)
            if success and version:
                update_desktop_shortcut(install_dir, version)

            print()
            print("=" * 60)
            print("  Rollback complete!")
            print("=" * 60)
            print()
            return 0
        else:
            return 1

    # Validate extras
    is_valid, invalid_extras = validate_extras(args.extras)
    if not is_valid:
        print_error(f"Invalid extras: {', '.join(invalid_extras)}")
        print(f"Valid extras: {', '.join(VALID_EXTRAS)}")
        return 1

    # Interactive confirmation (unless --yes)
    if not args.yes:
        extras_msg = f"[{args.extras}]" if args.extras != "all" else "[all]"
        print(f"This will upgrade WhisperJAV{extras_msg} to the latest version.")
        print("Your models and settings will be preserved.")
        if not args.no_snapshot:
            print("A snapshot will be created for rollback if needed.")
        print()
        try:
            response = input("Continue? [y/N]: ").strip().lower()
            if response not in ('y', 'yes'):
                print("Upgrade cancelled.")
                return 0
        except (EOFError, KeyboardInterrupt):
            print("\nUpgrade cancelled.")
            return 0

    # Check network
    if not check_network():
        print_error("No internet connection")
        print("Please check your network connection and try again.")
        return 1

    # Pre-upgrade compatibility check
    if not args.skip_compat_check and not args.wheel_only:
        print("\nChecking compatibility...")
        is_compat, warnings = check_upgrade_compatibility(install_dir, args.extras)

        if warnings:
            for warning in warnings:
                print_warning(warning)

        if not is_compat:
            print_error("Compatibility check failed.")
            print("Use --skip-compat-check to bypass this check.")
            return 1
        else:
            print_success("Compatibility check passed")

    # Create snapshot before upgrade (unless --no-snapshot or --wheel-only)
    snapshot_path = None
    if not args.no_snapshot and not args.wheel_only:
        print("\nCreating pre-upgrade snapshot...")
        snapshot_path = create_upgrade_snapshot(install_dir)
        if not snapshot_path:
            print_warning("Could not create snapshot - upgrade will proceed without rollback capability")

    # Wheel-only mode (hot-patch)
    if args.wheel_only:
        total_steps = 3

        print_step(1, total_steps, "Upgrading WhisperJAV package (wheel only)...")
        if not upgrade_package_wheel_only(install_dir):
            print_error("Upgrade failed.")
            return 1

        print_step(2, total_steps, "Verifying installation...")
        success, new_version = verify_installation(install_dir)

        if not success or not new_version:
            print_error("Could not verify installation")
            return 1

        print_success(f"WhisperJAV {new_version} installed successfully")

        print_step(3, total_steps, "Updating desktop shortcut...")
        update_desktop_shortcut(install_dir, new_version)

        print()
        print("=" * 60)
        print("  Hot-patch complete!")
        print("=" * 60)
        print()
        print(f"  New version: {new_version}")
        print()
        return 0

    # Full upgrade mode with extras support
    total_steps = 6

    # Step 1: Upgrade package with extras
    print_step(1, total_steps, f"Upgrading WhisperJAV[{args.extras}]...")
    upgrade_success = upgrade_package_with_extras(install_dir, args.extras)

    if not upgrade_success:
        print_error("Upgrade failed.")
        if snapshot_path:
            print("\nAttempting rollback to pre-upgrade state...")
            if rollback_to_snapshot(install_dir, snapshot_path):
                print_success("Rollback successful - installation restored")
            else:
                print_error("Rollback failed - installation may be in inconsistent state")
        return 1

    # Step 2: Fix package versions (numpy/librosa)
    print_step(2, total_steps, "Fixing package versions...")
    fix_package_versions(install_dir)

    # Step 3: Update launcher
    print_step(3, total_steps, "Updating launcher executable...")
    update_launcher(install_dir)

    # Step 4: Verify and get new version
    print_step(4, total_steps, "Verifying installation...")
    success, new_version = verify_installation(install_dir)

    if not success or not new_version:
        print_error("Could not verify installation")
        if snapshot_path:
            print("\nAttempting rollback to pre-upgrade state...")
            if rollback_to_snapshot(install_dir, snapshot_path):
                print_success("Rollback successful - installation restored")
        return 1

    print_success(f"WhisperJAV {new_version} installed successfully")

    # Step 5: Update shortcut and cleanup
    print_step(5, total_steps, "Updating desktop shortcut and cleaning up...")
    update_desktop_shortcut(install_dir, new_version)

    cleaned = cleanup_old_files(install_dir)
    if cleaned > 0:
        print_success(f"Cleaned up {cleaned} old file(s)")

    # Step 6: Cleanup old snapshots
    print_step(6, total_steps, "Managing snapshots...")
    removed = cleanup_old_snapshots(install_dir, keep=3)
    if removed > 0:
        print_success(f"Removed {removed} old snapshot(s)")
    else:
        print_success("Snapshot management complete")

    # Final summary
    print()
    print("=" * 60)
    print("  Upgrade complete!")
    print("=" * 60)
    print()
    print(f"  New version: {new_version}")
    print(f"  Extras: [{args.extras}]")
    print(f"  Installation: {install_dir}")
    print()
    print("  You can now launch WhisperJAV from your desktop shortcut.")
    print()
    print("  Note: Your AI models and settings have been preserved.")
    if snapshot_path:
        print(f"  Rollback snapshot: {snapshot_path.name}")
        print("  Use: whisperjav-upgrade --rollback to revert if needed")
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
