#!/usr/bin/env python3
"""
WhisperJAV Upgrade Script
=========================

Upgrades WhisperJAV installations to the latest version.

Usage:
    whisperjav-upgrade              # Interactive upgrade
    whisperjav-upgrade --check      # Check for updates only
    whisperjav-upgrade --yes        # Non-interactive mode
    whisperjav-upgrade --wheel-only # Hot-patch mode (package only)
    whisperjav-upgrade --version    # Show script version

Features:
    - Upgrades whisperjav package from GitHub @main branch with all dependencies
    - Fixes numpy/librosa versions (clearvoice metadata workaround)
    - Updates desktop shortcuts with correct version
    - Full cleanup of old version-specific files
    - Preserves user data (model cache, configs)
    - PyTorch won't be reinstalled if already satisfied
"""

import os
import sys
import glob
import shutil
import subprocess
import platform
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

# Version of this upgrade script
UPGRADE_SCRIPT_VERSION = "1.7.5"

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV Upgrade Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  whisperjav-upgrade              # Interactive upgrade
  whisperjav-upgrade --check      # Check for updates only
  whisperjav-upgrade --yes        # Non-interactive upgrade
  whisperjav-upgrade --wheel-only # Hot-patch mode (package only)
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

    # Interactive confirmation (unless --yes)
    if not args.yes:
        print("This will upgrade WhisperJAV to the latest version.")
        print("Your models and settings will be preserved.")
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

    # Full upgrade mode
    total_steps = 5

    # Step 1: Upgrade package
    print_step(1, total_steps, "Upgrading WhisperJAV package...")
    if not upgrade_package(install_dir):
        print_error("Upgrade failed. Your installation may be in an inconsistent state.")
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
        return 1

    print_success(f"WhisperJAV {new_version} installed successfully")

    # Step 5: Update shortcut and cleanup
    print_step(5, total_steps, "Updating desktop shortcut and cleaning up...")
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
