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
from pathlib import Path
from typing import Optional, Tuple

# Version of this upgrade script
UPGRADE_SCRIPT_VERSION = "1.7.4"

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
TORCH_DEPENDENT_PACKAGES = [
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main",
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
        import urllib.request
        urllib.request.urlopen("https://github.com", timeout=10)
        return True
    except Exception:
        return False


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

    Uses version-aware logic to minimize pip exposure and reduce
    risk of dependency conflicts.

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

    success = True

    # Get version-aware dependency list
    deps_to_install = get_required_dependencies(current_version)

    if not deps_to_install:
        print_success("No new dependencies required")
        return True

    # Install regular dependencies (no torch conflict)
    for dep in deps_to_install:
        try:
            result = subprocess.run(
                [str(pip_exe), 'install', dep],
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print_success(dep)
            else:
                # Check if already satisfied
                if "already satisfied" in result.stdout.lower():
                    print_success(f"{dep} (already installed)")
                else:
                    print_warning(f"{dep} - install had issues")
        except Exception as e:
            print_warning(f"{dep} - {e}")

    # Install torch-dependent packages with constraints to preserve CUDA torch
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
                    timeout=600  # 10 min timeout for large packages like NeMo
                )
                if result.returncode == 0:
                    # Extract package name for cleaner output
                    pkg_name = dep.split('@')[0].strip() if '@' in dep else dep
                    print_success(pkg_name)
                else:
                    if "already satisfied" in result.stdout.lower():
                        pkg_name = dep.split('@')[0].strip() if '@' in dep else dep
                        print_success(f"{pkg_name} (already installed)")
                    else:
                        print_warning(f"{dep} - install had issues: {result.stderr[:100]}")
            except subprocess.TimeoutExpired:
                print_warning(f"{dep} - installation timed out (may still be installing)")
            except Exception as e:
                print_warning(f"{dep} - {e}")

        # Cleanup constraints file
        if constraints_path and constraints_path.exists():
            try:
                constraints_path.unlink()
            except Exception:
                pass

    return success


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
    args = parser.parse_args()

    wheel_only_mode = args.wheel_only

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

    # Check network
    if not check_network():
        print_error("No internet connection")
        print("Please check your network connection and try again.")
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
    success, new_version = verify_installation(install_dir)

    if not success or not new_version:
        print_error("Could not verify installation")
        return 1

    print_success(f"WhisperJAV {new_version} installed successfully")

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
