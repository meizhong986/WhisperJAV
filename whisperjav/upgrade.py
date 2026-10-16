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
    whisperjav-upgrade --extras cli,gui   # Upgrade specific extras (source installs only)
    whisperjav-upgrade --list-snapshots   # List available rollback snapshots
    whisperjav-upgrade --rollback         # Rollback to previous version
    whisperjav-upgrade --version          # Show script version

Features:
    - Upgrades whisperjav to the newest release of its own series (1.10.x for 1.10),
      never to GitHub's main branch (v1.10.0, item 2e)
    - Keeps the PC's PyTorch build (CUDA 13.0, the CUDA 12.8 fallback, or CPU)
    - Installs the release's pinned package set (installer installs) or
      `uv sync --frozen` (source installs; extras selectable there only)
    - Automatic snapshot creation for safe rollback
    - Checks before any change that the release's pinned packages are available
    - Updates desktop shortcuts with correct version
    - Full cleanup of old version-specific files
    - Preserves user data (model cache, configs)
    - PyTorch won't be reinstalled if already satisfied
"""

import os
import re
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

# Where the package comes from.
#
# v1.10.0 (item 2e, spec H): the newest release tag of the installed series --
# v1.10.* for 1.10 -- read from GitHub's releases list; never `main`. A later 1.11
# or 2.0 is a fresh install, not an upgrade. WHISPERJAV_UPGRADE_REPO still
# overrides it (tests, a local wheel, a release candidate).
GITHUB_GIT_URL = "git+https://github.com/meizhong986/whisperjav.git"
GITHUB_RELEASES_API = os.environ.get(
    'WHISPERJAV_UPGRADE_RELEASES_API',
    'https://api.github.com/repos/meizhong986/whisperjav/releases?per_page=100'
)
GITHUB_REPO = os.environ.get('WHISPERJAV_UPGRADE_REPO') or GITHUB_GIT_URL

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
    "cli", "gui", "translate", "enhance",
    "huggingface", "analysis", "compatibility", "all",
]

IS_WINDOWS = platform.system() == 'Windows'

# PyTorch builds. v1.10.0: cu130 is the stack; cu128 is the per-PC fallback of item 2
# spec D. v1.10.0 (item 2f, owner W53-W56): every upgrade installs the new release's
# locked versions for the build this PC has -- uv.lock exported per build into
# whisperjav/installer/locked/ -- and never resolves version ranges.
PYTORCH_INDEXES = {
    "cpu": "https://download.pytorch.org/whl/cpu",
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu130": "https://download.pytorch.org/whl/cu130",
}
VARIANT_LABELS = {"cu130": "CUDA 13.0", "cu128": "CUDA 12.8", "cpu": "CPU"}
# uv.lock extra per build; macOS ("metal") uses cpu, whose index has the Apple wheels.
LOCKED_EXTRAS = {"cu130": "cu130", "cu128": "cu128", "cpu": "cpu", "metal": "cpu"}
TORCH_PACKAGE_NAMES = ("torch", "torchaudio", "torchvision")
ISSUES_URL = "https://github.com/meizhong986/WhisperJAV/issues"

# The owner's texts (item 2f, W58).
STACK_NOT_RECORDED_TEXT = (
    "This installation does not record its PyTorch build (whisperjav_stack.txt), so the upgrade "
    "cannot choose its pinned packages. Nothing was changed. Run the WhisperJAV {series} installer "
    "again to repair it.")
EXTRAS_INSTALLER_TEXT = (
    "--extras applies to source installs only. This installation came from the Windows installer, "
    "so an upgrade installs the release's full pinned package set. Run again without --extras.")


def _series_label() -> str:
    """'1.10' for WhisperJAV 1.10.x."""
    from whisperjav.__version__ import __version__
    return ".".join(__version__.split(".")[:2])


# =============================================================================
# Upgrade source: the newest release of this series (item 2e)
# =============================================================================

def _series(version: str) -> Optional[Tuple[int, int]]:
    """(major, minor) of a version or tag, e.g. "v1.10.2" -> (1, 10)."""
    import re
    m = re.match(r"^v?(\d+)\.(\d+)\.(\d+)", (version or "").strip())
    return (int(m.group(1)), int(m.group(2))) if m else None


def newest_release_in_series(current_version: str, releases: List[Dict]) -> Optional[str]:
    """Tag of the newest published, non-pre-release release in current_version's series.

    `releases` is GitHub's releases list (dicts with tag_name, draft, prerelease).
    """
    from whisperjav.version_checker import compare_versions

    series = _series(current_version)
    if series is None:
        return None
    best = None
    for rel in releases or []:
        tag = rel.get("tag_name") or ""
        if rel.get("draft") or rel.get("prerelease") or _series(tag) != series:
            continue
        if best is None or compare_versions(tag, best) > 0:
            best = tag
    return best


def _fetch_releases(timeout: int = 15) -> Optional[List[Dict]]:
    import urllib.request
    try:
        request = urllib.request.Request(
            GITHUB_RELEASES_API,
            headers={'Accept': 'application/vnd.github.v3+json', 'User-Agent': 'WhisperJAV-upgrade'},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode('utf-8'))
        return data if isinstance(data, list) else None
    except Exception:
        return None


def resolve_upgrade_source(current_version: Optional[str]) -> Tuple[Optional[str], str]:
    """(pip-installable source, explanation). Source None = do not upgrade.

    WHISPERJAV_UPGRADE_REPO wins. Otherwise the newest v<major>.<minor>.* release tag
    of the installed series; never main.
    """
    override = os.environ.get('WHISPERJAV_UPGRADE_REPO')
    if override:
        return override, f"WHISPERJAV_UPGRADE_REPO={override}"
    if not current_version or _series(current_version) is None:
        return None, "the installed version could not be read"
    releases = _fetch_releases()
    if releases is None:
        return None, "GitHub's releases list could not be read"
    tag = newest_release_in_series(current_version, releases)
    major, minor = _series(current_version)
    if tag is None:
        return None, f"no published {major}.{minor} release was found"
    return f"{GITHUB_GIT_URL}@{tag}", f"newest {major}.{minor} release: {tag}"


# =============================================================================
# The PC's stack (item 2 spec D): kept on every upgrade
# =============================================================================

def read_installed_stack(install_dir: Optional[Path] = None) -> Optional[str]:
    """cu130, cu128 or cpu, as the installer recorded it; None if not recorded."""
    try:
        from whisperjav.installer.cublas12 import read_stack
        return read_stack(str(install_dir) if install_dir else None)
    except Exception:
        return None


def place_cublas12_after_upgrade(install_dir: Path) -> None:
    """On a CUDA 13.0 Windows install, place cuBLAS 12 again after the package step.

    A reinstalled ctranslate2 brings its unused cuDNN DLL back, and a newer
    nvidia-cublas-cu12 must reach CTranslate2's folder (item 2 spec C).
    """
    if not IS_WINDOWS or read_installed_stack(install_dir) != "cu130":
        return
    python_exe = _find_python(install_dir)
    if python_exe is None:
        return
    # A source install's `uv sync` removes packages pyproject.toml does not list, and
    # nvidia-cublas-cu12 is one (review 2026-10-14): put it back before placing.
    try:
        from whisperjav.installer.cublas12 import CUBLAS12_PACKAGE, cublas12_source_dir  # noqa: F401
        probe = subprocess.run([str(python_exe), "-c", "import nvidia.cublas"],
                               capture_output=True, text=True, timeout=60)
        if probe.returncode != 0:
            uv_exe = shutil.which("uv")
            cmd = ([uv_exe, "pip", "install", "--python", str(python_exe), CUBLAS12_PACKAGE] if uv_exe
                   else [str(python_exe), "-m", "pip", "install", CUBLAS12_PACKAGE])
            subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except Exception as e:
        print_warning(f"Could not reinstall cuBLAS 12 for faster-whisper: {e}")
    try:
        r = subprocess.run([str(python_exe), "-m", "whisperjav.installer.cublas12", "place"],
                           capture_output=True, text=True, timeout=300)
        line = (r.stdout or r.stderr or "").strip().splitlines()[:1]
        if r.returncode == 0:
            print_success("cuBLAS 12 placed for faster-whisper")
        else:
            print_warning(f"Could not place cuBLAS 12 for faster-whisper: {line[0] if line else r.returncode}")
    except Exception as e:
        print_warning(f"Could not place cuBLAS 12 for faster-whisper: {e}")


# =============================================================================
# Release-pinned dependencies (item 2f)
# =============================================================================

def _requirement_name(line: str) -> str:
    """'torch==2.11.0+cu130' / 'stable-ts @ git+...' / 'x==1 ; marker' -> normalized name."""
    name = re.split(r"[=<>!~;@\[\s]", line.strip(), maxsplit=1)[0]
    return re.sub(r"[-_.]+", "-", name).lower()


def _pinned_version(line: str) -> Optional[str]:
    """'packaging==26.3' -> '26.3'; None for a line without ==, such as a git package."""
    m = re.match(r"^[^=<>!~;@\s]+\s*==\s*([^\s;]+)", line.strip())
    return m.group(1) if m else None


def _metadata_name_version(meta_file: Path) -> Tuple[Optional[str], Optional[str]]:
    """Name and Version from a METADATA or PKG-INFO file (its header block)."""
    name = version = None
    try:
        with open(meta_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    break
                if line.startswith("Name:"):
                    name = line[5:].strip()
                elif line.startswith("Version:"):
                    version = line[8:].strip()
    except OSError:
        pass
    return name, version


def _conda_python_packages(install_dir: Path) -> Tuple[Dict[str, Tuple[str, str]], List[str]]:
    """Python packages conda put into this installation, as pip names them.

    Owner W57 (D1), as in the installer: read from the *.dist-info / *.egg-info folder
    each conda-meta record lists, not from conda's package name. Returns
    ({normalized name: (name, version)}, conda packages with Python files but no such folder).
    """
    found: Dict[str, Tuple[str, str]] = {}
    unidentified: List[str] = []
    for record in sorted((install_dir / "conda-meta").glob("*.json")):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        conda_name = str(data.get("name", ""))
        if conda_name == "python":
            continue  # the interpreter; its one site-packages file is a README
        site = [str(f).replace("\\", "/") for f in (data.get("files") or [])
                if str(f).replace("\\", "/").lower().startswith("lib/site-packages/")]
        if not site:
            continue
        folders = sorted({"/".join(f.split("/")[:3]) for f in site
                          if f.split("/")[2].lower().endswith((".dist-info", ".egg-info"))})
        if not folders:
            unidentified.append(conda_name)
            continue
        for folder in folders:
            path = install_dir / folder
            if path.is_dir():
                path = path / ("METADATA" if folder.lower().endswith(".dist-info") else "PKG-INFO")
            name, version = _metadata_name_version(path)
            if not name or not version:
                parts = folder.split("/")[-1].rsplit(".", 1)[0].split("-")
                name = name or parts[0]
                version = version or (parts[1] if len(parts) > 1 else "")
            found[re.sub(r"[-_.]+", "-", name).lower()] = (name, version)
    return found, unidentified


def check_conda_packages(install_dir: Path, lines: List[str]) -> bool:
    """D1 (owner W57): conda's copy of a pinned package stays only at the pinned version.

    Every package conda installed that the new release pins is compared and printed, so
    a check that compared nothing shows as "Checked 0 packages".
    """
    conda_packages, unidentified = _conda_python_packages(install_dir)
    for conda_name in unidentified:
        print_warning(f"conda's {conda_name} put Python files here without package metadata, "
                      f"so it could not be compared.")
    compared, ok = 0, True
    for line in lines:
        key = _requirement_name(line)
        if key not in conda_packages:
            continue
        compared += 1
        name, conda_version = conda_packages[key]
        pinned = _pinned_version(line)
        if pinned is not None and pinned == conda_version:
            print(f"      Keeping conda's {name} {conda_version}: the same version this release is pinned to.")
            continue
        ok = False
        pinned_text = f"{name} {pinned}" if pinned else line.split(";")[0].strip()
        print_error(f"Stopped: conda installed {name} {conda_version}, but this release is pinned to "
                    f"{pinned_text}. Nothing was changed. Please report this so it can be fixed: {ISSUES_URL}")
    if ok:
        print(f"      Checked {compared} package{'' if compared == 1 else 's'} that conda installed against the "
              f"pinned list: all match.")
    return ok


def fetch_release_locked_file(source: str, stack: str) -> Optional[str]:
    """The new release's locked file for `stack`, read before anything changes (D5).

    `source` is what pip installs: git+https://github.com/<owner>/<repo>.git@<tag>, a
    git+file:// repository @<ref>, a local wheel, or a local source folder.
    """
    rel = f"whisperjav/installer/locked/requirements_{stack}.txt"
    try:
        m = re.match(r"^git\+https://github\.com/([^/]+)/([^/@]+?)(?:\.git)?(?:@(.+))?$", source)
        if m:
            import urllib.request
            url = f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}/{m.group(3) or 'HEAD'}/{rel}"
            request = urllib.request.Request(url, headers={'User-Agent': 'WhisperJAV-upgrade'})
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8")
        m = re.match(r"^git\+file://(.+?)(?:@([^@/]+))?$", source)
        if m:
            repo = m.group(1)
            if re.match(r"^/[A-Za-z]:", repo):
                repo = repo[1:]  # git+file:///c:/repos/x
            r = subprocess.run(["git", "-C", repo, "show", f"{m.group(2) or 'HEAD'}:{rel}"],
                               capture_output=True, text=True, encoding="utf-8", timeout=60)
            return r.stdout if r.returncode == 0 else None
        path = Path(source)
        if path.suffix == ".whl" and path.is_file():
            import zipfile
            with zipfile.ZipFile(path) as wheel:
                return wheel.read(rel).decode("utf-8")
        if (path / rel).is_file():
            return (path / rel).read_text(encoding="utf-8")
    except Exception:
        return None
    return None


def _split_pins(install_dir: Path, lines: List[str]) -> Tuple[List[str], List[str]]:
    """(lines pip installs from PyPI, PyTorch's three lines); conda's packages left out."""
    conda_packages, _ = _conda_python_packages(install_dir)
    torch_lines = [l for l in lines if _requirement_name(l) in TORCH_PACKAGE_NAMES]
    rest = [l for l in lines
            if _requirement_name(l) not in TORCH_PACKAGE_NAMES and _requirement_name(l) not in conda_packages]
    return rest, torch_lines


def _pin_commands(pip_exe: Path, stack: str, rest_file: str, torch_lines: List[str],
                  dry_run: bool) -> List[Tuple[List[str], str]]:
    """The two pip commands of a pinned install; the pre-check runs the same ones with --dry-run."""
    extra = ["--dry-run"] if dry_run else []
    return [
        ([str(pip_exe), "install", *extra, "--no-deps", "-r", rest_file], "dependencies"),
        ([str(pip_exe), "install", *extra, "--no-deps", *torch_lines,
          "--index-url", PYTORCH_INDEXES[stack]], "PyTorch"),
    ]


def _unavailable_package(stderr: str) -> Tuple[Optional[str], str]:
    """(package pip could not find, pip's error line) from a failed pip run."""
    errors = [l.strip() for l in (stderr or "").splitlines() if l.strip().startswith("ERROR:")]
    for line in errors:
        m = re.search(r"No matching distribution found for (\S+)", line)
        if m:
            return _requirement_name(m.group(1)), line
    return None, (errors[0] if errors else (stderr or "").strip()[:300])


def prepare_release_pins(install_dir: Path, stack: str, check_availability: bool = True) -> Optional[List[str]]:
    """Before anything changes (D5, owner W57 INPUT3): read the new release's locked file
    for this PC's build, compare conda's packages with it (D1), and -- unless skipped --
    run the pinned install as a pip dry run. None = stop; the reason has been printed.
    """
    text = fetch_release_locked_file(GITHUB_REPO, stack)
    if text is None:
        print_error(f"Upgrade stopped before any change: the new release's pinned package list for "
                    f"{VARIANT_LABELS[stack]} could not be read ({GITHUB_REPO}).")
        return None
    lines = [l.strip() for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not check_conda_packages(install_dir, lines):
        return None
    if not check_availability:
        return lines
    pip_exe = _find_pip(install_dir)
    if pip_exe is None:
        print_error("pip not found in installation")
        return None
    print(f"      Checking that this release's pinned packages are available for this PC "
          f"({VARIANT_LABELS[stack]})...")
    import tempfile
    rest, torch_lines = _split_pins(install_dir, lines)
    fd, rest_file = tempfile.mkstemp(prefix="whisperjav_pins_", suffix=".txt")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(rest) + "\n")
        for cmd, _ in _pin_commands(pip_exe, stack, rest_file, torch_lines, dry_run=True):
            # Bounded for real (#436): see _run_bounded. Git packages are cloned here.
            result = _run_bounded(cmd, timeout=1800)
            if result.returncode != 0:
                package, pip_line = _unavailable_package(result.stderr)
                if package:
                    print_error(f"Upgrade stopped before any change: {package} is not available for this PC "
                                f"({pip_line}).")
                else:
                    print_error(f"Upgrade stopped before any change: {pip_line}")
                return None
    except subprocess.TimeoutExpired:
        print_error("Upgrade stopped before any change: the check timed out.")
        return None
    finally:
        try:
            os.unlink(rest_file)
        except OSError:
            pass
    print_success(f"All {len(lines)} pinned packages are available.")
    return lines


def install_locked_requirements(install_dir: Path, stack: str, lines: List[str]) -> bool:
    """Install the new release's pinned packages (the lines prepare_release_pins read).

    Everything but PyTorch from PyPI, then PyTorch's three packages from the build's
    index; both --no-deps, as the file is complete. conda's packages, checked to be at
    the pinned version, are left out (owner W52, W57).
    """
    pip_exe = _find_pip(install_dir)
    if pip_exe is None:
        print_error("pip not found in installation")
        return False
    rest, torch_lines = _split_pins(install_dir, lines)

    print(f"      Upgrading with release-pinned dependencies ({VARIANT_LABELS[stack]})...")
    rest_file = install_dir / f"requirements_upgrade_{stack}.txt"
    try:
        rest_file.write_text("\n".join(rest) + "\n", encoding="utf-8")
        for cmd, _ in _pin_commands(pip_exe, stack, str(rest_file), torch_lines, dry_run=False):
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if result.returncode != 0:
                print_error(f"Installing the release's pinned packages failed: {(result.stderr or '')[:500]}")
                return False
    except subprocess.TimeoutExpired:
        print_error("Package upgrade timed out")
        return False
    except Exception as e:
        print_error(f"Installing the release's pinned packages failed: {e}")
        return False
    finally:
        try:
            rest_file.unlink()
        except OSError:
            pass
    print_success(f"Installed the release's pinned packages ({len(lines)}).")
    return True


# =============================================================================
# Source Install Detection
# =============================================================================

def _find_source_root() -> Optional[Path]:
    """Find the WhisperJAV source root (directory with pyproject.toml + whisperjav/).

    For source installs (editable pip install or uv sync), the project root
    is typically the cwd, or a parent/sibling of the install prefix.

    Returns:
        Path to source root, or None if not a source install.
    """
    candidates = [
        Path.cwd(),
        Path.cwd().parent,
        Path(__file__).parent.parent,  # whisperjav/ -> project root
    ]
    for candidate in candidates:
        if (candidate / "pyproject.toml").exists() and (candidate / "whisperjav").is_dir():
            # Extra check: does it have uv.lock? (confirms uv project)
            if (candidate / "uv.lock").exists():
                return candidate
    return None


def _detect_cuda_for_source() -> str:
    """Detect CUDA build for source-install upgrades.

    Returns: "cpu", "cu130", "cu128", or "metal". The stack the installer recorded
    wins (v1.10.0: an upgrade never moves a PC to another stack). Without a record,
    Linux gets cu128 (owner, W45) and Windows cu130 at driver 580+.
    """
    # Apple Silicon
    if sys.platform == "darwin" and platform.machine() == "arm64":
        return "metal"

    recorded = read_installed_stack()
    if recorded:
        return recorded

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=10
        )
        driver = result.stdout.strip().split('\n')[0]
        if driver:
            major = int(driver.split('.')[0])
            # Linux stays on CUDA 12.8 for 1.10.0 (owner, W45; see install.py).
            if major >= 580 and not sys.platform.startswith("linux"):
                return "cu130"
            return "cu128"
    except Exception:
        pass

    # Try importing torch to check existing CUDA
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; v = torch.version.cuda; print(v if v else 'cpu')"],
            capture_output=True, text=True, check=True, timeout=15
        )
        cuda_ver = result.stdout.strip()
        if cuda_ver and cuda_ver != "cpu":
            # e.g. "12.8" -> "cu128"
            normalized = cuda_ver.replace(".", "")
            key = f"cu{normalized}"
            if key in VARIANT_LABELS:
                return key
            return "cu130"
    except Exception:
        pass

    return "cpu"


def _wheel_fits_this_pc(platform_tag: str) -> bool:
    """Does a wheel's platform tag (win_amd64, manylinux_2_28_x86_64, ...) fit this PC?"""
    machine = platform.machine().lower()
    arches = {"amd64": ("x86_64", "amd64"), "x86_64": ("x86_64", "amd64"),
              "arm64": ("arm64", "aarch64"), "aarch64": ("arm64", "aarch64")}.get(machine, (machine,))
    if sys.platform == "win32":
        system = platform_tag.startswith("win")
    elif sys.platform.startswith("linux"):
        system = platform_tag.startswith(("manylinux", "musllinux", "linux"))
    elif sys.platform == "darwin":
        system = platform_tag.startswith("macosx")
        if system and platform_tag.endswith("universal2"):
            return True
    else:
        system = False
    return system and platform_tag.endswith(tuple(f"_{a}" for a in arches))


def lock_has_pytorch_for_this_pc(source_root: Path, cuda_version: str) -> bool:
    """D3 (owner W57): does uv.lock hold a PyTorch wheel of this build for this PC?
    The same check as install.py's."""
    index = f'source = {{ registry = "{PYTORCH_INDEXES[LOCKED_EXTRAS.get(cuda_version, "cpu")]}" }}'
    text = (source_root / "uv.lock").read_text(encoding="utf-8")
    for block in text.split("[[package]]"):
        if not block.lstrip().startswith('name = "torch"\n') or index not in block:
            continue
        for wheel in re.findall(r'url = "[^"]+/([^/"]+)\.whl"', block):
            if _wheel_fits_this_pc(wheel.rsplit("-", 1)[-1]):
                return True
    return False


def unsupported_source_build_message(source_root: Path, cuda_version: str) -> str:
    """Why a source install's PyTorch build cannot be upgraded on this PC, or "" (D3)."""
    from whisperjav.__version__ import __version__
    if sys.platform.startswith("linux") and cuda_version == "cu130":
        return (f"This installation uses CUDA 13.0, which WhisperJAV {__version__} does not offer on Linux. "
                f"Nothing was changed. Run install.py again with --cuda cu128 (the Linux default) or --cpu-only.")
    if not lock_has_pytorch_for_this_pc(source_root, cuda_version):
        label = VARIANT_LABELS.get(LOCKED_EXTRAS.get(cuda_version, "cpu"), "CPU")
        return (f"PyTorch has no {label} build for this computer ({sys.platform}, {platform.machine()}) "
                f"in WhisperJAV {__version__}, so it cannot be upgraded here. Nothing was changed.")
    return ""


def _detect_target_env_for_upgrade() -> tuple:
    """Detect active environment for upgrade targeting.

    Returns:
        (env_path, is_external): env_path is the target environment path,
        or None to let uv use .venv. is_external indicates whether the
        env is one we didn't create (conda/venv) — used to decide --inexact.
    """
    # VIRTUAL_ENV first — more specific (user explicitly activated)
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and Path(venv).is_dir():
        return venv, True

    # CONDA_PREFIX second — may be base env
    conda = os.environ.get("CONDA_PREFIX")
    if conda and Path(conda).is_dir():
        return conda, True

    return None, False  # No active env → uv uses .venv


def _run_uv_sync_upgrade(source_root: Path, extras: str = "all") -> bool:
    """Run git pull + uv sync for source installs.

    Args:
        source_root: Path to project root (has pyproject.toml + uv.lock).
        extras: Comma-separated extras or "all".

    Returns:
        True if successful.
    """
    import shutil

    # D3 (owner W57): a build 1.10 does not offer on this PC stops before anything changes.
    cuda_version = _detect_cuda_for_source()
    unsupported = unsupported_source_build_message(source_root, cuda_version)
    if unsupported:
        print_error(unsupported)
        return False

    # Step 1: git pull
    print("      Pulling latest code from GitHub...")
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=str(source_root),
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            # Try rebase if ff-only fails
            result = subprocess.run(
                ["git", "pull", "--rebase"],
                cwd=str(source_root),
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                print_error(f"git pull failed: {result.stderr[:300]}")
                return False
        pull_msg = result.stdout.strip().split('\n')[-1] if result.stdout.strip() else "up to date"
        print_success(f"git pull: {pull_msg}")
    except subprocess.TimeoutExpired:
        print_error("git pull timed out")
        return False
    except FileNotFoundError:
        print_error("git not found")
        return False

    # Step 2: Detect target environment
    target_env, is_external = _detect_target_env_for_upgrade()

    # Step 3: uv sync
    uv_exe = shutil.which("uv")
    if not uv_exe:
        print_error("uv not found — install from https://docs.astral.sh/uv/")
        return False

    # Build environment with UV_PROJECT_ENVIRONMENT if targeting an active env
    env = dict(os.environ)
    if target_env:
        env["UV_PROJECT_ENVIRONMENT"] = target_env
        env_type = "conda" if os.environ.get("CONDA_PREFIX") == target_env else "venv"
        print(f"      Upgrading into {env_type} environment: {target_env}")

    # --frozen: the versions uv.lock records, never re-resolved (item 2f). The three
    # PyTorch builds are extras that cannot be combined, so never --all-extras.
    cmd = ["uv", "sync", "--frozen"]

    # --inexact for external envs: don't remove packages we didn't install
    if is_external:
        cmd.append("--inexact")

    for extra in extras.split(","):
        extra = extra.strip()
        if extra:
            cmd.extend(["--extra", extra])
    variant = LOCKED_EXTRAS.get(cuda_version, "cpu")
    cmd.extend(["--extra", variant])

    print(f"      Upgrading with release-pinned dependencies ({VARIANT_LABELS[variant]})...")
    print(f"      Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=str(source_root),
            env=env,
            capture_output=True, text=True, timeout=1800
        )
        if result.returncode != 0:
            print_error(f"uv sync failed: {(result.stderr or '')[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print_error("uv sync timed out")
        return False

    print_success("uv sync completed successfully")
    return True


# =============================================================================
# GPU Protection
# =============================================================================

def _gpu_constraint_args(install_dir: Path) -> list:
    """Return pip constraint args to protect GPU packages during upgrades.

    The initial installer creates ``gpu_constraints.txt`` in the environment
    prefix, pinning torch/torchaudio to their exact CUDA versions.  If we
    don't pass ``-c gpu_constraints.txt`` on every ``pip install -U``, pip
    may silently replace CUDA wheels with CPU-only versions from PyPI.

    Returns:
        ['-c', '<path>'] if constraints file exists, else [].
    """
    constraints_path = install_dir / "gpu_constraints.txt"
    if constraints_path.exists():
        return ['-c', str(constraints_path)]
    return []


# =============================================================================
# Cross-Platform Helpers
# =============================================================================

def _find_python(install_dir: Path) -> Optional[Path]:
    """Find the Python executable inside an installation directory.

    Works for:
      - conda-constructor installs (Windows): prefix/python.exe
      - pip virtualenvs (Windows):            prefix/Scripts/python.exe
      - pip virtualenvs (Linux/macOS):        prefix/bin/python
      - System installs (Linux):              prefix/bin/python3
    """
    candidates = [
        install_dir / 'python.exe',          # conda-constructor (Windows)
        install_dir / 'python',              # rare direct layout
        install_dir / 'Scripts' / 'python.exe',  # pip venv (Windows)
        install_dir / 'bin' / 'python',      # pip venv / system (Linux/macOS)
        install_dir / 'bin' / 'python3',     # system (Linux/macOS)
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_pip(install_dir: Path) -> Optional[Path]:
    """Find the pip executable inside an installation directory.

    Works for:
      - conda-constructor (Windows): prefix/Scripts/pip.exe
      - pip virtualenvs (Windows):   prefix/Scripts/pip.exe
      - pip virtualenvs (Linux):     prefix/bin/pip
    """
    candidates = [
        install_dir / 'Scripts' / 'pip.exe',
        install_dir / 'Scripts' / 'pip',
        install_dir / 'bin' / 'pip',
        install_dir / 'bin' / 'pip3',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
        pip_exe = _find_pip(install_dir)
        if pip_exe is None:
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
    pip_exe = _find_pip(install_dir)
    if pip_exe is None:
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

def _kill_process_tree(pid: int) -> None:
    """Kill a process and everything it started (pip's git fetch, build steps)."""
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30)
        return
    try:
        from whisperjav.utils.process_manager import terminate_process_tree
        terminate_process_tree(pid, timeout=5.0)
    except Exception:
        try:
            os.kill(pid, 9)
        except OSError:
            pass


def _run_bounded(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a command and give up after `timeout` seconds, for real.

    subprocess.run(capture_output=True, timeout=...) can wait far past its
    timeout on Windows: after killing the child it waits for the output pipes,
    which a grandchild (pip's git fetch or a build step) keeps open. Reproduced
    2026-10-06: a 2 s timeout returned after 20 s, i.e. when the grandchild
    ended; with pip that can be never (#436: "whisperjav-upgrade hangs in the
    compatibility check"). Here output goes to temporary files and on timeout
    the whole process tree is killed; TimeoutExpired is raised as before.
    """
    import tempfile
    with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdout=out, stderr=err)
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_process_tree(proc.pid)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise
        out.seek(0)
        err.seek(0)
        decode = lambda b: b.decode("utf-8", errors="replace")
        return subprocess.CompletedProcess(cmd, returncode, decode(out.read()), decode(err.read()))


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

    Supports:
      - conda-constructor installs (Windows): python.exe directly in prefix
      - pip virtualenvs (any platform): python in bin/ or Scripts/
      - System pip installs (Linux): python3 in bin/

    Returns:
        Path to installation directory, or None if not found
    """
    # Check if running from within a Python environment that has whisperjav
    if hasattr(sys, 'prefix'):
        install_dir = Path(sys.prefix)
        if _find_python(install_dir) is not None:
            # Verify it's a WhisperJAV installation
            try:
                import whisperjav  # noqa: F811
                return install_dir
            except ImportError:
                pass

    # Check default Windows installation location (conda-constructor)
    if IS_WINDOWS:
        local_app_data = os.environ.get('LOCALAPPDATA', '')
        if local_app_data:
            default_path = Path(local_app_data) / 'WhisperJAV'
            if _find_python(default_path) is not None:
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
    python_exe = _find_python(install_dir)
    if python_exe is None:
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


def update_launcher(install_dir: Path) -> bool:
    """
    Update the WhisperJAV-GUI.exe launcher in the installation root.

    Windows-only (conda-constructor installs). On Linux/macOS pip installs,
    console_scripts entry points handle launching, so this is a no-op.

    Args:
        install_dir: Path to installation directory

    Returns:
        True if successful, False otherwise
    """
    if not IS_WINDOWS:
        # Linux/macOS pip installs use console_scripts entry points — no launcher exe needed
        print_success("Launcher update skipped (not needed on this platform)")
        return True

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
    pip_exe = _find_pip(install_dir)
    if pip_exe is None:
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


def upgrade_package_with_extras(install_dir: Path, extras: str,
                                locked_lines: Optional[List[str]] = None) -> bool:
    """
    Upgrade the whisperjav package with specific extras.

    For source installs (has uv.lock), uses ``git pull && uv sync``.
    For conda-constructor installs, the release's wheel, then its pinned packages.

    Args:
        install_dir: Path to installation directory.
        extras: Comma-separated extras or "all" (source installs only).
        locked_lines: the pinned lines prepare_release_pins read; read here if None.

    Returns:
        True if successful, False otherwise.
    """
    # Source install path: git pull + uv sync
    source_root = _find_source_root()
    if source_root:
        print(f"      Source install detected — using git pull + uv sync [{extras}]")
        return _run_uv_sync_upgrade(source_root, extras=extras)

    # Conda-constructor / pip install path (item 2f): the new release's package without
    # dependencies, then its locked file for the build this PC has. The installer
    # installed every extra, and the locked file is the whole set (owner W57 INPUT2;
    # main() stops --extras other than all before this).
    stack = read_installed_stack(install_dir)
    if stack not in VARIANT_LABELS:
        print_error(STACK_NOT_RECORDED_TEXT.format(series=_series_label()))
        return False
    if locked_lines is None:
        locked_lines = prepare_release_pins(install_dir, stack, check_availability=False)
        if locked_lines is None:
            return False
    if not upgrade_package_wheel_only(install_dir):
        return False
    return install_locked_requirements(install_dir, stack, locked_lines)


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
  whisperjav-upgrade --extras cli,gui   # Upgrade specific extras (source installs only)
  whisperjav-upgrade --wheel-only       # Hot-patch mode (package only)
  whisperjav-upgrade --list-snapshots   # List available rollback snapshots
  whisperjav-upgrade --rollback         # Rollback to previous version

Available extras (source installs only):
  cli        - Audio processing, VAD, scene detection
  gui         - PyWebView GUI interface
  translate   - AI-powered subtitle translation
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
        help="Extras to upgrade (source installs only; installer installs always get the release's "
             "full pinned set)."
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
        if IS_WINDOWS:
            print("Please run this script from within the WhisperJAV environment:")
            print("  %LOCALAPPDATA%\\WhisperJAV\\python.exe -m whisperjav.upgrade")
        else:
            print("Please ensure whisperjav is installed in the active Python environment:")
            print("  pip install whisperjav          # then: whisperjav-upgrade")
            print("  # or from a virtualenv:")
            print("  source /path/to/venv/bin/activate && whisperjav-upgrade")
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

    # D4 (owner W57 INPUT2): an installer install upgrades to the release's full pinned
    # set; --extras cannot narrow it, so it is refused rather than ignored.
    is_source_install = _find_source_root() is not None
    if not is_source_install and not args.wheel_only and args.extras.strip().lower() != "all":
        print_error(EXTRAS_INSTALLER_TEXT)
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

    # Where the new version comes from: the newest release of this series (item 2e).
    # Source installs (a git clone) pull their own branch instead.
    global GITHUB_REPO
    if not is_source_install:
        source, why = resolve_upgrade_source(current_version)
        if source is None:
            print_error(f"Cannot upgrade: {why}.")
            return 1
        tag = source.rsplit("@", 1)[-1] if "@" in source else ""
        if (tag and current_version and not os.environ.get('WHISPERJAV_UPGRADE_REPO')
                and _series(tag) == _series(current_version)):
            from whisperjav.version_checker import compare_versions
            if compare_versions(tag, current_version) <= 0:
                print_success(f"WhisperJAV {current_version} is the newest {tag[1:].rsplit('.', 1)[0]} release.")
                return 0
        GITHUB_REPO = source
        print(f"Upgrade source: {why}")

    # Pre-upgrade check (D5, owner W57 INPUT3), installer installs only: the new
    # release's pinned list for this PC's build, conda's packages compared with it, and
    # the pinned install run as a dry run -- the same commands the upgrade then runs.
    # Source installs need none: `uv sync --frozen` checks the lock itself.
    locked_lines = None
    if not is_source_install and not args.wheel_only:
        stack = read_installed_stack(install_dir)
        if stack not in VARIANT_LABELS:
            print_error(STACK_NOT_RECORDED_TEXT.format(series=_series_label()))
            return 1
        print("\nChecking compatibility...")
        locked_lines = prepare_release_pins(install_dir, stack,
                                            check_availability=not args.skip_compat_check)
        if locked_lines is None:
            return 1

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

        if IS_WINDOWS:
            print_step(3, total_steps, "Updating desktop shortcut...")
            update_desktop_shortcut(install_dir, new_version)
        else:
            print_step(3, total_steps, "Finalizing...")
            print_success("No shortcut update needed on this platform")

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
    upgrade_success = upgrade_package_with_extras(install_dir, args.extras, locked_lines)

    if not upgrade_success:
        print_error("Upgrade failed.")
        if snapshot_path:
            print("\nAttempting rollback to pre-upgrade state...")
            if rollback_to_snapshot(install_dir, snapshot_path):
                print_success("Rollback successful - installation restored")
            else:
                print_error("Rollback failed - installation may be in inconsistent state")
        return 1

    # Step 2: the versions are the release's locked ones (step 1, item 2f); the old
    # numpy/librosa "newest" step is gone. cuBLAS 12 is placed again on CUDA 13.0.
    print_step(2, total_steps, "Placing the CUDA 12 library for faster-whisper...")
    if IS_WINDOWS and read_installed_stack(install_dir) == "cu130":
        place_cublas12_after_upgrade(install_dir)
    else:
        print_success("Nothing to do for this PyTorch build.")

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

    # Step 5: Update shortcut (Windows) and cleanup
    if IS_WINDOWS:
        print_step(5, total_steps, "Updating desktop shortcut and cleaning up...")
        update_desktop_shortcut(install_dir, new_version)
    else:
        print_step(5, total_steps, "Cleaning up...")

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
    if IS_WINDOWS:
        print("  You can now launch WhisperJAV from your desktop shortcut.")
    else:
        print("  You can now run WhisperJAV with: whisperjav <video>")
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
