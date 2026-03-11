"""
Tests that check WhisperJAV dependency compatibility with numpy 2.x.

These tests inspect pyproject.toml and the installer registry to identify
dependencies that may or may not support numpy 2. They also check for known
version constraints.

No imports of WhisperJAV modules or heavy dependencies are needed.
"""

import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
REGISTRY_PATH = REPO_ROOT / "whisperjav" / "installer" / "core" / "registry.py"


# ---------------------------------------------------------------------------
# Known numpy 2 compatibility status of WhisperJAV's dependencies
# ---------------------------------------------------------------------------
# Status: "yes" = supports numpy 2, "no" = does not, "partial" = some versions,
#         "unknown" = not verified, "n/a" = doesn't depend on numpy
#
# Sources: PyPI, GitHub issues, release notes, CI testing (as of 2025-05)

DEPENDENCY_NUMPY2_STATUS: Dict[str, Dict] = {
    # Core ASR engines
    "openai-whisper": {
        "status": "partial",
        "min_version": "latest (main branch)",
        "notes": "whisper uses numpy internally for audio processing. "
                 "The main branch supports numpy 2 as of late 2024. "
                 "Older pip releases (20231117) may not.",
        "risk": "MEDIUM",
    },
    "stable-ts": {
        "status": "partial",
        "min_version": "latest (main branch)",
        "notes": "Fork from meizhong986. Depends on openai-whisper which "
                 "uses numpy. Should follow whisper's numpy 2 compatibility.",
        "risk": "MEDIUM",
    },
    "faster-whisper": {
        "status": "yes",
        "min_version": "1.1.0",
        "notes": "faster-whisper 1.1.0+ supports numpy 2. Uses CTranslate2 "
                 "which also supports numpy 2 since v4.5.0.",
        "risk": "LOW",
    },

    # PyTorch ecosystem
    "torch": {
        "status": "yes",
        "min_version": "2.3.0",
        "notes": "PyTorch 2.3+ officially supports numpy 2. "
                 "Earlier versions may work but are not officially tested.",
        "risk": "LOW",
    },
    "torchaudio": {
        "status": "yes",
        "min_version": "2.3.0",
        "notes": "Follows PyTorch version. 2.3+ supports numpy 2.",
        "risk": "LOW",
    },
    "torchvision": {
        "status": "yes",
        "min_version": "0.18.0",
        "notes": "Follows PyTorch version. Supports numpy 2 when paired "
                 "with torch 2.3+.",
        "risk": "LOW",
    },

    # Scientific stack
    "numpy": {
        "status": "yes",
        "min_version": "2.0.0",
        "notes": "This IS the package being upgraded. Currently pinned <2.0.",
        "risk": "N/A",
    },
    "scipy": {
        "status": "yes",
        "min_version": "1.12.0",
        "notes": "scipy 1.12+ supports numpy 2. WhisperJAV requires >=1.10.1.",
        "risk": "LOW",
    },
    "numba": {
        "status": "yes",
        "min_version": "0.59.0",
        "notes": "numba 0.59+ supports numpy 2. WhisperJAV requires >=0.58.0. "
                 "CRITICAL: numba must be built against numpy 2 ABI. "
                 "Binary incompatibility if numba was compiled with numpy 1.x.",
        "risk": "HIGH",
    },

    # Audio processing
    "librosa": {
        "status": "yes",
        "min_version": "0.10.2",
        "notes": "librosa 0.10.2+ supports numpy 2. WhisperJAV requires >=0.10.0.",
        "risk": "LOW",
    },
    "soundfile": {
        "status": "yes",
        "min_version": "0.12.0",
        "notes": "soundfile has minimal numpy dependency (just for array I/O). "
                 "Works with numpy 2.",
        "risk": "LOW",
    },
    "pyloudnorm": {
        "status": "yes",
        "min_version": "0.1.0",
        "notes": "Simple numpy usage (arithmetic). Should work with numpy 2.",
        "risk": "LOW",
    },
    "pydub": {
        "status": "n/a",
        "min_version": "any",
        "notes": "pydub does not depend on numpy.",
        "risk": "NONE",
    },

    # VAD
    "silero-vad": {
        "status": "yes",
        "min_version": "5.0",
        "notes": "Uses torch tensors primarily, minimal numpy interaction.",
        "risk": "LOW",
    },
    "ten-vad": {
        "status": "unknown",
        "min_version": "unknown",
        "notes": "Relatively new package. Uses numpy for audio conversion. "
                 "Needs testing.",
        "risk": "MEDIUM",
    },
    "auditok": {
        "status": "yes",
        "min_version": "0.2.0",
        "notes": "auditok uses numpy for audio array operations. "
                 "Basic operations (array, astype) are numpy 2 compatible.",
        "risk": "LOW",
    },

    # Enhancement backends
    "modelscope": {
        "status": "partial",
        "min_version": "1.20",
        "notes": "ModelScope has complex numpy usage. Some internal modules "
                 "may use deprecated numpy APIs. Needs thorough testing. "
                 "ZipEnhancer model pipeline goes through ModelScope.",
        "risk": "HIGH",
    },
    "clearvoice": {
        "status": "unknown",
        "min_version": "unknown",
        "notes": "ClearVoice fork from meizhong986. Uses numpy for audio "
                 "processing. Needs testing with numpy 2.",
        "risk": "MEDIUM",
    },

    # ML/Data
    "scikit-learn": {
        "status": "yes",
        "min_version": "1.4.0",
        "notes": "scikit-learn 1.4+ supports numpy 2. WhisperJAV requires >=1.3.0.",
        "risk": "LOW",
    },
    "datasets": {
        "status": "yes",
        "min_version": "3.0.0",
        "notes": "HuggingFace datasets 3.0+ supports numpy 2.",
        "risk": "LOW",
    },

    # Translation (no numpy dependency)
    "pysubtrans": {
        "status": "n/a",
        "min_version": "any",
        "notes": "Text processing only, no numpy dependency.",
        "risk": "NONE",
    },
    "openai": {
        "status": "n/a",
        "min_version": "any",
        "notes": "API client, no numpy dependency.",
        "risk": "NONE",
    },

    # Other
    "opencv-python": {
        "status": "yes",
        "min_version": "4.9.0",
        "notes": "OpenCV 4.9+ supports numpy 2. WhisperJAV requires >=4.0.",
        "risk": "MEDIUM",
    },
    "einops": {
        "status": "yes",
        "min_version": "0.7.0",
        "notes": "einops 0.7+ supports numpy 2.",
        "risk": "LOW",
    },
}


def _read_pyproject() -> str:
    """Read pyproject.toml content."""
    return PYPROJECT_PATH.read_text(encoding="utf-8")


def _extract_dependencies(content: str) -> List[str]:
    """Extract all dependency specifications from pyproject.toml."""
    deps = []
    # Match lines that look like dependency specs
    for line in content.splitlines():
        line = line.strip().strip('"').strip("'").strip(",")
        if not line or line.startswith("#") or line.startswith("["):
            continue
        # Match package names at start of line (pip-style specs)
        match = re.match(r'^([a-zA-Z][a-zA-Z0-9_.-]*)', line)
        if match:
            pkg = match.group(1).lower()
            if pkg in ("pip", "install", "requires", "python"):
                continue
            deps.append(line)
    return deps


def _find_numpy_constraint(content: str) -> Optional[str]:
    """Find the numpy version constraint in pyproject.toml."""
    for line in content.splitlines():
        if "numpy" in line.lower() and (">=" in line or "<" in line or "==" in line):
            return line.strip().strip('"').strip("'").strip(",")
    return None


# ===========================================================================
# Tests
# ===========================================================================


class TestNumpyVersionConstraint:
    """Verify current numpy version pinning."""

    def test_numpy_is_pinned_below_2(self):
        """Confirm that numpy is currently pinned to <2.0."""
        content = _read_pyproject()
        constraint = _find_numpy_constraint(content)
        assert constraint is not None, "No numpy constraint found in pyproject.toml"
        assert "<2.0" in constraint or "<2" in constraint, (
            f"Expected numpy to be pinned <2.0, got: {constraint}"
        )

    def test_numpy_constraint_details(self):
        """Document the exact numpy constraint and its location."""
        content = _read_pyproject()
        constraint = _find_numpy_constraint(content)
        print(f"Current numpy constraint: {constraint}")

        # Also check requirements.txt
        if REQUIREMENTS_PATH.exists():
            req_content = REQUIREMENTS_PATH.read_text(encoding="utf-8")
            req_constraint = None
            for line in req_content.splitlines():
                if "numpy" in line.lower() and (">=" in line or "<" in line):
                    req_constraint = line.strip()
                    break
            if req_constraint:
                print(f"requirements.txt constraint: {req_constraint}")

        # Check registry
        if REGISTRY_PATH.exists():
            reg_content = REGISTRY_PATH.read_text(encoding="utf-8")
            for i, line in enumerate(reg_content.splitlines(), 1):
                if "numpy" in line.lower() and ("version" in line.lower() or "<" in line or ">=" in line):
                    print(f"Registry constraint (line {i}): {line.strip()}")


class TestDependencyNumpyCompatibility:
    """Check each dependency's numpy 2 compatibility status."""

    @pytest.mark.parametrize(
        "dep_name,info",
        list(DEPENDENCY_NUMPY2_STATUS.items()),
        ids=list(DEPENDENCY_NUMPY2_STATUS.keys()),
    )
    def test_dependency_compatibility(self, dep_name: str, info: Dict):
        """
        Document each dependency's numpy 2 compatibility.
        Fails for HIGH risk dependencies to draw attention.
        """
        status = info["status"]
        risk = info["risk"]
        notes = info["notes"]

        print(f"\n{dep_name}:")
        print(f"  Status: {status}")
        print(f"  Risk: {risk}")
        print(f"  Min version for numpy 2: {info['min_version']}")
        print(f"  Notes: {notes}")

        if risk == "HIGH":
            pytest.xfail(
                f"{dep_name} is HIGH risk for numpy 2 migration: {notes}"
            )

    def test_compatibility_summary(self):
        """Produce a summary table of dependency numpy 2 compatibility."""
        report = [
            "Dependency Numpy 2 Compatibility Matrix",
            "=" * 70,
            f"{'Package':<25} {'Status':<10} {'Risk':<8} {'Min Version':<15}",
            "-" * 70,
        ]

        by_risk = {"HIGH": [], "MEDIUM": [], "LOW": [], "NONE": [], "N/A": []}

        for dep, info in sorted(DEPENDENCY_NUMPY2_STATUS.items()):
            status = info["status"]
            risk = info["risk"]
            min_ver = info["min_version"]
            report.append(f"{dep:<25} {status:<10} {risk:<8} {min_ver:<15}")
            by_risk.setdefault(risk, []).append(dep)

        report.extend([
            "",
            "Risk Summary:",
            f"  HIGH:   {', '.join(sorted(by_risk.get('HIGH', [])))}",
            f"  MEDIUM: {', '.join(sorted(by_risk.get('MEDIUM', [])))}",
            f"  LOW:    {', '.join(sorted(by_risk.get('LOW', [])))}",
            f"  NONE:   {', '.join(sorted(by_risk.get('NONE', [])))}",
        ])

        print("\n".join(report))


class TestMinimumVersionRequirements:
    """Check if WhisperJAV's dependency version constraints allow numpy 2 compatible versions."""

    def test_scipy_version_allows_numpy2(self):
        """scipy >=1.12.0 needed for numpy 2. WhisperJAV requires >=1.10.1."""
        content = _read_pyproject()
        match = re.search(r'scipy>=([\d.]+)', content)
        if match:
            version = match.group(1)
            parts = [int(x) for x in version.split(".")]
            if parts < [1, 12, 0]:
                pytest.xfail(
                    f"scipy>={version} allows versions that don't support numpy 2. "
                    f"Need scipy>=1.12.0 for numpy 2 compatibility."
                )

    def test_numba_version_allows_numpy2(self):
        """numba >=0.59.0 needed for numpy 2. WhisperJAV requires >=0.58.0."""
        content = _read_pyproject()
        match = re.search(r'numba>=([\d.]+)', content)
        if match:
            version = match.group(1)
            parts = [int(x) for x in version.split(".")]
            if parts < [0, 59, 0]:
                pytest.xfail(
                    f"numba>={version} allows versions that don't support numpy 2. "
                    f"Need numba>=0.59.0 for numpy 2 compatibility."
                )

    def test_librosa_version_allows_numpy2(self):
        """librosa >=0.10.2 needed for numpy 2. WhisperJAV requires >=0.10.0."""
        content = _read_pyproject()
        match = re.search(r'librosa>=([\d.]+)', content)
        if match:
            version = match.group(1)
            parts = [int(x) for x in version.split(".")]
            if parts < [0, 10, 2]:
                pytest.xfail(
                    f"librosa>={version} allows versions that don't support numpy 2. "
                    f"Need librosa>=0.10.2 for numpy 2 compatibility."
                )

    def test_scikit_learn_version_allows_numpy2(self):
        """scikit-learn >=1.4.0 needed for numpy 2. WhisperJAV requires >=1.3.0."""
        content = _read_pyproject()
        match = re.search(r'scikit-learn>=([\d.]+)', content)
        if match:
            version = match.group(1)
            parts = [int(x) for x in version.split(".")]
            if parts < [1, 4, 0]:
                pytest.xfail(
                    f"scikit-learn>={version} allows versions that don't support numpy 2. "
                    f"Need scikit-learn>=1.4.0 for numpy 2 compatibility."
                )

    def test_opencv_version_allows_numpy2(self):
        """opencv-python >=4.9.0 needed for numpy 2. WhisperJAV requires >=4.0."""
        content = _read_pyproject()
        match = re.search(r'opencv-python>=([\d.]+)', content)
        if match:
            version = match.group(1)
            parts = [int(x) for x in version.split(".")]
            if parts < [4, 9, 0]:
                pytest.xfail(
                    f"opencv-python>={version} allows versions that don't support numpy 2. "
                    f"Need opencv-python>=4.9.0 for numpy 2 compatibility."
                )


class TestPyvideotransNote:
    """Check the pyvideotrans compatibility note."""

    def test_pyvideotrans_constraint_documented(self):
        """
        The numpy <2.0 pin cites 'pyvideotrans compatibility' as the reason.
        Verify this is documented and check if pyvideotrans is still a concern.
        """
        content = _read_pyproject()
        has_pyvideotrans_note = "pyvideotrans" in content.lower()
        assert has_pyvideotrans_note, (
            "Expected pyvideotrans compatibility note in pyproject.toml "
            "next to numpy constraint"
        )

        # Check if pyvideotrans is actually used
        pyvideotrans_import = False
        for dirpath, dirnames, filenames in os.walk(REPO_ROOT / "whisperjav"):
            dirnames[:] = [d for d in dirnames if d != "__pycache__"]
            for f in filenames:
                if f.endswith(".py"):
                    fpath = Path(dirpath) / f
                    try:
                        text = fpath.read_text(encoding="utf-8", errors="replace")
                        if "pyvideotrans" in text:
                            pyvideotrans_import = True
                            break
                    except Exception:
                        pass
            if pyvideotrans_import:
                break

        if not pyvideotrans_import:
            print(
                "NOTE: 'pyvideotrans' is cited as the reason for numpy<2.0 pin, "
                "but no pyvideotrans import was found in the codebase. "
                "The constraint may be outdated."
            )


class TestRegistryNumpyConstraint:
    """Check the installer registry for numpy constraints."""

    def test_registry_matches_pyproject(self):
        """Verify installer registry has same numpy constraint as pyproject.toml."""
        if not REGISTRY_PATH.exists():
            pytest.skip("Registry file not found")

        pyproject_content = _read_pyproject()
        registry_content = REGISTRY_PATH.read_text(encoding="utf-8")

        # Extract numpy version from both
        pyproject_numpy = _find_numpy_constraint(pyproject_content)
        registry_numpy = None
        for line in registry_content.splitlines():
            if '"numpy"' in line or "'numpy'" in line:
                # Look at nearby lines for version
                continue
            if "numpy" in line and (">=1.26" in line or "<2.0" in line):
                registry_numpy = line.strip()
                break

        # Also search for version= near name="numpy"
        match = re.search(
            r'name\s*=\s*["\']numpy["\'].*?version\s*=\s*["\']([^"\']+)["\']',
            registry_content,
            re.DOTALL,
        )
        if match:
            registry_numpy = f"numpy{match.group(1)}"

        print(f"pyproject.toml: {pyproject_numpy}")
        print(f"registry.py:   {registry_numpy}")

        if pyproject_numpy and registry_numpy:
            # Both should contain <2.0
            assert "<2.0" in (registry_numpy or "") or "<2" in (registry_numpy or ""), (
                f"Registry numpy constraint doesn't match pyproject.toml. "
                f"Registry: {registry_numpy}, pyproject: {pyproject_numpy}"
            )


import os
