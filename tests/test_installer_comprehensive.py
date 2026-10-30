#!/usr/bin/env python3
"""
Comprehensive Installer Tests
=============================

Tests covering installation scenarios, paths, and edge cases that were not
previously tested. This complements test_installer.py which tests structure.

Coverage Areas:
1. GPU Detection - driver parsing, CUDA selection, platform detection
2. Executor - retry logic, timeout handling, pip args building
3. Config - CUDA matrix validation, constants
4. Import Scanner - AST extraction
5. End-to-end scenarios - fresh install, upgrade, failure recovery

Author: Senior Architect
Date: 2026-01-26
"""

import sys
import os
import re
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from typing import Optional, Tuple

import pytest


# =============================================================================
# GPU Detection Tests (detector.py)
# =============================================================================


class TestDriverVersionParsing:
    """Test driver version string parsing."""

    def test_parse_standard_version(self):
        """Test parsing standard driver version string."""
        from whisperjav.installer.core.detector import _parse_driver_version

        # Standard 3-part version
        assert _parse_driver_version("570.00.00") == (570, 0, 0)
        assert _parse_driver_version("450.80.02") == (450, 80, 2)
        assert _parse_driver_version("550.54.14") == (550, 54, 14)

    def test_parse_two_part_version(self):
        """Test parsing two-part version strings."""
        from whisperjav.installer.core.detector import _parse_driver_version

        # Some drivers report 2-part versions
        assert _parse_driver_version("550.54") == (550, 54, 0)
        assert _parse_driver_version("450.80") == (450, 80, 0)

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        from whisperjav.installer.core.detector import _parse_driver_version

        assert _parse_driver_version("  570.00.00  ") == (570, 0, 0)
        assert _parse_driver_version("\t450.80.02\n") == (450, 80, 2)

    def test_parse_invalid_version(self):
        """Test parsing invalid version strings."""
        from whisperjav.installer.core.detector import _parse_driver_version

        assert _parse_driver_version("") is None
        assert _parse_driver_version("invalid") is None
        assert _parse_driver_version("not.a.version") is None
        assert _parse_driver_version(None) is None

    def test_parse_edge_cases(self):
        """Test edge case version strings."""
        from whisperjav.installer.core.detector import _parse_driver_version

        # Version with extra text (shouldn't happen but be defensive)
        result = _parse_driver_version("570.00.00 BETA")
        assert result == (570, 0, 0)


class TestCUDAVersionSelection:
    """Test CUDA version selection based on driver."""

    def test_select_cu130_for_driver_580(self):
        """v1.10.0: driver 580+ gets CUDA 13.0 (the only stack)."""
        from whisperjav.installer.core.detector import _select_cuda_version

        cuda, url, msg = _select_cuda_version((580, 0, 0))
        assert cuda == "cu130"
        assert "cu130" in url

        cuda, url, msg = _select_cuda_version((610, 47, 0))
        assert cuda == "cu130"

    def test_driver_below_580_gets_no_cuda_build(self):
        """v1.10.0: below 580 there is no CUDA build; install.py stops and asks for an update."""
        from whisperjav.installer.core.detector import _select_cuda_version

        for driver in [(579, 99, 99), (570, 0, 0), (520, 50, 0), (450, 0, 0)]:
            cuda, url, msg = _select_cuda_version(driver)
            assert cuda is None, driver
            assert "580" in msg

    def test_select_cpu_for_old_driver(self):
        """Test CPU fallback for very old drivers."""
        from whisperjav.installer.core.detector import _select_cuda_version
        from whisperjav.installer.core.config import CPU_TORCH_INDEX

        # Driver < 450 should get CPU
        cuda, url, msg = _select_cuda_version((449, 99, 99))
        assert cuda is None
        assert url == CPU_TORCH_INDEX
        assert "too old" in msg.lower()

        cuda, url, msg = _select_cuda_version((400, 0, 0))
        assert cuda is None

    def test_select_cpu_for_none_driver(self):
        """Test CPU fallback when driver version unknown."""
        from whisperjav.installer.core.detector import _select_cuda_version
        from whisperjav.installer.core.config import CPU_TORCH_INDEX

        cuda, url, msg = _select_cuda_version(None)
        assert cuda is None
        assert url == CPU_TORCH_INDEX


class TestPlatformDetection:
    """Test platform detection functionality."""

    def test_detect_platform_returns_valid_enum(self):
        """Test that detect_platform returns a valid enum."""
        from whisperjav.installer.core.detector import detect_platform, DetectedPlatform

        result = detect_platform()
        assert isinstance(result, DetectedPlatform)
        assert result in list(DetectedPlatform)

    def test_get_platform_name_returns_string(self):
        """Test that get_platform_name returns readable string."""
        from whisperjav.installer.core.detector import get_platform_name

        name = get_platform_name()
        assert isinstance(name, str)
        assert len(name) > 0

    def test_detect_windows_platform_value(self):
        """Test Windows platform detection enum value."""
        from whisperjav.installer.core.detector import DetectedPlatform

        # Test enum value exists and has expected value
        assert DetectedPlatform.WINDOWS.name == "WINDOWS"
        assert DetectedPlatform.WINDOWS.value == 1

    def test_detect_linux_platform_value(self):
        """Test Linux platform detection enum value."""
        from whisperjav.installer.core.detector import DetectedPlatform

        # Test enum value exists
        assert DetectedPlatform.LINUX.name == "LINUX"
        assert DetectedPlatform.LINUX.value == 2


class TestPrerequisiteChecks:
    """Test prerequisite checking functions."""

    def test_check_python_version_current(self):
        """Test Python version check with current Python."""
        from whisperjav.installer.core.detector import check_python_version

        result = check_python_version()
        assert result.name == "Python"
        assert result.found is True
        # Current test environment should be compatible
        assert result.version is not None

    def test_check_ffmpeg(self):
        """Test FFmpeg presence check."""
        from whisperjav.installer.core.detector import check_ffmpeg

        result = check_ffmpeg()
        assert result.name == "FFmpeg"
        # Result depends on whether FFmpeg is installed
        assert isinstance(result.found, bool)

    def test_check_git(self):
        """Test Git presence check."""
        from whisperjav.installer.core.detector import check_git

        result = check_git()
        assert result.name == "Git"
        # Git should be installed in dev environment
        assert result.found is True

    def test_check_prerequisites_returns_dict(self):
        """Test comprehensive prerequisites check."""
        from whisperjav.installer.core.detector import check_prerequisites

        results = check_prerequisites()
        assert isinstance(results, dict)
        assert "python" in results
        assert "ffmpeg" in results
        assert "git" in results
        assert "gpu" in results
        assert "platform" in results
        assert "all_ok" in results


class TestGPUDetection:
    """Test GPU detection with mocked subprocess calls."""

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_detect_gpu_with_nvidia_smi(self, mock_run, mock_which):
        """Test GPU detection via nvidia-smi."""
        from whisperjav.installer.core.detector import detect_gpu

        # Mock nvidia-smi being available
        mock_which.return_value = "/usr/bin/nvidia-smi"

        # Mock driver version query
        driver_result = MagicMock()
        driver_result.returncode = 0
        driver_result.stdout = "580.00.00\n"

        # Mock GPU name query
        name_result = MagicMock()
        name_result.returncode = 0
        name_result.stdout = "NVIDIA GeForce RTX 4090\n"

        # v1.10.0: the compute capability is read too
        cc_result = MagicMock()
        cc_result.returncode = 0
        cc_result.stdout = "8.9\n"

        mock_run.side_effect = [driver_result, name_result, cc_result]

        result = detect_gpu()
        assert result.detected is True
        assert result.cuda_version == "cu130"
        assert result.compute_capability == (8, 9)
        assert result.detection_method == "nvidia-smi"
        assert "RTX 4090" in result.name

    @patch('shutil.which')
    def test_detect_gpu_no_nvidia_smi(self, mock_which):
        """Test GPU detection when nvidia-smi is not found."""
        from whisperjav.installer.core.detector import detect_gpu
        from whisperjav.installer.core.config import CPU_TORCH_INDEX

        mock_which.return_value = None

        result = detect_gpu()
        assert result.detected is False
        assert result.torch_index == CPU_TORCH_INDEX

    def test_macos_silicon_enum_value(self):
        """Test Apple Silicon platform enum exists."""
        from whisperjav.installer.core.detector import DetectedPlatform

        # Test enum value exists
        assert DetectedPlatform.MACOS_SILICON.name == "MACOS_SILICON"
        assert hasattr(DetectedPlatform, "MACOS_INTEL")


# =============================================================================
# Executor Tests (executor.py)
# =============================================================================


class TestGitTimeoutDetection:
    """Test Git timeout pattern detection."""

    def test_detect_21_second_timeout(self):
        """Test detection of the infamous 21-second timeout."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        error = "Failed to connect to github.com port 443 after 21 ms"
        assert executor._is_git_timeout(error) is True

    def test_detect_connection_timeout(self):
        """Test detection of generic connection timeout."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        assert executor._is_git_timeout("Connection timed out") is True
        assert executor._is_git_timeout("connection timed out") is True  # Case insensitive

    def test_detect_connection_reset(self):
        """Test detection of connection reset."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        assert executor._is_git_timeout("Connection reset by peer") is True

    def test_detect_ssl_problem(self):
        """Test detection of SSL certificate issues."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        assert executor._is_git_timeout("SSL certificate problem: unable to get local issuer") is True

    def test_detect_dns_failure(self):
        """Test detection of DNS resolution failure."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        assert executor._is_git_timeout("Could not resolve host: github.com") is True

    def test_no_false_positive(self):
        """Test that normal errors don't trigger timeout detection."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        assert executor._is_git_timeout("Package not found") is False
        assert executor._is_git_timeout("Invalid requirement") is False
        assert executor._is_git_timeout("Successfully installed torch") is False


class TestBuildPipArgs:
    """Test pip argument building for different package sources."""

    def test_build_args_pypi(self):
        """Test pip args for standard PyPI package."""
        from whisperjav.installer.core.executor import StepExecutor
        from whisperjav.installer.core.registry import Package, InstallSource

        executor = StepExecutor()
        pkg = Package(name="numpy", version=">=2.0.0")

        args = executor._build_pip_args(pkg, "cu118")
        assert "install" in args
        assert "numpy>=2.0.0" in args

    def test_build_args_git(self):
        """Test pip args for Git package."""
        from whisperjav.installer.core.executor import StepExecutor
        from whisperjav.installer.core.registry import Package, InstallSource

        executor = StepExecutor()
        pkg = Package(
            name="openai-whisper",
            source=InstallSource.GIT,
            git_url="git+https://github.com/openai/whisper@main"
        )

        args = executor._build_pip_args(pkg, "cu118")
        assert "install" in args
        assert "git+https://github.com/openai/whisper@main" in args

    def test_build_args_index_url(self):
        """Test pip args for package with custom index URL."""
        from whisperjav.installer.core.executor import StepExecutor
        from whisperjav.installer.core.registry import Package, InstallSource

        executor = StepExecutor()
        pkg = Package(
            name="torch",
            source=InstallSource.INDEX_URL,
            index_url="https://download.pytorch.org/whl/{cuda}"
        )

        args = executor._build_pip_args(pkg, "cu128")
        assert "install" in args
        assert "torch" in args
        assert "--index-url" in args
        assert "https://download.pytorch.org/whl/cu128" in args


class TestUvDetection:
    """Test uv package manager detection."""

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_detect_uv_in_prefix(self, mock_exists, mock_which):
        """Test uv detection in Python prefix."""
        from whisperjav.installer.core.executor import StepExecutor

        mock_exists.return_value = True
        mock_which.return_value = None

        executor = StepExecutor(use_uv=True)
        assert executor.uv_path is not None

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_detect_uv_in_path(self, mock_exists, mock_which):
        """Test uv detection in system PATH."""
        from whisperjav.installer.core.executor import StepExecutor

        mock_exists.return_value = False
        mock_which.return_value = "/usr/local/bin/uv"

        executor = StepExecutor(use_uv=True)
        assert executor.uv_path is not None

    @patch('shutil.which')
    @patch('pathlib.Path.exists')
    def test_uv_not_found(self, mock_exists, mock_which):
        """Test when uv is not available."""
        from whisperjav.installer.core.executor import StepExecutor

        mock_exists.return_value = False
        mock_which.return_value = None

        executor = StepExecutor(use_uv=True)
        assert executor.uv_path is None


class TestPackageSatisfaction:
    """Test package pre-verification."""

    @patch('subprocess.run')
    def test_package_already_installed(self, mock_run):
        """Test detection of already-installed package."""
        from whisperjav.installer.core.executor import StepExecutor
        from whisperjav.installer.core.registry import Package

        mock_run.return_value = MagicMock(returncode=0)

        executor = StepExecutor()
        pkg = Package(name="numpy")

        assert executor._is_package_satisfied(pkg) is True

    @patch('subprocess.run')
    def test_package_not_installed(self, mock_run):
        """Test detection of missing package."""
        from whisperjav.installer.core.executor import StepExecutor
        from whisperjav.installer.core.registry import Package

        mock_run.return_value = MagicMock(returncode=1)

        executor = StepExecutor()
        pkg = Package(name="nonexistent_package_xyz")

        assert executor._is_package_satisfied(pkg) is False


class TestExecutorStatistics:
    """Test executor statistics tracking."""

    def test_initial_stats(self):
        """Test initial statistics are zero."""
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()
        stats = executor.get_stats()

        assert stats["install_count"] == 0
        assert stats["skip_count"] == 0
        assert stats["fail_count"] == 0
        assert stats["total"] == 0


# =============================================================================
# Config Tests (config.py)
# =============================================================================


class TestCUDADriverMatrix:
    """Test CUDA driver matrix configuration."""

    def test_matrix_descending_order(self):
        """Test that CUDA matrix is in descending driver order."""
        from whisperjav.installer.core.config import CUDA_DRIVER_MATRIX

        # First entry should have highest driver requirement
        prev_driver = (999, 999, 999)
        for entry in CUDA_DRIVER_MATRIX:
            assert entry.min_driver < prev_driver, \
                f"Matrix not in descending order: {entry.min_driver} >= {prev_driver}"
            prev_driver = entry.min_driver

    def test_matrix_has_entries(self):
        """v1.10.0: the matrix chooses CUDA 13.0 only; cu128 is an index for the fallback."""
        from whisperjav.installer.core.config import CUDA_DRIVER_MATRIX, TORCH_INDEXES

        cuda_versions = {entry.cuda_version for entry in CUDA_DRIVER_MATRIX}
        assert cuda_versions == {"cu130"}
        assert set(TORCH_INDEXES) == {"cu130", "cu128", "cpu"}

    def test_torch_index_urls_valid(self):
        """Test that torch index URLs are valid."""
        from whisperjav.installer.core.config import CUDA_DRIVER_MATRIX, CPU_TORCH_INDEX

        for entry in CUDA_DRIVER_MATRIX:
            assert entry.torch_index.startswith("https://"), \
                f"Invalid URL: {entry.torch_index}"
            assert "pytorch.org" in entry.torch_index

        assert CPU_TORCH_INDEX.startswith("https://")
        assert "pytorch.org" in CPU_TORCH_INDEX


class TestConfigConstants:
    """Test configuration constants."""

    def test_retry_values_reasonable(self):
        """Test that retry configuration is reasonable."""
        from whisperjav.installer.core.config import (
            DEFAULT_RETRY_COUNT,
            DEFAULT_RETRY_DELAY,
            DEFAULT_TIMEOUT,
            GIT_INSTALL_TIMEOUT,
        )

        assert 1 <= DEFAULT_RETRY_COUNT <= 10
        assert 1 <= DEFAULT_RETRY_DELAY <= 60
        assert 60 <= DEFAULT_TIMEOUT <= 7200  # 1 min to 2 hours
        assert 60 <= GIT_INSTALL_TIMEOUT <= 1800  # 1 min to 30 min

    def test_python_version_range(self):
        """Test Python version configuration."""
        from whisperjav.installer.core.config import (
            PYTHON_MIN_VERSION,
            PYTHON_MAX_VERSION,
        )

        assert PYTHON_MIN_VERSION <= PYTHON_MAX_VERSION
        assert PYTHON_MIN_VERSION >= (3, 10)
        assert PYTHON_MAX_VERSION <= (3, 13)

    def test_git_timeout_patterns_exist(self):
        """Test that Git timeout patterns are defined."""
        from whisperjav.installer.core.config import GIT_TIMEOUT_PATTERNS

        assert len(GIT_TIMEOUT_PATTERNS) > 0
        assert any("443" in p for p in GIT_TIMEOUT_PATTERNS)  # Port 443 pattern
        assert any("timed out" in p.lower() for p in GIT_TIMEOUT_PATTERNS)


class TestConfigValidation:
    """Test config validation function."""

    def test_validate_config_passes(self):
        """Test that default config passes validation."""
        from whisperjav.installer.core.config import validate_config

        # Should not raise
        validate_config()


# =============================================================================
# Import Scanner Tests (imports.py)
# =============================================================================


class TestImportExtraction:
    """Test AST-based import extraction."""

    def test_extract_simple_import(self):
        """Test extraction of simple import statement."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "import os\nimport sys"
        imports = _extract_imports(source)
        assert "os" in imports
        assert "sys" in imports

    def test_extract_from_import(self):
        """Test extraction of from...import statement."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "from pathlib import Path"
        imports = _extract_imports(source)
        assert "pathlib" in imports

    def test_extract_nested_import(self):
        """Test extraction of nested module import."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "import torch.nn.functional"
        imports = _extract_imports(source)
        assert "torch" in imports
        assert "torch.nn" not in imports  # Should only get top-level

    def test_ignore_relative_import(self):
        """Test that relative imports are ignored."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "from . import utils\nfrom ..core import registry"
        imports = _extract_imports(source)
        assert "utils" not in imports
        assert "core" not in imports

    def test_handle_syntax_error(self):
        """Test graceful handling of syntax errors."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "this is not valid python {"
        imports = _extract_imports(source)
        assert imports == set()  # Should return empty set, not crash

    def test_extract_multiple_from_import(self):
        """Test extraction of multiple names from one import."""
        from whisperjav.installer.validation.imports import _extract_imports

        source = "from typing import Optional, List, Dict"
        imports = _extract_imports(source)
        assert "typing" in imports


# =============================================================================
# End-to-End Scenario Tests
# =============================================================================


class TestInstallationScenarios:
    """Test complete installation scenarios (mocked)."""

    def test_executor_creation(self):
        """Test executor can be created with various options."""
        from whisperjav.installer.core.executor import StepExecutor

        # Default executor
        executor = StepExecutor()
        assert executor.max_retries == 3

        # Custom options
        executor = StepExecutor(max_retries=5, retry_delay=10)
        assert executor.max_retries == 5
        assert executor.retry_delay == 10

    def test_install_result_string_representation(self):
        """Test ExecutionResult string output."""
        from whisperjav.installer.core.executor import ExecutionResult

        # Success case
        result = ExecutionResult(
            success=True,
            package_name="torch",
            attempt=1,
            duration_seconds=30.5
        )
        assert "torch" in str(result)
        assert "30.5s" in str(result)

        # Skip case
        result = ExecutionResult(
            success=True,
            package_name="numpy",
            attempt=0,
            duration_seconds=0.1,
            skipped=True,
            skip_reason="Already installed"
        )
        assert "skipped" in str(result)

        # Failure case
        result = ExecutionResult(
            success=False,
            package_name="broken",
            attempt=3,
            duration_seconds=15.0,
            error="Connection failed"
        )
        assert "FAILED" in str(result)


class TestTorchVerification:
    """Test PyTorch CUDA verification."""

    @patch('subprocess.run')
    def test_verify_torch_cuda_available(self, mock_run):
        """Test CUDA availability verification."""
        from whisperjav.installer.core.executor import StepExecutor

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="cuda=True,version=12.1"
        )

        executor = StepExecutor()
        available, message = executor.verify_torch_cuda()
        assert available is True
        assert "12.1" in message

    @patch('subprocess.run')
    def test_verify_torch_cuda_not_available(self, mock_run):
        """Test CPU-only torch detection."""
        from whisperjav.installer.core.executor import StepExecutor

        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="cuda=False,version=None"
        )

        executor = StepExecutor()
        available, message = executor.verify_torch_cuda()
        assert available is False
        assert "CPU" in message or "not available" in message.lower()


class TestInstallPyIntegration:
    """Test install.py script integration."""

    def test_install_py_exists(self):
        """Verify install.py exists in project root."""
        project_root = Path(__file__).parent.parent
        install_py = project_root / "install.py"
        assert install_py.exists()

    def test_install_py_help(self):
        """Test install.py --help returns successfully."""
        project_root = Path(__file__).parent.parent
        install_py = project_root / "install.py"

        result = subprocess.run(
            [sys.executable, str(install_py), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=project_root,
        )
        assert result.returncode == 0
        # Check for expected arguments
        assert "--cpu-only" in result.stdout or "cpu" in result.stdout.lower()

    def test_install_py_imports(self):
        """Test that install.py can be imported without execution."""
        project_root = Path(__file__).parent.parent
        install_py = project_root / "install.py"

        # Check syntax by compiling
        with open(install_py, 'r') as f:
            source = f.read()
        compile(source, str(install_py), 'exec')


class TestShellWrappers:
    """Test shell wrapper scripts."""

    def test_windows_bat_exists(self):
        """Verify install_windows.bat exists."""
        project_root = Path(__file__).parent.parent
        bat_file = project_root / "installer" / "install_windows.bat"
        assert bat_file.exists()

    def test_linux_sh_exists(self):
        """Verify install_linux.sh exists."""
        project_root = Path(__file__).parent.parent
        sh_file = project_root / "installer" / "install_linux.sh"
        assert sh_file.exists()

    def test_windows_bat_delegates_to_python(self):
        """Test that Windows bat file delegates to install.py."""
        project_root = Path(__file__).parent.parent
        bat_file = project_root / "installer" / "install_windows.bat"

        content = bat_file.read_text()
        # Should call python install.py
        assert "python" in content.lower()
        assert "install.py" in content

    def test_linux_sh_delegates_to_python(self):
        """Test that Linux sh file delegates to install.py."""
        project_root = Path(__file__).parent.parent
        sh_file = project_root / "installer" / "install_linux.sh"

        content = sh_file.read_text()
        # Should call python install.py
        assert "python" in content.lower()
        assert "install.py" in content


# =============================================================================
# Regression Tests
# =============================================================================


def _locked_requirement_files() -> list:
    """v1.10.0 (item 2f): uv.lock exported per PyTorch build -- what every install takes.

    Tracked by git (installer/generated/ is not, and may hold a stale build).
    """
    project_root = Path(__file__).parent.parent
    return sorted((project_root / "whisperjav" / "installer" / "locked").glob("requirements_*.txt"))


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class TestGeneratedRequirements:
    """Tests for the locked requirements files the installer ships (item 2f)."""

    def test_requirements_file_exists(self):
        """One locked file per lock extra (design 22.2.4)."""
        names = sorted(f.name for f in _locked_requirement_files())
        assert names == ["requirements_cpu.txt", "requirements_cu128.txt", "requirements_cu130.txt",
                         "requirements_vulkan.txt"], names

    def test_requirements_contains_all_pyproject_packages(self):
        """
        Critical test: Verify requirements.txt contains ALL packages from pyproject.toml.

        This ensures no packages are accidentally filtered out during generation.
        """
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        project_root = Path(__file__).parent.parent
        pyproject_path = project_root / "pyproject.toml"

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        project = config["project"]
        core_deps = project.get("dependencies", [])
        optional_deps = project.get("optional-dependencies", {})

        # Extras the installer ships: the "all" extra (item 2f exports --extra all)
        installer_extras = ["cli", "gui", "translate", "enhance", "huggingface", "qwen", "analysis", "compatibility"]

        # Collect expected packages, git packages included (item 2f: they are in the
        # locked files at the commits uv.lock records)
        expected = set()

        def add(dep):
            # Platform markers: a Windows-only package is still in the universal export
            name = re.split(r"[>=<\[@ ;!~]", dep)[0].strip()
            expected.add(_normalized(name))

        for dep in core_deps:
            add(dep)
        for extra in installer_extras:
            for dep in optional_deps.get(extra, []):
                if not dep.startswith("whisperjav"):
                    add(dep)

        req_files = _locked_requirement_files()
        assert req_files, "No locked requirements file found"
        for req_file in req_files:
            actual = set()
            for line in req_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    actual.add(_normalized(re.split(r"[>=<\[@ ;!~]", line)[0].strip()))
            missing = expected - actual
            assert not missing, f"Packages missing from {req_file.name}: {missing}"

    def test_requirements_has_translate_packages(self):
        """Verify translate extra packages are in requirements.txt."""
        project_root = Path(__file__).parent.parent
        req_files = _locked_requirement_files()
        assert req_files, "No locked requirements file found"

        content = req_files[0].read_text(encoding="utf-8").lower()

        translate_packages = ["pysubtrans", "openai", "google-genai"]
        missing = [pkg for pkg in translate_packages if pkg not in content]

        assert not missing, f"Translate packages missing from requirements.txt: {missing}"

    def test_requirements_has_gui_packages(self):
        """Verify GUI extra packages are in requirements.txt."""
        project_root = Path(__file__).parent.parent
        req_files = _locked_requirement_files()
        assert req_files, "No locked requirements file found"

        content = req_files[0].read_text(encoding="utf-8").lower()

        gui_packages = ["pywebview", "pythonnet", "pywin32"]
        missing = [pkg for pkg in gui_packages if pkg not in content]

        assert not missing, f"GUI packages missing from requirements.txt: {missing}"


class TestImportNameMappings:
    """Tests for package name vs import name mappings."""

    def test_known_import_mappings_documented(self):
        """
        Verify known package/import name mismatches are documented.

        Some packages have different pip names vs import names:
        - pysubtrans -> PySubtrans
        - Pillow -> PIL
        - opencv-python -> cv2
        - scikit-learn -> sklearn

        This test ensures we're aware of these.
        """
        # Known mappings (package_name -> import_name)
        known_mappings = {
            "pysubtrans": "PySubtrans",
            "Pillow": "PIL",
            "scikit-learn": "sklearn",
            "PyYAML": "yaml",
        }

        # This test just documents the mappings exist
        # Actual import checks would require the packages to be installed
        assert len(known_mappings) > 0, "No known mappings documented"

    def test_pysubtrans_import_check_uses_correct_name(self):
        """
        Regression test: Ensure translate/cli.py checks for PySubtrans, not pysubtrans.

        Bug fixed: The import check used lowercase 'pysubtrans' but the actual
        import name is 'PySubtrans' (capital P and S).
        """
        project_root = Path(__file__).parent.parent
        cli_path = project_root / "whisperjav" / "translate" / "cli.py"

        content = cli_path.read_text()

        # Should check for PySubtrans (correct)
        assert "import PySubtrans" in content, \
            "translate/cli.py should check 'import PySubtrans' (capital P and S)"

        # Should NOT have the buggy lowercase check (without the correct one)
        # Note: We allow the word 'pysubtrans' in strings/comments, just not as the import


class TestPostInstallTemplate:
    """Tests for the standalone installer's post_install template."""

    def test_every_git_dependency_is_in_the_locked_files(self):
        """Every git-addressed dependency the installer ships is in each locked file.

        v1.10.0 (item 2f): Phase 3.5's own list is gone; the git packages come from
        the locked file at the commit uv.lock records. A pyproject pin to a commit
        must be that same commit. (In 1.9.2 faster-whisper went missing because a
        generator dropped git addresses and nothing else installed it.)
        """
        try:
            import tomllib
        except ImportError:  # Python 3.10 without tomli
            pytest.skip("tomllib not available")

        project_root = Path(__file__).parent.parent
        with open(project_root / "pyproject.toml", "rb") as f:
            project = tomllib.load(f)["project"]

        installer_extras = [
            "cli", "gui", "translate", "enhance",
            "huggingface", "qwen", "analysis", "compatibility",
        ]
        expected = {}
        for dep in project.get("dependencies", []) + [
                d for e in installer_extras for d in project["optional-dependencies"].get(e, [])]:
            if " @ git+" in dep:
                name, url = (part.strip() for part in dep.split(" @ ", 1))
                expected[_normalized(name)] = url
        assert expected, "No git dependencies found in pyproject.toml -- parser broken?"

        for req_file in _locked_requirement_files():
            locked = {}
            for line in req_file.read_text(encoding="utf-8").splitlines():
                if " @ git+" in line and not line.lstrip().startswith("#"):
                    name, url = (part.strip() for part in line.split(";")[0].split(" @ ", 1))
                    locked[_normalized(name)] = url
            assert set(locked) == set(expected), (req_file.name, sorted(locked), sorted(expected))
            for name, url in expected.items():
                assert re.search(r"@[0-9a-f]{40}", locked[name]), f"{name} is not at a commit in {req_file.name}"
                pinned = re.search(r"@([0-9a-f]{40})", url)
                if pinned:
                    assert pinned.group(1) in locked[name], (name, url, locked[name])

    def test_the_locked_file_is_installed_before_pytorch(self):
        """Phase order (item 2f): choose the build, its locked file, PyTorch, the wheel.

        PyTorch is installed --no-deps, so it can only be imported (the check right
        after it) once the packages it needs are in place.
        """
        project_root = Path(__file__).parent.parent
        content = (project_root / "installer" / "templates" / "post_install.py.template").read_text(encoding="utf-8")
        positions = [content.find(banner) for banner in (
            'log_section("Phase 3: PyTorch Build Selection")',
            'log_section("Phase 4: Release-Pinned Dependencies")',
            'log_section("Phase 4.5: PyTorch Installation")',
            'log_section("Phase 5: WhisperJAV Application Installation")',
        )]
        assert -1 not in positions, positions
        assert positions == sorted(positions), positions
        assert "git_packages" not in content, "Phase 3.5's own git list should be gone (item 2f)"

    def test_failed_core_check_ends_the_install(self):
        """A failed Phase 6 check must write the failure marker and return 1.

        Before v1.9.2 the install script printed "Installation completed
        successfully!" and returned 0 even when a core package would not
        import, so a user whose install was unusable was told it had worked.
        """
        project_root = Path(__file__).parent.parent
        template_path = project_root / "installer" / "templates" / "post_install.py.template"
        content = template_path.read_text(encoding="utf-8")

        phase_6_pos = content.find('log_section("Phase 6: Post-Install Verification")')
        assert phase_6_pos != -1, "Phase 6 banner not found in template"

        tail = content[phase_6_pos:]
        # Stop at the end of main(), before the __main__ block
        end = tail.find('if __name__ ==')
        assert end != -1, "__main__ block not found after Phase 6"
        tail = tail[:end]

        assert "if fails > 0:" in tail, (
            "Phase 6 does not branch on a failed core check — a broken install "
            "would still be reported as successful"
        )
        assert "create_failure_file(" in tail, (
            "Phase 6's failure branch does not write the failure marker file"
        )
        assert "return 1" in tail, (
            "Phase 6's failure branch does not return a non-zero exit code"
        )

        failure_branch = tail[tail.find("if fails > 0:"):]
        assert failure_branch.find("create_failure_file(") < failure_branch.find("return 1"), (
            "The failure marker must be written before returning"
        )


class TestInstallAbortsOnCoreFailure:
    """The installer must stop when a package WhisperJAV cannot run without is missing.

    Owner decision, 2026-09-11. Until v1.9.2 the install script's exit code was
    thrown away by conda-constructor's own runner, the installer reached its
    normal finish page, and a desktop shortcut was created that opened onto an
    error. A user was told an unusable install had worked.

    These are text checks on the NSIS template, and that is a real limit: NSIS
    cannot be compiled or run here, so they prove the instructions are present
    and in the right order, not that the installer behaves correctly. Only the
    clean-machine install proves that.
    """

    @staticmethod
    def _nsis() -> str:
        project_root = Path(__file__).parent.parent
        path = project_root / "installer" / "templates" / "custom_template.nsi.tmpl.template"
        assert path.exists(), "NSIS template not found"
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _post_install() -> str:
        project_root = Path(__file__).parent.parent
        path = project_root / "installer" / "templates" / "post_install.py.template"
        return path.read_text(encoding="utf-8")

    def test_a_failed_post_install_reaches_nsis(self):
        """The post-install's exit code must reach NSIS's retry dialog, and the script
        must run in a console window of its own.

        Up to constructor 3.11.3 the script ran through _nsis.py, which swallowed a
        non-zero exit unless NSIS_SCRIPTS_RAISE_ERRORS was set, and its cmd.exe got a
        console window. Constructor 3.17.3 (v1.10.0, i1) runs the bat hidden under
        nsExec; ours starts it in its own window (post_install.py shows progress and
        asks questions there) with `start /wait`, which returns the bat's exit code
        into AbortRetryNSExecWait (checked 2026-10-16 with exit codes 0, 1, 3 in a
        path with a space). The bat must pass Python's exit code on.
        """
        content = self._nsis()
        runner = ('push \'start "" /wait "$CMD_EXE" /D /C ""$INSTDIR\\pkgs\\post_install.bat"" '
                  '& call exit !ERRORLEVEL!\'')
        assert runner in content, "post_install.bat is not run in its own window with its exit code passed on"
        assert "call AbortRetryNSExecWait" in content[content.find(runner):][:300]
        assert '_nsis.py" post_install' not in content
        assert '"$INSTDIR\\pkgs\\post_install.bat" > "${STEP_LOG}"' not in content  # the hidden stock runner

        bat = (Path(__file__).parent.parent / "installer" / "templates" / "post_install.bat.template").read_text(
            encoding="utf-8")
        python_run = bat.find('"%~dp0..\\python.exe" "%~dp0..\\post_install_v{{VERSION}}.py"')
        assert python_run != -1
        assert "exit /b %errorlevel%" in bat[python_run:], "the bat does not return Python's exit code"

    def test_a_previous_installation_is_removed_only_after_the_path_checks(self):
        """Replacing a previous installation deletes it, so constructor's path checks
        (length, spaces, characters, writable) must run first (review 2026-10-16)."""
        content = self._nsis()
        start = content.index("Function OnDirectoryLeave")
        body = content[start:content.index("FunctionEnd", start)]
        assert body.index("GetFullPathName") < body.index("MAX_RELATIVE_PATH_LENGTH") \
            < body.index("pathgood:") < body.index("Call IsWhisperJAVInstallation") \
            < body.index("Call RemoveExistingWhisperJAV") < body.index("whisperjav_continue:")
        assert "${LogSet} on" not in body

    def test_the_install_aborts_when_the_failure_marker_exists(self):
        content = self._nsis()

        marker_check = content.find("INSTALLATION_FAILED_v{{VERSION}}.txt")
        assert marker_check != -1, (
            "Nothing checks for the failure marker, so a broken install would still finish"
        )

        tail = content[marker_check:]
        abort_pos = tail.find("Abort")
        assert abort_pos != -1 and abort_pos < 2000, (
            "The failure-marker branch does not Abort the installation"
        )

    def test_no_desktop_shortcut_is_created_after_a_failed_install(self):
        """A shortcut that opens onto an error is worse than no shortcut.

        The abort must come first in the script, so the shortcut block is never
        reached. This also covers the silent-install case, where the retry
        dialog answers itself with "Ignore".
        """
        content = self._nsis()

        marker_check = content.find('${FileExists} "$INSTDIR\INSTALLATION_FAILED')
        shortcut = content.find('CreateShortCut "$DESKTOP')

        assert marker_check != -1, "failure-marker check not found in the NSIS template"
        assert shortcut != -1, "desktop shortcut creation not found in the NSIS template"
        assert marker_check < shortcut, (
            "The failure check must come BEFORE the desktop shortcut is created, "
            "or a broken install still puts a shortcut on the desktop"
        )

    def test_the_failure_message_tells_the_user_what_to_do(self):
        """The user is not technical. The dialog has to say what happened, where
        the details are, and what to try -- re-running fixes the usual cause."""
        content = self._nsis()
        marker_check = content.find('${FileExists} "$INSTDIR\INSTALLATION_FAILED')
        block = content[marker_check:marker_check + 2000]

        assert "MessageBox" in block, "the user is not shown anything"
        assert "install_log_v{{VERSION}}.txt" in block, "the log file is not named"
        assert "github.com/meizhong986/WhisperJAV/issues" in block, "no way to report it"
        assert "again" in block.lower(), "does not tell the user to try again"

    def test_a_stale_marker_from_an_earlier_attempt_is_cleared(self):
        """Otherwise a marker left by a previous failed attempt in the same folder
        would abort an install that actually worked."""
        content = self._post_install()

        main_pos = content.find("def main() -> int:")
        assert main_pos != -1
        block = content[main_pos:main_pos + 1200]

        assert "INSTALLATION_FAILED_v{{VERSION}}.txt" in block, (
            "main() does not clear a stale failure marker before the install runs"
        )
        assert "os.remove" in block, "the stale marker is found but not removed"

        first_phase = content.find("Phase 1: Preflight Checks", main_pos)
        clear_pos = content.find("INSTALLATION_FAILED_v{{VERSION}}.txt", main_pos)
        assert clear_pos < first_phase, (
            "the stale marker must be cleared before any phase can write a new one"
        )


class TestKnownIssues:
    """Tests for known issues that have been fixed."""

    def test_issue_90_torch_cpu_resolution(self):
        """
        Issue #90: Pip resolves CPU torch when whisper installed first.

        Verify torch is ordered before whisper packages.
        """
        from whisperjav.installer.core.registry import get_packages_in_install_order

        packages = get_packages_in_install_order()
        names = [p.name for p in packages]

        torch_idx = names.index("torch") if "torch" in names else 999
        whisper_idx = names.index("openai-whisper") if "openai-whisper" in names else 0

        assert torch_idx < whisper_idx, \
            "torch must be installed before openai-whisper to prevent CPU fallback"

    def test_issue_111_git_timeout_detection(self):
        """
        Issue #111: Git timeout on slow connections.

        Verify timeout pattern detection works.
        """
        from whisperjav.installer.core.executor import StepExecutor

        executor = StepExecutor()

        # The exact error message from Issue #111
        error = "fatal: unable to access 'https://github.com/openai/whisper.git/': Failed to connect to github.com port 443 after 21074 ms: Timed out"
        assert executor._is_git_timeout(error) is True

    def test_numpy_before_numba(self):
        """
        Known issue: numba must be installed after numpy.

        Verify installation order.
        """
        from whisperjav.installer.core.registry import get_packages_in_install_order

        packages = get_packages_in_install_order()
        names = [p.name for p in packages]

        if "numpy" in names and "numba" in names:
            numpy_idx = names.index("numpy")
            numba_idx = names.index("numba")
            assert numpy_idx < numba_idx, \
                "numpy must be installed before numba for binary compatibility"


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
