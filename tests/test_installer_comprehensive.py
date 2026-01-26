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

    def test_select_cu128_for_modern_driver(self):
        """Test CUDA 12.8 selection for driver 570+."""
        from whisperjav.installer.core.detector import _select_cuda_version

        # Driver 570+ should get cu128
        cuda, url, msg = _select_cuda_version((570, 0, 0))
        assert cuda == "cu128"
        assert "cu128" in url

        cuda, url, msg = _select_cuda_version((580, 10, 5))
        assert cuda == "cu128"

    def test_select_cu118_for_older_driver(self):
        """Test CUDA 11.8 selection for driver 450-569."""
        from whisperjav.installer.core.detector import _select_cuda_version

        # Driver 450-569 should get cu118
        cuda, url, msg = _select_cuda_version((450, 0, 0))
        assert cuda == "cu118"
        assert "cu118" in url

        cuda, url, msg = _select_cuda_version((569, 99, 99))
        assert cuda == "cu118"

        cuda, url, msg = _select_cuda_version((520, 50, 0))
        assert cuda == "cu118"

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
        driver_result.stdout = "570.00.00\n"

        # Mock GPU name query
        name_result = MagicMock()
        name_result.returncode = 0
        name_result.stdout = "NVIDIA GeForce RTX 4090\n"

        mock_run.side_effect = [driver_result, name_result]

        result = detect_gpu()
        assert result.detected is True
        assert result.cuda_version == "cu128"
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
        pkg = Package(name="numpy", version=">=1.26.0")

        args = executor._build_pip_args(pkg, "cu118")
        assert "install" in args
        assert "numpy>=1.26.0" in args

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
        """Test that CUDA matrix has at least cu118 and cu128."""
        from whisperjav.installer.core.config import CUDA_DRIVER_MATRIX

        cuda_versions = {entry.cuda_version for entry in CUDA_DRIVER_MATRIX}
        assert "cu118" in cuda_versions, "Missing cu118 entry"
        assert "cu128" in cuda_versions, "Missing cu128 entry"

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

        assert PYTHON_MIN_VERSION < PYTHON_MAX_VERSION
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
