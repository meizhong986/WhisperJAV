#!/usr/bin/env python3
"""
WhisperJAV Installation & Upgrade Manual Test Suite
====================================================

Automated testing for whisperjav-upgrade, version consistency, and CLI functionality.

This script tests key installation and upgrade scenarios to verify:
1. whisperjav-upgrade functions properly
2. Version consistency across components
3. CLI entry points work correctly
4. Post-install script structure is valid

NOTE: The standalone installer uses conda-constructor with post_install_v*.py,
not a standalone install.py. These tests focus on the development/upgrade workflow.

Usage:
    python test_installation_scenarios.py                    # Run all tests
    python test_installation_scenarios.py --quick            # Skip slow tests
    python test_installation_scenarios.py --verbose          # Show detailed output
    python test_installation_scenarios.py --keep-on-failure  # Don't cleanup failed tests
    python test_installation_scenarios.py --test upgrade     # Run only upgrade tests
    python test_installation_scenarios.py --test version     # Run only version tests

Requirements:
    - Python 3.10+
    - Run from the WhisperJAV repository root
    - Internet connection (for some tests)

Author: WhisperJAV Team
Date: 2025-01-27
"""

import os
import sys
import subprocess
import tempfile
import shutil
import argparse
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from datetime import datetime
import traceback


# =============================================================================
# Configuration
# =============================================================================

class TestConfig:
    """Test configuration and paths."""

    def __init__(self):
        # Detect repository root
        self.script_dir = Path(__file__).parent.resolve()
        self.repo_root = self.script_dir.parent.parent

        # Verify we're in the right place
        if not (self.repo_root / "whisperjav").exists():
            raise RuntimeError(
                f"Cannot find whisperjav package. "
                f"Expected at: {self.repo_root / 'whisperjav'}\n"
                f"Please run this script from the repository root."
            )

        # Key paths
        self.installer_dir = self.repo_root / "installer"
        self.generated_dir = self.installer_dir / "generated"
        self.whisperjav_pkg = self.repo_root / "whisperjav"
        self.version_file = self.whisperjav_pkg / "__version__.py"
        self.installer_version_file = self.installer_dir / "VERSION"

        # Test settings
        self.temp_base = Path(tempfile.gettempdir()) / "whisperjav_test"
        self.timeout_short = 30  # seconds
        self.timeout_medium = 120  # seconds
        self.timeout_long = 600  # seconds (10 min for full installs)

        # Python executable
        self.python = sys.executable

    def get_temp_dir(self, name: str) -> Path:
        """Get a temporary directory for a test."""
        path = self.temp_base / f"{name}_{int(time.time())}"
        path.mkdir(parents=True, exist_ok=True)
        return path


# =============================================================================
# Test Result Types
# =============================================================================

class TestStatus(Enum):
    """Test result status."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    status: TestStatus
    duration_sec: float
    message: str = ""
    stdout: str = ""
    stderr: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        status_icon = {
            TestStatus.PASSED: "[PASS]",
            TestStatus.FAILED: "[FAIL]",
            TestStatus.SKIPPED: "[SKIP]",
            TestStatus.ERROR: "[ERR ]",
        }
        return f"{status_icon[self.status]} {self.name} ({self.duration_sec:.2f}s)"


@dataclass
class TestSuite:
    """Collection of test results."""
    name: str
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def add(self, result: TestResult):
        self.results.append(result)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.PASSED)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.FAILED)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.SKIPPED)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == TestStatus.ERROR)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.passed / self.total) * 100


# =============================================================================
# Console Output Utilities
# =============================================================================

class Console:
    """Colored console output."""

    # ANSI color codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        # Enable ANSI on Windows
        if sys.platform == "win32":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            except Exception:
                pass

    def print(self, msg: str = "", end: str = "\n"):
        print(msg, end=end, flush=True)

    def header(self, title: str):
        self.print()
        self.print(f"{self.BOLD}{self.BLUE}{'=' * 70}{self.RESET}")
        self.print(f"{self.BOLD}{self.BLUE}  {title}{self.RESET}")
        self.print(f"{self.BOLD}{self.BLUE}{'=' * 70}{self.RESET}")

    def subheader(self, title: str):
        self.print()
        self.print(f"{self.CYAN}{'-' * 50}{self.RESET}")
        self.print(f"{self.CYAN}  {title}{self.RESET}")
        self.print(f"{self.CYAN}{'-' * 50}{self.RESET}")

    def success(self, msg: str):
        self.print(f"{self.GREEN}[OK]{self.RESET} {msg}")

    def error(self, msg: str):
        self.print(f"{self.RED}[ERROR]{self.RESET} {msg}")

    def warning(self, msg: str):
        self.print(f"{self.YELLOW}[WARN]{self.RESET} {msg}")

    def info(self, msg: str):
        self.print(f"{self.BLUE}[INFO]{self.RESET} {msg}")

    def debug(self, msg: str):
        if self.verbose:
            self.print(f"{self.CYAN}[DEBUG]{self.RESET} {msg}")

    def test_result(self, result: TestResult):
        if result.status == TestStatus.PASSED:
            color = self.GREEN
        elif result.status == TestStatus.FAILED:
            color = self.RED
        elif result.status == TestStatus.SKIPPED:
            color = self.YELLOW
        else:
            color = self.RED

        self.print(f"  {color}{result}{self.RESET}")

        if result.message and (result.status != TestStatus.PASSED or self.verbose):
            self.print(f"      {result.message}")

        if self.verbose and result.stdout:
            self.print(f"      stdout: {result.stdout[:200]}...")

        if result.status == TestStatus.FAILED and result.stderr:
            self.print(f"      stderr: {result.stderr[:500]}")


# =============================================================================
# Command Runner
# =============================================================================

class CommandRunner:
    """Execute commands and capture output."""

    def __init__(self, config: TestConfig, console: Console):
        self.config = config
        self.console = console

    def run(
        self,
        cmd: List[str],
        cwd: Optional[Path] = None,
        timeout: int = 120,
        env: Optional[Dict[str, str]] = None,
        check: bool = False
    ) -> Tuple[int, str, str]:
        """
        Run a command and return (returncode, stdout, stderr).

        Args:
            cmd: Command and arguments
            cwd: Working directory
            timeout: Timeout in seconds
            env: Environment variables (merged with current)
            check: If True, raise on non-zero exit

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        # Merge environment
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        # Log command
        cmd_str = " ".join(str(c) for c in cmd)
        self.console.debug(f"Running: {cmd_str}")
        if cwd:
            self.console.debug(f"  cwd: {cwd}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.config.repo_root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env
            )

            if check and result.returncode != 0:
                raise subprocess.CalledProcessError(
                    result.returncode, cmd, result.stdout, result.stderr
                )

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired as e:
            self.console.debug(f"Command timed out after {timeout}s")
            return -1, "", f"Timeout after {timeout}s"
        except Exception as e:
            self.console.debug(f"Command failed: {e}")
            return -1, "", str(e)

    def run_python(
        self,
        script: str,
        args: List[str] = None,
        cwd: Optional[Path] = None,
        timeout: int = 120,
        venv_python: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """Run a Python script."""
        python = str(venv_python) if venv_python else self.config.python
        cmd = [python, str(script)]
        if args:
            cmd.extend(args)
        return self.run(cmd, cwd=cwd, timeout=timeout)

    def run_module(
        self,
        module: str,
        args: List[str] = None,
        timeout: int = 120,
        venv_python: Optional[Path] = None
    ) -> Tuple[int, str, str]:
        """Run a Python module."""
        python = str(venv_python) if venv_python else self.config.python
        cmd = [python, "-m", module]
        if args:
            cmd.extend(args)
        return self.run(cmd, timeout=timeout)


# =============================================================================
# Test Cases: Installer Structure
# =============================================================================

class InstallerTests:
    """Tests for installer package structure and generated files."""

    def __init__(self, config: TestConfig, console: Console, runner: CommandRunner):
        self.config = config
        self.console = console
        self.runner = runner

    def test_version_file_exists(self) -> TestResult:
        """Verify installer/VERSION exists."""
        start = time.time()

        if not self.config.installer_version_file.exists():
            return TestResult(
                name="VERSION file exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Not found: {self.config.installer_version_file}"
            )

        return TestResult(
            name="VERSION file exists",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Found: {self.config.installer_version_file}"
        )

    def test_generated_dir_exists(self) -> TestResult:
        """Verify installer/generated directory exists."""
        start = time.time()

        if not self.config.generated_dir.exists():
            return TestResult(
                name="generated dir exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Not found: {self.config.generated_dir}"
            )

        return TestResult(
            name="generated dir exists",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Found: {self.config.generated_dir}"
        )

    def test_generated_files_present(self) -> TestResult:
        """Verify key generated files are present."""
        start = time.time()

        # Get version from __version__.py
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.__version__ import __version__; print(__version__)"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="generated files present",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Cannot get version",
                stderr=stderr
            )

        version = stdout.strip()

        # Check for expected generated files
        expected_files = [
            f"post_install_v{version}.py",
            f"requirements_v{version}.txt",
            f"construct_v{version}.yaml",
            f"build_installer_v{version}.bat",
            f"WhisperJAV_Launcher_v{version}.py",
        ]

        missing = []
        for filename in expected_files:
            if not (self.config.generated_dir / filename).exists():
                missing.append(filename)

        if missing:
            return TestResult(
                name="generated files present",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Missing files: {missing}"
            )

        return TestResult(
            name="generated files present",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"All {len(expected_files)} expected files present for v{version}"
        )

    def test_post_install_syntax(self) -> TestResult:
        """Verify post_install.py is valid Python."""
        start = time.time()

        # Get version
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.__version__ import __version__; print(__version__)"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="post_install syntax",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Cannot get version",
                stderr=stderr
            )

        version = stdout.strip()
        post_install = self.config.generated_dir / f"post_install_v{version}.py"

        if not post_install.exists():
            return TestResult(
                name="post_install syntax",
                status=TestStatus.SKIPPED,
                duration_sec=time.time() - start,
                message=f"File not found: {post_install}"
            )

        # Check syntax by compiling
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-m", "py_compile", str(post_install)],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="post_install syntax",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Syntax error in post_install.py",
                stderr=stderr
            )

        return TestResult(
            name="post_install syntax",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"post_install_v{version}.py is valid Python"
        )

    def test_wheel_exists(self) -> TestResult:
        """Verify wheel file was generated."""
        start = time.time()

        # Get version
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.__version__ import __version__; print(__version__)"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="wheel exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Cannot get version",
                stderr=stderr
            )

        version = stdout.strip()
        wheel_file = self.config.generated_dir / f"whisperjav-{version}-py3-none-any.whl"

        if not wheel_file.exists():
            return TestResult(
                name="wheel exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Not found: {wheel_file}"
            )

        # Check wheel size (should be at least 100KB)
        size_kb = wheel_file.stat().st_size / 1024

        if size_kb < 100:
            return TestResult(
                name="wheel exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Wheel too small: {size_kb:.1f}KB"
            )

        return TestResult(
            name="wheel exists",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Found: {wheel_file.name} ({size_kb:.1f}KB)"
        )

    def test_validation_script(self) -> TestResult:
        """Run the installer validation script."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.installer.validation",
            ["--quick"],
            timeout=self.config.timeout_medium
        )

        if code != 0:
            return TestResult(
                name="validation script",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Validation failed (exit {code})",
                stdout=stdout[:500],
                stderr=stderr[:500]
            )

        return TestResult(
            name="validation script",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="Installer validation passed"
        )

    def run_all(self, quick: bool = False) -> TestSuite:
        """Run all installer tests."""
        suite = TestSuite(name="Installer Tests")
        suite.start_time = datetime.now()

        tests = [
            self.test_version_file_exists,
            self.test_generated_dir_exists,
            self.test_generated_files_present,
            self.test_post_install_syntax,
            self.test_wheel_exists,
            self.test_validation_script,
        ]

        for test in tests:
            try:
                result = test()
                suite.add(result)
                self.console.test_result(result)
            except Exception as e:
                result = TestResult(
                    name=test.__name__,
                    status=TestStatus.ERROR,
                    duration_sec=0,
                    message=str(e),
                    stderr=traceback.format_exc()
                )
                suite.add(result)
                self.console.test_result(result)

        suite.end_time = datetime.now()
        return suite


# =============================================================================
# Test Cases: whisperjav-upgrade
# =============================================================================

class UpgradeTests:
    """Tests for whisperjav-upgrade functionality."""

    def __init__(self, config: TestConfig, console: Console, runner: CommandRunner):
        self.config = config
        self.console = console
        self.runner = runner

    def test_upgrade_module_exists(self) -> TestResult:
        """Verify upgrade module can be imported."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.upgrade",
            ["--help"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="upgrade module exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        return TestResult(
            name="upgrade module exists",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="whisperjav.upgrade module loads successfully"
        )

    def test_upgrade_help(self) -> TestResult:
        """Verify --help shows all expected options."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.upgrade",
            ["--help"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="upgrade --help",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        # Check for expected options
        expected_options = [
            "--check",
            "--yes",
            "--wheel-only",
            "--list-snapshots",
            "--rollback",
        ]

        missing = [opt for opt in expected_options if opt not in stdout]

        if missing:
            return TestResult(
                name="upgrade --help",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Missing options: {missing}",
                stdout=stdout
            )

        return TestResult(
            name="upgrade --help",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"All {len(expected_options)} expected options present"
        )

    def test_upgrade_version(self) -> TestResult:
        """Verify --version shows current version."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.upgrade",
            ["--version"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="upgrade --version",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        # Should contain version number
        if not any(c.isdigit() for c in stdout):
            return TestResult(
                name="upgrade --version",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="No version number in output",
                stdout=stdout
            )

        return TestResult(
            name="upgrade --version",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Version: {stdout.strip()}"
        )

    def test_upgrade_check(self) -> TestResult:
        """Verify --check shows update status."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.upgrade",
            ["--check"],
            timeout=self.config.timeout_medium  # May need network
        )

        # --check may return non-zero if no updates, that's OK
        output = stdout + stderr

        # Should show current version info
        if "version" not in output.lower() and "current" not in output.lower():
            return TestResult(
                name="upgrade --check",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Output doesn't show version info",
                stdout=stdout[:500]
            )

        return TestResult(
            name="upgrade --check",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="Check completed successfully"
        )

    def test_upgrade_list_snapshots(self) -> TestResult:
        """Verify --list-snapshots works."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.upgrade",
            ["--list-snapshots"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="upgrade --list-snapshots",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        # Should either show snapshots or "no snapshots"
        output_lower = (stdout + stderr).lower()
        if "snapshot" not in output_lower:
            return TestResult(
                name="upgrade --list-snapshots",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Output doesn't mention snapshots",
                stdout=stdout
            )

        return TestResult(
            name="upgrade --list-snapshots",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="Snapshot listing works"
        )

    def test_upgrade_validate_extras(self) -> TestResult:
        """Verify extras validation works.

        Note: Extras are only validated in the full upgrade path (not --check).
        When using --yes, the script proceeds to validation before prompting.
        """
        start = time.time()

        # The upgrade script validates extras before proceeding
        # Use --yes to auto-confirm but it should fail on invalid extras first
        # Since we can't run an actual upgrade, we test by importing the validate function
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c", """
from whisperjav.upgrade import validate_extras, VALID_EXTRAS
# Test valid extras
ok, invalid = validate_extras("cli,gui")
assert ok, f"Valid extras rejected: {invalid}"
# Test invalid extras
ok, invalid = validate_extras("invalid_extra")
assert not ok, "Invalid extra was accepted"
assert "invalid_extra" in invalid, f"Wrong invalid list: {invalid}"
# Test 'all'
ok, invalid = validate_extras("all")
assert ok, "'all' should be valid"
print(f"OK: Valid extras={VALID_EXTRAS}")
"""],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="upgrade extras validation",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Extras validation logic failed",
                stderr=stderr
            )

        return TestResult(
            name="upgrade extras validation",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="Extras validation works correctly"
        )

    def run_all(self, quick: bool = False) -> TestSuite:
        """Run all upgrade tests."""
        suite = TestSuite(name="whisperjav-upgrade Tests")
        suite.start_time = datetime.now()

        tests = [
            self.test_upgrade_module_exists,
            self.test_upgrade_help,
            self.test_upgrade_version,
            self.test_upgrade_check,
            self.test_upgrade_list_snapshots,
            self.test_upgrade_validate_extras,
        ]

        for test in tests:
            try:
                result = test()
                suite.add(result)
                self.console.test_result(result)
            except Exception as e:
                result = TestResult(
                    name=test.__name__,
                    status=TestStatus.ERROR,
                    duration_sec=0,
                    message=str(e),
                    stderr=traceback.format_exc()
                )
                suite.add(result)
                self.console.test_result(result)

        suite.end_time = datetime.now()
        return suite


# =============================================================================
# Test Cases: Version Consistency
# =============================================================================

class VersionTests:
    """Tests for version consistency across components."""

    def __init__(self, config: TestConfig, console: Console, runner: CommandRunner):
        self.config = config
        self.console = console
        self.runner = runner

    def test_version_file_exists(self) -> TestResult:
        """Verify __version__.py exists."""
        start = time.time()

        if not self.config.version_file.exists():
            return TestResult(
                name="version file exists",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Not found: {self.config.version_file}"
            )

        return TestResult(
            name="version file exists",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Found: {self.config.version_file}"
        )

    def test_version_importable(self) -> TestResult:
        """Verify version can be imported."""
        start = time.time()

        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.__version__ import __version__; print(__version__)"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="version importable",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Import failed: {stderr}"
            )

        version = stdout.strip()

        return TestResult(
            name="version importable",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Version: {version}"
        )

    def test_version_consistency(self) -> TestResult:
        """Verify WhisperJAV version is consistent across entry points.

        Note: The upgrade script has its own version (UPGRADE_SCRIPT_VERSION)
        which is intentionally different from the WhisperJAV package version.
        This test only checks WhisperJAV package version consistency.
        """
        start = time.time()

        versions = {}

        # Get version from __version__.py
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.__version__ import __version__; print(__version__)"],
            timeout=self.config.timeout_short
        )
        if code == 0:
            versions["__version__.py"] = stdout.strip()

        # Get version from whisperjav package
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "import whisperjav; print(whisperjav.__version__)"],
            timeout=self.config.timeout_short
        )
        if code == 0:
            versions["whisperjav"] = stdout.strip()

        # Get version from version_checker module (app version, not script version)
        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.version_checker import CURRENT_VERSION; print(CURRENT_VERSION)"],
            timeout=self.config.timeout_short
        )
        if code == 0:
            versions["version_checker"] = stdout.strip()

        # Check consistency
        unique_versions = set(versions.values())

        if len(unique_versions) > 1:
            return TestResult(
                name="version consistency",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Version mismatch: {versions}"
            )

        if len(unique_versions) == 0:
            return TestResult(
                name="version consistency",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Could not get any version"
            )

        return TestResult(
            name="version consistency",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"All versions match: {list(unique_versions)[0]}",
            details=versions
        )

    def run_all(self, quick: bool = False) -> TestSuite:
        """Run all version tests."""
        suite = TestSuite(name="Version Tests")
        suite.start_time = datetime.now()

        tests = [
            self.test_version_file_exists,
            self.test_version_importable,
            self.test_version_consistency,
        ]

        for test in tests:
            try:
                result = test()
                suite.add(result)
                self.console.test_result(result)
            except Exception as e:
                result = TestResult(
                    name=test.__name__,
                    status=TestStatus.ERROR,
                    duration_sec=0,
                    message=str(e),
                    stderr=traceback.format_exc()
                )
                suite.add(result)
                self.console.test_result(result)

        suite.end_time = datetime.now()
        return suite


# =============================================================================
# Test Cases: CLI Functional Tests
# =============================================================================

class CLITests:
    """Functional tests for CLI entry points."""

    def __init__(self, config: TestConfig, console: Console, runner: CommandRunner):
        self.config = config
        self.console = console
        self.runner = runner

    def test_main_help(self) -> TestResult:
        """Verify main CLI --help works."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.main",
            ["--help"],
            timeout=self.config.timeout_medium  # May load heavy modules
        )

        if code != 0:
            return TestResult(
                name="main CLI --help",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        # Check for key options
        if "--mode" not in stdout or "--model" not in stdout:
            return TestResult(
                name="main CLI --help",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message="Missing expected options",
                stdout=stdout[:500]
            )

        return TestResult(
            name="main CLI --help",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="CLI help works"
        )

    def test_gui_module_importable(self) -> TestResult:
        """Verify GUI module can be imported (fast, no window)."""
        start = time.time()

        code, stdout, stderr = self.runner.run(
            [self.config.python, "-c",
             "from whisperjav.webview_gui.main import main; print('OK')"],
            timeout=self.config.timeout_medium
        )

        if code != 0:
            return TestResult(
                name="GUI module importable",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Import failed",
                stderr=stderr
            )

        return TestResult(
            name="GUI module importable",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message=f"Import time: {time.time() - start:.2f}s"
        )

    def test_translate_help(self) -> TestResult:
        """Verify translate CLI --help works."""
        start = time.time()

        code, stdout, stderr = self.runner.run_module(
            "whisperjav.translate.cli",
            ["--help"],
            timeout=self.config.timeout_short
        )

        if code != 0:
            return TestResult(
                name="translate CLI --help",
                status=TestStatus.FAILED,
                duration_sec=time.time() - start,
                message=f"Exit code: {code}",
                stderr=stderr
            )

        return TestResult(
            name="translate CLI --help",
            status=TestStatus.PASSED,
            duration_sec=time.time() - start,
            message="Translate CLI help works"
        )

    def run_all(self, quick: bool = False) -> TestSuite:
        """Run all CLI tests."""
        suite = TestSuite(name="CLI Tests")
        suite.start_time = datetime.now()

        tests = [
            self.test_main_help,
            self.test_gui_module_importable,
            self.test_translate_help,
        ]

        for test in tests:
            try:
                result = test()
                suite.add(result)
                self.console.test_result(result)
            except Exception as e:
                result = TestResult(
                    name=test.__name__,
                    status=TestStatus.ERROR,
                    duration_sec=0,
                    message=str(e),
                    stderr=traceback.format_exc()
                )
                suite.add(result)
                self.console.test_result(result)

        suite.end_time = datetime.now()
        return suite


# =============================================================================
# Main Test Runner
# =============================================================================

class TestRunner:
    """Main test orchestrator."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.console = Console(verbose=args.verbose)
        self.config = TestConfig()
        self.runner = CommandRunner(self.config, self.console)
        self.suites: List[TestSuite] = []

    def run(self) -> int:
        """Run all tests and return exit code."""
        start_time = datetime.now()

        self.console.header("WhisperJAV Installation Test Suite")
        self.console.info(f"Repository: {self.config.repo_root}")
        self.console.info(f"Python: {self.config.python}")
        self.console.info(f"Mode: {'quick' if self.args.quick else 'full'}")
        self.console.info(f"Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Determine which tests to run
        test_filter = self.args.test.lower() if self.args.test else "all"

        try:
            # Version Tests (always run)
            if test_filter in ["all", "version"]:
                self.console.subheader("Version Consistency Tests")
                version_tests = VersionTests(self.config, self.console, self.runner)
                self.suites.append(version_tests.run_all(quick=self.args.quick))

            # Installer Structure Tests
            if test_filter in ["all", "installer"]:
                self.console.subheader("Installer Structure Tests")
                installer_tests = InstallerTests(self.config, self.console, self.runner)
                self.suites.append(installer_tests.run_all(quick=self.args.quick))

            # Upgrade Tests
            if test_filter in ["all", "upgrade"]:
                self.console.subheader("whisperjav-upgrade Tests")
                upgrade_tests = UpgradeTests(self.config, self.console, self.runner)
                self.suites.append(upgrade_tests.run_all(quick=self.args.quick))

            # CLI Tests
            if test_filter in ["all", "cli"]:
                self.console.subheader("CLI Functional Tests")
                cli_tests = CLITests(self.config, self.console, self.runner)
                self.suites.append(cli_tests.run_all(quick=self.args.quick))

        except KeyboardInterrupt:
            self.console.warning("Test run interrupted by user")
            return 130

        # Print summary
        end_time = datetime.now()
        self.print_summary(start_time, end_time)

        # Return exit code
        total_failed = sum(s.failed + s.errors for s in self.suites)
        return 0 if total_failed == 0 else 1

    def print_summary(self, start_time: datetime, end_time: datetime):
        """Print test summary."""
        self.console.header("Test Summary")

        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0

        for suite in self.suites:
            status_color = self.console.GREEN if suite.failed == 0 and suite.errors == 0 else self.console.RED
            self.console.print(
                f"  {suite.name}: "
                f"{self.console.GREEN}{suite.passed} passed{self.console.RESET}, "
                f"{self.console.RED}{suite.failed} failed{self.console.RESET}, "
                f"{self.console.YELLOW}{suite.skipped} skipped{self.console.RESET}"
            )
            total_passed += suite.passed
            total_failed += suite.failed
            total_skipped += suite.skipped
            total_errors += suite.errors

        duration = (end_time - start_time).total_seconds()

        self.console.print()
        self.console.print(f"{'=' * 50}")
        self.console.print(
            f"  Total: {total_passed + total_failed + total_skipped + total_errors} tests in {duration:.1f}s"
        )
        self.console.print(
            f"  {self.console.GREEN}Passed: {total_passed}{self.console.RESET}  "
            f"{self.console.RED}Failed: {total_failed}{self.console.RESET}  "
            f"{self.console.YELLOW}Skipped: {total_skipped}{self.console.RESET}  "
            f"Errors: {total_errors}"
        )

        if total_failed == 0 and total_errors == 0:
            self.console.print()
            self.console.success("All tests passed!")
        else:
            self.console.print()
            self.console.error(f"{total_failed + total_errors} test(s) failed")

            # List failed tests
            self.console.print()
            self.console.print("Failed tests:")
            for suite in self.suites:
                for result in suite.results:
                    if result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                        self.console.print(f"  - {suite.name}: {result.name}")
                        if result.message:
                            self.console.print(f"    {result.message}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WhisperJAV Installation & Upgrade Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_installation_scenarios.py                    # Run all tests
  python test_installation_scenarios.py --quick            # Skip slow tests
  python test_installation_scenarios.py --test installer   # Only installer tests
  python test_installation_scenarios.py --test upgrade     # Only upgrade tests
  python test_installation_scenarios.py --verbose          # Detailed output
        """
    )

    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip slow tests (venv creation, full installs)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed output"
    )

    parser.add_argument(
        "--keep-on-failure", "-k",
        action="store_true",
        help="Don't clean up temporary directories on failure"
    )

    parser.add_argument(
        "--test", "-t",
        choices=["all", "installer", "upgrade", "version", "cli"],
        default="all",
        help="Run specific test suite only"
    )

    args = parser.parse_args()

    runner = TestRunner(args)
    sys.exit(runner.run())


if __name__ == "__main__":
    main()
