#!/usr/bin/env python3
"""
Installation validation tests for WhisperJAV.

Tests the modular extras system, platform detection, and entry points.
"""

import subprocess
import sys
from pathlib import Path

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


class TestPyprojectToml:
    """Test pyproject.toml configuration."""

    def test_pyproject_exists(self):
        """Verify pyproject.toml exists."""
        pyproject = PROJECT_ROOT / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml not found"

    def test_pyproject_valid_toml(self):
        """Verify pyproject.toml is valid TOML."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        data = tomllib.loads(content)

        assert "project" in data, "Missing [project] section"
        assert "name" in data["project"], "Missing project name"
        assert data["project"]["name"] == "whisperjav"

    def test_pyproject_has_extras(self):
        """Verify extras are defined."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        data = tomllib.loads(content)

        extras = data.get("project", {}).get("optional-dependencies", {})
        expected_extras = [
            "cli", "gui", "translate", "llm", "enhance",
            "huggingface", "analysis", "compatibility", "all", "colab",
        ]

        for extra in expected_extras:
            assert extra in extras, f"Missing extra: {extra}"

    def test_pyproject_entry_points(self):
        """Verify entry points are defined."""
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        pyproject = PROJECT_ROOT / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        data = tomllib.loads(content)

        scripts = data.get("project", {}).get("scripts", {})
        expected_scripts = ["whisperjav", "whisperjav-gui", "whisperjav-translate"]

        for script in expected_scripts:
            assert script in scripts, f"Missing entry point: {script}"


class TestPlatformDetection:
    """Test platform detection utilities."""

    def test_platform_module_imports(self):
        """Verify platform module can be imported."""
        from whisperjav.utils.platform import (
            get_platform,
            is_windows,
            is_linux,
            is_macos,
            is_colab,
            is_notebook,
            get_platform_info,
            get_recommended_extras,
        )

        # Just verify they're callable
        assert callable(get_platform)
        assert callable(is_windows)
        assert callable(is_linux)
        assert callable(is_macos)
        assert callable(is_colab)
        assert callable(is_notebook)
        assert callable(get_platform_info)
        assert callable(get_recommended_extras)

    def test_get_platform_returns_valid(self):
        """Verify get_platform returns a valid platform name."""
        from whisperjav.utils.platform import get_platform

        platform = get_platform()
        assert platform in ("windows", "linux", "macos", "unknown")

    def test_platform_info_structure(self):
        """Verify get_platform_info returns expected structure."""
        from whisperjav.utils.platform import get_platform_info

        info = get_platform_info()
        assert isinstance(info, dict)
        assert "platform" in info
        assert "python_version" in info
        assert "is_notebook" in info

    def test_recommended_extras_returns_list(self):
        """Verify get_recommended_extras returns a list."""
        from whisperjav.utils.platform import get_recommended_extras

        extras = get_recommended_extras()
        assert isinstance(extras, list)
        assert len(extras) > 0


class TestConsoleUtilities:
    """Test console utilities."""

    def test_console_module_imports(self):
        """Verify console module can be imported."""
        from whisperjav.utils.console import (
            ensure_utf8_console,
            safe_print,
            suppress_dependency_warnings,
            setup_console,
            check_extras_installed,
            print_missing_extra_error,
        )

        assert callable(ensure_utf8_console)
        assert callable(safe_print)
        assert callable(suppress_dependency_warnings)
        assert callable(setup_console)
        assert callable(check_extras_installed)
        assert callable(print_missing_extra_error)

    def test_check_extras_installed(self):
        """Verify extras check functionality."""
        from whisperjav.utils.console import check_extras_installed

        # check_extras_installed returns (is_installed: bool, missing: list)
        # Check with a package that should be installed
        is_installed, missing = check_extras_installed(["pathlib"], "test")
        # pathlib is a builtin, should be "installed"
        assert isinstance(is_installed, bool)
        assert isinstance(missing, list)


class TestUpgradeModule:
    """Test upgrade module."""

    def test_upgrade_module_imports(self):
        """Verify upgrade module can be imported."""
        from whisperjav.upgrade import (
            VALID_EXTRAS,
            create_upgrade_snapshot,
            list_snapshots,
            rollback_to_snapshot,
            validate_extras,
            check_upgrade_compatibility,
            upgrade_package_with_extras,
        )

        assert isinstance(VALID_EXTRAS, list)
        assert callable(create_upgrade_snapshot)
        assert callable(list_snapshots)
        assert callable(rollback_to_snapshot)
        assert callable(validate_extras)
        assert callable(check_upgrade_compatibility)
        assert callable(upgrade_package_with_extras)

    def test_validate_extras_valid(self):
        """Test extras validation with valid values."""
        from whisperjav.upgrade import validate_extras

        is_valid, invalid = validate_extras("all")
        assert is_valid is True
        assert invalid == []

        is_valid, invalid = validate_extras("cli,gui")
        assert is_valid is True
        assert invalid == []

    def test_validate_extras_invalid(self):
        """Test extras validation with invalid values."""
        from whisperjav.upgrade import validate_extras

        is_valid, invalid = validate_extras("invalid_extra")
        assert is_valid is False
        assert "invalid_extra" in invalid


class TestEntryPoints:
    """Test CLI entry points."""

    def test_cli_module_exists(self):
        """Test whisperjav.cli module can be imported."""
        # The main entry point may have optional deps checks
        # So we just verify the module structure exists
        from whisperjav import cli
        assert hasattr(cli, "main")
        assert callable(cli.main)

    def test_upgrade_help(self):
        """Test whisperjav-upgrade --help."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.upgrade", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "--extras" in result.stdout
        assert "--rollback" in result.stdout
        assert "--list-snapshots" in result.stdout

    def test_upgrade_version(self):
        """Test whisperjav-upgrade --version."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.upgrade", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        assert "2.0.0" in result.stdout


class TestModuleImports:
    """Test that core modules can be imported."""

    def test_import_whisperjav(self):
        """Test main package import."""
        import whisperjav
        assert hasattr(whisperjav, "__version__")

    def test_import_pipelines(self):
        """Test pipeline imports."""
        from whisperjav.pipelines.faster_pipeline import FasterPipeline
        from whisperjav.pipelines.fast_pipeline import FastPipeline
        from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline

        assert callable(FasterPipeline)
        assert callable(FastPipeline)
        assert callable(FidelityPipeline)

    def test_import_utils(self):
        """Test utility imports."""
        from whisperjav.utils.logger import setup_logger
        from whisperjav.utils.platform import get_platform
        from whisperjav.utils.console import setup_console

        assert callable(setup_logger)
        assert callable(get_platform)
        assert callable(setup_console)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
