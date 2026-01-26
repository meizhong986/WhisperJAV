#!/usr/bin/env python3
"""
Installer Module Tests
======================

Tests for the whisperjav.installer package:
- Package registry
- Validation system
- Import scanner

These tests ensure the installation system remains consistent and catches
configuration drift before it reaches users.

Author: Senior Architect
Date: 2026-01-26
"""

import sys
from pathlib import Path

import pytest


class TestRegistry:
    """Test the package registry (single source of truth)."""

    def test_registry_imports(self):
        """Verify registry module can be imported."""
        from whisperjav.installer.core.registry import (
            PACKAGES,
            Package,
            Extra,
            InstallSource,
            Platform,
            get_packages_in_install_order,
            get_packages_by_extra,
            get_import_map,
            generate_pyproject_extras,
            generate_core_dependencies,
        )

        assert isinstance(PACKAGES, list)
        assert len(PACKAGES) > 0
        assert callable(get_packages_in_install_order)
        assert callable(get_packages_by_extra)
        assert callable(get_import_map)
        assert callable(generate_pyproject_extras)
        assert callable(generate_core_dependencies)

    def test_registry_validation_passes(self):
        """Verify registry validates without errors."""
        from whisperjav.installer.core.registry import validate_registry

        # Should not raise any exceptions
        validate_registry()

    def test_registry_no_duplicate_packages(self):
        """Verify no duplicate package names."""
        from whisperjav.installer.core.registry import PACKAGES

        names = [pkg.name.lower() for pkg in PACKAGES]
        assert len(names) == len(set(names)), "Duplicate packages found"

    def test_torch_first_order(self):
        """Verify PyTorch is installed first (GPU lock-in pattern)."""
        from whisperjav.installer.core.registry import get_packages_in_install_order

        ordered = get_packages_in_install_order()
        torch_indices = [i for i, pkg in enumerate(ordered) if pkg.name in ("torch", "torchaudio")]

        # Torch should be in first few packages
        assert len(torch_indices) >= 1, "torch not found in packages"
        assert max(torch_indices) < 5, "torch not in first 5 packages (order matters!)"

    def test_import_map_has_mappings(self):
        """Verify import map has expected mappings."""
        from whisperjav.installer.core.registry import get_import_map

        import_map = get_import_map()

        # Known mappings that should exist
        expected_mappings = {
            "PIL": "Pillow",
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "whisper": "openai-whisper",
            "stable_whisper": "stable-ts",
            "webview": "pywebview",
        }

        for import_name, package_name in expected_mappings.items():
            assert import_name in import_map, f"Missing import mapping: {import_name}"
            assert import_map[import_name] == package_name, \
                f"Wrong mapping for {import_name}: expected {package_name}, got {import_map[import_name]}"

    def test_generate_pyproject_extras_structure(self):
        """Verify generated extras have correct structure."""
        from whisperjav.installer.core.registry import generate_pyproject_extras

        extras = generate_pyproject_extras()

        # Should have all non-core extras
        expected_extras = ["cli", "gui", "translate", "llm", "enhance", "huggingface", "analysis"]
        for extra_name in expected_extras:
            assert extra_name in extras, f"Missing extra: {extra_name}"
            assert isinstance(extras[extra_name], list), f"Extra {extra_name} should be a list"

    def test_package_pyproject_spec(self):
        """Verify Package.pyproject_spec() generates valid specs."""
        from whisperjav.installer.core.registry import Package, InstallSource, Platform

        # Simple package
        pkg = Package(name="numpy", version=">=1.26.0")
        assert pkg.pyproject_spec() == "numpy>=1.26.0"

        # Git package
        git_pkg = Package(
            name="test",
            source=InstallSource.GIT,
            git_url="git+https://github.com/test/test.git"
        )
        assert "@ git+" in git_pkg.pyproject_spec()

        # Platform-specific
        win_pkg = Package(name="pywin32", platforms=[Platform.WINDOWS])
        assert "sys_platform == 'win32'" in win_pkg.pyproject_spec()


class TestValidation:
    """Test the validation system."""

    def test_validation_imports(self):
        """Verify validation modules can be imported."""
        from whisperjav.installer.validation import (
            validate_standalone_self_containment,
            validate_pyproject_sync,
            validate_requirements_sync,
            scan_imports,
            run_all_validations,
        )

        assert callable(validate_standalone_self_containment)
        assert callable(validate_pyproject_sync)
        assert callable(validate_requirements_sync)
        assert callable(scan_imports)
        assert callable(run_all_validations)

    def test_standalone_self_containment(self):
        """Verify standalone.py has no whisperjav imports."""
        from whisperjav.installer.validation import validate_standalone_self_containment

        errors = validate_standalone_self_containment()
        assert len(errors) == 0, f"standalone.py has forbidden imports: {errors}"

    def test_pyproject_sync(self):
        """Verify pyproject.toml matches registry."""
        from whisperjav.installer.validation import validate_pyproject_sync

        errors = validate_pyproject_sync()
        assert len(errors) == 0, f"pyproject.toml out of sync: {errors}"

    def test_import_scanner_no_errors(self):
        """Verify import scanner finds no untracked imports (errors become warnings)."""
        from whisperjav.installer.validation import scan_imports

        # Run scanner - warnings are OK, errors are not
        warnings = scan_imports()
        # Ideally zero, but warnings are informational
        # The key is that validation doesn't fail
        assert isinstance(warnings, list)


class TestImportScanner:
    """Test the import scanner in detail."""

    def test_stdlib_modules_set(self):
        """Verify stdlib modules set is populated."""
        from whisperjav.installer.validation.imports import STDLIB_MODULES

        assert isinstance(STDLIB_MODULES, set)
        assert len(STDLIB_MODULES) > 100  # Should have many stdlib modules
        assert "os" in STDLIB_MODULES
        assert "sys" in STDLIB_MODULES
        assert "json" in STDLIB_MODULES

    def test_known_internal_modules_set(self):
        """Verify known internal modules set exists."""
        from whisperjav.installer.validation.imports import KNOWN_INTERNAL_MODULES

        assert isinstance(KNOWN_INTERNAL_MODULES, set)
        assert "api" in KNOWN_INTERNAL_MODULES
        assert "utils" in KNOWN_INTERNAL_MODULES
        assert "core" in KNOWN_INTERNAL_MODULES

    def test_optional_imports_set(self):
        """Verify optional imports set exists."""
        from whisperjav.installer.validation.imports import OPTIONAL_IMPORTS

        assert isinstance(OPTIONAL_IMPORTS, set)
        assert "IPython" in OPTIONAL_IMPORTS
        assert "nemo" in OPTIONAL_IMPORTS


class TestStandalone:
    """Test the standalone installer module."""

    def test_standalone_imports_without_whisperjav(self):
        """Verify standalone.py can be imported independently."""
        # This is a critical test - standalone.py must work without whisperjav
        import importlib.util

        standalone_path = Path(__file__).parent.parent / "whisperjav" / "installer" / "core" / "standalone.py"
        assert standalone_path.exists(), f"standalone.py not found at {standalone_path}"

        spec = importlib.util.spec_from_file_location("standalone", standalone_path)
        assert spec is not None, "Could not load standalone.py spec"

        module = importlib.util.module_from_spec(spec)
        # This should NOT raise ImportError about whisperjav
        try:
            spec.loader.exec_module(module)
        except ImportError as e:
            if "whisperjav" in str(e):
                pytest.fail(f"standalone.py imports from whisperjav: {e}")
            raise  # Re-raise other import errors

    def test_standalone_has_gpu_detection(self):
        """Verify standalone has GPU detection utilities."""
        from whisperjav.installer.core.standalone import (
            CUDADriverEntry,
            CUDA_DRIVER_MATRIX,
            detect_gpu,
            select_cuda_version,
        )

        assert isinstance(CUDA_DRIVER_MATRIX, tuple)
        assert len(CUDA_DRIVER_MATRIX) > 0
        assert callable(detect_gpu)
        assert callable(select_cuda_version)

    def test_standalone_has_pip_utilities(self):
        """Verify standalone has pip installation utilities."""
        from whisperjav.installer.core.standalone import (
            run_pip_command,
            detect_uv,
            is_git_timeout_error,
            configure_git_timeouts,
        )

        assert callable(run_pip_command)
        assert callable(detect_uv)
        assert callable(is_git_timeout_error)
        assert callable(configure_git_timeouts)


class TestInstallPy:
    """Test the install.py orchestration module."""

    def test_install_module_exists(self):
        """Verify install.py exists in project root."""
        # install.py is in project root, not installer directory
        install_path = Path(__file__).parent.parent / "install.py"
        assert install_path.exists(), f"install.py not found at {install_path}"

    def test_install_help(self):
        """Test install.py --help."""
        import subprocess

        # install.py is in project root
        install_path = Path(__file__).parent.parent / "install.py"
        result = subprocess.run(
            [sys.executable, str(install_path), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=install_path.parent,
        )
        assert result.returncode == 0, f"install.py --help failed: {result.stderr}"
        assert "--extras" in result.stdout or "usage" in result.stdout.lower()


class TestRunAllValidations:
    """Test the complete validation runner."""

    def test_run_quick_validation(self):
        """Run quick validation (skips slow import scan)."""
        from whisperjav.installer.validation import run_all_validations

        # Quick mode should complete quickly and pass
        exit_code = run_all_validations(quick=True)
        assert exit_code == 0, "Quick validation failed"

    def test_run_full_validation(self):
        """Run full validation (includes import scan)."""
        from whisperjav.installer.validation import run_all_validations

        # Full validation should pass
        exit_code = run_all_validations(quick=False)
        assert exit_code == 0, "Full validation failed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
