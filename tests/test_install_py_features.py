#!/usr/bin/env python3
"""
Install.py Feature Tests
========================

Tests for features added to install.py in v1.8.2:
- Registry query functions (SSOT refactoring)
- Preflight checks (disk space, network, WebView2, VC++ Redist)
- Logging infrastructure
- Git timeout detection and configuration
- Failure file creation

These tests complement test_installer.py and test_installer_comprehensive.py
by testing the orchestration layer (install.py) rather than the installer module.

Author: Senior Architect
Date: 2026-01-28
"""

import sys
import os
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent


# Removed in 1.10.2 (step 5): the tests of the registry query functions, check_webview2_windows,
# check_vc_redist_windows, is_git_timeout_error and configure_git_for_slow_connections. install.py no longer has
# them since its source-install rework (design 22.6); is_git_timeout_error lives in
# whisperjav/installer/core/standalone.py and is tested in tests/test_installer.py.


# =============================================================================
# Preflight Check Tests
# =============================================================================


class TestPreflightChecks:
    """Test preflight check functions in install.py."""

    def test_check_disk_space_exists(self):
        """Verify check_disk_space function exists."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'check_disk_space')
            assert callable(install_module.check_disk_space)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_check_disk_space_returns_bool(self):
        """Test check_disk_space returns boolean."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            # Initialize logging to avoid errors
            install_module._init_logging(PROJECT_ROOT)

            # With a very low threshold, should pass
            result = install_module.check_disk_space(min_gb=0.001)
            assert isinstance(result, bool)
            assert result is True  # 1MB should always be available
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_check_disk_space_high_threshold_fails(self):
        """Test check_disk_space with impossibly high threshold."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            install_module._init_logging(PROJECT_ROOT)

            # 1 million GB should never be available
            result = install_module.check_disk_space(min_gb=1000000)
            assert result is False
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_check_network_exists(self):
        """Verify check_network function exists."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'check_network')
            assert callable(install_module.check_network)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    @pytest.mark.skipif(
        os.environ.get('CI') == 'true',
        reason="Network test may be flaky in CI"
    )
    def test_check_network_returns_bool(self):
        """Test check_network returns boolean."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            install_module._init_logging(PROJECT_ROOT)

            result = install_module.check_network(timeout=10)
            assert isinstance(result, bool)
        finally:
            sys.path.remove(str(PROJECT_ROOT))


# =============================================================================
# Logging Infrastructure Tests
# =============================================================================


class TestLoggingInfrastructure:
    """Test logging functions in install.py."""

    def test_logging_functions_exist(self):
        """Verify logging functions exist."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, '_init_logging')
            assert hasattr(install_module, 'log')
            assert hasattr(install_module, 'log_section')
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_init_logging_creates_file(self):
        """Test _init_logging creates log file."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                install_module._init_logging(tmppath / "install_log.txt")  # takes the file path

                log_file = tmppath / "install_log.txt"
                assert log_file.exists()

                content = log_file.read_text()
                assert "WhisperJAV Installation Log" in content
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_log_writes_to_file(self):
        """Test log() writes to file with timestamp."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            with tempfile.TemporaryDirectory() as tmpdir:
                tmppath = Path(tmpdir)
                install_module._init_logging(tmppath / "install_log.txt")  # takes the file path

                install_module.log("Test message 123")

                log_file = tmppath / "install_log.txt"
                content = log_file.read_text()
                assert "Test message 123" in content
                # Should have timestamp format [HH:MM:SS]
                assert "[" in content and "]" in content
        finally:
            sys.path.remove(str(PROJECT_ROOT))


# =============================================================================
# Failure File Tests
# =============================================================================


class TestFailureFile:
    """Test failure file creation."""

    def test_create_failure_file_exists(self):
        """Verify create_failure_file function exists."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'create_failure_file')
            assert callable(install_module.create_failure_file)
        finally:
            sys.path.remove(str(PROJECT_ROOT))


# =============================================================================
# Lazy Import Tests
# =============================================================================


class TestLazyImports:
    """Test PEP 562 lazy imports in whisperjav/__init__.py."""

    def test_lazy_import_works(self):
        """Test that lazy imports work correctly."""
        import whisperjav

        # __version__ should be immediately available
        assert hasattr(whisperjav, '__version__')
        assert whisperjav.__version__ is not None

    def test_lazy_import_deferred(self):
        """Test that heavy modules are not imported immediately."""
        import whisperjav

        # Check that __all__ is defined
        assert hasattr(whisperjav, '__all__')
        assert isinstance(whisperjav.__all__, list)

    def test_getattr_handles_unknown(self):
        """Test that __getattr__ raises AttributeError for unknown attrs."""
        import whisperjav

        with pytest.raises(AttributeError):
            _ = whisperjav.nonexistent_attribute_xyz


# =============================================================================
# Registry Enhancement Tests
# =============================================================================


class TestRegistryEnhancements:
    """Test registry enhancements (oss2 addition, etc.)."""

    def test_oss2_in_enhance_extra(self):
        """Verify oss2 is in ENHANCE extra (ModelScope dependency)."""
        from whisperjav.installer.core.registry import get_packages_by_extra, Extra

        enhance_packages = get_packages_by_extra(Extra.ENHANCE)
        package_names = [p.name.lower() for p in enhance_packages]

        assert 'oss2' in package_names, "oss2 should be in ENHANCE extra for ModelScope"

    def test_oss2_has_reason(self):
        """Verify oss2 package has a reason documented."""
        from whisperjav.installer.core.registry import get_packages_by_extra, Extra

        enhance_packages = get_packages_by_extra(Extra.ENHANCE)
        oss2_pkg = next((p for p in enhance_packages if p.name.lower() == 'oss2'), None)

        assert oss2_pkg is not None
        assert oss2_pkg.reason, "oss2 should have a documented reason"
        assert 'modelscope' in oss2_pkg.reason.lower() or 'aliyun' in oss2_pkg.reason.lower()


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
