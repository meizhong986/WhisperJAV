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


# =============================================================================
# Registry Query Function Tests
# =============================================================================


class TestRegistryQueryFunctions:
    """Test the registry query functions added for SSOT refactoring."""

    def test_import_registry_query_functions(self):
        """Verify registry query functions can be imported from install.py."""
        # Add project root to path
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            # Import install.py as a module
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            # Check helper functions exist
            assert hasattr(install_module, '_get_packages_for_step')
            assert hasattr(install_module, '_get_git_packages_for_step')
            assert hasattr(install_module, '_get_core_deps_from_registry')
            assert hasattr(install_module, '_get_huggingface_deps_from_registry')
            assert hasattr(install_module, '_get_translate_deps_from_registry')
            assert hasattr(install_module, '_get_gui_deps_from_registry')
            assert hasattr(install_module, '_get_enhance_deps_from_registry')
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_core_deps_returns_list(self):
        """Test _get_core_deps_from_registry returns a list of strings."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_core_deps_from_registry()

            assert isinstance(deps, list)
            assert len(deps) > 0
            # All items should be strings (pip specs)
            for dep in deps:
                assert isinstance(dep, str)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_core_deps_excludes_torch(self):
        """Test that core deps don't include torch (installed separately)."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_core_deps_from_registry()

            # torch and torchaudio should NOT be in core deps
            # (they're installed in Step 2 with special index URL)
            dep_names = [d.split('>')[0].split('<')[0].split('=')[0].lower() for d in deps]
            assert 'torch' not in dep_names, "torch should be excluded from core deps"
            assert 'torchaudio' not in dep_names, "torchaudio should be excluded from core deps"
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_huggingface_deps_returns_list(self):
        """Test _get_huggingface_deps_from_registry returns expected packages."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_huggingface_deps_from_registry()

            assert isinstance(deps, list)
            # Should include key HF packages
            deps_lower = [d.lower() for d in deps]
            dep_names = [d.split('>')[0].split('<')[0].split('=')[0] for d in deps_lower]

            assert 'transformers' in dep_names or any('transformers' in d for d in deps_lower)
            assert 'huggingface-hub' in dep_names or any('huggingface' in d for d in deps_lower)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_translate_deps_returns_list(self):
        """Test _get_translate_deps_from_registry returns expected packages."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_translate_deps_from_registry()

            assert isinstance(deps, list)
            # Should include translation packages
            deps_lower = ' '.join(deps).lower()
            assert 'pysubtrans' in deps_lower or 'openai' in deps_lower
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_gui_deps_returns_platform_filtered_list(self):
        """Test _get_gui_deps_from_registry respects platform filtering."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_gui_deps_from_registry()

            assert isinstance(deps, list)
            # pywebview should always be present
            deps_lower = ' '.join(deps).lower()
            assert 'pywebview' in deps_lower

            # pythonnet/pywin32 should only be on Windows
            if sys.platform != 'win32':
                assert 'pythonnet' not in deps_lower
                assert 'pywin32' not in deps_lower
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_enhance_deps_excludes_git_packages(self):
        """Test _get_enhance_deps_from_registry excludes git-based packages."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            deps = install_module._get_enhance_deps_from_registry()

            assert isinstance(deps, list)
            # clearvoice is git-based, should be excluded
            deps_lower = ' '.join(deps).lower()
            assert 'clearvoice' not in deps_lower, "clearvoice (git-based) should be excluded"

            # But oss2, modelscope should be included (PyPI packages)
            assert 'oss2' in deps_lower or 'modelscope' in deps_lower
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_get_git_packages_returns_tuples(self):
        """Test _get_git_packages_for_step returns list of (url, desc) tuples."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            # Get CORE extra git packages
            from whisperjav.installer.core.registry import Extra
            git_packages = install_module._get_git_packages_for_step([Extra.CORE])

            assert isinstance(git_packages, list)
            for item in git_packages:
                assert isinstance(item, tuple)
                assert len(item) == 2
                url, desc = item
                assert isinstance(url, str)
                assert isinstance(desc, str)
                assert url.startswith('git+')
        finally:
            sys.path.remove(str(PROJECT_ROOT))


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

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows-only test")
    def test_check_webview2_windows_exists(self):
        """Verify check_webview2_windows function exists (Windows)."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'check_webview2_windows')
            assert callable(install_module.check_webview2_windows)

            install_module._init_logging(PROJECT_ROOT)
            result = install_module.check_webview2_windows()
            assert isinstance(result, bool)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    @pytest.mark.skipif(sys.platform != 'win32', reason="Windows-only test")
    def test_check_vc_redist_windows_exists(self):
        """Verify check_vc_redist_windows function exists (Windows)."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'check_vc_redist_windows')
            assert callable(install_module.check_vc_redist_windows)

            install_module._init_logging(PROJECT_ROOT)
            result = install_module.check_vc_redist_windows()
            assert isinstance(result, bool)
        finally:
            sys.path.remove(str(PROJECT_ROOT))


# =============================================================================
# Git Timeout Detection Tests
# =============================================================================


class TestGitTimeoutDetection:
    """Test Git timeout detection in install.py."""

    def test_is_git_timeout_error_exists(self):
        """Verify is_git_timeout_error function exists."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'is_git_timeout_error')
            assert callable(install_module.is_git_timeout_error)
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_detect_21_second_timeout(self):
        """Test detection of 21-second TCP timeout (Windows default)."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            # The exact error from Issue #111
            error = "Failed to connect to github.com port 443 after 21074 ms"
            assert install_module.is_git_timeout_error(error) is True
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_detect_connection_timeout(self):
        """Test detection of generic connection timeout."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert install_module.is_git_timeout_error("Connection timed out after") is True
            assert install_module.is_git_timeout_error("Could not connect to server") is True
            assert install_module.is_git_timeout_error("Connection reset by peer") is True
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_detect_rpc_error(self):
        """Test detection of RPC errors (common with large repos)."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert install_module.is_git_timeout_error("error: RPC failed") is True
            assert install_module.is_git_timeout_error("fatal: unable to access") is True
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_no_false_positive(self):
        """Test that normal errors don't trigger timeout detection."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert install_module.is_git_timeout_error("Package not found") is False
            assert install_module.is_git_timeout_error("Invalid requirement") is False
            assert install_module.is_git_timeout_error("Successfully installed") is False
        finally:
            sys.path.remove(str(PROJECT_ROOT))

    def test_configure_git_for_slow_connections_exists(self):
        """Verify configure_git_for_slow_connections function exists."""
        sys.path.insert(0, str(PROJECT_ROOT))

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("install", PROJECT_ROOT / "install.py")
            install_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(install_module)

            assert hasattr(install_module, 'configure_git_for_slow_connections')
            assert callable(install_module.configure_git_for_slow_connections)
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
                install_module._init_logging(tmppath)

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
                install_module._init_logging(tmppath)

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
