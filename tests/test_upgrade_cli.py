#!/usr/bin/env python3
"""
Unit tests for whisperjav.upgrade CLI module.

Tests argument parsing, installation detection, and upgrade flow logic.
Does NOT test actual pip installations (those require integration tests).

Run with: pytest tests/test_upgrade_cli.py -v
"""

import os
import sys
import pytest
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import whisperjav.upgrade as upgrade


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_install_dir(tmp_path):
    """Create a mock WhisperJAV installation directory."""
    install_dir = tmp_path / "WhisperJAV"
    install_dir.mkdir()

    # Create mock python.exe
    (install_dir / "python.exe").touch()

    # Create Scripts directory with pip
    scripts_dir = install_dir / "Scripts"
    scripts_dir.mkdir()
    (scripts_dir / "pip.exe").touch()
    (scripts_dir / "whisperjav-gui.exe").touch()

    return install_dir


@pytest.fixture
def reset_module(monkeypatch):
    """Reset module-level constants after each test."""
    original_repo = upgrade.GITHUB_REPO
    yield
    monkeypatch.setattr(upgrade, 'GITHUB_REPO', original_repo)


# =============================================================================
# Argument Parsing Tests
# =============================================================================

class TestParseArgs:
    """Tests for parse_args() function."""

    def test_no_args(self):
        """Test parsing with no arguments."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade']):
            args = upgrade.parse_args()
            assert args.check is False
            assert args.yes is False
            assert args.wheel_only is False
            assert args.version is False
            assert args.force is False

    def test_check_flag(self):
        """Test --check / -c flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--check']):
            args = upgrade.parse_args()
            assert args.check is True

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '-c']):
            args = upgrade.parse_args()
            assert args.check is True

    def test_yes_flag(self):
        """Test --yes / -y flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--yes']):
            args = upgrade.parse_args()
            assert args.yes is True

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '-y']):
            args = upgrade.parse_args()
            assert args.yes is True

    def test_wheel_only_flag(self):
        """Test --wheel-only / -w flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--wheel-only']):
            args = upgrade.parse_args()
            assert args.wheel_only is True

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '-w']):
            args = upgrade.parse_args()
            assert args.wheel_only is True

    def test_version_flag(self):
        """Test --version / -v flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--version']):
            args = upgrade.parse_args()
            assert args.version is True

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '-v']):
            args = upgrade.parse_args()
            assert args.version is True

    def test_force_flag(self):
        """Test --force flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--force']):
            args = upgrade.parse_args()
            assert args.force is True

    def test_combined_flags(self):
        """Test combining multiple flags."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '-y', '-w', '--force']):
            args = upgrade.parse_args()
            assert args.yes is True
            assert args.wheel_only is True
            assert args.force is True
            assert args.check is False

    def test_check_with_force(self):
        """Test --check --force combination."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--check', '--force']):
            args = upgrade.parse_args()
            assert args.check is True
            assert args.force is True


# =============================================================================
# Installation Detection Tests
# =============================================================================

class TestDetectInstallation:
    """Tests for detect_installation() function."""

    def test_detect_from_sys_prefix(self, temp_install_dir, monkeypatch):
        """Test detection when running from within installation."""
        # Mock sys.prefix to point to our temp dir
        monkeypatch.setattr(sys, 'prefix', str(temp_install_dir))

        # Mock whisperjav import
        mock_whisperjav = MagicMock()
        monkeypatch.setitem(sys.modules, 'whisperjav', mock_whisperjav)

        result = upgrade.detect_installation()
        assert result == temp_install_dir

    def test_detect_from_localappdata(self, tmp_path, monkeypatch):
        """Test detection from LOCALAPPDATA default location."""
        # Create installation in fake LOCALAPPDATA
        fake_localappdata = tmp_path / "localappdata"
        fake_localappdata.mkdir()
        whisperjav_dir = fake_localappdata / "WhisperJAV"
        whisperjav_dir.mkdir()
        (whisperjav_dir / "python.exe").touch()

        monkeypatch.setenv('LOCALAPPDATA', str(fake_localappdata))
        monkeypatch.setattr(sys, 'prefix', '/some/other/path')

        # Mock platform
        with patch('whisperjav.upgrade.platform.system', return_value='Windows'):
            result = upgrade.detect_installation()
            assert result == whisperjav_dir

    def test_no_installation_found(self, monkeypatch):
        """Test when no installation can be found."""
        monkeypatch.setattr(sys, 'prefix', '/nonexistent')
        monkeypatch.setenv('LOCALAPPDATA', '/nonexistent')

        # Make sure whisperjav import fails
        if 'whisperjav' in sys.modules:
            monkeypatch.delitem(sys.modules, 'whisperjav', raising=False)

        with patch('whisperjav.upgrade.platform.system', return_value='Windows'):
            result = upgrade.detect_installation()
            assert result is None


# =============================================================================
# Version Detection Tests
# =============================================================================

class TestGetCurrentVersion:
    """Tests for get_current_version() function."""

    def test_get_version_success(self, temp_install_dir, monkeypatch):
        """Test successful version detection via subprocess."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1.7.4\n"

        with patch('subprocess.run', return_value=mock_result):
            version = upgrade.get_current_version(temp_install_dir)
            assert version == "1.7.4"

    def test_get_version_failure(self, temp_install_dir):
        """Test version detection when subprocess fails."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""

        with patch('subprocess.run', return_value=mock_result):
            version = upgrade.get_current_version(temp_install_dir)
            assert version is None

    def test_get_version_no_python(self, tmp_path):
        """Test version detection when python.exe doesn't exist."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        version = upgrade.get_current_version(empty_dir)
        assert version is None


# =============================================================================
# Network Check Tests
# =============================================================================

class TestCheckNetwork:
    """Tests for check_network() function."""

    def test_network_available(self):
        """Test when network is available."""
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.return_value.__enter__ = MagicMock()
            mock_urlopen.return_value.__exit__ = MagicMock()
            assert upgrade.check_network() is True

    def test_network_unavailable(self):
        """Test when network is unavailable."""
        with patch('urllib.request.urlopen', side_effect=Exception("No network")):
            assert upgrade.check_network() is False


# =============================================================================
# Environment Variable Configuration Tests
# =============================================================================

class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    def test_custom_upgrade_repo(self, monkeypatch):
        """Test that WHISPERJAV_UPGRADE_REPO overrides default."""
        custom_repo = "git+https://github.com/test/stub.git@v1.7.5"
        monkeypatch.setenv('WHISPERJAV_UPGRADE_REPO', custom_repo)

        # Re-import to pick up new env var
        if 'whisperjav.upgrade' in sys.modules:
            del sys.modules['whisperjav.upgrade']

        import whisperjav.upgrade as u
        assert u.GITHUB_REPO == custom_repo

    def test_default_upgrade_repo(self):
        """Test default GITHUB_REPO value."""
        assert "github.com/meizhong986/whisperjav" in upgrade.GITHUB_REPO


# =============================================================================
# Upgrade Package Tests (with mocked subprocess)
# =============================================================================

class TestUpgradePackage:
    """Tests for upgrade_package() function with mocked subprocess."""

    def test_upgrade_success(self, temp_install_dir, capsys):
        """Test successful package upgrade."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed whisperjav"
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result):
            result = upgrade.upgrade_package(temp_install_dir)
            assert result is True

        captured = capsys.readouterr()
        assert "successfully" in captured.out.lower()

    def test_upgrade_failure(self, temp_install_dir, capsys):
        """Test failed package upgrade."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: Could not find package"

        with patch('subprocess.run', return_value=mock_result):
            result = upgrade.upgrade_package(temp_install_dir)
            assert result is False

        captured = capsys.readouterr()
        assert "failed" in captured.out.lower()

    def test_upgrade_timeout(self, temp_install_dir, capsys):
        """Test package upgrade timeout."""
        with patch('subprocess.run', side_effect=subprocess.TimeoutExpired("pip", 1800)):
            result = upgrade.upgrade_package(temp_install_dir)
            assert result is False

        captured = capsys.readouterr()
        assert "timed out" in captured.out.lower()

    def test_upgrade_no_pip(self, tmp_path, capsys):
        """Test upgrade when pip is not found."""
        empty_dir = tmp_path / "nopip"
        empty_dir.mkdir()

        result = upgrade.upgrade_package(empty_dir)
        assert result is False

        captured = capsys.readouterr()
        assert "pip not found" in captured.out.lower()


# =============================================================================
# Wheel-Only Upgrade Tests
# =============================================================================

class TestUpgradePackageWheelOnly:
    """Tests for upgrade_package_wheel_only() function."""

    def test_wheel_only_success(self, temp_install_dir, capsys):
        """Test successful wheel-only upgrade."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Successfully installed whisperjav"
        mock_result.stderr = ""

        with patch('subprocess.run', return_value=mock_result) as mock_run:
            result = upgrade.upgrade_package_wheel_only(temp_install_dir)
            assert result is True

            # Verify --no-deps was passed
            call_args = mock_run.call_args[0][0]
            assert '--no-deps' in call_args

        captured = capsys.readouterr()
        assert "wheel only" in captured.out.lower()

    def test_wheel_only_failure(self, temp_install_dir, capsys):
        """Test failed wheel-only upgrade."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Error"

        with patch('subprocess.run', return_value=mock_result):
            result = upgrade.upgrade_package_wheel_only(temp_install_dir)
            assert result is False


# =============================================================================
# Cleanup Tests
# =============================================================================

class TestCleanupOldFiles:
    """Tests for cleanup_old_files() function."""

    def test_cleanup_removes_old_files(self, tmp_path):
        """Test that old version-specific files are removed."""
        install_dir = tmp_path / "install"
        install_dir.mkdir()

        # Create old files that should be cleaned up
        old_files = [
            "install_log_v1.5.1.txt",
            "install_log_v1.6.0.txt",
            "WhisperJAV_Launcher_v1.5.1.py",
            "post_install_v1.5.1.py",
            "requirements_v1.5.1.txt",
        ]

        for f in old_files:
            (install_dir / f).touch()

        # Create files that should NOT be cleaned
        keep_files = [
            "important.txt",
            "python.exe",
            "whisperjav_config.json",
        ]

        for f in keep_files:
            (install_dir / f).touch()

        cleaned = upgrade.cleanup_old_files(install_dir)

        assert cleaned == len(old_files)

        # Verify old files are gone
        for f in old_files:
            assert not (install_dir / f).exists()

        # Verify kept files remain
        for f in keep_files:
            assert (install_dir / f).exists()

    def test_cleanup_empty_directory(self, tmp_path):
        """Test cleanup on directory with no matching files."""
        install_dir = tmp_path / "empty"
        install_dir.mkdir()

        cleaned = upgrade.cleanup_old_files(install_dir)
        assert cleaned == 0


# =============================================================================
# Check for Updates CLI Tests
# =============================================================================

class TestCheckForUpdatesCLI:
    """Tests for check_for_updates_cli() function."""

    def test_cli_update_available(self, capsys):
        """Test CLI output when update is available."""
        mock_result = MagicMock()
        mock_result.update_available = True
        mock_result.current_version = "1.7.4"
        mock_result.latest_version = "1.8.0"
        mock_result.error = None
        mock_result.version_info = MagicMock()
        mock_result.version_info.release_url = "https://example.com"
        mock_result.version_info.release_notes = "New features"

        # Patch where check_for_updates is actually used (imported in upgrade.py)
        with patch('whisperjav.version_checker.check_for_updates', return_value=mock_result):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                with patch('whisperjav.version_checker.CURRENT_VERSION', '1.7.4'):
                    exit_code = upgrade.check_for_updates_cli(force=True)

        assert exit_code == 1  # Update available
        captured = capsys.readouterr()
        assert "update available" in captured.out.lower()
        assert "1.8.0" in captured.out

    def test_cli_no_update(self, capsys):
        """Test CLI output when no update is available."""
        mock_result = MagicMock()
        mock_result.update_available = False
        mock_result.current_version = "1.8.0"
        mock_result.latest_version = "1.8.0"
        mock_result.error = None

        with patch('whisperjav.version_checker.check_for_updates', return_value=mock_result):
            with patch('whisperjav.version_checker.CURRENT_VERSION', '1.8.0'):
                exit_code = upgrade.check_for_updates_cli(force=True)

        assert exit_code == 0  # Up to date
        captured = capsys.readouterr()
        assert "latest version" in captured.out.lower()

    def test_cli_error(self, capsys):
        """Test CLI output when error occurs."""
        mock_result = MagicMock()
        mock_result.update_available = False
        mock_result.error = "Network error"

        with patch('whisperjav.version_checker.check_for_updates', return_value=mock_result):
            with patch('whisperjav.version_checker.CURRENT_VERSION', '1.7.4'):
                exit_code = upgrade.check_for_updates_cli(force=True)

        assert exit_code == 2  # Error
        captured = capsys.readouterr()
        assert "error" in captured.out.lower()


# =============================================================================
# Main Entry Point Tests
# =============================================================================

class TestMain:
    """Tests for main() entry point."""

    def test_main_version_flag(self, capsys):
        """Test main() with --version flag."""
        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--version']):
            exit_code = upgrade.main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert upgrade.UPGRADE_SCRIPT_VERSION in captured.out

    def test_main_check_flag(self, capsys):
        """Test main() with --check flag delegates to check_for_updates_cli."""
        # Mock the actual function that does the work
        mock_result = MagicMock()
        mock_result.update_available = False
        mock_result.current_version = "1.7.4"
        mock_result.error = None

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--check']):
            with patch('whisperjav.version_checker.check_for_updates', return_value=mock_result):
                with patch('whisperjav.version_checker.CURRENT_VERSION', '1.7.4'):
                    exit_code = upgrade.main()

        assert exit_code == 0  # No update available
        captured = capsys.readouterr()
        assert "latest version" in captured.out.lower()

    def test_main_check_with_force(self, capsys):
        """Test main() with --check --force passes force=True."""
        mock_result = MagicMock()
        mock_result.update_available = False
        mock_result.current_version = "1.7.4"
        mock_result.error = None

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--check', '--force']):
            with patch('whisperjav.version_checker.check_for_updates', return_value=mock_result) as mock_check:
                with patch('whisperjav.version_checker.CURRENT_VERSION', '1.7.4'):
                    exit_code = upgrade.main()

        # Verify force=True was passed
        mock_check.assert_called_once_with(force=True)

    def test_main_no_installation(self, capsys, monkeypatch, tmp_path):
        """Test main() when no installation is found."""
        # Create isolation from real environment
        fake_prefix = tmp_path / "fake_env"
        fake_prefix.mkdir()

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--yes']):
            with patch.object(sys, 'prefix', str(fake_prefix)):
                with patch('whisperjav.upgrade.platform.system', return_value='Windows'):
                    # Point LOCALAPPDATA to empty temp dir
                    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
                    exit_code = upgrade.main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower()


# =============================================================================
# Launcher Update Tests
# =============================================================================

class TestUpdateLauncher:
    """Tests for update_launcher() function."""

    def test_update_launcher_success(self, temp_install_dir, capsys):
        """Test successful launcher update."""
        # Create source file
        (temp_install_dir / "Scripts" / "whisperjav-gui.exe").touch()

        result = upgrade.update_launcher(temp_install_dir)

        assert result is True
        assert (temp_install_dir / "WhisperJAV-GUI.exe").exists()

    def test_update_launcher_no_source(self, temp_install_dir, capsys):
        """Test launcher update when source doesn't exist."""
        # Remove the source file
        (temp_install_dir / "Scripts" / "whisperjav-gui.exe").unlink()

        result = upgrade.update_launcher(temp_install_dir)

        # Should return True (non-fatal) but with warning
        assert result is True
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "fallback" in captured.out.lower()


# =============================================================================
# Integration Test: Full Upgrade Flow (mocked)
# =============================================================================

class TestFullUpgradeFlow:
    """Integration test for the full upgrade flow with mocked subprocess."""

    def test_full_upgrade_flow_success(self, temp_install_dir, capsys, monkeypatch):
        """Test complete upgrade flow succeeds."""
        # Isolate from real environment
        monkeypatch.setattr(sys, 'prefix', str(temp_install_dir))

        # Mock all subprocess calls
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "1.8.0"
        mock_result.stderr = ""

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--yes']):
            with patch('whisperjav.upgrade.detect_installation', return_value=temp_install_dir):
                with patch('subprocess.run', return_value=mock_result):
                    with patch('whisperjav.upgrade.check_network', return_value=True):
                        with patch('whisperjav.upgrade.update_desktop_shortcut', return_value=True):
                            exit_code = upgrade.main()

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "complete" in captured.out.lower()

    def test_full_upgrade_flow_network_error(self, temp_install_dir, capsys, monkeypatch):
        """Test upgrade fails gracefully on network error."""
        # Isolate from real environment
        monkeypatch.setattr(sys, 'prefix', str(temp_install_dir))

        with patch.object(sys, 'argv', ['whisperjav-upgrade', '--yes']):
            with patch('whisperjav.upgrade.detect_installation', return_value=temp_install_dir):
                with patch('whisperjav.upgrade.get_current_version', return_value="1.7.4"):
                    with patch('whisperjav.upgrade.check_network', return_value=False):
                        exit_code = upgrade.main()

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "internet" in captured.out.lower() or "network" in captured.out.lower()


# =============================================================================
# Run directly for debugging
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
