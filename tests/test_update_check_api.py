#!/usr/bin/env python3
"""
Unit tests for WhisperJAV GUI Update Check API.

Tests the update check modal feature including:
- check_for_updates() API method
- dismiss_update_notification() API method
- get_dismissed_update() API method
- open_url() API method
- start_update() API method

Run with: pytest tests/test_update_check_api.py -v
"""

import json
import os
import pytest
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple
from unittest.mock import patch, MagicMock, Mock


# =============================================================================
# Mock Classes to simulate version_checker types
# =============================================================================

class MockVersionInfo(NamedTuple):
    """Mock VersionInfo to avoid importing actual module."""
    version: str
    release_url: str
    release_notes: str
    published_at: str
    is_prerelease: bool
    installer_url: str = None


class MockUpdateCheckResult(NamedTuple):
    """Mock UpdateCheckResult to avoid importing actual module."""
    update_available: bool
    current_version: str
    latest_version: str
    version_info: MockVersionInfo = None
    from_cache: bool = False
    error: str = None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory."""
    cache_dir = tmp_path / ".whisperjav_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_api(tmp_path, monkeypatch):
    """
    Create a mock API instance with patched dependencies.

    This patches the environment to use temp directories and
    avoids importing heavy dependencies like webview.
    """
    # Patch LOCALAPPDATA to use temp directory
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

    # Create a minimal API class for testing
    # We import just the methods we need to test
    from whisperjav.webview_gui.api import WhisperJAVAPI

    # Create API instance (may need window mock)
    with patch('webview.create_window'):
        api = WhisperJAVAPI.__new__(WhisperJAVAPI)
        api.window = None
        api._process = None
        api._output_queue = None
        api._log_thread = None
        api._status = {"running": False}

    return api


@pytest.fixture
def sample_update_result():
    """Sample update check result with update available."""
    return MockUpdateCheckResult(
        update_available=True,
        current_version="1.7.5",
        latest_version="1.8.0",
        version_info=MockVersionInfo(
            version="1.8.0",
            release_url="https://github.com/meizhong986/whisperjav/releases/tag/v1.8.0",
            release_notes="## What's New\n- Feature A\n- Feature B\n- Bug fixes",
            published_at="2025-01-15T10:00:00Z",
            is_prerelease=False,
            installer_url="https://github.com/meizhong986/whisperjav/releases/download/v1.8.0/WhisperJAV-1.8.0-Windows-x86_64.exe"
        ),
        from_cache=False,
        error=None
    )


@pytest.fixture
def sample_no_update_result():
    """Sample update check result with no update available."""
    return MockUpdateCheckResult(
        update_available=False,
        current_version="1.8.0",
        latest_version="1.8.0",
        version_info=None,
        from_cache=True,
        error=None
    )


@pytest.fixture
def sample_error_result():
    """Sample update check result with error."""
    return MockUpdateCheckResult(
        update_available=False,
        current_version="1.7.5",
        latest_version=None,
        version_info=None,
        from_cache=False,
        error="Network error: Unable to connect to GitHub"
    )


# =============================================================================
# check_for_updates() Tests
# =============================================================================

class TestCheckForUpdates:
    """Tests for check_for_updates() API method."""

    def test_update_available_returns_correct_structure(self, mock_api, sample_update_result):
        """Test that update available response has correct structure."""
        with patch('whisperjav.version_checker.check_for_updates', return_value=sample_update_result):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                result = mock_api.check_for_updates(force=True)

        assert result["success"] is True
        assert result["update_available"] is True
        assert result["current_version"] == "1.7.5"
        assert result["latest_version"] == "1.8.0"
        assert result["notification_level"] == "minor"
        assert result["release_url"] is not None
        assert result["release_notes"] is not None
        assert result["from_cache"] is False

    def test_no_update_available(self, mock_api, sample_no_update_result):
        """Test response when no update is available."""
        with patch('whisperjav.version_checker.check_for_updates', return_value=sample_no_update_result):
            result = mock_api.check_for_updates(force=False)

        assert result["success"] is True
        assert result["update_available"] is False
        assert result["current_version"] == "1.8.0"
        assert result["from_cache"] is True
        # No notification_level, release_url, release_notes when no update
        assert "notification_level" not in result or result.get("notification_level") is None

    def test_error_handling(self, mock_api, sample_error_result):
        """Test response when update check fails."""
        with patch('whisperjav.version_checker.check_for_updates', return_value=sample_error_result):
            result = mock_api.check_for_updates(force=True)

        assert result["success"] is True  # Method succeeded, but check had error
        assert result["update_available"] is False
        assert "error" in result
        assert "Network error" in result["error"]

    def test_exception_handling(self, mock_api):
        """Test that exceptions are caught and returned as error."""
        with patch('whisperjav.version_checker.check_for_updates', side_effect=Exception("Import failed")):
            result = mock_api.check_for_updates(force=True)

        assert result["success"] is False
        assert "Import failed" in result["error"]
        assert result["update_available"] is False

    def test_force_bypasses_cache(self, mock_api, sample_update_result):
        """Test that force=True bypasses cache."""
        mock_check = Mock(return_value=sample_update_result)
        with patch('whisperjav.version_checker.check_for_updates', mock_check):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                mock_api.check_for_updates(force=True)

        # Verify force=True was passed
        mock_check.assert_called_once_with(force=True)

    def test_release_notes_truncation(self, mock_api):
        """Test that long release notes are truncated."""
        long_notes = "A" * 1000  # 1000 character release notes
        result_with_long_notes = MockUpdateCheckResult(
            update_available=True,
            current_version="1.7.5",
            latest_version="1.8.0",
            version_info=MockVersionInfo(
                version="1.8.0",
                release_url="https://example.com",
                release_notes=long_notes,
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=False
            ),
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=result_with_long_notes):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                result = mock_api.check_for_updates(force=True)

        # Release notes should be truncated to 500 chars
        assert len(result["release_notes"]) <= 500


class TestNotificationLevels:
    """Tests for notification level classification."""

    @pytest.mark.parametrize("current,latest,expected_level", [
        ("1.7.5", "1.7.6", "patch"),      # Patch update
        ("1.7.5", "1.8.0", "minor"),      # Minor update
        ("1.7.5", "2.0.0", "major"),      # Major update
        ("1.7.5", "1.7.5", "none"),       # No update
    ])
    def test_notification_level_classification(self, mock_api, current, latest, expected_level):
        """Test that notification levels are correctly classified."""
        result = MockUpdateCheckResult(
            update_available=(current != latest),
            current_version=current,
            latest_version=latest,
            version_info=MockVersionInfo(
                version=latest,
                release_url="https://example.com",
                release_notes="Test notes",
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=False
            ) if current != latest else None,
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=result):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value=expected_level):
                response = mock_api.check_for_updates(force=True)

        if current != latest:
            assert response.get("notification_level") == expected_level


# =============================================================================
# dismiss_update_notification() Tests
# =============================================================================

class TestDismissUpdateNotification:
    """Tests for dismiss_update_notification() API method."""

    def test_dismiss_creates_file(self, mock_api, tmp_path, monkeypatch):
        """Test that dismissing creates a dismissal file."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        result = mock_api.dismiss_update_notification("1.8.0")

        assert result["success"] is True

        # Check file was created
        dismissal_file = tmp_path / '.whisperjav_cache' / 'update_dismissed.json'
        assert dismissal_file.exists()

        # Verify contents
        with open(dismissal_file) as f:
            data = json.load(f)
        assert data["version"] == "1.8.0"
        assert "dismissed_at" in data

    def test_dismiss_overwrites_previous(self, mock_api, tmp_path, monkeypatch):
        """Test that new dismissal overwrites previous."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        # First dismissal
        mock_api.dismiss_update_notification("1.7.6")

        # Second dismissal
        mock_api.dismiss_update_notification("1.8.0")

        # Check only latest version is stored
        dismissal_file = tmp_path / '.whisperjav_cache' / 'update_dismissed.json'
        with open(dismissal_file) as f:
            data = json.load(f)
        assert data["version"] == "1.8.0"

    def test_dismiss_error_handling(self, mock_api, monkeypatch):
        """Test error handling when dismissal fails."""
        # Mock the file open to raise an error
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            with patch('pathlib.Path.mkdir'):  # Don't fail on mkdir
                result = mock_api.dismiss_update_notification("1.8.0")

        # Should return error, not crash
        assert result["success"] is False
        assert "error" in result


# =============================================================================
# get_dismissed_update() Tests
# =============================================================================

class TestGetDismissedUpdate:
    """Tests for get_dismissed_update() API method."""

    def test_get_dismissed_when_exists(self, mock_api, tmp_path, monkeypatch):
        """Test retrieving dismissed update info."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        # Create dismissal first
        mock_api.dismiss_update_notification("1.8.0")

        result = mock_api.get_dismissed_update()

        assert result["success"] is True
        assert result["dismissed"]["version"] == "1.8.0"
        assert "dismissed_at" in result["dismissed"]

    def test_get_dismissed_when_not_exists(self, mock_api, tmp_path, monkeypatch):
        """Test when no dismissal exists."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))
        # Don't create any dismissal file

        result = mock_api.get_dismissed_update()

        assert result["success"] is True
        assert result["dismissed"] is None


# =============================================================================
# open_url() Tests
# =============================================================================

class TestOpenUrl:
    """Tests for open_url() API method."""

    def test_open_url_success(self, mock_api):
        """Test successful URL opening."""
        with patch('webbrowser.open', return_value=True) as mock_open:
            result = mock_api.open_url("https://github.com/meizhong986/WhisperJAV/releases")

        assert result["success"] is True
        mock_open.assert_called_once_with("https://github.com/meizhong986/WhisperJAV/releases")

    def test_open_url_github_releases(self, mock_api):
        """Test opening GitHub releases page specifically."""
        with patch('webbrowser.open', return_value=True) as mock_open:
            result = mock_api.open_url("https://github.com/meizhong986/WhisperJAV/releases")

        assert result["success"] is True
        # Verify the exact URL used for major updates
        mock_open.assert_called_with("https://github.com/meizhong986/WhisperJAV/releases")

    def test_open_url_error_handling(self, mock_api):
        """Test error handling when browser fails to open."""
        with patch('webbrowser.open', side_effect=Exception("No browser available")):
            result = mock_api.open_url("https://example.com")

        assert result["success"] is False
        assert "No browser available" in result["error"]

    def test_open_url_various_urls(self, mock_api):
        """Test opening various URL types."""
        urls = [
            "https://github.com/meizhong986/WhisperJAV",
            "https://github.com/meizhong986/WhisperJAV/releases/tag/v1.8.0",
            "https://github.com/meizhong986/WhisperJAV/issues",
        ]

        for url in urls:
            with patch('webbrowser.open', return_value=True) as mock_open:
                result = mock_api.open_url(url)
                assert result["success"] is True
                mock_open.assert_called_with(url)


# =============================================================================
# start_update() Tests
# =============================================================================

class TestStartUpdate:
    """Tests for start_update() API method."""

    def test_start_update_spawns_wrapper(self, mock_api, monkeypatch):
        """Test that start_update spawns the update wrapper process."""
        mock_popen = MagicMock()

        with patch('subprocess.Popen', mock_popen):
            with patch.object(Path, 'exists', return_value=True):
                result = mock_api.start_update(wheel_only=False)

        assert result["success"] is True
        assert result["should_exit"] is True
        assert "Update process started" in result["message"]

        # Verify Popen was called
        mock_popen.assert_called_once()
        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        # Should include update_wrapper module
        assert "-m" in cmd
        assert "whisperjav.update_wrapper" in cmd

    def test_start_update_with_wheel_only(self, mock_api):
        """Test start_update with wheel_only=True."""
        mock_popen = MagicMock()

        with patch('subprocess.Popen', mock_popen):
            with patch.object(Path, 'exists', return_value=True):
                result = mock_api.start_update(wheel_only=True)

        assert result["success"] is True

        # Verify --wheel-only flag is present
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert "--wheel-only" in cmd

    def test_start_update_includes_pid(self, mock_api):
        """Test that start_update passes current PID to wrapper."""
        mock_popen = MagicMock()
        current_pid = os.getpid()

        with patch('subprocess.Popen', mock_popen):
            with patch.object(Path, 'exists', return_value=True):
                mock_api.start_update()

        call_args = mock_popen.call_args
        cmd = call_args[0][0]

        # Should include --pid argument
        assert "--pid" in cmd
        pid_idx = cmd.index("--pid")
        assert cmd[pid_idx + 1] == str(current_pid)

    def test_start_update_error_handling(self, mock_api):
        """Test error handling when spawn fails."""
        with patch('subprocess.Popen', side_effect=Exception("Spawn failed")):
            with patch.object(Path, 'exists', return_value=True):
                result = mock_api.start_update()

        assert result["success"] is False
        assert "Spawn failed" in result["error"]
        assert result["should_exit"] is False

    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
    def test_start_update_windows_detached(self, mock_api):
        """Test that Windows uses DETACHED_PROCESS flag."""
        mock_popen = MagicMock()

        with patch('subprocess.Popen', mock_popen):
            with patch.object(Path, 'exists', return_value=True):
                mock_api.start_update()

        call_kwargs = mock_popen.call_args[1]

        # Should use Windows-specific creation flags
        assert "creationflags" in call_kwargs
        expected_flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        assert call_kwargs["creationflags"] == expected_flags


# =============================================================================
# Integration Tests
# =============================================================================

class TestUpdateCheckIntegration:
    """Integration tests for the complete update check flow."""

    def test_full_update_check_flow(self, mock_api, sample_update_result, tmp_path, monkeypatch):
        """Test complete flow: check -> show update -> user action."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        # Step 1: Check for updates
        with patch('whisperjav.version_checker.check_for_updates', return_value=sample_update_result):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                check_result = mock_api.check_for_updates(force=True)

        assert check_result["update_available"] is True
        assert check_result["notification_level"] == "minor"

        # Step 2: User decides to dismiss
        dismiss_result = mock_api.dismiss_update_notification(check_result["latest_version"])
        assert dismiss_result["success"] is True

        # Step 3: Verify dismissal was recorded
        dismissed = mock_api.get_dismissed_update()
        assert dismissed["dismissed"]["version"] == "1.8.0"

    def test_major_update_triggers_download(self, mock_api, tmp_path, monkeypatch):
        """Test that major updates trigger download page instead of update."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        major_update = MockUpdateCheckResult(
            update_available=True,
            current_version="1.7.5",
            latest_version="2.0.0",
            version_info=MockVersionInfo(
                version="2.0.0",
                release_url="https://github.com/meizhong986/whisperjav/releases/tag/v2.0.0",
                release_notes="## Breaking Changes\n- New architecture",
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=False
            ),
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=major_update):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='major'):
                result = mock_api.check_for_updates(force=True)

        # For major updates, frontend should use open_url() instead of start_update()
        assert result["notification_level"] == "major"

        # Verify open_url works for this case
        with patch('webbrowser.open', return_value=True) as mock_open:
            url_result = mock_api.open_url("https://github.com/meizhong986/WhisperJAV/releases")

        assert url_result["success"] is True

    def test_patch_update_triggers_inline_update(self, mock_api, tmp_path, monkeypatch):
        """Test that patch updates trigger inline update."""
        monkeypatch.setenv('LOCALAPPDATA', str(tmp_path))

        patch_update = MockUpdateCheckResult(
            update_available=True,
            current_version="1.7.5",
            latest_version="1.7.6",
            version_info=MockVersionInfo(
                version="1.7.6",
                release_url="https://github.com/meizhong986/whisperjav/releases/tag/v1.7.6",
                release_notes="## Bug Fixes\n- Fixed MPS fallback",
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=False
            ),
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=patch_update):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='patch'):
                result = mock_api.check_for_updates(force=True)

        # For patch updates, frontend should use start_update()
        assert result["notification_level"] == "patch"

        # Verify start_update works
        mock_popen = MagicMock()
        with patch('subprocess.Popen', mock_popen):
            with patch.object(Path, 'exists', return_value=True):
                update_result = mock_api.start_update(wheel_only=False)

        assert update_result["success"] is True
        assert update_result["should_exit"] is True


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_network_timeout_handling(self, mock_api):
        """Test handling of network timeout."""
        timeout_result = MockUpdateCheckResult(
            update_available=False,
            current_version="1.7.5",
            latest_version=None,
            version_info=None,
            from_cache=False,
            error="Connection timed out after 10 seconds"
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=timeout_result):
            result = mock_api.check_for_updates(force=True)

        assert result["update_available"] is False
        assert "timed out" in result["error"]

    def test_invalid_version_format(self, mock_api):
        """Test handling of invalid version format."""
        invalid_result = MockUpdateCheckResult(
            update_available=False,
            current_version="unknown",
            latest_version=None,
            version_info=None,
            from_cache=False,
            error="Could not parse version"
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=invalid_result):
            result = mock_api.check_for_updates(force=True)

        assert result["success"] is True  # API call succeeded
        assert "error" in result

    def test_prerelease_handling(self, mock_api):
        """Test that prerelease versions are identified."""
        prerelease = MockUpdateCheckResult(
            update_available=True,
            current_version="1.7.5",
            latest_version="1.8.0-rc1",
            version_info=MockVersionInfo(
                version="1.8.0-rc1",
                release_url="https://example.com",
                release_notes="Release candidate",
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=True
            ),
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=prerelease):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='minor'):
                result = mock_api.check_for_updates(force=True)

        assert result["is_prerelease"] is True

    def test_empty_release_notes(self, mock_api):
        """Test handling of empty release notes."""
        empty_notes = MockUpdateCheckResult(
            update_available=True,
            current_version="1.7.5",
            latest_version="1.7.6",
            version_info=MockVersionInfo(
                version="1.7.6",
                release_url="https://example.com",
                release_notes="",
                published_at="2025-01-15T10:00:00Z",
                is_prerelease=False
            ),
            from_cache=False,
            error=None
        )

        with patch('whisperjav.version_checker.check_for_updates', return_value=empty_notes):
            with patch('whisperjav.version_checker.get_update_notification_level', return_value='patch'):
                result = mock_api.check_for_updates(force=True)

        assert result["release_notes"] == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
