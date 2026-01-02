#!/usr/bin/env python3
"""
Unit tests for whisperjav.version_checker module.

Tests version parsing, comparison, update checking, and notification logic.
Uses unittest.mock to mock urllib.request calls.

Run with: pytest tests/test_upgrade_version_checker.py -v
"""

import json
import os
import pytest
import tempfile
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock


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
def mock_version_checker(temp_cache_dir, monkeypatch):
    """
    Fixture that imports version_checker with mocked paths.

    This re-imports the module with test-specific settings.
    """
    # Set environment variables before import
    monkeypatch.setenv('WHISPERJAV_CACHE_HOURS', '6')

    # Clear any cached imports
    import sys
    if 'whisperjav.version_checker' in sys.modules:
        del sys.modules['whisperjav.version_checker']

    # Patch the cache directory
    import whisperjav.version_checker as vc
    monkeypatch.setattr(vc, 'CACHE_DIR', temp_cache_dir)
    monkeypatch.setattr(vc, 'VERSION_CACHE_FILE', temp_cache_dir / 'version_check.json')

    return vc


@pytest.fixture
def sample_github_release():
    """Sample GitHub API release response."""
    return {
        "tag_name": "v1.8.0",
        "html_url": "https://github.com/meizhong986/whisperjav/releases/tag/v1.8.0",
        "body": "## What's New\n- Feature A\n- Feature B",
        "published_at": "2025-01-15T10:00:00Z",
        "prerelease": False,
        "assets": [
            {
                "name": "WhisperJAV-1.8.0-Windows-x86_64.exe",
                "browser_download_url": "https://github.com/meizhong986/whisperjav/releases/download/v1.8.0/WhisperJAV-1.8.0-Windows-x86_64.exe"
            }
        ]
    }


@pytest.fixture
def sample_security_release():
    """Sample GitHub API release response with security fix."""
    return {
        "tag_name": "v1.7.5-security",
        "html_url": "https://github.com/meizhong986/whisperjav/releases/tag/v1.7.5-security",
        "body": "## CRITICAL SECURITY UPDATE\n- Fixed vulnerability in input handling",
        "published_at": "2025-01-15T10:00:00Z",
        "prerelease": False,
        "assets": []
    }


# =============================================================================
# Version Parsing Tests
# =============================================================================

class TestParseVersion:
    """Tests for parse_version() function."""

    def test_simple_version(self, mock_version_checker):
        """Test parsing simple X.Y.Z version."""
        vc = mock_version_checker
        assert vc.parse_version("1.7.4") == (1, 7, 4, "")

    def test_version_with_v_prefix(self, mock_version_checker):
        """Test parsing version with 'v' prefix."""
        vc = mock_version_checker
        assert vc.parse_version("v1.7.4") == (1, 7, 4, "")

    def test_version_with_suffix(self, mock_version_checker):
        """Test parsing version with suffix (e.g., .post4)."""
        vc = mock_version_checker
        assert vc.parse_version("1.7.3.post4") == (1, 7, 3, ".post4")

    def test_version_with_rc_suffix(self, mock_version_checker):
        """Test parsing version with release candidate suffix."""
        vc = mock_version_checker
        assert vc.parse_version("2.0.0-rc1") == (2, 0, 0, "-rc1")

    def test_version_with_security_suffix(self, mock_version_checker):
        """Test parsing version with security suffix."""
        vc = mock_version_checker
        assert vc.parse_version("1.7.5-security") == (1, 7, 5, "-security")

    def test_invalid_version(self, mock_version_checker):
        """Test parsing invalid version returns zeros."""
        vc = mock_version_checker
        assert vc.parse_version("invalid") == (0, 0, 0, "")
        assert vc.parse_version("") == (0, 0, 0, "")

    def test_major_only_version(self, mock_version_checker):
        """Test parsing version without all components."""
        vc = mock_version_checker
        # This doesn't match X.Y.Z pattern, so returns zeros
        assert vc.parse_version("1") == (0, 0, 0, "")


# =============================================================================
# Version Comparison Tests
# =============================================================================

class TestCompareVersions:
    """Tests for compare_versions() function."""

    def test_equal_versions(self, mock_version_checker):
        """Test comparing equal versions."""
        vc = mock_version_checker
        assert vc.compare_versions("1.7.4", "1.7.4") == 0
        assert vc.compare_versions("v1.7.4", "1.7.4") == 0

    def test_major_version_difference(self, mock_version_checker):
        """Test comparing different major versions."""
        vc = mock_version_checker
        assert vc.compare_versions("1.7.4", "2.0.0") == -1
        assert vc.compare_versions("2.0.0", "1.7.4") == 1

    def test_minor_version_difference(self, mock_version_checker):
        """Test comparing different minor versions."""
        vc = mock_version_checker
        assert vc.compare_versions("1.7.4", "1.8.0") == -1
        assert vc.compare_versions("1.8.0", "1.7.4") == 1

    def test_patch_version_difference(self, mock_version_checker):
        """Test comparing different patch versions."""
        vc = mock_version_checker
        assert vc.compare_versions("1.7.4", "1.7.5") == -1
        assert vc.compare_versions("1.7.5", "1.7.4") == 1

    def test_release_vs_prerelease(self, mock_version_checker):
        """Test that release version > prerelease."""
        vc = mock_version_checker
        # Release (no suffix) should be greater than prerelease
        assert vc.compare_versions("1.7.4", "1.7.4-rc1") == 1
        assert vc.compare_versions("1.7.4-rc1", "1.7.4") == -1

    def test_suffix_comparison(self, mock_version_checker):
        """Test comparing versions with different suffixes."""
        vc = mock_version_checker
        # Alphabetical suffix comparison when both have suffixes
        assert vc.compare_versions("1.7.4-alpha", "1.7.4-beta") == -1
        assert vc.compare_versions("1.7.4-rc1", "1.7.4-rc2") == -1

    def test_post_suffix(self, mock_version_checker):
        """Test versions with .post suffix."""
        vc = mock_version_checker
        # 1.7.4 (no suffix) > 1.7.4.post1 (has suffix)
        assert vc.compare_versions("1.7.4", "1.7.4.post1") == 1


# =============================================================================
# Notification Level Tests
# =============================================================================

class TestGetUpdateNotificationLevel:
    """Tests for get_update_notification_level() function."""

    def test_no_update_available(self, mock_version_checker):
        """Test when no update is available."""
        vc = mock_version_checker
        result = vc.UpdateCheckResult(
            update_available=False,
            current_version="1.7.4"
        )
        assert vc.get_update_notification_level(result) == 'none'

    def test_major_version_update(self, mock_version_checker):
        """Test major version update notification level."""
        vc = mock_version_checker
        result = vc.UpdateCheckResult(
            update_available=True,
            current_version="1.7.4",
            latest_version="2.0.0",
            version_info=vc.VersionInfo(
                version="2.0.0",
                release_url="https://example.com",
                release_notes="New major version",
                published_at="2025-01-15",
                is_prerelease=False
            )
        )
        assert vc.get_update_notification_level(result) == 'major'

    def test_minor_version_update(self, mock_version_checker):
        """Test minor version update notification level."""
        vc = mock_version_checker
        result = vc.UpdateCheckResult(
            update_available=True,
            current_version="1.7.4",
            latest_version="1.8.0",
            version_info=vc.VersionInfo(
                version="1.8.0",
                release_url="https://example.com",
                release_notes="New minor version",
                published_at="2025-01-15",
                is_prerelease=False
            )
        )
        assert vc.get_update_notification_level(result) == 'minor'

    def test_patch_version_update(self, mock_version_checker):
        """Test patch version update notification level."""
        vc = mock_version_checker
        result = vc.UpdateCheckResult(
            update_available=True,
            current_version="1.7.4",
            latest_version="1.7.5",
            version_info=vc.VersionInfo(
                version="1.7.5",
                release_url="https://example.com",
                release_notes="Bug fixes",
                published_at="2025-01-15",
                is_prerelease=False
            )
        )
        assert vc.get_update_notification_level(result) == 'patch'

    def test_security_update(self, mock_version_checker):
        """Test security update triggers critical level."""
        vc = mock_version_checker
        result = vc.UpdateCheckResult(
            update_available=True,
            current_version="1.7.4",
            latest_version="1.7.5",
            version_info=vc.VersionInfo(
                version="1.7.5",
                release_url="https://example.com",
                release_notes="SECURITY: Fixed vulnerability in parser",
                published_at="2025-01-15",
                is_prerelease=False
            )
        )
        assert vc.get_update_notification_level(result) == 'critical'

    def test_critical_keyword_detection(self, mock_version_checker):
        """Test various critical keywords in release notes."""
        vc = mock_version_checker

        critical_keywords = ['security', 'critical', 'urgent', 'vulnerability']

        for keyword in critical_keywords:
            result = vc.UpdateCheckResult(
                update_available=True,
                current_version="1.7.4",
                latest_version="1.7.5",
                version_info=vc.VersionInfo(
                    version="1.7.5",
                    release_url="https://example.com",
                    release_notes=f"This is a {keyword} fix",
                    published_at="2025-01-15",
                    is_prerelease=False
                )
            )
            assert vc.get_update_notification_level(result) == 'critical', f"Failed for keyword: {keyword}"


# =============================================================================
# Should Show Notification Tests
# =============================================================================

class TestShouldShowNotification:
    """Tests for should_show_notification() function."""

    def test_none_level_never_shows(self, mock_version_checker):
        """Test that 'none' level never shows notification."""
        vc = mock_version_checker
        assert vc.should_show_notification('none') is False
        assert vc.should_show_notification('none', None) is False
        assert vc.should_show_notification('none', datetime.now().isoformat()) is False

    def test_critical_always_shows(self, mock_version_checker):
        """Test that 'critical' level always shows."""
        vc = mock_version_checker
        # Even if dismissed recently, critical should show
        assert vc.should_show_notification('critical') is True
        assert vc.should_show_notification('critical', datetime.now().isoformat()) is True

    def test_major_always_shows(self, mock_version_checker):
        """Test that 'major' level always shows."""
        vc = mock_version_checker
        assert vc.should_show_notification('major') is True
        assert vc.should_show_notification('major', datetime.now().isoformat()) is True

    def test_minor_respects_monthly_interval(self, mock_version_checker):
        """Test that 'minor' level respects 30-day interval."""
        vc = mock_version_checker

        # Never dismissed - show
        assert vc.should_show_notification('minor') is True

        # Dismissed yesterday - don't show
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        assert vc.should_show_notification('minor', yesterday) is False

        # Dismissed 31 days ago - show
        old_date = (datetime.now() - timedelta(days=31)).isoformat()
        assert vc.should_show_notification('minor', old_date) is True

    def test_patch_respects_weekly_interval(self, mock_version_checker):
        """Test that 'patch' level respects 7-day interval."""
        vc = mock_version_checker

        # Never dismissed - show
        assert vc.should_show_notification('patch') is True

        # Dismissed yesterday - don't show
        yesterday = (datetime.now() - timedelta(days=1)).isoformat()
        assert vc.should_show_notification('patch', yesterday) is False

        # Dismissed 8 days ago - show
        old_date = (datetime.now() - timedelta(days=8)).isoformat()
        assert vc.should_show_notification('patch', old_date) is True

    def test_invalid_timestamp_shows(self, mock_version_checker):
        """Test that invalid timestamp causes notification to show."""
        vc = mock_version_checker
        assert vc.should_show_notification('minor', 'invalid-date') is True


# =============================================================================
# Helper for mocking urllib.request
# =============================================================================

def create_mock_response(data, status_code=200):
    """Create a mock urllib response object."""
    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps(data).encode('utf-8')
    mock_response.status = status_code
    mock_response.__enter__ = MagicMock(return_value=mock_response)
    mock_response.__exit__ = MagicMock(return_value=False)
    return mock_response


# =============================================================================
# HTTP Mocking Tests (using unittest.mock for urllib)
# =============================================================================

class TestCheckForUpdates:
    """Tests for check_for_updates() with mocked HTTP."""

    def test_update_available(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test check_for_updates when newer version exists."""
        vc = mock_version_checker

        # Mock current version to be older
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Mock urllib.request.urlopen
        mock_response = create_mock_response(sample_github_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is True
        assert result.current_version == '1.7.4'
        assert result.latest_version == '1.8.0'
        assert result.version_info is not None
        assert result.version_info.version == '1.8.0'
        assert result.error is None
        assert result.from_cache is False

    def test_no_update_available(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test check_for_updates when current version is latest."""
        vc = mock_version_checker

        # Mock current version to match latest
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.8.0')

        mock_response = create_mock_response(sample_github_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is False
        assert result.current_version == '1.8.0'
        assert result.latest_version == '1.8.0'

    def test_network_error(self, mock_version_checker, monkeypatch):
        """Test check_for_updates handles network errors gracefully."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Mock network error
        with patch.object(urllib.request, 'urlopen', side_effect=urllib.error.URLError("Connection refused")):
            result = vc.check_for_updates(force=True)

        assert result.update_available is False
        assert result.error is not None
        assert "Could not check" in result.error

    def test_rate_limit_error(self, mock_version_checker, monkeypatch):
        """Test check_for_updates handles rate limit (403) gracefully."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Mock 403 error
        mock_fp = MagicMock()
        mock_fp.read.return_value = b'{"message": "rate limit"}'
        http_error = urllib.error.HTTPError(
            url=vc.GITHUB_API_URL,
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=mock_fp
        )

        with patch.object(urllib.request, 'urlopen', side_effect=http_error):
            result = vc.check_for_updates(force=True)

        assert result.update_available is False
        assert result.error is not None

    def test_no_releases_404(self, mock_version_checker, monkeypatch):
        """Test check_for_updates handles 404 (no releases) gracefully."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Mock 404 error
        mock_fp = MagicMock()
        mock_fp.read.return_value = b'{"message": "Not Found"}'
        http_error = urllib.error.HTTPError(
            url=vc.GITHUB_API_URL,
            code=404,
            msg="Not Found",
            hdrs={},
            fp=mock_fp
        )

        with patch.object(urllib.request, 'urlopen', side_effect=http_error):
            result = vc.check_for_updates(force=True)

        assert result.update_available is False
        assert result.error is not None

    def test_cache_is_saved(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test that successful check saves to cache."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        mock_response = create_mock_response(sample_github_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        # Check cache file was created
        assert vc.VERSION_CACHE_FILE.exists()

        with open(vc.VERSION_CACHE_FILE, 'r') as f:
            cache_data = json.load(f)

        assert cache_data['latest_version'] == '1.8.0'
        assert 'checked_at' in cache_data

    def test_cache_is_used(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test that cached result is used when not expired."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Create cache file
        cache_data = {
            'latest_version': '1.7.5',
            'checked_at': datetime.now().isoformat(),
            'version_info': {
                'version': '1.7.5',
                'release_url': 'https://example.com',
                'release_notes': 'Cached notes',
                'published_at': '2025-01-15',
                'is_prerelease': False,
                'download_url': None
            }
        }

        vc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(vc.VERSION_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)

        # No HTTP mock needed - should use cache
        result = vc.check_for_updates(force=False)

        assert result.from_cache is True
        assert result.latest_version == '1.7.5'

    def test_expired_cache_triggers_refresh(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test that expired cache triggers fresh check."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Create expired cache file (7 hours old, beyond 6 hour default)
        expired_time = datetime.now() - timedelta(hours=7)
        cache_data = {
            'latest_version': '1.7.5',
            'checked_at': expired_time.isoformat(),
            'version_info': {
                'version': '1.7.5',
                'release_url': 'https://example.com',
                'release_notes': 'Old cached notes',
                'published_at': '2025-01-14',
                'is_prerelease': False,
                'download_url': None
            }
        }

        vc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(vc.VERSION_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)

        # Mock fresh API response
        mock_response = create_mock_response(sample_github_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=False)

        assert result.from_cache is False
        assert result.latest_version == '1.8.0'  # Fresh data, not cached

    def test_force_bypasses_cache(self, mock_version_checker, sample_github_release, monkeypatch):
        """Test that force=True bypasses cache."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Create fresh cache file
        cache_data = {
            'latest_version': '1.7.5',
            'checked_at': datetime.now().isoformat(),
            'version_info': {
                'version': '1.7.5',
                'release_url': 'https://example.com',
                'release_notes': 'Cached notes',
                'published_at': '2025-01-15',
                'is_prerelease': False,
                'download_url': None
            }
        }

        vc.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(vc.VERSION_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f)

        # Mock fresh API response
        mock_response = create_mock_response(sample_github_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.from_cache is False
        assert result.latest_version == '1.8.0'  # Fresh data

    def test_windows_installer_asset_detection(self, mock_version_checker, monkeypatch):
        """Test that Windows installer asset URL is extracted."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        release_with_assets = {
            "tag_name": "v1.8.0",
            "html_url": "https://github.com/meizhong986/whisperjav/releases/tag/v1.8.0",
            "body": "New version",
            "published_at": "2025-01-15T10:00:00Z",
            "prerelease": False,
            "assets": [
                {
                    "name": "WhisperJAV-1.8.0-Windows-x86_64.exe",
                    "browser_download_url": "https://example.com/download.exe"
                },
                {
                    "name": "checksums.txt",
                    "browser_download_url": "https://example.com/checksums.txt"
                }
            ]
        }

        mock_response = create_mock_response(release_with_assets)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.version_info is not None
        assert result.version_info.download_url == "https://example.com/download.exe"


# =============================================================================
# Environment Variable Configuration Tests
# =============================================================================

class TestEnvironmentConfiguration:
    """Tests for environment variable configuration."""

    def test_custom_api_url(self, monkeypatch, temp_cache_dir):
        """Test that WHISPERJAV_UPDATE_API_URL overrides default."""
        custom_url = "https://api.github.com/repos/test/stub/releases/latest"
        monkeypatch.setenv('WHISPERJAV_UPDATE_API_URL', custom_url)

        # Re-import to pick up new env var
        import sys
        if 'whisperjav.version_checker' in sys.modules:
            del sys.modules['whisperjav.version_checker']

        import whisperjav.version_checker as vc

        assert vc.GITHUB_API_URL == custom_url

    def test_custom_cache_hours(self, monkeypatch, temp_cache_dir):
        """Test that WHISPERJAV_CACHE_HOURS overrides default."""
        monkeypatch.setenv('WHISPERJAV_CACHE_HOURS', '0')

        import sys
        if 'whisperjav.version_checker' in sys.modules:
            del sys.modules['whisperjav.version_checker']

        import whisperjav.version_checker as vc

        assert vc.CACHE_DURATION_HOURS == 0

    def test_custom_releases_url(self, monkeypatch, temp_cache_dir):
        """Test that WHISPERJAV_RELEASES_URL overrides default."""
        custom_url = "https://github.com/test/stub/releases"
        monkeypatch.setenv('WHISPERJAV_RELEASES_URL', custom_url)

        import sys
        if 'whisperjav.version_checker' in sys.modules:
            del sys.modules['whisperjav.version_checker']

        import whisperjav.version_checker as vc

        assert vc.GITHUB_RELEASES_URL == custom_url


# =============================================================================
# Integration Tests with Test Stub Repository
# =============================================================================

class TestWithTestStubRepo:
    """
    Integration tests using mocked responses that simulate the test stub repo.

    These tests verify the full flow works with realistic data.
    """

    def test_stub_v1_7_4_to_v2_0_0_major_update(self, mock_version_checker, monkeypatch):
        """Test upgrade from 1.7.4 to 2.0.0 (major version)."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        # Simulate test stub's v2.0.0 release
        stub_release = {
            "tag_name": "v2.0.0",
            "html_url": "https://github.com/meizhong986/whisperjav-test/releases/tag/v2.0.0",
            "body": "# WhisperJAV v2.0.0\n\n**MAJOR VERSION RELEASE**\n\n## Breaking Changes\n- New config format",
            "published_at": "2025-01-15T10:00:00Z",
            "prerelease": False,
            "assets": []
        }

        mock_response = create_mock_response(stub_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is True
        assert result.latest_version == '2.0.0'
        assert vc.get_update_notification_level(result) == 'major'

    def test_stub_v1_7_4_to_v1_8_0_minor_update(self, mock_version_checker, monkeypatch):
        """Test upgrade from 1.7.4 to 1.8.0 (minor version)."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        stub_release = {
            "tag_name": "v1.8.0",
            "html_url": "https://github.com/meizhong986/whisperjav-test/releases/tag/v1.8.0",
            "body": "# WhisperJAV v1.8.0\n\nMinor version update with new features.",
            "published_at": "2025-01-15T10:00:00Z",
            "prerelease": False,
            "assets": []
        }

        mock_response = create_mock_response(stub_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is True
        assert result.latest_version == '1.8.0'
        assert vc.get_update_notification_level(result) == 'minor'

    def test_stub_v1_7_4_to_v1_7_5_patch_update(self, mock_version_checker, monkeypatch):
        """Test upgrade from 1.7.4 to 1.7.5 (patch version)."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        stub_release = {
            "tag_name": "v1.7.5",
            "html_url": "https://github.com/meizhong986/whisperjav-test/releases/tag/v1.7.5",
            "body": "# WhisperJAV v1.7.5\n\nPatch release with bug fixes.",
            "published_at": "2025-01-15T10:00:00Z",
            "prerelease": False,
            "assets": []
        }

        mock_response = create_mock_response(stub_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is True
        assert result.latest_version == '1.7.5'
        assert vc.get_update_notification_level(result) == 'patch'

    def test_stub_security_release_critical(self, mock_version_checker, monkeypatch):
        """Test security release triggers critical notification."""
        vc = mock_version_checker
        monkeypatch.setattr(vc, 'CURRENT_VERSION', '1.7.4')

        stub_release = {
            "tag_name": "v1.7.5-security",
            "html_url": "https://github.com/meizhong986/whisperjav-test/releases/tag/v1.7.5-security",
            "body": "# CRITICAL SECURITY UPDATE\n\nFixed vulnerability in parser.",
            "published_at": "2025-01-15T10:00:00Z",
            "prerelease": False,
            "assets": []
        }

        mock_response = create_mock_response(stub_release)

        with patch.object(urllib.request, 'urlopen', return_value=mock_response):
            result = vc.check_for_updates(force=True)

        assert result.update_available is True
        assert vc.get_update_notification_level(result) == 'critical'


# =============================================================================
# Run directly for debugging
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
