#!/usr/bin/env python3
"""
WhisperJAV Version Checker

Checks for new versions on GitHub and provides update notifications.
Uses GitHub API with caching and retry logic.
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from datetime import datetime, timedelta

# Import version - handle both installed and dev scenarios
try:
    from whisperjav.__version__ import __version__ as CURRENT_VERSION
except ImportError:
    CURRENT_VERSION = "0.0.0"

# =============================================================================
# Configurable Endpoints (for testing)
# =============================================================================
# These can be overridden via environment variables to point at test stubs:
#
#   WHISPERJAV_UPDATE_API_URL - GitHub API endpoint for release info
#   WHISPERJAV_RELEASES_URL   - Human-readable releases page URL
#
# Example (pointing at test stub repo):
#   set WHISPERJAV_UPDATE_API_URL=https://api.github.com/repos/meizhong986/whisperjav-test/releases/latest
#   set WHISPERJAV_RELEASES_URL=https://github.com/meizhong986/whisperjav-test/releases
#
# Example (local mock server):
#   set WHISPERJAV_UPDATE_API_URL=http://localhost:8000/api/releases/latest
#   set WHISPERJAV_RELEASES_URL=http://localhost:8000/releases
# =============================================================================

# GitHub API endpoint
GITHUB_API_URL = os.environ.get(
    'WHISPERJAV_UPDATE_API_URL',
    'https://api.github.com/repos/meizhong986/whisperjav/releases/latest'
)
GITHUB_RELEASES_URL = os.environ.get(
    'WHISPERJAV_RELEASES_URL',
    'https://github.com/meizhong986/WhisperJAV/releases'
)

# Cache settings
CACHE_DIR = Path(os.environ.get('LOCALAPPDATA', Path.home())) / '.whisperjav_cache'
VERSION_CACHE_FILE = CACHE_DIR / 'version_check.json'
CACHE_DURATION_HOURS = int(os.environ.get('WHISPERJAV_CACHE_HOURS', '6'))


@dataclass
class VersionInfo:
    """Information about an available version."""
    version: str
    release_url: str
    release_notes: str
    published_at: str
    is_prerelease: bool
    download_url: Optional[str] = None


@dataclass
class UpdateCheckResult:
    """Result of an update check."""
    update_available: bool
    current_version: str
    latest_version: Optional[str] = None
    version_info: Optional[VersionInfo] = None
    error: Optional[str] = None
    from_cache: bool = False


def parse_version(version_str: str) -> Tuple[int, int, int, str]:
    """
    Parse version string into components.

    Args:
        version_str: Version string like "1.7.3", "1.7.3.post4", "v1.7.4"

    Returns:
        Tuple of (major, minor, patch, suffix)
    """
    import re

    # Remove leading 'v' if present
    version_str = version_str.lstrip('v')

    match = re.match(r'^(\d+)\.(\d+)\.(\d+)(.*)$', version_str)
    if match:
        return (int(match.group(1)), int(match.group(2)),
                int(match.group(3)), match.group(4) or "")
    return (0, 0, 0, "")


def compare_versions(v1: str, v2: str) -> int:
    """
    Compare two version strings.

    Returns:
        -1 if v1 < v2
         0 if v1 == v2
         1 if v1 > v2
    """
    p1 = parse_version(v1)
    p2 = parse_version(v2)

    # Compare major, minor, patch
    for i in range(3):
        if p1[i] < p2[i]:
            return -1
        if p1[i] > p2[i]:
            return 1

    # Compare suffix (empty suffix > any suffix for release versions)
    s1, s2 = p1[3], p2[3]
    if s1 == s2:
        return 0
    if not s1:  # v1 is release, v2 has suffix
        return 1
    if not s2:  # v2 is release, v1 has suffix
        return -1
    # Both have suffixes - alphabetical comparison
    return -1 if s1 < s2 else 1


def _load_cache() -> Optional[Dict]:
    """Load cached version check result."""
    try:
        if VERSION_CACHE_FILE.exists():
            with open(VERSION_CACHE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Check if cache is still valid
                cached_time = datetime.fromisoformat(data.get('checked_at', ''))
                if datetime.now() - cached_time < timedelta(hours=CACHE_DURATION_HOURS):
                    return data
    except Exception:
        pass
    return None


def _save_cache(data: Dict):
    """Save version check result to cache."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        data['checked_at'] = datetime.now().isoformat()
        with open(VERSION_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _fetch_latest_release(timeout: int = 10, retries: int = 2) -> Optional[Dict]:
    """
    Fetch latest release info from GitHub API.

    Args:
        timeout: Request timeout in seconds
        retries: Number of retry attempts

    Returns:
        Release data dict or None on failure
    """
    last_error = None

    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(
                GITHUB_API_URL,
                headers={
                    'Accept': 'application/vnd.github.v3+json',
                    'User-Agent': f'WhisperJAV/{CURRENT_VERSION}'
                }
            )

            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode('utf-8'))

        except urllib.error.HTTPError as e:
            if e.code == 403:
                # Rate limited - don't retry
                last_error = "GitHub API rate limit exceeded"
                break
            elif e.code == 404:
                # No releases found
                last_error = "No releases found"
                break
            last_error = f"HTTP {e.code}"

        except urllib.error.URLError as e:
            last_error = f"Network error: {e.reason}"

        except Exception as e:
            last_error = str(e)

        # Wait before retry
        if attempt < retries:
            time.sleep(1)

    return None


def check_for_updates(force: bool = False) -> UpdateCheckResult:
    """
    Check if a newer version of WhisperJAV is available.

    Args:
        force: If True, bypass cache and check immediately

    Returns:
        UpdateCheckResult with update information
    """
    current = CURRENT_VERSION

    # Check cache first (unless forced)
    if not force:
        cached = _load_cache()
        if cached:
            latest = cached.get('latest_version')
            if latest:
                update_available = compare_versions(current, latest) < 0

                version_info = None
                if cached.get('version_info'):
                    vi = cached['version_info']
                    version_info = VersionInfo(
                        version=vi.get('version', latest),
                        release_url=vi.get('release_url', GITHUB_RELEASES_URL),
                        release_notes=vi.get('release_notes', ''),
                        published_at=vi.get('published_at', ''),
                        is_prerelease=vi.get('is_prerelease', False),
                        download_url=vi.get('download_url')
                    )

                return UpdateCheckResult(
                    update_available=update_available,
                    current_version=current,
                    latest_version=latest,
                    version_info=version_info,
                    from_cache=True
                )

    # Fetch from GitHub
    release_data = _fetch_latest_release()

    if not release_data:
        return UpdateCheckResult(
            update_available=False,
            current_version=current,
            error="Could not check for updates"
        )

    # Parse release data
    latest = release_data.get('tag_name', '').lstrip('v')

    if not latest:
        return UpdateCheckResult(
            update_available=False,
            current_version=current,
            error="Invalid release data"
        )

    version_info = VersionInfo(
        version=latest,
        release_url=release_data.get('html_url', GITHUB_RELEASES_URL),
        release_notes=release_data.get('body', ''),
        published_at=release_data.get('published_at', ''),
        is_prerelease=release_data.get('prerelease', False),
        download_url=None
    )

    # Find Windows installer asset
    for asset in release_data.get('assets', []):
        name = asset.get('name', '')
        if 'Windows' in name and name.endswith('.exe'):
            version_info.download_url = asset.get('browser_download_url')
            break

    # Save to cache
    cache_data = {
        'latest_version': latest,
        'version_info': {
            'version': version_info.version,
            'release_url': version_info.release_url,
            'release_notes': version_info.release_notes[:1000],  # Truncate
            'published_at': version_info.published_at,
            'is_prerelease': version_info.is_prerelease,
            'download_url': version_info.download_url
        }
    }
    _save_cache(cache_data)

    # Compare versions
    update_available = compare_versions(current, latest) < 0

    return UpdateCheckResult(
        update_available=update_available,
        current_version=current,
        latest_version=latest,
        version_info=version_info,
        from_cache=False
    )


def get_update_notification_level(result: UpdateCheckResult) -> str:
    """
    Determine the notification level for an update.

    Returns:
        'critical' - Security/critical update (always show)
        'major' - Major version update (show every launch)
        'minor' - Minor version update (show monthly)
        'patch' - Patch update (subtle, dismissable)
        'none' - No update available
    """
    if not result.update_available or not result.version_info:
        return 'none'

    current = parse_version(result.current_version)
    latest = parse_version(result.version_info.version)

    # Check for critical keywords in release notes
    notes = (result.version_info.release_notes or '').lower()
    if any(word in notes for word in ['security', 'critical', 'urgent', 'vulnerability']):
        return 'critical'

    # Compare version components
    if latest[0] > current[0]:
        return 'major'  # Major version bump (1.x -> 2.x)

    if latest[1] > current[1]:
        return 'minor'  # Minor version bump (1.7 -> 1.8)

    return 'patch'  # Patch version bump (1.7.3 -> 1.7.4)


def should_show_notification(level: str, last_dismissed: Optional[str] = None) -> bool:
    """
    Determine if notification should be shown based on level and dismiss history.

    Args:
        level: Notification level from get_update_notification_level
        last_dismissed: ISO timestamp of last dismissal

    Returns:
        True if notification should be shown
    """
    if level == 'none':
        return False

    if level == 'critical':
        return True  # Always show critical

    if not last_dismissed:
        return True  # Never dismissed, show it

    try:
        dismissed_time = datetime.fromisoformat(last_dismissed)
        now = datetime.now()

        if level == 'major':
            return True  # Show every launch

        if level == 'minor':
            # Show monthly
            return (now - dismissed_time).days >= 30

        if level == 'patch':
            # Show weekly
            return (now - dismissed_time).days >= 7

    except Exception:
        return True

    return True


# CLI interface
if __name__ == "__main__":
    print(f"Current version: {CURRENT_VERSION}")
    print("Checking for updates...")

    result = check_for_updates(force=True)

    if result.error:
        print(f"Error: {result.error}")
    elif result.update_available:
        print(f"Update available: {result.latest_version}")
        print(f"Release URL: {result.version_info.release_url}")
        if result.version_info.download_url:
            print(f"Download: {result.version_info.download_url}")
    else:
        print("You have the latest version!")
