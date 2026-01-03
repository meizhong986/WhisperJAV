#!/usr/bin/env python3
"""
WhisperJAV Version Checker

Checks for new versions on GitHub and provides update notifications.
Supports both stable releases and development (latest commit) updates.
Uses GitHub API with caching and retry logic.
"""

import os
import sys
import json
import time
import subprocess
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass, field
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

# GitHub API endpoints
GITHUB_REPO = os.environ.get('WHISPERJAV_REPO', 'meizhong986/whisperjav')
GITHUB_API_URL = os.environ.get(
    'WHISPERJAV_UPDATE_API_URL',
    f'https://api.github.com/repos/{GITHUB_REPO}/releases/latest'
)
GITHUB_COMMITS_API_URL = os.environ.get(
    'WHISPERJAV_COMMITS_API_URL',
    f'https://api.github.com/repos/{GITHUB_REPO}/commits/main'
)
GITHUB_COMPARE_API_URL = os.environ.get(
    'WHISPERJAV_COMPARE_API_URL',
    f'https://api.github.com/repos/{GITHUB_REPO}/compare'
)
GITHUB_RELEASES_URL = os.environ.get(
    'WHISPERJAV_RELEASES_URL',
    f'https://github.com/{GITHUB_REPO}/releases'
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
class CommitInfo:
    """Information about a single commit."""
    sha: str
    short_sha: str
    message: str
    author: str
    date: str
    url: str


@dataclass
class DevUpdateInfo:
    """Information about development (latest commit) updates."""
    commits_ahead: int
    latest_commit: Optional[CommitInfo] = None
    recent_commits: List[CommitInfo] = field(default_factory=list)
    base_tag: Optional[str] = None  # The release tag we're comparing against


@dataclass
class UpdateCheckResult:
    """Result of an update check."""
    # Current installation info
    current_version: str
    current_commit: Optional[str] = None  # Short SHA if known
    installation_type: str = "release"  # "release" or "dev"

    # Stable release track
    update_available: bool = False
    latest_version: Optional[str] = None
    version_info: Optional[VersionInfo] = None

    # Development track
    dev_update_available: bool = False
    dev_info: Optional[DevUpdateInfo] = None

    # Meta
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


def _fetch_github_api(url: str, timeout: int = 10) -> Optional[Dict]:
    """
    Fetch data from GitHub API.

    Args:
        url: GitHub API URL
        timeout: Request timeout in seconds

    Returns:
        JSON response dict or None on failure
    """
    try:
        request = urllib.request.Request(
            url,
            headers={
                'Accept': 'application/vnd.github.v3+json',
                'User-Agent': f'WhisperJAV/{CURRENT_VERSION}'
            }
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None


def _get_current_commit() -> Optional[str]:
    """
    Get the current installation's commit hash.

    Tries multiple methods:
    1. Check if running from git repo (development)
    2. Check stored commit hash from pip install

    Returns:
        Short commit hash (7 chars) or None
    """
    # Method 1: Running from git repo (development)
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).parent.parent  # whisperjav package root
        )
        if result.returncode == 0:
            return result.stdout.strip()[:7]
    except Exception:
        pass

    # Method 2: Check for stored commit hash (from pip install git+...)
    try:
        commit_file = CACHE_DIR / 'installed_commit.txt'
        if commit_file.exists():
            return commit_file.read_text().strip()[:7]
    except Exception:
        pass

    return None


def _fetch_latest_commit(timeout: int = 10) -> Optional[CommitInfo]:
    """
    Fetch the latest commit from the main branch.

    Returns:
        CommitInfo or None on failure
    """
    data = _fetch_github_api(GITHUB_COMMITS_API_URL, timeout)
    if not data:
        return None

    try:
        commit = data.get('commit', {})
        return CommitInfo(
            sha=data.get('sha', ''),
            short_sha=data.get('sha', '')[:7],
            message=commit.get('message', '').split('\n')[0][:100],  # First line, truncated
            author=commit.get('author', {}).get('name', 'Unknown'),
            date=commit.get('author', {}).get('date', ''),
            url=data.get('html_url', '')
        )
    except Exception:
        return None


def _fetch_commits_since_tag(tag: str, timeout: int = 10) -> Optional[DevUpdateInfo]:
    """
    Fetch commits between a release tag and the main branch.

    Args:
        tag: Release tag to compare from (e.g., 'v1.7.5')

    Returns:
        DevUpdateInfo with commit list, or None on failure
    """
    # Use GitHub compare API: /repos/{owner}/{repo}/compare/{base}...{head}
    compare_url = f"{GITHUB_COMPARE_API_URL}/{tag}...main"
    data = _fetch_github_api(compare_url, timeout)

    if not data:
        return None

    try:
        commits_data = data.get('commits', [])
        ahead_by = data.get('ahead_by', len(commits_data))

        # Parse recent commits (up to 10)
        recent_commits = []
        for commit_data in commits_data[-10:]:  # Last 10 commits
            commit = commit_data.get('commit', {})
            recent_commits.append(CommitInfo(
                sha=commit_data.get('sha', ''),
                short_sha=commit_data.get('sha', '')[:7],
                message=commit.get('message', '').split('\n')[0][:80],
                author=commit.get('author', {}).get('name', 'Unknown'),
                date=commit.get('author', {}).get('date', ''),
                url=commit_data.get('html_url', '')
            ))

        # Reverse to show newest first
        recent_commits.reverse()

        return DevUpdateInfo(
            commits_ahead=ahead_by,
            latest_commit=recent_commits[0] if recent_commits else None,
            recent_commits=recent_commits,
            base_tag=tag
        )
    except Exception:
        return None


def check_for_updates(force: bool = False, include_dev: bool = True) -> UpdateCheckResult:
    """
    Check if a newer version of WhisperJAV is available.

    Checks both:
    1. Stable releases (tagged versions on GitHub)
    2. Development updates (commits on main branch since last release)

    Args:
        force: If True, bypass cache and check immediately
        include_dev: If True, also check for dev updates (commits since release)

    Returns:
        UpdateCheckResult with both release and dev update information
    """
    current = CURRENT_VERSION
    current_commit = _get_current_commit()

    # Determine installation type
    # If we can get a commit hash, likely installed from git
    installation_type = "dev" if current_commit else "release"

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

                # Reconstruct dev_info from cache if available
                dev_info = None
                dev_update_available = False
                if cached.get('dev_info'):
                    di = cached['dev_info']
                    commits_ahead = di.get('commits_ahead', 0)
                    dev_update_available = commits_ahead > 0

                    recent_commits = []
                    for c in di.get('recent_commits', []):
                        recent_commits.append(CommitInfo(
                            sha=c.get('sha', ''),
                            short_sha=c.get('short_sha', ''),
                            message=c.get('message', ''),
                            author=c.get('author', ''),
                            date=c.get('date', ''),
                            url=c.get('url', '')
                        ))

                    latest_commit = None
                    if di.get('latest_commit'):
                        lc = di['latest_commit']
                        latest_commit = CommitInfo(
                            sha=lc.get('sha', ''),
                            short_sha=lc.get('short_sha', ''),
                            message=lc.get('message', ''),
                            author=lc.get('author', ''),
                            date=lc.get('date', ''),
                            url=lc.get('url', '')
                        )

                    dev_info = DevUpdateInfo(
                        commits_ahead=commits_ahead,
                        latest_commit=latest_commit,
                        recent_commits=recent_commits,
                        base_tag=di.get('base_tag')
                    )

                return UpdateCheckResult(
                    current_version=current,
                    current_commit=current_commit,
                    installation_type=installation_type,
                    update_available=update_available,
                    latest_version=latest,
                    version_info=version_info,
                    dev_update_available=dev_update_available,
                    dev_info=dev_info,
                    from_cache=True
                )

    # Fetch latest release from GitHub
    release_data = _fetch_latest_release()

    if not release_data:
        return UpdateCheckResult(
            current_version=current,
            current_commit=current_commit,
            installation_type=installation_type,
            error="Could not check for updates"
        )

    # Parse release data
    latest = release_data.get('tag_name', '').lstrip('v')
    release_tag = release_data.get('tag_name', f'v{latest}')

    if not latest:
        return UpdateCheckResult(
            current_version=current,
            current_commit=current_commit,
            installation_type=installation_type,
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

    # Compare versions for stable track
    update_available = compare_versions(current, latest) < 0

    # Fetch dev track info (commits since release)
    dev_info = None
    dev_update_available = False

    if include_dev:
        dev_info = _fetch_commits_since_tag(release_tag)
        if dev_info:
            # Dev update available if there are commits ahead
            # AND user doesn't already have the latest commit
            if dev_info.commits_ahead > 0:
                if current_commit and dev_info.latest_commit:
                    # User has a commit hash - check if it matches latest
                    dev_update_available = current_commit != dev_info.latest_commit.short_sha
                else:
                    # Can't compare commits, assume update available
                    dev_update_available = True

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

    # Cache dev info if available
    if dev_info:
        cache_data['dev_info'] = {
            'commits_ahead': dev_info.commits_ahead,
            'base_tag': dev_info.base_tag,
            'latest_commit': {
                'sha': dev_info.latest_commit.sha,
                'short_sha': dev_info.latest_commit.short_sha,
                'message': dev_info.latest_commit.message,
                'author': dev_info.latest_commit.author,
                'date': dev_info.latest_commit.date,
                'url': dev_info.latest_commit.url
            } if dev_info.latest_commit else None,
            'recent_commits': [
                {
                    'sha': c.sha,
                    'short_sha': c.short_sha,
                    'message': c.message,
                    'author': c.author,
                    'date': c.date,
                    'url': c.url
                }
                for c in dev_info.recent_commits[:5]  # Cache up to 5 commits
            ]
        }

    _save_cache(cache_data)

    return UpdateCheckResult(
        current_version=current,
        current_commit=current_commit,
        installation_type=installation_type,
        update_available=update_available,
        latest_version=latest,
        version_info=version_info,
        dev_update_available=dev_update_available,
        dev_info=dev_info,
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
    if _get_current_commit():
        print(f"Current commit: {_get_current_commit()}")
    print("Checking for updates...\n")

    result = check_for_updates(force=True)

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    # Stable release track
    print("=" * 50)
    print("[STABLE RELEASE]")
    print("=" * 50)
    if result.update_available:
        print(f"  Update available: v{result.latest_version}")
        print(f"  Release URL: {result.version_info.release_url}")
        if result.version_info.download_url:
            print(f"  Download: {result.version_info.download_url}")
    else:
        print(f"  [OK] You have the latest stable release (v{result.latest_version})")

    # Development track
    print("\n" + "=" * 50)
    print("[DEVELOPMENT] (main branch)")
    print("=" * 50)
    if result.dev_info:
        if result.dev_info.commits_ahead > 0:
            print(f"  {result.dev_info.commits_ahead} commit(s) ahead of {result.dev_info.base_tag}")
            if result.dev_update_available:
                print("  -> Updates available!")
            else:
                print("  [OK] You have the latest development version")

            if result.dev_info.recent_commits:
                print("\n  Recent commits:")
                for commit in result.dev_info.recent_commits[:5]:
                    # Encode safely for Windows console
                    msg = commit.message[:60].encode('ascii', 'replace').decode('ascii')
                    print(f"    * [{commit.short_sha}] {msg}")
        else:
            print("  [OK] Release is up to date with main branch")
    else:
        print("  Could not fetch development info")

    print("\n" + "=" * 50)
    print("To update:")
    print("  Stable:  pip install -U whisperjav")
    print("  Dev:     pip install -U git+https://github.com/meizhong986/whisperjav.git")
