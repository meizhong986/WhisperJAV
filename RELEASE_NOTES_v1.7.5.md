# WhisperJAV v1.7.5 Release Notes

**Release Date:** January 2026
**Release Type:** Hotfix

## Overview

Version 1.7.5 is a hotfix release that addresses critical issues and introduces experimental features for the upgrade system.

## Critical Bug Fixes

### Fixed: Missing FFmpeg in Installer
- **Issue:** The conda-constructor installer was not correctly bundling FFmpeg, causing "ffmpeg not found" errors
- **Solution:** Updated `construct_v1.7.5.yaml` to properly include FFmpeg in the base environment
- **Impact:** Users can now process videos immediately after installation without additional configuration

## New Experimental Features

### Self-Upgrade System (EXPERIMENTAL)

A new command-line tool for checking and applying updates:

```bash
# Check for updates
whisperjav-upgrade --check

# Upgrade to latest version (with confirmation)
whisperjav-upgrade

# Upgrade without confirmation (for automation)
whisperjav-upgrade --yes

# Wheel-only upgrade (skip installer dependencies)
whisperjav-upgrade --wheel-only
```

**Important Notes:**
- This feature is **EXPERIMENTAL** and still in evaluation phase
- We recommend backing up important files before upgrading
- The upgrade preserves your settings and cached AI models
- If something goes wrong, reinstall from the latest installer
- Please report any issues at: https://github.com/meizhong986/WhisperJAV/issues

### Update Notification System

- GUI now checks for updates on startup
- Notification displayed when new versions are available
- Configurable notification intervals by update severity:
  - Critical/security updates: Always shown
  - Major updates: Shown every launch
  - Minor updates: Shown monthly
  - Patch updates: Shown weekly (dismissable)

## Technical Details

### Version Checker (`whisperjav/version_checker.py`)
- Uses GitHub API to check for latest releases
- Implements caching (6 hours default) to reduce API calls
- Supports environment variables for testing:
  - `WHISPERJAV_UPDATE_API_URL` - Custom API endpoint
  - `WHISPERJAV_RELEASES_URL` - Custom releases page
  - `WHISPERJAV_CACHE_HOURS` - Cache duration

### Upgrade Script (`whisperjav/upgrade.py`)
- Detects installation directory automatically
- Validates network connectivity before upgrading
- Extracts current version from installed package
- Downloads and installs wheel from GitHub releases
- Cleans up old version files after upgrade

### Test Coverage
- 80 automated tests for upgrade system
- `tests/test_upgrade_version_checker.py` - 44 tests
- `tests/test_upgrade_cli.py` - 36 tests

## Files Changed

### New Files
- `whisperjav/version_checker.py` - Version checking and update notifications
- `whisperjav/upgrade.py` - Self-upgrade CLI tool
- `tests/test_upgrade_version_checker.py` - Version checker tests
- `tests/test_upgrade_cli.py` - Upgrade CLI tests
- `installer/upgrade_whisperjav.py` - Standalone upgrade script for installer

### Modified Files
- `whisperjav/__version__.py` - Bumped to 1.7.5
- `setup.py` - Added `whisperjav-upgrade` console script entry point
- `installer/generated/*_v1.7.5.*` - All v1.7.5 installer files generated

## Upgrade Instructions

### From v1.7.4 (Recommended)
```bash
# Option 1: Use the new upgrade command (if available)
whisperjav-upgrade --yes

# Option 2: Manual pip upgrade
pip install --upgrade git+https://github.com/meizhong986/whisperjav.git@v1.7.5
```

### Fresh Installation
Download the installer from the GitHub releases page:
https://github.com/meizhong986/WhisperJAV/releases/tag/v1.7.5

## Known Issues

- Self-upgrade on Windows may require running as Administrator if installed in Program Files
- Update notifications in GUI may not appear immediately after upgrade (requires GUI restart)

## Contributors

- MeiZhong - Development and testing
- Claude (Anthropic) - Code assistance and documentation

---

For questions or issues, please open a GitHub issue at:
https://github.com/meizhong986/WhisperJAV/issues
