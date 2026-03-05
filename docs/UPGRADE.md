# WhisperJAV Upgrade Guide

This guide covers how to upgrade WhisperJAV to the latest version.

## GUI Users (Recommended)

The GUI will automatically check for updates on startup and display a notification banner when a new version is available.

### Update Notification Banner

When an update is available, you'll see a colored banner at the top of the window:

| Color | Update Type | Description |
|-------|-------------|-------------|
| Red | Critical | Security or critical bug fix - always shown |
| Blue | Major | Major version update (e.g., 1.7 → 2.0) |
| Green | Minor | New features (e.g., 1.7.4 → 1.8.0) |
| Gray | Patch | Bug fixes (e.g., 1.7.4 → 1.7.5) |

### One-Click Update

1. Click **Update Now** on the notification banner
2. Confirm the update in the dialog
3. The GUI will close automatically
4. The update runs in the background
5. The GUI relaunches when complete

### View Release Notes

Click **Release Notes** on the banner to see what's new in the update.

### Dismiss Notifications

Click the **×** button to dismiss the notification. You'll be reminded:
- Patch updates: Weekly
- Minor updates: Monthly
- Major updates: Every launch
- Critical updates: Always shown (cannot dismiss)

## CLI Users

Use the `whisperjav-upgrade` command for command-line upgrades.

### Check for Updates

```bash
whisperjav-upgrade --check
```

This shows if an update is available without installing it.

### Interactive Upgrade

```bash
whisperjav-upgrade
```

You'll be prompted to confirm before the upgrade begins.

### Non-Interactive Upgrade

```bash
whisperjav-upgrade --yes
```

Automatically confirms the upgrade (useful for scripts).

### Hot-Patch Mode

```bash
whisperjav-upgrade --wheel-only
```

Only updates the WhisperJAV package itself, skipping dependency installation. This is faster and safer for post-release patches (e.g., 1.7.4 → 1.7.4.post1).

### Show Script Version

```bash
whisperjav-upgrade --version
```

### Force Update Check

```bash
whisperjav-upgrade --check --force
```

Bypasses the 6-hour cache to check for updates immediately.

## What Gets Updated

| Component | Full Upgrade | Wheel-Only |
|-----------|--------------|------------|
| WhisperJAV code | Yes | Yes |
| New dependencies | Yes | No |
| numpy/librosa fix | Yes | No |
| Desktop shortcut | Yes | Yes |
| Old files cleanup | Yes | No |

## What Gets Preserved

The upgrade preserves:
- AI models (`~/.cache/whisper/`, `~/.cache/huggingface/`)
- Configuration (`whisperjav_config.json`)
- Cache data (`.whisperjav_cache/`)

## Troubleshooting

See [UPGRADE_TROUBLESHOOTING.md](UPGRADE_TROUBLESHOOTING.md) for common issues and solutions.

## Manual Rollback

If an upgrade fails, see [MANUAL_ROLLBACK.md](MANUAL_ROLLBACK.md) for recovery options.

## Running Standalone Upgrade Script

If the installed upgrade command doesn't work, you can download and run the standalone script:

```cmd
cd %LOCALAPPDATA%\WhisperJAV
curl -O https://raw.githubusercontent.com/meizhong986/whisperjav/main/installer/upgrade_whisperjav.py
python.exe upgrade_whisperjav.py
```

## Technical Details

### Version Check Caching

Update checks are cached for 6 hours to avoid excessive API calls. Use `--force` to bypass the cache.

### Update Process Flow

1. **GUI detects update** → Shows notification banner
2. **User clicks "Update Now"** → Confirmation dialog
3. **GUI spawns update wrapper** → Background process
4. **GUI exits** → Releases file locks
5. **Wrapper waits for GUI** → Ensures clean exit
6. **Upgrade script runs** → Updates packages
7. **Wrapper relaunches GUI** → Fresh start

### Files Created

- `update.log` - Detailed upgrade log (in installation directory)
- `.whisperjav_cache/version_check.json` - Cached update check result
- `.whisperjav_cache/update_dismissed.json` - Dismissed notification tracking
