# Manual Rollback Guide

If an upgrade fails or causes issues, follow these steps to restore WhisperJAV to a working state.

## Quick Reference

| Scenario | Solution |
|----------|----------|
| Upgrade failed mid-way | [Option A: Reinstall specific version](#option-a-reinstall-specific-version) |
| New version has bugs | [Option A: Reinstall specific version](#option-a-reinstall-specific-version) |
| Complete corruption | [Option B: Fresh install](#option-b-fresh-install) |
| Can't run Python at all | [Option B: Fresh install](#option-b-fresh-install) |

## Option A: Reinstall Specific Version

Use this if Python still works but WhisperJAV is broken.

### Step 1: Open Command Prompt

1. Press `Win + R`
2. Type `cmd` and press Enter

### Step 2: Navigate to WhisperJAV

```cmd
cd %LOCALAPPDATA%\WhisperJAV
```

### Step 3: Reinstall the version you want

**To reinstall latest stable:**
```cmd
python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@main
```

**To install a specific version (e.g., v1.7.3):**
```cmd
python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@v1.7.3
```

**Available version tags:**
- `v1.7.4` - Latest
- `v1.7.3` - Previous stable
- `v1.7.2`, `v1.7.1`, `v1.7.0` - Older versions

### Step 4: Verify installation

```cmd
python.exe -c "from whisperjav import __version__; print(__version__)"
```

## Option B: Fresh Install

Use this if the Python environment is completely broken.

### Step 1: Uninstall WhisperJAV

1. Open **Settings** → **Apps** → **Apps & features**
2. Search for "WhisperJAV"
3. Click **Uninstall**

Or manually delete:
```cmd
rmdir /s /q "%LOCALAPPDATA%\WhisperJAV"
```

### Step 2: Download fresh installer

Go to [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) and download the latest `.exe` installer.

### Step 3: Run the installer

The installer will create a fresh environment with all dependencies.

## Preserving Your Data

Your AI models and cache are stored separately and survive reinstallation:

| Data | Location | Preserved? |
|------|----------|------------|
| Whisper models | `~/.cache/whisper/` | Yes |
| HuggingFace models | `~/.cache/huggingface/` | Yes |
| WhisperJAV config | `%LOCALAPPDATA%\WhisperJAV\whisperjav_config.json` | Deleted on uninstall |

**To preserve config before uninstall:**
```cmd
copy "%LOCALAPPDATA%\WhisperJAV\whisperjav_config.json" "%USERPROFILE%\Desktop\"
```

## Common Issues

### "pip is not recognized"

Your PATH is not set correctly. Use the full path:
```cmd
%LOCALAPPDATA%\WhisperJAV\python.exe -m pip install ...
```

### "Permission denied"

Close all WhisperJAV windows and try again. If still failing:
1. Open Task Manager (`Ctrl+Shift+Esc`)
2. End any `python.exe` or `pythonw.exe` processes from WhisperJAV
3. Try the command again

### "Network error"

Check your internet connection. If behind a proxy:
```cmd
set HTTPS_PROXY=http://your-proxy:port
python.exe -m pip install ...
```

## Getting Help

If none of the above works:

1. Open an issue at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues)
2. Include:
   - What you were doing when it broke
   - Error messages (screenshot or copy-paste)
   - Windows version
   - Output of: `python.exe --version`
