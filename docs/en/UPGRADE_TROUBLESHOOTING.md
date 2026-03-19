# Upgrade Troubleshooting Guide

This guide helps resolve common issues during WhisperJAV upgrades.

## Preflight Check Failures

### "Insufficient disk space"

**Error:** `Disk space: X.X GB available, need 5 GB`

**Solution:**
1. Free up disk space on the drive where WhisperJAV is installed
2. Delete old files: `%TEMP%\*`, browser cache, old downloads
3. Run Disk Cleanup: `cleanmgr`

### "Wrong Python environment detected"

**Error:**
```
Running from: C:\some\other\python
Expected:     C:\Users\...\AppData\Local\WhisperJAV
```

**Solution:**
Run the upgrade script using WhisperJAV's Python:
```cmd
%LOCALAPPDATA%\WhisperJAV\python.exe upgrade_whisperjav.py
```

### "WhisperJAV GUI is running"

**Error:** `GUI check failed: WhisperJAV GUI is running`

**Solution:**
1. Close the WhisperJAV GUI window
2. If it doesn't close, open Task Manager and end:
   - `pythonw.exe` (WhisperJAV GUI)
   - `WhisperJAV-GUI.exe`
3. Try the upgrade again

### "No internet connection"

**Error:** `Network: no connection to GitHub`

**Solution:**
1. Check your internet connection
2. Try accessing https://github.com in a browser
3. If behind a corporate firewall/proxy:
   ```cmd
   set HTTPS_PROXY=http://your-proxy:port
   python.exe upgrade_whisperjav.py
   ```

## Upgrade Execution Failures

### "Package upgrade failed"

**Symptoms:** Upgrade stops during "Upgrading WhisperJAV package..."

**Causes:**
- Network timeout
- GitHub temporarily unavailable
- pip cache corruption

**Solutions:**

1. **Retry the upgrade:**
   ```cmd
   python.exe upgrade_whisperjav.py
   ```

2. **Clear pip cache and retry:**
   ```cmd
   python.exe -m pip cache purge
   python.exe upgrade_whisperjav.py
   ```

3. **Use wheel-only mode** (if you just need the latest code):
   ```cmd
   python.exe upgrade_whisperjav.py --wheel-only
   ```

### "Dependency installation failed"

**Symptoms:** Some packages fail to install during dependency phase

**Solutions:**

1. **Continue anyway** - Most dependency failures are non-fatal
2. **Install manually:**
   ```cmd
   python.exe -m pip install <package-name>
   ```

### "Permission denied" / "Access denied"

**Symptoms:** File operations fail

**Solutions:**
1. Close ALL WhisperJAV windows
2. Run upgrade from an elevated command prompt (Run as Administrator)
3. Temporarily disable antivirus real-time protection

### "Could not update launcher"

**Symptoms:** Desktop shortcut not updated

**This is non-fatal.** WhisperJAV still works. To fix manually:
1. Right-click the desktop shortcut
2. Click Properties
3. Update the target path

## Post-Upgrade Issues

### "Import error" after upgrade

**Symptoms:** WhisperJAV won't start, shows import errors

**Solutions:**

1. **Reinstall with dependencies:**
   ```cmd
   python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@main
   ```

2. **Check for conflicting packages:**
   ```cmd
   python.exe -m pip check
   ```

### "CUDA not available" after upgrade

**Symptoms:** GPU acceleration stopped working

**Cause:** CPU-only PyTorch was installed over CUDA version

**Solution:**
```cmd
REM Check your CUDA version first
nvidia-smi

REM Reinstall CUDA PyTorch (example for CUDA 12.1)
python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall
```

### Version didn't change after upgrade

**Symptoms:** `--version` shows old version

**Solutions:**
1. Clear Python's import cache:
   ```cmd
   python.exe -c "import whisperjav; print(whisperjav.__file__)"
   ```
2. Reinstall:
   ```cmd
   python.exe -m pip install --force-reinstall --no-cache-dir git+https://github.com/meizhong986/whisperjav.git@main
   ```

## Advanced Options

### Skip preflight checks

**Use with caution** - only if you know what you're doing:
```cmd
python.exe upgrade_whisperjav.py --skip-preflight
```

### Hot-patch mode

Only updates WhisperJAV code, skips all dependencies:
```cmd
python.exe upgrade_whisperjav.py --wheel-only
```

### Non-interactive mode

For scripted upgrades:
```cmd
python.exe upgrade_whisperjav.py --yes
```

## When All Else Fails

See [Manual Rollback Guide](MANUAL_ROLLBACK.md) for complete recovery options.

## Reporting Issues

If you encounter an issue not covered here:

1. Open [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues)
2. Include:
   - Full console output from the upgrade attempt
   - Windows version (`winver`)
   - Python version (`python.exe --version`)
   - Current WhisperJAV version (if known)
