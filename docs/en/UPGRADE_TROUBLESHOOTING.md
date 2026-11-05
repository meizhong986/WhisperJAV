# Upgrade troubleshooting

This page is for installs made with the Windows installer.

## WhisperJAV no longer starts after an upgrade

For example, the upgrade's last lines say "Rollback failed - installation may be in inconsistent state", or `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version` ends with `ModuleNotFoundError: No module named 'whisperjav'`. Uninstall WhisperJAV and install the newest release with the installer: see [Repair or go back](MANUAL_ROLLBACK.md).

## "[WinError 5] Access is denied" during the upgrade

WhisperJAV is installed on another drive than Windows' temporary folder (normally drive C:), and a program from the install's `Scripts` folder was running while the upgrade replaced it: `whisperjav-upgrade.exe` itself, when the upgrade was started with it, or another WhisperJAV window. The upgrade stops halfway, and WhisperJAV usually no longer starts: repair it as above. pip's advice in the same message ("Consider using the `--user` option or check the permissions.") does not apply. Next time, close every WhisperJAV window and start the upgrade with the install's `python.exe`.

## "No internet connection" or "Cannot upgrade: GitHub's releases list could not be read"

The upgrade first checks that github.com can be reached, then reads GitHub's list of releases. Nothing was changed. Check the connection and try again. The upgrade downloads from GitHub, PyPI and download.pytorch.org; users in China: keep your VPN on during the whole upgrade.

## "Upgrade stopped before any change: …"

Nothing was changed. The rest of the message says why: the new release's list of packages could not be read (check the connection and try again; in China, keep your VPN on), a package of the new release is not available for your PC, or the check timed out. If it happens again, open an issue with the whole output (see below).

## "Stopped: conda installed …, but this release is pinned to …"

Nothing was changed. Report it with the link the message gives.

## "CUDA not available" after the upgrade

**Cause:** usually a CPU-only PyTorch was installed over the CUDA one (for example by a plain `pip install`), or the NVIDIA driver changed.

1. Run `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --check`: it names the accelerator of the install and what PyTorch can use.
2. Check the driver with `nvidia-smi` (580 or newer on Windows, 570 or newer on Linux).
3. Run the WhisperJAV installer again: it keeps the accelerator of the install and puts back the PyTorch build that belongs to it.

## Do not repair with pip

Do not run `pip install -U whisperjav` or `pip install git+https://github.com/…`: on Windows, PyPI serves the CPU-only build of PyTorch, so it can replace your GPU build, and the other packages may not be the versions WhisperJAV was tested with.

## Reporting issues

Open [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) and include:

- the whole output of the upgrade command (all of it, not only the last lines);
- the output of `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --check`, if WhisperJAV still starts;
- the drive WhisperJAV is installed on, and your Windows version (`winver`).
