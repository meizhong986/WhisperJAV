# Repair WhisperJAV or go back to an earlier version

This page is for installs made with the Windows installer. The upgrade tool's own rollback (`--rollback`, and the one it tries after a failed upgrade) does not work on these installs in 1.10: it stops with an error. Use the steps below.

## Repair an install that no longer starts

1. Uninstall WhisperJAV: double-click `Uninstall-WhisperJAV.exe` in the install folder (by default `%LOCALAPPDATA%\WhisperJAV`), or open Windows **Settings > Apps**, find **WhisperJAV** and choose **Uninstall**. After an upgrade, the Apps entry still names the version you first installed.
2. When the uninstaller has finished, download the installer of the newest release from [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) and install it. The uninstall leaves the old desktop icon (for example "WhisperJAV v1.10.0"); delete it and start WhisperJAV from the icon the new install makes.
3. The installer chooses the accelerator as at your first install: a supported NVIDIA GPU is used without a question; otherwise a window asks. To make the install for another accelerator, run the installer again into the same folder afterwards. On an Own PyTorch install, add your PyTorch again with the command it shows at the end.

## Go back to an earlier 1.10 release

1. Uninstall WhisperJAV as above.
2. Download the installer of the earlier 1.10 release you want from its page under [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) (for example v1.10.0) and install it.

## What stays

Your settings (`%APPDATA%\WhisperJAV`) and your downloaded models (`%USERPROFILE%\.cache`) are outside the install folder: the uninstall and the new install leave them as they are. The XXL Faster Whisper settings **Executable** and **Extra Args** are the exception: WhisperJAV saves them inside the install folder, so enter them again after a new install.

## Do not repair with pip

Do not run `pip install --force-reinstall git+https://github.com/…` or `pip install -U whisperjav`: on Windows, PyPI serves the CPU-only build of PyTorch, so it can replace your GPU build, and the other packages may not be the versions WhisperJAV was tested with.

## Getting help

See [Reporting issues](UPGRADE_TROUBLESHOOTING.md#reporting-issues).
