# WhisperJAV Upgrade Guide

**From 1.9.x to 1.10:** install 1.10 fresh. 1.10 needs Python 3.13, and a 1.9.x install has Python 3.10, so its `whisperjav-upgrade` cannot install 1.10. Your settings are kept.

**Within 1.10:** the upgrade installs the newest 1.10 release with the package versions it was tested with, and keeps the accelerator your install is made for (`whisperjav_stack.txt` in the install folder). On an Own PyTorch install it never installs, replaces or removes PyTorch.

## Windows installer installs

The GUI tells you when a new version is out: a notice with the new version appears in the header, and **Check for Updates** under the palette icon (🎨) in the header checks at once. Its buttons open the release page on GitHub; the GUI does not upgrade WhisperJAV itself. Upgrade in Command Prompt:

1. Close WhisperJAV (every window of it).
2. Open Command Prompt (press Start, type `cmd`, press Enter) and run:
   ```
   "%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade
   ```
   If you installed into another folder, put that folder in place of `%LOCALAPPDATA%\WhisperJAV`, here and below.
3. When it asks `Continue? [y/N]:`, type `y` and press Enter.
4. When it says "Upgrade complete!", check the version:
   ```
   "%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version
   ```

Start the upgrade with the install's `python.exe`, as above, not with `Scripts\whisperjav-upgrade.exe`: when WhisperJAV is installed on another drive than Windows' temporary folder (normally drive C:), `whisperjav-upgrade.exe` of 1.10.0 and 1.10.1 stops halfway and WhisperJAV no longer starts.

The same command with one option added:

| Add | What it does |
|---|---|
| `--check` | Says whether GitHub has a newer release than yours; changes nothing. It counts every release, while the upgrade installs only the newest 1.10 release. Its last line says "Run 'whisperjav-upgrade' to update."; use the command above instead. The answer is kept for 6 hours; add `--force` as well to ask again at once. |
| `--yes` | Upgrades without asking first. |
| `--wheel-only` | Updates WhisperJAV alone; the packages it uses (CrispASR, PyTorch and the others) keep their versions. |

Do not update with `pip install -U whisperjav…`: on Windows, PyPI serves the CPU-only build of PyTorch, so it can replace your GPU build without a word.

## If the upgrade fails

Before a full upgrade changes anything, the upgrade tool saves a list of the installed packages (not with `--wheel-only`), and after a failure it tries to put that list back. On installs made with the Windows installer this does not work in 1.10: it stops with "Rollback failed". `--rollback`, which the tool suggests when it finishes, stops with an error too.

If WhisperJAV no longer starts after an upgrade, uninstall it and install the newest release with the installer: see [Repair or go back](MANUAL_ROLLBACK.md). For the upgrade tool's messages, see [Troubleshooting](UPGRADE_TROUBLESHOOTING.md).

## Source installs (Windows, macOS, Linux)

In the `WhisperJAV` folder, with the same environment active as at install (for `--local` on Windows, run `.venv\Scripts\activate` first), run `git fetch --tags`, then `git checkout v1.10.x` (the release you want), then the install script again as at install (on Windows with `--local` if you used it). It offers your install's accelerator as the default. Do not use `whisperjav-upgrade` on a source install: in that folder it runs `git pull`, which fails on a clone of a release.

## What stays

Your settings (`%APPDATA%\WhisperJAV` on Windows) and your downloaded models (`%USERPROFILE%\.cache`) are outside the install folder. The upgrade does not change them, and neither does an uninstall or a new install. The XXL Faster Whisper settings **Executable** and **Extra Args** are the exception: WhisperJAV saves them inside the install folder, so enter them again after an upgrade or a new install.
