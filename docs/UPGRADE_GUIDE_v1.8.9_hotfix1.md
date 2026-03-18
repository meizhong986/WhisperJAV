# Upgrade Guide: v1.8.9 to v1.8.9 Hotfix 1

This guide is for users already on v1.8.9 who want to apply the hotfix without reinstalling everything.

The hotfix fixes 3 bugs and adds Portuguese translation. No dependencies change — it's a code-only update.

---

## What the hotfix fixes

1. **Ollama translation 404 error** — `--provider ollama` always failed with HTTP 404 (#132)
2. **GUI shows old interface after upgrade** — WebView2 cached stale HTML/CSS from the previous version (#236)
3. **Icon crash on first launch** — `OverflowError: int too long to convert` on 64-bit Windows (#235)
4. **Portuguese translation** — Added as a new target language (#238)

---

## How to upgrade

### Option 1: One command (recommended)

Open a terminal and run:

```
whisperjav-upgrade
```

This works for both Windows installer users and source install users. It auto-detects your install type and does the right thing.

**What it does behind the scenes:**
- Source installs: `git pull` + `uv sync`
- Installer installs: `pip install -U` from GitHub

### Option 2: Manual upgrade (source install)

If you installed from git:

```
cd whisperjav
git pull
```

That's it. No `uv sync` needed — no dependencies changed in this hotfix.

### Option 3: Manual upgrade (Windows installer)

If you used the .exe installer and prefer to upgrade manually:

```
cd %LOCALAPPDATA%\WhisperJAV
Scripts\pip install -U "whisperjav @ git+https://github.com/meizhong986/whisperjav.git"
```

### Option 4: Fresh install (Windows installer .exe)

If you prefer a clean install, download the hotfix installer from the [Releases page](https://github.com/meizhong986/WhisperJAV/releases). This installs everything from scratch (10-20 minutes).

### Colab / Kaggle

Re-run your install cell. It always pulls the latest version from GitHub:

```python
!pip install "whisperjav[colab] @ git+https://github.com/meizhong986/whisperjav.git"
```

---

## After upgrading

- **GUI users**: The hotfix automatically clears the old WebView2 cache on first launch. You should see the updated interface immediately. No manual cache clearing needed.
- **Ollama users**: `--provider ollama` should now work without the 404 error.
- **Verify your version**: In the GUI, the header shows the version. Or run:

```
python -c "from whisperjav.__version__ import __version__; print(__version__)"
```

Expected output: `1.8.9.post1`

---

## Troubleshooting

**"whisperjav-upgrade" not found:**
Your install might not have the upgrade script on PATH. Use Option 2 or 3 instead.

**GUI still shows old interface after upgrade:**
Close the GUI completely and relaunch. The cache is cleared on the next startup after the upgrade.

**Permission errors on Windows:**
Close any running WhisperJAV windows first, then retry the upgrade command.
