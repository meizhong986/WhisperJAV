**Sanitizer hardening + small fixes**
The Japanese SRT sanitizer is more robust. Two smaller issues with Colab
install and Ollama translation are also fixed.

---


Subtitle Sanitization

- Hardening of the Japanese SRT sanitization. (#287)


Ollama

- **Curated model list + output length control** — one model in the
  curated list emitted chain-of-thought text that broke the translation
  format; replaced with an instruct-tuned equivalent. Also added
  `--ollama-max-tokens` so output length can be tuned for setups that
  need a different default. (#271)


Colab

- **`--provider local` translation on Colab** — the `[llm]` extra was
  missing from `install_colab.sh`, causing
  `ModuleNotFoundError: starlette_context` when starting local LLM
  translation. The extra is now installed. (#291)

---

### How to Upgrade or Install

**Upgrade from 1.8.10:**

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.11"
```

**Fresh Install:**

### Windows — Standalone Installer (.exe)


1. Download **WhisperJAV-1.8.11-Windows-x86_64.exe** from the Assets below
2. Run the installer (no admin rights required)
3. Wait 10-20 minutes for setup to complete
4. Launch from the Desktop shortcut

Installs to `%LOCALAPPDATA%\WhisperJAV`. A desktop shortcut is created automatically. Your GPU is detected automatically.

### macOS

Requires [Git](https://git-scm.com/downloads). The install script checks for everything else (Xcode CLI Tools, Python, FFmpeg, PortAudio) and tells you exactly what to install if anything is missing. Open Terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.11
installer/install_mac.sh
```

After installation, open the `whisperjav` folder in Finder and double-click **WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10-3.12. The install script handles PEP 668 (externally-managed) environments on Debian 12+ / Ubuntu 24.04+. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.11
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.11
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.




## Compatibility

Same as v1.8.10 — no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| Ollama | 0.3.0+ recommended |

## Known Issues

- **Windows standalone installer does not add its bundled tools to user PATH.**
  WhisperJAV ships its own ffmpeg 7.1 (pinned `>=6,<8`) inside the install
  directory's `Library\bin`, but the installer does not persistently add
  that folder to the user's PATH. If you have a separate ffmpeg on your
  PATH — especially ffmpeg 8.x — WhisperJAV subprocesses that run outside
  the GUI launcher may resolve to that other ffmpeg instead of the bundled
  one. Behavior can be unexpected if the third-party ffmpeg has different
  default flags or filter names. Workaround: launch via the Desktop
  shortcut (which activates the environment) or manually add
  `<install-dir>\Library\bin` to your user PATH. A proper fix is planned
  for v1.8.12.

- **Apple Silicon MPS + whisper-large-v3-turbo** — produces garbage output
  on MPS for this specific model. Use `--hf-device cpu` or the default
  kotoba model. (#198, #227)

- **Ollama download progress** — the download progress bar in the popup is
  indeterminate (pulsing). Real progress is shown in the terminal. A hint
  in the popup directs you to check there.

## What's Next

**v1.8.12** — installer PATH fix (above), a ZipEnhancer Colab init bug
(#290), Qwen3-ASR `transformers` version pin (#280), and verification of
the Colab fix in this release.

**v1.9.0** — full Ollama migration (remove llama-cpp-python), standalone
subtitle merge CLI (#230), Chinese GUI partial i18n (#175, #180), speaker
diarization (#248, #252).

---

Thanks to everyone who reported issues and tested on their own material.
