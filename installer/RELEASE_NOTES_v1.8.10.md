**Quality + Stability + Ollama (beta)**
Transcription accuracy significantly improved in aggressive mode
Ollama integration is now available in GUI 
Many bugs fixed across the board

---


Transcription Quality Improvement

- **Aggressive sensitivity retune** — I ran the aggressive preset against Netflix ground truth in my test cases. I found that several parameters were working against each other. Coverage improved from 76.5% to 92.6% after the accuracy upgrade.


Ollama GUI Integration

- **Ollama as a provider** — Pre-installed Ollama is required. WhisperJAV integrates into your installed Ollama. In GUI select "Ollama" from the provider dropdown in the Ensemble tab or Standalone Translation tab. Wait for Ollama server to be discovered and connect. Select model or provide model. 

- **Three-state onboarding** — When you select Ollama, the GUI detects your setup and shows the right panel: not installed (download link), no model (recommended model with copyable `ollama pull` command), or connected (green status dot).

- **In-GUI model download** — Selecting an uninstalled model opens a confirmation popup with the model name and download size. Click "Download" to pull it directly. Progress streams to the terminal.

- **VRAM-aware model recommendations** — The GUI detects your GPU VRAM and recommends the best model: gemma3:4b for 4GB, qwen2.5:7b for 8GB, gemma3:12b for 12GB, qwen2.5:14b for 16GB+.

- **Automatic VRAM cleanup** — When ensemble translation finishes, the Ollama model is automatically unloaded from VRAM. Previously it stayed loaded indefinitely, blocking other GPU workloads.

- **Auto-start server** — If Ollama is installed but not running, the GUI starts it automatically when you select the Ollama provider.

- **llama-cpp deprecated** — The "Local LLM" option is renamed to "llama-cpp (deprecated)". I plan to remove it entirely in v1.9.0 in favor of Ollama.

Speech Enhanceer for VAD ONLY option

- **Checkbox in ensemble UI** — The "Enhance for VAD only" option (use enhanced audio for speech detection but original audio for transcription) is now accessible as a checkbox below each enhancer dropdown in the ensemble tab. Previously this was only available inside the Qwen Customize Parameters modal. Works for all pipelines. (#253)

---
Bug Fixes

- **XXL exe path lost on restart** — The BYOP panel's XXL executable path was not restored when reopening the app. Caused by a race condition between DOMContentLoaded and pywebview API readiness. Fixed by reloading BYOP preferences after pywebview is ready.

- **XXL --model hardcoded** — The `--model` flag was hardcoded in the XXL runner, preventing users from changing it. Moved to user-editable Extra Args field (default: `--model large-v3`).

- **Silero VAD crashes in Colab/Kaggle** — `torch.hub.load()` calls `input()` for interactive trust confirmation, which crashes with EOFError in non-interactive environments. Fixed with `trust_repo=True`. (#253)

- **Config contamination between runs** — VAD parameters from one run could leak into the next due to mutable default dictionaries. Added a contamination firewall that deep-copies config at pipeline entry.

- **GUI ensemble presets not applied** — Switching between pipelines of the same type (e.g., balanced to fast, both using Faster-Whisper) did not refresh the preset values. The GUI now always reloads presets on pipeline change.

- **Colab/Kaggle notebook fixes** — Added `llvmlite>=0.46.0` to install cells and `TORCH_HUB_TRUST_REPO=1` environment variable. (#253, #231)

- **Translation diagnostic hardening** — Fixed 12+ issues in translation pipeline: truthful statistics, debug wiring, temperature clobber, Qwen3 thinking model support, ground-truth detection, instruction delivery, PySubtrans integration diagnostics, and Ollama server-side log guidance.

- **Output directory not honored** — Fixed output path handling and added artifact cleanup safety guard.

---

### How to Upgrade or Install

**Upgrade from 1.8.9:**

```
pip install -U --no-deps "whisperjav @ git+https://github.com/meizhong986/whisperjav.git@v1.8.10"
```

**Fresh Install:**

### Windows — Standalone Installer (.exe)


1. Download **WhisperJAV-1.8.9-Windows-x86_64.exe** from the Assets below
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
git checkout v1.8.9
installer/install_mac.sh
```

After installation, open the `whisperjav` folder in Finder and double-click **WhisperJAV.command** to launch the GUI.

### Linux

Requires Git and Python 3.10-3.12. The install script handles PEP 668 (externally-managed) environments on Debian 12+ / Ubuntu 24.04+. Open a terminal and run:

```bash
cd ~
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.9
installer/install_linux.sh
```

After installation, launch the GUI with `./WhisperJAV.sh`.

### Windows — Source Install

Requires [Git](https://git-scm.com/downloads) and [Python 3.10-3.12](https://www.python.org/downloads/). Open a terminal and run:

```
cd %USERPROFILE%
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.9
installer\install_windows.bat
```

After installation, double-click **WhisperJAV.bat** to launch the GUI.




## Compatibility

Same as v1.8.9 — no dependency changes.

| Component | Supported Versions |
|-----------|-------------------|
| Python | 3.10, 3.11, 3.12 |
| PyTorch | 2.4.0 - 2.10.x |
| CUDA | 11.8+ (12.4+ recommended) |
| Ollama | 0.3.0+ recommended |

## Known Issues

- **Apple Silicon MPS + whisper-large-v3-turbo** — Produces garbage output on MPS for this specific model. Use `--hf-device cpu` or the default kotoba model. (#198, #227)

- **Ollama download progress** — The download progress bar in the popup is indeterminate (pulsing). Real progress is shown in the terminal. A hint in the popup directs you to check there.

## What's Next (v1.9.0)

- Full Ollama migration — remove llama-cpp-python dependency entirely
- Standalone subtitle merge CLI tool (`whisperjav-merge`) (#230)
- Chinese GUI (partial i18n) (#175, #180)
- Speaker diarization (#248, #252)

---
## Technical Details


### New Developer Tools

- `scripts/whisper_param_tuner.py` — Standalone CLI for testing Whisper parameters against ground truth SRT in seconds (vs 8+ min full pipeline). Supports `--sweep` for grid search, `--gt-focus` for specific subtitle IDs, both OpenAI and Faster-Whisper backends.

- Diagnostic JSON — Saved to `{scene_name}_diagnostic.json` alongside scene SRTs. Contains full segment data including no_speech_prob, compression_ratio, temperature per segment.

