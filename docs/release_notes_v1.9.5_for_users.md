# WhisperJAV 1.9.5

## Key points

- Fix release for 1.9.4. No library underneath WhisperJAV changes; on 1.9.3 or 1.9.4, upgrade with
  `whisperjav-upgrade --wheel-only` (steps below).
- The GUI no longer writes subtitles into WhisperJAV's program folder when the saved output folder
  is empty.
- The Fidelity pipeline works again in the Colab notebook.
- New Ensemble defaults: pass 1 Qwen3-ASR with FireRedVAD, pass 2 anime-whisper with TEN VAD, both
  Aggressive.

## Changes

**Fixes**

- GUI, *Remember settings* on: an empty saved output folder sent subtitles into WhisperJAV's program
  folder. They now go next to the source video, and the *Save next to source video* box shows this. (#454)
- Colab: the 1.9.4 notebook replaced Colab's PyTorch, after which the Fidelity pipeline crashed and
  wrote no subtitles. The notebook now keeps Colab's PyTorch and adds the CUDA 12 libraries that
  faster-whisper needs. Tested on a T4 with Balanced and Fidelity in two passes. (#456, #447)
- Translation with Ollama: `<summary>` and `<scene>` blocks that a local model repeats in its reply
  could appear as subtitle lines. WhisperJAV now removes them before reading the reply. If you still
  see such lines, please report them in #455 with the model name. (#455, PR #443)

**Changed defaults**

- DeepSeek: the default model is now `deepseek-flash`, DeepSeek's current name for its fast model.
  The old name, `deepseek-v4-flash`, is still accepted, and a saved choice is kept. WhisperJAV turns
  DeepSeek's thinking (reasoning) off for both. (PR #457)
- Ensemble tab: pass 1 starts as Qwen3-ASR with FireRedVAD, pass 2 as anime-whisper with TEN VAD,
  both Aggressive. Pass 2 stays off until you tick it. With *Remember settings* on, your saved
  choices are kept.
- Command line, two-pass runs: when pass 1 is Qwen3-ASR or pass 2 is anime-whisper, a sensitivity or
  speech segmenter you do not give now defaults to Aggressive with FireRedVAD (pass 1) or Aggressive
  with TEN (pass 2). 1.9.4 used Balanced with WhisperSeg. To keep the 1.9.4 behaviour, add
  `--pass1-sensitivity balanced --pass1-speech-segmenter whisperseg` (or the `--pass2-` versions).
- If pass 1 is Qwen3-ASR with FireRedVAD and FireRedVAD is not installed, the run stops before
  transcribing and says so. FireRedVAD is installed by default since 1.9.2.

**Documentation**

- The Chinese Windows install guides now point to `whisperjav-upgrade` instead of `pip install -U`.

## Installation

Do not upgrade with `pip install -U whisperjav`: on Windows it replaces your GPU build of PyTorch
with a CPU-only one. Your settings and models are kept in every case below.

### Upgrade from 1.9.3 or 1.9.4

Only WhisperJAV itself is replaced. Close WhisperJAV first.

**Windows (installed with the .exe)**

1. Open Command Prompt: press Start, type `cmd`, press Enter.
2. Go to the installation folder. The default is:
   ```
   cd /d "%LOCALAPPDATA%\WhisperJAV"
   ```
   If you chose another folder during installation, use that folder instead.
3. Run:
   ```
   Scripts\whisperjav-upgrade.exe --wheel-only
   ```
4. Check the version; it should print `WhisperJAV 1.9.5`:
   ```
   Scripts\whisperjav.exe --version
   ```

**Linux and macOS (installed from source)**

1. Open a terminal and activate the environment WhisperJAV is installed in, for example:
   ```
   source ~/.venv/whisperjav/bin/activate
   ```
   (`~/venvs/whisperjav` if you followed the macOS guide; `.venv` inside your `whisperjav` folder if
   the installer created the environment.)
2. Run:
   ```
   whisperjav-upgrade --wheel-only
   whisperjav --version
   ```

### Upgrade from 1.9.2 or older

Do a fresh install as described below. On Windows, install into the same folder; the installer
offers to remove the old version first.

### Fresh install: Windows

1. Download `WhisperJAV-1.9.5-Windows-x86_64.exe` from the
   [Releases page](https://github.com/meizhong986/WhisperJAV/releases/latest).
2. Run it and follow the steps. It detects your NVIDIA driver and installs the matching PyTorch build,
   so keep the internet connection on during installation.
3. Start WhisperJAV from the desktop shortcut.

### Fresh install: macOS (Apple Silicon)

Needs Xcode Command Line Tools, Homebrew, Python 3.10, 3.11 or 3.12, FFmpeg, PortAudio and Git
([macOS guide](https://github.com/meizhong986/WhisperJAV/blob/main/docs/en/guides/installation_mac_apple_silicon.md),
section 2). Then:

```
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
python3 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
bash installer/install_mac.sh
```

### Fresh install: Linux

Needs an NVIDIA driver 450 or newer (for GPU use), Python 3.10, 3.11 or 3.12 with `venv`, FFmpeg and
Git ([Linux guide](https://github.com/meizhong986/WhisperJAV/blob/main/docs/en/guides/installation_linux.md),
"Prerequisites"). Then:

```
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
bash installer/install_linux.sh
```

### Google Colab

Open the notebook from the README. It installs 1.9.5.

## Credits

- Angelholl reported the output-folder problem (#454).
- wahkeenet (#456) and a-pikachu (#447) reported the Colab problems.
- Untitled572 wrote the Ollama tag fix (PR #443) and the DeepSeek model change (PR #457).
