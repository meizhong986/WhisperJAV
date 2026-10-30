# WhisperJAV Installation Guide for Apple Silicon Macs

**Version:** 1.10.0

On an Apple silicon Mac, WhisperJAV 1.10.0 is installed for the **Apple GPU (Metal)** by itself: the GPU inside the Apple chip. Some pipelines use it, others run on the CPU (see [Where each pipeline runs](#4-where-each-pipeline-runs)). There is no `.exe`-style installer for the Mac; you install from the source code with one script.

Nobody in the project has run WhisperJAV 1.10.0 on a Mac yet. The parts marked "not yet tested on a Mac" are there and should work, but we could not check them. Please tell us how it works for you: https://github.com/meizhong986/WhisperJAV/issues

---

## Contents

1. [Your Mac](#1-your-mac)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Where each pipeline runs](#4-where-each-pipeline-runs)
5. [Running WhisperJAV](#5-running-whisperjav)
6. [Updating](#6-updating)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. Your Mac

| | WhisperJAV 1.10.0 needs |
|---|---|
| Chip | Apple silicon (M1 or newer) |
| macOS | 14 (Sonoma) or newer |
| Memory | 8 GB (16 GB recommended) |
| Disk | about 10 GB for the install and the models it downloads on first use |
| Python | 3.13, for Apple silicon (arm64) |

**Not supported** (the installer says so and installs nothing): Intel Macs, because PyTorch 2.11, which WhisperJAV 1.10.0 needs, is not made for them; macOS 13 and older, because ONNX Runtime 1.29 needs macOS 14 or newer. A Python running under Rosetta (as an Intel program) on an Apple silicon Mac is refused too: use an arm64 Python, for example from Homebrew.

---

## 2. Prerequisites

Open **Terminal** and run these one by one.

1. **Xcode Command Line Tools:**
   ```bash
   xcode-select --install
   ```
2. **Homebrew** (see [brew.sh](https://brew.sh)):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. **Python 3.13, FFmpeg, PortAudio and Git:**
   ```bash
   brew install python@3.13 ffmpeg portaudio git
   ```
   PortAudio is needed by PyAudio, which the auditok scene detector uses.

---

## 3. Installation

```bash
# A virtual environment with Python 3.13 (Homebrew's Python does not allow installing into it directly)
python3.13 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
python3 --version          # should print Python 3.13.x

# The source code of the release
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV

# The installer
chmod +x installer/install_mac.sh
./installer/install_mac.sh
```

The script first checks that the Mac is supported (Apple silicon, macOS 14 or newer), then the Xcode tools, Homebrew, Python, FFmpeg and PortAudio, and then runs `install.py`. It installs exactly the package versions this release was tested with (`uv sync` from the lock file): PyTorch with Apple GPU support, and CrispASR's macOS build. It prints:

> Apple silicon Mac: installing for the Apple GPU (Metal) — PyTorch with Apple GPU support, and CrispASR's macOS build.

At the end it creates `WhisperJAV.command` in the `WhisperJAV` folder: double-click it in Finder to start the GUI (you can drag it to the Dock).

Users in China: the installer downloads from GitHub, PyPI, download.pytorch.org and Hugging Face. Keep your VPN on during the whole installation.

**Options** (passed on to `install.py`):

| Option | What it does |
|---|---|
| `--accelerator own-pytorch` | keep the PyTorch already in this environment; install everything else — experimental |
| `--no-speech-enhancement` | leave out the speech enhancement packages |
| `--minimal` | command line only: leaves out the GUI and, among others, the packages the WhisperSeg speech segmenter needs |
| `--dev` | editable install, for development |

`--accelerator cpu` (or the old `--cpu-only`) gives the same Apple GPU install: "On an Apple silicon Mac, WhisperJAV is installed for the Apple GPU (Metal), which includes the CPU. To run one job on the CPU, add --device cpu to that run."

The install's accelerator is recorded in `whisperjav_stack.txt` in the environment's folder.

---

## 4. Where each pipeline runs

| Pipeline | On an Apple silicon Mac |
|---|---|
| **transformers** | the Apple GPU (Metal) — **recommended on a Mac** |
| **fidelity** | asks PyTorch for the Apple GPU (Metal); not yet tested on a Mac. If it fails, use the transformers pipeline. |
| **Parakeet-CrispASR, Cohere-CrispASR** (GGUF) | ask for the Apple GPU (Metal); not yet tested on a Mac |
| **balanced, fast, faster** | the CPU (not accelerated), several times slower: their engine (CTranslate2) cannot use the Apple GPU |
| **ChronosJAV** (Qwen3-ASR, anime-whisper) | the CPU (not accelerated) in 1.10.0: its ASR model does not use the Apple GPU yet |
| WhisperSeg speech segmenter | the CPU, in every install |

When the Apple GPU can be used, nothing is shown at the start of a run; each pipeline says in the console where it runs as it starts (for example, balanced, fast and faster say they run on the CPU). If the Apple GPU cannot be used, a box at the start says where each part runs. The installer ends with:

> On an Apple silicon Mac, the transformers pipeline is recommended: it asks for the Apple GPU (Metal).

---

## 5. Running WhisperJAV

Activate the environment first (`source ~/venvs/whisperjav/bin/activate`), or double-click `WhisperJAV.command` for the GUI.

```bash
whisperjav-gui                                  # the GUI
whisperjav video.mp4 --mode transformers        # on the Apple GPU (recommended)
whisperjav video.mp4                            # balanced, on the CPU
whisperjav --check                              # accelerator, where each part runs, CrispASR, models
whisperjav video.mp4 --device cpu               # this run on the CPU
```

The GUI uses macOS's own WebKit; nothing else needs installing for it.

---

## 6. Updating

Within 1.10, update the source code and run the installer again:

```bash
source ~/venvs/whisperjav/bin/activate
cd WhisperJAV
git fetch --tags
git checkout v1.10.x      # the release you want
./installer/install_mac.sh
```

A 1.9.x environment cannot be updated to 1.10.0: 1.10.0 needs Python 3.13. Make a new Python 3.13 environment and install there.

---

## 7. Troubleshooting

**"WhisperJAV 1.10.0 does not support Intel Macs" / "does not support macOS …".** The Mac is outside the supported list; nothing was installed.

**"Python 3.x is not supported".** Activate the Python 3.13 environment (section 3) and run the script again.

**`externally-managed-environment`.** You ran it with Homebrew's Python directly. Make and activate a virtual environment (section 3).

**PyAudio / PortAudio errors.** `brew install portaudio`, then run the installer again.

**FFmpeg not found.** `brew install ffmpeg`, then open a new Terminal.

**Slow processing.** Most pipelines run on the CPU on a Mac (section 4). Use the transformers pipeline for the Apple GPU.

**fidelity fails on the Apple GPU.** Use the transformers pipeline, or add `--device cpu` to run fidelity on the CPU.

**The GUI does not open.** Check that `python3 -c "import webview"` works in the environment; if not, run `xcode-select --install` and the installer again. The command line works without the GUI.

**Anything else.** Open an issue at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) with the installer's output.

Your settings are in `~/Library/Application Support/WhisperJAV`.
