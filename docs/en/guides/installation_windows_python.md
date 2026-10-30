# WhisperJAV 1.10.0 — Windows Installation from Source (Python)

This guide installs WhisperJAV from its source code into a Python environment you manage. If you just want it to work, use the [Windows Installer](installation_windows_standalone.md) instead.

---

## Contents

1. [What you need](#1-what-you-need)
2. [Get the source code](#2-get-the-source-code)
3. [Choose the Python environment](#3-choose-the-python-environment)
4. [Run the installer](#4-run-the-installer)
5. [What the install is made for](#5-what-the-install-is-made-for)
6. [Check and run](#6-check-and-run)
7. [Updating](#7-updating)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. What you need

| Software | Version | Where to get it |
|---|---|---|
| Windows | 10 or 11, 64-bit (x86-64); Windows on ARM is not supported | — |
| Python | **3.13** (other versions are refused) | [python.org](https://www.python.org/downloads/) or conda/Miniconda |
| Git | any recent version | [git-scm.com](https://git-scm.com/download/win) |
| FFmpeg | 6.x or 7.x, on your PATH | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), or `winget install --id=Gyan.FFmpeg -e` |
| Visual C++ Redistributable | 2015–2022 (x64) | [microsoft.com](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| WebView2 Runtime | any, for the GUI | [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703) |
| NVIDIA driver | 580 or newer, for the NVIDIA GPU (CUDA) install | [nvidia.com/drivers](https://www.nvidia.com/drivers) |

Check them in a Command Prompt:

```cmd
python --version
git --version
ffmpeg -version
nvidia-smi
```

`nvidia-smi` fails without an NVIDIA GPU; that is fine. The installer installs `uv` (the package installer it uses) itself if it is missing.

Users in China: the installer downloads from GitHub, PyPI, download.pytorch.org and Hugging Face. Keep your VPN on during the whole installation.

---

## 2. Get the source code

```cmd
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
```

Keep the folder path short, in plain Western letters, without spaces if you can.

---

## 3. Choose the Python environment

The installer installs into the Python environment that is active when you run it:

- **A venv:**
  ```cmd
  python -m venv whisperjav-env
  whisperjav-env\Scripts\activate
  ```
- **A conda environment:**
  ```cmd
  conda create -n whisperjav python=3.13 -y
  conda activate whisperjav
  ```
- **No active environment:** the installer creates `.venv` inside the `WhisperJAV` folder and installs there. `--local` does the same even when an environment is active.

Never install into your system-wide Python.

---

## 4. Run the installer

From the `WhisperJAV` folder, with the environment active:

```cmd
installer\install_windows.bat
```

or, the same:

```cmd
python install.py
```

| Option | What it does |
|---|---|
| `--accelerator nvidia` | made for the NVIDIA GPU (CUDA) |
| `--accelerator vulkan` | made for GPU through Vulkan (AMD, Intel or older NVIDIA card) — experimental |
| `--accelerator own-pytorch` | keep the PyTorch already in this environment; install everything else — experimental |
| `--accelerator cpu` | made for the CPU only (not accelerated) |
| `--no-speech-enhancement` | leave out the speech enhancement packages |
| `--minimal` | command line only: leaves out the GUI and, among others, the packages the WhisperSeg speech segmenter needs |
| `--dev` | editable install, for development |
| `--local` | install into `.venv` in the source folder even if an environment is active |
| `--log-file PATH` | write the log to PATH |

The installer first checks that the computer is supported, then installs exactly the package versions this release was tested with (`uv sync` from the lock file), for the accelerator of the install. It never rewrites the lock file.

---

## 5. What the install is made for

Without `--accelerator`:

- **A supported NVIDIA GPU** (GTX 16 / RTX 20 or newer, driver 580 or newer): made for the NVIDIA GPU (CUDA 13.0), without a question.
- **Otherwise** the installer asks the same question as the Windows installer's window:

  ```
  Type 1 = AMD, 2 = Intel, 3 = older NVIDIA card, 4 = my own PyTorch, 5 = CPU only, 6 = stop, then Enter:
  ```

  1, 2 and 3 make the install for GPU through Vulkan; it needs the Vulkan runtime (`vulkan-1.dll`), which comes with your graphics driver — without it, the CPU-only install is made. With an NVIDIA GPU that 1.10 cannot use (an older card or driver), the reason is shown first. There is no time limit. 6 stops without installing anything.
- **No console to answer** (an automated run): the CPU-only install without a supported NVIDIA GPU; a stop with an NVIDIA GPU that 1.10 cannot use.

What each accelerator gives each pipeline: [Accelerated or not](../getting-started/index.md#accelerated-or-not).

**Own PyTorch.** Install your PyTorch build into the environment first (PyTorch 2.11 is recommended), then run `python install.py --accelerator own-pytorch`. WhisperJAV installs everything except PyTorch and never installs, replaces or removes PyTorch in this environment. Without a PyTorch in the active environment, the installer stops and installs nothing.

The choice is recorded in `whisperjav_stack.txt` in the environment's folder. Running the installer again in the same environment keeps it; `--accelerator` chooses another.

---

## 6. Check and run

With the environment active:

```cmd
whisperjav --check
whisperjav-gui
whisperjav video.mp4
```

`whisperjav --check` lists the accelerator of the install, where each part runs, CrispASR and the models. At the start of each run, WhisperJAV says where each part runs (on an NVIDIA GPU (CUDA) install whose GPU is usable, it says nothing). `--device cpu` runs one job on the CPU.

---

## 7. Updating

Within 1.10, update the source code and run the installer again, with the same Python environment active as when you installed (none, if the install went into `.venv`); it keeps the accelerator of the environment:

```cmd
git fetch --tags
git checkout v1.10.x
python install.py
```

(`v1.10.x` stands for the release you want.) A 1.9.x environment cannot be updated to 1.10.0: 1.10.0 needs Python 3.13. Make a new Python 3.13 environment and install there.

Do not update with a plain `pip install -U whisperjav…`: on Windows, PyPI serves the CPU-only build of PyTorch, so it can replace your GPU build without a word.

---

## 8. Troubleshooting

**"WhisperJAV 1.10.0 does not support …".** The computer is outside the [supported list](../getting-started/index.md#supported-computers); the message says why. Nothing was installed.

**The installer refuses your Python.** WhisperJAV 1.10.0 needs Python 3.13. Make a 3.13 environment (section 3).

**`--accelerator own-pytorch` stops: "none was found".** Install your PyTorch and torchaudio into the active environment first, then run the installer again.

**The run stops: "THE NVIDIA GPU CANNOT BE USED".** The install is made for the NVIDIA GPU and the GPU cannot be used now (for example a driver change). The box names the reason and the fix; `--device cpu` runs that one job on the CPU.

**The GGUF pipelines stop: "CrispASR could not be loaded".** With the GPU-through-Vulkan or Own PyTorch install on Windows, this usually means the Vulkan runtime (`vulkan-1.dll`) is missing: update the graphics driver, or run `python install.py --accelerator cpu`.

**`ffmpeg` not found.** Put FFmpeg's `bin` folder on your PATH and open a new Command Prompt.

**The GUI does not open.** Install WebView2. To see the error, run `python -m whisperjav.webview_gui.main`.

**Anything else.** The log names the step that failed. Open an issue at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) with it.
