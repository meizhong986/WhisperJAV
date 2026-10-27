# WhisperJAV Linux Installation Guide

**Version:** 1.10.0

Nobody in the project has run WhisperJAV 1.10.0 on Linux yet. Please tell us how it works for you: https://github.com/meizhong986/WhisperJAV/issues

---

## Contents

1. [Your computer](#your-computer)
2. [System packages](#system-packages)
3. [Python 3.13](#python-313)
4. [NVIDIA driver](#nvidia-driver)
5. [Installation](#installation)
6. [What the install is made for](#what-the-install-is-made-for)
7. [Check and run](#check-and-run)
8. [Updating](#updating)
9. [Troubleshooting](#troubleshooting)
10. [Uninstalling](#uninstalling)

---

## Your computer

| | WhisperJAV 1.10.0 needs |
|---|---|
| Processor | x86-64 or ARM64 |
| Linux | glibc 2.35 or newer: Ubuntu 22.04 or newer, Debian 12 or newer, Fedora 36 or newer, openSUSE Leap 15.6, and distributions built on them; rolling distributions such as Arch |
| Python | 3.13 |
| RAM | 8 GB (16 GB recommended) |
| Disk | about 10 GB for the install and the models it downloads on first use |
| NVIDIA GPU (optional) | GTX 16 / RTX 20 series or newer, driver 570 or newer |

**Not supported** (the installer says so and installs nothing): glibc older than 2.35 — Ubuntu 20.04, Debian 11, openSUSE Leap 15.5 (glibc 2.31); Red Hat Enterprise Linux 9, Rocky Linux 9, AlmaLinux 9 (glibc 2.34). To see your glibc version: `ldd --version`.

**AMD and Intel GPUs on Linux:** in 1.10.0, the GGUF pipelines (Parakeet-CrispASR, Cohere-CrispASR) run on the CPU on Linux, because CrispASR publishes no Linux GPU package for AMD or Intel yet. With your own PyTorch build for your GPU (for example ROCm for an AMD GPU), the PyTorch pipelines can ask for it — see [Own PyTorch](#own-pytorch).

---

## System packages

Install these before the WhisperJAV installer; pip cannot install them.

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential ffmpeg git
sudo apt-get install -y libsndfile1 libsndfile1-dev
# TEN VAD speech segmenter
sudo apt-get install -y libc++1 libc++abi1
# PyAudio, which auditok scene detection uses
sudo apt-get install -y portaudio19-dev
# GUI: Ubuntu 24.04+ ships WebKit 4.1, 22.04 ships 4.0; this installs whichever is there
sudo apt-get install -y libgtk-3-dev
sudo apt-get install -y libwebkit2gtk-4.1-dev gir1.2-webkit2-4.1 \
  || sudo apt-get install -y libwebkit2gtk-4.0-dev gir1.2-webkit2-4.0
```

### Fedora

```bash
sudo dnf install -y python3-pip python3-devel gcc gcc-c++ ffmpeg git
sudo dnf install -y libsndfile libsndfile-devel portaudio-devel
# GUI
sudo dnf install -y webkit2gtk4.1-devel gtk3-devel
```

On Fedora, the full `ffmpeg` package comes from RPM Fusion; enable RPM Fusion first if `dnf` does not find `ffmpeg`.

### Arch Linux / Manjaro

```bash
sudo pacman -S --noconfirm python python-pip base-devel ffmpeg git libsndfile portaudio
# GUI
sudo pacman -S --noconfirm webkit2gtk gtk3
```

### GUI backend (all distributions)

The system packages above provide WebKit, but they are **not enough on their own**. pywebview ships **no Linux backend by default**; GTK and Qt are both opt-in. And apt/dnf packages land system-wide, so a virtual environment created without `--system-site-packages` cannot see the `gi` (PyGObject) bindings.

Following the distribution instructions alone therefore still ends in `ModuleNotFoundError: No module named 'gi'` when you launch `whisperjav-gui` (#366). Install a backend *into the virtual environment* as well:

```bash
# Qt backend - simplest, needs no system GTK bindings
pip install "pywebview[qt]"

# or, to use GTK instead, expose the system bindings to the venv:
#   python3.13 -m venv --system-site-packages whisperjav-env
```

This step is Linux-only. Windows and macOS need nothing extra.

---

## Python 3.13

WhisperJAV 1.10.0 runs on Python 3.13 only. `python3 --version` shows your system Python; if it is not 3.13:

- **Ubuntu 22.04 / 24.04:** install 3.13 from the deadsnakes PPA:
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
  ```
- **Fedora:** `sudo dnf install -y python3.13 python3.13-devel` where your release offers it.
- **Any distribution:** `uv` can download Python 3.13 for you. Install `uv` first (`pip install --user uv`, or see [uv's installation page](https://docs.astral.sh/uv/getting-started/installation/)), then `uv python install 3.13`, then make the environment with `uv venv --python 3.13 ~/.venv/whisperjav` (instead of the `python3.13 -m venv` line below).

Then make and activate a virtual environment (Debian 12+ and Ubuntu 24.04+ do not allow installing into the system Python):

```bash
python3.13 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
```

---

## NVIDIA driver

For the NVIDIA GPU (CUDA) install you need the NVIDIA driver, 570 or newer; you do not need the CUDA Toolkit (PyTorch brings its own CUDA runtime). On Linux, 1.10.0 installs PyTorch for CUDA 12.8.

```bash
nvidia-smi          # shows the driver version, or fails without a driver
```

To install the driver:

- **Ubuntu:** `sudo ubuntu-drivers autoinstall`, or `sudo apt-get install -y nvidia-driver-570`, then `sudo reboot`.
- **Fedora:** with RPM Fusion enabled, `sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda`, then `sudo reboot`.
- **Arch:** `sudo pacman -S nvidia nvidia-utils`, then `sudo reboot`.
- **Data-centre GPUs (Ubuntu):** `sudo apt-get install -y nvidia-headless-570-server nvidia-utils-570-server`, then `sudo reboot`.

---

## Installation

With the virtual environment active:

```bash
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

`./installer/install_linux.sh` checks first that the computer is supported, then runs `python3 install.py` with the options you give it. `python3 install.py` directly does the same.

| Option | What it does |
|---|---|
| `--accelerator nvidia` | made for the NVIDIA GPU (CUDA 12.8 on Linux) |
| `--accelerator own-pytorch` | keep the PyTorch already in this environment; install everything else — experimental |
| `--accelerator cpu` | made for the CPU only (not accelerated) |
| `--no-speech-enhancement` | leave out the speech enhancement packages |
| `--minimal` | command line only: leaves out the GUI and, among others, the packages the WhisperSeg speech segmenter needs |
| `--dev` | editable install, for development |
| `--local` | install into `.venv` in the source folder even if an environment is active |

(`--accelerator vulkan` is for Windows only: CrispASR has no Vulkan package for Linux.)

The installer installs exactly the package versions this release was tested with (`uv sync` from the lock file) for the accelerator of the install, and makes a GUI launcher, `WhisperJAV.sh`, in the source folder. Without an active environment it creates `.venv` in the source folder.

Users in China: the installer downloads from GitHub, PyPI, download.pytorch.org and Hugging Face. Keep your VPN on during the whole installation.

---

## What the install is made for

Without `--accelerator`:

- **A supported NVIDIA GPU** (driver 570 or newer): made for the NVIDIA GPU (CUDA 12.8), without a question.
- **Otherwise** the installer asks:

  ```
  No supported NVIDIA GPU found. What should this install be made for? 1 = my own PyTorch (already installed in this Python environment, for example ROCm for an AMD GPU); 2 = CPU only (not accelerated); 3 = stop. Type 1, 2 or 3 and press Enter:
  ```

  There is no time limit. With an NVIDIA GPU that 1.10 cannot use (an older card or driver), the reason is shown first. Without a console to answer (an automated run), the CPU-only install is made — or, with an NVIDIA GPU 1.10 cannot use, the installer stops.

| Install made for | GGUF pipelines | Other pipelines |
|---|---|---|
| NVIDIA GPU (CUDA), x86-64 | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) |
| NVIDIA GPU (CUDA), ARM64 | CPU (not accelerated) | the PyTorch pipelines on the NVIDIA GPU; balanced, fast and faster on the CPU |
| Own PyTorch | CPU (not accelerated) | fidelity, transformers, ChronosJAV and the speech enhancers ask for the GPU your PyTorch reports (not tested); balanced, fast and faster on the CPU unless CTranslate2, their engine, sees an NVIDIA GPU |
| CPU only | CPU (not accelerated) | CPU (not accelerated), several times slower |

The WhisperSeg speech segmenter runs on the CPU in every install.

### Own PyTorch

Install your PyTorch build into the environment first (PyTorch 2.11 is recommended; for an AMD GPU, the ROCm build from pytorch.org), then:

```bash
python3 install.py --accelerator own-pytorch
```

WhisperJAV installs everything except PyTorch, and never installs, replaces or removes PyTorch in this environment — not at install, upgrade or repair. Without a PyTorch in the active environment, the installer stops and installs nothing. This is experimental in 1.10.0: we could not test it; please tell us how it works: https://github.com/meizhong986/WhisperJAV/issues

The choice is recorded in `whisperjav_stack.txt` in the environment's folder. Running the installer again in the same environment keeps it; `--accelerator` chooses another.

---

## Check and run

```bash
whisperjav --check          # accelerator, where each part runs, CrispASR, models
whisperjav-gui              # or ./WhisperJAV.sh in the source folder
whisperjav video.mp4
```

At the start of each run, WhisperJAV says where each part runs (on an NVIDIA GPU (CUDA) install whose GPU is usable, it says nothing). `--device cpu` runs one job on the CPU.

**On a server without a display:** use the command line; the GUI packages are not needed for it.

---

## Updating

Within 1.10, update the source code and run the installer again, with the same Python environment active as when you installed (none, if the install went into `.venv`); it keeps the accelerator of the environment:

```bash
git fetch --tags
git checkout v1.10.x      # the release you want
python3 install.py
```

A 1.9.x environment cannot be updated to 1.10.0: 1.10.0 needs Python 3.13. Make a new Python 3.13 environment and install there.

---

## Troubleshooting

**"WhisperJAV 1.10.0 does not support this Linux".** Its glibc is older than 2.35; the message names the version. Nothing was installed. Use a newer distribution release.

**The run stops: "THE NVIDIA GPU CANNOT BE USED".** The install is made for the NVIDIA GPU and the GPU cannot be used now. The box names the reason (for example the driver) and the fix; `--device cpu` runs that one job on the CPU. `nvidia-smi` shows whether the driver is loaded.

**`ModuleNotFoundError: No module named 'gi'`.** See [GUI backend](#gui-backend-all-distributions).

**Library not found (`libsndfile`, `portaudio`, `libc++`).** Install the [system packages](#system-packages) and run the installer again.

**Git timeouts.** Run the installer again; it retries with longer Git timeouts. Behind a proxy, set `https_proxy` before running it.

**Anything else.** Open an issue at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) with the installer's output.

---

## Uninstalling

```bash
rm -rf ~/.venv/whisperjav          # the virtual environment (or the .venv in the source folder)
rm -rf ~/WhisperJAV                # the source folder, wherever you cloned it
rm -rf ~/.cache/huggingface ~/.cache/whisper   # downloaded models (optional; other programs may use them too)
```

Your settings are in `~/.config/WhisperJAV`.
