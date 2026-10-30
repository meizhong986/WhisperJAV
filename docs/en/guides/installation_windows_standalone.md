# WhisperJAV Windows Installer Guide

**Version:** 1.10.0
**Installer:** `WhisperJAV-1.10.0-Windows-x86_64.exe` (no administrator rights needed)

---

## Contents

1. [Before you install](#before-you-install)
2. [Installation steps](#installation-steps)
3. [What this install is made for](#what-this-install-is-made-for)
4. [First run](#first-run)
5. [Checking your installation](#checking-your-installation)
6. [Coming from 1.9.x](#coming-from-19x)
7. [Updating within 1.10](#updating-within-110)
8. [Running the installer again](#running-the-installer-again)
9. [Uninstalling](#uninstalling)
10. [Troubleshooting](#troubleshooting)
11. [Technical details](#technical-details)

---

## Before you install

### Your computer

| | Needed |
|---|---|
| Windows | Windows 10 or 11, 64-bit (x86-64). Windows on ARM (for example Snapdragon laptops) is not supported: the installer stops on its first screen. |
| RAM | 8 GB (16 GB recommended) |
| Free disk space | about 10 GB, for the install and the models it downloads on first use |
| Internet | while installing, and on the first run of each model |
| WebView2 | Microsoft Edge WebView2 Runtime, for the GUI (the installer offers it if it is missing) |

### Your graphics card

The installer makes this install for one accelerator (see [Accelerated or not](../getting-started/index.md#accelerated-or-not)):

- **An NVIDIA GPU that 1.10 supports** — GTX 16 series, RTX 20 series or newer, with driver 580 or newer: the install is made for the NVIDIA GPU (CUDA) by itself, without a question.
- **Anything else** — an AMD or Intel graphics card, an NVIDIA card older than 1.10 supports, an older driver, or no graphics card WhisperJAV can use: a window asks what the install should be made for.

To check your NVIDIA driver version, right-click the desktop, open the NVIDIA Control Panel and look at "Driver Version" in the bottom-left corner. To update it: [nvidia.com/drivers](https://www.nvidia.com/drivers), then restart the PC.

### Antivirus

Some antivirus programs block the installer's downloads. If that happens, add an exclusion for `%LOCALAPPDATA%\WhisperJAV`, or pause real-time scanning while installing.

### Users in China

The installer downloads from GitHub, PyPI, download.pytorch.org and Hugging Face. Keep your VPN on during the whole installation.

---

## Installation steps

1. Download `WhisperJAV-1.10.0-Windows-x86_64.exe` from the [releases page](https://github.com/meizhong986/WhisperJAV/releases).
2. Double-click it. You do not need to run it as administrator. If Windows SmartScreen warns, click **More info**, then **Run anyway**.
3. Accept the licence.
4. Choose **Just Me** (recommended; no administrator rights) or **All Users**.
5. Choose the folder. The default is `C:\Users\<you>\AppData\Local\WhisperJAV`. Keep the path short, and avoid spaces if you can.
6. Click **Install**.

After the files are unpacked, a console window runs the rest of the installation. It shows each phase:

| Phase | What happens |
|---|---|
| 1 | Checks: free disk space, the connection to PyPI, the Visual C++ runtime, WebView2 |
| 2 | Looks for an NVIDIA GPU and its driver |
| 2.5 | Decides what the install is made for; shows the window when it has to ask (see below) |
| 3–4.5 | Installs the package versions this release was tested with, then PyTorch for that accelerator |
| 5 | Installs WhisperJAV |
| 5.1 | NVIDIA GPU (CUDA) install only: places the CUDA 12 library balanced, fast and faster need; if this PC cannot load it, installs the CUDA 12.8 build of PyTorch instead |
| 5.5–5.8 | Creates the `WhisperJAV-GUI.exe` launcher, downloads the FireRedVAD speech detection model, checks the icons |
| 6 | Checks that the installed program starts |

At the end, a summary shows what was installed and what the install is made for, and a desktop shortcut "WhisperJAV v1.10.0" is created. The installation takes about 10 to 30 minutes, depending on your connection.

---

## What this install is made for

### A supported NVIDIA GPU

No question is asked. The install is made for the NVIDIA GPU (CUDA 13.0). If this PC cannot load the CUDA 12 library that balanced, fast and faster need, the installer says so in a framed message and installs the CUDA 12.8 build instead.

### No supported NVIDIA GPU: the window

When the installer finds no NVIDIA GPU that 1.10 supports, a window titled **"WhisperJAV — no supported NVIDIA GPU found"** asks:

> WhisperJAV found no NVIDIA graphics card that version 1.10 supports (graphics found: …). What should this install be made for?

| Choice | The install is made for |
|---|---|
| AMD graphics card | GPU through Vulkan |
| Intel graphics card | GPU through Vulkan |
| NVIDIA graphics card older than WhisperJAV 1.10 supports | GPU through Vulkan |
| I have my own PyTorch for my GPU (advanced) | Own PyTorch |
| None, or use the CPU only | CPU only (not accelerated) |

The window explains:

> With an AMD, Intel or older NVIDIA card, the GGUF pipelines (Parakeet-CrispASR and Cohere-CrispASR) ask for your graphics card through Vulkan. This is experimental in 1.10.0: it has not been tested on AMD or Intel graphics cards, and we would like to hear how it works for you. All other pipelines run on the CPU (not accelerated), several times slower than on a supported NVIDIA GPU. With your own PyTorch, WhisperJAV installs everything except PyTorch, and you add your PyTorch build afterwards (the command is shown at the end). With the CPU only, every pipeline runs on the CPU.

Without an answer, the CPU-only version is installed after 20 seconds. **Stop** ends the installer without installing anything.

GPU through Vulkan needs the Vulkan runtime (`vulkan-1.dll`), which comes with your graphics driver. If it is missing, the installer says so and installs the CPU-only version; update the graphics driver and run the installer again to change that.

### An NVIDIA GPU that 1.10 cannot use

The window starts with the reason, and **Stop** is pre-selected, with no time limit:

- **An older card** (below GTX 16 / RTX 20): "Your graphics card, … (compute capability …), is older than WhisperJAV 1.10 supports. 1.10 needs a GTX 16-series, RTX 20-series or newer card." and "WhisperJAV 1.9.5 still supports this card: https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5." If you are sure the card is supported, **My card is identified wrongly** shows where to report it, then stops.
- **An older driver** (below 580): "Your NVIDIA driver (…) is older than WhisperJAV 1.10 needs (580 or newer). Update it from https://www.nvidia.com/drivers, restart the PC, and run this installer again to get the NVIDIA GPU version. Or choose below."
- **A card Windows lists but WhisperJAV cannot reach** — the driver may be missing or not working, or, on a laptop with two graphics chips, the NVIDIA chip may be switched off: install or repair the NVIDIA driver (580 or newer), restart the PC and run the installer again. Or choose below.

### Own PyTorch

WhisperJAV installs everything except PyTorch, and never installs, replaces or removes PyTorch in this install at install or upgrade. Running the installer again replaces the whole install folder, PyTorch included: the window keeps Own PyTorch and shows the command again. At the end, a window shows the one command to run, with WhisperJAV's Python filled in:

```
"<install folder>\python.exe" -m pip install torch torchaudio --index-url <the address of your PyTorch build>
```

PyTorch 2.11 is recommended. The same command is saved in `<install folder>\OWN_PYTORCH_NEXT_STEP.txt`. Until you have run it, every WhisperJAV run stops at its start and shows the command again.

### Where the install's choice is kept

The installer writes the accelerator into `whisperjav_stack.txt` in the install folder. Every run, `whisperjav-upgrade` and every later run of the installer follow it.

### When the installer stops on purpose

If you choose Stop, a closing dialog says: "WhisperJAV was not installed. The installer stopped for the reason shown in its console window." The reason is also saved in `INSTALLATION_STOPPED_v1.10.0.txt` in the install folder.

---

## First run

Double-click the **WhisperJAV v1.10.0** shortcut, or `WhisperJAV-GUI.exe` in the install folder.

The first time a pipeline needs a model, WhisperJAV downloads it (for example about 3 GB for Whisper large-v3, 674 MB for Parakeet-CrispASR). Later runs reuse it.

At the start of each run, WhisperJAV checks the install:

- **NVIDIA GPU (CUDA) install, GPU usable:** nothing is shown; the run goes on.
- **NVIDIA GPU (CUDA) install, GPU not usable now** (for example after a driver change): the run does not start. A box names the reason and the fix. To run on the CPU this time, add `--device cpu`; in the GUI, the dialog **"The NVIDIA GPU cannot be used"** offers **Run on the CPU this time**.
- **Every other install:** a box says what the install is made for and where each part runs on this computer, then the run goes on.

Basic workflow: add files, choose a mode (Balanced is a good start), click **Start**. The subtitles are written to the output folder you chose.

---

## Checking your installation

Open a Command Prompt (press Start, type `cmd`, press Enter) and run:

```cmd
cd /d "%LOCALAPPDATA%\WhisperJAV"
Scripts\whisperjav.exe --check
```

(Use your install folder if you chose another one.)

It lists Python, PyTorch, the accelerator of the install, where each part runs, FFmpeg, CrispASR and the models. On an NVIDIA GPU (CUDA) install, for example:

```
✓ PyTorch: PyTorch 2.11.0+cu130: NVIDIA GPU (CUDA 13.0), NVIDIA GeForce RTX 3060
✓ CrispASR: 0.8.44+cuda (CUDA build), GGUF pipelines: asks for the NVIDIA GPU (CUDA)
ℹ Accelerator: Accelerator of this install: NVIDIA GPU (CUDA 13.0).
ℹ CTranslate2: CTranslate2 (balanced, fast, faster): NVIDIA GPU (CUDA)
```

`Scripts\whisperjav.exe --version` prints the version: `WhisperJAV 1.10.0`.

The installation log is `install_log_v1.10.0.txt` in the install folder.

---

## Coming from 1.9.x

Install 1.10.0 fresh: 1.10.0 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to 1.10.0. If you install into the folder of your old version, the installer finds it and offers to replace it: everything in that folder is removed first. Your settings in `%APPDATA%\WhisperJAV` and your downloaded models stay.

---

## Updating within 1.10

Later 1.10 releases can be installed with `whisperjav-upgrade`, which keeps the accelerator of your install and its PyTorch:

1. Open a Command Prompt.
2. `cd /d "%LOCALAPPDATA%\WhisperJAV"` (or your install folder).
3. `Scripts\whisperjav-upgrade.exe`
4. Check the version: `Scripts\whisperjav.exe --version`

Do not update with a plain `pip install -U whisperjav…`: on Windows, PyPI serves the CPU-only build of PyTorch, so it can replace your GPU build without a word.

---

## Running the installer again

Run the installer again into the same folder to repair the install or to change what it is made for. A window titled **"WhisperJAV — this install's accelerator"** says "This install is made for: …. Keep it, or choose another." The current accelerator is pre-selected; without an answer, the install is kept as it is after 20 seconds. If the NVIDIA GPU cannot be used at that moment, the reason is shown above the choice. The installer removes the old install folder first (your settings and downloaded models stay) and keeps only its record of the accelerator; on an Own PyTorch install, add your PyTorch again with the command shown at the end.

---

## Uninstalling

Open Windows **Settings > Apps**, find **WhisperJAV** and choose **Uninstall**. The downloaded models stay in `%USERPROFILE%\.cache` (Hugging Face and Whisper caches) and your settings in `%APPDATA%\WhisperJAV`; delete those folders by hand to free the space.

---

## Troubleshooting

**"No supported NVIDIA GPU found" although you have an NVIDIA card.** Read the first line of the window: it names the reason (old card, old driver, or a card WhisperJAV cannot reach). Update or repair the driver, restart the PC and run the installer again. On a laptop with two graphics chips, make sure the NVIDIA chip is switched on.

**The run stops: "THE NVIDIA GPU CANNOT BE USED".** The install is made for the NVIDIA GPU, and the GPU cannot be used now. The box names the reason and the fix. `--device cpu` runs that one job on the CPU.

**Processing is slow.** Look at the box at the start of the run, or run `whisperjav --check`: they say where each part runs. On a CPU-only install, every pipeline runs on the CPU, several times slower than on a GPU.

**The GGUF pipelines stop: "CrispASR could not be loaded".** With the GPU-through-Vulkan or Own PyTorch install, this usually means the Vulkan runtime (`vulkan-1.dll`) is missing: update the graphics driver, or run the installer again and choose CPU only.

**Downloads fail or time out.** Check the connection; behind a firewall, allow `python.exe`, `uv.exe` and `git.exe`. The installer retries GitHub downloads with longer timeouts. In China, keep your VPN on for the whole installation.

**The GUI does not open, or the window is blank.** Install WebView2 from [Microsoft](https://go.microsoft.com/fwlink/p/?LinkId=2124703). To see error messages, run `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.webview_gui.main` in a Command Prompt.

**The installation failed.** Open `install_log_v1.10.0.txt` in the install folder: it names the phase that failed. `INSTALLATION_FAILED_v1.10.0.txt` holds the reason. Run the installer again; it offers to replace the partial install. If it keeps failing, open an issue at [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) with the log attached.

---

## Technical details

### Silent installation

```cmd
cmd /C START /WAIT WhisperJAV-1.10.0-Windows-x86_64.exe /S /D=C:\WhisperJAV
```

`/S` installs without the installer's screens; `/D=` sets the folder and must come last. `WhisperJAV-1.10.0-Windows-x86_64.exe /?` lists every option. The window of [What this install is made for](#what-this-install-is-made-for) still opens when the installer has to ask: on a PC without a supported NVIDIA GPU it installs the CPU-only version after 20 seconds without an answer; with an NVIDIA GPU that 1.10 cannot use, it waits for an answer.

### Files in the install folder

| File | What it is |
|---|---|
| `install_log_v1.10.0.txt` | the installation log |
| `whisperjav_stack.txt` | what the install is made for |
| `OWN_PYTORCH_NEXT_STEP.txt` | Own PyTorch install only: the command that adds your PyTorch |
| `INSTALLATION_STOPPED_v1.10.0.txt` | only when the installer stopped on purpose: the reason |
| `INSTALLATION_FAILED_v1.10.0.txt` | only when the installation failed: the reason |
| `WhisperJAV-GUI.exe` | the GUI launcher |

### Addresses the installer downloads from

| Address | For |
|---|---|
| `pypi.org`, `files.pythonhosted.org` | Python packages |
| `download.pytorch.org` | PyTorch |
| `github.com` | CrispASR's GPU builds and packages installed from GitHub |
| `huggingface.co` | models (installation and first run) |
| `aka.ms`, `go.microsoft.com` | the Visual C++ runtime and WebView2, if missing |
