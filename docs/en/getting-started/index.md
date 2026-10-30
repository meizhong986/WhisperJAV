# Getting Started

WhisperJAV 1.10.0 runs on Windows, Linux and Apple silicon Macs. Choose the way of installing that fits your computer.

## Which install method should I use?

| If you... | Use this |
|-----------|----------|
| Just want it to work (Windows) | [Windows Installer](../guides/installation_windows_standalone.md) |
| Manage your own Python environment on Windows | [Windows Python Install](../guides/installation_windows_python.md) |
| Have an Apple silicon Mac (M1 or newer) | [macOS Guide](../guides/installation_mac_apple_silicon.md) |
| Run Linux | [Linux Guide](../guides/installation_linux.md) |

**Coming from 1.9.x?** Install 1.10.0 fresh. 1.10.0 needs Python 3.13, so `whisperjav-upgrade` cannot take a 1.9.x install to 1.10.0. Your settings in the user settings folder are kept.

**Google Colab and Kaggle** are not supported by 1.10.0: it is made and tested for installs on your own computer only. The Colab expert notebook keeps installing 1.9.5.

## Accelerated or not

A GPU — a graphics card, or the GPU inside an Apple chip — does the heavy work of turning speech into text many times faster than the CPU. Each WhisperJAV install is made for one accelerator, which you choose when you install: NVIDIA GPU (CUDA), GPU through Vulkan (AMD, Intel or older NVIDIA card, Windows), Apple GPU (Metal), your own PyTorch, or CPU only (not accelerated). At the start of a run, WhisperJAV says where each part runs whenever the install's GPU is not in use. Not accelerated works, but several times slower.

What each computer gets:

| Your computer | The install is made for | GGUF pipelines (Parakeet-CrispASR, Cohere-CrispASR) | All other pipelines |
|---|---|---|---|
| Windows PC with an NVIDIA GPU that 1.10 supports (GTX 16 / RTX 20 or newer), driver 580 or newer | NVIDIA GPU (CUDA), chosen by itself | NVIDIA GPU (CUDA) | NVIDIA GPU (CUDA) |
| Windows PC with an AMD or Intel graphics card, or an NVIDIA card older than 1.10 supports | GPU through Vulkan, your own PyTorch, or CPU only — the installer asks | GPU through Vulkan: your graphics card (experimental, not tested) | CPU (not accelerated), several times slower |
| Windows PC without a graphics card WhisperJAV can use | CPU only — the installer asks | CPU (not accelerated) | CPU (not accelerated) |
| Linux PC with a supported NVIDIA GPU, driver 570 or newer | NVIDIA GPU (CUDA), chosen by itself | NVIDIA GPU (CUDA) on x86-64; CPU on ARM64 | NVIDIA GPU (CUDA); on ARM64, balanced, fast and faster on the CPU |
| Linux PC with an AMD or Intel GPU, or no GPU | your own PyTorch, or CPU only — the installer asks | CPU (not accelerated) | your own PyTorch: as below; CPU only: CPU (not accelerated) |
| Apple silicon Mac, macOS 14 or newer | Apple GPU (Metal), chosen by itself | ask for the Apple GPU (Metal) — not yet tested on a Mac | transformers: Apple GPU (Metal); fidelity: asks for it, not yet tested on a Mac; balanced, fast, faster and ChronosJAV: CPU (not accelerated) |
| Any supported computer with your own PyTorch build for your GPU (for example ROCm for an AMD GPU) | Own PyTorch: WhisperJAV installs everything except PyTorch | Windows: asks for your graphics card through Vulkan (experimental, not tested); Linux: CPU; Mac: asks for the Apple GPU (Metal), not yet tested on a Mac | fidelity, transformers, ChronosJAV and the speech enhancers ask for the GPU your PyTorch reports (not tested); balanced, fast and faster run on the CPU unless CTranslate2, their engine, sees an NVIDIA GPU |

The WhisperSeg speech segmenter runs on the CPU in every install.

**Changing the accelerator later:** run the installer again. It shows the choice, with the accelerator of your install pre-selected. `whisperjav-upgrade` never changes it.

**One run on the CPU:** add `--device cpu` to that run. In the GUI, the Customize window of an Ensemble pass has a Device menu; "CPU (not accelerated)" puts that pass on the CPU (not offered for the GGUF pipelines). When the GPU of an NVIDIA GPU install cannot be used, the GUI offers "Run on the CPU this time".

**Intel GPUs in WhisperJAV 1.10.0:** on Windows, the GGUF pipelines (Parakeet-CrispASR and Cohere-CrispASR) can use an Intel GPU through Vulkan — experimental, not tested. WhisperJAV 1.10.0 does not use Intel GPUs through PyTorch (Intel's XPU builds): with such a PyTorch, the other pipelines run on the CPU. Intel GPU support through PyTorch is planned for research in 1.10.1.

**Experimental:** GPUs through Vulkan on Windows, and your own PyTorch, are experimental in WhisperJAV 1.10.0: we have done everything we can to make them work, but we could not test them on AMD or Intel graphics cards. Please tell us how they work on your PC: https://github.com/meizhong986/WhisperJAV/issues

## Supported computers

Every installer checks the computer before anything is installed. On a computer outside this list it says why and installs nothing.

| | WhisperJAV 1.10.0 supports | Not supported |
|---|---|---|
| Windows | Windows 10 or 11, 64-bit (x86-64) | Windows on ARM (for example Snapdragon laptops); Windows older than 10 |
| Linux | x86-64 or ARM64 with glibc 2.35 or newer: Ubuntu 22.04 or newer, Debian 12 or newer, Fedora 36 or newer, openSUSE Leap 15.6, and distributions built on them; rolling distributions | glibc older than 2.35: Ubuntu 20.04, Debian 11, openSUSE Leap 15.5 (glibc 2.31); Red Hat Enterprise Linux 9, Rocky Linux 9, AlmaLinux 9 (glibc 2.34) |
| Mac | Apple silicon (M1 or newer), macOS 14 (Sonoma) or newer | Intel Macs; macOS 13 and older |
| NVIDIA GPU, for the NVIDIA GPU (CUDA) install | GTX 16 / RTX 20 series or newer; driver 580 or newer on Windows, 570 or newer on Linux | An older card or driver can still install: the installer offers the other accelerators |
| Python (source install) | 3.13 | other versions |
| Hosted notebooks | — | Google Colab, Kaggle |

The GGUF pipelines (Parakeet-CrispASR, Cohere-CrispASR) need a processor with AVX2 (Intel since 2013, AMD since 2015); every other pipeline runs without it.

Other needs: 8 GB of RAM (16 GB recommended); about 10 GB of free disk space for the install and the models it downloads on first use; FFmpeg (the Windows installer brings it); an internet connection while installing and on the first run of each model.

After installation, launch the GUI with `whisperjav-gui` and follow the [GUI User Guide](../guides/gui_user_guide.md).
