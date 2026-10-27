# 快速入门

WhisperJAV 1.10.0 可在 Windows、Linux 和 Apple silicon Mac 上运行。请根据你的电脑选择安装方式。

## 我应该选择哪种安装方式？

| 如果你... | 请使用 |
|-----------|--------|
| 只想在 Windows 上开箱即用 | [Windows 安装程序](../guides/installation_windows_standalone.md) |
| 在 Windows 上自行管理 Python 环境 | [Windows Python 安装](../guides/installation_windows_python.md) |
| 使用 Apple silicon Mac（M1 或更新） | [macOS 指南](../guides/installation_mac_apple_silicon.md) |
| 使用 Linux | [Linux 指南](../guides/installation_linux.md) |

**从 1.9.x 升级？** 请全新安装 1.10.0。1.10.0 需要 Python 3.13，因此 `whisperjav-upgrade` 无法把 1.9.x 安装升级到 1.10.0。用户设置文件夹中的设置会保留。

**Google Colab 和 Kaggle** 不受 1.10.0 支持：1.10.0 只为在你自己的电脑上安装而制作和测试。Colab 专家版 notebook 继续安装 1.9.5。

## 是否加速 {#accelerated-or-not}

GPU——显卡，或 Apple 芯片内置的 GPU——承担把语音转成文字的繁重工作，速度比 CPU 快很多倍。每个 WhisperJAV 安装都只为一种加速器制作，在安装时由你选择：NVIDIA GPU（CUDA）、通过 Vulkan 使用的 GPU（AMD、Intel 或较旧的 NVIDIA 显卡，仅 Windows）、Apple GPU（Metal）、你自己的 PyTorch，或仅 CPU（不加速）。每次运行开始时，只要没有使用安装的 GPU，WhisperJAV 就会说明各部分在哪里运行。不加速也能用，但会慢好几倍。

每种电脑得到的安装：

| 你的电脑 | 安装为哪种加速器制作 | GGUF 处理管线（Parakeet-CrispASR、Cohere-CrispASR） | 其他所有处理管线 |
|---|---|---|---|
| Windows 电脑，带 1.10 支持的 NVIDIA GPU（GTX 16 / RTX 20 或更新），驱动 580 或更新 | NVIDIA GPU（CUDA），自动选择 | NVIDIA GPU（CUDA） | NVIDIA GPU（CUDA） |
| Windows 电脑，带 AMD 或 Intel 显卡，或比 1.10 支持范围更旧的 NVIDIA 显卡 | 通过 Vulkan 使用 GPU、你自己的 PyTorch 或仅 CPU——安装程序会询问 | 通过 Vulkan：使用你的显卡（实验性，未经测试） | CPU（不加速），慢好几倍 |
| Windows 电脑，没有 WhisperJAV 能用的显卡 | 仅 CPU——安装程序会询问 | CPU（不加速） | CPU（不加速） |
| Linux 电脑，带受支持的 NVIDIA GPU，驱动 570 或更新 | NVIDIA GPU（CUDA），自动选择 | x86-64 上为 NVIDIA GPU（CUDA）；ARM64 上为 CPU | NVIDIA GPU（CUDA）；ARM64 上 balanced、fast 和 faster 在 CPU 上 |
| Linux 电脑，带 AMD 或 Intel GPU，或没有 GPU | 你自己的 PyTorch 或仅 CPU——安装程序会询问 | CPU（不加速） | 自己的 PyTorch：见下一行；仅 CPU：CPU（不加速） |
| Apple silicon Mac，macOS 14 或更新 | Apple GPU（Metal），自动选择 | 请求使用 Apple GPU（Metal）——尚未在 Mac 上测试 | transformers：Apple GPU（Metal）；fidelity：请求使用它，尚未在 Mac 上测试；balanced、fast、faster 和 ChronosJAV：CPU（不加速） |
| 任何受支持的电脑，你有适合自己 GPU 的 PyTorch 版本（例如 AMD GPU 的 ROCm 版） | 自己的 PyTorch：WhisperJAV 安装除 PyTorch 以外的一切 | Windows：请求通过 Vulkan 使用你的显卡（实验性，未经测试）；Linux：CPU；Mac：请求使用 Apple GPU（Metal），尚未在 Mac 上测试 | fidelity、transformers、ChronosJAV 和语音增强器请求使用你的 PyTorch 报告的 GPU（未经测试）；balanced、fast 和 faster 在 CPU 上运行，除非它们的引擎 CTranslate2 能看到 NVIDIA GPU |

WhisperSeg 语音分割器在所有安装中都在 CPU 上运行。

**以后更换加速器：** 再次运行安装程序。它会显示选项，并预先选中当前安装的加速器。`whisperjav-upgrade` 从不更改它。

**某一次运行改用 CPU：** 在那次运行中加上 `--device cpu`。在 GUI 中，集成模式通道的 Customize 窗口有一个 Device 菜单；选 "CPU (not accelerated)" 会让该通道使用 CPU（GGUF 处理管线不提供此项）。当 NVIDIA GPU 安装的 GPU 无法使用时，GUI 会提供 "Run on the CPU this time"（这次在 CPU 上运行）。

**WhisperJAV 1.10.0 中的 Intel GPU：** 在 Windows 上，GGUF 处理管线（Parakeet-CrispASR 和 Cohere-CrispASR）可以通过 Vulkan 使用 Intel GPU——实验性，未经测试。WhisperJAV 1.10.0 不通过 PyTorch（Intel 的 XPU 版本）使用 Intel GPU：使用这种 PyTorch 时，其他处理管线在 CPU 上运行。通过 PyTorch 支持 Intel GPU 计划在 1.10.1 中研究。

**实验性：** 在 WhisperJAV 1.10.0 中，Windows 上通过 Vulkan 使用 GPU，以及你自己的 PyTorch，都属于实验性功能：我们已尽一切努力让它们可用，但无法在 AMD 或 Intel 显卡上测试。请告诉我们它们在你的电脑上表现如何：https://github.com/meizhong986/WhisperJAV/issues

## 受支持的电脑 {#supported-computers}

所有安装程序都会在安装任何内容之前检查电脑。如果电脑不在此列表中，安装程序会说明原因，并且不安装任何内容。

| | WhisperJAV 1.10.0 支持 | 不支持 |
|---|---|---|
| Windows | Windows 10 或 11，64 位（x86-64） | Windows on ARM（例如 Snapdragon 笔记本）；早于 Windows 10 的版本 |
| Linux | glibc 2.35 或更新的 x86-64 或 ARM64：Ubuntu 22.04 或更新、Debian 12 或更新、Fedora 36 或更新、openSUSE Leap 15.6，以及基于它们的发行版；滚动更新发行版 | glibc 早于 2.35：Ubuntu 20.04、Debian 11、openSUSE Leap 15.5（glibc 2.31）；Red Hat Enterprise Linux 9、Rocky Linux 9、AlmaLinux 9（glibc 2.34） |
| Mac | Apple silicon（M1 或更新），macOS 14（Sonoma）或更新 | Intel Mac；macOS 13 及更早 |
| NVIDIA GPU（用于 NVIDIA GPU（CUDA）安装） | GTX 16 / RTX 20 系列或更新；Windows 驱动 580 或更新，Linux 驱动 570 或更新 | 较旧的显卡或驱动仍可安装：安装程序会提供其他加速器 |
| Python（源码安装） | 3.13 | 其他版本 |
| 托管 notebook | — | Google Colab、Kaggle |

GGUF 处理管线（Parakeet-CrispASR、Cohere-CrispASR）需要支持 AVX2 的处理器（Intel 2013 年以后、AMD 2015 年以后）；其他所有处理管线不需要。

其他要求：8 GB 内存（推荐 16 GB）；约 10 GB 可用磁盘空间，用于安装和首次使用时下载的模型；FFmpeg（Windows 安装程序已附带）；安装时以及每个模型首次运行时需要联网。

安装完成后，使用 `whisperjav-gui` 启动图形界面，然后参照 [GUI 用户指南](../guides/gui_user_guide.md) 开始使用。
