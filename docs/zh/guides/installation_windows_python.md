# WhisperJAV 1.10.0 — 在 Windows 上从源码安装（Python）

本指南把 WhisperJAV 从源码安装到你自己管理的 Python 环境中。如果只想开箱即用，请改用 [Windows 安装程序](installation_windows_standalone.md)。

---

## 目录

1. [需要的软件](#1-what-you-need)
2. [获取源码](#2-get-the-source-code)
3. [选择 Python 环境](#3-choose-the-python-environment)
4. [运行安装程序](#4-run-the-installer)
5. [安装为哪种加速器制作](#5-what-the-install-is-made-for)
6. [检查与运行](#6-check-and-run)
7. [更新](#7-updating)
8. [故障排除](#8-troubleshooting)

---

## 1. 需要的软件 {#1-what-you-need}

| 软件 | 版本 | 获取方式 |
|---|---|---|
| Windows | 10 或 11，64 位（x86-64）；不支持 Windows on ARM | — |
| Python | **3.13**（其他版本会被拒绝） | [python.org](https://www.python.org/downloads/) 或 conda/Miniconda |
| Git | 任意较新版本 | [git-scm.com](https://git-scm.com/download/win) |
| FFmpeg | 6.x 或 7.x，需在 PATH 中 | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/)，或 `winget install --id=Gyan.FFmpeg -e` |
| Visual C++ Redistributable | 2015–2022（x64） | [microsoft.com](https://aka.ms/vs/17/release/vc_redist.x64.exe) |
| WebView2 Runtime | 任意版本，供 GUI 使用 | [microsoft.com](https://go.microsoft.com/fwlink/p/?LinkId=2124703) |
| NVIDIA 驱动 | 580 或更新，用于 NVIDIA GPU（CUDA）安装 | [nvidia.com/drivers](https://www.nvidia.com/drivers) |

在命令提示符中检查：

```cmd
python --version
git --version
ffmpeg -version
nvidia-smi
```

没有 NVIDIA GPU 时 `nvidia-smi` 会失败，这没有关系。如果缺少 `uv`（安装程序使用的软件包安装工具），安装程序会自行安装。

中国用户：安装程序会从 GitHub、PyPI、download.pytorch.org 和 Hugging Face 下载。请在整个安装过程中保持 VPN 开启。

---

## 2. 获取源码 {#2-get-the-source-code}

```cmd
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
```

文件夹路径尽量短，只用西文字母，尽量不含空格。

---

## 3. 选择 Python 环境 {#3-choose-the-python-environment}

安装程序会安装到运行它时处于激活状态的 Python 环境：

- **venv：**
  ```cmd
  python -m venv whisperjav-env
  whisperjav-env\Scripts\activate
  ```
- **conda 环境：**
  ```cmd
  conda create -n whisperjav python=3.13 -y
  conda activate whisperjav
  ```
- **没有激活的环境：** 安装程序会在 `WhisperJAV` 文件夹中创建 `.venv` 并安装到那里。即使有激活的环境，`--local` 也会这样做。

不要安装到系统级的 Python 中。

---

## 4. 运行安装程序 {#4-run-the-installer}

在 `WhisperJAV` 文件夹中，激活环境后运行：

```cmd
installer\install_windows.bat
```

或者（效果相同）：

```cmd
python install.py
```

| 选项 | 作用 |
|---|---|
| `--accelerator nvidia` | 为 NVIDIA GPU（CUDA）制作 |
| `--accelerator vulkan` | 为通过 Vulkan 使用 GPU 制作（AMD、Intel 或较旧的 NVIDIA 显卡）——实验性 |
| `--accelerator own-pytorch` | 保留这个环境中已有的 PyTorch，安装其他一切——实验性 |
| `--accelerator cpu` | 为仅 CPU 制作（不加速） |
| `--no-speech-enhancement` | 不安装语音增强软件包 |
| `--minimal` | 仅命令行：不安装 GUI，不安装的内容中还包括 WhisperSeg 语音分割器需要的软件包 |
| `--dev` | 可编辑安装，用于开发 |
| `--local` | 即使有激活的环境，也安装到源码文件夹中的 `.venv` |
| `--log-file PATH` | 把日志写到 PATH |

安装程序首先检查电脑是否受支持，然后按安装的加速器，精确安装本版本测试过的软件包版本（从锁定文件执行 `uv sync`）。它从不改写锁定文件。

---

## 5. 安装为哪种加速器制作 {#5-what-the-install-is-made-for}

不加 `--accelerator` 时：

- **受支持的 NVIDIA GPU**（GTX 16 / RTX 20 或更新，驱动 580 或更新）：为 NVIDIA GPU（CUDA 13.0）制作，不会提问。
- **其他情况**，安装程序会提出与 Windows 安装程序窗口相同的问题：

  ```
  Type 1 = AMD, 2 = Intel, 3 = older NVIDIA card, 4 = my own PyTorch, 5 = CPU only, 6 = stop, then Enter:
  ```

  （意思是：输入 1 = AMD、2 = Intel、3 = 较旧的 NVIDIA 显卡、4 = 我自己的 PyTorch、5 = 仅 CPU、6 = 停止，然后按 Enter。）1、2、3 使安装为"通过 Vulkan 使用 GPU"制作，这需要随显卡驱动安装的 Vulkan 运行库（`vulkan-1.dll`）——缺少时会制作仅 CPU 的安装。有 1.10 无法使用的 NVIDIA GPU（较旧的显卡或驱动）时，会先显示原因。没有时间限制。6 会停止，不安装任何内容。
- **没有可以回答的控制台**（自动运行）：没有受支持的 NVIDIA GPU 时制作仅 CPU 的安装；有 1.10 无法使用的 NVIDIA GPU 时停止。

每种加速器下各处理管线在哪里运行：见[是否加速](../getting-started/index.md#accelerated-or-not)。

**自己的 PyTorch。** 先把你的 PyTorch 版本安装到环境中（推荐 PyTorch 2.11），然后运行 `python install.py --accelerator own-pytorch`。WhisperJAV 安装除 PyTorch 以外的一切，并且从不在这个环境中安装、替换或删除 PyTorch。如果激活的环境中没有 PyTorch，安装程序会停止，不安装任何内容。

选择会记录在环境文件夹中的 `whisperjav_stack.txt`。在同一个环境中再次运行安装程序会保留它；`--accelerator` 可以另选。

---

## 6. 检查与运行 {#6-check-and-run}

激活环境后：

```cmd
whisperjav --check
whisperjav-gui
whisperjav video.mp4
```

`whisperjav --check` 会列出安装的加速器、各部分在哪里运行、CrispASR 和模型。每次运行开始时，WhisperJAV 会说明各部分在哪里运行（在 GPU 可用的 NVIDIA GPU（CUDA）安装上不显示）。`--device cpu` 让某一次运行在 CPU 上进行。

---

## 7. 更新 {#7-updating}

在 1.10 内，更新源码后再次运行安装程序，并激活与安装时相同的 Python 环境（如果安装在 `.venv` 中，则不激活任何环境）；它会保留环境的加速器：

```cmd
git fetch --tags
git checkout v1.10.x
python install.py
```

（`v1.10.x` 代表你想要的版本。）1.9.x 环境无法更新到 1.10.0：1.10.0 需要 Python 3.13。请新建一个 Python 3.13 环境并安装到那里。

不要用普通的 `pip install -U whisperjav…` 更新：在 Windows 上，PyPI 提供的是仅 CPU 版的 PyTorch，可能会在毫无提示的情况下替换你的 GPU 版本。

---

## 8. 故障排除 {#8-troubleshooting}

**"WhisperJAV 1.10.0 does not support …"。** 电脑不在[受支持列表](../getting-started/index.md#supported-computers)中；提示会说明原因。没有安装任何内容。

**安装程序拒绝你的 Python。** WhisperJAV 1.10.0 需要 Python 3.13。请创建一个 3.13 环境（第 3 节）。

**`--accelerator own-pytorch` 停止："none was found"。** 先把你的 PyTorch 和 torchaudio 安装到激活的环境中，再次运行安装程序。

**运行停止："THE NVIDIA GPU CANNOT BE USED"。** 安装为 NVIDIA GPU 制作，而 GPU 现在不可用（例如更换了驱动）。框中会说明原因和解决办法；`--device cpu` 让这一次在 CPU 上运行。

**GGUF 处理管线停止："CrispASR could not be loaded"。** 在 Windows 上的"通过 Vulkan 使用 GPU"或"自己的 PyTorch"安装中，这通常意味着缺少 Vulkan 运行库（`vulkan-1.dll`）：更新显卡驱动，或运行 `python install.py --accelerator cpu`。

**找不到 `ffmpeg`。** 把 FFmpeg 的 `bin` 文件夹加入 PATH，并打开新的命令提示符。

**GUI 无法打开。** 安装 WebView2。要查看错误，运行 `python -m whisperjav.webview_gui.main`。

**其他问题。** 日志会指出失败的步骤。请在 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题并附上日志。
