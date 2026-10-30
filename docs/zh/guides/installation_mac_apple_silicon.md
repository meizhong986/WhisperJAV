# WhisperJAV Apple silicon Mac 安装指南

**版本：** 1.10.0

在 Apple silicon Mac 上，WhisperJAV 1.10.0 会自动为 **Apple GPU（Metal）** 制作安装：即 Apple 芯片内置的 GPU。部分处理管线使用它，其他的在 CPU 上运行（见[各处理管线在哪里运行](#4-where-each-pipeline-runs)）。Mac 没有类似 `.exe` 的安装程序；你需要用一个脚本从源码安装。

项目中还没有人在 Mac 上运行过 WhisperJAV 1.10.0。标为"尚未在 Mac 上测试"的部分已经具备，应当可以工作，但我们无法检查。请告诉我们它在你的 Mac 上表现如何：https://github.com/meizhong986/WhisperJAV/issues

---

## 目录

1. [你的 Mac](#1-your-mac)
2. [准备工作](#2-prerequisites)
3. [安装](#3-installation)
4. [各处理管线在哪里运行](#4-where-each-pipeline-runs)
5. [运行 WhisperJAV](#5-running-whisperjav)
6. [更新](#6-updating)
7. [故障排除](#7-troubleshooting)

---

## 1. 你的 Mac {#1-your-mac}

| | WhisperJAV 1.10.0 需要 |
|---|---|
| 芯片 | Apple silicon（M1 或更新） |
| macOS | 14（Sonoma）或更新 |
| 内存 | 8 GB（推荐 16 GB） |
| 磁盘 | 约 10 GB，用于安装和首次使用时下载的模型 |
| Python | 3.13，Apple silicon（arm64）版 |

**不支持**（安装程序会说明原因，不安装任何内容）：Intel Mac，因为 WhisperJAV 1.10.0 需要的 PyTorch 2.11 没有为它们制作；macOS 13 及更早，因为 ONNX Runtime 1.29 需要 macOS 14 或更新。在 Apple silicon Mac 上通过 Rosetta（作为 Intel 程序）运行的 Python 也会被拒绝：请使用 arm64 的 Python，例如 Homebrew 提供的。

---

## 2. 准备工作 {#2-prerequisites}

打开 **终端**，逐条运行：

1. **Xcode 命令行工具：**
   ```bash
   xcode-select --install
   ```
2. **Homebrew**（见 [brew.sh](https://brew.sh)）：
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
3. **Python 3.13、FFmpeg、PortAudio 和 Git：**
   ```bash
   brew install python@3.13 ffmpeg portaudio git
   ```
   PortAudio 是 PyAudio 需要的，auditok 场景检测器会用到 PyAudio。

---

## 3. 安装 {#3-installation}

```bash
# 使用 Python 3.13 的虚拟环境（Homebrew 的 Python 不允许直接安装到其中）
python3.13 -m venv ~/venvs/whisperjav
source ~/venvs/whisperjav/bin/activate
python3 --version          # 应显示 Python 3.13.x

# 该版本的源码
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV

# 安装脚本
chmod +x installer/install_mac.sh
./installer/install_mac.sh
```

脚本先检查 Mac 是否受支持（Apple silicon、macOS 14 或更新），然后检查 Xcode 工具、Homebrew、Python、FFmpeg 和 PortAudio，再运行 `install.py`。它会精确安装本版本测试过的软件包版本（从锁定文件执行 `uv sync`）：带 Apple GPU 支持的 PyTorch，以及 CrispASR 的 macOS 版。它会显示：

> Apple silicon Mac: installing for the Apple GPU (Metal) — PyTorch with Apple GPU support, and CrispASR's macOS build.

（意思是：Apple silicon Mac：为 Apple GPU（Metal）安装——带 Apple GPU 支持的 PyTorch，以及 CrispASR 的 macOS 版。）

最后，它会在 `WhisperJAV` 文件夹中创建 `WhisperJAV.command`：在访达中双击它即可启动 GUI（也可以拖到程序坞）。

中国用户：安装程序会从 GitHub、PyPI、download.pytorch.org 和 Hugging Face 下载。请在整个安装过程中保持 VPN 开启。

**选项**（传给 `install.py`）：

| 选项 | 作用 |
|---|---|
| `--accelerator own-pytorch` | 保留这个环境中已有的 PyTorch，安装其他一切——实验性 |
| `--no-speech-enhancement` | 不安装语音增强软件包 |
| `--minimal` | 仅命令行：不安装 GUI，不安装的内容中还包括 WhisperSeg 语音分割器需要的软件包 |
| `--dev` | 可编辑安装，用于开发 |

`--accelerator cpu`（或旧的 `--cpu-only`）得到的仍是 Apple GPU 安装："On an Apple silicon Mac, WhisperJAV is installed for the Apple GPU (Metal), which includes the CPU. To run one job on the CPU, add --device cpu to that run."（意思是：在 Apple silicon Mac 上，WhisperJAV 为 Apple GPU（Metal）安装，其中也包括 CPU；要让某一次运行使用 CPU，在那次运行中加上 --device cpu。）

安装的加速器记录在环境文件夹中的 `whisperjav_stack.txt`。

---

## 4. 各处理管线在哪里运行 {#4-where-each-pipeline-runs}

| 处理管线 | 在 Apple silicon Mac 上 |
|---|---|
| **transformers** | Apple GPU（Metal）——**在 Mac 上推荐** |
| **fidelity** | 向 PyTorch 请求使用 Apple GPU（Metal）；尚未在 Mac 上测试。如果失败，请使用 transformers 处理管线。 |
| **Parakeet-CrispASR、Cohere-CrispASR**（GGUF） | 请求使用 Apple GPU（Metal）；尚未在 Mac 上测试 |
| **balanced、fast、faster** | CPU（不加速），慢好几倍：它们的引擎（CTranslate2）无法使用 Apple GPU |
| **ChronosJAV**（Qwen3-ASR、anime-whisper） | 在 1.10.0 中为 CPU（不加速）：其 ASR 模型尚不使用 Apple GPU |
| WhisperSeg 语音分割器 | CPU，所有安装中都是 |

Apple GPU 可用时，运行开始时不显示任何内容；每个处理管线启动时会在控制台中说明自己在哪里运行（例如 balanced、fast 和 faster 会说明它们在 CPU 上运行）。如果 Apple GPU 不可用，运行开始时会有一个框说明各部分在哪里运行。安装脚本结束时会显示：

> On an Apple silicon Mac, the transformers pipeline is recommended: it asks for the Apple GPU (Metal).

（意思是：在 Apple silicon Mac 上推荐 transformers 处理管线：它请求使用 Apple GPU（Metal）。）

---

## 5. 运行 WhisperJAV {#5-running-whisperjav}

先激活环境（`source ~/venvs/whisperjav/bin/activate`），或双击 `WhisperJAV.command` 打开 GUI。

```bash
whisperjav-gui                                  # GUI
whisperjav video.mp4 --mode transformers        # 在 Apple GPU 上（推荐）
whisperjav video.mp4                            # balanced，在 CPU 上
whisperjav --check                              # 加速器、各部分在哪里运行、CrispASR、模型
whisperjav video.mp4 --device cpu               # 这次运行使用 CPU
```

GUI 使用 macOS 自带的 WebKit，不需要另外安装任何东西。

---

## 6. 更新 {#6-updating}

在 1.10 内，更新源码后再次运行安装脚本：

```bash
source ~/venvs/whisperjav/bin/activate
cd WhisperJAV
git fetch --tags
git checkout v1.10.x      # 你想要的版本
./installer/install_mac.sh
```

1.9.x 环境无法更新到 1.10.0：1.10.0 需要 Python 3.13。请新建一个 Python 3.13 环境并安装到那里。

---

## 7. 故障排除 {#7-troubleshooting}

**"WhisperJAV 1.10.0 does not support Intel Macs" / "does not support macOS …"。** 这台 Mac 不在受支持列表中；没有安装任何内容。

**"Python 3.x is not supported"。** 激活 Python 3.13 环境（第 3 节）后再次运行脚本。

**`externally-managed-environment`。** 你直接用 Homebrew 的 Python 运行了它。请创建并激活虚拟环境（第 3 节）。

**PyAudio / PortAudio 错误。** 运行 `brew install portaudio`，然后再次运行安装脚本。

**找不到 FFmpeg。** 运行 `brew install ffmpeg`，然后打开新的终端。

**处理很慢。** 在 Mac 上，大多数处理管线在 CPU 上运行（第 4 节）。请使用 transformers 处理管线以使用 Apple GPU。

**fidelity 在 Apple GPU 上失败。** 改用 transformers 处理管线，或加上 `--device cpu` 让 fidelity 在 CPU 上运行。

**GUI 无法打开。** 检查环境中 `python3 -c "import webview"` 是否成功；如果不成功，运行 `xcode-select --install` 后再次运行安装脚本。不用 GUI 也可以使用命令行。

**其他问题。** 请在 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题，并附上安装脚本的输出。

设置保存在 `~/Library/Application Support/WhisperJAV`。
