# WhisperJAV Linux 安装指南

**版本：** 1.10.0

项目中还没有人在 Linux 上运行过 WhisperJAV 1.10.0。请告诉我们它在你的电脑上表现如何：https://github.com/meizhong986/WhisperJAV/issues

---

## 目录

1. [你的电脑](#your-computer)
2. [系统软件包](#system-packages)
3. [Python 3.13](#python-313)
4. [NVIDIA 驱动](#nvidia-driver)
5. [安装](#installation)
6. [安装为哪种加速器制作](#what-the-install-is-made-for)
7. [检查与运行](#check-and-run)
8. [更新](#updating)
9. [故障排除](#troubleshooting)
10. [卸载](#uninstalling)

---

## 你的电脑 {#your-computer}

| | WhisperJAV 1.10.0 需要 |
|---|---|
| 处理器 | x86-64 或 ARM64 |
| Linux | glibc 2.35 或更新：Ubuntu 22.04 或更新、Debian 12 或更新、Fedora 36 或更新、openSUSE Leap 15.6，以及基于它们的发行版；Arch 等滚动更新发行版 |
| Python | 3.13 |
| 内存 | 8 GB（推荐 16 GB） |
| 磁盘 | 约 10 GB，用于安装和首次使用时下载的模型 |
| NVIDIA GPU（可选） | GTX 16 / RTX 20 系列或更新，驱动 570 或更新 |

**不支持**（安装程序会说明原因，不安装任何内容）：glibc 早于 2.35——Ubuntu 20.04、Debian 11、openSUSE Leap 15.5（glibc 2.31）；Red Hat Enterprise Linux 9、Rocky Linux 9、AlmaLinux 9（glibc 2.34）。查看 glibc 版本：`ldd --version`。

**Linux 上的 AMD 和 Intel GPU：** 在 1.10.0 中，GGUF 处理管线（Parakeet-CrispASR、Cohere-CrispASR）在 Linux 上运行于 CPU，因为 CrispASR 尚未发布适用于 AMD 或 Intel 的 Linux GPU 软件包。使用适合你 GPU 的自有 PyTorch 版本（例如 AMD GPU 的 ROCm 版）时，PyTorch 处理管线可以请求使用它——见[自己的 PyTorch](#own-pytorch)。

---

## 系统软件包 {#system-packages}

请在运行 WhisperJAV 安装程序之前安装它们；pip 无法安装这些软件包。

### Ubuntu / Debian

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential ffmpeg git
sudo apt-get install -y libsndfile1 libsndfile1-dev
# TEN VAD 语音分割器
sudo apt-get install -y libc++1 libc++abi1
# PyAudio，auditok 场景检测会用到
sudo apt-get install -y portaudio19-dev
# GUI：Ubuntu 24.04+ 提供 WebKit 4.1，22.04 提供 4.0；下面的命令会安装其中存在的那个
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

在 Fedora 上，完整的 `ffmpeg` 软件包来自 RPM Fusion；如果 `dnf` 找不到 `ffmpeg`，请先启用 RPM Fusion。

### Arch Linux / Manjaro

```bash
sudo pacman -S --noconfirm python python-pip base-devel ffmpeg git libsndfile portaudio
# GUI
sudo pacman -S --noconfirm webkit2gtk gtk3
```

### GUI 后端（所有发行版） {#gui-backend-all-distributions}

上面的系统软件包提供了 WebKit，但**单靠它们还不够**。pywebview 在 Linux 上**默认不带任何后端**；GTK 和 Qt 都需要另外选择。而且 apt/dnf 软件包安装在系统范围，未使用 `--system-site-packages` 创建的虚拟环境看不到 `gi`（PyGObject）绑定。

因此，只按发行版的说明操作，启动 `whisperjav-gui` 时仍会出现 `ModuleNotFoundError: No module named 'gi'`（#366）。还需要*在虚拟环境中*安装一个后端：

```bash
# Qt 后端——最简单，不需要系统 GTK 绑定
pip install "pywebview[qt]"

# 或者，改用 GTK：让虚拟环境能看到系统绑定
#   python3.13 -m venv --system-site-packages whisperjav-env
```

这一步只针对 Linux。Windows 和 macOS 不需要任何额外操作。

---

## Python 3.13 {#python-313}

WhisperJAV 1.10.0 只能在 Python 3.13 上运行。`python3 --version` 显示系统 Python 的版本；如果不是 3.13：

- **Ubuntu 22.04 / 24.04：** 从 deadsnakes PPA 安装 3.13：
  ```bash
  sudo add-apt-repository ppa:deadsnakes/ppa
  sudo apt-get update
  sudo apt-get install -y python3.13 python3.13-venv python3.13-dev
  ```
- **Fedora：** 在你的版本提供时，运行 `sudo dnf install -y python3.13 python3.13-devel`。
- **任何发行版：** `uv` 可以为你下载 Python 3.13。先安装 `uv`（`pip install --user uv`，或见 [uv 的安装页面](https://docs.astral.sh/uv/getting-started/installation/)），然后 `uv python install 3.13`，然后用 `uv venv --python 3.13 ~/.venv/whisperjav` 创建环境（代替下面的 `python3.13 -m venv` 一行）。

然后创建并激活虚拟环境（Debian 12+ 和 Ubuntu 24.04+ 不允许安装到系统 Python 中）：

```bash
python3.13 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
```

---

## NVIDIA 驱动 {#nvidia-driver}

NVIDIA GPU（CUDA）安装需要 NVIDIA 驱动 570 或更新；不需要 CUDA Toolkit（PyTorch 自带 CUDA 运行库）。在 Linux 上，1.10.0 安装适用于 CUDA 12.8 的 PyTorch。

```bash
nvidia-smi          # 显示驱动版本；没有驱动时会失败
```

安装驱动：

- **Ubuntu：** `sudo ubuntu-drivers autoinstall`，或 `sudo apt-get install -y nvidia-driver-570`，然后 `sudo reboot`。
- **Fedora：** 启用 RPM Fusion 后，`sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda`，然后 `sudo reboot`。
- **Arch：** `sudo pacman -S nvidia nvidia-utils`，然后 `sudo reboot`。
- **数据中心 GPU（Ubuntu）：** `sudo apt-get install -y nvidia-headless-570-server nvidia-utils-570-server`，然后 `sudo reboot`。

---

## 安装 {#installation}

在激活虚拟环境后：

```bash
git clone --branch v1.10.0 https://github.com/meizhong986/WhisperJAV.git
cd WhisperJAV
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

`./installer/install_linux.sh` 先检查电脑是否受支持，然后带着你给出的选项运行 `python3 install.py`。直接运行 `python3 install.py` 效果相同。

| 选项 | 作用 |
|---|---|
| `--accelerator nvidia` | 为 NVIDIA GPU 制作（Linux 上为 CUDA 12.8） |
| `--accelerator own-pytorch` | 保留这个环境中已有的 PyTorch，安装其他一切——实验性 |
| `--accelerator cpu` | 为仅 CPU 制作（不加速） |
| `--no-speech-enhancement` | 不安装语音增强软件包 |
| `--minimal` | 仅命令行：不安装 GUI，不安装的内容中还包括 WhisperSeg 语音分割器需要的软件包 |
| `--dev` | 可编辑安装，用于开发 |
| `--local` | 即使有激活的环境，也安装到源码文件夹中的 `.venv` |

（`--accelerator vulkan` 仅适用于 Windows：CrispASR 没有 Linux 的 Vulkan 软件包。）

安装程序按安装的加速器，精确安装本版本测试过的软件包版本（从锁定文件执行 `uv sync`），并在源码文件夹中创建 GUI 启动器 `WhisperJAV.sh`。没有激活的环境时，它会在源码文件夹中创建 `.venv`。

中国用户：安装程序会从 GitHub、PyPI、download.pytorch.org 和 Hugging Face 下载。请在整个安装过程中保持 VPN 开启。

---

## 安装为哪种加速器制作 {#what-the-install-is-made-for}

不加 `--accelerator` 时：

- **受支持的 NVIDIA GPU**（驱动 570 或更新）：为 NVIDIA GPU（CUDA 12.8）制作，不会提问。
- **其他情况**，安装程序会询问：

  ```
  No supported NVIDIA GPU found. What should this install be made for? 1 = my own PyTorch (already installed in this Python environment, for example ROCm for an AMD GPU); 2 = CPU only (not accelerated); 3 = stop. Type 1, 2 or 3 and press Enter:
  ```

  （意思是：没有找到受支持的 NVIDIA GPU；这次安装要为什么制作？1 = 我自己的 PyTorch（已安装在这个 Python 环境中，例如 AMD GPU 的 ROCm 版）；2 = 仅 CPU（不加速）；3 = 停止。）没有时间限制。有 1.10 无法使用的 NVIDIA GPU（较旧的显卡或驱动）时，会先显示原因。没有可以回答的控制台时（自动运行），会制作仅 CPU 的安装——如果有 1.10 无法使用的 NVIDIA GPU，安装程序会停止。

| 安装为哪种加速器制作 | GGUF 处理管线 | 其他处理管线 |
|---|---|---|
| NVIDIA GPU（CUDA），x86-64 | NVIDIA GPU（CUDA） | NVIDIA GPU（CUDA） |
| NVIDIA GPU（CUDA），ARM64 | CPU（不加速） | PyTorch 处理管线在 NVIDIA GPU 上；balanced、fast 和 faster 在 CPU 上 |
| 自己的 PyTorch | CPU（不加速） | fidelity、transformers、ChronosJAV 和语音增强器请求使用你的 PyTorch 报告的 GPU（未经测试）；balanced、fast 和 faster 在 CPU 上运行，除非它们的引擎 CTranslate2 能看到 NVIDIA GPU |
| 仅 CPU | CPU（不加速） | CPU（不加速），慢好几倍 |

WhisperSeg 语音分割器在所有安装中都在 CPU 上运行。

### 自己的 PyTorch {#own-pytorch}

先把你的 PyTorch 版本安装到环境中（推荐 PyTorch 2.11；AMD GPU 请用 pytorch.org 上的 ROCm 版），然后：

```bash
python3 install.py --accelerator own-pytorch
```

WhisperJAV 安装除 PyTorch 以外的一切，并且从不在这个环境中安装、替换或删除 PyTorch——安装、升级和修复时都是如此。如果激活的环境中没有 PyTorch，安装程序会停止，不安装任何内容。这在 1.10.0 中属于实验性功能：我们无法测试；请告诉我们效果如何：https://github.com/meizhong986/WhisperJAV/issues

选择会记录在环境文件夹中的 `whisperjav_stack.txt`。在同一个环境中再次运行安装程序会保留它；`--accelerator` 可以另选。

---

## 检查与运行 {#check-and-run}

```bash
whisperjav --check          # 加速器、各部分在哪里运行、CrispASR、模型
whisperjav-gui              # 或在源码文件夹中运行 ./WhisperJAV.sh
whisperjav video.mp4
```

每次运行开始时，WhisperJAV 会说明各部分在哪里运行（在 GPU 可用的 NVIDIA GPU（CUDA）安装上不显示）。`--device cpu` 让某一次运行在 CPU 上进行。

**在没有显示器的服务器上：** 使用命令行；命令行不需要 GUI 软件包。

---

## 更新 {#updating}

在 1.10 内，更新源码后再次运行安装程序，并激活与安装时相同的 Python 环境（如果安装在 `.venv` 中，则不激活任何环境）；它会保留环境的加速器：

```bash
git fetch --tags
git checkout v1.10.x      # 你想要的版本
python3 install.py
```

1.9.x 环境无法更新到 1.10.0：1.10.0 需要 Python 3.13。请新建一个 Python 3.13 环境并安装到那里。

---

## 故障排除 {#troubleshooting}

**"WhisperJAV 1.10.0 does not support this Linux"。** 它的 glibc 早于 2.35；提示会写出版本。没有安装任何内容。请使用较新的发行版版本。

**运行停止："THE NVIDIA GPU CANNOT BE USED"。** 安装为 NVIDIA GPU 制作，而 GPU 现在不可用。框中会说明原因（例如驱动）和解决办法；`--device cpu` 让这一次在 CPU 上运行。`nvidia-smi` 会显示驱动是否已加载。

**`ModuleNotFoundError: No module named 'gi'`。** 见 [GUI 后端](#gui-backend-all-distributions)。

**找不到库（`libsndfile`、`portaudio`、`libc++`）。** 安装[系统软件包](#system-packages)后再次运行安装程序。

**Git 超时。** 再次运行安装程序；它会用更长的 Git 超时重试。在代理后面时，运行前设置 `https_proxy`。

**其他问题。** 请在 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题，并附上安装程序的输出。

---

## 卸载 {#uninstalling}

```bash
rm -rf ~/.venv/whisperjav          # 虚拟环境（或源码文件夹中的 .venv）
rm -rf ~/WhisperJAV                # 源码文件夹（你克隆到的位置）
rm -rf ~/.cache/huggingface ~/.cache/whisper   # 下载的模型（可选；其他程序也可能在用）
```

设置保存在 `~/.config/WhisperJAV`。
