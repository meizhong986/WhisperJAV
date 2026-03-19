# WhisperJAV Linux 安装指南

**版本：** 1.8.9
**最后更新：** 2026-03-19
**适用平台：** Ubuntu、Debian、Fedora、RHEL、Arch Linux、Google Colab、Kaggle

---

## 目录

1. [系统要求](#系统要求)
2. [前提条件](#前提条件)
   - [Ubuntu / Debian](#ubuntu--debian)
   - [Fedora / RHEL / CentOS Stream](#fedora--rhel--centos-stream)
   - [Arch Linux / Manjaro](#arch-linux--manjaro)
3. [NVIDIA 驱动程序和 CUDA 设置](#nvidia-驱动程序和-cuda-设置)
4. [安装方法](#安装方法)
   - [方法 1：源码安装（推荐）](#方法-1源码安装推荐)
   - [方法 2：pip 安装并指定 Extras](#方法-2pip-安装并指定-extras)
   - [方法 3：Conda 环境](#方法-3conda-环境)
5. [GPU 验证](#gpu-验证)
6. [安装特定 Extras](#安装特定-extras)
7. [无图形界面服务器设置](#无图形界面服务器设置)
8. [Google Colab 设置](#google-colab-设置)
9. [Kaggle 设置](#kaggle-设置)
10. [运行应用程序](#运行应用程序)
11. [Systemd 服务设置](#systemd-服务设置)
12. [故障排除](#故障排除)
13. [性能调优](#性能调优)
14. [卸载](#卸载)

---

## 系统要求

### 硬件

| 组件 | 最低 | 推荐 | Qwen3-ASR |
|------|------|------|-----------|
| CPU | 4 核（x86_64）| 8+ 核 | 8+ 核 |
| 内存 | 8 GB | 16 GB | 32 GB |
| GPU 显存 | 4 GB（基本）| 8 GB | 16+ GB |
| 磁盘空间 | 15 GB（安装）| 50 GB（安装 + 模型 + 临时文件）| 50+ GB |
| 网络 | 安装时必需 | 宽带用于模型下载 | 3-10 GB 模型下载 |

### 支持的 GPU

| GPU 系列 | 显存 | 推荐模式 | 备注 |
|----------|------|----------|------|
| RTX 4090/4080/4070 | 12-24 GB | 所有模式、Qwen3-ASR | 最佳性能 |
| RTX 3090/3080/3070 | 8-24 GB | 所有模式、Qwen3-ASR | 优秀 |
| RTX 3060/3050 | 6-12 GB | Balanced、Fast | 12 GB 版可使用 Qwen |
| RTX 2080/2070/2060 | 6-11 GB | Balanced、Fast | 良好 |
| GTX 1080 Ti/1070 | 8-11 GB | Balanced、Fast | 够用 |
| Tesla V100/A100 | 16-80 GB | 所有模式 | 数据中心 GPU |
| 无 GPU（仅 CPU）| 无 | 仅 Faster 模式 | 慢 10-50 倍 |

### 软件

| 组件 | 要求 | 备注 |
|------|------|------|
| Linux 内核 | 4.15+ | 推荐 5.4+ 以支持新款 NVIDIA 驱动程序 |
| Python | 3.10、3.11 或 3.12 | 3.9 和 3.13+ 不受支持 |
| NVIDIA 驱动程序 | 450+（cu118）或 570+（cu128）| GPU 加速所需 |
| FFmpeg | 4.0+ | 音视频处理所需 |
| Git | 2.0+ | 从 GitHub 安装包所需 |
| GCC / build-essential | 任意近期版本 | 编译扩展所需 |

---

## 前提条件

在运行 WhisperJAV 安装程序之前，请先安装以下系统包。这些是系统级库，pip 无法安装。

### Ubuntu / Debian

```bash
# 更新包列表
sudo apt-get update

# 必需：Python、构建工具、FFmpeg、Git
sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    ffmpeg \
    git

# 音频处理库
sudo apt-get install -y \
    libsndfile1 \
    libsndfile1-dev

# 可选：用于 TEN VAD 原生库
sudo apt-get install -y libc++1 libc++abi1

# 可选：用于 PyAudio/auditok（麦克风输入）
sudo apt-get install -y portaudio19-dev

# 可选：用于 GUI（whisperjav-gui）
sudo apt-get install -y \
    libwebkit2gtk-4.0-dev \
    libgtk-3-dev \
    gir1.2-webkit2-4.0
```

**Ubuntu 20.04 (Focal) 用户：** 默认 Python 是 3.8，版本过旧。请从 deadsnakes PPA 安装 Python 3.10+：

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
# 后续所有命令中使用 python3.11 代替 python3
```

### Fedora / RHEL / CentOS Stream

```bash
# 必需：Python、构建工具、FFmpeg、Git
sudo dnf install -y \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    ffmpeg \
    git

# 音频处理库
sudo dnf install -y libsndfile libsndfile-devel

# 可选：用于 PyAudio/auditok
sudo dnf install -y portaudio-devel

# 可选：用于 GUI
sudo dnf install -y \
    webkit2gtk4.0-devel \
    gtk3-devel
```

**RHEL/CentOS：** FFmpeg 不在默认仓库中。请先启用 RPM Fusion：

```bash
# RHEL 9 / CentOS Stream 9
sudo dnf install -y \
    https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-9.noarch.rpm \
    https://mirrors.rpmfusion.org/nonfree/el/rpmfusion-nonfree-release-9.noarch.rpm
sudo dnf install -y ffmpeg
```

### Arch Linux / Manjaro

```bash
# 必需：Python、构建工具、FFmpeg、Git
sudo pacman -S --noconfirm \
    python \
    python-pip \
    base-devel \
    ffmpeg \
    git

# 音频处理库
sudo pacman -S --noconfirm libsndfile

# 可选：用于 PyAudio/auditok
sudo pacman -S --noconfirm portaudio

# 可选：用于 GUI
sudo pacman -S --noconfirm webkit2gtk gtk3
```

---

## NVIDIA 驱动程序和 CUDA 设置

WhisperJAV 使用 PyTorch 进行 GPU 推理。您需要安装 NVIDIA 驱动程序，但不需要单独安装 CUDA Toolkit -- PyTorch 自带 CUDA 运行时。

### 检查当前驱动程序

```bash
# 检查是否已安装 NVIDIA 驱动程序
nvidia-smi
```

如果找不到 `nvidia-smi`，您需要安装 NVIDIA 驱动程序。

### 安装 NVIDIA 驱动程序

**Ubuntu / Debian：**

```bash
# 方法 1：Ubuntu 推荐的驱动程序工具（最简单）
sudo ubuntu-drivers autoinstall
sudo reboot

# 方法 2：指定驱动程序版本
sudo apt-get install -y nvidia-driver-570
sudo reboot
```

**Fedora：**

```bash
# 先启用 RPM Fusion 仓库（见上文），然后：
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
sudo reboot
```

**Arch Linux：**

```bash
sudo pacman -S nvidia nvidia-utils
sudo reboot
```

### 验证驱动程序版本

安装并重启后：

```bash
nvidia-smi
```

查看输出中的驱动程序版本。这决定了 PyTorch 将使用的 CUDA 版本：

| 驱动程序版本 | CUDA 支持 | PyTorch 索引 |
|-------------|-----------|-------------|
| 570+ | CUDA 12.8 | 最佳性能（默认）|
| 450-569 | CUDA 11.8 | 通用回退方案 |
| < 450 | 无 | 仅 CPU（请更新驱动程序！）|

### 数据中心/云端 GPU

对于 Tesla、A100、H100 或其他数据中心 GPU，安装数据中心驱动程序：

```bash
# Ubuntu
sudo apt-get install -y nvidia-headless-570-server nvidia-utils-570-server
sudo reboot
```

---

## 安装方法

### 方法 1：源码安装（推荐）

此方法使用自动化安装程序，处理 GPU 检测、安装顺序和重试逻辑。

```bash
# 步骤 1：克隆仓库
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# 步骤 2：创建并激活虚拟环境
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# 步骤 3：运行安装程序
python install.py
```

安装程序将：
1. 检查 Python 版本、FFmpeg、Git、磁盘空间和网络
2. 检测 GPU 并选择最优 CUDA 版本
3. 安装包含 GPU 支持的 PyTorch（或回退到 CPU）
4. 按正确顺序安装所有依赖项
5. 安装 WhisperJAV
6. 验证安装

**安装程序选项：**

```bash
# 仅 CPU（无 GPU）
python install.py --cpu-only

# 强制指定 CUDA 版本
python install.py --cuda118     # 用于较旧驱动程序（450+）
python install.py --cuda128     # 用于新款驱动程序（570+）

# 跳过可选功能
python install.py --no-speech-enhancement
python install.py --minimal     # 仅转录功能

# 包含本地大语言模型翻译
python install.py --local-llm          # 预编译 wheel（快速）
python install.py --local-llm-build    # 从源码编译（慢）
python install.py --no-local-llm       # 跳过且不提示

# 开发模式（可编辑安装）
python install.py --dev

# 跳过预检
python install.py --skip-preflight
```

**替代方案：使用 shell 包装器：**

```bash
chmod +x installer/install_linux.sh
./installer/install_linux.sh
```

shell 包装器会检查 PEP 668（外部管理的 Python）并委托给 `install.py`。

### 方法 2：pip 安装并指定 Extras

如果您想更精细地控制安装内容，可以直接使用 pip。但必须先安装 PyTorch。

```bash
# 步骤 1：创建并激活虚拟环境
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# 步骤 2：升级 pip
pip install --upgrade pip

# 步骤 3：安装包含 CUDA 的 PyTorch（必须最先安装！）
# 驱动程序 570+：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# 驱动程序 450-569：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# 仅 CPU：
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 步骤 4：安装 WhisperJAV 及所需 extras
pip install "whisperjav[cli] @ git+https://github.com/meizhong986/whisperjav.git"

# 或从本地克隆安装：
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
pip install -e ".[cli]"
```

**重要：** 始终先使用 `--index-url` 安装 PyTorch，然后再安装 WhisperJAV。如果跳过此步骤，pip 将安装仅 CPU 版本的 PyTorch，性能会慢 10-50 倍。

### 方法 3：Conda 环境

```bash
# 步骤 1：创建 conda 环境
conda create -n whisperjav python=3.11 -y
conda activate whisperjav

# 步骤 2：通过 conda 安装 PyTorch（自动处理 CUDA）
conda install pytorch torchaudio pytorch-cuda=12.8 -c pytorch -c nvidia -y
# 或用于 CUDA 11.8：
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 步骤 3：安装 conda 未提供的系统依赖项
conda install ffmpeg -c conda-forge -y

# 步骤 4：安装 WhisperJAV
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
# 或从本地克隆：
cd whisperjav
pip install -e ".[all]"
```

---

## GPU 验证

安装完成后，验证 GPU 支持是否正常工作：

```bash
# 快速检查：CUDA 是否可用？
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('No GPU')"
python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# 完整诊断：
python3 -m whisperjav.utils.preflight_check -v

# 设备检测报告：
python3 -m whisperjav.utils.device_detector
```

GPU 正常工作时的预期输出：

```
CUDA available: True
GPU: NVIDIA GeForce RTX 4090
CUDA version: 12.8
```

如果 CUDA 显示 False，请参阅[故障排除：CUDA 未检测到](#cuda-未检测到)。

---

## 安装特定 Extras

WhisperJAV 使用模块化的 extras 系统。仅安装您需要的功能：

```bash
# 激活 venv 并先安装 PyTorch 之后：

# 仅 CLI（转录，无 GUI）
pip install "whisperjav[cli] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + 翻译
pip install "whisperjav[cli,translate] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + GUI
pip install "whisperjav[cli,gui] @ git+https://github.com/meizhong986/whisperjav.git"

# CLI + Qwen3-ASR（大型模型，需要 8+ GB 显存）
pip install "whisperjav[cli,qwen] @ git+https://github.com/meizhong986/whisperjav.git"

# Unix 优化（CLI + 翻译 + 增强 + huggingface，无 GUI）
pip install "whisperjav[unix] @ git+https://github.com/meizhong986/whisperjav.git"

# 所有功能
pip install "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### 可用 Extras

| Extra | 说明 | 需要的系统依赖项 |
|-------|------|------------------|
| `cli` | 音频处理、语音活动检测、场景检测 | libsndfile |
| `gui` | PyWebView GUI 界面 | libwebkit2gtk-4.0-dev、libgtk-3-dev |
| `translate` | AI 字幕翻译（云端 API）| 无 |
| `llm` | 本地大语言模型服务器（FastAPI）| 无 |
| `enhance` | 语音增强（ClearVoice、BS-RoFormer）| libsndfile |
| `huggingface` | HuggingFace Transformers 集成 | 无 |
| `qwen` | Qwen3-ASR 处理管线（需要 huggingface）| 无（推荐 8+ GB 显存）|
| `analysis` | 可视化和分析工具 | 无 |
| `compatibility` | pyvideotrans 集成 | 无 |
| `all` | 所有功能组合 | 以上全部 |
| `unix` | CLI + 翻译 + 增强 + huggingface + 分析 + 兼容性 | libsndfile |
| `colab` | 针对 Google Colab 优化 | 不适用（Colab 已预装大部分）|
| `kaggle` | 针对 Kaggle 优化 | 不适用 |
| `dev` | 开发工具（pytest、ruff）| 无 |

---

## 无图形界面服务器设置

用于无显示器的服务器（仅 SSH、云端虚拟机、CI/CD）：

```bash
# 步骤 1：安装前提条件（不需要 GUI 包）
sudo apt-get install -y \
    python3 python3-pip python3-venv python3-dev \
    build-essential ffmpeg git libsndfile1

# 步骤 2：创建 venv
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate

# 步骤 3：安装 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 步骤 4：安装 WhisperJAV（unix extra = 无 GUI 依赖项）
pip install "whisperjav[unix] @ git+https://github.com/meizhong986/whisperjav.git"

# 或使用安装程序的最小化标志：
python install.py --minimal
```

**无图形界面运行要点：**
- 使用 `[unix]` extra 或 `[cli]` extra 代替 `[all]` 以跳过 GUI 依赖项
- GUI（`whisperjav-gui`）需要显示服务器和 WebKit2GTK -- 在服务器上请跳过
- CLI 模式（`whisperjav`）完全支持无图形界面运行
- 如果出现 matplotlib 警告，请设置 `MPLBACKEND=Agg`（无显示器用于绘图）

---

## Google Colab 设置

WhisperJAV 包含专用的 Colab 安装程序，可自动处理所有设置。

### 快速入门

在 Colab 笔记本单元格中：

```python
# 单元格 1：克隆并安装
!git clone https://github.com/meizhong986/WhisperJAV.git
!bash WhisperJAV/installer/install_colab.sh
```

```python
# 单元格 2：上传或挂载您的视频
from google.colab import drive
drive.mount('/content/drive')
```

```python
# 单元格 3：转录
!MPLBACKEND=Agg /content/whisperjav_env/bin/whisperjav \
    /content/drive/MyDrive/video.mp4 \
    --mode balanced \
    --sensitivity aggressive
```

### Colab 安装程序执行的操作

1. 安装 `uv` 包管理器（比 pip 快 10-100 倍）
2. 在 `/content/whisperjav_env` 创建隔离的虚拟环境
3. 安装与 Colab GPU 匹配的、包含 CUDA 支持的 PyTorch
4. 安装系统库（portaudio、libsndfile、ffmpeg、libc++）
5. 安装 WhisperJAV 及所有 extras，包括 Qwen3-ASR
6. 尝试从预编译 wheels 安装 llama-cpp-python（可选）

### Colab 使用技巧

- 使用 `MPLBACKEND=Agg` 避免 matplotlib 显示错误
- 挂载 Google Drive 以持久保存输出字幕
- 加载别名文件以使用更短的命令：
  ```bash
  !source /content/whisperjav_aliases.sh
  ```
- 调试模式：`!bash WhisperJAV/installer/install_colab.sh --debug`

---

## Kaggle 设置

与 Colab 类似，但使用基于 pip 的方式：

```python
# 单元格 1：安装
!pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
!pip install "whisperjav[kaggle] @ git+https://github.com/meizhong986/whisperjav.git"

# 单元格 2：验证
!python -c "import whisperjav; print(whisperjav.__version__)"

# 单元格 3：转录
!whisperjav /kaggle/input/your-dataset/video.mp4 --mode balanced
```

---

## 运行应用程序

### CLI 使用

```bash
# 先激活虚拟环境
source ~/.venv/whisperjav/bin/activate

# 基本转录
whisperjav video.mp4

# 指定模式和灵敏度
whisperjav video.mp4 --mode balanced --sensitivity aggressive

# Faster 模式（精度较低，速度快）
whisperjav video.mp4 --mode faster

# 启用语音增强
whisperjav video.mp4 --mode balanced --enhance

# 使用 Qwen3-ASR 处理管线（需要 [qwen] extra）
whisperjav video.mp4 --mode qwen --input-mode assembly

# 带翻译
whisperjav video.mp4 --translate --translate-provider deepseek

# 批量处理（目录中所有 .mp4 文件）
whisperjav /path/to/videos/ --mode balanced

# 指定输出目录
whisperjav video.mp4 --output-dir /path/to/subtitles/

# 帮助
whisperjav --help

# 环境预检
whisperjav --check
```

### GUI 使用

```bash
# 需要 [gui] extra 和 WebKit2GTK
source ~/.venv/whisperjav/bin/activate
whisperjav-gui
```

**注意：** GUI 需要显示服务器（X11 或 Wayland）和 WebKit2GTK。除非使用 X11 转发或 VNC，否则无法通过 SSH 使用。

### 翻译

```bash
# 翻译已有字幕
whisperjav-translate -i subtitles.srt --provider deepseek

# 使用特定指令翻译
whisperjav-translate -i subtitles.srt --provider gemini --instructions standard
```

---

## Systemd 服务设置

用于在服务器上自动化/定时转录：

### 创建服务文件

```bash
sudo tee /etc/systemd/system/whisperjav-batch.service << 'EOF'
[Unit]
Description=WhisperJAV Batch Transcription
After=network.target

[Service]
Type=oneshot
User=your-username
Group=your-group
WorkingDirectory=/home/your-username
Environment="PATH=/home/your-username/.venv/whisperjav/bin:/usr/local/bin:/usr/bin"
Environment="MPLBACKEND=Agg"
ExecStart=/home/your-username/.venv/whisperjav/bin/whisperjav \
    /data/incoming/ \
    --mode balanced \
    --output-dir /data/subtitles/
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
```

### 创建定时器以定期运行

```bash
sudo tee /etc/systemd/system/whisperjav-batch.timer << 'EOF'
[Unit]
Description=Run WhisperJAV batch transcription hourly

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
EOF
```

### 启用并启动

```bash
sudo systemctl daemon-reload
sudo systemctl enable whisperjav-batch.timer
sudo systemctl start whisperjav-batch.timer

# 检查状态
sudo systemctl status whisperjav-batch.timer

# 查看日志
journalctl -u whisperjav-batch.service -f
```

---

## 故障排除

### CUDA 未检测到

**症状：** `torch.cuda.is_available()` 返回 `False`

**诊断步骤：**

```bash
# 步骤 1：检查 NVIDIA 驱动程序是否已加载
nvidia-smi

# 步骤 2：检查 PyTorch 是否安装了 CUDA 版本
python3 -c "import torch; print(torch.version.cuda)"
# 应输出 "12.8" 或 "11.8"，而非 "None"

# 步骤 3：检查驱动程序兼容性
python3 -c "import torch; print(torch.__version__)"
nvidia-smi | head -3
# 将驱动程序版本与 CUDA 要求进行比较
```

**常见原因和解决方案：**

| 原因 | 解决方案 |
|------|----------|
| 安装了仅 CPU 版本的 PyTorch | `pip uninstall torch torchaudio && pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128` |
| 驱动程序对于 CUDA 12.8 过旧 | 更新驱动程序：`sudo apt install nvidia-driver-570` 或使用 `--cuda118` |
| 未安装 NVIDIA 驱动程序 | 安装驱动程序（见 [NVIDIA 驱动程序设置](#nvidia-驱动程序和-cuda-设置)）|
| 在容器中运行但未传递 GPU | 向 Docker 传递 `--gpus all`：`docker run --gpus all ...` |
| 加载了 Nouveau 驱动程序而非 nvidia | 禁用 nouveau：`echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf && sudo update-initramfs -u && sudo reboot` |

### 库未找到错误

**`OSError: sndfile library not found`**

```bash
# Ubuntu/Debian
sudo apt-get install -y libsndfile1 libsndfile1-dev

# Fedora/RHEL
sudo dnf install -y libsndfile libsndfile-devel

# Arch
sudo pacman -S libsndfile
```

**`ModuleNotFoundError: No module named '_tkinter'`**

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-tk

# Fedora
sudo dnf install -y python3-tkinter
```

**`ImportError: libwebkit2gtk-4.0.so: cannot open shared object file`**

GUI 需要 WebKit2GTK。仅使用 CLI 时不需要。

```bash
# Ubuntu/Debian
sudo apt-get install -y libwebkit2gtk-4.0-dev

# Fedora
sudo dnf install -y webkit2gtk4.0-devel
```

### 权限拒绝

**`ERROR: Could not install packages due to an EnvironmentError: [Errno 13] Permission denied`**

您正在尝试在没有虚拟环境的情况下安装到系统 Python。请创建虚拟环境：

```bash
python3 -m venv ~/.venv/whisperjav
source ~/.venv/whisperjav/bin/activate
# 然后重试安装
```

**`error: externally-managed-environment`**（PEP 668）

同样的解决方案 -- 创建并激活虚拟环境。此错误出现在 Debian 12+、Ubuntu 24.04+ 及类似的现代发行版上。

如果 `python3 -m venv` 失败并提示 "No module named venv"：

```bash
# Ubuntu/Debian
sudo apt-get install -y python3-venv
# 或对于特定版本：
sudo apt-get install -y python3.11-venv
```

### Git 超时/网络问题

**症状：** `Failed to connect to github.com port 443 after 21 ms`

这在中国大陆防火墙（GFW）后或 VPN 连接慢时常见。

```bash
# 选项 1：安装程序在重试时自动配置 Git 超时
# 只需重新运行安装程序 -- 它会检测并处理此问题

# 选项 2：手动配置 Git
git config --global http.connectTimeout 120
git config --global http.timeout 300
git config --global http.maxRetries 5

# 选项 3：使用代理
export https_proxy=http://your-proxy:port
export http_proxy=http://your-proxy:port
```

### PyTorch 版本不匹配

**症状：** `RuntimeError: CUDA error: CUBLAS_STATUS_NOT_INITIALIZED` 或类似错误

```bash
# 检查当前版本
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# 重新安装匹配版本
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### numba / llvmlite 错误

**症状：** `ImportError: numba needs NumPy 1.x` 或 `Cannot import llvmlite`

```bash
# 重新安装 numpy 和 numba
pip install "numpy>=2.0.0"
pip install --force-reinstall "numba>=0.60.0"
```

### 语音增强安装失败

**症状：** modelscope / clearvoice 安装失败

```bash
# 这些包是可选的。不安装它们也可以：
python install.py --no-speech-enhancement

# 或单独安装特定后端：
pip install clearvoice       # 仅 ClearVoice
pip install bs-roformer-infer # 仅 BS-RoFormer
```

### 内存不足（OOM）

**症状：** 转录时出现 `CUDA out of memory`

```bash
# 使用较小的模型
whisperjav video.mp4 --mode faster

# 减少 Qwen 处理管线的批量大小
whisperjav video.mp4 --mode qwen --input-mode vad_slicing

# 监控 GPU 显存
watch -n 1 nvidia-smi
```

---

## 性能调优

### 显存管理

| GPU 显存 | 推荐设置 |
|----------|----------|
| 4 GB | 仅 `--mode faster` |
| 6 GB | `--mode fast` 或 `--mode balanced` 配合小模型 |
| 8 GB | `--mode balanced --sensitivity balanced` |
| 12 GB | `--mode balanced --sensitivity aggressive` |
| 16+ GB | 所有模式，包括 `--mode qwen --input-mode assembly` |
| 24+ GB | 所有模式，大批量大小 |

### Qwen3-ASR 专项调优

Qwen3-ASR 需要较大的显存。根据 GPU 选择输入模式：

| 输入模式 | 显存占用 | 质量 | 速度 |
|----------|----------|------|------|
| `assembly` | 最高（文本生成与对齐分离）| 长场景最佳 | 中等 |
| `context_aware` | 高（耦合 ASR + 对齐）| 对话最佳 | 较慢 |
| `vad_slicing` | 较低（短片段）| 适合噪声音频 | 最快 |

```bash
# Assembly 模式（推荐用于 16+ GB 显存）
whisperjav video.mp4 --mode qwen --input-mode assembly

# VAD slicing 模式（用于 8 GB 显存）
whisperjav video.mp4 --mode qwen --input-mode vad_slicing
```

### 环境变量

```bash
# 限制 GPU 显存使用（总显存的比例）
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# 在多 GPU 系统中使用特定 GPU
export CUDA_VISIBLE_DEVICES=0

# 禁用 matplotlib 显示（无图形界面服务器）
export MPLBACKEND=Agg

# 在 Ampere+ GPU 上启用 TF32 以加速推理
export TORCH_ALLOW_TF32=1
```

### 批量处理优化

处理多个文件时：

```bash
# 处理所有 .mp4 文件
whisperjav /path/to/videos/ --mode balanced

# 使用 screen/tmux 处理长时间运行的任务
tmux new -s whisperjav
whisperjav /path/to/videos/ --mode balanced --sensitivity aggressive
# 按 Ctrl+B, D 分离；tmux attach -t whisperjav 重新连接
```

---

## 卸载

```bash
# 删除虚拟环境
rm -rf ~/.venv/whisperjav

# 删除缓存的模型（可选，节省磁盘空间）
rm -rf ~/.cache/whisper
rm -rf ~/.cache/huggingface

# 删除桌面条目（如已创建）
rm -f ~/.local/share/applications/whisperjav.desktop

# 删除源码（如已克隆）
rm -rf ~/whisperjav  # 根据实际路径调整
```

---

## 附录：架构概览

### 安装流程

```
install_linux.sh（薄包装器）
    |
    v
install.py（编排器）
    |
    +-- 预检（磁盘、网络）
    +-- detect_gpu() --> CUDA 版本选择
    +-- 步骤 1：pip 升级
    +-- 步骤 2：PyTorch（通过 --index-url 锁定 GPU 版本）
    +-- 步骤 3：核心依赖项（numpy、scipy、numba、音频库）
    +-- 步骤 4：Whisper 包（openai-whisper、stable-ts、faster-whisper）
    +-- 步骤 5：可选包（HuggingFace、Qwen、翻译、语音活动检测、语音增强、GUI）
    +-- 步骤 6：WhisperJAV（--no-deps 以保留 GPU 版 torch）
    +-- 验证
```

### 为什么必须先安装 PyTorch

PyPI 上的 PyTorch 是仅 CPU 版本。如果直接运行 `pip install whisperjav`，pip 会从 PyPI 解析 torch 并安装仅 CPU 版本，导致推理速度慢 10-50 倍。通过先使用 `--index-url https://download.pytorch.org/whl/cu128` 安装 torch，GPU 版本会被"锁定"，后续包会将其视为已满足。

### 包注册表

所有包定义位于 `whisperjav/installer/core/registry.py`。这是以下信息的唯一真实来源：
- 包名称和版本
- 安装顺序（PyTorch 优先，numba 在 numpy 之后等）
- 每个包属于哪个 extras
- 平台特定的包（仅 Windows、仅 Linux）
- 导入名称映射（例如 `opencv-python` 导入为 `cv2`）

添加或修改依赖项时，请更新注册表并运行验证：

```bash
python -m whisperjav.installer.validation
```
