# WhisperJAV Windows 安装程序指南

**版本：** 1.10.0
**安装程序：** `WhisperJAV-1.10.0-Windows-x86_64.exe`（无需管理员权限）

---

## 目录

1. [安装前准备](#before-you-install)
2. [安装步骤](#installation-steps)
3. [安装为哪种加速器制作](#what-this-install-is-made-for)
4. [首次运行](#first-run)
5. [检查安装](#checking-your-installation)
6. [从 1.9.x 升级](#coming-from-19x)
7. [在 1.10 内更新](#updating-within-110)
8. [再次运行安装程序](#running-the-installer-again)
9. [卸载](#uninstalling)
10. [故障排除](#troubleshooting)
11. [技术细节](#technical-details)

---

## 安装前准备 {#before-you-install}

### 你的电脑

| | 要求 |
|---|---|
| Windows | Windows 10 或 11，64 位（x86-64）。不支持 Windows on ARM（例如 Snapdragon 笔记本）：安装程序会在第一个界面停止。 |
| 内存 | 8 GB（推荐 16 GB） |
| 可用磁盘空间 | 约 10 GB，用于安装和首次使用时下载的模型 |
| 网络 | 安装时以及每个模型首次运行时 |
| WebView2 | Microsoft Edge WebView2 Runtime，供 GUI 使用（缺失时安装程序会提供） |

### 你的显卡

安装程序会把这次安装制作为一种加速器的安装（见[是否加速](../getting-started/index.md#accelerated-or-not)）：

- **1.10 支持的 NVIDIA GPU**——GTX 16 系列、RTX 20 系列或更新，驱动 580 或更新：安装自动为 NVIDIA GPU（CUDA）制作，不会提问。
- **其他情况**——AMD 或 Intel 显卡、比 1.10 支持范围更旧的 NVIDIA 显卡、较旧的驱动，或没有 WhisperJAV 能用的显卡：会弹出一个窗口，询问这次安装要为什么制作。

查看 NVIDIA 驱动版本：右键点击桌面，打开 NVIDIA 控制面板，在左下角查看"驱动程序版本"。更新驱动：[nvidia.com/drivers](https://www.nvidia.com/drivers)，然后重启电脑。

### 杀毒软件

一些杀毒软件会阻止安装程序下载。遇到这种情况时，请为 `%LOCALAPPDATA%\WhisperJAV` 添加排除项，或在安装期间暂停实时扫描。

### 中国用户

安装程序会从 GitHub、PyPI、download.pytorch.org 和 Hugging Face 下载。请在整个安装过程中保持 VPN 开启。

---

## 安装步骤 {#installation-steps}

1. 从[发布页面](https://github.com/meizhong986/WhisperJAV/releases)下载 `WhisperJAV-1.10.0-Windows-x86_64.exe`。
2. 双击运行。无需以管理员身份运行。如果 Windows SmartScreen 发出警告，点击 **更多信息**，再点击 **仍要运行**。
3. 接受许可协议。
4. 选择 **Just Me**（推荐；无需管理员权限）或 **All Users**。
5. 选择文件夹。默认为 `C:\Users\<你的用户名>\AppData\Local\WhisperJAV`。路径尽量短，尽量不含空格。
6. 点击 **Install**。

文件解压后，一个控制台窗口会完成其余安装，并显示各个阶段：

| 阶段 | 内容 |
|---|---|
| 1 | 检查：磁盘剩余空间、到 PyPI 的连接、Visual C++ 运行库、WebView2 |
| 2 | 查找 NVIDIA GPU 及其驱动 |
| 2.5 | 决定安装为哪种加速器制作；需要询问时显示窗口（见下文） |
| 3–4.5 | 安装本版本测试过的软件包版本，然后安装适合该加速器的 PyTorch |
| 5 | 安装 WhisperJAV |
| 5.1 | 仅 NVIDIA GPU（CUDA）安装：放置 balanced、fast 和 faster 需要的 CUDA 12 库；如果这台电脑无法加载它，改为安装 PyTorch 的 CUDA 12.8 版 |
| 5.5–5.8 | 创建 `WhisperJAV-GUI.exe` 启动器，下载 FireRedVAD 语音检测模型，检查图标 |
| 6 | 检查已安装的程序能否启动 |

最后会显示一份摘要，说明安装了什么以及安装为哪种加速器制作，并创建桌面快捷方式 "WhisperJAV v1.10.0"。安装需要约 10 到 30 分钟，取决于网速。

---

## 安装为哪种加速器制作 {#what-this-install-is-made-for}

### 受支持的 NVIDIA GPU

不会提问。安装为 NVIDIA GPU（CUDA 13.0）制作。如果这台电脑无法加载 balanced、fast 和 faster 需要的 CUDA 12 库，安装程序会用带边框的消息说明，并改为安装 CUDA 12.8 版。

### 没有受支持的 NVIDIA GPU：窗口

如果安装程序找不到 1.10 支持的 NVIDIA GPU，会弹出标题为 **"WhisperJAV — no supported NVIDIA GPU found"** 的窗口并询问：

> WhisperJAV found no NVIDIA graphics card that version 1.10 supports (graphics found: …). What should this install be made for?

（意思是：WhisperJAV 没有找到 1.10 支持的 NVIDIA 显卡。这次安装要为什么制作？）

| 选项 | 安装为哪种加速器制作 |
|---|---|
| AMD graphics card（AMD 显卡） | 通过 Vulkan 使用 GPU |
| Intel graphics card（Intel 显卡） | 通过 Vulkan 使用 GPU |
| NVIDIA graphics card older than WhisperJAV 1.10 supports（比 1.10 支持范围更旧的 NVIDIA 显卡） | 通过 Vulkan 使用 GPU |
| I have my own PyTorch for my GPU (advanced)（我有适合自己 GPU 的 PyTorch，进阶） | 自己的 PyTorch |
| None, or use the CPU only（没有，或仅使用 CPU） | 仅 CPU（不加速） |

窗口中的说明（意思）：使用 AMD、Intel 或较旧的 NVIDIA 显卡时，GGUF 处理管线（Parakeet-CrispASR 和 Cohere-CrispASR）请求通过 Vulkan 使用你的显卡。这在 1.10.0 中是实验性功能：尚未在 AMD 或 Intel 显卡上测试，我们希望听到它在你电脑上的表现。其他所有处理管线在 CPU 上运行（不加速），比在受支持的 NVIDIA GPU 上慢好几倍。选择自己的 PyTorch 时，WhisperJAV 安装除 PyTorch 以外的一切，之后由你添加自己的 PyTorch 版本（安装结束时会显示命令）。选择仅 CPU 时，所有处理管线都在 CPU 上运行。

如果 20 秒内没有回答，会安装仅 CPU 版本。**Stop** 会结束安装程序，不安装任何内容。

通过 Vulkan 使用 GPU 需要 Vulkan 运行库（`vulkan-1.dll`），它随显卡驱动一起安装。如果缺失，安装程序会说明并安装仅 CPU 版本；更新显卡驱动后再次运行安装程序即可改变。

### 1.10 无法使用的 NVIDIA GPU

窗口会先说明原因，并预先选中 **Stop**，没有时间限制：

- **较旧的显卡**（低于 GTX 16 / RTX 20）："Your graphics card, … (compute capability …), is older than WhisperJAV 1.10 supports. 1.10 needs a GTX 16-series, RTX 20-series or newer card." 以及 "WhisperJAV 1.9.5 still supports this card: https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5."（意思是：你的显卡比 1.10 支持的更旧；1.9.5 仍然支持这张显卡。）如果你确定显卡受支持，**My card is identified wrongly**（我的显卡被认错了）会显示报告问题的地址，然后停止。
- **较旧的驱动**（低于 580）："Your NVIDIA driver (…) is older than WhisperJAV 1.10 needs (580 or newer). Update it from https://www.nvidia.com/drivers, restart the PC, and run this installer again to get the NVIDIA GPU version. Or choose below."（意思是：驱动太旧；更新驱动并重启后再次运行安装程序，即可得到 NVIDIA GPU 版本；或在下面另选。）
- **Windows 能列出、但 WhisperJAV 无法访问的显卡**——驱动可能缺失或不正常，或者在有两块显示芯片的笔记本上，NVIDIA 芯片可能已关闭：安装或修复 NVIDIA 驱动（580 或更新），重启电脑后再次运行安装程序。或在下面另选。

### 自己的 PyTorch

WhisperJAV 安装除 PyTorch 以外的一切，并且在安装和升级时从不在这个安装中安装、替换或删除 PyTorch。再次运行安装程序会替换整个安装文件夹，包括 PyTorch：窗口会保留"自己的 PyTorch"并再次显示命令。安装结束时，一个窗口会显示要运行的那一条命令，其中已填好 WhisperJAV 的 Python：

```
"<安装文件夹>\python.exe" -m pip install torch torchaudio --index-url <你的 PyTorch 版本的地址>
```

推荐 PyTorch 2.11。同一条命令也保存在 `<安装文件夹>\OWN_PYTORCH_NEXT_STEP.txt` 中。在你运行它之前，每次 WhisperJAV 运行都会在开始时停止，并再次显示这条命令。

### 安装的选择保存在哪里

安装程序把加速器写入安装文件夹中的 `whisperjav_stack.txt`。每次运行、`whisperjav-upgrade` 以及以后再次运行安装程序都遵循它。

### 安装程序有意停止时

如果你选择 Stop，一个结束对话框会显示："WhisperJAV was not installed. The installer stopped for the reason shown in its console window."（意思是：WhisperJAV 未安装；停止原因见控制台窗口。）原因也保存在安装文件夹中的 `INSTALLATION_STOPPED_v1.10.0.txt`。

---

## 首次运行 {#first-run}

双击 **WhisperJAV v1.10.0** 快捷方式，或安装文件夹中的 `WhisperJAV-GUI.exe`。

处理管线第一次需要某个模型时，WhisperJAV 会下载它（例如 Whisper large-v3 约 3 GB，Parakeet-CrispASR 674 MB）。之后的运行会直接使用。

每次运行开始时，WhisperJAV 会检查安装：

- **NVIDIA GPU（CUDA）安装，GPU 可用：** 不显示任何内容，运行继续。
- **NVIDIA GPU（CUDA）安装，GPU 现在不可用**（例如更换驱动之后）：运行不会开始。一个框会说明原因和解决办法。要这次在 CPU 上运行，加上 `--device cpu`；在 GUI 中，对话框 **"The NVIDIA GPU cannot be used"** 提供 **Run on the CPU this time**（这次在 CPU 上运行）。
- **其他所有安装：** 一个框说明安装为哪种加速器制作，以及各部分在这台电脑上的运行位置，然后运行继续。

基本流程：添加文件，选择模式（Balanced 是不错的起点），点击 **Start**。字幕会写入你选择的输出文件夹。

---

## 检查安装 {#checking-your-installation}

打开命令提示符（按开始键，输入 `cmd`，按 Enter）并运行：

```cmd
cd /d "%LOCALAPPDATA%\WhisperJAV"
Scripts\whisperjav.exe --check
```

（如果你选择了其他安装文件夹，请改用那个文件夹。）

它会列出 Python、PyTorch、安装的加速器、各部分在哪里运行、FFmpeg、CrispASR 和模型。在 NVIDIA GPU（CUDA）安装上，例如：

```
✓ PyTorch: PyTorch 2.11.0+cu130: NVIDIA GPU (CUDA 13.0), NVIDIA GeForce RTX 3060
✓ CrispASR: 0.8.44+cuda (CUDA build), GGUF pipelines: asks for the NVIDIA GPU (CUDA)
ℹ Accelerator: Accelerator of this install: NVIDIA GPU (CUDA 13.0).
ℹ CTranslate2: CTranslate2 (balanced, fast, faster): NVIDIA GPU (CUDA)
```

`Scripts\whisperjav.exe --version` 显示版本：`WhisperJAV 1.10.0`。

安装日志是安装文件夹中的 `install_log_v1.10.0.txt`。

---

## 从 1.9.x 升级 {#coming-from-19x}

请全新安装 1.10.0：1.10.0 需要 Python 3.13，因此 `whisperjav-upgrade` 无法把 1.9.x 安装升级到 1.10.0。如果安装到旧版本所在的文件夹，安装程序会发现它并提出替换：该文件夹中的所有内容会先被删除。`%APPDATA%\WhisperJAV` 中的设置和已下载的模型会保留。

---

## 在 1.10 内更新 {#updating-within-110}

以后的 1.10 版本可以用 `whisperjav-upgrade` 安装，它会保留安装的加速器及其 PyTorch：

1. 打开命令提示符。
2. `cd /d "%LOCALAPPDATA%\WhisperJAV"`（或你的安装文件夹）。
3. `Scripts\whisperjav-upgrade.exe`
4. 检查版本：`Scripts\whisperjav.exe --version`

不要用普通的 `pip install -U whisperjav…` 更新：在 Windows 上，PyPI 提供的是仅 CPU 版的 PyTorch，可能会在毫无提示的情况下替换你的 GPU 版本。

---

## 再次运行安装程序 {#running-the-installer-again}

把安装程序再次运行到同一个文件夹，可以修复安装或更改它为哪种加速器制作。标题为 **"WhisperJAV — this install's accelerator"** 的窗口会显示 "This install is made for: …. Keep it, or choose another."（意思是：这次安装为……制作；保留它，或另选一个。）当前加速器已预先选中；20 秒内没有回答时，安装保持原样。如果当时 NVIDIA GPU 不可用，原因会显示在选项上方。安装程序会先删除旧的安装文件夹（设置和已下载的模型保留），只保留它所记录的加速器；在"自己的 PyTorch"安装中，请用安装结束时显示的命令重新添加你的 PyTorch。

---

## 卸载 {#uninstalling}

打开 Windows **设置 > 应用**，找到 **WhisperJAV**，选择 **卸载**。下载的模型保留在 `%USERPROFILE%\.cache`（Hugging Face 和 Whisper 缓存），设置保留在 `%APPDATA%\WhisperJAV`；如需释放空间，请手动删除这些文件夹。

---

## 故障排除 {#troubleshooting}

**有 NVIDIA 显卡，却显示 "No supported NVIDIA GPU found"。** 请阅读窗口的第一行：它说明了原因（显卡较旧、驱动较旧，或 WhisperJAV 无法访问的显卡）。更新或修复驱动，重启电脑后再次运行安装程序。在有两块显示芯片的笔记本上，请确认 NVIDIA 芯片已开启。

**运行停止："THE NVIDIA GPU CANNOT BE USED"。** 安装为 NVIDIA GPU 制作，而 GPU 现在不可用。框中会说明原因和解决办法。`--device cpu` 让这一次在 CPU 上运行。

**处理很慢。** 查看运行开始时的框，或运行 `whisperjav --check`：它们会说明各部分在哪里运行。在仅 CPU 安装中，所有处理管线都在 CPU 上运行，比在 GPU 上慢好几倍。

**GGUF 处理管线停止："CrispASR could not be loaded"。** 在"通过 Vulkan 使用 GPU"或"自己的 PyTorch"安装中，这通常意味着缺少 Vulkan 运行库（`vulkan-1.dll`）：更新显卡驱动，或再次运行安装程序并选择仅 CPU。

**下载失败或超时。** 检查网络连接；在防火墙后面时，允许 `python.exe`、`uv.exe` 和 `git.exe` 访问网络。安装程序会用更长的超时重试 GitHub 下载。在中国，请在整个安装过程中保持 VPN 开启。

**GUI 无法打开，或窗口空白。** 从 [Microsoft](https://go.microsoft.com/fwlink/p/?LinkId=2124703) 安装 WebView2。要查看错误信息，在命令提示符中运行 `"%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.webview_gui.main`。

**安装失败。** 打开安装文件夹中的 `install_log_v1.10.0.txt`：它会指出失败的阶段。`INSTALLATION_FAILED_v1.10.0.txt` 记录了原因。再次运行安装程序；它会提出替换不完整的安装。如果仍然失败，请在 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题，并附上日志。

---

## 技术细节 {#technical-details}

### 静默安装

```cmd
cmd /C START /WAIT WhisperJAV-1.10.0-Windows-x86_64.exe /S /D=C:\WhisperJAV
```

`/S` 不显示安装程序的界面进行安装；`/D=` 设置文件夹，必须放在最后。`WhisperJAV-1.10.0-Windows-x86_64.exe /?` 列出所有选项。需要询问时，[安装为哪种加速器制作](#what-this-install-is-made-for)中的窗口仍会打开：在没有受支持 NVIDIA GPU 的电脑上，20 秒内没有回答时安装仅 CPU 版本；有 1.10 无法使用的 NVIDIA GPU 时，它会一直等待回答。

### 安装文件夹中的文件

| 文件 | 说明 |
|---|---|
| `install_log_v1.10.0.txt` | 安装日志 |
| `whisperjav_stack.txt` | 安装为哪种加速器制作 |
| `OWN_PYTORCH_NEXT_STEP.txt` | 仅"自己的 PyTorch"安装：添加你的 PyTorch 的命令 |
| `INSTALLATION_STOPPED_v1.10.0.txt` | 仅当安装程序有意停止时：原因 |
| `INSTALLATION_FAILED_v1.10.0.txt` | 仅当安装失败时：原因 |
| `WhisperJAV-GUI.exe` | GUI 启动器 |

### 安装程序访问的地址

| 地址 | 用途 |
|---|---|
| `pypi.org`、`files.pythonhosted.org` | Python 软件包 |
| `download.pytorch.org` | PyTorch |
| `github.com` | CrispASR 的 GPU 版本，以及从 GitHub 安装的软件包 |
| `huggingface.co` | 模型（安装时和首次运行时） |
| `aka.ms`、`go.microsoft.com` | Visual C++ 运行库和 WebView2（缺失时） |
