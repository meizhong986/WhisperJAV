# WhisperJAV Windows 独立安装程序安装指南

**版本：** 1.8.9
**最后更新：** 2026-03-19
**安装程序类型：** 独立 .exe（无需管理员权限）

---

## 系统要求

### 最低要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 10（64 位）或 Windows 11 |
| 内存 | 8 GB |
| 可用磁盘空间 | 8 GB（安装 + AI 模型 + 缓存）|
| 网络 | 安装过程中需要联网以下载依赖项 |
| 运行时 | Microsoft Edge WebView2（安装程序会在缺失时提示安装）|

### 推荐配置（最佳性能）

| 组件 | 推荐 |
|------|------|
| GPU | NVIDIA GPU，6+ GB 显存（RTX 2060 或更好）|
| GPU 驱动程序 | NVIDIA 驱动程序 570+（支持 CUDA 12.8）|
| 内存 | 16 GB |
| 存储 | SSD（处理速度显著提升）|
| 网络 | 宽带连接（用于首次下载模型）|

### GPU 兼容性

WhisperJAV 使用 NVIDIA CUDA 进行 GPU 加速转录。安装程序会自动检测您的 GPU 并安装相应版本：

| 您的 NVIDIA 驱动程序版本 | 安装的 CUDA 版本 | 兼容的 GPU |
|--------------------------|-----------------|------------|
| 570 或更新 | CUDA 12.8（最优）| RTX 20xx、30xx、40xx、50xx 系列 |
| 450 - 569 | CUDA 11.8（通用）| 所有支持 CUDA 的 NVIDIA GPU |
| 无 NVIDIA GPU | 仅 CPU（较慢）| 任何系统（比 GPU 慢 6-10 倍）|

如果您没有 NVIDIA GPU，WhisperJAV 仍然可以在纯 CPU 模式下工作。处理速度会较慢（每小时视频大约需要 30-60 分钟，而使用 GPU 仅需 5-10 分钟）。

### 支持的 NVIDIA GPU

- **GeForce：** GTX 1060+、RTX 2060+、RTX 3060+、RTX 4060+、RTX 5070+
- **Quadro / RTX Professional：** Quadro RTX、RTX A 系列
- **数据中心：** T4、A10、A100、H100、B100

---

## 安装前准备

### 1. 更新 NVIDIA 驱动程序（推荐）

为获得最佳性能，请将 NVIDIA 驱动程序更新至 570 或更新版本：

1. 访问 [nvidia.com/drivers](https://www.nvidia.com/drivers)
2. 选择您的 GPU 型号
3. 下载并安装最新的 "Game Ready" 或 "Studio" 驱动程序
4. 安装后重启计算机

您可以通过以下方式检查当前驱动程序版本：
- 右键点击桌面，选择"NVIDIA 控制面板"
- 查看左下角的"驱动程序版本"

### 2. 确认 WebView2 已安装

WhisperJAV 使用 Microsoft Edge WebView2 作为图形界面。大多数 Windows 10 和 11 系统已预装此组件。如果您不确定，安装程序会自动检查并在需要时提供下载链接。

手动检查方法：
1. 打开 Windows 设置中的"添加或删除程序"
2. 搜索 "WebView2"
3. 如果显示 "Microsoft Edge WebView2 Runtime"，说明已就绪

如未安装，请从以下地址下载：
[https://go.microsoft.com/fwlink/p/?LinkId=2124703](https://go.microsoft.com/fwlink/p/?LinkId=2124703)

### 3. 释放磁盘空间

确保 WhisperJAV 安装目标磁盘至少有 8 GB 可用空间。空间大致分配如下：

| 组件 | 大小 |
|------|------|
| 基础安装（Python + conda）| ~500 MB |
| 包含 CUDA 的 PyTorch | ~2.5 GB |
| Python 依赖项 | ~1.5 GB |
| AI 模型（首次使用时下载）| ~3 GB |
| **合计** | **~7.5 GB** |

### 4. 临时排除杀毒软件（如有需要）

部分杀毒软件可能会标记安装程序或安装后的下载内容。如果遇到问题：

1. 临时禁用实时扫描，或
2. 为 `%LOCALAPPDATA%\WhisperJAV` 添加排除规则

---

## 安装步骤

### 步骤 1：下载安装程序

从 [GitHub Releases 页面](https://github.com/meizhong986/whisperjav/releases) 下载 `WhisperJAV-1.8.9-Windows-x86_64.exe`。

安装程序文件大小约为 150 MB。

### 步骤 2：运行安装程序

双击下载的 `.exe` 文件。**不需要**以管理员身份运行。

如果 Windows SmartScreen 显示警告：
1. 点击"更多信息"
2. 点击"仍要运行"

### 步骤 3：接受许可协议

阅读并接受 MIT 许可协议以继续。

### 步骤 4：选择安装类型

- **仅为当前用户安装（推荐）：** 仅为当前用户安装，无需管理员权限。
- **为所有用户安装：** 为计算机上的所有用户安装，需要管理员权限。

### 步骤 5：选择安装位置

默认安装位置为：
```
C:\Users\[用户名]\AppData\Local\WhisperJAV
```

您可以更改为任意位置。请注意以下建议：
- 尽量避免路径中包含空格（例如 "Program Files"）
- 除非已启用长路径支持，否则避免路径超过 46 个字符
- 避免路径中包含特殊字符（重音字母、中日韩字符）

### 步骤 6：安装选项

安装程序将显示可选设置：
- **添加到 PATH：** 默认启用。允许从命令行运行 `whisperjav`，并支持后续通过 `pip install -U` 进行升级。
- **创建快捷方式：** 在桌面创建快捷方式以便快速访问。

点击"安装"开始。

### 步骤 7：安装后配置（自动执行）

基础环境解压完成后，安装后脚本会自动运行。这是安装过程中耗时最长的部分。一个控制台窗口将显示进度：

#### 阶段 1：预检（< 1 分钟）
- 磁盘空间验证（最低 8 GB）
- 网络连接检查（到 PyPI）
- Visual C++ Redistributable 检查（缺失时自动安装）
- WebView2 运行时检查（缺失时提示下载）

#### 阶段 2：GPU 检测（< 1 分钟）
- 检测 NVIDIA GPU 及驱动程序版本
- 选择适当的 CUDA 版本

#### 阶段 3：PyTorch 安装（3-5 分钟）
- 下载并安装包含 CUDA 支持的 PyTorch（~2.5 GB）
- 如果未检测到 GPU，系统将询问是否安装仅 CPU 版本的 PyTorch

#### 阶段 3.5：核心 Whisper 包（2-3 分钟）
- 安装 OpenAI Whisper、Stable-TS、FFmpeg-Python（来自 GitHub）
- 这些包使用 git，对于防火墙后的用户可能触发 Git 超时处理

#### 阶段 4：Python 依赖项（3-5 分钟）
- 从需求文件安装所有剩余 Python 包
- 约 500 MB 的下载量

#### 阶段 5：WhisperJAV 应用程序（< 1 分钟）
- 从捆绑的 wheel 包安装 WhisperJAV 应用程序

#### 阶段 5.3：本地大语言模型翻译（可选，交互式）
- 系统将提示安装本地大语言模型以进行离线翻译
- 输入 "Y" 安装（推荐）或 "N" 跳过
- 如果 30 秒内未响应，默认选择 "Y"
- 如果现在跳过，之后仍可安装

#### 阶段 5.5-5.8：启动器和图标设置（< 1 分钟）
- 在安装目录中创建 `WhisperJAV-GUI.exe` 启动器
- 验证图标文件

所有阶段完成后，桌面上将创建名为 "WhisperJAV v1.8.9" 的快捷方式。

### 步骤 8：完成

安装摘要将显示：
- 安装目录
- Python 版本
- PyTorch 版本和 CUDA 状态
- WebView2 状态
- 安装耗时

按 Enter 关闭安装程序窗口。

**总安装时间：** 根据网速和硬件情况，约 10-20 分钟。

---

## 首次运行体验

### 启动 WhisperJAV

双击桌面上的 **"WhisperJAV v1.8.9"** 快捷方式。或者，双击安装目录中的 `WhisperJAV-GUI.exe`。

### 首次转录：AI 模型下载

首次处理视频时，WhisperJAV 需要下载 AI 模型。这是一次性操作：

| 模型 | 大小 | 下载时间 |
|------|------|----------|
| Whisper Large-v3 | ~3 GB | 5-10 分钟 |

下载进度将在 GUI 中显示。初次下载后，模型将缓存在本地，无需再次下载。

模型存储位置：
```
C:\Users\[用户名]\.cache\whisper\
```

### 基本工作流程

1. **添加文件：** 点击 "Add File(s)" 或拖放视频/音频文件
2. **选择模式：** 选择 "Balanced" 获得最佳质量，选择 "Faster" 获得更快速度
3. **开始处理：** 点击 "Start"
4. **找到字幕：** 输出的 SRT 文件保存在输入文件旁边的 `_output` 文件夹中

---

## 验证安装

### 检查 GPU 加速

安装完成后，从安装目录打开命令提示符并运行：

```cmd
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

GPU 系统的预期输出：
```
CUDA available: True
GPU: NVIDIA GeForce RTX 4070
```

### 检查 WhisperJAV 版本

```cmd
python -c "from whisperjav.__version__ import __version__; print(f'WhisperJAV {__version__}')"
```

预期输出：
```
WhisperJAV 1.8.9
```

### 检查安装日志

安装日志位于：
```
[安装目录]\install_log_v1.8.9.txt
```

此文件包含安装过程中每个步骤的详细信息，对于诊断问题非常有用。

---

## 从旧版本升级

### 自动升级检测

如果将 WhisperJAV v1.8.9 安装到与旧版本相同的目录，安装程序将：

1. 检测现有的 WhisperJAV 安装
2. 询问是否要替换
3. 确认后，将在安装前干净地删除旧版本

升级期间，`%APPDATA%\WhisperJAV` 中的用户配置将被保留。

### 通过 pip 手动升级

如果已启用 PATH（默认设置），无需重新下载安装程序即可升级：

```cmd
pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"
```

### 保留配置

用户设置存储在：
```
%APPDATA%\WhisperJAV\whisperjav_config.json
```

此文件在升级或重新安装时不会被删除。如果您从 v1.8.0 之前的版本升级，安装程序将自动从旧位置迁移您的配置。

---

## 卸载

### 方法 1：Windows 设置（推荐）

1. 打开 Windows 设置 > 应用 > 应用和功能
2. 搜索 "WhisperJAV"
3. 点击"卸载"
4. 按提示操作

### 方法 2：卸载脚本

运行安装目录中的 `uninstall_v1.8.9.bat` 文件。此交互式脚本将：

1. 删除桌面快捷方式
2. 删除开始菜单快捷方式
3. 询问是否删除用户配置文件
4. 询问是否删除缓存的 AI 模型（~3 GB）
5. 删除安装目录

### 方法 3：手动删除

1. **删除安装目录：**
   ```
   %LOCALAPPDATA%\WhisperJAV
   ```
   （或您自定义的安装位置）

2. **删除桌面快捷方式：**
   ```
   %USERPROFILE%\Desktop\WhisperJAV v1.8.9.lnk
   ```

3. **（可选）删除缓存的 AI 模型**以释放约 3 GB 空间：
   ```
   %USERPROFILE%\.cache\whisper
   ```

4. **（可选）删除用户配置：**
   ```
   %APPDATA%\WhisperJAV
   ```

5. **（可选）清理 PATH 条目：**
   如果安装时启用了 "Add to PATH"，请从用户 PATH 中删除以下条目：
   - `[安装目录]\Scripts`
   - `[安装目录]\Library\bin`

---

## 故障排除

### 安装问题

#### "NVIDIA driver not found" 或 "No NVIDIA GPU detected"

**原因：** 未安装 NVIDIA GPU，或驱动程序未安装/版本过旧。

**解决方案：**
- 如果您有 NVIDIA GPU，请从 [nvidia.com/drivers](https://www.nvidia.com/drivers) 下载并安装最新驱动程序
- 如果没有 NVIDIA GPU，在提示时接受仅 CPU 安装。处理速度会较慢但功能完全正常。

#### "WebView2 runtime not detected"

**原因：** 未安装 Microsoft Edge WebView2。图形界面需要此组件。

**解决方案：**
1. 安装程序将自动打开下载页面
2. 从 [Microsoft](https://go.microsoft.com/fwlink/p/?LinkId=2124703) 下载并安装 "Evergreen Standalone Installer"
3. 安装完成后，在安装程序窗口中按 Enter 继续

#### "Network connection failed"

**原因：** 无网络连接或防火墙阻止下载。

**解决方案：**
- 检查网络连接
- 临时禁用 VPN 或代理
- 检查防火墙设置：允许 `python.exe`、`pip.exe`、`git.exe` 和 `uv.exe` 访问网络
- 如果在企业防火墙后，请联系 IT 部门

#### "Git connection timeout" 或 "Failed to connect to github.com"

**原因：** GitHub 被屏蔽或访问缓慢（在中国大陆防火墙后或某些 VPN 配置下常见）。

**解决方案：**
- 安装程序会自动检测并配置延长的 Git 超时时间
- 等待自动重试（最多 3 次）
- 如果重试失败，请尝试：
  - 切换到不同的 VPN 节点
  - 使用 GitHub 镜像/代理
  - 在网络拥堵较低的时段运行安装程序

#### "Out of disk space"

**原因：** 安装目标磁盘可用空间不足 8 GB。

**解决方案：**
- 释放磁盘空间后重新运行安装程序
- 选择空间更大的其他磁盘
- 注意 AI 模型（首次使用时下载）还需要约 3 GB

#### "Installation failed after retries"

**解决方案：**
1. 检查安装目录中的 `install_log_v1.8.9.txt` 获取具体错误信息
2. 常见原因：
   - 杀毒软件阻止下载：为安装目录添加排除规则
   - 网络超时：使用更好的网络连接重试
   - 下载损坏：删除安装目录后重新开始
3. 尝试以管理员身份运行安装程序（右键 > 以管理员身份运行）

#### "Post-install script failed"

**原因：** 自动化包安装阶段发生错误。

**解决方案：**
1. 检查 `install_log_v1.8.9.txt` 确定失败的具体阶段
2. 如果在 PyTorch 安装阶段失败：检查 GPU 驱动程序版本
3. 如果在依赖项安装阶段失败：检查网络连接
4. 重新运行安装程序；它会检测到现有的不完整安装并提供替换选项

### 运行时问题

#### "GUI 无法启动"或出现空白窗口

**解决方案：**
1. 确认 WebView2 已安装（见上文）
2. 尝试从命令行启动以查看错误信息：
   ```cmd
   cd %LOCALAPPDATA%\WhisperJAV
   python.exe -m whisperjav.webview_gui.main
   ```
3. 检查杀毒软件是否阻止了应用程序

#### 处理速度非常慢

**解决方案：**
1. 查看处理开始时的控制台输出，确认是否已启用 GPU 加速
2. 如果显示 "CPU-only mode"，说明 GPU 未被使用
3. 验证 CUDA 安装：`python -c "import torch; print(torch.cuda.is_available())"`
4. 关闭其他占用 GPU 的应用程序（游戏、视频编辑器）
5. 使用 "Faster" 模式以牺牲部分精度换取更快的结果

#### "Model download stuck"

**解决方案：**
- 大型模型（约 3 GB）根据网络状况需要 5-20 分钟下载
- 下载中断后可以恢复
- 如果卡住超过 30 分钟，请取消并检查网络连接
- 模型缓存在 `%USERPROFILE%\.cache\whisper\`

#### 处理过程中应用程序崩溃

**解决方案：**
1. 确保有足够内存（大型模型推荐 16 GB）
2. 尝试一次只处理一个文件
3. 使用较小的 Whisper 模型（例如 "medium" 代替 "large-v3"）
4. 查看 GUI 控制台中的错误信息
5. 在 [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) 报告持续崩溃的问题

---

## 静默安装/无人值守安装

安装程序支持命令行选项以进行自动化部署：

```cmd
WhisperJAV-1.8.9-Windows-x86_64.exe /S /D=C:\WhisperJAV
```

### 可用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `/S` | 静默模式（无 GUI）| 关闭 |
| `/Q` | 安静模式（抑制控制台输出）| 关闭 |
| `/D=<路径>` | 安装目录（必须是最后一个参数）| `%LOCALAPPDATA%\WhisperJAV` |
| `/InstallationType=AllUsers` | 为所有用户安装（需要管理员权限）| JustMe |
| `/AddToPath=0` 或 `1` | 添加到系统 PATH | 1（启用）|
| `/NoShortcuts=0` 或 `1` | 跳过快捷方式创建 | 0（创建快捷方式）|
| `/InstallLocalLLM=0` 或 `1` | 安装本地大语言模型翻译 | 提示选择 |
| `/NoRegistry=0` 或 `1` | 跳过注册表条目 | 0 |
| `/?` | 显示帮助 | -- |

### 示例

静默安装到自定义目录：
```cmd
cmd /C START /WAIT WhisperJAV-1.8.9-Windows-x86_64.exe /S /D=D:\WhisperJAV
```

静默安装，跳过本地大语言模型，不修改 PATH：
```cmd
cmd /C START /WAIT WhisperJAV-1.8.9-Windows-x86_64.exe /S /AddToPath=0 /InstallLocalLLM=0
```

---

## 常见问题

### 问：需要 NVIDIA GPU 吗？

**答：** 不需要。WhisperJAV 可以在任何 Windows 10/11 系统上运行。没有 NVIDIA GPU 时将以纯 CPU 模式运行，速度约慢 6-10 倍。偶尔使用的话，CPU 模式完全可以胜任。如果经常使用或处理长视频，强烈推荐使用 NVIDIA GPU。

### 问：需要多少磁盘空间？

**答：** 总共约 8 GB：约 4.5 GB 用于安装，约 3 GB 用于 AI 模型（首次使用时下载）。如果还安装了用于翻译的本地大语言模型，则额外需要约 4-8 GB。

### 问：可以安装到网络驱动器吗？

**答：** 不推荐。WhisperJAV 需要快速的磁盘访问来处理临时文件和加载模型。建议安装到本地 SSD 以获得最佳体验。

### 问：我的数据会被发送到网络吗？

**答：** 不会。所有转录都在您的计算机本地完成。仅在安装期间（下载包）和首次运行时（下载 AI 模型）需要联网。之后，WhisperJAV 可以完全离线工作。唯一的例外是如果您使用云端翻译提供商（DeepSeek、Gemini 等），它们需要 API 密钥并会将字幕文本发送到这些服务。

### 问：可以与其他 Python 安装并存吗？

**答：** 可以。WhisperJAV 安装了自己独立的 Python 环境，不会干扰系统上现有的任何 Python 安装。

### 问：如何更新到新版本？

**答：** 下载并运行新的安装程序。如果安装到同一目录，安装程序会检测到旧版本并提供替换选项。您的用户设置会被保留。或者，如果已启用 PATH，可以运行：`pip install -U "whisperjav[all] @ git+https://github.com/meizhong986/whisperjav.git"`

### 问：支持哪些视频格式？

**答：** WhisperJAV 支持 FFmpeg 能读取的所有格式，涵盖几乎所有常见的视频和音频格式：MP4、MKV、AVI、MOV、WMV、FLV、WebM、MP3、WAV、FLAC、AAC、OGG 等。

### 问：字幕文件保存在哪里？

**答：** 字幕文件（SRT 格式）保存在输入视频文件旁边的 `_output` 子文件夹中。例如，如果视频位于 `D:\Videos\movie.mp4`，字幕将位于 `D:\Videos\movie_output\movie.srt`。

### 问：安装程序卡在 "Extracting packages"

**答：** 此阶段解压 conda 包，根据磁盘速度可能需要 2-5 分钟。虽然看起来可能像是卡住了，但实际上正在工作。可以通过任务管理器查看磁盘活动来确认。

### 问：安装完成后可以移动安装目录吗？

**答：** 不可以。移动安装目录会破坏内部路径。如果需要更改位置，请卸载后重新安装到新位置。

### 问：安装时的"本地大语言模型"选项是什么？

**答：** 这会安装 llama-cpp-python，使您能够使用计算机上的本地 AI 模型翻译字幕，无需 API 密钥或网络连接。它需要额外的磁盘空间（4-8 GB），配合 NVIDIA GPU 效果最佳。安装时可以跳过，之后需要时再添加。

### 问：我在企业防火墙/中国大陆防火墙后

**答：** 安装程序内置了对 GitHub 慢速或被屏蔽连接的检测和处理机制。它会自动配置延长的超时时间并重试失败的下载。如果安装仍然失败，请尝试使用能够访问 github.com 和 pypi.org 的 VPN 或代理。

---

## 技术详情

### 安装目录结构

安装完成后，目录结构如下：

```
%LOCALAPPDATA%\WhisperJAV\
    python.exe                    # Python 解释器
    pythonw.exe                   # Python（无控制台窗口）
    WhisperJAV-GUI.exe            # GUI 启动器
    whisperjav_icon.ico           # 应用程序图标
    install_log_v1.8.9.txt        # 安装日志
    uninstall_v1.8.9.bat          # 卸载脚本
    Lib\                          # Python 标准库
        site-packages\
            whisperjav\           # WhisperJAV 应用程序代码
            torch\                # PyTorch
            ...
    Scripts\                      # 可执行脚本
        whisperjav.exe            # CLI 入口点
        whisperjav-gui.exe        # GUI 入口点
        whisperjav-translate.exe  # 翻译 CLI
        pip.exe                   # 包管理器
        ...
    Library\
        bin\                      # FFmpeg、git 及其他工具
            ffmpeg.exe
            git.exe
            ...
```

### 日志文件位置

| 日志 | 路径 | 用途 |
|------|------|------|
| 安装日志 | `[安装目录]\install_log_v1.8.9.txt` | 详细安装进度 |
| 失败标记 | `[安装目录]\INSTALLATION_FAILED_v1.8.9.txt` | 仅在安装失败时创建 |
| 卸载日志 | `%TEMP%\whisperjav_uninstall_v1.8.9.txt` | 卸载详情 |

### 网络要求

安装期间需要能够访问以下域名：

| 域名 | 用途 |
|------|------|
| `pypi.org` | Python 包下载 |
| `files.pythonhosted.org` | Python 包文件 |
| `github.com` | 基于 Git 的包下载 |
| `download.pytorch.org` | PyTorch CUDA wheels |
| `huggingface.co` | AI 模型下载（首次运行时）|
| `aka.ms` | VC++ Redistributable（如需要）|
| `go.microsoft.com` | WebView2 运行时（如需要）|

---

## 支持

如果您遇到本指南未涵盖的问题：

1. 检查安装日志 `install_log_v1.8.9.txt`
2. 在 [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) 搜索已有问题
3. 提交新 issue，请包含：
   - 您的 Windows 版本
   - GPU 型号和驱动程序版本
   - 安装日志文件
   - 问题描述
