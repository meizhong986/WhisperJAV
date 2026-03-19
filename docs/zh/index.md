# WhisperJAV

**日本成人视频 AI 字幕生成器**

WhisperJAV 是一款开源工具，利用 OpenAI Whisper 及其他语音识别模型从视频和音频文件中生成精准的日语字幕，并针对日语语言处理进行了专门优化。

---

!!! note "翻译进行中"
    中文文档正在建设中。目前仅首页已翻译，其他页面将自动显示英文原版。

    欢迎在 [GitHub](https://github.com/meizhong986/whisperjav/issues) 上协助改进翻译。

---

## 快速入门

### 1. 安装

=== "Windows（安装程序）"

    从 [最新发布页面](https://github.com/meizhong986/whisperjav/releases/latest) 下载 `WhisperJAV-x.x.x-Windows-x86_64.exe`，运行即可——无需管理员权限。

    [完整 Windows 安装指南](guides/installation_windows_standalone.md){ .md-button }

=== "Windows（Python）"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    installer\install_windows.bat
    ```

    [完整 Python 安装指南](guides/installation_windows_python.md){ .md-button }

=== "macOS"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    python3 -m venv ~/venvs/whisperjav
    source ~/venvs/whisperjav/bin/activate
    ./installer/install_mac.sh
    ```

    [完整 macOS 指南](guides/installation_mac_apple_silicon.md){ .md-button }

=== "Linux"

    ```bash
    git clone https://github.com/meizhong986/whisperjav.git
    cd whisperjav
    ./installer/install_linux.sh
    ```

    [完整 Linux 指南](guides/installation_linux.md){ .md-button }

### 2. 启动 GUI

```bash
whisperjav-gui
```

或使用桌面快捷方式（Windows 安装程序）。

### 3. 转录

1. 将视频拖入应用
2. 点击 **Start**
3. SRT 字幕文件将生成在视频旁边

就这么简单。如需更多控制，请参阅 [GUI 用户指南](guides/gui_user_guide.md)。

---

## 功能一览

| 功能 | 说明 |
|------|------|
| **多处理管线** | Balanced、Fast、Faster、Fidelity、Transformers——在速度与精度间权衡 |
| **集成模式** | 使用两个不同后端进行双重处理，合并获得最佳结果 |
| **BYOP XXL** | 自带管线——在集成模式 Pass 2 中使用 Faster Whisper XXL |
| **ChronosJAV** | 专用管线，搭配 anime-whisper 和 Kotoba 模型，适合动漫/JAV 内容 |
| **Qwen3-ASR** | 替代语音识别引擎，日语表现出色 |
| **AI 翻译** | 使用 Ollama、DeepSeek、Gemini、Claude、GPT 或本地大语言模型翻译字幕 |
| **语音增强** | ClearVoice 降噪、BS-RoFormer 人声分离、FFmpeg DSP 链 |
| **场景检测** | Auditok、Silero、Semantic——分割长音频以提高准确度 |
| **GUI + CLI** | 功能完整的 GUI 用于交互操作，CLI 用于自动化和脚本 |

---

## 最新动态

请查看 [最新发布说明](https://github.com/meizhong986/whisperjav/releases/latest) 了解当前版本的更新内容。

---

## 获取帮助

- [常见问题](faq.md) — 常见问题解答
- [升级故障排除](UPGRADE_TROUBLESHOOTING.md) — 解决升级问题
- [GitHub Issues](https://github.com/meizhong986/whisperjav/issues) — 错误报告和功能请求
