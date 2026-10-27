# CLI 参考

WhisperJAV 提供多个命令行工具。

---

## whisperjav（主 CLI）

将视频/音频文件转录为 SRT 字幕。

```bash
whisperjav [OPTIONS] INPUT [INPUT...]
```

### 基本选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `INPUT` | （必填） | 一个或多个视频/音频文件路径 |
| `--mode` | `balanced` | 处理管线：`balanced`、`fast`、`faster`、`fidelity`、`transformers`、`qwen` |
| `--sensitivity` | `aggressive` | 检测灵敏度：`aggressive`、`balanced`、`conservative` |
| `--language` | `japanese` | 源音频语言：`japanese`、`korean`、`chinese`、`english` |
| `--output-dir` | 与源文件相同 | 输出 SRT 文件的目录 |
| `--output-format` | `srt` | 输出格式：`srt`、`vtt`、`both` |

### 模型选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `--model` | （自动） | Whisper 模型：`large-v2`、`large-v3`、`turbo` |
| `--translate` | 关闭 | 在转录过程中翻译为英语 |

### 集成模式选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `--ensemble` | 关闭 | 启用集成模式 |
| `--ensemble-serial` | 关闭 | 先完整处理每个文件再处理下一个 |
| `--pass1-pipeline` | `balanced` | 第一通道的处理管线；也可以是 `parakeet-crispasr`、`cohere-crispasr`（GGUF，见 [Parakeet-CrispASR 与 Cohere-CrispASR](../pipelines/crispasr.md)） |
| `--pass2-pipeline` | （无） | 第二通道的处理管线；指定它即启用第二通道 |
| `--merge-strategy` | `pass1_primary` | 合并策略：`pass1_primary`、`pass2_primary`、`smart_merge`、`full_merge`、`pass1_overlap`、`pass2_overlap`、`longest` |

### 处理选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `--device` | `auto` | 本次运行的 ASR 和其他模型在哪里运行，适用于所有处理管线。`auto` = 按安装的加速器：能用时用其 GPU，否则用 CPU（不加速）；`cpu` = 这次在 CPU 上运行；`cuda` = 需要 NVIDIA GPU。见[是否加速](../getting-started/index.md#accelerated-or-not)。 |
| `--async-processing` | 关闭 | 启用异步处理 |
| `--temp-dir` | 系统临时目录 | 自定义临时文件目录 |
| `--keep-temp` | 关闭 | 保留中间文件 |
| `--debug` | 关闭 | 启用调试日志 |

### 示例

```bash
# 基本转录
whisperjav video.mp4

# Faster 模式配合保守灵敏度
whisperjav video.mp4 --mode faster --sensitivity conservative

# 集成模式串行处理
whisperjav *.mp4 --ensemble --ensemble-serial --merge-strategy smart_merge

# VTT 输出
whisperjav video.mp4 --output-format vtt

# 自定义输出目录
whisperjav video.mp4 --output-dir ./subtitles/

# 这次在 CPU 上运行（不加速）
whisperjav video.mp4 --mode faster --device cpu

# Parakeet-CrispASR 作为一个集成模式通道
whisperjav video.mp4 --ensemble --pass1-pipeline parakeet-crispasr

# 检查安装：加速器、各部分在哪里运行、模型
whisperjav --check
```

### 运行没有开始时

| 退出码 | 含义 |
|---|---|
| 0 | 运行完成 |
| 1 | 运行完成，但至少有一个文件失败（或处于 `--fail-on` 指定的状态） |
| 2 | 运行没有开始：有运行无法使用的设置（提示会说明是哪个） |
| 3 | 运行没有开始：安装为 NVIDIA GPU 制作，而 GPU 现在不可用。框中会说明原因和解决办法；`--device cpu` 让这一次在 CPU 上运行 |
| 4 | 运行没有开始：PyTorch 缺失或损坏（在“自己的 PyTorch”安装中：尚未添加）。提示会给出修复的命令 |

当安装的 GPU 可用时（NVIDIA GPU 或 Apple GPU 安装），运行开始时不显示任何内容。在其他所有安装上，会有一个框说明各部分在这台电脑上的运行位置，然后运行继续。以前版本的 `--hf-device` 和 `--qwen-device` 仍可使用，作为 `--device` 的别名。

---

## whisperjav-gui

启动 GUI 应用。

```bash
whisperjav-gui
```

无需额外参数。所有配置通过 GUI 界面完成。

---

## whisperjav-translate

独立的字幕翻译工具。

```bash
whisperjav-translate [OPTIONS] -i INPUT.srt
```

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `-i`、`--input` | （必填） | 输入 SRT 文件 |
| `--provider` | `deepseek` | AI 提供商 |
| `--model` | （自动） | 模型名称 |
| `--api-key` | （环境变量） | API 密钥（或通过环境变量设置） |
| `--target-language` | `English` | 目标语言 |
| `--tone` | `standard` | 翻译风格：`standard`、`adult` |

---

## whisperjav-upgrade

将 WhisperJAV 升级到最新版本。

```bash
whisperjav-upgrade [OPTIONS]
```

| 选项 | 描述 |
|------|------|
| `--check` | 检查更新但不升级 |
| `--wheel-only` | 仅升级代码（跳过依赖重新安装） |
| `--list-snapshots` | 显示可用的回滚点 |
| `--rollback` | 回滚到上一版本 |
| `--extras` | 仅升级指定的 extras（例如 `cli,gui`） |

详见[升级指南](../UPGRADE.md)。
