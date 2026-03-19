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
| `--mode` | `balanced` | 处理管线：`balanced`、`fast`、`faster`、`fidelity`、`transformers` |
| `--sensitivity` | `aggressive` | 检测灵敏度：`aggressive`、`balanced`、`conservative` |
| `--language` | `ja` | 源语言代码（`ja`、`ko`、`zh`、`en`） |
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
| `--pass1-pipeline` | `balanced` | 第一通道的处理管线 |
| `--pass2-pipeline` | `qwen` | 第二通道的处理管线 |
| `--merge-strategy` | `smart` | 合并策略：`pass1_primary`、`smart`、`full`、`pass2_primary`、`longest` |

### 处理选项

| 选项 | 默认值 | 描述 |
|------|--------|------|
| `--cpu-only` | 关闭 | 强制 CPU 模式（不使用 GPU） |
| `--async` | 关闭 | 启用异步处理 |
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
whisperjav *.mp4 --ensemble --ensemble-serial --merge-strategy smart

# VTT 输出
whisperjav video.mp4 --output-format vtt

# 自定义输出目录
whisperjav video.mp4 --output-dir ./subtitles/

# 纯 CPU 模式
whisperjav video.mp4 --mode faster --cpu-only
```

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
