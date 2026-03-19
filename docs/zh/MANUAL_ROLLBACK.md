# 手动回滚指南

如果升级失败或导致问题，请按照以下步骤将 WhisperJAV 恢复到可用状态。

## 快速参考

| 场景 | 解决方案 |
|------|----------|
| 升级中途失败 | [方案 A：重新安装指定版本](#方案-a重新安装指定版本) |
| 新版本有 bug | [方案 A：重新安装指定版本](#方案-a重新安装指定版本) |
| 环境完全损坏 | [方案 B：全新安装](#方案-b全新安装) |
| Python 完全无法运行 | [方案 B：全新安装](#方案-b全新安装) |

## 方案 A：重新安装指定版本

当 Python 仍可运行但 WhisperJAV 出现问题时使用此方案。

### 第 1 步：打开命令提示符

1. 按 `Win + R`
2. 输入 `cmd` 并按回车

### 第 2 步：导航到 WhisperJAV 目录

```cmd
cd %LOCALAPPDATA%\WhisperJAV
```

### 第 3 步：重新安装所需版本

**重新安装最新稳定版：**
```cmd
python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@main
```

**安装特定版本（例如 v1.7.3）：**
```cmd
python.exe -m pip install --force-reinstall git+https://github.com/meizhong986/whisperjav.git@v1.7.3
```

**可用版本标签：**
- `v1.7.4` - 最新版
- `v1.7.3` - 上一个稳定版
- `v1.7.2`、`v1.7.1`、`v1.7.0` - 更早版本

### 第 4 步：验证安装

```cmd
python.exe -c "from whisperjav import __version__; print(__version__)"
```

## 方案 B：全新安装

当 Python 环境完全损坏时使用此方案。

### 第 1 步：卸载 WhisperJAV

1. 打开 **设置** → **应用** → **应用和功能**
2. 搜索 "WhisperJAV"
3. 点击 **卸载**

或手动删除：
```cmd
rmdir /s /q "%LOCALAPPDATA%\WhisperJAV"
```

### 第 2 步：下载新安装程序

前往 [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) 下载最新的 `.exe` 安装程序。

### 第 3 步：运行安装程序

安装程序将创建一个包含所有依赖的全新环境。

## 保留你的数据

AI 模型和缓存存储在独立位置，重新安装后仍会保留：

| 数据 | 位置 | 是否保留？ |
|------|------|-----------|
| Whisper 模型 | `~/.cache/whisper/` | 是 |
| HuggingFace 模型 | `~/.cache/huggingface/` | 是 |
| WhisperJAV 配置 | `%LOCALAPPDATA%\WhisperJAV\whisperjav_config.json` | 卸载时删除 |

**卸载前备份配置：**
```cmd
copy "%LOCALAPPDATA%\WhisperJAV\whisperjav_config.json" "%USERPROFILE%\Desktop\"
```

## 常见问题

### "pip is not recognized"

PATH 环境变量设置不正确。请使用完整路径：
```cmd
%LOCALAPPDATA%\WhisperJAV\python.exe -m pip install ...
```

### "Permission denied"

关闭所有 WhisperJAV 窗口后重试。如果仍然失败：
1. 打开任务管理器（`Ctrl+Shift+Esc`）
2. 结束所有来自 WhisperJAV 的 `python.exe` 或 `pythonw.exe` 进程
3. 重新执行命令

### "Network error"

检查网络连接。如果使用代理：
```cmd
set HTTPS_PROXY=http://your-proxy:port
python.exe -m pip install ...
```

## 获取帮助

如果以上方法都无效：

1. 前往 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题
2. 请附上以下信息：
   - 出现问题时你正在进行的操作
   - 错误信息（截图或复制粘贴）
   - Windows 版本
   - 以下命令的输出：`python.exe --version`
