# WhisperJAV 升级指南

本指南介绍如何将 WhisperJAV 升级到最新版本。

## GUI 用户（推荐方式）

GUI 启动时会自动检查更新，并在有新版本时显示通知横幅。

### 更新提示

当有可用更新时，标题栏会出现一个小图标，显示新版本号（例如 "v1.8.10 available"）。点击可查看更新日志和发布说明。

仅在**关键**更新时，窗口顶部才会显示醒目的彩色横幅，确保你不会错过。

### 一键更新

1. 点击标题栏中的更新提示（或关键更新的横幅）
2. 查看更新日志，然后点击 **Update Now**（立即更新）
3. 在对话框中确认更新
4. GUI 将自动关闭
5. 更新在后台运行
6. 完成后 GUI 自动重新启动

## CLI 用户

使用 `whisperjav-upgrade` 命令进行命令行升级。

### 检查更新

```bash
whisperjav-upgrade --check
```

仅显示是否有可用更新，不会执行安装。

### 交互式升级

```bash
whisperjav-upgrade
```

升级开始前会提示你进行确认。

### 非交互式升级

```bash
whisperjav-upgrade --yes
```

自动确认升级（适用于脚本自动化）。

### 热补丁模式

```bash
whisperjav-upgrade --wheel-only
```

仅更新 WhisperJAV 软件包本身，跳过依赖安装。适用于发布后的小补丁（例如 1.7.4 → 1.7.4.post1），速度更快、更安全。

### 显示脚本版本

```bash
whisperjav-upgrade --version
```

### 强制检查更新

```bash
whisperjav-upgrade --check --force
```

绕过 6 小时的缓存，立即检查更新。

## 更新内容

| 组件 | 完整升级 | 仅更新包 |
|------|----------|----------|
| WhisperJAV 代码 | 是 | 是 |
| 新增依赖 | 是 | 否 |
| numpy/librosa 修复 | 是 | 否 |
| 桌面快捷方式 | 是 | 是 |
| 旧文件清理 | 是 | 否 |

## 保留的内容

升级过程会保留以下内容：
- AI 模型（`~/.cache/whisper/`、`~/.cache/huggingface/`）
- 配置文件（`whisperjav_config.json`）
- 缓存数据（`.whisperjav_cache/`）

## 故障排除

请参阅 [UPGRADE_TROUBLESHOOTING.md](UPGRADE_TROUBLESHOOTING.md) 了解常见问题及解决方案。

## 手动回滚

如果升级失败，请参阅 [MANUAL_ROLLBACK.md](MANUAL_ROLLBACK.md) 了解恢复方法。

## 运行独立升级脚本

如果已安装的升级命令无法使用，你可以下载并运行独立脚本：

```cmd
cd %LOCALAPPDATA%\WhisperJAV
curl -O https://raw.githubusercontent.com/meizhong986/whisperjav/main/installer/upgrade_whisperjav.py
python.exe upgrade_whisperjav.py
```

## 技术细节

### 版本检查缓存

更新检查结果会缓存 6 小时，以避免频繁调用 API。使用 `--force` 可绕过缓存。

### 更新流程

1. **GUI 检测到更新** → 显示通知横幅
2. **用户点击 "Update Now"** → 确认对话框
3. **GUI 启动更新进程** → 后台进程
4. **GUI 退出** → 释放文件锁
5. **更新进程等待 GUI** → 确保干净退出
6. **升级脚本运行** → 更新软件包
7. **更新进程重启 GUI** → 全新启动

### 创建的文件

- `update.log` - 详细的升级日志（位于安装目录）
- `.whisperjav_cache/version_check.json` - 缓存的更新检查结果
- `.whisperjav_cache/update_dismissed.json` - 已忽略的通知记录
