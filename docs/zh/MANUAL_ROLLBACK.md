# 修复 WhisperJAV 或回到旧版本

本页适用于用 Windows 安装程序制作的安装。升级工具自带的回滚（`--rollback`，以及升级失败后它尝试的回滚）在 1.10 中对这类安装不起作用：它会报错停止。请按下面的步骤操作。

## 修复无法启动的安装

1. 卸载 WhisperJAV：双击安装文件夹（默认为 `%LOCALAPPDATA%\WhisperJAV`）中的 `Uninstall-WhisperJAV.exe`，或打开 Windows **设置 > 应用**，找到 **WhisperJAV** 并选择 **卸载**。升级之后，"应用"中的条目仍显示你最初安装的版本。
2. 卸载完成后，从 [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) 下载最新版本的安装程序并安装。卸载会留下旧的桌面图标（例如 "WhisperJAV v1.10.0"）；请删除它，并从新安装创建的图标启动 WhisperJAV。
3. 安装程序会像第一次安装时那样选择加速器：有受支持的 NVIDIA GPU 时直接使用，不询问；否则会弹出窗口询问。要改为其他加速器，之后把安装程序再次运行到同一个文件夹。在"自己的 PyTorch"安装中，请用安装结束时显示的命令重新添加你的 PyTorch。

## 回到较早的 1.10 版本

1. 按上文卸载 WhisperJAV。
2. 在 [WhisperJAV Releases](https://github.com/meizhong986/WhisperJAV/releases) 中打开你要的较早 1.10 版本的页面（例如 v1.10.0），下载它的安装程序并安装。

## 保留的内容

设置（`%APPDATA%\WhisperJAV`）和已下载的模型（`%USERPROFILE%\.cache`）都在安装文件夹之外：卸载和重新安装都不会改动它们。例外是 XXL Faster Whisper 的 **可执行文件** 和 **额外参数** 设置：WhisperJAV 把它们保存在安装文件夹中，重新安装后请重新填写。

## 不要用 pip 修复

不要运行 `pip install --force-reinstall git+https://github.com/…` 或 `pip install -U whisperjav`：在 Windows 上，PyPI 提供的是仅 CPU 版的 PyTorch，可能会替换你的 GPU 版本，其他软件包也可能不是 WhisperJAV 测试过的版本。

## 获取帮助

参见[报告问题](UPGRADE_TROUBLESHOOTING.md#reporting-issues)。
