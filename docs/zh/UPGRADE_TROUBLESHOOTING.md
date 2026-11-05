# 升级故障排除

本页适用于用 Windows 安装程序制作的安装。

## 升级后 WhisperJAV 无法启动

例如升级的最后几行显示 "Rollback failed - installation may be in inconsistent state"，或者运行 `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version` 时以 `ModuleNotFoundError: No module named 'whisperjav'` 结尾。请卸载 WhisperJAV，再用安装程序安装最新版本：参见[修复或回到旧版本](MANUAL_ROLLBACK.md)。

## 升级时出现 "[WinError 5] Access is denied"

WhisperJAV 安装在与 Windows 临时文件夹（通常在 C 盘）不同的盘上，而升级替换安装文件夹 `Scripts` 中的程序时，其中某个程序正在运行：用 `whisperjav-upgrade.exe` 启动升级时就是它本身，也可能是另一个 WhisperJAV 窗口。升级会在中途停止，之后 WhisperJAV 通常无法启动：请按上一节修复。同一条信息中 pip 的建议（"Consider using the `--user` option or check the permissions."）不适用。下次请先关闭所有 WhisperJAV 窗口，并用安装文件夹中的 `python.exe` 启动升级。

## "No internet connection" 或 "Cannot upgrade: GitHub's releases list could not be read"

升级会先检查能否访问 github.com，然后读取 GitHub 上的版本列表。没有做任何更改。请检查网络连接后重试。升级会从 GitHub、PyPI 和 download.pytorch.org 下载；中国用户请在整个升级过程中保持 VPN 开启。

## "Upgrade stopped before any change: …"

没有做任何更改。信息的其余部分说明了原因：无法读取新版本的软件包列表（请检查网络连接后重试；在中国请保持 VPN 开启），新版本的某个软件包没有适用于你电脑的版本，或者检查超时。如果再次出现，请附上完整输出提交问题（见下文）。

## "Stopped: conda installed …, but this release is pinned to …"

没有做任何更改。请用提示信息中给出的链接报告此问题。

## 升级后提示 "CUDA not available"

**原因：** 通常是 CPU 版 PyTorch 覆盖了 CUDA 版本（例如普通的 `pip install` 造成的），或 NVIDIA 驱动发生了变化。

1. 运行 `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --check`：它会说明安装的加速器以及 PyTorch 能使用什么。
2. 用 `nvidia-smi` 检查驱动（Windows 580 或更新，Linux 570 或更新）。
3. 再次运行 WhisperJAV 安装程序：它会保留安装的加速器，并恢复与之对应的 PyTorch 版本。

## 不要用 pip 修复

不要运行 `pip install -U whisperjav` 或 `pip install git+https://github.com/…`：在 Windows 上，PyPI 提供的是仅 CPU 版的 PyTorch，可能会替换你的 GPU 版本，其他软件包也可能不是 WhisperJAV 测试过的版本。

## 报告问题 {#reporting-issues}

请在 [GitHub Issues](https://github.com/meizhong986/WhisperJAV/issues) 提交问题，并附上：

- 升级命令的完整输出（全部内容，而不只是最后几行）；
- 如果 WhisperJAV 仍能启动，运行 `"%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --check` 的输出；
- WhisperJAV 所在的盘，以及你的 Windows 版本（`winver`）。
