# WhisperJAV 升级指南

**从 1.9.x 升级到 1.10：** 请全新安装 1.10。1.10 需要 Python 3.13，而 1.9.x 安装使用的是 Python 3.10，因此它的 `whisperjav-upgrade` 无法安装 1.10。设置会保留。

**在 1.10 内：** 升级会安装 1.10 系列中最新的版本，使用该版本测试过的软件包版本，并保留安装的加速器（安装文件夹中的 `whisperjav_stack.txt`）。在"自己的 PyTorch"安装中，升级从不安装、替换或删除 PyTorch。

## Windows 安装程序制作的安装 {#windows-installer-installs}

GUI 会提示有新版本：顶部栏会出现显示新版本号的提示，顶部栏调色板图标（🎨）中的 **检查更新** 会立即检查。它的按钮会打开 GitHub 上的发布页面；GUI 本身不会升级 WhisperJAV。请在命令提示符中升级：

1. 关闭 WhisperJAV（它的所有窗口）。
2. 打开命令提示符（按"开始"，输入 `cmd`，按 Enter），运行：
   ```
   "%LOCALAPPDATA%\WhisperJAV\python.exe" -m whisperjav.upgrade
   ```
   如果安装到了其他文件夹，请在这里和下文中用那个文件夹代替 `%LOCALAPPDATA%\WhisperJAV`。
3. 当它询问 `Continue? [y/N]:` 时，输入 `y` 并按 Enter。
4. 当它显示 "Upgrade complete!" 时，检查版本：
   ```
   "%LOCALAPPDATA%\WhisperJAV\Scripts\whisperjav.exe" --version
   ```

请如上用安装文件夹中的 `python.exe` 启动升级，不要用 `Scripts\whisperjav-upgrade.exe`：当 WhisperJAV 安装在与 Windows 临时文件夹（通常在 C 盘）不同的盘上时，1.10.0 和 1.10.1 的 `whisperjav-upgrade.exe` 会在中途停止，之后 WhisperJAV 无法启动。

同一命令加上一个选项：

| 加上 | 作用 |
|---|---|
| `--check` | 显示 GitHub 上是否有比你的版本更新的版本；不做任何更改。它看的是所有版本，而升级只安装 1.10 系列中最新的版本。最后一行会说 "Run 'whisperjav-upgrade' to update."；请改用上面的命令。结果保留 6 小时；再加上 `--force` 可立即重新查询。 |
| `--yes` | 升级前不再询问。 |
| `--wheel-only` | 只更新 WhisperJAV 本身；它使用的软件包（CrispASR、PyTorch 等）保持原版本。 |

不要用 `pip install -U whisperjav…` 更新：在 Windows 上，PyPI 提供的是仅 CPU 版的 PyTorch，可能会在毫无提示的情况下替换你的 GPU 版本。

## 如果升级失败

完整升级在做任何更改之前，升级工具会保存已安装软件包的列表（`--wheel-only` 时不保存），失败后会尝试恢复这个列表。在用 Windows 安装程序制作的安装中，这在 1.10 里不起作用：它会以 "Rollback failed" 停止。升级完成时工具建议的 `--rollback` 同样会报错停止。

如果升级后 WhisperJAV 无法启动，请卸载它，再用安装程序安装最新版本：参见[修复或回到旧版本](MANUAL_ROLLBACK.md)。升级工具的提示信息请参见[故障排除](UPGRADE_TROUBLESHOOTING.md)。

## 源码安装（Windows、macOS、Linux） {#source-installs-windows-macos-linux}

在 `WhisperJAV` 文件夹中，激活与安装时相同的环境（Windows 上用 `--local` 安装的，先运行 `.venv\Scripts\activate`），运行 `git fetch --tags`，然后 `git checkout v1.10.x`（你要的版本），再像安装时一样运行安装脚本（Windows 上如果当时用了 `--local`，就加上它）。它会把当前安装的加速器作为默认选项。不要在源码安装上使用 `whisperjav-upgrade`：在那个文件夹中它会运行 `git pull`，而这在切换到某个版本标签的克隆上会失败。

## 保留的内容

设置（Windows 上为 `%APPDATA%\WhisperJAV`）和已下载的模型（`%USERPROFILE%\.cache`）都在安装文件夹之外。升级不会更改它们，卸载和重新安装也不会。例外是 XXL Faster Whisper 的 **可执行文件** 和 **额外参数** 设置：WhisperJAV 把它们保存在安装文件夹中，升级或重新安装后请重新填写。
