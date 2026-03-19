# 快速入门

WhisperJAV 可在 Windows、macOS 和 Linux 上运行。请根据你的环境选择合适的安装方式。

## 我应该选择哪种安装方式？

| 如果你... | 请使用 |
|-----------|--------|
| 只想在 Windows 上开箱即用 | [Windows 独立安装程序](../guides/installation_windows_standalone.md) |
| 自行管理 Python 环境 | [Windows Python 安装](../guides/installation_windows_python.md) |
| 使用 Apple Silicon Mac | [macOS 指南](../guides/installation_mac_apple_silicon.md) |
| 使用 Linux（Ubuntu、Fedora、Arch） | [Linux 指南](../guides/installation_linux.md) |
| 使用 Google Colab 或 Kaggle | 参见仓库中的 notebook |

## 系统要求

| 组件 | 最低要求 | 推荐配置 |
|------|----------|----------|
| **Python** | 3.10 | 3.12 |
| **GPU** | 无（CPU 可用） | NVIDIA 显卡，6GB+ 显存 |
| **内存** | 8 GB | 16 GB |
| **磁盘** | 5 GB | 10 GB（含模型） |
| **FFmpeg** | 必需 | 安装程序已内置 |

## GPU 支持

| GPU | 加速支持 | 备注 |
|-----|----------|------|
| NVIDIA (CUDA) | 完整支持 | 性能最佳。需要驱动版本 450+。 |
| Apple Silicon (MPS) | 部分支持 | 仅 Transformers 模式可用，其他模式回退至 CPU。 |
| AMD (ROCm) | 不支持 | 请使用 CPU 模式 |
| Intel (oneAPI) | 不支持 | 请使用 CPU 模式 |
| 无 GPU | CPU 模式 | 可以运行，但速度明显较慢 |

安装完成后，使用 `whisperjav-gui` 启动图形界面，然后参照 [GUI 用户指南](../guides/gui_user_guide.md) 开始使用。
