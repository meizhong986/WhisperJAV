# Getting Started

WhisperJAV runs on Windows, macOS, and Linux. Choose the installation method that fits your setup.

## Which install method should I use?

| If you... | Use this |
|-----------|----------|
| Just want it to work (Windows) | [Windows Standalone Installer](../guides/installation_windows_standalone.md) |
| Manage your own Python environment | [Windows Python Install](../guides/installation_windows_python.md) |
| Have an Apple Silicon Mac | [macOS Guide](../guides/installation_mac_apple_silicon.md) |
| Run Linux (Ubuntu, Fedora, Arch) | [Linux Guide](../guides/installation_linux.md) |
| Use Google Colab or Kaggle | See the notebooks in the repository |

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Python** | 3.10 | 3.12 |
| **GPU** | None (CPU works) | NVIDIA with 6GB+ VRAM |
| **RAM** | 8 GB | 16 GB |
| **Disk** | 5 GB | 10 GB (with models) |
| **FFmpeg** | Required | Bundled with installer |

## GPU Support

| GPU | Acceleration | Notes |
|-----|-------------|-------|
| NVIDIA (CUDA) | Full | Best performance. Driver 450+ required. |
| Apple Silicon (MPS) | Partial | Transformers mode only. Other modes fall back to CPU. |
| AMD (ROCm) | Not supported | Use CPU mode |
| Intel (oneAPI) | Not supported | Use CPU mode |
| No GPU | CPU mode | Works but significantly slower |

After installation, launch the GUI with `whisperjav-gui` and follow the [GUI User Guide](../guides/gui_user_guide.md).
