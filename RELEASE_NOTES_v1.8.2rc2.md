# WhisperJAV v1.8.2-rc2 Release Notes

Pre-release for testing. Feedback welcome.

---

## Supported Platforms

| Platform | GPU Acceleration | Notes |
|----------|------------------|-------|
| Windows x64 | NVIDIA CUDA | Driver 450+ required |
| Linux x64 | NVIDIA CUDA | Driver 450+ required |
| macOS Apple Silicon | Metal | M1/M2/M3/M4 native |
| macOS Intel | CPU only | No GPU acceleration |
| Google Colab | NVIDIA CUDA (cu126) | T4/V100/A100 GPUs |

---

## What's Changed

### CUDA Configuration Simplified

Reduced official CUDA support to 4 versions to improve maintainability:

| CUDA | Driver | Purpose | Wheel Source |
|------|--------|---------|--------------|
| cu128 | 570+ | Primary | HuggingFace |
| cu118 | 450+ | Legacy fallback | HuggingFace |
| cu126 | 560+ | Google Colab | HuggingFace |
| cu130 | 575+ | Development only | JamePeng GitHub |

- Created `llama_cuda_config.py` as single source of truth
- Removed cu121, cu124 (not officially supported)
- AVX2 wheels only (AMD Ryzen compatible, no AVX512)

### Local LLM Improvements

- Added install-time validation to catch DLL issues early
- Improved server diagnostics (GPU layers, inference speed, batch time estimates)
- Fixed driver-to-CUDA mapping (575+→cu130, 570+→cu128, 560+→cu126, 450+→cu118)

---

## Issues Fixed

| Issue | Symptom | Fix |
|-------|---------|-----|
| #132 | Local LLM timeout on Colab | Fixed dependency resolution, race condition, temp file handling |
| #132 | Semantic scene detection ignored | Fixed merge fallback logic |
| - | Inconsistent CUDA version mappings | Centralized in `llama_cuda_config.py` |
| - | cu124 returned for driver 550-559 | Now correctly maps to cu118 |

---

## Local LLM Wheel Installation Flow

When you use `--provider local` for translation, the system installs llama-cpp-python automatically:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Try HuggingFace wheel (AVX2, most compatible)           │
│     └─ Fails? ↓                                             │
│  2. Try JamePeng GitHub wheel                               │
│     └─ Fails? ↓                                             │
│  3. Build from source (requires build tools, ~10 min)       │
└─────────────────────────────────────────────────────────────┘
```

| Source | Pros | Cons |
|--------|------|------|
| HuggingFace | Fast, AVX2 (AMD compatible) | Limited CUDA versions |
| JamePeng GitHub | More CUDA versions | May have AVX512 (AMD issues) |
| Source build | Always works | Slow, needs compiler |

**CUDA version selection:**
- System detects your CUDA version from PyTorch
- Tries exact match first, then falls back to older compatible versions
- Example: cu130 system tries cu130 → cu128 → cu126 → cu118

---

## Known Limitations

- **cu118 wheels**: Not yet uploaded to HuggingFace (falls back to JamePeng or source build)
- **No CPU fallback wheel**: Source build required if CUDA wheels fail

---

## Upgrade Instructions

### Windows Users (Fresh Install Recommended)

Download and install from standalone `.exe` (attached below).

### Expert Users (Windows / macOS / Linux)

```bash
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav
git checkout v1.8.2-rc2
```

| Platform | Command |
|----------|---------|
| Windows | `installer\install_windows.bat` |
| macOS | `./installer/install_linux.sh` |
| Linux | `./installer/install_linux.sh` |

---

## Files Changed

- `whisperjav/translate/llama_cuda_config.py` (new)
- `whisperjav/translate/llama_build_utils.py`
- `whisperjav/translate/local_backend.py`
- `installer/templates/post_install.py.template`
- `installer/install_colab.sh`
- `notebook/WhisperJAV_colab_parallel.ipynb`
- `notebook/WhisperJAV_colab_parallel_expert.ipynb`

---

**Full Changelog:** [v1.8.2-rc1...v1.8.2-rc2](https://github.com/meizhong986/WhisperJAV/compare/v1.8.2-rc1...v1.8.2-rc2)
