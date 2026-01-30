# WhisperJAV v1.8.2rc1 Release Notes

**Release Date:** January 2026
**Type:** Pre-release (Release Candidate 1)
**Branch:** main

---

## Overview

v1.8.2rc1 addresses critical Local LLM translation issues reported after v1.8.1, along with installation process improvements and Colab/Kaggle notebook updates.

---

## What's Fixed

### Local LLM Translation - Root Cause Fix

After comprehensive analysis of issues #149, #148, #146, #139, and #132, we identified the **common root cause**: In v1.8.1, the installer's LLM installation prompt defaulted to **SKIP** if users didn't respond within the timeout, leading to fragile runtime installation attempts.

| Issue | Symptom | Root Cause | Status |
|-------|---------|------------|--------|
| #149 | `Failed to load shared library 'ggml.dll'` | LLM skipped during install, runtime download got wrong wheel | **Fixed** |
| #148 | Server starts, 502 errors on every request | Health check passed but model loading failed | **Fixed** |
| #146 | DLL load errors persist after v1.8.1 update | Same as #149 | **Fixed** |
| #139 | Translation failed with DLL errors | Same as #149 | **Fixed** |
| #132 | Timeout on Colab/Kaggle | Runtime install fragile + insufficient health check | **Fixed** |

### Fix 1: Installer Default Changed (Addresses #149, #146, #139)

**Before (v1.8.1):**
```
Install local LLM translation? (Y/n): [30 second timeout → SKIP]
```

**After (v1.8.2rc1):**
```
Install local LLM translation? (Y/n): [30 second timeout → INSTALL]
```

Users who don't respond now get llama-cpp-python installed during setup when the environment is optimal (VC++ just installed, network verified, correct CUDA detection).

### Fix 2: Improved Server Health Check (Addresses #148, #132)

**Before:** Health check only verified HTTP server responded (`GET /v1/models`)

**Problem:** llama-cpp-python uses **lazy model loading** - the model isn't loaded until the first inference request. So the server could "pass" health check but fail on first real translation request.

**After:** Two-phase health check:
1. **Phase 1:** Wait for HTTP server to respond (existing)
2. **Phase 2:** Send minimal inference request to verify model actually loads

If model loading fails, users now get specific diagnostics:
- "Model failed to load during inference - CUDA/GPU error detected"
- "Inference verification timed out - Model may be loading on CPU"
- Actionable guidance: "Try: --translate-gpu-layers 0 for CPU mode"

### Fix 3: Installation Process Improvements

| Improvement | Description |
|-------------|-------------|
| **Preflight checks** | Disk space, network, WebView2 verification before installation starts |
| **Git timeout handling** | Automatic retry with extended timeouts for GFW/VPN users (Issue #111) |
| **Structured logging** | Detailed `install_log_v1.8.2rc1.txt` for troubleshooting |
| **timed_input()** | Cross-platform timeout for prompts, enables unattended installation |
| **oss2 dependency removed** | Eliminated China-specific cloud SDK dependency |

### Fix 4: Colab/Kaggle Notebook Updates

All notebooks updated for v1.8.2:

| Notebook | Changes |
|----------|---------|
| `WhisperJAV_colab_edition.ipynb` | Version bump, Python 3.13+ check |
| `WhisperJAV_colab_edition_expert.ipynb` | Version bump, Python 3.13+ check |
| `WhisperJAV_colab_parallel.ipynb` | Version bump, Python 3.13+ check, git pinned to @v1.8.2rc1 |
| `WhisperJAV_colab_parallel_expert.ipynb` | Version bump, Python 3.13+ check, git pinned, Kaggle HF_HOME/TMPDIR fix |
| `install_colab.sh` | Branch pinned to v1.8.2rc1 |

**Kaggle-specific fix:** Added cache directory setup to prevent model downloads to RAM disk (OOM issues):
```python
os.environ["HF_HOME"] = "/kaggle/working/.cache"
os.environ["TMPDIR"] = "/kaggle/working/temp"
```

---

## Technical Details

### Health Check Implementation

The new `_wait_for_server()` function in `local_backend.py`:

```python
def _wait_for_server(port: int, max_wait: int = 300) -> Tuple[bool, Optional[str]]:
    """
    Two-phase readiness check:
    1. HTTP Ready: Wait for /v1/models endpoint to respond
    2. Inference Ready: Make a small completion request to verify model loads
    """
    # Phase 1: HTTP connectivity (existing)
    # Phase 2: Inference verification (NEW)
    test_payload = {"prompt": "Hello", "max_tokens": 1}
    # If this fails, model loading or CUDA is broken
```

Returns `(success, error_message)` tuple for better diagnostics.

### Installer Wheel Selection Priority

1. **HuggingFace wheels** (AVX2 only, most compatible with AMD Ryzen)
2. **JamePeng GitHub wheels** (may have AVX512)
3. **Source build** (slowest but guaranteed compatible)

Official HuggingFace wheels: cu128 (primary), cu126 (Colab), cu118 (legacy), metal

---

## Known Limitations

- **CUDA 13.0 not supported**: PyTorch 2.x may bundle CUDA 13.0, but no llama-cpp-python wheels exist for cu130. System falls back to cu128 or source build.
- **No CPU fallback wheel**: If all CUDA wheels fail, source build is required (needs build tools)

---

## Upgrade Instructions

### From v1.8.1 (Recommended)

```bash
# Quick wheel-only upgrade
whisperjav-upgrade --wheel-only

# Or full upgrade with dependency check
whisperjav-upgrade
```

### Fresh Installation

Download the installer when released:
- `WhisperJAV-1.8.2rc1-Windows-x86_64.exe`

Or install from source:
```bash
pip install git+https://github.com/meizhong986/whisperjav.git@v1.8.2rc1
```

---

## Testing Checklist

Before promoting to stable release:

- [ ] Windows installer: Verify LLM installs by default on timeout
- [ ] Health check: Verify 502 errors are caught during startup
- [ ] Colab notebook: Verify Python 3.13+ check works
- [ ] Kaggle notebook: Verify HF_HOME prevents OOM
- [ ] Local LLM: End-to-end translation test with `--provider local`

---

## Related Issues

- #149 - Failed to start local server (ggml.dll)
- #148 - Local LLM Not working (502 errors)
- #146 - Local server error
- #139 - Translated Failed
- #132 - Local LLM not working in Google Colab
- #111 - Git timeout behind GFW

---

## Acknowledgments

Thanks to all users who reported issues and provided detailed logs, enabling this root cause analysis.

---

**Full Changelog:** [v1.8.1...v1.8.2rc1](https://github.com/meizhong986/WhisperJAV/compare/v1.8.1...v1.8.2rc1)
