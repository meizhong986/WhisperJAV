# WhisperJAV v1.5.3 Release Notes

**Release Date:** November 2025
**Previous Version:** v1.5.1

---

## What's New at a Glance

Version 1.5.3 brings **five major improvements** to WhisperJAV:

1. **Multi-Platform GPU Support** - Now works on Apple Silicon Macs and latest NVIDIA GPUs
2. **Resume Translation** - Never lose translation progress again with auto-save
3. **Multi-Language Support** - Chinese and Korean transcription (experimental)
4. **GUI Enhancements** - Clearer options, better user experience
5. **Critical Bug Fixes** - Windows UTF-8 encoding and CUDA detection fixed

---

## Who Benefits Most from This Release?

- **Mac users with M1/M2/M3/M4/M5 chips**: Can now use WhisperJAV with native GPU acceleration
- **RTX 50 series (Blackwell) owners**: Full support for latest NVIDIA architecture
- **Translation users**: Auto-save prevents lost progress and saves API costs
- **Windows users with Japanese filenames**: UTF-8 encoding fixes prevent processing failures
- **CPU-only users**: Better interface with acceptance checkbox to skip GPU warnings

---

## 1. Multi-Platform GPU Support (NEW)

### What's New

WhisperJAV now **automatically detects and uses the best available GPU** on your system, with expanded platform support.

### Supported Hardware

**NVIDIA GPUs:**
- RTX 20 series (Turing)
- RTX 30 series (Ampere)
- RTX 40 series (Ada Lovelace)
- **RTX 50 series (Blackwell)** - NEW
- Professional cards (A100, H100, L40, etc.)

**Apple Silicon:**
- **M1, M2, M3, M4, M5 chips** - NEW
- Native Metal Performance Shaders (MPS) acceleration
- Optimized for macOS GPU hardware

**AMD GPUs:**
- Detection support via ROCm
- Limited processing support (falls back to CPU for stability)

**CPU Fallback:**
- Automatic fallback if no compatible GPU detected
- Works on any system (slower but functional)

### User Impact

- **Mac users**: WhisperJAV now runs natively with GPU acceleration on Apple Silicon
- **Latest GPU support**: RTX 50 series and newest hardware fully supported
- **Zero configuration**: Automatic device detection - just run the application
- **Better error messages**: Clear feedback when GPU isn't available

### Technical Details

New `device_detector.py` module provides:
- Intelligent GPU architecture detection
- Platform-specific optimization selection
- CUDA/MPS/ROCm capability checking
- Graceful degradation to CPU when needed

**Files Changed:**
- `whisperjav/utils/device_detector.py` (NEW - 265 lines)
- `whisperjav/utils/preflight_check.py` (enhanced)
- `installer/post_install_v1.5.1.py` (CUDA detection improved)

---

## 2. Resume Translation Feature (NEW)

### What's New

Translation can now be **interrupted and resumed** without losing progress. The system automatically saves project state after each batch.

### How It Works

1. **Auto-Save**: Project state saved after each translation batch completes
2. **Error Recovery**: Progress preserved even if translation fails mid-batch
3. **Resume**: Re-run the same command to continue from last completed batch
4. **Cleanup**: Project files automatically removed after successful completion

### User Benefits

**No More Lost Work:**
- Interrupt translation anytime (Ctrl+C or close window)
- Resume later from exactly where you left off
- Network issues don't erase hours of progress

**Cost Savings:**
- Completed batches never re-translated
- Saves API calls to DeepSeek, Claude, Gemini, etc.
- Partial results preserved for inspection

**Better Reliability:**
- Graceful handling of network errors
- API rate limits don't lose progress
- System crashes recover cleanly

### Example Usage

```bash
# Start translation
whisperjav-translate -i subtitles.srt

# [Translation interrupted after 50/100 batches]

# Resume from batch 51
whisperjav-translate -i subtitles.srt
# Output: "Resuming from batch 51 of 100..."
```

### Technical Details

**Implementation:**
- Persistent project files (`.subtrans` format)
- Batch-level checkpointing
- Save-on-error exception handling
- Atomic batch completion tracking

**Files Changed:**
- `whisperjav/translate/core.py` (35 lines modified)

---

## 3. Multi-Language Transcription (EXPERIMENTAL)

### What's New

WhisperJAV GUI now supports **Chinese and Korean** transcription in addition to Japanese.

### Supported Source Languages

| Language | Status | Notes |
|----------|--------|-------|
| Japanese (日本語) | Optimized | Primary focus, best results |
| Korean (한국어) | Experimental | Basic support, may need tuning |
| Chinese (中文) | Experimental | Basic support, may need tuning |
| English | Experimental | Basic support |

### How to Use

**In GUI:**
1. Select "Source audio language" dropdown
2. Choose your video's language
3. Process as normal

**From CLI:**
- CLI remains optimized for Japanese
- Same flags work for other languages (experimental)

### Important Notes

- **Chinese/Korean support is experimental** - accuracy may be lower than Japanese
- Post-processing optimized for Japanese may not work perfectly for other languages
- Consider this a preview - full optimization coming in future releases

### User Impact

- **Expanded use cases**: Process Chinese and Korean AV content
- **Learning curve reduced**: Same interface for multiple languages
- **Foundation for future**: Platform ready for additional language optimization

---

## 4. GUI Improvements

### New Features

#### 1. Source Language Selection
- Clear dropdown menu for input audio language
- Options: Japanese, Korean, Chinese, English
- Replaces confusing "language" setting

#### 2. Clearer Output Format Labels
- **"Native (transcribe in source language)"** - Default option
- **"Direct to English (translate via Whisper)"** - Translation option
- Eliminates confusion about what each setting does

#### 3. CPU Mode Acceptance Checkbox
- **"Accept CPU-only mode (skip GPU warning)"**
- Prevents repeated GPU warning prompts
- Better experience for users without GPU

#### 4. Drag & Drop Support Enhanced
- Improved file handling
- Better visual feedback
- More reliable operation

#### 5. UI Scrolling Fixes
- Fixed console log performance issues
- Smoother scrolling with large outputs
- Better rendering of progress updates

### User Impact

**Less Confusion:**
- Settings labeled in plain language
- Clear explanation of what each option does
- No more "What does this do?" moments

**Better UX for CPU Users:**
- Stop being nagged about GPU every time
- One checkbox to acknowledge CPU mode
- Cleaner workflow

**Smoother Interface:**
- Console no longer lags with large outputs
- Progress updates render smoothly
- Overall more responsive feel

### Technical Details

**Files Changed:**
- `whisperjav/webview_gui/assets/index.html` (new form fields)
- `whisperjav/webview_gui/assets/app.js` (76+ lines, drag & drop, scrolling)
- `whisperjav/webview_gui/assets/style.css` (UI styling)
- `whisperjav/webview_gui/main.py` (enhanced GUI logic)

---

## 5. Critical Bug Fixes

### 5.1 Windows UTF-8 Encoding Fix

**Problem:**
- Subprocess calls failed on Windows with non-ASCII characters
- Japanese filenames and paths caused crashes
- Translation and processing would fail silently

**Solution:**
- Added explicit `encoding='utf-8'` to all `subprocess.run()` calls
- Ensures proper handling of Japanese, Chinese, Korean characters
- Works consistently across Windows locales

**Impact:**
- **Japanese filenames now work reliably** on Windows
- No more mysterious failures with Asian characters
- Translation works correctly with Unicode paths

**Files Fixed:**
- `whisperjav/main.py`
- `whisperjav/modules/audio_extraction.py`
- `whisperjav/modules/media_discovery.py`
- `whisperjav/utils/preflight_check.py`

---

### 5.2 CUDA Version Comparison Fix

**Problem:**
- Type mismatch in CUDA version checking (string vs float)
- Incorrect "incompatible CUDA version" warnings
- False positives prevented valid GPU usage

**Solution:**
- Proper type conversion before comparison
- Consistent version parsing logic
- Added unit tests to prevent regression

**Impact:**
- **Accurate CUDA toolkit detection**
- No more false warnings about incompatible versions
- Correct GPU capability assessment

**Test Coverage:**
- `tests/test_preflight_cuda_version.py` (NEW)

**Files Fixed:**
- `whisperjav/utils/preflight_check.py`

---

### 5.3 UI Scrolling Performance

**Problem:**
- Console output lagged with large logs
- UI became unresponsive during processing
- Poor user experience with real-time updates

**Solution:**
- Optimized scroll handling in JavaScript
- Better DOM update batching
- Improved rendering performance

**Impact:**
- **Smoother GUI experience** during processing
- Real-time updates don't cause lag
- Better overall responsiveness

**Files Fixed:**
- `whisperjav/webview_gui/assets/app.js`

---

## System Requirements (Updated)

### Supported Platforms

| Platform | GPU Support | Status |
|----------|-------------|--------|
| **Windows 10/11** (64-bit) | NVIDIA (CUDA) | Fully Supported |
| **macOS** (Apple Silicon) | M1/M2/M3/M4/M5 (MPS) | Fully Supported (NEW) |
| **macOS** (Intel) | CPU only | Supported (slower) |
| **Linux** | NVIDIA (CUDA) | Fully Supported |
| **Linux** | AMD (ROCm) | Limited (experimental) |

### GPU Requirements

**NVIDIA GPUs:**
- CUDA 11.8+ required
- RTX 20/30/40/50 series recommended
- 4-6 GB VRAM recommended

**Apple Silicon:**
- M1 or newer (M1/M2/M3/M4/M5)
- macOS 12.3+ (MPS support)
- 8 GB unified memory minimum

**AMD GPUs:**
- ROCm detection only
- Falls back to CPU for processing
- Future optimization planned

### Memory Requirements

- **8 GB RAM minimum** (system memory)
- **16 GB RAM recommended** for large files (>2 hours)
- **GPU: 4-6 GB VRAM** for optimal performance

### Software Requirements

- **Python**: 3.9, 3.10, 3.11, or 3.12 (3.13+ incompatible)
- **FFmpeg**: Required (must be in PATH)
- **WebView2**: Required for GUI (Windows only, auto-prompted)

---

## Installation & Upgrading

### New Installation

#### Windows (Installer - Recommended)
```bash
# Download WhisperJAV-1.5.3-Windows-x86_64.exe (coming soon)
# Run installer and follow prompts
# Desktop shortcut created automatically
```

#### macOS (Apple Silicon)
```bash
# Install Homebrew if needed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 ffmpeg

# Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main

# Launch GUI
whisperjav-gui
```

#### From Source (All Platforms)
```bash
# Clone repository
git clone https://github.com/meizhong986/whisperjav.git
cd whisperjav

# Install in development mode
pip install -e . -U

# Launch
whisperjav-gui  # GUI
whisperjav video.mp4  # CLI
```

---

### Upgrading from v1.5.1

#### If Using Installer (Windows)
1. **Option A**: Uninstall v1.5.1 first (clean install)
2. **Option B**: Install v1.5.3 alongside (keeps both versions)
3. Run new v1.5.3 installer
4. **Models and cache preserved** automatically

#### If Using pip
```bash
# Upgrade to latest version
pip install -U --no-deps git+https://github.com/meizhong986/whisperjav.git@main

# Verify version
python -c "from whisperjav import __version__; print(__version__)"
# Output: 1.5.3
```

#### Cache and Settings
- **Configuration files**: Automatically migrated
- **Model cache**: Preserved (no re-download needed)
- **Custom settings**: Retained from previous version

---

## Platform-Specific Installation Notes

### Windows (NVIDIA GPU)

**Prerequisites:**
1. NVIDIA GPU with CUDA 11.8+ support
2. Latest NVIDIA drivers
3. WebView2 (installer prompts if missing)

**Installation:**
- Use Windows installer (easiest)
- Or install via pip with CUDA-enabled PyTorch

---

### macOS (Apple Silicon)

**Prerequisites:**
1. M1/M2/M3/M4/M5 Mac
2. macOS 12.3 or newer
3. Xcode Command Line Tools

**Installation:**
```bash
# Install Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 ffmpeg

# Install PyTorch with MPS support
pip install torch torchvision torchaudio

# Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

**Performance Notes:**
- M1/M2/M3/M4/M5 use Metal Performance Shaders (MPS)
- Significantly faster than CPU-only mode
- Performance comparable to entry-level NVIDIA GPUs

---

### Linux (NVIDIA GPU)

**Prerequisites:**
1. NVIDIA GPU with CUDA 11.8+
2. NVIDIA drivers (proprietary recommended)
3. CUDA Toolkit (optional, PyTorch bundles its own)

**Installation:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pip ffmpeg

# Install WhisperJAV
pip install git+https://github.com/meizhong986/whisperjav.git@main
```

---

## Known Issues

### 1. Chinese/Korean Support Incomplete
- **Status**: Experimental feature, work in progress
- **Impact**: Lower accuracy than Japanese transcription
- **Workaround**: Use balanced mode for best results
- **Fix**: Full optimization planned for v1.6.0

### 2. AMD GPU Support Limited
- **Status**: Detection only, processing uses CPU
- **Impact**: No GPU acceleration for AMD users
- **Workaround**: Use CPU mode (works but slower)
- **Fix**: ROCm optimization planned for future release

### 3. Resume Translation Requires PySubtrans 0.7.0+
- **Status**: Older PySubtrans versions don't support persistence
- **Impact**: Resume feature won't work with outdated dependencies
- **Workaround**: Upgrade PySubtrans: `pip install -U pysubtrans`
- **Fix**: Installer bundles correct version automatically

### 4. GUI Requires WebView2 on Windows
- **Status**: WebView2 not installed by default on Windows 10 (pre-2020)
- **Impact**: GUI won't launch without WebView2
- **Workaround**: Installer prompts for automatic download
- **Fix**: None needed (WebView2 is standard on Windows 11)

---

## Technical Changes Summary

### New Modules
- `whisperjav/utils/device_detector.py` - Multi-platform GPU detection (265 lines)
- `tests/test_preflight_cuda_version.py` - CUDA version unit tests

### Modified Core Modules
- `whisperjav/translate/core.py` - Resume capability (35 lines modified)
- `whisperjav/utils/preflight_check.py` - Enhanced device detection
- `whisperjav/webview_gui/main.py` - Multi-language support (76+ lines)
- `whisperjav/webview_gui/assets/*.js/html/css` - GUI improvements

### Subprocess Encoding Fixes
All `subprocess.run()` calls updated with `encoding='utf-8'`:
- `whisperjav/main.py`
- `whisperjav/modules/audio_extraction.py`
- `whisperjav/modules/media_discovery.py`
- `whisperjav/utils/preflight_check.py`

### Installer Updates
- PyInstaller spec fixes (Qt dependencies)
- Better dependency handling
- WebView2 detection improvements
- Platform-specific install scripts

---

## Git Commit History

**Changes from v1.5.1 to v1.5.3:**

- 15 commits
- 57 files changed
- 4,339 insertions, 256 deletions
- Development period: October 31 - November 9, 2025

**Key Commits:**
1. `739d9a4` - Added: resume translation, More Fixes: UI scrolling
2. `7e37783` - Several Fixes: Apple Silicon, Blackwell cuda, device-check, drag&drop
3. `3a79a39` - Fixes for multiple enhancements
4. `5103a3c` - Fix CUDA version comparison in preflight check
5. `defdc8f` - Fix Windows subprocess UTF-8 encoding
6. `a894001` - Support for Chinese AV videos (experimental)

---

## Acknowledgments

Special thanks to:
- Community testers who reported UTF-8 encoding issues on Windows
- Apple Silicon Mac users who requested native GPU support
- Contributors who tested experimental Chinese/Korean features
- All users who provided feedback on v1.5.1

---

## Questions or Problems?

- **Report issues**: https://github.com/meizhong986/WhisperJAV/issues
- **Documentation**: See README.md and CLAUDE.md
- **GUI Guide**: See GUI_USER_GUIDE.md

---

## License

WhisperJAV is released under the MIT License. See LICENSE file for details.

---

**Version 1.5.3 - November 2025**
*Multi-Platform Release - Expanding Access, Improving Reliability*
