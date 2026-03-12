# WhisperJAV Installation System — State Review for External Reviewers

**Date:** 2026-03-12
**Current stable release:** v1.8.7
**Current dev branch:** main (12 commits ahead of v1.8.7, targeting v1.8.8)

---

## Table of Contents

1. [Background & Historical Context](#1-background--historical-context)
2. [v1.8.7 Stable: Released Installation Process](#2-v187-stable-released-installation-process)
3. [Current Dev State: Post-uv Migration](#3-current-dev-state-post-uv-migration)
4. [Known Issues & Open Questions](#4-known-issues--open-questions)
5. [File Reference](#5-file-reference)

---

## 1. Background & Historical Context

### What is WhisperJAV?

WhisperJAV is a subtitle generation tool optimized for Japanese media, using OpenAI Whisper with custom enhancements for Japanese language processing, scene detection, and voice activity detection. It targets consumers (non-developers wanting subtitles) and developers/researchers alike.

### The GPU Lock-in Problem (Issue #90)

The central installation challenge for any Whisper-based tool:

1. Standard `pip install` resolves all dependencies at once
2. PyTorch on PyPI is **CPU-only** (~200MB)
3. Whisper packages depend on torch
4. Result: pip installs CPU torch even when user has an RTX 4090
5. Inference is **6-10x slower** on CPU vs GPU

**Solution (v1.0-v1.8.7):** Install PyTorch **first** with `--index-url` pointing to CUDA wheels, then install everything else. This "locks in" the GPU version so subsequent packages see torch as already satisfied.

### Installation Evolution Timeline

| Version | Approach | Key Change |
|---------|----------|------------|
| v1.0-v1.6 | Manual pip commands in batch scripts | 950+ line .bat files with duplicated logic per platform |
| v1.7.0 | Unified installer module (`whisperjav/installer/`) | Centralized GPU detection, registry, retry logic |
| v1.8.0 | conda-constructor .exe installer | Bundled Python + uv binary for end users |
| v1.8.2 | Package registry as single source of truth | `registry.py` defines all packages; validation catches drift |
| v1.8.7 | 6-stage pip pipeline via `install.py` | Registry-driven, but still multiple sequential pip calls |
| **Dev (post-v1.8.7)** | **uv sync** replaces pip pipeline | Single command, lockfile, named GPU indexes |

### Two Installation Paths

WhisperJAV has always maintained two distinct installation paths:

| Path | Target User | Method | Produced By |
|------|------------|--------|-------------|
| **Conda-constructor installer** | End users (Windows) | Download .exe → run → done | `build_release.py` → NSIS .exe (~150MB) |
| **Source install** | Developers, Linux/macOS users | `git clone` → `python install.py` | Direct from repo |

Both paths share the same core logic (GPU detection, package registry, preflight checks) via the `whisperjav/installer/` module.

---

## 2. v1.8.7 Stable: Released Installation Process

### 2.1 Source Installation (`install.py` — 1,741 lines)

The v1.8.7 `install.py` uses a **6-stage sequential pip pipeline** driven by the package registry.

#### Preflight Checks

Before installation begins, the script validates:

| Check | Method | Failure Behavior |
|-------|--------|-----------------|
| Python version | `sys.version_info` | Hard fail if not 3.10-3.12 |
| FFmpeg | `shutil.which("ffmpeg")` | Warning (needed for audio extraction) |
| Git | `shutil.which("git")` | Hard fail (needed for git-based deps) |
| Disk space | `shutil.disk_usage()` | Hard fail if < 8GB free |
| Network | `urllib.request.urlopen("https://pypi.org")` | Hard fail if PyPI unreachable |
| WebView2 | Windows registry check | Warning (needed for GUI) |
| VC++ Redist | Windows registry check | Warning (needed for some packages) |

#### The 6-Stage Pipeline

```
Step 1/6: Upgrade pip
    └─ pip install --upgrade pip

Step 2/6: Install PyTorch (GPU Lock-in)
    └─ pip install torch torchaudio torchvision --index-url <cuda_url>
    └─ CUDA URL selected by GPU detection (cu128, cu118, or cpu)
    └─ Apple Silicon: installs from default PyPI (MPS support built-in)

Step 3/6: Install Core Dependencies
    └─ numpy, scipy, librosa, soundfile, pydub, pysrt, srt, etc.
    └─ Packages queried from registry.py (single source of truth)
    └─ numpy installed BEFORE numba (binary ABI compatibility)

Step 4/6: Install Whisper Packages
    └─ openai-whisper (git+https://github.com/openai/whisper@main)
    └─ stable-ts (git+https://github.com/meizhong986/stable-ts-fix-setup.git)
    └─ ffmpeg-python (git+https://github.com/kkroening/ffmpeg-python.git)
    └─ faster-whisper (from PyPI)

Step 5/6: Install Optional Packages
    └─ HuggingFace (transformers, accelerate, huggingface-hub)
    └─ Qwen3-ASR
    └─ Translation (pysubtrans, openai, google-genai)
    └─ LLM server (uvicorn, fastapi)
    └─ llama-cpp-python (interactive prompt, platform-specific wheel logic)
    └─ VAD (silero-vad, auditok)
    └─ Speech Enhancement (modelscope, clearvoice, bs-roformer) — allow_fail=True
    └─ GUI (pywebview, pythonnet on Windows)

Step 6/6: Install WhisperJAV
    └─ pip install --no-deps .          (standard)
    └─ pip install --no-deps -e .       (with --dev flag)
    └─ --no-deps prevents pip from re-resolving and overwriting GPU torch
```

#### Post-Install Verification

```python
# 1. Import test
python -c "import whisperjav; print(whisperjav.__version__)"

# 2. Torch CUDA verification
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

#### Post-Install Summary

Prints CLI usage instructions:
```
whisperjav video.mp4 --mode balanced
whisperjav-gui
whisperjav --help
```

#### GPU Detection Flow

```
1. Check --cpu-only or --cuda flags (explicit override)
2. Try whisperjav.installer.detect_gpu() (nvidia-smi + pynvml)
   └─ Returns GPUInfo with name, driver_version, cuda_version
3. Check Apple Silicon (sys.platform == "darwin" + arm64)
4. Fallback: basic nvidia-smi query
5. Default: "cpu"

Driver → CUDA mapping:
  Driver 570+ → cu128 (CUDA 12.8)
  Driver 450+ → cu118 (CUDA 11.8)
  No GPU      → cpu
```

#### Key Architectural Decisions in v1.8.7

1. **Registry-driven:** Package lists queried from `whisperjav/installer/core/registry.py` at runtime, not hardcoded
2. **StepExecutor:** Centralized pip execution with retry logic (3 attempts, 5s delay)
3. **allow_fail=True:** Speech enhancement and GUI packages can fail without aborting installation
4. **Git timeout handling:** Detects 21-second TCP timeout pattern (common behind GFW), auto-reconfigures git
5. **Interactive prompts with timeout:** LLM installation prompt defaults to "yes" after 30 seconds

### 2.2 Conda-Constructor Installer (Windows .exe)

The end-user installer is built with conda-constructor and produces a ~150MB Windows executable.

#### Build Pipeline

```
installer/VERSION              ← Version config (major.minor.patch.prerelease)
         │
installer/build_release.py     ← 5-phase orchestrator
         │
         ├─ Phase 1: Update whisperjav/__version__.py
         ├─ Phase 2: Generate 11 files from templates/
         ├─ Phase 3: Build whisperjav-{version}-py3-none-any.whl
         ├─ Phase 4: Copy static files (LICENSE, icon)
         └─ Phase 5: Run validation
         │
installer/generated/           ← Output directory
         ├─ construct_v{VER}.yaml          (conda-constructor config)
         ├─ post_install_v{VER}.bat        (batch wrapper)
         ├─ post_install_v{VER}.py         (2,820 lines of post-install logic)
         ├─ requirements_v{VER}.txt        (~65 packages)
         ├─ build_installer_v{VER}.bat     (runs conda-constructor)
         ├─ validate_installer_v{VER}.py   (pre-build checks)
         ├─ WhisperJAV_Launcher_v{VER}.py  (GUI launcher)
         ├─ custom_template_v{VER}.nsi.tmpl(NSIS UI customization)
         └─ ...
```

#### What the .exe Installer Does

```
User downloads WhisperJAV-1.8.7-Windows-x86_64.exe
         │
    ┌────┴────┐
    │  NSIS   │  ← GUI installer (choose install location)
    │ Wizard  │     Default: %LOCALAPPDATA%\WhisperJAV (no admin required)
    └────┬────┘
         │
    ┌────┴─────────────────┐
    │ conda-constructor    │  ← Extracts minimal conda environment:
    │ (base environment)   │     Python 3.10.18, pip, git, ffmpeg, menuinst
    └────┬─────────────────┘
         │
    ┌────┴─────────────────┐
    │ post_install.py      │  ← 2,820-line script that runs:
    │ (heavy lifting)      │
    │  1. GPU detection    │     nvidia-smi or pynvml
    │  2. CUDA selection   │     cu128/cu118/cpu based on driver
    │  3. Disk/network     │     8GB free, PyPI reachable
    │  4. VC++ Redist      │     Auto-download if missing
    │  5. WebView2 check   │     Prompt to install if missing
    │  6. PyTorch install  │     uv pip install torch --index-url <cuda>
    │  7. GPU constraints  │     Write pin file to protect CUDA wheels
    │  8. All other deps   │     uv pip install -r requirements.txt --constraint gpu_constraints.txt
    │  9. WhisperJAV wheel │     uv pip install whisperjav-{ver}-py3-none-any.whl
    │ 10. Git config       │     Timeout/proxy configuration
    │ 11. Verification     │     Import test + CUDA test
    └────┬─────────────────┘
         │
    ┌────┴──────────┐
    │ Desktop       │  ← Created by NSIS custom template
    │ Shortcut      │     Points to WhisperJAV_Launcher_v{VER}.py
    └───────────────┘
```

#### Key Details

- **Install location:** `%LOCALAPPDATA%\WhisperJAV` (no admin, user-writable)
- **Install time:** 10-20 minutes (internet-dependent, downloads ~2-4GB of packages)
- **First run:** Additional 5-10 minutes for AI model download (~3GB)
- **uv.exe bundled:** ~30MB binary included for 10-30x faster pip operations
- **GPU constraint file:** After PyTorch install, pins torch/torchaudio/torchvision versions to prevent subsequent pip installs from replacing CUDA wheels with CPU versions
- **GFW support:** Detects China network issues (21s TCP timeout) and auto-configures git with extended timeouts and proxy detection

### 2.3 Platform Wrapper Scripts

Thin wrappers that delegate to `install.py`:

| Script | Platform | What It Does |
|--------|----------|--------------|
| `installer/install_windows.bat` | Windows | Verifies directory, forwards args to `python install.py` |
| `installer/install_linux.sh` | Linux | Same + documents system deps (python3-dev, build-essential, libsndfile1) |
| `installer/install_mac.sh` | macOS | Same + documents Homebrew deps (portaudio) |

### 2.4 Upgrade System (`whisperjav/upgrade.py` — 1,520 lines)

The v1.8.7 upgrade system handles conda-constructor installs:

```
whisperjav-upgrade
    │
    ├─ Check GitHub releases/latest for new version
    ├─ Create snapshot (rollback point)
    ├─ pip install git+https://github.com/meizhong986/whisperjav.git@main
    │   └─ --upgrade --constraint gpu_constraints.txt
    │   └─ GPU constraints protect CUDA PyTorch wheels
    ├─ Fix known compatibility issues (numpy, librosa)
    ├─ Update desktop shortcuts
    ├─ Clean old version-specific files
    └─ Preserve user data (.whisperjav_cache, configs)
```

### 2.5 Validation & Quality Gates

#### Runtime Preflight (`whisperjav/utils/preflight_check.py`)

Runs before every WhisperJAV execution:

| Check | Status | Action |
|-------|--------|--------|
| Python 3.10-3.12 | PASS/FAIL | Abort if wrong version |
| CUDA availability | PASS/WARN | Continue with warning if no GPU |
| PyTorch CUDA compat | PASS/WARN | Detect driver/toolkit mismatch |
| GPU memory | INFO | Recommend model size |
| FFmpeg on PATH | FAIL | Abort (required for audio) |
| Disk space | WARN | Warn if < 8GB |
| Critical imports | WARN | Warn if packages missing |

#### CI/CD Validation (`whisperjav/installer/validation/`)

| Validator | What It Checks |
|-----------|---------------|
| `sync.py` | pyproject.toml extras match registry.py definitions |
| `imports.py` | All imports in codebase are covered by registry (no ghost dependencies) |
| `__main__.py` | Runner: `python -m whisperjav.installer.validation` |

### 2.6 Package Registry (`whisperjav/installer/core/registry.py`)

**The single source of truth** for all ~81 packages. Each package is a dataclass:

```python
@dataclass
class Package:
    name: str                    # PyPI name
    version: str                 # Version constraint
    extra: Extra                 # Which pyproject.toml extra
    source: InstallSource        # PYPI, GIT, INDEX_URL, WHEEL_URL
    order: int                   # Install priority (10=first, 90=last)
    git_url: Optional[str]       # For git-based packages
    index_url: Optional[str]     # For PyTorch CUDA index
    import_name: Optional[str]   # When import name differs (cv2 for opencv-python)
    reason: str                  # Why this package is needed
```

**Installation order ranges:**

```
10-19  PyTorch         MUST be first (GPU lock-in)
20-29  Scientific      numpy before numba (ABI compatibility)
30-39  Whisper         Depends on torch being present
40-49  Audio/CLI       soundfile, pydub, VAD
50-59  GUI             pywebview, pythonnet
60-69  Translation     pysubtrans, openai, google-genai
70-79  Enhancement     modelscope, clearvoice, bs-roformer
80-89  HuggingFace     transformers, accelerate
90-99  Compat/dev      pyvideotrans, ruff
```

### 2.7 Standalone Module (`whisperjav/installer/core/standalone.py`)

Self-contained utilities bundled with the conda-constructor installer. **Critical constraint: ZERO imports from whisperjav.*** — this file runs before WhisperJAV is installed.

Contains:
- GPU detection (nvidia-smi, pynvml fallback)
- CUDA driver matrix (570+ → cu128, 450+ → cu118)
- Git timeout pattern detection
- Retry logic (3 attempts, 5s delay, 30min timeout)
- Version parsing utilities

---

## 3. Current Dev State: Post-uv Migration

### 3.1 What Changed (12 commits since v1.8.7)

```
2e28f2a feat: migrate install.py from 6-stage pip to uv sync    ← THE BIG CHANGE
2119d28 docs+tests: add dependency cross-match table + 65 tests
0ab96b1 fix: harden installation — GPU protection + graceful pip errors
dea0fa2 fix: remove np.int_ from metadata_manager (numpy 2.x)
d5b8296 fix: correct numpy 2.x migration — proper lower bounds
a2ab596 fix: align silero-vad>=6.2 across all sources
2da8b22 feat: lift numpy<2.0 pin — allow numpy 2.x
04e52fb docs: update issue tracker
badfc2b docs: add v1.8.8 research reports
9dc8f1f fix(#198): force greedy decoding on MPS
c052865 fix: numpy constraint in Phase 3 torch install
d1ee06e fix: pre-install numpy<2 before Phase 3 PyTorch
```

### 3.2 The uv Migration: Before vs After

#### Before (v1.8.7): 6-stage pip pipeline

```
install.py (1,741 lines)
    │
    ├─ Step 1: pip install --upgrade pip
    ├─ Step 2: pip install torch --index-url <cuda>
    ├─ Step 3: pip install numpy scipy librosa ...
    ├─ Step 4: pip install git+whisper git+stable-ts ...
    ├─ Step 5: pip install transformers pywebview ... + llama-cpp
    └─ Step 6: pip install --no-deps .

    6+ separate pip invocations, each resolving dependencies independently
    Registry queried at runtime for package lists
    No lockfile — versions resolved fresh each time
```

#### After (current dev): Single uv sync

```
install.py (833 lines)
    │
    ├─ Preflight: uv, disk, network, Python, FFmpeg, Git
    ├─ GPU detection → select PyTorch index
    ├─ uv sync --extra cli --extra gui ... --index pytorch=<cuda_url>
    │   └─ Single command installs ALL packages from uv.lock
    │   └─ Named 'pytorch' index routes torch/torchaudio to CUDA wheels
    │   └─ All other packages resolve from PyPI (default)
    └─ Optional: llama-cpp-python (separate uv pip install)

    1 primary command replaces 6 pip steps
    uv.lock pins all ~246 packages with exact versions + hashes
    pyproject.toml [tool.uv.sources] handles GPU routing declaratively
```

### 3.3 New: `install.py` (833 lines, uv-based)

#### Flow

```
main()
 ├─ Parse arguments
 ├─ Validate source directory (pyproject.toml exists)
 ├─ Initialize logging
 │
 ├─ Preflight Checks
 │   ├─ check_uv()           ← NEW: auto-installs uv via pip if missing
 │   ├─ check_disk_space()
 │   ├─ check_network()
 │   ├─ Python version (via installer module or fallback)
 │   ├─ FFmpeg
 │   └─ Git
 │
 ├─ GPU Detection
 │   └─ detect_cuda_version() → "cpu" | "cu118" | "cu124" | "cu128" | "metal"
 │
 ├─ Determine Extras
 │   ├─ --minimal: ["cli"]
 │   └─ default:   ["cli", "gui", "translate", "llm", "huggingface",
 │                   "qwen", "analysis", "compatibility", "enhance"]
 │
 ├─ run_uv_sync(extras, cuda_version, dev)     ← SINGLE COMMAND
 │   ├─ Build cmd: uv sync --extra cli --extra gui ...
 │   ├─ Add: --python sys.executable
 │   ├─ Add: --index pytorch=<cuda_url>  (if not metal)
 │   ├─ Add: --verbose
 │   ├─ Stream output live (subprocess.Popen, line-buffered)
 │   └─ Tee to log file
 │
 ├─ Retry without --extra enhance (if first attempt fails)
 │
 ├─ Optional: _install_local_llm()
 │   └─ Uses nested _uv_pip_install() for platform-specific wheel
 │
 ├─ _verify_installation()
 │   └─ .venv Python: import whisperjav, check torch.cuda
 │
 └─ Summary
     └─ If .venv exists: print activation instructions
     └─ Print usage commands
```

#### Key Functions

| Function | Lines | Purpose |
|----------|-------|---------|
| `_uv_cmd()` | 161-170 | Returns `["uv"]` or `[sys.executable, "-m", "uv"]` |
| `check_uv()` | 177-227 | Verify uv installed; auto-bootstrap via pip if missing |
| `detect_cuda_version()` | 320-370 | GPU detection with flag overrides |
| `run_uv_sync()` | 377-456 | **Core:** runs `uv sync` with extras and pytorch index |
| `_install_local_llm()` | 463-554 | Platform-specific llama-cpp-python install |
| `_venv_python()` | 587-596 | Find .venv Python for verification |
| `_verify_installation()` | 599-626 | Import test + torch CUDA test |

### 3.4 New: pyproject.toml uv Configuration

```toml
# Named index for PyTorch GPU wheels
[tool.uv]
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu128"   # Default: CUDA 12.8
explicit = true  # ONLY torch/torchaudio use this index

# Per-package source routing with platform markers
[tool.uv.sources]
torch = [
    { index = "pytorch", marker = "sys_platform != 'darwin'" },
]
torchaudio = [
    { index = "pytorch", marker = "sys_platform != 'darwin'" },
]
# macOS: torch comes from default PyPI (has built-in MPS support for Apple Silicon)
# Linux/Windows: torch comes from named pytorch CUDA index
```

**New `[torch]` extra:**
```toml
[project.optional-dependencies]
torch = ["torch", "torchaudio"]
cli = ["whisperjav[torch]", "soundfile", "pydub", ...]
```

**Override CUDA at install time:**
```bash
uv sync --extra all --index pytorch=https://download.pytorch.org/whl/cu118   # CUDA 11.8
uv sync --extra all --index pytorch=https://download.pytorch.org/whl/cpu     # CPU only
```

### 3.5 New: `uv.lock` (Lockfile)

- Pins all ~246 packages with exact versions + hashes
- Generated by `uv lock` (run by developer when updating dependencies)
- Consumed by `uv sync` (deterministic, no resolution at install time)
- Replaces the deleted `requirements.txt`

**What the lockfile provides:**
1. Exact reproducibility across all users
2. No resolution at install time (fast, no surprises)
3. Hash verification (tamper protection)
4. Transitive dependency pinning
5. Conflict prevention (caught at dev time, not user install time)

### 3.6 New: Upgrade Path for Source Installs

Added to `whisperjav/upgrade.py`:

```python
def _find_source_root() -> Optional[Path]:
    """Detect uv-based source install by looking for pyproject.toml + uv.lock"""

def _detect_cuda_for_source() -> str:
    """GPU detection for upgrade path: nvidia-smi → torch.version.cuda → cpu"""

def _run_uv_sync_upgrade(source_root, extras="all") -> bool:
    """
    1. git pull (fetch latest code)
    2. Detect CUDA version
    3. uv sync --extra <extras> --index pytorch=<cuda_url>
    """
```

The existing pip-based upgrade path (for conda-constructor installs) is unchanged.

### 3.7 What Did NOT Change

| Component | Status | Why |
|-----------|--------|-----|
| Conda-constructor installer | Unchanged | Uses `uv pip install`, not `uv sync`. Separate path. |
| `post_install.py.template` | Unchanged | conda-constructor post-install. No pyproject.toml in installed env. |
| `whisperjav/installer/core/registry.py` | Unchanged | Still the single source of truth for package definitions |
| `whisperjav/installer/core/standalone.py` | Unchanged | GPU detection, self-contained utilities |
| `whisperjav/utils/preflight_check.py` | Unchanged | Runtime environment validation |
| `whisperjav/installer/validation/` | Minor update | Added "torch" to known composite extras whitelist |
| Platform scripts (.bat, .sh) | Docs only | Updated CLI flag documentation |

---

## 4. Known Issues & Open Questions

### 4.1 Environment Targeting Problem (Current Focus)

**Problem:** `uv sync` always creates `.venv` in the project directory, even when the user has an active conda or venv environment.

- Packages go into `.venv`, not the user's env
- Console scripts (`whisperjav`, `whisperjav-gui`) land in `.venv/Scripts/`, not on PATH
- User must double-activate (their env + `.venv`) or use `uv run`
- Returning users who type `whisperjav` get "command not found"

**Root cause:** `uv sync` ignores `VIRTUAL_ENV` and `CONDA_PREFIX` by design.

**Proposed fix:** Set `UV_PROJECT_ENVIRONMENT` env var to the active environment path before calling `uv sync`. This is uv's officially supported mechanism for this use case. Additionally, `--inexact` flag may be needed when installing into conda environments (to avoid removing conda-installed packages not in the lockfile).

### 4.2 `.python-version` File

The project has a `.python-version` file (contains `3.11`) which causes uv to download CPython 3.11 instead of using the user's Python. Current workaround: `--python sys.executable` flag. Needs a proper solution (possibly removing the file or using `UV_PROJECT_ENVIRONMENT` which overrides it).

### 4.3 Two Installation Paths Diverging

The conda-constructor path uses `uv pip install` (no lockfile, manual GPU constraint files) while the source path now uses `uv sync` (lockfile, declarative GPU routing). These paths will continue to diverge unless unified.

### 4.4 Consumer UX Gap

Current source install flow ends with:
```
Packages were installed into the project's .venv.
Activate it before running WhisperJAV:
    .venv\Scripts\activate
```

This is developer-facing, not consumer-facing. A consumer-ready installation would need:
- Automatic environment activation or smart launcher
- Desktop shortcut that "just works"
- No terminal knowledge required

---

## 5. File Reference

### Source Installation Files

| File | Lines | Role |
|------|-------|------|
| `install.py` | 833 | Main source installer (uv sync, current dev) |
| `pyproject.toml` | ~460 | Project metadata, extras, uv config |
| `uv.lock` | ~5,000 | Lockfile (all packages pinned with hashes) |
| `installer/install_windows.bat` | ~150 | Windows wrapper → install.py |
| `installer/install_linux.sh` | ~150 | Linux wrapper → install.py |
| `installer/install_mac.sh` | ~120 | macOS wrapper → install.py |

### Conda-Constructor Installer Files

| File | Lines | Role |
|------|-------|------|
| `installer/build_release.py` | 533 | 5-phase release build orchestrator |
| `installer/VERSION` | ~20 | Version config (major.minor.patch.prerelease) |
| `installer/templates/construct.yaml.template` | 65 | conda-constructor config |
| `installer/templates/post_install.py.template` | 2,820 | Post-install logic (GPU, deps, checks) |
| `installer/templates/post_install.bat.template` | ~30 | Batch wrapper for post_install.py |
| `installer/templates/requirements.txt.template` | 69 | Pip requirements (~65 packages) |
| `installer/templates/WhisperJAV_Launcher.py.template` | ~100 | GUI launcher script |
| `installer/templates/build_installer.bat.template` | ~50 | Runs conda-constructor |
| `installer/templates/validate_installer.py.template` | ~200 | Pre-build validation |
| `installer/templates/custom_template.nsi.tmpl.template` | ~100 | NSIS UI customization |

### Core Installer Module

| File | Lines | Role |
|------|-------|------|
| `whisperjav/installer/__init__.py` | 363 | Package API, architecture docs |
| `whisperjav/installer/core/registry.py` | ~700 | Package registry (single source of truth) |
| `whisperjav/installer/core/standalone.py` | ~300 | GPU detection, CUDA matrix (zero whisperjav imports) |
| `whisperjav/installer/core/executor.py` | ~400 | StepExecutor with retry logic |
| `whisperjav/installer/validation/sync.py` | 218 | Registry ↔ pyproject.toml sync validation |
| `whisperjav/installer/validation/imports.py` | 322 | Ghost dependency detection (AST-based) |
| `whisperjav/installer/validation/__main__.py` | ~50 | Validation runner |

### Upgrade System

| File | Lines | Role |
|------|-------|------|
| `whisperjav/upgrade.py` | 1,520 | Upgrade script (pip + uv sync paths) |

### Runtime Validation

| File | Lines | Role |
|------|-------|------|
| `whisperjav/utils/preflight_check.py` | ~500 | Pre-execution environment validation |

---

*Document prepared for external review of WhisperJAV installation architecture.*
*For questions, see: https://github.com/meizhong986/WhisperJAV/issues*
