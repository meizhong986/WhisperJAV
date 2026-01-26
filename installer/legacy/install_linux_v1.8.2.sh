#!/bin/bash
# ==============================================================================
# WhisperJAV Linux Installation Script (v1.8.0)
# ==============================================================================
#
# This script handles the staged installation of WhisperJAV on Linux systems,
# working around pip dependency resolution conflicts.
#
# Usage:
#   chmod +x install_linux.sh
#   ./install_linux.sh
#
# Options:
#   --cpu-only              Install CPU-only PyTorch (no CUDA)
#   --cuda118               Install PyTorch for CUDA 11.8 (universal fallback for older drivers)
#   --cuda128               Install PyTorch for CUDA 12.8 (default for driver 570+)
#   --no-speech-enhancement Skip speech enhancement packages
#   --minimal               Install minimal version (no speech enhancement, no TEN VAD)
#   --dev                   Install in development/editable mode
#   --local-llm             Install local LLM (tries prebuilt wheel first)
#   --local-llm-build       Install local LLM (builds from source)
#
# Note: CUDA 12.1/12.4/12.6 support removed in v1.8.0 (PyTorch 2.7.x dropped these)
#
# ==============================================================================

set -e  # Exit on error

# Initialize paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INSTALL_LOG="$SCRIPT_DIR/install_log_linux.txt"
echo "" > "$INSTALL_LOG"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$INSTALL_LOG"
}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log "============================================================"
log "  WhisperJAV Linux Installation Script"
log "  Started: $(date)"
log "============================================================"

echo "============================================================"
echo "  WhisperJAV Linux Installation Script"
echo "============================================================"
echo ""

# ==============================================================================
# SYSTEM DEPENDENCIES (Issue #33)
# ==============================================================================
# Before running this script, ensure you have the following system packages:
#
# Debian/Ubuntu:
#   sudo apt-get update
#   sudo apt-get install -y python3-dev build-essential ffmpeg libsndfile1
#
# Fedora/RHEL:
#   sudo dnf install python3-devel gcc ffmpeg libsndfile
#
# Arch Linux:
#   sudo pacman -S python ffmpeg libsndfile
#
# If you encounter build errors for audio packages, also install:
#   sudo apt-get install -y portaudio19-dev  # For PyAudio/sounddevice issues
# ==============================================================================

# Check for FFmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${RED}Error: FFmpeg is not installed.${NC}"
    echo "Please install FFmpeg first:"
    echo "  Debian/Ubuntu: sudo apt-get install ffmpeg"
    echo "  Fedora/RHEL:   sudo dnf install ffmpeg"
    echo "  Arch Linux:    sudo pacman -S ffmpeg"
    log "ERROR: FFmpeg not found"
    exit 1
fi
echo -e "${GREEN}FFmpeg found: $(ffmpeg -version | head -n1)${NC}"
log "FFmpeg found"

# Check for Git
if ! command -v git >/dev/null 2>&1; then
    echo -e "${RED}Error: Git is not installed.${NC}"
    echo "Please install Git first:"
    echo "  Debian/Ubuntu: sudo apt-get install git"
    echo "  Fedora/RHEL:   sudo dnf install git"
    echo "  Arch Linux:    sudo pacman -S git"
    log "ERROR: Git not found"
    exit 1
fi
echo -e "${GREEN}Git found${NC}"
log "Git found"

# ==============================================================================
# PEP 668 / Virtual Environment Check (Debian 12+, Ubuntu 24.04+)
# ==============================================================================
# Modern Linux distributions mark system Python as "externally-managed" which
# prevents pip from installing packages system-wide. This is a safety feature.
# Users should run this script inside a virtual environment.

check_venv_requirement() {
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}Virtual environment detected: $VIRTUAL_ENV${NC}"
        log "Virtual environment detected: $VIRTUAL_ENV"
        return 0
    fi

    # Check if system has PEP 668 marker (externally-managed-environment)
    PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null)
    if [[ -f "$PYTHON_STDLIB/EXTERNALLY-MANAGED" ]]; then
        echo ""
        echo -e "${YELLOW}============================================================${NC}"
        echo -e "${YELLOW}  WARNING: System Python is Externally Managed (PEP 668)${NC}"
        echo -e "${YELLOW}============================================================${NC}"
        echo ""
        echo "Your system (likely Debian 12+ or Ubuntu 24.04+) prevents pip from"
        echo "installing packages system-wide. You have two options:"
        echo ""
        echo -e "${GREEN}Option 1 (Recommended): Create and activate a virtual environment${NC}"
        echo "  python3 -m venv ~/.venv/whisperjav"
        echo "  source ~/.venv/whisperjav/bin/activate"
        echo "  ./install_linux.sh"
        echo ""
        echo -e "${YELLOW}Option 2: Use --break-system-packages (not recommended)${NC}"
        echo "  pip3 install --break-system-packages <package>"
        echo ""
        echo "Or install via pipx for isolated installation:"
        echo "  pipx install whisperjav"
        echo ""
        log "WARNING: PEP 668 externally-managed environment detected, no venv active"

        # Ask user if they want to continue anyway
        echo -e "${YELLOW}Do you want to continue anyway? Installation will likely fail.${NC}"
        read -t 30 -p "Continue? (y/N): " CONTINUE_ANYWAY || CONTINUE_ANYWAY="n"
        if [[ ! "$CONTINUE_ANYWAY" =~ ^[Yy]$ ]]; then
            echo ""
            echo "Installation cancelled. Please create a virtual environment first."
            exit 1
        fi
        echo ""
        echo -e "${YELLOW}Continuing at your own risk...${NC}"
        log "User chose to continue without venv"
    else
        # No PEP 668, but still recommend venv
        echo -e "${YELLOW}Note: Running outside a virtual environment. Consider using one.${NC}"
        log "No venv detected, but system allows global pip installs"
    fi
}

check_venv_requirement

# Parse arguments
CPU_ONLY=false
CUDA_VERSION="auto"
NO_SPEECH_ENHANCEMENT=false
MINIMAL=false
DEV_MODE=false
LOCAL_LLM=false
LOCAL_LLM_BUILD=false

for arg in "$@"; do
    case $arg in
        --cpu-only)
            CPU_ONLY=true
            CUDA_VERSION="cpu"
            ;;
        --cuda118)
            CUDA_VERSION="cuda118"
            ;;
        --cuda128)
            CUDA_VERSION="cuda128"
            ;;
        # Legacy options for backward compatibility (map to supported versions)
        --cuda121|--cuda124|--cuda126)
            echo -e "${YELLOW}Warning: $arg is deprecated. Using CUDA 11.8 instead.${NC}"
            echo "         PyTorch 2.7.x dropped CUDA 12.1/12.4 support."
            echo "         Use --cuda128 for CUDA 12.8 or --cuda118 for CUDA 11.8."
            CUDA_VERSION="cuda118"
            ;;
        --no-speech-enhancement)
            NO_SPEECH_ENHANCEMENT=true
            ;;
        --minimal)
            MINIMAL=true
            NO_SPEECH_ENHANCEMENT=true
            ;;
        --dev)
            DEV_MODE=true
            ;;
        --local-llm)
            LOCAL_LLM=true
            ;;
        --local-llm-build)
            LOCAL_LLM=true
            LOCAL_LLM_BUILD=true
            ;;
        --help|-h)
            echo ""
            echo "WhisperJAV Linux Installation Script (v1.8.0)"
            echo ""
            echo "Usage: ./install_linux.sh [options]"
            echo ""
            echo "Options:"
            echo "  --cpu-only              Install CPU-only PyTorch (no CUDA)"
            echo "  --cuda118               Install PyTorch for CUDA 11.8 (universal fallback)"
            echo "  --cuda128               Install PyTorch for CUDA 12.8 (default for driver 570+)"
            echo "  --no-speech-enhancement Skip speech enhancement packages"
            echo "  --minimal               Minimal install (no speech enhancement)"
            echo "  --dev                   Install in development/editable mode"
            echo "  --local-llm             Install local LLM (tries prebuilt wheel first)"
            echo "  --local-llm-build       Install local LLM (builds from source)"
            echo "  --help, -h              Show this help message"
            echo ""
            echo "CUDA version selection:"
            echo "  Driver 570+ → CUDA 12.8 (auto-selected)"
            echo "  Driver 450+ → CUDA 11.8 (fallback)"
            echo ""
            echo "The script will auto-detect your GPU and select the appropriate"
            echo "CUDA version. Use --cuda118 or --cuda128 to override."
            echo ""
            exit 0
            ;;
    esac
done

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
log "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    echo -e "${RED}Error: Python 3.10 or higher is required. Found: $PYTHON_VERSION${NC}"
    echo -e "${RED}       Python 3.9 is no longer supported due to pysubtrans dependency.${NC}"
    log "ERROR: Python 3.10+ required. Found: $PYTHON_VERSION"
    exit 1
fi

if [ "$PYTHON_MINOR" -gt 12 ]; then
    echo -e "${YELLOW}Warning: Python 3.13+ may have compatibility issues with openai-whisper${NC}"
    log "WARNING: Python 3.13+ detected"
fi

echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"
log "Python $PYTHON_VERSION detected"
echo ""

# Auto-detect NVIDIA GPU and driver version
if [ "$CPU_ONLY" = false ]; then
    echo -e "${YELLOW}Checking for NVIDIA GPU...${NC}"
    log "Checking for NVIDIA GPU..."
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        if [ -n "$GPU_NAME" ]; then
            echo -e "${GREEN}NVIDIA GPU detected: $GPU_NAME${NC}"
            echo -e "${GREEN}Driver version: $DRIVER_VERSION${NC}"
            log "NVIDIA GPU detected: $GPU_NAME"
            log "Driver version: $DRIVER_VERSION"

            # Auto-select CUDA version based on driver if not specified
            # Simplified matrix for v1.8.0: CUDA 12.8 (modern) or 11.8 (universal fallback)
            if [ "$CUDA_VERSION" = "auto" ]; then
                DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)
                if [ "$DRIVER_MAJOR" -ge 570 ] 2>/dev/null; then
                    CUDA_VERSION="cuda128"
                    echo -e "${GREEN}Auto-selecting CUDA 12.8 based on driver $DRIVER_VERSION${NC}"
                    log "Auto-selecting CUDA 12.8"
                elif [ "$DRIVER_MAJOR" -ge 450 ] 2>/dev/null; then
                    CUDA_VERSION="cuda118"
                    echo -e "${GREEN}Auto-selecting CUDA 11.8 (universal fallback) for driver $DRIVER_VERSION${NC}"
                    echo -e "${YELLOW}Tip: Update driver to 570+ for CUDA 12.8 performance benefits${NC}"
                    log "Auto-selecting CUDA 11.8"
                else
                    echo -e "${RED}Driver $DRIVER_VERSION is too old for CUDA${NC}"
                    echo -e "${YELLOW}Switching to CPU-only installation${NC}"
                    log "Driver too old - switching to CPU-only"
                    CPU_ONLY=true
                    CUDA_VERSION="cpu"
                fi
            fi
        else
            echo -e "${RED}nvidia-smi found but no GPU detected${NC}"
            log "WARNING: nvidia-smi found but no GPU detected"
            CPU_ONLY=true
            CUDA_VERSION="cpu"
        fi
    elif [ -f "/proc/driver/nvidia/version" ]; then
        echo -e "${GREEN}NVIDIA driver detected${NC}"
        log "NVIDIA driver detected"
        if [ "$CUDA_VERSION" = "auto" ]; then
            # Conservative fallback when we can't determine driver version
            CUDA_VERSION="cuda118"
            echo -e "${YELLOW}Using CUDA 11.8 (universal fallback)${NC}"
            log "Using CUDA 11.8 (fallback)"
        fi
    else
        echo -e "${RED}No NVIDIA GPU detected!${NC}"
        echo -e "${YELLOW}Switching to CPU-only installation automatically.${NC}"
        log "No NVIDIA GPU - switching to CPU-only"
        CPU_ONLY=true
        CUDA_VERSION="cpu"
    fi
    echo ""
fi

# Display configuration
echo "============================================================"
echo "  Installation Configuration"
echo "============================================================"
if [ "$CPU_ONLY" = true ]; then
    echo "  PyTorch: CPU-only"
    log "Configuration: PyTorch=CPU-only"
else
    echo "  PyTorch: $CUDA_VERSION"
    log "Configuration: PyTorch=$CUDA_VERSION"
fi
if [ "$NO_SPEECH_ENHANCEMENT" = true ]; then
    echo "  Speech Enhancement: Disabled"
    log "Configuration: Speech Enhancement=Disabled"
else
    echo "  Speech Enhancement: Enabled"
    log "Configuration: Speech Enhancement=Enabled"
fi
if [ "$DEV_MODE" = true ]; then
    echo "  Mode: Development (editable)"
    log "Configuration: Mode=Development"
else
    echo "  Mode: Standard"
    log "Configuration: Mode=Standard"
fi
echo "============================================================"
echo ""

# ==============================================================================
# Phase 1: Upgrade pip
# ==============================================================================
log ""
log "============================================================"
log "  Phase 1: Upgrade pip"
log "============================================================"

echo -e "${YELLOW}Step 1/7: Upgrading pip...${NC}"
python3 -m pip install --upgrade pip
log "pip upgraded"
echo ""

# ==============================================================================
# Phase 2: Install PyTorch
# ==============================================================================
log ""
log "============================================================"
log "  Phase 2: PyTorch Installation"
log "============================================================"

echo -e "${YELLOW}Step 2/7: Installing PyTorch...${NC}"

# Select PyTorch URL based on CUDA version
# v1.8.0: Simplified to CUDA 12.8 + 11.8 only (aligned with PyTorch 2.7.x)
case $CUDA_VERSION in
    cpu)
        TORCH_URL="https://download.pytorch.org/whl/cpu"
        ;;
    cuda118)
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        ;;
    cuda128)
        TORCH_URL="https://download.pytorch.org/whl/cu128"
        ;;
    *)
        # Fallback to CUDA 11.8 (universal compatibility)
        TORCH_URL="https://download.pytorch.org/whl/cu118"
        ;;
esac

echo "Installing PyTorch ($CUDA_VERSION)..."
log "Installing PyTorch ($CUDA_VERSION) from $TORCH_URL..."
pip3 install --progress-bar on torch torchaudio --index-url "$TORCH_URL"
log "PyTorch installed"
echo ""

# ==============================================================================
# Phase 3: Core Dependencies
# ==============================================================================
log ""
log "============================================================"
log "  Phase 3: Core Dependencies"
log "============================================================"

echo -e "${YELLOW}Step 3/7: Installing core dependencies...${NC}"

# Phase 3.1: Core scientific stack (MUST install first)
# NumPy 1.26.x for pyvideotrans compatibility (v1.8.0)
log "Phase 3.1: Installing scientific stack..."
pip3 install "numpy>=1.26.0,<2.0" "scipy>=1.10.1" "librosa>=0.10.0"

# Phase 3.2: Audio and utility packages (including fsspec constraint)
log "Phase 3.2: Installing audio/utility packages..."
pip3 install --progress-bar on soundfile pydub tqdm colorama requests regex "psutil>=5.9.0" "fsspec>=2025.3.0"

# Phase 3.3: Subtitle and async packages
log "Phase 3.3: Installing subtitle/async packages..."
pip3 install pysrt srt aiofiles jsonschema pyloudnorm

# Phase 3.4: Config and optimization packages
# numba>=0.58.0 supports NumPy 1.22-2.0
log "Phase 3.4: Installing config packages..."
pip3 install "pydantic>=2.0,<3.0" "PyYAML>=6.0" "numba>=0.58.0"

# Phase 3.5: pyvideotrans compatibility packages (Phase 1 prep)
log "Phase 3.5: Installing pyvideotrans compatibility packages..."
pip3 install "av>=13.0.0" "imageio>=2.31.0" "imageio-ffmpeg>=0.4.9" "httpx>=0.27.0" "websockets>=13.0" "soxr>=0.3.0" || {
    echo -e "${YELLOW}Warning: pyvideotrans prep packages failed (non-fatal)${NC}"
    log "WARNING: pyvideotrans prep packages failed"
}

# Phase 3.6: Image packages (non-fatal)
pip3 install Pillow || echo -e "${YELLOW}Warning: Pillow installation failed (non-fatal)${NC}"

log "Core dependencies installed"
echo ""

# ==============================================================================
# Phase 4: Whisper Packages
# ==============================================================================
log ""
log "============================================================"
log "  Phase 4: Whisper Packages"
log "============================================================"

echo -e "${YELLOW}Step 4/7: Installing Whisper packages...${NC}"
log "Installing Whisper packages..."

pip3 install git+https://github.com/openai/whisper@main
log "openai-whisper installed"

pip3 install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main
log "stable-ts installed"

# ffmpeg-python: PyPI tarball fails, use git URL
pip3 install git+https://github.com/kkroening/ffmpeg-python.git || {
    echo -e "${YELLOW}Warning: ffmpeg-python from git failed, trying PyPI...${NC}"
    log "WARNING: ffmpeg-python git failed, trying PyPI"
    pip3 install ffmpeg-python
}
log "ffmpeg-python installed"

pip3 install "faster-whisper>=1.1.0"
log "faster-whisper installed"
echo ""

# ==============================================================================
# Phase 5: Optional Packages
# ==============================================================================
log ""
log "============================================================"
log "  Phase 5: Optional Packages"
log "============================================================"

echo -e "${YELLOW}Step 5/7: Installing optional packages...${NC}"

# HuggingFace / Transformers
log "Installing HuggingFace packages..."
pip3 install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0"

# hf_xet for faster HuggingFace downloads (optional)
pip3 install hf_xet 2>/dev/null || log "Note: hf_xet not installed (optional)"

# Translation (pysubtrans requires Python 3.10+)
log "Installing translation packages..."
pip3 install "pysubtrans>=1.5.0" "openai>=1.35.0" "google-genai>=1.39.0"

# Local LLM Translation (llama-cpp-python from JamePeng fork) - OPTIONAL
# Only installed if --local-llm or --local-llm-build is specified
# Note: Apple Silicon builds from source with Metal (~10min, fast)
#       Intel Mac uses CPU build (no Metal/GPU acceleration)
#       Linux uses prebuilt wheels or builds from source
if [ "$LOCAL_LLM" = true ]; then
    log "Installing llama-cpp-python for local LLM translation..."
    echo "Installing llama-cpp-python for local LLM translation..."

    # Detect platform
    IS_APPLE_SILICON=false
    IS_INTEL_MAC=false
    if [ "$(uname)" = "Darwin" ]; then
        if [ "$(uname -m)" = "arm64" ]; then
            IS_APPLE_SILICON=true
        else
            IS_INTEL_MAC=true
        fi
    fi

    if [ "$IS_APPLE_SILICON" = true ]; then
        # Apple Silicon: build from source with Metal (fast ~10min)
        echo "  Apple Silicon detected - building from source with Metal support."
        log "Apple Silicon detected - building with Metal"
        SOURCE_INFO=$(PYTHONPATH="$REPO_ROOT" python3 -c "from install import get_llama_cpp_source_info; url,backend,cmake,env=get_llama_cpp_source_info(); print(f'{url}|{backend}|{cmake or \"\"}|{env.get(\"CMAKE_BUILD_PARALLEL_LEVEL\", \"\")}')" 2>/dev/null)
        SOURCE_URL=$(echo "$SOURCE_INFO" | cut -d'|' -f1)
        SOURCE_BACKEND=$(echo "$SOURCE_INFO" | cut -d'|' -f2)
        SOURCE_CMAKE=$(echo "$SOURCE_INFO" | cut -d'|' -f3)
        SOURCE_PARALLEL=$(echo "$SOURCE_INFO" | cut -d'|' -f4)

        echo "  Backend: $SOURCE_BACKEND"
        log "llama-cpp-python backend: $SOURCE_BACKEND"
        if [ -n "$SOURCE_PARALLEL" ]; then
            echo "  Setting CMAKE_BUILD_PARALLEL_LEVEL=$SOURCE_PARALLEL"
            export CMAKE_BUILD_PARALLEL_LEVEL="$SOURCE_PARALLEL"
        fi
        if [ -n "$SOURCE_CMAKE" ]; then
            echo "  Setting CMAKE_ARGS=$SOURCE_CMAKE"
            export CMAKE_ARGS="$SOURCE_CMAKE"
        fi
        pip3 install "$SOURCE_URL" || {
            echo -e "${YELLOW}Warning: llama-cpp-python build failed (local LLM translation will not work)${NC}"
            log "WARNING: llama-cpp-python source build failed"
        }
    elif [ "$IS_INTEL_MAC" = true ]; then
        # Intel Mac: CPU-only build (no Metal support)
        if [ "$LOCAL_LLM_BUILD" = true ]; then
            echo "  Intel Mac detected - building CPU-only version."
            log "Intel Mac detected - building CPU-only"
            SOURCE_INFO=$(PYTHONPATH="$REPO_ROOT" python3 -c "from install import get_llama_cpp_source_info; url,backend,cmake,env=get_llama_cpp_source_info(); print(f'{url}|{backend}|{cmake or \"\"}|{env.get(\"CMAKE_BUILD_PARALLEL_LEVEL\", \"\")}')" 2>/dev/null)
            SOURCE_URL=$(echo "$SOURCE_INFO" | cut -d'|' -f1)
            SOURCE_BACKEND=$(echo "$SOURCE_INFO" | cut -d'|' -f2)
            SOURCE_PARALLEL=$(echo "$SOURCE_INFO" | cut -d'|' -f4)

            echo "  Backend: $SOURCE_BACKEND"
            log "llama-cpp-python backend: $SOURCE_BACKEND"
            if [ -n "$SOURCE_PARALLEL" ]; then
                echo "  Setting CMAKE_BUILD_PARALLEL_LEVEL=$SOURCE_PARALLEL"
                export CMAKE_BUILD_PARALLEL_LEVEL="$SOURCE_PARALLEL"
            fi
            pip3 install "$SOURCE_URL" || {
                echo -e "${YELLOW}Warning: llama-cpp-python build failed (local LLM translation will not work)${NC}"
                log "WARNING: llama-cpp-python source build failed"
            }
        else
            echo "  Intel Mac detected - no prebuilt wheels available."
            echo "  To build CPU-only version, use --local-llm-build."
            echo "  Skipping local LLM installation."
            log "Intel Mac - skipping (use --local-llm-build to build)"
        fi
    else
        # Linux: try prebuilt wheel first, fall back to source build
        WHEEL_INFO=$(PYTHONPATH="$REPO_ROOT" python3 -c "from install import get_llama_cpp_prebuilt_wheel; result=get_llama_cpp_prebuilt_wheel(); print(f'{result[0] or \"\"}|{result[1] or \"\"}')" 2>/dev/null)
        WHEEL_URL=$(echo "$WHEEL_INFO" | cut -d'|' -f1)
        WHEEL_BACKEND=$(echo "$WHEEL_INFO" | cut -d'|' -f2)

        if [ -n "$WHEEL_URL" ]; then
            # Prebuilt wheel available - use it
            echo "  Backend: $WHEEL_BACKEND"
            log "llama-cpp-python backend: $WHEEL_BACKEND"
            pip3 install "$WHEEL_URL" && {
                pip3 install "llama-cpp-python[server]"
                log "llama-cpp-python installed from prebuilt wheel"
            } || {
                echo -e "${YELLOW}Warning: Prebuilt wheel failed, local LLM translation may not work${NC}"
                log "WARNING: llama-cpp-python prebuilt wheel failed"
            }
        elif [ "$LOCAL_LLM_BUILD" = true ]; then
            # No prebuilt wheel, but user opted for source build
            SOURCE_INFO=$(PYTHONPATH="$REPO_ROOT" python3 -c "from install import get_llama_cpp_source_info; url,backend,cmake,env=get_llama_cpp_source_info(); print(f'{url}|{backend}|{cmake or \"\"}|{env.get(\"CMAKE_BUILD_PARALLEL_LEVEL\", \"\")}')" 2>/dev/null)
            SOURCE_URL=$(echo "$SOURCE_INFO" | cut -d'|' -f1)
            SOURCE_BACKEND=$(echo "$SOURCE_INFO" | cut -d'|' -f2)
            SOURCE_CMAKE=$(echo "$SOURCE_INFO" | cut -d'|' -f3)
            SOURCE_PARALLEL=$(echo "$SOURCE_INFO" | cut -d'|' -f4)

            echo "  Backend: $SOURCE_BACKEND"
            log "llama-cpp-python backend: $SOURCE_BACKEND"
            if [ -n "$SOURCE_PARALLEL" ]; then
                echo "  Setting CMAKE_BUILD_PARALLEL_LEVEL=$SOURCE_PARALLEL"
                export CMAKE_BUILD_PARALLEL_LEVEL="$SOURCE_PARALLEL"
            fi
            if [ -n "$SOURCE_CMAKE" ]; then
                echo "  Setting CMAKE_ARGS=$SOURCE_CMAKE"
                export CMAKE_ARGS="$SOURCE_CMAKE"
            fi
            pip3 install "$SOURCE_URL" || {
                echo -e "${YELLOW}Warning: llama-cpp-python build failed (local LLM translation will not work)${NC}"
                log "WARNING: llama-cpp-python source build failed"
            }
        else
            # --local-llm specified but no prebuilt wheel available
            echo "  No prebuilt wheel available for your platform."
            echo "  To build from source, use --local-llm-build."
            echo "  Skipping local LLM installation."
            log "No prebuilt wheel available, skipping (use --local-llm-build to build)"
        fi
    fi
else
    echo "Skipping local LLM (use --local-llm or --local-llm-build to install)"
    log "Skipping local LLM (not requested)"
fi

# VAD
log "Installing VAD packages..."
pip3 install "silero-vad>=6.0" auditok

if [ "$MINIMAL" = false ]; then
    pip3 install ten-vad || echo -e "${YELLOW}Warning: ten-vad installation failed (optional)${NC}"
    pip3 install "scikit-learn>=1.3.0"  # Semantic scene detection
fi
log "Optional packages installed"
echo ""

# ==============================================================================
# Phase 6: Speech Enhancement (optional but recommended)
# ==============================================================================
log ""
log "============================================================"
log "  Phase 6: Speech Enhancement"
log "============================================================"

if [ "$NO_SPEECH_ENHANCEMENT" = false ]; then
    echo -e "${YELLOW}Step 6/7: Installing speech enhancement packages...${NC}"
    echo "Note: These packages can be tricky. If installation fails, re-run with --no-speech-enhancement"
    echo "      Speech enhancement improves transcription quality in noisy audio."
    echo ""
    log "Installing speech enhancement packages..."

    # Install ModelScope dependencies first (CRITICAL: datasets must be <4.0)
    log "Installing ModelScope dependencies..."
    pip3 install addict simplejson sortedcontainers packaging
    pip3 install "datasets>=2.14.0,<4.0" || {
        echo -e "${YELLOW}Warning: datasets installation failed - modelscope may not work${NC}"
        log "WARNING: datasets installation failed"
    }

    # Install ModelScope (ZipEnhancer SOTA speech enhancement)
    echo "Installing ModelScope (ZipEnhancer)..."
    log "Installing ModelScope..."
    pip3 install "modelscope>=1.20" || {
        echo -e "${YELLOW}Warning: modelscope installation failed (optional)${NC}"
        log "WARNING: modelscope installation failed"
    }

    # Install ClearVoice from fork with relaxed librosa constraint
    echo "Installing ClearVoice (48kHz denoising)..."
    log "Installing ClearVoice..."
    pip3 install "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice" || {
        echo -e "${YELLOW}Warning: clearvoice installation failed (optional)${NC}"
        log "WARNING: clearvoice installation failed"
    }

    # BS-RoFormer (vocal isolation)
    echo "Installing BS-RoFormer (vocal isolation)..."
    log "Installing BS-RoFormer..."
    pip3 install bs-roformer-infer || {
        echo -e "${YELLOW}Warning: bs-roformer-infer installation failed (optional)${NC}"
        log "WARNING: bs-roformer-infer installation failed"
    }

    # ONNX Runtime for ZipEnhancer ONNX mode
    echo "Installing ONNX Runtime..."
    pip3 install "onnxruntime>=1.16.0" || log "Note: onnxruntime installation failed (optional)"

    log "Speech enhancement packages installed"
else
    echo -e "${YELLOW}Step 6/7: Skipping speech enhancement (--no-speech-enhancement)${NC}"
    log "Skipping speech enhancement (--no-speech-enhancement)"
fi
echo ""

# ==============================================================================
# Phase 7: WhisperJAV Application
# ==============================================================================
log ""
log "============================================================"
log "  Phase 7: WhisperJAV Application"
log "============================================================"

echo -e "${YELLOW}Step 7/7: Installing WhisperJAV...${NC}"
if [ "$DEV_MODE" = true ]; then
    log "Installing WhisperJAV in development mode..."
    pip3 install --no-deps -e .
else
    log "Installing WhisperJAV..."
    pip3 install --no-deps git+https://github.com/meizhong986/whisperjav.git
fi
log "WhisperJAV installed"
echo ""

# ==============================================================================
# Verification Phase
# ==============================================================================
log ""
log "============================================================"
log "  Verification Phase"
log "============================================================"

echo -e "${YELLOW}Verifying installation...${NC}"
log "Verifying installation..."

# Verify WhisperJAV
python3 -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully!')" && {
    echo -e "${GREEN}Installation complete!${NC}"
    log "WhisperJAV verified successfully"
} || {
    echo -e "${RED}Installation may have issues. Check the output above.${NC}"
    log "WARNING: WhisperJAV verification failed"
}

# Verify PyTorch CUDA
python3 -c "import torch; cuda_status = 'ENABLED' if torch.cuda.is_available() else 'DISABLED'; print(f'CUDA acceleration: {cuda_status}')" 2>/dev/null

# Verify critical packages
echo ""
echo "Verifying critical packages:"
log "Verifying critical packages..."

python3 -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>/dev/null || echo "  numpy: FAILED"
python3 -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>/dev/null || echo "  scipy: FAILED"
python3 -c "import librosa; print(f'  librosa: {librosa.__version__}')" 2>/dev/null || echo "  librosa: FAILED"
python3 -c "import faster_whisper; print(f'  faster-whisper: {faster_whisper.__version__}')" 2>/dev/null || echo "  faster-whisper: FAILED"
python3 -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>/dev/null || echo "  transformers: FAILED"

if [ "$NO_SPEECH_ENHANCEMENT" = false ]; then
    echo ""
    echo "Speech enhancement packages:"
    python3 -c "import modelscope; print(f'  modelscope: {modelscope.__version__}')" 2>/dev/null || echo "  modelscope: NOT INSTALLED"
    python3 -c "import clearvoice; print('  clearvoice: installed')" 2>/dev/null || echo "  clearvoice: NOT INSTALLED"
fi

echo ""

# ==============================================================================
# Summary
# ==============================================================================
log ""
log "============================================================"
log "  Installation Complete!"
log "============================================================"

echo "============================================================"
echo "  Installation Summary"
echo "============================================================"
echo ""
echo "  CPU-only mode: $CPU_ONLY"
echo "  Speech enhancement: $([ "$NO_SPEECH_ENHANCEMENT" = true ] && echo 'Disabled' || echo 'Enabled')"
echo "  Local LLM: $([ "$LOCAL_LLM" = true ] && echo 'Installed' || echo 'Not installed')"
echo ""
echo "  To run WhisperJAV:"
echo "    whisperjav video.mp4 --mode balanced"
echo ""
if [ "$LOCAL_LLM" = true ]; then
    echo "  To translate with local LLM:"
    echo "    whisperjav video.mp4 --translate --translate-provider local"
    echo ""
else
    echo "  To enable local LLM translation, re-install with:"
    echo "    ./install_linux.sh --local-llm-build    (builds from source)"
    echo "    ./install_linux.sh --local-llm          (tries prebuilt wheel first)"
    echo ""
fi
echo "  For help:"
echo "    whisperjav --help"
echo ""
echo "  Log file: $INSTALL_LOG"
echo ""
echo "  If you encounter issues with speech enhancement, re-run with:"
echo "    ./install_linux.sh --no-speech-enhancement"
echo ""

# ==============================================================================
# Optional: Create Desktop Entry
# ==============================================================================
create_desktop_entry() {
    DESKTOP_DIR="$HOME/.local/share/applications"
    DESKTOP_FILE="$DESKTOP_DIR/whisperjav.desktop"

    # Find whisperjav-gui executable
    WHISPERJAV_GUI=$(which whisperjav-gui 2>/dev/null || echo "")
    if [[ -z "$WHISPERJAV_GUI" ]]; then
        echo -e "${YELLOW}Could not find whisperjav-gui in PATH. Desktop entry skipped.${NC}"
        return
    fi

    mkdir -p "$DESKTOP_DIR"

    cat > "$DESKTOP_FILE" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=WhisperJAV
Comment=Japanese AV Subtitle Generator with AI-powered transcription
Exec=$WHISPERJAV_GUI
Icon=whisperjav
Terminal=false
Categories=AudioVideo;Audio;Video;
Keywords=whisper;subtitle;transcription;japanese;
EOF

    chmod +x "$DESKTOP_FILE"
    echo -e "${GREEN}Desktop entry created: $DESKTOP_FILE${NC}"
    log "Desktop entry created: $DESKTOP_FILE"

    # Update desktop database if available
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true
    fi
}

echo -e "${YELLOW}Would you like to create a desktop launcher entry?${NC}"
read -t 15 -p "Create desktop entry? (y/N): " CREATE_DESKTOP || CREATE_DESKTOP="n"
if [[ "$CREATE_DESKTOP" =~ ^[Yy]$ ]]; then
    create_desktop_entry
else
    echo "Skipping desktop entry creation."
    echo "You can launch WhisperJAV from the terminal with: whisperjav-gui"
fi
echo ""

log "Installation completed successfully at $(date)"
