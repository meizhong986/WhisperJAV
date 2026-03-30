#!/bin/bash
# ==============================================================================
# WhisperJAV Google Colab Installation Script
# ==============================================================================
#
# Installs WhisperJAV directly into Colab's system Python.
# Colab already has PyTorch + CUDA, so we skip torch installation entirely.
#
# Key features:
#   - Uses uv for fast package installation
#   - No venv — installs into system Python (reuses Colab's torch stack)
#   - Installs system dependencies (portaudio19-dev for pyaudio/auditok)
#   - Optional llama-cpp-python for local LLM translation (prebuilt wheel only)
#
# Usage:
#   !git clone https://github.com/meizhong986/WhisperJAV.git
#   !bash WhisperJAV/installer/install_colab.sh
#
# Debug mode (verbose output):
#   !bash WhisperJAV/installer/install_colab.sh --debug
#
# ==============================================================================

# Configuration
WHISPERJAV_REPO="https://github.com/meizhong986/WhisperJAV.git"
WHISPERJAV_BRANCH="main"
HF_WHEEL_REPO="mei986/whisperjav-wheels"
LLAMA_CPP_VERSION="0.3.21"

# Debug mode
DEBUG=false
if [[ "$1" == "--debug" ]] || [[ "$1" == "-d" ]]; then
    DEBUG=true
    set -x  # Print commands as they execute
fi

# Error handling - trap errors and show what failed
set -e
trap 'exit_code=$?; echo ""; echo "ERROR: Command failed at line $LINENO: $BASH_COMMAND"; echo "Exit code: $exit_code"; exit 1' ERR

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

section() {
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  $1${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ==============================================================================
# ENVIRONMENT DETECTION
# ==============================================================================

section "WhisperJAV Colab Installer"

# Check if running on Colab
if [[ ! -d "/content" ]]; then
    error "This script is designed for Google Colab."
    error "For regular Linux installation, use: installer/install_linux.sh"
    exit 1
fi

info "Detected Google Colab environment"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1 | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)
    info "GPU: $GPU_NAME"
    info "Driver: $DRIVER_VERSION"
else
    warn "nvidia-smi not found. GPU acceleration may not work."
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f2)
info "Python: $PYTHON_VERSION"

if [[ "$PYTHON_MAJOR" -ne 3 ]] || [[ "$PYTHON_MINOR" -lt 10 ]] || [[ "$PYTHON_MINOR" -gt 12 ]]; then
    error "Python 3.10-3.12 required. Found: $PYTHON_VERSION"
    exit 1
fi

# Verify Colab's PyTorch is available
TORCH_CHECK=$(python3 -c "
try:
    import torch
    print(f'VERSION:{torch.__version__}')
    print(f'CUDA:{torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU:{torch.cuda.get_device_name(0)}')
except Exception as e:
    print(f'ERROR:{e}')
" 2>&1)

if echo "$TORCH_CHECK" | grep -q "^ERROR:"; then
    error "Colab's PyTorch not found. This is unexpected."
    error "$TORCH_CHECK"
    exit 1
fi

TORCH_VERSION=$(echo "$TORCH_CHECK" | grep "^VERSION:" | cut -d: -f2)
CUDA_AVAILABLE=$(echo "$TORCH_CHECK" | grep "^CUDA:" | cut -d: -f2)
info "PyTorch: $TORCH_VERSION (Colab pre-installed)"
if [[ "$CUDA_AVAILABLE" == "True" ]]; then
    GPU_DETECTED=$(echo "$TORCH_CHECK" | grep "^GPU:" | cut -d: -f2)
    info "CUDA available: $GPU_DETECTED"
else
    warn "CUDA not available (CPU mode)"
fi

# ==============================================================================
# STEP 1: INSTALL UV PACKAGE MANAGER
# ==============================================================================

section "Step 1/3: Installing uv package manager"

if command -v uv &> /dev/null; then
    info "uv already installed: $(uv --version)"
else
    info "Downloading uv (fast Python package manager)..."
    if ! curl -LfS https://astral.sh/uv/install.sh | sh; then
        error "Failed to download/install uv"
        exit 1
    fi
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        success "uv installed: $(uv --version)"
    else
        error "uv installation completed but uv command not found"
        exit 1
    fi
fi

# Ensure uv is in PATH for subsequent commands
export PATH="$HOME/.local/bin:$PATH"

# ==============================================================================
# STEP 2: INSTALL WHISPERJAV (into system Python — reuses Colab's torch)
# ==============================================================================

section "Step 2/3: Installing WhisperJAV"

# Install system dependencies required for building Python packages
info "Installing system dependencies..."
SYS_PKGS="portaudio19-dev libc++1 libc++abi1 libsndfile1 ffmpeg libgl1"
if apt-get update -qq > /dev/null 2>&1 && apt-get install -y -qq $SYS_PKGS > /dev/null 2>&1; then
    success "System dependencies installed ($SYS_PKGS)"
else
    warn "Could not install some system dependencies: $SYS_PKGS"
    warn "TEN VAD or pyaudio may not work - Silero fallback will be used"
fi

info "Installing from $WHISPERJAV_REPO@$WHISPERJAV_BRANCH"
info "Reusing Colab's PyTorch $TORCH_VERSION (no reinstall needed)"
info "This may take 1-3 minutes..."

# Install WhisperJAV with all extras directly into system Python.
# --system: required since we're not in a venv
# --no-build-isolation: not needed, uv handles this
# Colab's torch/numpy/etc. are reused — only WhisperJAV's own deps are installed.
if ! uv pip install --system "git+${WHISPERJAV_REPO}@${WHISPERJAV_BRANCH}#egg=whisperjav[cli,enhance,translate,huggingface,qwen,analysis,compatibility]"; then
    error "WhisperJAV installation failed"
    exit 1
fi

# Verify WhisperJAV installation
info "Verifying WhisperJAV installation..."

# Override MPLBACKEND - Colab sets it to 'matplotlib_inline' which can
# cause issues in subprocess context. Use 'Agg' (non-interactive).
export MPLBACKEND=Agg

if python3 -c "import whisperjav; print('OK')" 2>&1; then
    success "WhisperJAV installed successfully"
else
    error "WhisperJAV import failed. Running diagnostic..."
    echo ""
    python3 -c "import whisperjav" 2>&1 || true
    echo ""
    exit 1
fi

# Verify CLI is available
WHISPERJAV_BIN=$(which whisperjav 2>/dev/null || true)
if [[ -n "$WHISPERJAV_BIN" ]]; then
    success "CLI available: $WHISPERJAV_BIN"
else
    error "WhisperJAV CLI not found in PATH"
    exit 1
fi

# ==============================================================================
# STEP 3: INSTALL LLAMA-CPP-PYTHON (OPTIONAL - for local LLM translation)
# ==============================================================================

section "Step 3/3: Installing llama-cpp-python (optional)"

info "llama-cpp-python enables local LLM translation without cloud APIs"
info "Skipping source builds (takes 7+ minutes) - using prebuilt wheels only"
echo ""

LLAMA_INSTALLED=false

# Determine wheel filename for cu126
PY_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}"
WHEEL_NAME="llama_cpp_python-${LLAMA_CPP_VERSION}-${PY_TAG}-${PY_TAG}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

info "Looking for prebuilt wheel: $WHEEL_NAME"

# --- Attempt 1: HuggingFace (mei986/whisperjav-wheels) ---
info "Checking HuggingFace for prebuilt cu126 wheel..."

HF_WHEEL_URL="https://huggingface.co/datasets/${HF_WHEEL_REPO}/resolve/main/llama-cpp-python/cu126/${WHEEL_NAME}"

if curl --output /dev/null --silent --head --fail "$HF_WHEEL_URL"; then
    info "Found wheel on HuggingFace, downloading..."
    WHEEL_PATH="/tmp/${WHEEL_NAME}"

    if curl -L --progress-bar -o "$WHEEL_PATH" "$HF_WHEEL_URL"; then
        if uv pip install --system "$WHEEL_PATH"; then
            success "llama-cpp-python installed from HuggingFace (cu126)"
            LLAMA_INSTALLED=true
            rm -f "$WHEEL_PATH"
        else
            warn "HuggingFace wheel installation failed"
            rm -f "$WHEEL_PATH"
        fi
    else
        warn "Failed to download wheel from HuggingFace"
    fi
else
    info "No cu126 wheel on HuggingFace (not yet uploaded)"
fi

# --- Attempt 2: JamePeng GitHub Releases ---
if [[ "$LLAMA_INSTALLED" == "false" ]]; then
    info "Checking JamePeng GitHub releases..."

    # Query GitHub API for cu126 releases
    GITHUB_RELEASES=$(curl -s "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=20" 2>&1)

    # Look for cu126-linux release with matching Python version
    # Extract both the download URL and the original filename
    GITHUB_WHEEL_INFO=$(echo "$GITHUB_RELEASES" | python3 -c "
import sys, json
try:
    releases = json.load(sys.stdin)
    for release in releases:
        tag = release.get('tag_name', '')
        if '-cu126-' in tag and '-linux-' in tag:
            for asset in release.get('assets', []):
                name = asset.get('name', '')
                if name.endswith('.whl') and '${PY_TAG}' in name and 'linux' in name:
                    url = asset.get('browser_download_url', '')
                    print(f'{name}\t{url}')
                    sys.exit(0)
except Exception as e:
    print(f'# Error: {e}', file=sys.stderr)
" 2>&1)

    # Parse filename and URL from tab-separated output
    GITHUB_WHEEL_FILENAME=$(echo "$GITHUB_WHEEL_INFO" | cut -f1)
    GITHUB_WHEEL_URL=$(echo "$GITHUB_WHEEL_INFO" | cut -f2)

    if [[ -n "$GITHUB_WHEEL_URL" ]] && [[ ! "$GITHUB_WHEEL_URL" =~ ^# ]] && [[ -n "$GITHUB_WHEEL_FILENAME" ]]; then
        info "Found wheel on JamePeng GitHub: $GITHUB_WHEEL_FILENAME"
        info "Downloading..."
        # Use the original wheel filename (PEP 427 compliant) — not a generic name
        WHEEL_PATH="/tmp/${GITHUB_WHEEL_FILENAME}"

        if curl -L --progress-bar -o "$WHEEL_PATH" "$GITHUB_WHEEL_URL"; then
            if uv pip install --system "$WHEEL_PATH"; then
                success "llama-cpp-python installed from JamePeng GitHub (cu126)"
                LLAMA_INSTALLED=true
                rm -f "$WHEEL_PATH"
            else
                warn "GitHub wheel installation failed"
                rm -f "$WHEEL_PATH"
            fi
        else
            warn "Failed to download wheel from GitHub"
        fi
    else
        info "No matching cu126 wheel found on JamePeng GitHub"
    fi
fi

# --- No prebuilt wheel available ---
if [[ "$LLAMA_INSTALLED" == "false" ]]; then
    echo ""
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn "  No prebuilt cu126 wheel available"
    warn "  Skipping llama-cpp-python (source build takes too long)"
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    info "Local LLM translation will not be available."
    info "You can still use cloud translation providers (deepseek, gemini, etc.)"
    info ""
    info "To install llama-cpp-python manually later:"
    info "  CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python"
fi

# ==============================================================================
# INSTALLATION COMPLETE
# ==============================================================================

section "Installation Complete!"

echo ""
echo -e "${GREEN}WhisperJAV has been installed into Colab's system Python.${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  HOW TO USE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Transcribe a video:"
echo -e "  ${GREEN}whisperjav /content/drive/MyDrive/video.mp4${NC}"
echo ""
echo "Transcribe with options:"
echo -e "  ${GREEN}whisperjav /content/drive/MyDrive/video.mp4 --mode balanced --sensitivity aggressive${NC}"
echo ""
echo "Translate subtitles (cloud API):"
echo -e "  ${GREEN}whisperjav-translate -i /content/drive/MyDrive/video.srt --provider deepseek${NC}"
echo ""

if [[ "$LLAMA_INSTALLED" == "true" ]]; then
    echo "Translate subtitles (local LLM):"
    echo -e "  ${GREEN}whisperjav-translate -i /content/drive/MyDrive/video.srt --provider local${NC}"
    echo ""
fi

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

success "Installation complete!"
