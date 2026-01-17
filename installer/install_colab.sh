#!/bin/bash
# ==============================================================================
# WhisperJAV Google Colab Installation Script
# ==============================================================================
#
# This script provides fast, isolated installation of WhisperJAV on Google Colab.
#
# Key features:
#   - Uses uv for 10-100x faster package installation
#   - Creates isolated venv to avoid numpy 2.x conflicts with Colab's ecosystem
#   - Configures PyTorch for cu126 (Colab's CUDA version)
#   - Handles llama-cpp-python installation (prebuilt wheel or source build)
#
# Usage (in Colab notebook):
#   !curl -LsSf https://raw.githubusercontent.com/meizhong986/WhisperJAV/main/installer/install_colab.sh | bash
#
# Or clone and run:
#   !git clone https://github.com/meizhong986/WhisperJAV.git
#   !bash WhisperJAV/installer/install_colab.sh
#
# ==============================================================================

set -e  # Exit on error

# Configuration
VENV_PATH="/content/whisperjav_env"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu126"
WHISPERJAV_REPO="https://github.com/meizhong986/WhisperJAV.git"
WHISPERJAV_BRANCH="main"
HF_WHEEL_REPO="mei986/whisperjav-wheels"
LLAMA_CPP_VERSION="0.3.21"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

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
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
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

# ==============================================================================
# STEP 1: INSTALL UV PACKAGE MANAGER
# ==============================================================================

section "Step 1/5: Installing uv package manager"

if command -v uv &> /dev/null; then
    info "uv already installed: $(uv --version)"
else
    info "Downloading uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"

    if command -v uv &> /dev/null; then
        success "uv installed: $(uv --version)"
    else
        error "Failed to install uv"
        exit 1
    fi
fi

# Ensure uv is in PATH for subsequent commands
export PATH="$HOME/.local/bin:$PATH"

# ==============================================================================
# STEP 2: CREATE ISOLATED VIRTUAL ENVIRONMENT
# ==============================================================================

section "Step 2/5: Creating isolated environment"

info "Creating venv at $VENV_PATH"
info "(This isolates WhisperJAV from Colab's numpy 2.x ecosystem)"

# Remove existing venv if present
if [[ -d "$VENV_PATH" ]]; then
    warn "Removing existing environment..."
    rm -rf "$VENV_PATH"
fi

# Create venv with uv (80x faster than python -m venv)
uv venv "$VENV_PATH" --python "python$PYTHON_MAJOR.$PYTHON_MINOR"

if [[ -f "$VENV_PATH/bin/python" ]]; then
    success "Virtual environment created"
else
    error "Failed to create virtual environment"
    exit 1
fi

# Helper function to run pip in venv
venv_pip() {
    uv pip install --python "$VENV_PATH/bin/python" "$@"
}

# ==============================================================================
# STEP 3: INSTALL PYTORCH (cu126)
# ==============================================================================

section "Step 3/5: Installing PyTorch (cu126)"

info "Installing PyTorch with CUDA 12.6 support..."
info "Index URL: $PYTORCH_INDEX"

venv_pip torch torchvision torchaudio --index-url "$PYTORCH_INDEX"

# Verify PyTorch installation
TORCH_VERSION=$("$VENV_PATH/bin/python" -c "import torch; print(torch.__version__)" 2>/dev/null)
CUDA_AVAILABLE=$("$VENV_PATH/bin/python" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)

if [[ -n "$TORCH_VERSION" ]]; then
    success "PyTorch installed: $TORCH_VERSION"
    if [[ "$CUDA_AVAILABLE" == "True" ]]; then
        success "CUDA is available"
    else
        warn "CUDA not available (CPU mode)"
    fi
else
    error "PyTorch installation verification failed"
    exit 1
fi

# ==============================================================================
# STEP 4: INSTALL WHISPERJAV
# ==============================================================================

section "Step 4/5: Installing WhisperJAV"

info "Installing from $WHISPERJAV_REPO@$WHISPERJAV_BRANCH"

venv_pip "git+${WHISPERJAV_REPO}@${WHISPERJAV_BRANCH}"

# Verify WhisperJAV installation
if "$VENV_PATH/bin/python" -c "import whisperjav" 2>/dev/null; then
    success "WhisperJAV installed successfully"
else
    error "WhisperJAV installation verification failed"
    exit 1
fi

# ==============================================================================
# STEP 5: INSTALL LLAMA-CPP-PYTHON (for local translation)
# ==============================================================================

section "Step 5/5: Installing llama-cpp-python (local LLM)"

LLAMA_INSTALLED=false

# Determine wheel filename for cu126
PY_TAG="cp${PYTHON_MAJOR}${PYTHON_MINOR}"
WHEEL_NAME="llama_cpp_python-${LLAMA_CPP_VERSION}-${PY_TAG}-${PY_TAG}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl"

# --- Attempt 1: HuggingFace (mei986/whisperjav-wheels) ---
info "Checking HuggingFace for prebuilt cu126 wheel..."

HF_WHEEL_URL="https://huggingface.co/datasets/${HF_WHEEL_REPO}/resolve/main/llama-cpp-python/cu126/${WHEEL_NAME}"

if curl --output /dev/null --silent --head --fail "$HF_WHEEL_URL"; then
    info "Found wheel on HuggingFace, downloading..."
    WHEEL_PATH="/tmp/${WHEEL_NAME}"
    curl -L -o "$WHEEL_PATH" "$HF_WHEEL_URL"

    if venv_pip "$WHEEL_PATH" 2>/dev/null; then
        success "llama-cpp-python installed from HuggingFace (cu126)"
        LLAMA_INSTALLED=true
        rm -f "$WHEEL_PATH"
    else
        warn "HuggingFace wheel installation failed"
        rm -f "$WHEEL_PATH"
    fi
else
    info "No cu126 wheel on HuggingFace (not yet uploaded)"
fi

# --- Attempt 2: JamePeng GitHub Releases ---
if [[ "$LLAMA_INSTALLED" == "false" ]]; then
    info "Checking JamePeng GitHub releases..."

    # Query GitHub API for cu126 releases
    GITHUB_RELEASES=$(curl -s "https://api.github.com/repos/JamePeng/llama-cpp-python/releases?per_page=20" 2>/dev/null)

    # Look for cu126-linux release with matching Python version
    GITHUB_WHEEL_URL=$(echo "$GITHUB_RELEASES" | python3 -c "
import sys, json
try:
    releases = json.load(sys.stdin)
    for release in releases:
        tag = release.get('tag_name', '')
        if '-cu126-' in tag and '-linux-' in tag:
            for asset in release.get('assets', []):
                name = asset.get('name', '')
                if name.endswith('.whl') and '${PY_TAG}' in name and 'linux' in name:
                    print(asset.get('browser_download_url', ''))
                    sys.exit(0)
except:
    pass
" 2>/dev/null)

    if [[ -n "$GITHUB_WHEEL_URL" ]]; then
        info "Found wheel on JamePeng GitHub, downloading..."
        WHEEL_PATH="/tmp/llama_cpp_python_github.whl"
        curl -L -o "$WHEEL_PATH" "$GITHUB_WHEEL_URL"

        if venv_pip "$WHEEL_PATH" 2>/dev/null; then
            success "llama-cpp-python installed from JamePeng GitHub (cu126)"
            LLAMA_INSTALLED=true
            rm -f "$WHEEL_PATH"
        else
            warn "GitHub wheel installation failed"
            rm -f "$WHEEL_PATH"
        fi
    else
        info "No matching cu126 wheel found on JamePeng GitHub"
    fi
fi

# --- Attempt 3: Build from source ---
if [[ "$LLAMA_INSTALLED" == "false" ]]; then
    echo ""
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    warn "  No prebuilt cu126 wheel available"
    warn "  Building llama-cpp-python from source..."
    warn "  This may take ~10 minutes"
    warn "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""

    # Detect compute capability for optimized build
    COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [[ -z "$COMPUTE_CAP" ]]; then
        # Fallback: T4 is compute capability 7.5
        COMPUTE_CAP="75"
    fi
    info "Building for compute capability: sm_$COMPUTE_CAP"

    # Set build optimization environment variables
    CPU_CORES=$(nproc)
    PARALLEL_JOBS=$((CPU_CORES * 3 / 4))  # Use 75% of cores
    [[ $PARALLEL_JOBS -lt 2 ]] && PARALLEL_JOBS=2
    [[ $PARALLEL_JOBS -gt 16 ]] && PARALLEL_JOBS=16

    export CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=${COMPUTE_CAP}"
    export CMAKE_BUILD_PARALLEL_LEVEL="$PARALLEL_JOBS"

    info "Build config: $PARALLEL_JOBS parallel jobs, CUDA arch sm_$COMPUTE_CAP"

    # Build from JamePeng fork (better maintained)
    if venv_pip "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git"; then
        success "llama-cpp-python built and installed from source"
        LLAMA_INSTALLED=true
    else
        warn "Source build failed. Local LLM translation will not be available."
        warn "You can still use cloud translation providers (deepseek, gemini, etc.)"
    fi
fi

# Install server extras if llama-cpp is installed
if [[ "$LLAMA_INSTALLED" == "true" ]]; then
    venv_pip "llama-cpp-python[server]" 2>/dev/null || true
fi

# ==============================================================================
# INSTALLATION COMPLETE
# ==============================================================================

section "Installation Complete!"

echo ""
echo -e "${GREEN}WhisperJAV has been installed in an isolated environment.${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  HOW TO USE${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Transcribe a video:"
echo -e "  ${GREEN}$VENV_PATH/bin/whisperjav /content/drive/MyDrive/video.mp4${NC}"
echo ""
echo "Transcribe with options:"
echo -e "  ${GREEN}$VENV_PATH/bin/whisperjav /content/drive/MyDrive/video.mp4 --mode balanced --sensitivity aggressive${NC}"
echo ""
echo "Translate subtitles (local LLM):"
echo -e "  ${GREEN}$VENV_PATH/bin/whisperjav-translate -i /content/drive/MyDrive/video.srt --provider local${NC}"
echo ""
echo "Translate subtitles (cloud API):"
echo -e "  ${GREEN}$VENV_PATH/bin/whisperjav-translate -i /content/drive/MyDrive/video.srt --provider deepseek${NC}"
echo ""
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Create convenience aliases file
cat > /content/whisperjav_aliases.sh << 'EOF'
# WhisperJAV convenience aliases
alias whisperjav='/content/whisperjav_env/bin/whisperjav'
alias whisperjav-translate='/content/whisperjav_env/bin/whisperjav-translate'
EOF

echo "Optional: Run this to enable short commands in your session:"
echo -e "  ${GREEN}source /content/whisperjav_aliases.sh${NC}"
echo ""
echo "Then you can use:"
echo -e "  ${GREEN}whisperjav /content/drive/MyDrive/video.mp4${NC}"
echo ""
