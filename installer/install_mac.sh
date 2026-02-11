#!/bin/bash
# ==============================================================================
# WhisperJAV macOS Installation Script - Thin Wrapper
# ==============================================================================
#
# ARCHITECTURAL NOTE:
# -------------------
# This is a THIN WRAPPER that delegates to install.py.
# macOS-specific shell-level checks (Xcode CLI, Homebrew, PEP 668) live here.
# All real installation logic is in install.py / whisperjav/installer/.
#
# The Python-level install.py already handles macOS well (detector.py has
# DetectedPlatform.MACOS_SILICON, Metal messaging, etc). This script fills
# the gap at the shell wrapper layer.
#
# Options (forwarded to install.py):
#   --cpu-only              Install CPU-only PyTorch
#   --no-speech-enhancement Skip speech enhancement packages
#   --minimal               Minimal install (transcription only)
#   --dev                   Install in development/editable mode
#   --local-llm             Install local LLM support (prebuilt wheel)
#   --local-llm-build       Install local LLM support (build from source)
#   --help                  Show help message
#
# Author: Senior Architect
# Date: 2026-02-10
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# Architecture Detection
# ==============================================================================

detect_architecture() {
    local arch
    arch="$(uname -m)"
    if [[ "$arch" == "arm64" ]]; then
        echo -e "${GREEN}Architecture: Apple Silicon (M-series) - native GPU acceleration available${NC}"
    elif [[ "$arch" == "x86_64" ]]; then
        echo -e "${YELLOW}Architecture: Intel Mac - CPU-only mode (no GPU acceleration)${NC}"
    else
        echo -e "${YELLOW}Architecture: $arch${NC}"
    fi
}

# ==============================================================================
# Prerequisite Checks
# ==============================================================================

check_xcode_cli_tools() {
    if xcode-select -p &>/dev/null; then
        echo -e "${GREEN}Xcode CLI Tools: installed${NC}"
    else
        echo ""
        echo -e "${RED}Xcode Command Line Tools are not installed.${NC}"
        echo "They are required for compiling Python packages."
        echo ""
        echo "Install them with:"
        echo "  xcode-select --install"
        echo ""
        echo "Then re-run this script."
        exit 1
    fi
}

check_homebrew() {
    if command -v brew &>/dev/null; then
        echo -e "${GREEN}Homebrew: installed${NC}"
    else
        echo ""
        echo -e "${YELLOW}Homebrew is not installed.${NC}"
        echo "Homebrew is recommended for managing Python and FFmpeg on macOS."
        echo ""
        echo "Install it with:"
        echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
        echo ""
        echo -e "${YELLOW}Continuing without Homebrew...${NC}"
        echo ""
    fi
}

check_python() {
    if command -v python3 &>/dev/null; then
        local py_version
        py_version="$(python3 --version 2>&1)"
        echo -e "${GREEN}Python: $py_version${NC}"

        # Check version range (3.10-3.12)
        local py_minor
        py_minor="$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null)"
        if [[ "$py_minor" -lt 10 || "$py_minor" -gt 12 ]]; then
            echo ""
            echo -e "${RED}Python 3.$py_minor is not supported.${NC}"
            echo "WhisperJAV requires Python 3.10, 3.11, or 3.12."
            echo ""
            echo "Install a supported version with:"
            echo "  brew install python@3.12"
            echo ""
            echo "Or use pyenv:"
            echo "  brew install pyenv"
            echo "  pyenv install 3.12"
            echo "  pyenv global 3.12"
            exit 1
        fi
    else
        echo ""
        echo -e "${RED}Python 3 is not installed.${NC}"
        echo "Install it with:"
        echo "  brew install python@3.12"
        echo ""
        echo "Or download from: https://www.python.org/downloads/"
        exit 1
    fi
}

check_ffmpeg() {
    if command -v ffmpeg &>/dev/null; then
        echo -e "${GREEN}FFmpeg: installed${NC}"
    else
        echo ""
        echo -e "${YELLOW}FFmpeg is not installed.${NC}"
        echo "FFmpeg is required for audio/video processing."
        echo ""
        echo "Install it with:"
        echo "  brew install ffmpeg"
        echo ""
        echo -e "${YELLOW}Continuing without FFmpeg (install.py will check again)...${NC}"
        echo ""
    fi
}

# ==============================================================================
# PEP 668 / Virtual Environment Check (macOS-specific messaging)
# ==============================================================================
#
# WHY THIS CHECK:
# Homebrew Python 3.12+ and some system Pythons set EXTERNALLY-MANAGED,
# which prevents pip from installing packages globally. We detect this
# early and give macOS-appropriate guidance.
#
check_venv_requirement() {
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}Virtual environment: $VIRTUAL_ENV${NC}"
        return 0
    fi

    # Check if system has PEP 668 marker
    local PYTHON_STDLIB
    PYTHON_STDLIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('stdlib'))" 2>/dev/null)
    if [[ -f "$PYTHON_STDLIB/EXTERNALLY-MANAGED" ]]; then
        echo ""
        echo -e "${YELLOW}============================================================${NC}"
        echo -e "${YELLOW}  WARNING: Externally Managed Python (PEP 668)${NC}"
        echo -e "${YELLOW}============================================================${NC}"
        echo ""
        echo "Your Python installation (likely from Homebrew) restricts pip from"
        echo "installing packages system-wide. You need a virtual environment."
        echo ""
        echo -e "${GREEN}Option 1 (Recommended): Create a virtual environment${NC}"
        echo "  python3 -m venv ~/.venv/whisperjav"
        echo "  source ~/.venv/whisperjav/bin/activate"
        echo "  ./installer/install_mac.sh"
        echo ""
        echo -e "${GREEN}Option 2: Use pyenv to manage Python versions${NC}"
        echo "  brew install pyenv"
        echo "  pyenv install 3.12"
        echo "  pyenv global 3.12"
        echo ""

        # Ask user if they want to continue
        echo -e "${YELLOW}Do you want to continue anyway? Installation will likely fail.${NC}"
        read -t 30 -p "Continue? (y/N): " CONTINUE_ANYWAY || CONTINUE_ANYWAY="n"
        if [[ ! "$CONTINUE_ANYWAY" =~ ^[Yy]$ ]]; then
            echo ""
            echo "Installation cancelled. Please create a virtual environment first."
            exit 1
        fi
        echo ""
        echo -e "${YELLOW}Continuing at your own risk...${NC}"
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================

echo "============================================================"
echo "  WhisperJAV macOS Installation (via install.py)"
echo "============================================================"
echo ""

detect_architecture
check_xcode_cli_tools
check_homebrew
check_python
check_ffmpeg
echo ""

# Check PEP 668 / venv requirement
check_venv_requirement

# Navigate to repository root
cd "$REPO_ROOT"

# Verify install.py exists
if [[ ! -f "install.py" ]]; then
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  ERROR: install.py not found${NC}"
    echo -e "${RED}============================================================${NC}"
    echo ""
    echo "  This script must be run from the WhisperJAV repository."
    echo "  Expected location: whisperjav/installer/install_mac.sh"
    echo ""
    echo "  To install WhisperJAV:"
    echo "    git clone https://github.com/meizhong986/whisperjav.git"
    echo "    cd whisperjav"
    echo "    chmod +x installer/install_mac.sh"
    echo "    ./installer/install_mac.sh"
    echo ""
    exit 1
fi

# ==============================================================================
# Forward to install.py
# ==============================================================================

echo ""

# Run install.py with all arguments
python3 install.py "$@"
EXIT_CODE=$?

# ==============================================================================
# Return exit code from install.py
# ==============================================================================

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  Installation failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}============================================================${NC}"
    echo "  Check the output above for error details."
    echo "  For help: python3 install.py --help"
    echo ""
    exit $EXIT_CODE
fi

# ==============================================================================
# macOS Post-Install: Verify GUI and give guidance
# ==============================================================================

echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  macOS Post-Install Check${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

# Verify pywebview installed correctly
if python3 -c "import webview" 2>/dev/null; then
    echo -e "${GREEN}  GUI: ready${NC}"
    echo "  pywebview is installed. The GUI uses macOS WebKit — no"
    echo "  additional downloads or runtimes needed."
    echo ""
    echo "  Launch the GUI with:"
    echo -e "    ${GREEN}whisperjav-gui${NC}"
else
    echo -e "${YELLOW}  GUI: pywebview did not install successfully${NC}"
    echo ""
    echo "  The GUI needs pywebview, which uses macOS WebKit (built-in)."
    echo "  Try installing it manually:"
    echo "    pip install pywebview"
    echo ""
    echo "  If that fails, check that Xcode CLI Tools are installed:"
    echo "    xcode-select --install"
    echo ""
    echo "  You can still use WhisperJAV via the command line:"
    echo "    whisperjav video.mp4 --mode balanced"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Installation complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""

exit 0
