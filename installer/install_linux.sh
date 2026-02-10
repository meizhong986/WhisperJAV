#!/bin/bash
# ==============================================================================
# WhisperJAV Linux Installation Script - Thin Wrapper
# ==============================================================================
#
# ARCHITECTURAL NOTE:
# -------------------
# This is a THIN WRAPPER that delegates to install.py.
#
# WHY THIN WRAPPER:
# Before v2.0: This was 800+ lines with duplicated GPU detection, retry logic,
#              timeout handling, etc. Changes had to be made in 3 places.
# After v2.0:  All logic is in the unified installer module (Python).
#              This script just forwards arguments to install.py.
#
# WHAT THIS SCRIPT DOES:
# 1. Check for Python and virtual environment (PEP 668)
# 2. Navigate to repository root
# 3. Forward all arguments to python install.py
# 4. Return the exit code from install.py
#
# WHY KEEP SHELL SCRIPT AT ALL:
# - Handles PEP 668 (externally-managed) warning for Debian 12+/Ubuntu 24.04+
# - Provides familiar entry point for Linux users
# - Can be chmod +x and run directly
#
# For the full implementation, see:
#   - install.py (root) - Main installation script
#   - whisperjav/installer/ - Unified installer module
#
# Options (forwarded to install.py):
#   --cpu-only              Install CPU-only PyTorch
#   --cuda118               Install PyTorch for CUDA 11.8
#   --cuda128               Install PyTorch for CUDA 12.8 (default)
#   --no-speech-enhancement Skip speech enhancement packages
#   --minimal               Minimal install (transcription only)
#   --dev                   Install in development/editable mode
#   --local-llm             Install local LLM support (prebuilt wheel)
#   --local-llm-build       Install local LLM support (build from source)
#   --help                  Show help message
#
# Author: Senior Architect
# Date: 2026-01-26
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory and repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ==============================================================================
# PEP 668 / Virtual Environment Check
# ==============================================================================
#
# WHY THIS CHECK:
# Modern Linux distributions (Debian 12+, Ubuntu 24.04+) mark system Python
# as "externally-managed" which prevents pip from installing packages globally.
# This is a safety feature to prevent breaking system Python.
#
# WHY CHECK HERE (not in install.py):
# - Shell script can detect venv before Python runs
# - Can provide immediate, clear guidance to user
# - Python script would fail with confusing pip error message
#
check_venv_requirement() {
    # Check if we're in a virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${GREEN}Virtual environment detected: $VIRTUAL_ENV${NC}"
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
    fi
}

# ==============================================================================
# Main Execution
# ==============================================================================

# Detect macOS and redirect to dedicated installer
if [[ "$(uname -s)" == "Darwin" ]]; then
    echo -e "${YELLOW}macOS detected. Please use the dedicated macOS installer:${NC}"
    echo "  chmod +x installer/install_mac.sh"
    echo "  ./installer/install_mac.sh"
    echo ""
    echo "Or run install.py directly:"
    echo "  python3 install.py $@"
    exit 1
fi

echo "============================================================"
echo "  WhisperJAV Linux Installation (via install.py)"
echo "============================================================"
echo ""

# Check for Python
if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.10-3.12 first:"
    echo "  Debian/Ubuntu: sudo apt-get install python3 python3-pip python3-venv"
    echo "  Fedora/RHEL:   sudo dnf install python3 python3-pip"
    echo "  Arch Linux:    sudo pacman -S python python-pip"
    exit 1
fi
echo -e "${GREEN}Python found: $(python3 --version)${NC}"

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
    echo "  Expected location: whisperjav/installer/install_linux.sh"
    echo ""
    echo "  To install WhisperJAV:"
    echo "    git clone https://github.com/meizhong986/whisperjav.git"
    echo "    cd whisperjav"
    echo "    chmod +x installer/install_linux.sh"
    echo "    ./installer/install_linux.sh"
    echo ""
    exit 1
fi

# ==============================================================================
# Forward to install.py
# ==============================================================================
#
# WHY FORWARD ALL ARGUMENTS:
# - install.py handles argument parsing with argparse
# - All validation, help text, etc. is maintained in one place
# - No need to duplicate argument parsing in shell script
#
echo ""

# Run install.py with all arguments
python3 install.py "$@"
EXIT_CODE=$?

# ==============================================================================
# Return exit code from install.py
# ==============================================================================
#
# WHY PRESERVE EXIT CODE:
# - Allows CI/CD systems to detect installation failures
# - Allows other scripts to check if installation succeeded
#
if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo -e "${RED}============================================================${NC}"
    echo -e "${RED}  Installation failed with exit code $EXIT_CODE${NC}"
    echo -e "${RED}============================================================${NC}"
    echo "  Check the output above for error details."
    echo "  For help: python3 install.py --help"
    echo ""
fi

exit $EXIT_CODE
