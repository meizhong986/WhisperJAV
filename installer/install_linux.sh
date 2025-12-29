#!/bin/bash
# ==============================================================================
# WhisperJAV Linux Installation Script
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
#   --cpu-only    Install CPU-only PyTorch (no CUDA)
#   --no-speech-enhancement    Skip speech enhancement packages (clearvoice, etc.)
#   --minimal     Install minimal version (no speech enhancement, no TEN VAD)
#
# ==============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: FFmpeg is not installed.${NC}"
    echo "Please install FFmpeg first:"
    echo "  Debian/Ubuntu: sudo apt-get install ffmpeg"
    echo "  Fedora/RHEL:   sudo dnf install ffmpeg"
    echo "  Arch Linux:    sudo pacman -S ffmpeg"
    exit 1
fi
echo -e "${GREEN}FFmpeg found: $(ffmpeg -version | head -n1)${NC}"

# Parse arguments
CPU_ONLY=false
NO_SPEECH_ENHANCEMENT=false
MINIMAL=false

for arg in "$@"; do
    case $arg in
        --cpu-only)
            CPU_ONLY=true
            ;;
        --no-speech-enhancement)
            NO_SPEECH_ENHANCEMENT=true
            ;;
        --minimal)
            MINIMAL=true
            NO_SPEECH_ENHANCEMENT=true
            ;;
    esac
done

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${RED}Error: Python 3.9 or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi

if [ "$PYTHON_MINOR" -gt 12 ]; then
    echo -e "${YELLOW}Warning: Python 3.13+ may have compatibility issues with openai-whisper${NC}"
fi

echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}Step 1/6: Upgrading pip...${NC}"
python3 -m pip install --upgrade pip
echo ""

# Install PyTorch first (before anything else)
echo -e "${YELLOW}Step 2/6: Installing PyTorch...${NC}"
if [ "$CPU_ONLY" = true ]; then
    echo "Installing CPU-only PyTorch..."
    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing PyTorch with CUDA support..."
    pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
fi
echo ""

# Install core dependencies that don't have conflicts
echo -e "${YELLOW}Step 3/6: Installing core dependencies...${NC}"
pip3 install numpy>=2.0 scipy>=1.10.1 librosa>=0.11.0
pip3 install soundfile pydub tqdm colorama requests regex
pip3 install pysrt srt aiofiles jsonschema pyloudnorm
pip3 install pydantic>=2.0 PyYAML>=6.0
echo ""

# Install Whisper packages from git (avoid PyPI conflicts)
echo -e "${YELLOW}Step 4/6: Installing Whisper packages...${NC}"
pip3 install git+https://github.com/openai/whisper@main
pip3 install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main
pip3 install faster-whisper>=1.1.0
echo ""

# Install optional packages
echo -e "${YELLOW}Step 5/6: Installing optional packages...${NC}"

# HuggingFace / Transformers
pip3 install huggingface-hub>=0.25.0 transformers>=4.40.0 accelerate>=0.26.0

# Translation
pip3 install PySubtrans>=0.7.0 openai>=1.35.0 google-genai>=1.39.0

# VAD
pip3 install silero-vad>=6.0 auditok

if [ "$MINIMAL" = false ]; then
    pip3 install ten-vad || echo -e "${YELLOW}Warning: ten-vad installation failed (optional)${NC}"
fi

# Speech Enhancement (optional, can cause conflicts)
if [ "$NO_SPEECH_ENHANCEMENT" = false ]; then
    echo ""
    echo -e "${YELLOW}Installing speech enhancement packages...${NC}"
    echo "Note: These packages can be tricky. If installation fails, re-run with --no-speech-enhancement"

    # Install modelscope and its dependencies first
    pip3 install addict simplejson sortedcontainers packaging
    pip3 install "datasets>=2.14.0,<4.0"
    pip3 install modelscope>=1.20 || echo -e "${YELLOW}Warning: modelscope installation failed${NC}"

    # Install clearvoice from fork (supports numpy 2.x)
    pip3 install "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice" || \
        echo -e "${YELLOW}Warning: clearvoice installation failed (speech enhancement will be disabled)${NC}"

    # BS-RoFormer
    pip3 install bs-roformer-infer || echo -e "${YELLOW}Warning: bs-roformer-infer installation failed${NC}"

    pip3 install onnxruntime>=1.16.0
fi
echo ""

# Install WhisperJAV itself
echo -e "${YELLOW}Step 6/6: Installing WhisperJAV...${NC}"
pip3 install --no-deps git+https://github.com/meizhong986/whisperjav.git
echo ""

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python3 -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully!')" && \
    echo -e "${GREEN}Installation complete!${NC}" || \
    echo -e "${RED}Installation may have issues. Check the output above.${NC}"

echo ""
echo "============================================================"
echo "  Installation Summary"
echo "============================================================"
echo ""
echo "  CPU-only mode: $CPU_ONLY"
echo "  Speech enhancement: $([ "$NO_SPEECH_ENHANCEMENT" = true ] && echo 'Disabled' || echo 'Enabled')"
echo ""
echo "  To run WhisperJAV:"
echo "    whisperjav video.mp4 --mode balanced"
echo ""
echo "  For help:"
echo "    whisperjav --help"
echo ""
