@echo off
REM ==============================================================================
REM WhisperJAV Windows Installation Script
REM ==============================================================================
REM
REM This script handles the staged installation of WhisperJAV on Windows,
REM working around pip dependency resolution conflicts (Issue #90).
REM
REM Prerequisites:
REM   - Python 3.9-3.12 in PATH
REM   - FFmpeg in PATH (download from https://www.gyan.dev/ffmpeg/builds/)
REM   - Git in PATH
REM
REM Usage:
REM   install_windows.bat [options]
REM
REM Options:
REM   --cpu-only              Install CPU-only PyTorch (no CUDA)
REM   --cuda118               Install PyTorch for CUDA 11.8
REM   --cuda121               Install PyTorch for CUDA 12.1 (default)
REM   --cuda124               Install PyTorch for CUDA 12.4
REM   --no-speech-enhancement Skip speech enhancement packages
REM   --minimal               Minimal install (transcription only)
REM   --dev                   Install in development/editable mode
REM
REM ==============================================================================

setlocal EnableDelayedExpansion

echo ============================================================
echo   WhisperJAV Windows Installation Script
echo ============================================================
echo.

REM Parse arguments
set "CPU_ONLY=0"
set "CUDA_VERSION=cuda121"
set "NO_SPEECH_ENHANCEMENT=0"
set "MINIMAL=0"
set "DEV_MODE=0"

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--cpu-only" (
    set "CPU_ONLY=1"
    set "CUDA_VERSION=cpu"
)
if /i "%~1"=="--cuda118" set "CUDA_VERSION=cuda118"
if /i "%~1"=="--cuda121" set "CUDA_VERSION=cuda121"
if /i "%~1"=="--cuda124" set "CUDA_VERSION=cuda124"
if /i "%~1"=="--no-speech-enhancement" set "NO_SPEECH_ENHANCEMENT=1"
if /i "%~1"=="--minimal" (
    set "MINIMAL=1"
    set "NO_SPEECH_ENHANCEMENT=1"
)
if /i "%~1"=="--dev" set "DEV_MODE=1"
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
shift
goto :parse_args
:done_parsing

REM Check Python
echo [Step 0/7] Checking prerequisites...
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.9-3.12 and add it to PATH
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Python %PYTHON_VERSION% detected

REM Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.9+ required. Found: %PYTHON_VERSION%
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 9 (
    echo ERROR: Python 3.9+ required. Found: %PYTHON_VERSION%
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% GTR 12 (
    echo WARNING: Python 3.13+ may have compatibility issues with openai-whisper
)

REM Check FFmpeg
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg not found in PATH
    echo FFmpeg is required for audio/video processing.
    echo Download from: https://www.gyan.dev/ffmpeg/builds/
    echo Add ffmpeg.exe location to your PATH before using WhisperJAV.
) else (
    for /f "tokens=1-3" %%a in ('ffmpeg -version 2^>^&1 ^| findstr /B "ffmpeg version"') do (
        echo FFmpeg found: %%a %%b %%c
    )
)

REM Check Git
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found in PATH
    echo Git is required for installing packages from GitHub.
    echo Download from: https://git-scm.com/download/win
    exit /b 1
)
echo Git found

REM Auto-detect NVIDIA GPU
if "%CPU_ONLY%"=="0" (
    echo.
    echo Checking for NVIDIA GPU...
    nvidia-smi --query-gpu=name --format=csv,noheader >nul 2>&1
    if errorlevel 1 (
        echo No NVIDIA GPU detected!
        echo ============================================================
        echo   WARNING: Switching to CPU-only installation automatically.
        echo   To force CUDA install, use a specific --cuda*** flag.
        echo ============================================================
        set "CUDA_VERSION=cpu"
        set "CPU_ONLY=1"
    ) else (
        for /f "delims=" %%g in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>nul') do (
            echo NVIDIA GPU detected: %%g
        )
    )
)

REM Check we're in the right directory
if not exist "setup.py" (
    echo.
    echo ERROR: setup.py not found. Run this script from the WhisperJAV source directory.
    echo.
    echo   git clone https://github.com/meizhong986/whisperjav.git
    echo   cd whisperjav
    echo   installer\install_windows.bat
    echo.
    exit /b 1
)

REM Set PyTorch index URL
if "%CUDA_VERSION%"=="cpu" set "TORCH_URL=https://download.pytorch.org/whl/cpu"
if "%CUDA_VERSION%"=="cuda118" set "TORCH_URL=https://download.pytorch.org/whl/cu118"
if "%CUDA_VERSION%"=="cuda121" set "TORCH_URL=https://download.pytorch.org/whl/cu121"
if "%CUDA_VERSION%"=="cuda124" set "TORCH_URL=https://download.pytorch.org/whl/cu124"

REM Display configuration
echo.
echo ============================================================
echo   Installation Configuration
echo ============================================================
echo   PyTorch: %CUDA_VERSION%
if "%NO_SPEECH_ENHANCEMENT%"=="1" (
    echo   Speech Enhancement: No
) else (
    echo   Speech Enhancement: Yes
)
if "%DEV_MODE%"=="1" (
    echo   Mode: Development ^(editable^)
) else (
    echo   Mode: Standard
)
echo ============================================================
echo.

REM Step 1: Upgrade pip
echo ============================================================
echo   [Step 1/7] Upgrading pip
echo ============================================================
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip
    exit /b 1
)
echo.

REM Step 2: Install PyTorch
echo ============================================================
echo   [Step 2/7] Installing PyTorch ^(%CUDA_VERSION%^)
echo ============================================================
python -m pip install torch torchaudio --index-url %TORCH_URL%
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)
echo.

REM Step 3: Install core dependencies
echo ============================================================
echo   [Step 3/7] Installing core dependencies
echo ============================================================
python -m pip install "numpy>=2.0" "scipy>=1.10.1" "librosa>=0.11.0"
python -m pip install soundfile pydub tqdm colorama requests regex
python -m pip install pysrt srt aiofiles jsonschema pyloudnorm
python -m pip install "pydantic>=2.0,<3.0" "PyYAML>=6.0" numba
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    exit /b 1
)
echo.

REM Step 4: Install Whisper packages from GitHub
echo ============================================================
echo   [Step 4/7] Installing Whisper packages
echo ============================================================
python -m pip install git+https://github.com/openai/whisper@main
if errorlevel 1 (
    echo ERROR: Failed to install openai-whisper
    exit /b 1
)
python -m pip install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main
if errorlevel 1 (
    echo ERROR: Failed to install stable-ts
    exit /b 1
)
python -m pip install git+https://github.com/kkroening/ffmpeg-python.git
python -m pip install "faster-whisper>=1.1.0"
if errorlevel 1 (
    echo ERROR: Failed to install faster-whisper
    exit /b 1
)
echo.

REM Step 5: Install optional packages
echo ============================================================
echo   [Step 5/7] Installing optional packages
echo ============================================================

REM HuggingFace / Transformers
python -m pip install "huggingface-hub>=0.25.0" "transformers>=4.40.0" "accelerate>=0.26.0"

REM Translation
python -m pip install "PySubtrans>=0.7.0" "openai>=1.35.0" "google-genai>=1.39.0"

REM VAD
python -m pip install "silero-vad>=6.0" auditok

if "%MINIMAL%"=="0" (
    echo Installing TEN VAD ^(optional^)...
    python -m pip install ten-vad 2>nul
    if errorlevel 1 echo WARNING: ten-vad installation failed ^(optional^)

    python -m pip install "scikit-learn>=1.3.0"
)
echo.

REM Step 6: Speech Enhancement (optional)
if "%NO_SPEECH_ENHANCEMENT%"=="0" (
    echo ============================================================
    echo   [Step 6/7] Installing speech enhancement packages
    echo ============================================================
    echo Note: These packages can be tricky. Failures here are non-fatal.
    echo.

    python -m pip install addict simplejson sortedcontainers packaging
    python -m pip install "datasets>=2.14.0,<4.0"

    echo Installing ModelScope...
    python -m pip install "modelscope>=1.20" 2>nul
    if errorlevel 1 echo WARNING: modelscope installation failed ^(optional^)

    echo Installing ClearVoice...
    python -m pip install "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice" 2>nul
    if errorlevel 1 echo WARNING: clearvoice installation failed ^(optional^)

    echo Installing BS-RoFormer...
    python -m pip install bs-roformer-infer 2>nul
    if errorlevel 1 echo WARNING: bs-roformer-infer installation failed ^(optional^)

    python -m pip install "onnxruntime>=1.16.0" 2>nul
    echo.
) else (
    echo ============================================================
    echo   [Step 6/7] Skipping speech enhancement ^(--no-speech-enhancement^)
    echo ============================================================
    echo.
)

REM GUI dependencies (Windows-specific)
echo Installing GUI dependencies...
python -m pip install "pywebview>=5.0.0" 2>nul
python -m pip install "pythonnet>=3.0" "pywin32>=305" 2>nul
echo.

REM Step 7: Install WhisperJAV
echo ============================================================
echo   [Step 7/7] Installing WhisperJAV
echo ============================================================
if "%DEV_MODE%"=="1" (
    python -m pip install --no-deps -e .
) else (
    python -m pip install --no-deps .
)
if errorlevel 1 (
    echo ERROR: Failed to install WhisperJAV
    exit /b 1
)
echo.

REM Verify installation
echo ============================================================
echo   Verifying Installation
echo ============================================================
python -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully!')"
if errorlevel 1 (
    echo WARNING: Could not verify installation
)
echo.

REM Summary
echo ============================================================
echo   Installation Complete!
echo ============================================================
echo.
echo   PyTorch: %CUDA_VERSION%
if "%NO_SPEECH_ENHANCEMENT%"=="1" (
    echo   Speech Enhancement: Disabled
) else (
    echo   Speech Enhancement: Enabled
)
echo.
echo   To run WhisperJAV:
echo     whisperjav video.mp4 --mode balanced
echo.
echo   To run with GUI:
echo     whisperjav-gui
echo.
echo   For help:
echo     whisperjav --help
echo.
echo   If you encounter issues with speech enhancement, re-run with:
echo     installer\install_windows.bat --no-speech-enhancement
echo.

exit /b 0

:show_help
echo.
echo WhisperJAV Windows Installation Script
echo.
echo Usage: install_windows.bat [options]
echo.
echo Options:
echo   --cpu-only              Install CPU-only PyTorch ^(no CUDA^)
echo   --cuda118               Install PyTorch for CUDA 11.8
echo   --cuda121               Install PyTorch for CUDA 12.1 ^(default^)
echo   --cuda124               Install PyTorch for CUDA 12.4
echo   --no-speech-enhancement Skip speech enhancement packages
echo   --minimal               Minimal install ^(transcription only^)
echo   --dev                   Install in development/editable mode
echo   --help, -h              Show this help message
echo.
echo Examples:
echo   install_windows.bat                    # Standard install with CUDA 12.1
echo   install_windows.bat --cpu-only         # CPU-only install
echo   install_windows.bat --minimal --dev    # Minimal dev install
echo.
exit /b 0
