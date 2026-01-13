@echo off
REM ==============================================================================
REM WhisperJAV Windows Installation Script
REM ==============================================================================
REM
REM This script handles the staged installation of WhisperJAV on Windows,
REM working around pip dependency resolution conflicts (Issue #90).
REM
REM Prerequisites:
REM   - Python 3.10-3.12 in PATH (3.9 no longer supported due to pysubtrans)
REM   - FFmpeg in PATH (download from https://www.gyan.dev/ffmpeg/builds/)
REM   - Git in PATH
REM
REM Usage:
REM   install_windows.bat [options]
REM
REM Options:
REM   --cpu-only              Install CPU-only PyTorch (no CUDA)
REM   --cuda118               Install PyTorch for CUDA 11.8
REM   --cuda121               Install PyTorch for CUDA 12.1
REM   --cuda124               Install PyTorch for CUDA 12.4
REM   --cuda126               Install PyTorch for CUDA 12.6
REM   --cuda128               Install PyTorch for CUDA 12.8 (default for driver 570+)
REM   --no-speech-enhancement Skip speech enhancement packages
REM   --minimal               Minimal install (transcription only)
REM   --dev                   Install in development/editable mode
REM   --local-llm             Install local LLM (fast - prebuilt wheel only)
REM   --local-llm-build       Install local LLM (slow - builds from source if needed)
REM
REM ==============================================================================

setlocal EnableDelayedExpansion

REM Initialize log file
set "INSTALL_LOG=%~dp0install_log_windows.txt"
echo. > "%INSTALL_LOG%"
call :log "============================================================"
call :log "  WhisperJAV Windows Installation Script"
call :log "  Started: %DATE% %TIME%"
call :log "============================================================"

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
set "LOCAL_LLM=0"
set "LOCAL_LLM_BUILD=0"

:parse_args
if "%~1"=="" goto :done_parsing
if /i "%~1"=="--cpu-only" (
    set "CPU_ONLY=1"
    set "CUDA_VERSION=cpu"
)
if /i "%~1"=="--cuda118" set "CUDA_VERSION=cuda118"
if /i "%~1"=="--cuda121" set "CUDA_VERSION=cuda121"
if /i "%~1"=="--cuda124" set "CUDA_VERSION=cuda124"
if /i "%~1"=="--cuda126" set "CUDA_VERSION=cuda126"
if /i "%~1"=="--cuda128" set "CUDA_VERSION=cuda128"
if /i "%~1"=="--no-speech-enhancement" set "NO_SPEECH_ENHANCEMENT=1"
if /i "%~1"=="--minimal" (
    set "MINIMAL=1"
    set "NO_SPEECH_ENHANCEMENT=1"
)
if /i "%~1"=="--dev" set "DEV_MODE=1"
if /i "%~1"=="--local-llm" set "LOCAL_LLM=1"
if /i "%~1"=="--local-llm-build" (
    set "LOCAL_LLM=1"
    set "LOCAL_LLM_BUILD=1"
)
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
shift
goto :parse_args
:done_parsing

REM ============================================================
REM Phase 1: Preflight Checks
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 1: Preflight Checks"
call :log "============================================================"

REM Check disk space (require 8GB free)
echo [Step 0/7] Checking prerequisites...
call :log "Checking disk space..."
for /f "tokens=3" %%a in ('dir /-c "%~d0\" 2^>nul ^| findstr /c:"bytes free"') do set "FREE_BYTES=%%a"
set "FREE_BYTES=%FREE_BYTES:,=%"
REM Simple check: if free bytes string is less than 10 chars, likely less than 8GB
if defined FREE_BYTES (
    call :log "Free disk space check: PASSED"
) else (
    call :log "WARNING: Could not determine disk space"
)

REM Check network connectivity (use curl instead of ping - PyPI blocks ICMP)
call :log "Checking network connectivity to PyPI..."
curl -s --head --max-time 5 https://pypi.org >nul 2>&1
if errorlevel 1 (
    echo WARNING: Cannot reach pypi.org. Network connection may be required.
    call :log "WARNING: Network check failed (pypi.org unreachable)"
) else (
    call :log "Network check: OK"
)

REM Check Python
call :log "Checking Python..."
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found in PATH
    echo Please install Python 3.10-3.12 and add it to PATH
    call :log "ERROR: Python not found in PATH"
    call :create_failure_file "Python not found in PATH"
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYTHON_VERSION=%%v
echo Python %PYTHON_VERSION% detected
call :log "Python %PYTHON_VERSION% detected"

REM Extract major.minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set "PY_MAJOR=%%a"
    set "PY_MINOR=%%b"
)

if %PY_MAJOR% LSS 3 (
    echo ERROR: Python 3.10+ required. Found: %PYTHON_VERSION%
    echo        Python 3.9 is no longer supported due to pysubtrans dependency.
    call :log "ERROR: Python 3.10+ required. Found: %PYTHON_VERSION%"
    call :create_failure_file "Python 3.10+ required"
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% LSS 10 (
    echo ERROR: Python 3.10+ required. Found: %PYTHON_VERSION%
    echo        Python 3.9 is no longer supported due to pysubtrans dependency.
    call :log "ERROR: Python 3.10+ required. Found: %PYTHON_VERSION%"
    call :create_failure_file "Python 3.10+ required"
    exit /b 1
)
if %PY_MAJOR%==3 if %PY_MINOR% GTR 12 (
    echo WARNING: Python 3.13+ may have compatibility issues with openai-whisper
    call :log "WARNING: Python 3.13+ detected - may have compatibility issues"
)

REM Check FFmpeg
call :log "Checking FFmpeg..."
where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo WARNING: FFmpeg not found in PATH
    echo FFmpeg is required for audio/video processing.
    echo Download from: https://www.gyan.dev/ffmpeg/builds/
    echo Add ffmpeg.exe location to your PATH before using WhisperJAV.
    call :log "WARNING: FFmpeg not found in PATH"
) else (
    for /f "tokens=1-3" %%a in ('ffmpeg -version 2^>^&1 ^| findstr /B "ffmpeg version"') do (
        echo FFmpeg found: %%a %%b %%c
        call :log "FFmpeg found: %%a %%b %%c"
    )
)

REM Check Git
call :log "Checking Git..."
where git >nul 2>&1
if errorlevel 1 (
    echo ERROR: Git not found in PATH
    echo Git is required for installing packages from GitHub.
    echo Download from: https://git-scm.com/download/win
    call :log "ERROR: Git not found in PATH"
    call :create_failure_file "Git not found in PATH"
    exit /b 1
)
echo Git found
call :log "Git found"

REM Check WebView2 (Windows GUI requirement)
call :log "Checking WebView2 runtime..."
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
if errorlevel 1 (
    reg query "HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}" >nul 2>&1
)
if errorlevel 1 (
    echo.
    echo ============================================================
    echo   WARNING: WebView2 Runtime Not Detected
    echo ============================================================
    echo   The WhisperJAV GUI requires Microsoft Edge WebView2.
    echo   Download from: https://go.microsoft.com/fwlink/p/?LinkId=2124703
    echo   Install WebView2 before using whisperjav-gui.
    echo ============================================================
    echo.
    call :log "WARNING: WebView2 runtime NOT DETECTED"
) else (
    echo WebView2 runtime: Detected
    call :log "WebView2 runtime: Detected"
)

REM ============================================================
REM Phase 2: GPU and CUDA Detection
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 2: GPU and CUDA Detection"
call :log "============================================================"

if "%CPU_ONLY%"=="0" (
    echo.
    echo Checking for NVIDIA GPU...
    call :log "Checking for NVIDIA GPU..."

    REM Try to get driver version for smarter CUDA selection
    REM Note: Using temp file because for /f mangles comma in --format=csv,noheader
    set "DRIVER_VERSION="
    set "GPU_NAME="
    nvidia-smi --query-gpu=driver_version,name --format=csv,noheader > "%TEMP%\whisperjav_gpu.txt" 2>nul
    for /f "tokens=1,2 delims=," %%a in (%TEMP%\whisperjav_gpu.txt) do (
        set "DRIVER_VERSION=%%a"
        set "GPU_NAME=%%b"
    )
    del "%TEMP%\whisperjav_gpu.txt" 2>nul

    if defined GPU_NAME (
        echo NVIDIA GPU detected: !GPU_NAME!
        echo Driver version: !DRIVER_VERSION!
        call :log "NVIDIA GPU detected: !GPU_NAME!"
        call :log "Driver version: !DRIVER_VERSION!"

        REM Auto-select CUDA version based on driver if user didn't specify
        if "%CUDA_VERSION%"=="cuda121" (
            REM Check if driver supports newer CUDA
            REM Driver 570+ supports CUDA 12.8
            REM Driver 560+ supports CUDA 12.6
            REM Driver 551+ supports CUDA 12.4
            REM Driver 531+ supports CUDA 12.1
            for /f "tokens=1 delims=." %%d in ("!DRIVER_VERSION!") do set "DRIVER_MAJOR=%%d"
            if defined DRIVER_MAJOR (
                if !DRIVER_MAJOR! GEQ 570 (
                    echo Auto-selecting CUDA 12.8 based on driver !DRIVER_VERSION!
                    call :log "Auto-selecting CUDA 12.8 based on driver"
                    set "CUDA_VERSION=cuda128"
                ) else if !DRIVER_MAJOR! GEQ 560 (
                    echo Auto-selecting CUDA 12.6 based on driver !DRIVER_VERSION!
                    call :log "Auto-selecting CUDA 12.6 based on driver"
                    set "CUDA_VERSION=cuda126"
                ) else if !DRIVER_MAJOR! GEQ 551 (
                    echo Auto-selecting CUDA 12.4 based on driver !DRIVER_VERSION!
                    call :log "Auto-selecting CUDA 12.4 based on driver"
                    set "CUDA_VERSION=cuda124"
                ) else (
                    echo Using CUDA 12.1 for driver !DRIVER_VERSION!
                    call :log "Using CUDA 12.1 for driver"
                )
            )
        )
    ) else (
        echo No NVIDIA GPU detected!
        echo ============================================================
        echo   WARNING: Switching to CPU-only installation automatically.
        echo   To force CUDA install, use a specific --cuda*** flag.
        echo ============================================================
        call :log "No NVIDIA GPU detected - switching to CPU-only"
        set "CUDA_VERSION=cpu"
        set "CPU_ONLY=1"
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
    call :log "ERROR: setup.py not found"
    call :create_failure_file "setup.py not found - wrong directory"
    exit /b 1
)

REM Set PyTorch index URL with version pinning for stability
if "%CUDA_VERSION%"=="cpu" (
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
    set "TORCH_PACKAGES=torch torchaudio"
)
if "%CUDA_VERSION%"=="cuda118" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu118"
    set "TORCH_PACKAGES=torch torchaudio"
)
if "%CUDA_VERSION%"=="cuda121" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu121"
    set "TORCH_PACKAGES=torch torchaudio"
)
if "%CUDA_VERSION%"=="cuda124" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu124"
    set "TORCH_PACKAGES=torch torchaudio"
)
if "%CUDA_VERSION%"=="cuda126" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu126"
    set "TORCH_PACKAGES=torch torchaudio"
)
if "%CUDA_VERSION%"=="cuda128" (
    set "TORCH_URL=https://download.pytorch.org/whl/cu128"
    set "TORCH_PACKAGES=torch torchaudio"
)

REM Display configuration
echo.
echo ============================================================
echo   Installation Configuration
echo ============================================================
echo   PyTorch: %CUDA_VERSION%
call :log "Configuration: PyTorch=%CUDA_VERSION%"
if "%NO_SPEECH_ENHANCEMENT%"=="1" (
    echo   Speech Enhancement: No
    call :log "Configuration: Speech Enhancement=No"
) else (
    echo   Speech Enhancement: Yes
    call :log "Configuration: Speech Enhancement=Yes"
)
if "%DEV_MODE%"=="1" (
    echo   Mode: Development ^(editable^)
    call :log "Configuration: Mode=Development"
) else (
    echo   Mode: Standard
    call :log "Configuration: Mode=Standard"
)
echo ============================================================
echo.

REM ============================================================
REM Phase 3: PyTorch Installation
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 3: PyTorch Installation"
call :log "============================================================"

echo ============================================================
echo   [Step 1/7] Upgrading pip
echo ============================================================
call :run_pip_with_retry "install --upgrade pip" "Upgrade pip"
if errorlevel 1 (
    call :create_failure_file "Failed to upgrade pip"
    exit /b 1
)
echo.

echo ============================================================
echo   [Step 2/7] Installing PyTorch ^(%CUDA_VERSION%^)
echo ============================================================
call :log "Installing PyTorch with %CUDA_VERSION%..."
call :run_pip_with_retry "install %TORCH_PACKAGES% --index-url %TORCH_URL%" "Install PyTorch"
if errorlevel 1 (
    call :create_failure_file "Failed to install PyTorch"
    exit /b 1
)

REM Verify PyTorch installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>nul
if errorlevel 1 (
    echo WARNING: Could not verify PyTorch installation
    call :log "WARNING: PyTorch verification failed"
) else (
    call :log "PyTorch verified successfully"
)
echo.

REM ============================================================
REM Phase 4: Core Dependencies (with constraints)
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 4: Core Dependencies"
call :log "============================================================"

echo ============================================================
echo   [Step 3/7] Installing core dependencies
echo ============================================================

REM Phase 4.1: Core scientific stack (MUST install first to establish versions)
REM scipy>=1.14.0 required for NumPy 2.0 ABI compatibility
call :log "Phase 4.1: Installing core scientific stack..."
call :run_pip_with_retry "install numpy>=2.0 scipy>=1.14.0 librosa>=0.11.0" "Install scientific packages"
if errorlevel 1 goto :install_failed

REM Phase 4.2: Audio and utility packages
call :log "Phase 4.2: Installing audio/utility packages..."
call :run_pip_with_retry "install --progress-bar on soundfile pydub tqdm colorama requests regex psutil>=5.9.0 fsspec>=2025.3.0" "Install audio/utility packages"
if errorlevel 1 goto :install_failed

REM Phase 4.3: Subtitle and async packages
call :log "Phase 4.3: Installing subtitle/async packages..."
call :run_pip_with_retry "install pysrt srt aiofiles jsonschema pyloudnorm" "Install subtitle/async packages"
if errorlevel 1 goto :install_failed

REM Phase 4.4: Config and optimization packages
REM numba>=0.60.0 required for NumPy 2.0 compatibility
call :log "Phase 4.4: Installing config packages..."
call :run_pip_with_retry "install pydantic>=2.0,<3.0 PyYAML>=6.0 numba>=0.60.0" "Install config packages"
if errorlevel 1 goto :install_failed

REM Phase 4.5: Image/plotting packages (non-fatal)
call :run_pip_with_retry "install Pillow" "Install image packages"
if errorlevel 1 (
    echo WARNING: Pillow installation failed (non-fatal)
    call :log "WARNING: Pillow installation failed"
)
echo.

REM ============================================================
REM Phase 5: Whisper Packages
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 5: Whisper Packages"
call :log "============================================================"

echo ============================================================
echo   [Step 4/7] Installing Whisper packages
echo ============================================================

call :run_pip_with_retry "install git+https://github.com/openai/whisper@main" "Install openai-whisper"
if errorlevel 1 (
    call :create_failure_file "Failed to install openai-whisper"
    exit /b 1
)

call :run_pip_with_retry "install git+https://github.com/meizhong986/stable-ts-fix-setup.git@main" "Install stable-ts"
if errorlevel 1 (
    call :create_failure_file "Failed to install stable-ts"
    exit /b 1
)

REM ffmpeg-python: PyPI tarball fails, use git URL
call :run_pip_with_retry "install git+https://github.com/kkroening/ffmpeg-python.git" "Install ffmpeg-python (git)"
if errorlevel 1 (
    echo WARNING: ffmpeg-python from git failed, trying PyPI version...
    call :log "WARNING: ffmpeg-python git failed, trying PyPI..."
    call :run_pip_with_retry "install ffmpeg-python" "Install ffmpeg-python (PyPI fallback)"
)

call :run_pip_with_retry "install faster-whisper>=1.1.0" "Install faster-whisper"
if errorlevel 1 (
    call :create_failure_file "Failed to install faster-whisper"
    exit /b 1
)
echo.

REM ============================================================
REM Phase 6: Optional Packages
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 6: Optional Packages"
call :log "============================================================"

echo ============================================================
echo   [Step 5/7] Installing optional packages
echo ============================================================

REM HuggingFace / Transformers
call :log "Installing HuggingFace packages..."
call :run_pip_with_retry "install huggingface-hub>=0.25.0 transformers>=4.40.0 accelerate>=0.26.0" "Install HuggingFace"
if errorlevel 1 (
    echo WARNING: HuggingFace packages failed (non-fatal)
    call :log "WARNING: HuggingFace packages failed"
)

REM hf_xet for faster HuggingFace downloads (optional)
python -m pip install hf_xet 2>nul
if errorlevel 1 (
    call :log "Note: hf_xet not installed (optional, faster HF downloads)"
)

REM Translation (pysubtrans requires Python 3.10+)
call :log "Installing translation packages..."
call :run_pip_with_retry "install pysubtrans>=1.5.0 openai>=1.35.0 google-genai>=1.39.0" "Install translation packages"
if errorlevel 1 (
    echo WARNING: Translation packages failed (non-fatal)
    call :log "WARNING: Translation packages failed"
)

REM Local LLM Translation (llama-cpp-python from JamePeng fork) - OPTIONAL
REM Only installed if --local-llm or --local-llm-build is specified
if "%LOCAL_LLM%"=="1" (
    call :log "Installing llama-cpp-python for local LLM translation..."
    echo Installing llama-cpp-python for local LLM translation...

    REM Try prebuilt wheel first
    python -c "from install import get_llama_cpp_prebuilt_wheel; result=get_llama_cpp_prebuilt_wheel(); print(f'WHEEL_URL={result[0] or \"\"}'); print(f'WHEEL_BACKEND={result[1] or \"\"}')" > "%TEMP%\llama_wheel.txt" 2>nul
    set "WHEEL_URL="
    set "WHEEL_BACKEND="
    for /f "tokens=1,* delims==" %%a in (%TEMP%\llama_wheel.txt) do set "%%a=%%b"
    del "%TEMP%\llama_wheel.txt" 2>nul

    if not "!WHEEL_URL!"=="" (
        REM Prebuilt wheel available - use it
        echo   Backend: !WHEEL_BACKEND!
        call :log "llama-cpp-python backend: !WHEEL_BACKEND!"
        python -m pip install "!WHEEL_URL!" 2>nul
        if errorlevel 1 (
            echo WARNING: Prebuilt wheel failed, local LLM translation may not work
            call :log "WARNING: llama-cpp-python prebuilt wheel failed"
        ) else (
            python -m pip install "llama-cpp-python[server]" 2>nul
            call :log "llama-cpp-python installed from prebuilt wheel"
        )
    ) else if "%LOCAL_LLM_BUILD%"=="1" (
        REM No prebuilt wheel, but user opted for source build
        python -c "from install import get_llama_cpp_source_info; url,backend,cmake=get_llama_cpp_source_info(); print(f'SOURCE_URL={url}'); print(f'SOURCE_BACKEND={backend}'); print(f'SOURCE_CMAKE={cmake or \"\"}')" > "%TEMP%\llama_source.txt" 2>nul
        set "SOURCE_URL="
        set "SOURCE_BACKEND="
        set "SOURCE_CMAKE="
        for /f "tokens=1,* delims==" %%a in (%TEMP%\llama_source.txt) do set "%%a=%%b"
        del "%TEMP%\llama_source.txt" 2>nul

        echo   Backend: !SOURCE_BACKEND!
        call :log "llama-cpp-python backend: !SOURCE_BACKEND!"
        if not "!SOURCE_CMAKE!"=="" (
            echo   Setting CMAKE_ARGS=!SOURCE_CMAKE!
            set "CMAKE_ARGS=!SOURCE_CMAKE!"
        )
        python -m pip install "!SOURCE_URL!" 2>nul
        if errorlevel 1 (
            echo WARNING: llama-cpp-python build failed ^(local LLM translation will not work^)
            call :log "WARNING: llama-cpp-python source build failed"
        ) else (
            call :log "llama-cpp-python built from source"
        )
    ) else (
        REM --local-llm specified but no prebuilt wheel available
        echo   No prebuilt wheel available for your platform.
        echo   To build from source, use --local-llm-build instead.
        echo   Skipping local LLM installation.
        call :log "No prebuilt wheel available, skipping (use --local-llm-build to build)"
    )
) else (
    echo Skipping local LLM ^(use --local-llm or --local-llm-build to install^)
    call :log "Skipping local LLM (not requested)"
)

REM VAD
call :log "Installing VAD packages..."
call :run_pip_with_retry "install silero-vad>=6.0 auditok" "Install VAD packages"
if errorlevel 1 (
    echo WARNING: VAD packages failed (non-fatal)
    call :log "WARNING: VAD packages failed"
)

if "%MINIMAL%"=="0" (
    echo Installing TEN VAD ^(optional^)...
    call :log "Installing TEN VAD..."
    python -m pip install ten-vad 2>nul
    if errorlevel 1 (
        echo WARNING: ten-vad installation failed ^(optional^)
        call :log "WARNING: ten-vad installation failed"
    )

    REM scikit-learn>=1.5.0 required for NumPy 2.0 compatibility
    call :run_pip_with_retry "install scikit-learn>=1.5.0" "Install scikit-learn"
    if errorlevel 1 (
        echo WARNING: scikit-learn failed (non-fatal)
        call :log "WARNING: scikit-learn failed"
    )
)
echo.

REM Speech Enhancement (optional but recommended)
if "%NO_SPEECH_ENHANCEMENT%"=="0" (
    echo ============================================================
    echo   [Step 6/7] Installing speech enhancement packages
    echo ============================================================
    echo Note: These packages can be tricky. Failures here are non-fatal.
    echo       Speech enhancement improves transcription quality in noisy audio.
    echo.
    call :log "Installing speech enhancement packages..."

    REM Install ModelScope dependencies first (CRITICAL: datasets must be <4.0)
    call :log "Installing ModelScope dependencies..."
    python -m pip install addict simplejson sortedcontainers packaging 2>nul
    python -m pip install "datasets>=2.14.0,<4.0" 2>nul
    if errorlevel 1 (
        echo WARNING: datasets installation failed - modelscope may not work
        call :log "WARNING: datasets installation failed"
    )

    REM Install ModelScope (ZipEnhancer SOTA speech enhancement)
    echo Installing ModelScope ^(ZipEnhancer^)...
    call :log "Installing ModelScope..."
    python -m pip install "modelscope>=1.20" 2>nul
    if errorlevel 1 (
        echo WARNING: modelscope installation failed ^(optional^)
        call :log "WARNING: modelscope installation failed"
    )

    REM Install ClearVoice from NumPy 2.x compatible fork
    echo Installing ClearVoice ^(48kHz denoising^)...
    call :log "Installing ClearVoice..."
    python -m pip install "git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice" 2>nul
    if errorlevel 1 (
        echo WARNING: clearvoice installation failed ^(optional^)
        call :log "WARNING: clearvoice installation failed"
    )

    REM Install BS-RoFormer (vocal isolation)
    echo Installing BS-RoFormer ^(vocal isolation^)...
    call :log "Installing BS-RoFormer..."
    python -m pip install bs-roformer-infer 2>nul
    if errorlevel 1 (
        echo WARNING: bs-roformer-infer installation failed ^(optional^)
        call :log "WARNING: bs-roformer-infer installation failed"
    )

    REM ONNX Runtime for ZipEnhancer ONNX mode
    echo Installing ONNX Runtime...
    python -m pip install "onnxruntime>=1.16.0" 2>nul
    if errorlevel 1 (
        call :log "Note: onnxruntime installation failed (optional)"
    )
    echo.
) else (
    echo ============================================================
    echo   [Step 6/7] Skipping speech enhancement ^(--no-speech-enhancement^)
    echo ============================================================
    call :log "Skipping speech enhancement (--no-speech-enhancement)"
    echo.
)

REM GUI dependencies (Windows-specific)
call :log "Installing GUI dependencies..."
echo Installing GUI dependencies...
python -m pip install "pywebview>=5.0.0" 2>nul
python -m pip install "pythonnet>=3.0" "pywin32>=305" 2>nul
echo.

REM ============================================================
REM Phase 7: WhisperJAV Application
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Phase 7: WhisperJAV Application"
call :log "============================================================"

echo ============================================================
echo   [Step 7/7] Installing WhisperJAV
echo ============================================================
if "%DEV_MODE%"=="1" (
    call :log "Installing WhisperJAV in development mode..."
    call :run_pip_with_retry "install --no-deps -e ." "Install WhisperJAV (editable)"
) else (
    call :log "Installing WhisperJAV from GitHub (latest)..."
    call :run_pip_with_retry "install --no-deps git+https://github.com/meizhong986/whisperjav.git" "Install WhisperJAV"
)
if errorlevel 1 (
    call :create_failure_file "Failed to install WhisperJAV"
    exit /b 1
)
echo.

REM ============================================================
REM Verification Phase: Check Critical Dependencies
REM ============================================================
call :log ""
call :log "============================================================"
call :log "  Verifying Installation"
call :log "============================================================"

echo ============================================================
echo   Verifying Installation
echo ============================================================

REM Verify WhisperJAV
python -c "import whisperjav; print(f'WhisperJAV {whisperjav.__version__} installed successfully!')"
if errorlevel 1 (
    echo WARNING: Could not verify WhisperJAV installation
    call :log "WARNING: WhisperJAV verification failed"
) else (
    call :log "WhisperJAV verified successfully"
)

REM Verify PyTorch CUDA status
python -c "import torch; cuda_status = 'ENABLED' if torch.cuda.is_available() else 'DISABLED'; print(f'CUDA acceleration: {cuda_status}')" 2>nul

REM Verify critical packages
echo.
echo Verifying critical packages:
call :log "Verifying critical packages..."

python -c "import numpy; print(f'  numpy: {numpy.__version__}')" 2>nul || echo   numpy: FAILED
python -c "import scipy; print(f'  scipy: {scipy.__version__}')" 2>nul || echo   scipy: FAILED
python -c "import librosa; print(f'  librosa: {librosa.__version__}')" 2>nul || echo   librosa: FAILED
python -c "import faster_whisper; print(f'  faster-whisper: {faster_whisper.__version__}')" 2>nul || echo   faster-whisper: FAILED
python -c "import transformers; print(f'  transformers: {transformers.__version__}')" 2>nul || echo   transformers: FAILED

if "%NO_SPEECH_ENHANCEMENT%"=="0" (
    echo.
    echo Speech enhancement packages:
    python -c "import modelscope; print(f'  modelscope: {modelscope.__version__}')" 2>nul || echo   modelscope: NOT INSTALLED
    python -c "import clearvoice; print('  clearvoice: installed')" 2>nul || echo   clearvoice: NOT INSTALLED
)

echo.

REM Summary
call :log ""
call :log "============================================================"
call :log "  Installation Complete!"
call :log "============================================================"

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
if "%LOCAL_LLM%"=="1" (
    echo   Local LLM: Installed
) else (
    echo   Local LLM: Not installed
)
echo.
echo   To run WhisperJAV:
echo     whisperjav video.mp4 --mode balanced
echo.
echo   To run with GUI:
echo     whisperjav-gui
echo.
if "%LOCAL_LLM%"=="1" (
    echo   To translate with local LLM:
    echo     whisperjav video.mp4 --translate --translate-provider local
    echo.
) else (
    echo   To enable local LLM translation, re-install with:
    echo     installer\install_windows.bat --local-llm          ^(fast - prebuilt wheel^)
    echo     installer\install_windows.bat --local-llm-build    ^(slow - builds if needed^)
    echo.
)
echo   For help:
echo     whisperjav --help
echo.
echo   Log file: %INSTALL_LOG%
echo.
echo   If you encounter issues with speech enhancement, re-run with:
echo     installer\install_windows.bat --no-speech-enhancement
echo.

call :log "Installation completed successfully at %DATE% %TIME%"
exit /b 0

REM ============================================================
REM Helper Functions
REM ============================================================

:log
REM Log message to file with timestamp
echo [%DATE% %TIME%] %~1 >> "%INSTALL_LOG%"
goto :eof

:run_pip_with_retry
REM Run pip command with up to 3 retries
REM Enhanced with Git timeout detection for GFW/VPN users (Issue #111)
REM Usage: call :run_pip_with_retry "pip args" "description"
set "PIP_ARGS=%~1"
set "PIP_DESC=%~2"
set "RETRY_COUNT=0"
set "PIP_OUTPUT_FILE=%TEMP%\pip_output_%RANDOM%.txt"

:retry_loop
set /a RETRY_COUNT+=1
call :log "Attempt %RETRY_COUNT%/3: %PIP_DESC%"
echo   [%RETRY_COUNT%/3] %PIP_DESC%...

REM Run pip and capture output for timeout detection
python -m pip %PIP_ARGS% > "%PIP_OUTPUT_FILE%" 2>&1
type "%PIP_OUTPUT_FILE%"
set "PIP_RESULT=!ERRORLEVEL!"

if !PIP_RESULT! NEQ 0 (
    REM Check if this looks like a Git timeout error (common with GFW/VPN)
    REM Pattern: "Failed to connect to github.com port 443 after 21" or similar
    findstr /C:"after 21" /C:"Connection timed out" /C:"Could not connect" "%PIP_OUTPUT_FILE%" >nul 2>&1
    if !ERRORLEVEL! EQU 0 (
        if "!GIT_TIMEOUTS_CONFIGURED!"=="" (
            echo.
            echo   ============================================================
            echo   Git connection timeout detected!
            echo   This is common for users behind GFW or slow VPN connections.
            echo   Configuring Git with extended timeouts...
            echo   ============================================================
            call :log "Git timeout detected - configuring extended timeouts"
            call :configure_git_timeouts
        )
    )
    
    if %RETRY_COUNT% LSS 3 (
        echo   Retrying in 5 seconds...
        call :log "Failed, retrying in 5 seconds..."
        timeout /t 5 /nobreak >nul
        del "%PIP_OUTPUT_FILE%" 2>nul
        goto :retry_loop
    ) else (
        echo   ERROR: %PIP_DESC% failed after 3 attempts
        call :log "ERROR: %PIP_DESC% failed after 3 attempts"
        del "%PIP_OUTPUT_FILE%" 2>nul
        exit /b 1
    )
) else (
    call :log "SUCCESS: %PIP_DESC%"
)
del "%PIP_OUTPUT_FILE%" 2>nul
exit /b 0

:configure_git_timeouts
REM Configure Git with extended timeouts for slow/unstable connections (Issue #111)
REM This helps users behind GFW or using slow VPN connections
call :log "Configuring Git timeouts for slow connections..."
echo   Applying Git timeout configuration:
git config --global http.connectTimeout 120 2>nul && echo     + http.connectTimeout = 120 seconds
git config --global http.timeout 300 2>nul && echo     + http.timeout = 300 seconds
git config --global http.lowSpeedLimit 0 2>nul && echo     + http.lowSpeedLimit = 0 (disabled)
git config --global http.lowSpeedTime 999999 2>nul && echo     + http.lowSpeedTime = 999999
git config --global http.postBuffer 524288000 2>nul && echo     + http.postBuffer = 500MB
git config --global http.maxRetries 5 2>nul && echo     + http.maxRetries = 5
REM Set environment variables for current session
set "GIT_HTTP_CONNECT_TIMEOUT=120"
set "GIT_HTTP_TIMEOUT=300"
set "GIT_TIMEOUTS_CONFIGURED=1"
call :log "Git timeout configuration complete"
echo.
goto :eof

:create_failure_file
REM Create a failure marker file with error details
set "FAILURE_FILE=%~dp0INSTALLATION_FAILED.txt"
echo WhisperJAV Installation Failed > "%FAILURE_FILE%"
echo ================================ >> "%FAILURE_FILE%"
echo. >> "%FAILURE_FILE%"
echo Error: %~1 >> "%FAILURE_FILE%"
echo Time: %DATE% %TIME% >> "%FAILURE_FILE%"
echo. >> "%FAILURE_FILE%"
echo Troubleshooting: >> "%FAILURE_FILE%"
echo - Check install_log_windows.txt for details >> "%FAILURE_FILE%"
echo - Ensure Python 3.10-3.12, FFmpeg, and Git are in PATH >> "%FAILURE_FILE%"
echo - Try: pip cache purge ^&^& pip install --upgrade pip >> "%FAILURE_FILE%"
echo - For network issues, check firewall/proxy settings >> "%FAILURE_FILE%"
echo. >> "%FAILURE_FILE%"
echo Support: https://github.com/meizhong986/WhisperJAV/issues >> "%FAILURE_FILE%"
call :log "ERROR: Created failure file - %~1"
goto :eof

:install_failed
call :create_failure_file "Core dependencies installation failed"
exit /b 1

:show_help
echo.
echo WhisperJAV Windows Installation Script
echo.
echo Usage: install_windows.bat [options]
echo.
echo Options:
echo   --cpu-only              Install CPU-only PyTorch ^(no CUDA^)
echo   --cuda118               Install PyTorch for CUDA 11.8
echo   --cuda121               Install PyTorch for CUDA 12.1
echo   --cuda124               Install PyTorch for CUDA 12.4
echo   --cuda126               Install PyTorch for CUDA 12.6
echo   --cuda128               Install PyTorch for CUDA 12.8 ^(default for driver 570+^)
echo   --no-speech-enhancement Skip speech enhancement packages
echo   --minimal               Minimal install ^(transcription only^)
echo   --dev                   Install in development/editable mode
echo   --local-llm             Install local LLM ^(fast - prebuilt wheel only^)
echo   --local-llm-build       Install local LLM ^(slow - builds from source if needed^)
echo   --help, -h              Show this help message
echo.
echo Examples:
echo   install_windows.bat                    # Standard install with auto CUDA detection
echo   install_windows.bat --cpu-only         # CPU-only install
echo   install_windows.bat --cuda128          # Force CUDA 12.8
echo   install_windows.bat --minimal --dev    # Minimal dev install
echo   install_windows.bat --local-llm        # Include local LLM ^(prebuilt wheel^)
echo   install_windows.bat --local-llm-build  # Include local LLM ^(build if no wheel^)
echo.
echo The script will auto-detect your GPU and select the appropriate
echo CUDA version ^(12.8 for driver 570+, 12.6 for 560+, etc.^).
echo Use --cuda*** flags to override.
echo.
exit /b 0
