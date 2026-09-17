#!/usr/bin/env python3
"""Pre-flight environment validation for WhisperJAV.

This module ensures the runtime environment meets all requirements,
with special focus on CUDA availability and compatibility.
"""

import sys
import os
import io
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import after UTF-8 fix to avoid encoding issues
from whisperjav.utils.device_detector import get_best_device, is_gpu_available

# Fix stdout/stderr encoding for Windows before any print statements
# This is critical for unicode characters (⚠️, ✓, •, etc.) in console output
def _ensure_utf8_output():
    """Ensure stdout and stderr use UTF-8 encoding to handle unicode characters."""
    if sys.stdout is not None and (not hasattr(sys.stdout, 'encoding') or sys.stdout.encoding.lower() != 'utf-8'):
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else io.BufferedWriter(io.FileIO(1, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass  # Silently fail if wrapping not possible

    if sys.stderr is not None and (not hasattr(sys.stderr, 'encoding') or sys.stderr.encoding.lower() != 'utf-8'):
        try:
            sys.stderr = io.TextIOWrapper(
                sys.stderr.buffer if hasattr(sys.stderr, 'buffer') else io.BufferedWriter(io.FileIO(2, 'w')),
                encoding='utf-8',
                errors='replace',
                line_buffering=True
            )
        except (AttributeError, OSError):
            pass  # Silently fail if wrapping not possible

# Apply UTF-8 fix immediately
_ensure_utf8_output()

# Use colorama for cross-platform colored output if available
try:
    from colorama import init, Fore, Style
    init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    # Fallback color definitions
    class Fore:
        RED = GREEN = YELLOW = CYAN = RESET = ''
    class Style:
        BRIGHT = RESET_ALL = ''


class CheckStatus(Enum):
    """Status of a pre-flight check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    INFO = "INFO"


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    status: CheckStatus
    message: str
    details: Optional[List[str]] = None
    fatal: bool = False


class PreflightChecker:
    """Comprehensive environment checker for WhisperJAV."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[CheckResult] = []
        
    def run_all_checks(self) -> bool:
        """Run all pre-flight checks and return overall status."""
        print(f"\n{Fore.CYAN}WhisperJAV Pre-flight Environment Check{Style.RESET_ALL}")
        print("=" * 60)
        
        # Run checks in order of importance
        self._check_python_version()
        self._check_cuda_availability()
        self._check_pytorch_cuda()
        self._check_gpu_memory()
        self._check_ffmpeg()
        self._check_disk_space()
        self._check_dependencies()
        self._check_downloaded_segmenter_models()
        
        # Display results
        self._display_results()
        
        # Return False if any fatal check failed
        return not any(r.fatal and r.status == CheckStatus.FAIL for r in self.results)
    
    def _check_python_version(self):
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version >= (3, 10) and version < (3, 13):
            self.results.append(CheckResult(
                name="Python Version",
                status=CheckStatus.PASS,
                message=f"Python {version_str} is supported"
            ))
        else:
            self.results.append(CheckResult(
                name="Python Version",
                status=CheckStatus.FAIL,
                message=f"Python {version_str} is not supported",
                details=["Supported versions: 3.10, 3.11, 3.12 (3.13+ breaks openai-whisper)"],
                fatal=True
            ))
    
    def _check_cuda_availability(self):
        """Check for CUDA availability - this is mandatory."""
        try:
            import torch

            if torch.cuda.is_available():
                # These calls can throw RuntimeError if CUDA driver is incompatible
                try:
                    device_count = torch.cuda.device_count()
                    device_name = torch.cuda.get_device_name(0)
                    cuda_version = torch.version.cuda

                    # #411: a card this build has no kernels for is not a usable GPU.
                    from whisperjav.utils.device_detector import cuda_build_supports_device
                    try:
                        capability = tuple(torch.cuda.get_device_capability(0))
                        arch_list = list(torch.cuda.get_arch_list())
                    except Exception:  # noqa: BLE001 - cannot judge; report availability only
                        capability, arch_list = None, []
                    if capability and not cuda_build_supports_device(capability, arch_list):
                        self.results.append(CheckResult(
                            name="CUDA Availability",
                            status=CheckStatus.FAIL,
                            message="GPU present but not supported by this PyTorch build",
                            details=[
                                f"Primary GPU: {device_name} (compute capability "
                                f"{capability[0]}.{capability[1]})",
                                f"This build has kernels for: {', '.join(arch_list)}",
                                "",
                                "Solutions:",
                                "1. Install a PyTorch build with kernels for this card",
                                "   (see https://pytorch.org/get-started/locally/)",
                                "2. Or use --accept-cpu-mode to run in CPU mode (slower)",
                            ],
                            fatal=True
                        ))
                        return

                    self.results.append(CheckResult(
                        name="CUDA Availability",
                        status=CheckStatus.PASS,
                        message=f"CUDA {cuda_version} available with {device_count} GPU(s)",
                        details=[f"Primary GPU: {device_name}"]
                    ))
                except RuntimeError as e:
                    error_msg = str(e)
                    # Handle CUDA driver version mismatch
                    if "driver version is insufficient" in error_msg.lower():
                        self.results.append(CheckResult(
                            name="CUDA Availability",
                            status=CheckStatus.FAIL,
                            message="CUDA driver version is too old",
                            details=[
                                "Your NVIDIA driver is too old for the installed PyTorch CUDA version.",
                                "",
                                "Solutions:",
                                "1. Update NVIDIA drivers from: https://www.nvidia.com/drivers",
                                "2. Or reinstall PyTorch with CPU-only version:",
                                "   pip uninstall torch torchvision torchaudio",
                                "   pip install torch torchvision torchaudio",
                                "3. Or use --accept-cpu-mode to run in CPU mode (slower)",
                                "",
                                f"Error: {error_msg}",
                            ],
                            fatal=True
                        ))
                    else:
                        self.results.append(CheckResult(
                            name="CUDA Availability",
                            status=CheckStatus.FAIL,
                            message="CUDA initialization failed",
                            details=[f"Error: {error_msg}"],
                            fatal=True
                        ))
            else:
                # This is a fatal error for WhisperJAV
                self.results.append(CheckResult(
                    name="CUDA Availability",
                    status=CheckStatus.FAIL,
                    message="No CUDA-capable GPU detected",
                    details=[
                        "WhisperJAV requires an NVIDIA GPU with CUDA support and cuda enabled torch.",
                        "",
                        "Possible solutions:",
                        "1. Ensure you have an CUDA version above 11.8 and CUDNN. ",
                        "2. Ensure you have CUDA enabled torch and torchaudio installed. ",
                        "3. Verify your torch is not CPU version. ",
                        "",
                    ],
                    fatal=True
                ))
        except ImportError:
            self.results.append(CheckResult(
                name="CUDA Availability",
                status=CheckStatus.FAIL,
                message="CUDA enabled PyTorch not installed",
                details=["Please complete installation first"],
                fatal=True
            ))
    
    def _check_pytorch_cuda(self):
        """Check PyTorch CUDA configuration."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                return  # Already reported in CUDA check
            
            # Check if PyTorch was built with CUDA
            if hasattr(torch, '_C') and hasattr(torch._C, '_cuda_getCompiledVersion'):
                compiled_cuda = torch._C._cuda_getCompiledVersion()
                runtime_cuda = torch.version.cuda
                
                # Convert compiled_cuda integer to version string for comparison
                # compiled_cuda is an integer like 12090 representing CUDA 12.9.0
                # We extract major.minor (12.9) to match runtime_cuda format (string "12.9")
                # Note: Patch version is intentionally truncated as runtime_cuda doesn't include it
                if isinstance(compiled_cuda, int):
                    major = compiled_cuda // 1000
                    minor = (compiled_cuda % 1000) // 10
                    compiled_cuda_str = f"{major}.{minor}"
                else:
                    compiled_cuda_str = str(compiled_cuda)
                
                if compiled_cuda_str == runtime_cuda:
                    self.results.append(CheckResult(
                        name="PyTorch CUDA Build",
                        status=CheckStatus.PASS,
                        message=f"PyTorch compiled for CUDA {runtime_cuda}"
                    ))
                else:
                    self.results.append(CheckResult(
                        name="PyTorch CUDA Build",
                        status=CheckStatus.WARN,
                        message="CUDA version mismatch",
                        details=[
                            f"PyTorch compiled for: CUDA {compiled_cuda_str}",
                            f"Runtime CUDA version: {runtime_cuda}",
                            "This may cause compatibility issues"
                        ]
                    ))
            
            # Test basic CUDA operations
            try:
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                self.results.append(CheckResult(
                    name="CUDA Operations",
                    status=CheckStatus.PASS,
                    message="Basic CUDA operations working"
                ))
            except Exception as e:
                self.results.append(CheckResult(
                    name="CUDA Operations",
                    status=CheckStatus.FAIL,
                    message="CUDA operations failed",
                    details=[str(e)],
                    fatal=True
                ))
                
        except ImportError:
            pass  # Already handled above
    
    def _check_gpu_memory(self):
        """Check available GPU memory."""
        try:
            import torch
            
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.get_device_properties(0).total_memory
                gpu_mem_gb = gpu_mem / (1024**3)
                
                if gpu_mem_gb >= 8:
                    status = CheckStatus.PASS
                    message = f"GPU memory: {gpu_mem_gb:.1f} GB (Excellent)"
                elif gpu_mem_gb >= 6:
                    status = CheckStatus.PASS
                    message = f"GPU memory: {gpu_mem_gb:.1f} GB (Good)"
                elif gpu_mem_gb >= 4:
                    status = CheckStatus.WARN
                    message = f"GPU memory: {gpu_mem_gb:.1f} GB (Minimum)"
                else:
                    status = CheckStatus.WARN
                    message = f"GPU memory: {gpu_mem_gb:.1f} GB (Low)"
                
                details = []
                if gpu_mem_gb < 6:
                    details.append("Consider using --mode faster for better performance")
                    details.append("Large videos may require chunk processing")
                
                self.results.append(CheckResult(
                    name="GPU Memory",
                    status=status,
                    message=message,
                    details=details if details else None
                ))
        except:
            pass  # Skip if CUDA not available
    
    def _check_ffmpeg(self):
        """Check ffmpeg availability."""
        ffmpeg_path = shutil.which('ffmpeg')
        
        if ffmpeg_path:
            # Try to get version
            try:
                import subprocess
                result = subprocess.run(
                    ['ffmpeg', '-version'],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=5
                )
                version_line = result.stdout.split('\n')[0]
                
                self.results.append(CheckResult(
                    name="FFmpeg",
                    status=CheckStatus.PASS,
                    message="FFmpeg is installed",
                    details=[version_line] if self.verbose else None
                ))
            except:
                self.results.append(CheckResult(
                    name="FFmpeg",
                    status=CheckStatus.PASS,
                    message="FFmpeg is installed"
                ))
        else:
            self.results.append(CheckResult(
                name="FFmpeg",
                status=CheckStatus.FAIL,
                message="FFmpeg not found in PATH",
                details=[
                    "FFmpeg is required for audio/video processing",
                    "",
                    "Installation instructions:",
                    "- Windows: Download from https://ffmpeg.org/download.html",
                    "- macOS: brew install ffmpeg",
                    "- Linux: sudo apt install ffmpeg (or equivalent)"
                ],
                fatal=True
            ))
    
    def _check_disk_space(self):
        """Check available disk space."""
        try:
            path = Path.cwd()
            stat = os.statvfs(path)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
            
            if free_gb >= 50:
                status = CheckStatus.PASS
                message = f"Free disk space: {free_gb:.1f} GB"
            elif free_gb >= 20:
                status = CheckStatus.WARN
                message = f"Free disk space: {free_gb:.1f} GB (Low)"
                details = ["Recommend at least 50 GB for processing large videos"]
            else:
                status = CheckStatus.FAIL
                message = f"Free disk space: {free_gb:.1f} GB (Critical)"
                details = ["Insufficient space for video processing"]
                
            self.results.append(CheckResult(
                name="Disk Space",
                status=status,
                message=message,
                details=details if status != CheckStatus.PASS else None,
                fatal=(status == CheckStatus.FAIL)
            ))
        except:
            # Skip on error (e.g., Windows without os.statvfs)
            pass
    
    def _check_dependencies(self):
        """Check critical Python dependencies."""
        critical_deps = [
            'whisper',
            'faster_whisper',
            'torch',
            'torchaudio',
            'numpy',
            'ffmpeg'
        ]

        optional_deps = {
            'stable_whisper': "Required only for legacy fast/faster pipelines",
            # v1.9.2: FireRedVAD ships in the [cli] extra. Since 2026-09-12 it is
            # also the DEFAULT speech segmenter for the fidelity pipeline, so a
            # plain `--mode fidelity` needs it -- not just an explicit
            # --speech-segmenter firered-vad.
            'fireredvad': "FireRedVAD speech segmenter -- the default for --mode "
                          "fidelity and for a fidelity ensemble pass, and selectable "
                          "anywhere with --speech-segmenter firered-vad; "
                          "pip install fireredvad",
        }
        
        missing = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if missing:
            self.results.append(CheckResult(
                name="Python Dependencies",
                status=CheckStatus.FAIL,
                message=f"Missing {len(missing)} critical dependencies",
                details=[f"Missing: {', '.join(missing)}"],
                fatal=True
            ))
        else:
            self.results.append(CheckResult(
                name="Python Dependencies",
                status=CheckStatus.PASS,
                message="All critical dependencies installed"
            ))

        optional_missing_details = []
        for dep, description in optional_deps.items():
            try:
                __import__(dep)
            except ImportError:
                optional_missing_details.append(f"{dep}: {description}")

        if optional_missing_details:
            self.results.append(CheckResult(
                name="Optional Components",
                status=CheckStatus.WARN,
                message="Optional dependencies are missing",
                details=optional_missing_details,
                fatal=False
            ))
    
    def _check_downloaded_segmenter_models(self):
        """Are the speech-segmenter models that are fetched at runtime present?

        FireRedVAD is the fidelity pipeline's speech segmenter since 2026-09-12 and
        its model is downloaded on first use. A WARN, not a FAIL: a user who only
        runs Balanced never needs it, and --check has no pipeline to go on.
        """
        for backend, (_m, _f, display) in _RUNTIME_DOWNLOADED_SEGMENTERS.items():
            # download=False: --check reports, it does not change the machine.
            ok = ensure_segmenter_model_available(
                backend, download=False, exit_on_fail=False)
            if ok:
                self.results.append(CheckResult(
                    name=f"{display} model",
                    status=CheckStatus.PASS,
                    message="Downloaded and ready",
                ))
            else:
                self.results.append(CheckResult(
                    name=f"{display} model",
                    status=CheckStatus.WARN,
                    message="Not downloaded yet",
                    details=[
                        f"{display} finds the speech for the fidelity pipeline. Its "
                        "model is downloaded once from Hugging Face, and it is not "
                        "on this machine yet. --check does not download it.",
                        "It is fetched during installation, or at the start of the "
                        "first fidelity run. To avoid needing the network then, run "
                        "one fidelity job while online, or use "
                        "--speech-segmenter silero-v3.1, which needs no download.",
                    ],
                    fatal=False,
                ))

    def _display_results(self):
        """Display all check results in a formatted manner."""
        print()
        
        # Group results by status
        failures = [r for r in self.results if r.status == CheckStatus.FAIL]
        warnings = [r for r in self.results if r.status == CheckStatus.WARN]
        passes = [r for r in self.results if r.status == CheckStatus.PASS]
        
        # Display each group
        for result in passes:
            self._display_result(result, Fore.GREEN)
        
        for result in warnings:
            self._display_result(result, Fore.YELLOW)
            
        for result in failures:
            self._display_result(result, Fore.RED)
        
        # Summary
        print("\n" + "=" * 60)
        if failures:
            fatal_count = sum(1 for r in failures if r.fatal)
            print(f"{Fore.RED}✗ {len(failures)} check(s) failed ({fatal_count} fatal){Style.RESET_ALL}")
            if fatal_count > 0:
                print(f"{Fore.RED}WhisperJAV cannot run until these issues are resolved.{Style.RESET_ALL}")
        elif warnings:
            print(f"{Fore.YELLOW}⚠ All checks passed with {len(warnings)} warning(s){Style.RESET_ALL}")
            print("WhisperJAV can run but may have reduced performance.")
        else:
            print(f"{Fore.GREEN}✓ All checks passed!{Style.RESET_ALL}")
            print("Your environment is ready for WhisperJAV.")
        print("=" * 60 + "\n")
    
    def _display_result(self, result: CheckResult, color: str):
        """Display a single check result."""
        status_symbol = {
            CheckStatus.PASS: "✓",
            CheckStatus.FAIL: "✗",
            CheckStatus.WARN: "⚠",
            CheckStatus.INFO: "ℹ"
        }[result.status]
        
        print(f"{color}{status_symbol} {result.name}: {result.message}{Style.RESET_ALL}")
        
        if result.details and (self.verbose or result.status in [CheckStatus.FAIL, CheckStatus.WARN]):
            for detail in result.details:
                if detail:  # Skip empty lines
                    print(f"  {detail}")
                else:
                    print()


# Speech segmenters whose model is downloaded on first use instead of shipping in
# the wheel. Name → the callable that fetches it (raising on failure) and the
# human name used in the message.
#
# WHY THIS EXISTS AT START-UP. The fidelity pipeline catches a segmenter failure
# PER SCENE and carries on (pipelines/fidelity_pipeline.py). A model that cannot be
# fetched therefore fails every scene, produces an empty subtitle file, and the run
# summary calls the file "empty" -- which the default --fail-on does not fail on.
# The user is handed an empty .srt by a run that exited 0. Fetching here turns that
# into one message before any audio is read. (Owner, 2026-09-12, his option 3.)
_RUNTIME_DOWNLOADED_SEGMENTERS = {
    "firered-vad": (
        "whisperjav.modules.speech_segmentation.backends.firered_vad",
        "ensure_model_downloaded",
        "FireRedVAD",
    ),
}


def ensure_segmenter_model_available(backend, *, model_dir=None,
                                     download: bool = True,
                                     exit_on_fail: bool = True) -> bool:
    """
    Make sure a speech segmenter's model is on this machine before a run starts.

    Returns True when there is nothing to do (the segmenter ships its model, or is
    not one we know) or when the fetch succeeded. On failure it prints what the user
    can do about it and, by default, ends the run with status 1 -- the same status
    the GPU start-up check uses when it cannot proceed. Pass exit_on_fail=False to
    get False back instead, which is what --check does.

    ``model_dir`` is the directory the run's segmenter config names, if any, so this
    check looks in the same place the segmenter will and cannot pass while the run
    then fails, or the reverse.

    ``download=False`` reports what is already on the machine without fetching
    anything -- what ``--check`` wants, since a diagnostic must not change the
    machine it is diagnosing.

    This is a better MESSAGE, earlier; it is not the guarantee. The guarantee is in
    ``FireRedVadSpeechSegmenter.__init__``, which resolves the model when the
    segmenter is actually built and so cannot be walked around by a path this check
    does not predict.
    """
    entry = _RUNTIME_DOWNLOADED_SEGMENTERS.get(backend or "")
    if entry is None:
        return True
    module_name, func_name, display = entry
    try:
        import importlib
        getattr(importlib.import_module(module_name), func_name)(
            model_dir, download=download)
        return True
    except Exception as exc:
        if not exit_on_fail:
            return False
        offline = bool(os.environ.get("HF_HUB_OFFLINE"))
        lines = [
            f"{display}'s speech-detection model is not on this machine",
            "",
            f"This run needs {display} to find the speech in your audio, and its",
            "model is downloaded once from Hugging Face. That did not work:",
            "",
            f"  {((str(exc).splitlines() or ['unknown error'])[0])[:BOX_WIDTH - 8]}",
            "",
            "Nothing has been transcribed. Without the model every scene would",
            "fail and you would be handed an empty subtitle file.",
            "",
            "What you can do:",
            "  - Connect to the internet and run this once. The download is",
            "    small (about 2 MB) and is kept for every run after it.",
            "  - Or pick a speech segmenter that needs no download:",
            "      --speech-segmenter silero-v3.1",
            "    In the Ensemble tab, set that pass's Speech Segmenter to",
            "    Silero v3.1.",
            "  - Or download the model on another machine and point at it:",
            "      huggingface-cli download FireRedTeam/FireRedVAD \\",
            "        --local-dir <folder>",
            "    In China, ModelScope serves the same files:",
            "      modelscope download --model xukaituo/FireRedVAD \\",
            "        --local_dir <folder>",
            "    Then set WHISPERJAV_FIREREDVAD_MODEL_DIR to <folder>.",
        ]
        if offline:
            lines += [
                "",
                "Downloads are switched off for this run (HF_HUB_OFFLINE is set,",
                "which is what --offline does), so nothing can be fetched now.",
                "The model has to have been downloaded once beforehand.",
            ]
        _print_box(lines, Fore.RED)
        sys.exit(1)



def ensure_speech_enhancer_available(backend, *, exit_on_fail: bool = True) -> bool:
    """
    Make sure an audio clean-up that must not fail quietly can actually run,
    before any audio is read.

    Most clean-up backends degrade to "no enhancement" with a warning, which is
    right for one that only helps a little. A backend listed in the factory's
    FATAL_WHEN_UNAVAILABLE was chosen because the audio needs it -- falling back
    would hand the user a poor subtitle file from a run that exited 0. Owner,
    2026-09-17, about htdemucs: "I think it should fail with good user
    communication."

    This is a better MESSAGE, earlier. The guarantee is in the backend itself and
    in pipeline_helper, which raise SpeechEnhancerUnavailable however the backend
    is reached, including by a path this check does not predict.

    Returns True when there is nothing to check or the backend is ready. On
    failure it prints what the user can do and, by default, ends the run with
    status 1 -- the same status the other start-up checks use. Pass
    exit_on_fail=False to get False back instead, which is what --check wants.
    """
    if not backend or backend == "none":
        return True

    # Names may arrive as "backend:detail" from --passN-speech-enhancer.
    backend = str(backend).split(":", 1)[0].strip()

    try:
        from whisperjav.modules.speech_enhancement.factory import (
            FATAL_WHEN_UNAVAILABLE,
            SpeechEnhancerFactory,
        )
    except Exception:
        return True  # Enhancement is not installed at all; nothing to promise.

    if backend not in FATAL_WHEN_UNAVAILABLE:
        return True

    available, hint = SpeechEnhancerFactory.is_backend_available(backend)
    if available:
        return True
    if not exit_on_fail:
        return False

    lines = [
        f"The {backend} audio clean-up is not installed",
        "",
        "This run was asked to isolate the voice from the music and effects",
        f"with {backend}, and the program that does it is not on this machine.",
        "",
        "Nothing has been transcribed. Going ahead without it would hand you",
        "subtitles made from the untouched audio -- which is the thing you",
        "asked to have cleaned up -- and the run would look successful.",
        "",
        "What you can do:",
        f"  - Install it, once, with:  {hint}",
        "    It is a normal install: nothing is compiled, and it brings a",
        "    model of about 84 MB the first time you run it.",
        "  - Or pick a clean-up that is already installed:",
        "      --speech-enhancer is chosen per pass, e.g.",
        "      --pass1-speech-enhancer bs-roformer  (isolates the voice too)",
        "      --pass1-speech-enhancer zipenhancer   (reduces noise)",
        "      --pass1-speech-enhancer ffmpeg-dsp    (levels and filters)",
        "    In the Ensemble tab, set that pass's Audio Clean-up to one of",
        "    those instead.",
        "  - Or leave the clean-up out and transcribe the audio as it is.",
        "",
        "Note: the model comes from Meta's own site, not from Hugging Face,",
        "so a Hugging Face mirror or --offline does not apply to it.",
    ]
    _print_box(lines, Fore.RED)
    sys.exit(1)

def run_preflight_checks(verbose: bool = False, exit_on_fail: bool = True) -> bool:
    """Run pre-flight checks and optionally exit on failure.
    
    Args:
        verbose: Show detailed information for all checks
        exit_on_fail: Exit the program if fatal checks fail
        
    Returns:
        True if all checks passed or only warnings, False if fatal errors
    """
    checker = PreflightChecker(verbose=verbose)
    success = checker.run_all_checks()
    
    if not success and exit_on_fail:
        sys.exit(1)
    
    return success

# Set once the user has answered "yes" (or passed consent on the command line), so the
# question is asked once per run: the check runs at import time and again in main(),
# and spawned worker processes re-run the module level of whisperjav.main.
CPU_ACCEPTED_ENV = "WHISPERJAV_CPU_ACCEPTED"


def cpu_consent_in_argv(argv) -> bool:
    """True when the command line already answers "use the CPU": --accept-cpu-mode,
    or an explicit --device cpu (either spelling)."""
    argv = list(argv)
    if "--accept-cpu-mode" in argv or "--device=cpu" in argv:
        return True
    for i, tok in enumerate(argv[:-1]):
        if tok == "--device" and argv[i + 1] == "cpu":
            return True
    return False


def _stdin_is_interactive() -> bool:
    """True when a person can answer a question on this console.

    The GUI marks its child processes with WHISPERJAV_NO_CONSOLE=1 because on
    Windows even the null device reports itself as a terminal, so isatty() alone
    would leave a GUI worker waiting for an answer nobody can give.
    """
    if os.environ.get("WHISPERJAV_NO_CONSOLE") == "1":
        return False
    try:
        return bool(sys.stdin) and sys.stdin.isatty()
    except Exception:  # noqa: BLE001 - a broken stdin means nobody can answer
        return False


def _ask(prompt: str):
    """Read one answer. None when there was nobody to answer (end of input on a
    piped stdin, which Windows reports as a terminal); an interrupt is 'no'."""
    try:
        return input(prompt).strip().lower()
    except EOFError:
        return None
    except KeyboardInterrupt:
        return ""


BOX_WIDTH = 72
CPU_QUESTION = "Continue on the CPU anyway? [y/N] "


def _print_box(lines, colour):
    """Print `lines` inside a double-ruled box so the question cannot be missed
    (owner, 2026-09-06: "printed out in a very very visible manner")."""
    inner = BOX_WIDTH - 2
    print(f"{colour}╔{'═' * inner}╗")
    print(f"║{' ' * inner}║")
    for line in lines:
        print(f"║  {line.ljust(inner - 2)}║")
    print(f"║{' ' * inner}║")
    print(f"╚{'═' * inner}╝{Style.RESET_ALL}")


def _ask_to_continue_on_cpu(colour) -> bool:
    """Stop and ask whether to proceed on the CPU or abort. Never decides alone,
    never continues after a timeout. Returns True only on an explicit yes; a
    "yes" is remembered for this run and its worker processes. Where nobody can
    answer (the GUI's child process, a piped run) it aborts and says how to
    answer in advance. Anything else exits with status 1."""
    if _stdin_is_interactive():
        _print_box([
            "YOUR ANSWER IS NEEDED  -  nothing has been processed yet.",
            "",
            "Continue on the CPU anyway?  (much slower than a GPU)",
            "",
            "   type  y  then Enter   ->  continue on the CPU",
            "   Enter, or  n          ->  abort",
        ], colour)
        answer = _ask(CPU_QUESTION)
        if answer in ("y", "yes"):
            os.environ[CPU_ACCEPTED_ENV] = "1"   # remembered for this run and its workers
            print(f"\n{Fore.GREEN}✓ Continuing on the CPU (you confirmed).{Style.RESET_ALL}\n")
            return True
        if answer is not None:
            print(f"\n{Fore.RED}Aborted. Nothing was processed.{Style.RESET_ALL}\n")
            sys.exit(1)
        print()
        _print_box([
            "NO ANSWER RECEIVED  -  input ended before you answered.",
            "Nothing was processed.",
            "",
            "To continue on the CPU, run again with  --accept-cpu-mode",
            "(in the GUI: tick 'Accept CPU-only mode', then Start again).",
        ], Fore.RED)
        print()
        sys.exit(1)
    _print_box([
        "STOPPED  -  no console to ask on.  Nothing was processed.",
        "",
        "To continue on the CPU, run again with  --accept-cpu-mode",
        "(in the GUI: tick 'Accept CPU-only mode', then Start again).",
    ], Fore.RED)
    print()
    sys.exit(1)


def enforce_gpu_requirement(accept_cpu_mode=False):
    """
    The start-up check every run passes through. A usable GPU (CUDA or MPS)
    passes silently. Otherwise the run STOPS and ASKS whether to proceed on the
    CPU or abort (owner, 2026-09-06), both when no GPU is present and when one
    is present but this PyTorch build has no kernels for it (#411).

    Args:
        accept_cpu_mode: True when the command line already answered
            (--accept-cpu-mode, --device cpu, or the GUI's "Accept CPU-only
            mode" box): no question is asked.

    Returns:
        bool: True when a GPU is usable or the user chose the CPU; otherwise
        the process exits with status 1 and nothing has been processed.
    """
    # Skip check entirely if user explicitly accepted CPU mode
    if accept_cpu_mode:
        if os.environ.get(CPU_ACCEPTED_ENV) != "1":
            print(f"{Fore.YELLOW}ℹ CPU mode accepted in advance (--accept-cpu-mode, --device cpu, or the "
                  f"GUI's 'Accept CPU-only mode' box); the GPU check is skipped.{Style.RESET_ALL}")
        os.environ[CPU_ACCEPTED_ENV] = "1"
        return True
    if os.environ.get(CPU_ACCEPTED_ENV) == "1":
        return True   # already answered in this run (or by the parent process)

    try:
        import torch

        # Check for any GPU (CUDA or MPS)
        best_device = get_best_device()
        if best_device in ('cuda', 'mps'):
            return True

        # #411 (owner, 2026-09-06): a GPU is present but this PyTorch build has no
        # kernels for it. The check stops and ASKS whether to proceed on the CPU or
        # abort; it never decides by itself and never continues after a timeout.
        # Where nobody can answer (the GUI's child process, a piped run) it aborts
        # and says how to answer: --accept-cpu-mode, or the GUI's "Accept CPU-only
        # mode" box, both handled at the top of this function.
        from whisperjav.utils import device_detector as _dd
        if _dd.CUDA_UNUSABLE_REASON:
            print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
            print(f"{Fore.RED}❌ GPU not supported by this PyTorch build{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}\n")
            print(f"  {_dd.CUDA_UNUSABLE_REASON}\n")
            print("Running on this card would end as an empty file reported as success.\n")
            print(f"{Fore.CYAN}Your options:{Style.RESET_ALL}")
            print("  - Install a PyTorch build with kernels for this card")
            print("    (see https://pytorch.org/get-started/locally/), or")
            print("  - Continue on the CPU instead (much slower; the ChronosJAV pipelines")
            print("    pick CUDA on their own and may still fail there).")
            print("\n  Run 'whisperjav --check' for detailed diagnostics\n")
            return _ask_to_continue_on_cpu(Fore.RED)

        # No GPU at all (no CUDA, no MPS): explain, then the same question
        # (owner, 2026-09-06: this path asks too; the 30 s auto-continue is gone).
        print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠  No GPU found{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")

        print("WhisperJAV works best with GPU acceleration.")
        print("We detected that no compatible GPU is currently available.\n")

        print(f"{Fore.CYAN}What this means:{Style.RESET_ALL}")
        print("  • CPU-only processing will be significantly slower (10-50x)")
        print("  • Large video files may take hours instead of minutes")
        print("  • You may encounter memory issues with longer videos\n")

        print(f"{Fore.CYAN}Supported GPU platforms:{Style.RESET_ALL}")
        print("  • NVIDIA GPUs (CUDA) - RTX 20/30/40/50 series, Blackwell, etc.")
        print("  • Apple Silicon (MPS) - M1/M2/M3/M4/M5 chips")
        print("  • AMD GPUs (ROCm) - Limited support, see documentation\n")

        print(f"{Fore.CYAN}To enable GPU acceleration:{Style.RESET_ALL}")

        current_platform = platform.system()
        if current_platform == "Darwin":
            print("  macOS detected:")
            print("  1. If you have Apple Silicon (M1/M2/M3/M4/M5):")
            print("     pip install --upgrade torch torchvision torchaudio")
            print("  2. If you have Intel Mac with AMD GPU, GPU acceleration not supported")
        elif current_platform == "Windows":
            print("  Windows detected:")
            print("  1. Ensure you have an NVIDIA GPU")
            print("  2. Install latest NVIDIA drivers from nvidia.com")
            print("  3. Reinstall PyTorch with CUDA:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        else:
            print("  Linux detected:")
            print("  1. For NVIDIA GPUs:")
            print("     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
            print("  2. For AMD GPUs (experimental):")
            print("     See https://pytorch.org/get-started/locally/ for ROCm installation")

        print("\n  Run 'whisperjav --check' for detailed diagnostics\n")
        return _ask_to_continue_on_cpu(Fore.YELLOW)

    except ImportError:
        # PyTorch not installed - this is a critical error
        print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.RED}❌ Critical Error: PyTorch Not Installed{Style.RESET_ALL}")
        print(f"{Fore.RED}{'='*70}{Style.RESET_ALL}\n")
        print("WhisperJAV requires PyTorch to function.")
        print("Please install it using:\n")
        print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
        print("\nOr for CPU-only (slower):")
        print("  pip install torch torchvision torchaudio")
        print(f"\n{Fore.RED}{'='*70}{Style.RESET_ALL}\n")
        input("Press Enter to exit...")
        sys.exit(1)
        
        
if __name__ == "__main__":
    # Run checks when module is executed directly
    import argparse
    parser = argparse.ArgumentParser(description="WhisperJAV environment checker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    run_preflight_checks(verbose=args.verbose)