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
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Import after UTF-8 fix to avoid encoding issues
from whisperjav.utils.device_detector import get_best_device, is_gpu_available
from whisperjav.installer.core import accelerators as _ACC  # standard library only

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


# v1.10.0 (item 2): texts the owner approved (W43, M1b) or that wait for his wording (DRAFT).
RELEASE_195_URL = "https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5"
MIN_COMPUTE_CAPABILITY_110 = (7, 5)
OLD_CARD_TEXT_110 = f"WhisperJAV 1.10 does not support this card; WhisperJAV 1.9.5 does: {RELEASE_195_URL}"
# DRAFT, not yet approved by the owner:
CUBLAS12_MISSING_TEXT = ("The CUDA 12 files that Balanced, Fast and Faster need on the GPU ({files}) "
                         "are missing from the ctranslate2 folder. Run the WhisperJAV installer again.")
# DRAFT, not yet approved by the owner (W43 (3) approved that the line exists):
CU128_STACK_TEXT = ("This PC runs the CUDA 12.8 build of WhisperJAV: when it was installed, the CUDA 12 "
                    "library could not be loaded next to CUDA 13.")

# 1.10.0, the CrispASR pipelines (route B design 12, T3 and T4; W193 5, W203 I2).
CRISPASR_NAME = "CrispASR"
CRISPASR_TEXT = "{version} ({build}), GGUF pipelines: {device}{why}"
CRISPASR_BUILD_NAMES = {"cuda": "CUDA build", "cpu": "CPU build", "vulkan": "Vulkan build"}
# T46 (design 22.10), from the accelerator table (W241 Q1).
CRISPASR_DEVICE_NAMES = _ACC.CHECK_CRISPASR_DEVICE
CRISPASR_NOT_INSTALLED_TEXT = ("not installed; the CrispASR pipelines cannot run here. Run the WhisperJAV "
                               "installer again.")
CRISPASR_NO_BUILD_TEXT = ("no CrispASR build for this computer ({platform}, {machine}); the CrispASR "
                          "pipelines cannot run here")
CRISPASR_NO_AVX2_TEXT = "this processor lacks AVX2, which CrispASR needs; the CrispASR pipelines cannot run here"
CRISPASR_DEVICE_PROBLEM = "{version} ({build}): {reason}"
CRISPASR_MODEL_NAME = "CrispASR model {model}"
CRISPASR_MODEL_TEXT = "{label}, for {pipeline}: {state}"
# What uses an aligner or timing-model file (item 12). Texts approved by the owner as drafted (W268);
# design note 2026-10-28 section 9, Q11.
CRISPASR_DEFAULT_ALIGNER_FOR = "ChronosJAV and Cohere-CrispASR (aligner)"
CRISPASR_ALIGNER_FOR = "ChronosJAV (aligner)"
CRISPASR_TIMING_FOR = "ChronosJAV (timing model)"
CRISPASR_DOWNLOADED = "downloaded"
CRISPASR_MODEL_NOT_DOWNLOADED = "not downloaded ({size})"
CRISPASR_CHECK_FAILED = "could not be checked: {error}"


def cublas12_files_missing() -> List[str]:
    """Placed cuBLAS 12 files missing on a CUDA 13.0 install ([] otherwise; spec C)."""
    try:
        from whisperjav.installer.cublas12 import missing_files, read_stack
        if read_stack() != "cu130":
            return []
        return missing_files()
    except Exception:
        return []


def _nvidia_smi_gpu_name() -> str:
    """The first NVIDIA GPU nvidia-smi lists, or "" (for --check's CPU-only warning, T49)."""
    import subprocess
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    except Exception:  # noqa: BLE001 - no nvidia-smi means no NVIDIA GPU to point at
        return ""
    lines = (r.stdout or "").strip().splitlines()
    return lines[0].strip() if r.returncode == 0 and lines else ""


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
        self._check_cublas12_for_faster_whisper()
        self._check_gpu_memory()
        self._check_ffmpeg()
        self._check_disk_space()
        self._check_dependencies()
        self._check_downloaded_segmenter_models()
        self._check_crispasr()

        # Display results
        self._display_results()
        
        # Return False if any fatal check failed
        return not any(r.fatal and r.status == CheckStatus.FAIL for r in self.results)
    
    def _check_cublas12_for_faster_whisper(self):
        """v1.10.0: the stack this PC was installed with, and its cuBLAS 12 files."""
        try:
            from whisperjav.installer.cublas12 import read_stack
            stack = read_stack()
        except Exception:
            return
        if stack == "cu128":
            self.results.append(CheckResult(
                name="CUDA build", status=CheckStatus.INFO, message=CU128_STACK_TEXT))
            return
        if stack != "cu130":
            return
        missing = cublas12_files_missing()
        if missing:
            self.results.append(CheckResult(
                name="cuBLAS 12 for faster-whisper", status=CheckStatus.FAIL,
                message=CUBLAS12_MISSING_TEXT.format(files=", ".join(missing))))
        else:
            self.results.append(CheckResult(
                name="cuBLAS 12 for faster-whisper", status=CheckStatus.PASS,
                message="cublas64_12.dll and cublasLt64_12.dll are in the ctranslate2 folder"))

    def _check_python_version(self):
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version[:2] == (3, 13):
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
                details=["Supported version: 3.13 (WhisperJAV 1.10)"],
                fatal=True
            ))
    
    def _check_cuda_availability(self):
        """The Accelerators group of --check (design 22.5.5; owner W232 2, W225).

        First the accelerator this install is made for, then one line per engine: PyTorch,
        CTranslate2, ONNX Runtime (CrispASR has its own line). A FAIL -- and exit status 1 --
        only where the install does not fit the computer: an NVIDIA install whose GPU cannot be
        used. Every other install gets INFO lines; an Apple GPU is a PASS.
        """
        from whisperjav.installer.core import accelerators
        has_record, value = accelerators.read_record()
        acc = accelerators.get(value)
        made_for = acc.name if acc else ("not recorded" if not has_record else "not known to this release")
        self.results.append(CheckResult(name="Accelerator", status=CheckStatus.INFO,
                                        message=f"Accelerator of this install: {made_for}."))
        try:
            import torch
        except ImportError:
            self.results.append(CheckResult(name="PyTorch", status=CheckStatus.FAIL,
                                            message="PyTorch is not installed", fatal=True))
            return
        version = torch.__version__
        best = get_best_device()
        own = " (your own)" if value == "own-pytorch" else ""
        if best == "cuda":
            card = torch.cuda.get_device_name(0)
            if getattr(torch.version, "hip", None):
                self.results.append(CheckResult(name="PyTorch", status=CheckStatus.INFO,
                                                message=f"PyTorch {version}{own}: AMD GPU (ROCm), {card} — not tested"))
            else:
                self.results.append(CheckResult(
                    name="PyTorch", status=CheckStatus.PASS,
                    message=f"PyTorch {version}{own}: NVIDIA GPU (CUDA {torch.version.cuda}), {card}"))
        elif best == "mps":
            self.results.append(CheckResult(
                name="PyTorch", status=CheckStatus.PASS,
                message=f"PyTorch {version}{own}: Apple GPU (Metal, through MPS) — not yet tested on a Mac"))
        elif acc is not None and acc.nvidia:
            reason, fix = _nvidia_reason(torch)
            if fix == FIX_DRIVER:
                details = ["1. Update the NVIDIA driver (580 or newer) from https://www.nvidia.com/drivers",
                           "2. Or run the WhisperJAV installer again and choose another accelerator",
                           "3. Or add --device cpu to run on the CPU (not accelerated)"]
            else:
                details = ["1. Run the WhisperJAV installer again",
                           "2. Or add --device cpu to run on the CPU (not accelerated)"]
            self.results.append(CheckResult(
                name="PyTorch", status=CheckStatus.FAIL,
                message=f"PyTorch {version}: the NVIDIA GPU cannot be used: {reason}",
                details=details, fatal=True))
        else:
            self.results.append(CheckResult(name="PyTorch", status=CheckStatus.INFO,
                                            message=f"PyTorch {version}{own}: CPU (not accelerated)"))

        # The run's own rule (device_detector.ctranslate2_device): the GPU only when PyTorch's GPU is
        # NVIDIA's and CTranslate2 sees it too. CTranslate2's Windows wheel sees an NVIDIA card even on
        # a CPU-only install, where the run uses the CPU.
        ct2 = ("NVIDIA GPU (CUDA)" if best == "cuda" and not getattr(torch.version, "hip", None)
               and _ctranslate2_sees_cuda() else "CPU (not accelerated)")
        self.results.append(CheckResult(name="CTranslate2", status=CheckStatus.INFO,
                                        message=f"CTranslate2 (balanced, fast, faster): {ct2}"))
        self.results.append(CheckResult(
            name="ONNX Runtime", status=CheckStatus.INFO,
            message="ONNX Runtime (WhisperSeg speech segmenter): CPU (not accelerated) in every install"))

        if value == "cpu":
            gpu = _nvidia_smi_gpu_name()
            if gpu:
                self.results.append(CheckResult(
                    name="Accelerator", status=CheckStatus.WARN,
                    message=(f"This PC has a supported NVIDIA GPU ({gpu}), but this install is made for the "
                             "CPU only. Run the WhisperJAV installer again to use the GPU (for a source "
                             "install: python install.py --accelerator nvidia).")))

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
                # Exit status 1 only where an NVIDIA install does not fit (W232 2): an Own PyTorch
                # (for example ROCm) or other install gets a warning (review P7 finding 15).
                from whisperjav.installer.core import accelerators
                acc = accelerators.get(accelerators.read_record()[1])
                nvidia_install = acc is not None and acc.nvidia
                self.results.append(CheckResult(
                    name="CUDA Operations",
                    status=CheckStatus.FAIL if nvidia_install else CheckStatus.WARN,
                    message="CUDA operations failed",
                    details=[str(e)],
                    fatal=nvidia_install
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

    def _check_crispasr(self):
        """CrispASR and its models, for the CrispASR pipelines (1.10.0, route B).

        One line for CrispASR -- its version, its build and the device it would run on with --device
        auto -- and one per model the CrispASR pipelines offer. It never downloads anything, never loads
        CrispASR's libraries and is never fatal. A model not downloaded is INFO: the normal state until
        a CrispASR pipeline first runs, which fetches it. A computer that cannot run CrispASR is a WARN.
        """
        try:
            from whisperjav.crispasr import install, manifest
        except Exception:
            return  # the CrispASR package cannot be loaded; there is nothing to report on
        self.results.append(self._crispasr_line(CRISPASR_NAME, self._crispasr_runtime_result, install))
        listed = set()
        for pipeline, names in manifest.PIPELINE_MODELS.items():
            for name in names:
                if name in listed:
                    continue
                listed.add(name)
                self.results.append(self._crispasr_line(
                    CRISPASR_MODEL_NAME.format(model=name), self._crispasr_model_result,
                    install, manifest.MODELS[name], pipeline))
        # ChronosJAV's aligners and timing model, and Cohere-CrispASR's aligner (item 12).
        for name, model in manifest.ALIGNER_MODELS.items():
            used_by = (CRISPASR_DEFAULT_ALIGNER_FOR if name == manifest.DEFAULT_ALIGNER
                       else CRISPASR_ALIGNER_FOR)
            self.results.append(self._crispasr_line(
                CRISPASR_MODEL_NAME.format(model=name), self._crispasr_model_result, install, model, used_by))
        for name, model in manifest.TIMING_MODELS.items():
            self.results.append(self._crispasr_line(
                CRISPASR_MODEL_NAME.format(model=name), self._crispasr_model_result,
                install, model, CRISPASR_TIMING_FOR))

    @staticmethod
    def _crispasr_line(name: str, make, *args) -> "CheckResult":
        """One CrispASR line from ``make(*args)``; a check that fails is a WARN line, not a crash."""
        try:
            return make(*args)
        except Exception as e:  # noqa: BLE001 -- a diagnostic must not crash --check
            return CheckResult(name=name, status=CheckStatus.WARN,
                               message=CRISPASR_CHECK_FAILED.format(error=e))

    @staticmethod
    def _crispasr_model_result(install, model, pipeline: str) -> "CheckResult":
        """The line for one CrispASR model file and what uses it (master plan 1.5, 3.3; item 12)."""
        downloaded = install.model_status(model).downloaded
        state = (CRISPASR_DOWNLOADED if downloaded
                 else CRISPASR_MODEL_NOT_DOWNLOADED.format(size=install.mb(model.size)))
        return CheckResult(
            name=CRISPASR_MODEL_NAME.format(model=model.name),
            status=CheckStatus.PASS if downloaded else CheckStatus.INFO,
            message=CRISPASR_MODEL_TEXT.format(label=model.label, pipeline=pipeline, state=state),
        )

    @staticmethod
    def _crispasr_runtime_result(install) -> "CheckResult":
        """The CrispASR line: what the run would use with --device auto (route B design 9.1)."""
        if not install.platform_supported():
            return CheckResult(name=CRISPASR_NAME, status=CheckStatus.WARN,
                               message=CRISPASR_NO_BUILD_TEXT.format(platform=sys.platform,
                                                                     machine=platform.machine()))
        if not install.crispasr_installed():
            return CheckResult(name=CRISPASR_NAME, status=CheckStatus.WARN, message=CRISPASR_NOT_INSTALLED_TEXT)
        if install.has_avx2() is False:
            return CheckResult(name=CRISPASR_NAME, status=CheckStatus.WARN, message=CRISPASR_NO_AVX2_TEXT)
        version, build = install.installed_build()
        build_name = CRISPASR_BUILD_NAMES.get(build, build)
        try:
            choice = install.choose_device(None, build, version)
        except install.CrispASRSetupError as e:  # pragma: no cover - auto never refuses
            return CheckResult(name=CRISPASR_NAME, status=CheckStatus.WARN,
                               message=CRISPASR_DEVICE_PROBLEM.format(version=version, build=build_name, reason=e))
        why = f": {choice.reason}" if choice.reason else ""
        return CheckResult(name=CRISPASR_NAME, status=CheckStatus.PASS,
                           message=CRISPASR_TEXT.format(version=version, build=build_name,
                                                        device=CRISPASR_DEVICE_NAMES.get(choice.device, choice.device),
                                                        why=why))

    def _display_results(self):
        """Display all check results in a formatted manner."""
        print()
        
        # Group results by status
        failures = [r for r in self.results if r.status == CheckStatus.FAIL]
        warnings = [r for r in self.results if r.status == CheckStatus.WARN]
        passes = [r for r in self.results if r.status == CheckStatus.PASS]
        infos = [r for r in self.results if r.status == CheckStatus.INFO]

        # Display each group
        for result in passes:
            self._display_result(result, Fore.GREEN)

        # v1.10.0: the CUDA 12.8 line (W43 (3)) is INFO; nothing was shown for INFO before.
        for result in infos:
            self._display_result(result, Fore.CYAN)

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



def ensure_crispasr_ready(passes, *, device=None, temp_dir=None, exit_on_fail: bool = True) -> bool:
    """
    Make CrispASR, the device and each CrispASR pass's model ready before a run starts.

    ``passes`` lists (pass number, pipeline, --passN-model value or None) for every Ensemble pass
    that names a CrispASR pipeline; ``device`` is --device. A missing model is downloaded here, unless
    --offline, and staged under a plain-ASCII name when its path has other characters (route B design
    4.2): a missing package, a processor without AVX2, --device cuda without an NVIDIA GPU, a missing
    file under --offline or a failed download then stops the run before any audio is read. Checked
    only inside the pass, a two-pass run would first run the other pass in full, for nothing.

    The pass worker prepares the same again before its first film; by then that is a quick check. On
    failure this prints what the user can do and ends the run with status 1, as the other start-up
    checks here do; ``exit_on_fail=False`` returns False instead.
    """
    import tempfile

    from whisperjav.crispasr import install, manifest

    stage_root = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "whisperjav"
    try:
        for _n, pipeline, model_name in passes:
            model = manifest.model_for(pipeline, model_name)
            install.prepare_runtime(device, model)
            path = install.ensure_model(model)
            install.stage_model(path, stage_root, model.sha256)
        return True
    except install.CrispASRSetupError as exc:
        if not exit_on_fail:
            return False
        names = ", ".join(f"{pipeline} (pass {n})" for n, pipeline, _m in passes)
        lines = ["CrispASR is not ready for this run", ""]
        # Wrapped, so the box keeps its frame however many passes are named.
        lines += textwrap.wrap(f"This run uses {names}. WhisperJAV checks CrispASR and downloads its model "
                               f"the first time it is needed. That did not work:", BOX_WIDTH - 4)
        lines += [""]
        lines += [f"  {chunk}" for chunk in textwrap.wrap(str(exc), BOX_WIDTH - 8)[:4]]
        lines += [
            "",
            "Nothing has been transcribed.",
        ]
        if install.hub_offline():
            lines += [
                "",
                "Downloads are switched off for this run (--offline), so",
                "nothing can be fetched now. Run once without --offline first.",
            ]
        _print_box(lines, Fore.RED)
        sys.exit(1)


def ensure_speech_enhancer_available(backend, *, exit_on_fail: bool = True) -> bool:
    """
    Make sure a chosen audio clean-up can actually run, before any audio is read.

    Owner, 2026-09-17: "if user selected any but they cannot run then it is a
    failure and the process shall stop with helpful communication." Choosing a
    clean-up is a deliberate act. Falling back to no clean-up would hand the user
    subtitles made from the untouched audio, from a run that exited 0, with only
    a warning in the log to explain it.

    Applies to every clean-up that has to be installed (the factory's
    FATAL_WHEN_UNAVAILABLE). "none" and "ffmpeg-dsp" cannot be missing.

    This is a better MESSAGE, earlier. The guarantee is in pipeline_helper, which
    raises SpeechEnhancerUnavailable however the backend is reached, including by
    a path this check does not predict.

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

    # Offer the clean-ups this machine can actually run, rather than a fixed list
    # that may name something else that is also missing.
    alternatives = []
    for other in SpeechEnhancerFactory.list_backends():
        if other in ("none", backend):
            continue
        ok, _ = SpeechEnhancerFactory.is_backend_available(other)
        if ok:
            alternatives.append(other)

    lines = [
        f"The {backend} audio clean-up cannot run on this machine",
        "",
        f"This run was asked to clean up the audio with {backend}, and it is",
        "not installed.",
        "",
        "Nothing has been transcribed. Going ahead without it would hand you",
        "subtitles made from the untouched audio -- the thing you asked to",
        "have cleaned up -- and the run would look successful.",
        "",
        "What you can do:",
        f"  - Install it, once:  {hint}",
    ]
    if backend == "htdemucs":
        lines += [
            "    Nothing is compiled, and it brings a model of about 84 MB the",
            "    first time you run it.",
        ]
    if alternatives:
        lines += [
            "  - Or pick a clean-up that is already installed:",
        ]
        lines += [f"      {name}" for name in alternatives]
        lines += [
            "    On the command line that is --passN-speech-enhancer; in the",
            "    Ensemble tab it is that pass's Audio Clean-up.",
        ]
    lines += [
        "  - Or leave the clean-up out and transcribe the audio as it is.",
    ]
    if backend == "htdemucs":
        lines += [
            "",
            "Note: the model comes from Meta's own site, not from Hugging Face,",
            "so a Hugging Face mirror or --offline does not apply to it.",
        ]
    _print_box(lines, Fore.RED)
    sys.exit(1)


def ensure_segmenter_backend_available(backend, *, exit_on_fail: bool = True) -> bool:
    """
    Make sure a chosen speech segmenter can be built, before any audio is read.

    The same rule as the clean-ups above, for the same reason. A segmenter that
    cannot be built already stopped the run -- but deep inside the recogniser and
    with the words "Speech Segmenter not configured - this is an architecture
    violation", which tells a user nothing. This says it up front, in plain
    language, and names the segmenters this machine can run.

    ``ensure_segmenter_model_available`` is the sibling of this: that one is
    about a MODEL that has to be downloaded, this one about a PACKAGE that has to
    be installed.
    """
    if not backend or backend in ("none", "faster-whisper"):
        return True

    try:
        from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
    except Exception:
        return True

    try:
        available, hint = SpeechSegmenterFactory.is_backend_available(backend)
    except Exception:
        return True  # Unknown name: argparse already refuses those.

    if available:
        return True
    if not exit_on_fail:
        return False

    alternatives = []
    try:
        for other in SpeechSegmenterFactory.list_backends():
            if other in ("none", backend):
                continue
            try:
                ok, _ = SpeechSegmenterFactory.is_backend_available(other)
            except Exception:
                continue
            if ok:
                alternatives.append(other)
    except Exception:
        pass

    lines = [
        f"The {backend} speech detector cannot run on this machine",
        "",
        f"This run needs {backend} to find the speech in your audio, and it is",
        "not installed.",
        "",
        "Nothing has been transcribed. Without a working speech detector",
        "every scene would fail and you would get an empty subtitle file.",
        "",
        "What you can do:",
        f"  - Install it, once:  {hint}",
    ]
    if alternatives:
        lines += ["  - Or pick a speech detector that is already installed:"]
        lines += [f"      {name}" for name in alternatives[:8]]
        lines += [
            "    On the command line that is --speech-segmenter (or",
            "    --passN-speech-segmenter); in the Ensemble tab it is that",
            "    pass's Speech Segmenter.",
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

# ---------------------------------------------------------------------------------------------
# The start of each run (design section 22.5; owner W234 D-A, W237 1)
#
# Every install of 1.10 is made for one accelerator, chosen when installing and recorded in
# whisperjav_stack.txt. No run asks whether to go on without a GPU (the 1.9 question, the
# --accept-cpu-mode answer and the GUI's "Accept CPU-only mode" box had no job left once the
# install records what it is made for). The start of a run is one of three things:
#   silent   -- the install's GPU can be used (NVIDIA, or the Apple GPU), or, without a record,
#               PyTorch finds a GPU it can use;
#   notice   -- a box that says where each part runs, then the run goes on;
#   stop     -- only an NVIDIA install whose GPU cannot be used now (exit status 3); --device cpu
#               runs it on the CPU instead. PyTorch missing is stopped earlier, in whisperjav.main.
# ---------------------------------------------------------------------------------------------

BOX_WIDTH = 72


def _print_box(lines, colour):
    """Print `lines` inside a double-ruled box so the message cannot be missed
    (owner, 2026-09-06: "printed out in a very very visible manner"). Used by the stops of
    the readiness checks above."""
    inner = BOX_WIDTH - 2
    print(f"{colour}╔{'═' * inner}╗")
    print(f"║{' ' * inner}║")
    for line in lines:
        print(f"║  {line.ljust(inner - 2)}║")
    print(f"║{' ' * inner}║")
    print(f"╚{'═' * inner}╝{Style.RESET_ALL}")

# Exit status of a run that did not start because this install's NVIDIA GPU cannot be used
# (design 22.5.4; 22.13 detail 2). The GUI offers "Run on the CPU this time" for it.
EXIT_NVIDIA_UNUSABLE = 3

# Where the parts run: the accelerator table holds these lines (design 22.6.8; W241 Q1).
NOTICE_MADE_FOR = _ACC.NOTICE_MADE_FOR
NOTICE_NO_RECORD = _ACC.NOTICE_NO_RECORD
NOTICE_GGUF = _ACC.NOTICE_GGUF
NOTICE_OTHERS = _ACC.NOTICE_OTHERS
NOTICE_DEVICE_CPU = _ACC.NOTICE_DEVICE_CPU
NOTICE_DEVICE_CPU_ASR_MODELS = _ACC.NOTICE_DEVICE_CPU_ASR_MODELS
NOTICE_DEVICE_CPU_CRISPASR = _ACC.NOTICE_DEVICE_CPU_CRISPASR
CPU_WHERE = _ACC.CPU_WHERE
GGUF_VULKAN_WHERE = _ACC.GGUF_VULKAN_WHERE
GGUF_METAL_WHERE = _ACC.GGUF_METAL_WHERE
GGUF_CUDA_WHERE = _ACC.GGUF_CUDA_WHERE
GGUF_CPU_WHERE = _ACC.GGUF_CPU_WHERE
GGUF_MISSING_WHERE = _ACC.GGUF_MISSING_WHERE
OWN_PYTORCH_WHERE = _ACC.OWN_PYTORCH_WHERE

NVIDIA_STOP_TEXT = ("This install is made for an NVIDIA GPU, but the GPU cannot be used now: {reason}. "
                    "Nothing was processed. To fix it: {fix}. To run on the CPU (not accelerated) this "
                    "time, add --device cpu.")
FIX_DRIVER = ("update the NVIDIA driver (580 or newer) from https://www.nvidia.com/drivers, or run the "
              "WhisperJAV installer again and choose another accelerator")
FIX_INSTALLER = "run the WhisperJAV installer again; it checks this card and offers what fits it"
REASON_NO_CUDA_BUILD = "this install's PyTorch has no NVIDIA GPU support (it may have been replaced)"
REASON_NOT_FOUND = ("PyTorch finds no NVIDIA GPU it can use (the driver may be missing, too old or not working, "
                    "or the card was removed)")


def _box(lines, colour):
    """A double-ruled box so the text cannot be missed (owner, 2026-09-06: "printed out in a very
    very visible manner")."""
    inner = BOX_WIDTH - 2
    out = [f"{colour}╔{'═' * inner}╗", f"║{' ' * inner}║"]
    for line in lines:
        for piece in (textwrap.wrap(line, inner - 4, subsequent_indent="  ") or [""]):
            out.append(f"║  {piece.ljust(inner - 2)}║")
    out += [f"║{' ' * inner}║", f"╚{'═' * inner}╝{Style.RESET_ALL}"]
    return "\n".join(out)


def _to_log_file(text: str, level: int) -> None:
    """Write `text` to the --log-file only: the box itself is already on the console."""
    import logging
    from whisperjav.utils.logger import logger as wj_logger
    for handler in wj_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.emit(logging.LogRecord(wj_logger.name, level, __file__, 0, text, None, None))


def _ctranslate2_sees_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _gguf_where(best: str) -> str:
    """Where the GGUF pipelines run with --device auto, from the installed CrispASR build."""
    try:
        from whisperjav.crispasr import install as crisp
        if not crisp.crispasr_installed():
            return GGUF_MISSING_WHERE
        _version, build = crisp.installed_build()
    except Exception:  # noqa: BLE001
        return GGUF_CPU_WHERE
    if build == "vulkan":
        return GGUF_VULKAN_WHERE
    if build == "cuda":
        return GGUF_CUDA_WHERE if best == "cuda" else GGUF_CPU_WHERE
    if sys.platform == "darwin" and best == "mps":
        return GGUF_METAL_WHERE
    return GGUF_CPU_WHERE


def _gpu_label(torch_module) -> str:
    try:
        if torch_module.cuda.is_available():
            name = torch_module.cuda.get_device_name(0)
            kind = "AMD GPU (ROCm)" if getattr(torch_module.version, "hip", None) else "NVIDIA GPU (CUDA)"
            return f"its {kind}, {name}"
        if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
            return "the Apple GPU (Metal)"
    except Exception:  # noqa: BLE001
        pass
    return "the GPU it reports"


def _nvidia_reason(torch_module):
    """(reason, fix) for an NVIDIA install whose GPU cannot be used now."""
    from whisperjav.utils import device_detector as dd
    if dd.CUDA_UNUSABLE_REASON:
        cap = dd.CUDA_UNUSABLE_CAPABILITY
        if cap is not None and tuple(cap) < MIN_COMPUTE_CAPABILITY_110:
            return dd.CUDA_UNUSABLE_REASON.rstrip(".") + ". " + OLD_CARD_TEXT_110, FIX_INSTALLER
        return dd.CUDA_UNUSABLE_REASON.rstrip("."), FIX_INSTALLER
    if not getattr(torch_module.version, "cuda", None):
        return REASON_NO_CUDA_BUILD, FIX_INSTALLER
    return REASON_NOT_FOUND, FIX_DRIVER


def start_of_run_check(device: Optional[str] = None) -> str:
    """The start of a run, by the install record. Returns "silent" or "notice"; exits with status 3
    for the one stop. Never asks anything (design 22.5.2)."""
    import torch
    from whisperjav.installer.core import accelerators

    has_record, value = accelerators.read_record()
    acc = accelerators.get(value)
    best = get_best_device()  # also sets device_detector.CUDA_UNUSABLE_REASON (#411)

    if acc is not None and acc.nvidia and best != "cuda" and device != "cpu":
        reason, fix = _nvidia_reason(torch)
        text = NVIDIA_STOP_TEXT.format(reason=reason, fix=fix)
        print("\n" + _box(["THE NVIDIA GPU CANNOT BE USED", "", text], Fore.RED) + "\n", flush=True)
        _to_log_file(text, 40)
        sys.exit(EXIT_NVIDIA_UNUSABLE)

    if device != "cpu":
        if acc is not None and acc.nvidia:
            return _cublas12_warning_and_silent()
        if value == "metal" and best == "mps":
            return "silent"
        if acc is None and best in ("cuda", "mps"):
            return _cublas12_warning_and_silent()

    lines = [NOTICE_MADE_FOR.format(name=acc.name) if acc is not None else NOTICE_NO_RECORD]
    if device == "cpu":
        # CrispASR's aligner and timing model cannot be held on the CPU (item 12, probe P1): said only
        # where this install's CrispASR build would ask for a GPU; then "every pipeline" is not said.
        if _gguf_where(best) in (GGUF_CUDA_WHERE, GGUF_VULKAN_WHERE, GGUF_METAL_WHERE):
            lines += [NOTICE_DEVICE_CPU_ASR_MODELS, NOTICE_DEVICE_CPU_CRISPASR]
        else:
            lines.append(NOTICE_DEVICE_CPU)
    else:
        lines.append(NOTICE_GGUF.format(where=_gguf_where(best)))
        if value == "own-pytorch" and best in ("cuda", "mps"):
            ct2 = "the NVIDIA GPU (CUDA)" if _ctranslate2_sees_cuda() else "the CPU (not accelerated)"
            lines.append(NOTICE_OTHERS.format(where=OWN_PYTORCH_WHERE.format(gpu=_gpu_label(torch), ct2=ct2)))
        else:
            lines.append(NOTICE_OTHERS.format(where=CPU_WHERE))
    print("\n" + _box(lines, Fore.YELLOW) + "\n", flush=True)
    _to_log_file(" ".join(lines), 30)
    return "notice"


def _cublas12_warning_and_silent() -> str:
    missing = cublas12_files_missing()
    if missing:
        print(f"\n{Fore.YELLOW}{CUBLAS12_MISSING_TEXT.format(files=', '.join(missing))}{Style.RESET_ALL}\n")
    return "silent"


if __name__ == "__main__":
    # Run checks when module is executed directly
    import argparse
    parser = argparse.ArgumentParser(description="WhisperJAV environment checker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    run_preflight_checks(verbose=args.verbose)