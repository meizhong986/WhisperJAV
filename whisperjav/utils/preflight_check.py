#!/usr/bin/env python3
"""Pre-flight environment validation for WhisperJAV.

This module ensures the runtime environment meets all requirements,
with special focus on CUDA availability and compatibility.
"""

import time
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
        
        # Display results
        self._display_results()
        
        # Return False if any fatal check failed
        return not any(r.fatal and r.status == CheckStatus.FAIL for r in self.results)
    
    def _check_python_version(self):
        """Check Python version compatibility."""
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"
        
        if version >= (3, 9) and version < (3, 13):
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
                details=["Supported versions: 3.9, 3.10, 3.11, 3.12 (3.13+ breaks openai-whisper)"],
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
            'stable_whisper': "Required only for legacy fast/faster pipelines"
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

def _wait_for_keypress_with_timeout(timeout_seconds=30):
    """
    Cross-platform implementation to wait for any keypress with timeout.
    Returns True if key was pressed, False if timeout occurred.
    """
    if sys.platform == 'win32':
        # Windows implementation using msvcrt
        try:
            import msvcrt
            start_time = time.time()
            while (time.time() - start_time) < timeout_seconds:
                if msvcrt.kbhit():
                    msvcrt.getch()  # Consume the keypress
                    return True
                time.sleep(0.1)
            return False
        except ImportError:
            # Fallback if msvcrt not available
            time.sleep(timeout_seconds)
            return False
    else:
        # Unix-like systems (Linux, macOS) using select
        try:
            import select
            print("Press any key to continue immediately, or wait for auto-continue...")
            # Use select to wait for stdin with timeout
            rlist, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
            if rlist:
                sys.stdin.readline()  # Consume the input
                return True
            return False
        except (ImportError, OSError):
            # Fallback if select not available or stdin not supported
            time.sleep(timeout_seconds)
            return False


def enforce_gpu_requirement(accept_cpu_mode=False, timeout_seconds=30):
    """
    Check for GPU (CUDA or MPS) availability with friendly warning and optional bypass.

    Args:
        accept_cpu_mode: If True, skip the warning entirely (from --accept-cpu flag)
        timeout_seconds: How long to wait for user acknowledgment (default: 30)

    Returns:
        bool: True if GPU is available or user accepted CPU mode, False otherwise
    """
    # Skip check entirely if user explicitly accepted CPU mode
    if accept_cpu_mode:
        print(f"{Fore.YELLOW}ℹ GPU check bypassed via --accept-cpu-mode flag.{Style.RESET_ALL}")
        return True

    try:
        import torch

        # Check for any GPU (CUDA or MPS)
        best_device = get_best_device()
        if best_device in ('cuda', 'mps'):
            return True

        # No GPU detected - show friendly warning
        print(f"\n{Fore.YELLOW}{'='*70}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}⚠  GPU Performance Warning{Style.RESET_ALL}")
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

        print(f"{Fore.YELLOW}You can continue with CPU-only mode, but expect slower performance.{Style.RESET_ALL}\n")

        print(f"{Fore.GREEN}Press any key to continue with CPU mode...{Style.RESET_ALL}")
        print(f"(Auto-continuing in {timeout_seconds} seconds, or use --accept-cpu-mode to skip this warning)")
        print(f"{Fore.YELLOW}{'='*70}{Style.RESET_ALL}\n")

        # Wait for keypress or timeout
        key_pressed = _wait_for_keypress_with_timeout(timeout_seconds)

        if key_pressed:
            print(f"\n{Fore.GREEN}✓ Continuing with CPU mode (user confirmed)...{Style.RESET_ALL}\n")
        else:
            print(f"\n{Fore.GREEN}✓ Auto-continuing with CPU mode after timeout...{Style.RESET_ALL}\n")

        return True

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