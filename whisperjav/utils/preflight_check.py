#!/usr/bin/env python3
"""Pre-flight environment validation for WhisperJAV.

This module ensures the runtime environment meets all requirements,
with special focus on CUDA availability and compatibility.
"""

import time 
import sys
import os
import shutil
import platform
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

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
        
        if version >= (3, 8) and version < (3, 13):
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
                details=["Supported versions: 3.8, 3.9, 3.10, 3.11, 3.12"],
                fatal=True
            ))
    
    def _check_cuda_availability(self):
        """Check for CUDA availability - this is mandatory."""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                cuda_version = torch.version.cuda
                
                self.results.append(CheckResult(
                    name="CUDA Availability",
                    status=CheckStatus.PASS,
                    message=f"CUDA {cuda_version} available with {device_count} GPU(s)",
                    details=[f"Primary GPU: {device_name}"]
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
            'stable_whisper',
            'faster_whisper',
            'torch',
            'torchaudio',
            'numpy',
            'ffmpeg'
        ]
        
        missing = []
        for dep in critical_deps:
            try:
                __import__(dep)
            except ImportError:
                missing.append(dep)
        
        if not missing:
            self.results.append(CheckResult(
                name="Python Dependencies",
                status=CheckStatus.PASS,
                message="All critical dependencies installed"
            ))
        else:
            self.results.append(CheckResult(
                name="Python Dependencies",
                status=CheckStatus.FAIL,
                message=f"Missing {len(missing)} critical dependencies",
                details=[f"Missing: {', '.join(missing)}"],
                fatal=True
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

def enforce_cuda_requirement():
    """Simple CUDA enforcement for main.py integration."""
    print("\nInitial check for CUDA enabled torch. WhisperJAV requires an NVIDIA GPU with CUDA support.")
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"\n{Fore.RED}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.RED}❌ CUDA Required - CPU Mode Not Supported{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}")
            print("\nWhisperJAV requires GPU with CUDA support, and CUDA enabled torch.")
            print("If you already have CUDA, please check that torch is cuda enabled.")
            print("Uninstall cpu torch and reinstall torch and torch audio with CUDA support\n.")
            print("Run 'whisperjav --check' for detailed diagnostics.")
            print(f"{Fore.RED}{'='*60}{Style.RESET_ALL}\n")

            # Wait for user input before exiting
            input("Press Enter to exit...")
            sys.exit(1)
    except ImportError:
        print(f"\n{Fore.RED}PyTorch not installed. Please complete installation first.{Style.RESET_ALL}\n")
        input("Press Enter to exit...")
        sys.exit(1)
        
        
if __name__ == "__main__":
    # Run checks when module is executed directly
    import argparse
    parser = argparse.ArgumentParser(description="WhisperJAV environment checker")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show detailed information")
    args = parser.parse_args()
    
    run_preflight_checks(verbose=args.verbose)