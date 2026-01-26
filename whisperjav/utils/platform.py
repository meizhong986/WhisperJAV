#!/usr/bin/env python3
"""
Platform detection utilities for WhisperJAV.

Provides consistent platform detection across all modules for:
- Operating system detection (Windows, Linux, macOS)
- Runtime environment detection (Colab, Kaggle, Jupyter)
- Architecture detection (x86_64, arm64)

Usage:
    from whisperjav.utils.platform import is_windows, is_colab, get_platform_info

    if is_windows():
        # Windows-specific code
        pass
"""
import os
import sys
import platform as platform_module
from typing import Literal

# Type alias for platform names
PlatformType = Literal["windows", "linux", "macos", "unknown"]


def get_platform() -> PlatformType:
    """
    Get the current operating system platform.

    Returns:
        One of: "windows", "linux", "macos", "unknown"

    Example:
        >>> get_platform()
        'windows'
    """
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "linux":
        return "linux"
    elif sys.platform == "darwin":
        return "macos"
    else:
        return "unknown"


def is_windows() -> bool:
    """Check if running on Windows."""
    return sys.platform == "win32"


def is_linux() -> bool:
    """Check if running on Linux."""
    return sys.platform == "linux"


def is_macos() -> bool:
    """Check if running on macOS."""
    return sys.platform == "darwin"


def is_unix() -> bool:
    """Check if running on Unix-like system (Linux or macOS)."""
    return sys.platform in ("linux", "darwin")


def is_colab() -> bool:
    """
    Check if running in Google Colab.

    Returns:
        True if running in Google Colab environment.
    """
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """
    Check if running in Kaggle notebook.

    Returns:
        True if running in Kaggle environment.
    """
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def is_notebook() -> bool:
    """
    Check if running in any Jupyter notebook environment.

    Includes: Jupyter Notebook, JupyterLab, Google Colab, Kaggle, VS Code notebooks.

    Returns:
        True if running in a notebook environment.
    """
    # Check for Colab/Kaggle first
    if is_colab() or is_kaggle():
        return True

    # Check for IPython/Jupyter
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            # Check if we're in a ZMQ interactive shell (Jupyter)
            if "IPKernelApp" in ipython.config:
                return True
            # Check shell type
            shell = ipython.__class__.__name__
            if shell in ("ZMQInteractiveShell", "TerminalInteractiveShell"):
                return "ZMQ" in shell
        return False
    except (ImportError, AttributeError):
        return False


def is_interactive() -> bool:
    """
    Check if running in an interactive environment.

    Returns:
        True if running interactively (notebook or interactive terminal).
    """
    if is_notebook():
        return True

    # Check if stdin is a TTY (interactive terminal)
    try:
        return sys.stdin.isatty()
    except AttributeError:
        return False


def get_architecture() -> str:
    """
    Get the CPU architecture.

    Returns:
        Architecture string like "x86_64", "arm64", "aarch64".
    """
    return platform_module.machine()


def is_arm() -> bool:
    """Check if running on ARM architecture (including Apple Silicon)."""
    arch = get_architecture().lower()
    return "arm" in arch or "aarch" in arch


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (M1/M2/M3)."""
    return is_macos() and is_arm()


def get_platform_info() -> dict:
    """
    Get detailed platform information.

    Returns:
        Dictionary with comprehensive platform details.

    Example:
        >>> get_platform_info()
        {
            'platform': 'windows',
            'python_version': '3.10.12',
            'machine': 'AMD64',
            'processor': 'Intel64 Family 6...',
            'is_colab': False,
            'is_kaggle': False,
            'is_notebook': False,
            'is_interactive': True,
            'is_arm': False,
        }
    """
    return {
        "platform": get_platform(),
        "python_version": platform_module.python_version(),
        "python_implementation": platform_module.python_implementation(),
        "machine": get_architecture(),
        "processor": platform_module.processor(),
        "system": platform_module.system(),
        "release": platform_module.release(),
        "is_colab": is_colab(),
        "is_kaggle": is_kaggle(),
        "is_notebook": is_notebook(),
        "is_interactive": is_interactive(),
        "is_arm": is_arm(),
    }


def get_recommended_extras() -> list[str]:
    """
    Get recommended extras for the current platform.

    Returns:
        List of recommended extra names for pip install.

    Example:
        >>> get_recommended_extras()
        ['cli', 'gui', 'translate']  # On Windows
        ['cli', 'translate']  # On Linux/Colab
    """
    extras = ["cli"]  # CLI is always recommended

    if is_colab() or is_kaggle():
        # Notebook environments: no GUI, add huggingface for model downloads
        extras.extend(["translate", "huggingface"])
    elif is_windows():
        # Windows: full experience with GUI
        extras.extend(["gui", "translate"])
    elif is_macos():
        # macOS: GUI works with WebKit
        extras.extend(["gui", "translate"])
    elif is_linux():
        # Linux: GUI needs WebKit, recommend CLI-focused
        extras.extend(["translate"])

    return extras


def print_platform_summary():
    """Print a summary of the current platform (useful for debugging)."""
    info = get_platform_info()
    print("Platform Information:")
    print(f"  OS: {info['platform']} ({info['system']} {info['release']})")
    print(f"  Python: {info['python_version']} ({info['python_implementation']})")
    print(f"  Architecture: {info['machine']}")
    print(f"  Environment: ", end="")

    envs = []
    if info['is_colab']:
        envs.append("Google Colab")
    if info['is_kaggle']:
        envs.append("Kaggle")
    if info['is_notebook']:
        envs.append("Notebook")
    if info['is_interactive']:
        envs.append("Interactive")

    print(", ".join(envs) if envs else "Standard")
    print(f"  Recommended extras: {', '.join(get_recommended_extras())}")


# Module-level exports
__all__ = [
    "get_platform",
    "is_windows",
    "is_linux",
    "is_macos",
    "is_unix",
    "is_colab",
    "is_kaggle",
    "is_notebook",
    "is_interactive",
    "get_architecture",
    "is_arm",
    "is_apple_silicon",
    "get_platform_info",
    "get_recommended_extras",
    "print_platform_summary",
]
