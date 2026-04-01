#!/usr/bin/env python3
"""
Console utilities for WhisperJAV.

Provides:
- UTF-8 encoding fix for Windows console
- UTF-8 mode relaunch for CLI entry points (fixes open() default encoding)
- Safe print functions that handle encoding errors
- Warning suppression for common noisy dependencies

This module should be imported early, before other dependencies,
to ensure console encoding is properly configured.

Usage:
    from whisperjav.utils.console import ensure_utf8_console, safe_print

    # At module level (before other imports):
    ensure_utf8_console()

    # For safe printing:
    safe_print("Text with unicode: \u3042\u3044\u3046")
"""
import io
import os
import sys
import warnings
from typing import Any, TextIO


def relaunch_for_utf8(module_name: str) -> None:
    """
    On Windows with non-UTF-8 locale, relaunch the process in UTF-8 mode.

    Third-party libraries (PySubtrans, etc.) may use open() without explicit
    encoding, which defaults to the system locale (GBK on Chinese Windows).
    This causes UnicodeDecodeError when processing non-ASCII content like
    Chinese translations. See GitHub issue #190.

    The only way to fix open() default encoding in Python 3.10-3.14 is to
    start the interpreter with ``-X utf8`` or ``PYTHONUTF8=1``. This function
    transparently relaunches the process with UTF-8 mode enabled.

    Call at the very start of CLI entry points, before any other code.
    Does nothing on non-Windows or if already in UTF-8 mode.

    Args:
        module_name: The module to run via ``-m`` (e.g., 'whisperjav.translate.cli')
    """
    if os.name != 'nt':
        return

    if getattr(sys.flags, 'utf8_mode', False):
        return

    # Safety guard against infinite relaunch (shouldn't happen since
    # utf8_mode would be True after relaunch, but protects edge cases)
    if os.environ.get('_WHISPERJAV_UTF8') == '1':
        return

    if not sys.executable:
        return

    import subprocess

    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8:replace'
    env['_WHISPERJAV_UTF8'] = '1'

    result = subprocess.run(
        [sys.executable, '-X', 'utf8', '-m', module_name] + sys.argv[1:],
        env=env,
    )
    sys.exit(result.returncode)


def ensure_utf8_console() -> None:
    """
    Ensure stdout and stderr use UTF-8 encoding on Windows.

    This fixes issues with unicode characters (especially Japanese)
    not displaying correctly in Windows console. Should be called
    early in the application startup, before any output.

    Also sets PYTHONUTF8=1 in the environment so any child processes
    inherit UTF-8 mode.

    Safe to call multiple times - will only apply the fix once.

    Example:
        # At the top of your main module:
        from whisperjav.utils.console import ensure_utf8_console
        ensure_utf8_console()
    """
    # Set env vars for child processes (subprocess, os.system, etc.)
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8:replace')

    _fix_stream_encoding(sys.stdout, 1)
    _fix_stream_encoding(sys.stderr, 2)


def _fix_stream_encoding(stream: TextIO | None, fd: int) -> None:
    """
    Fix encoding for a single stream.

    Prefers reconfigure() (Python 3.7+) which modifies the stream in-place,
    preserving any cached references. Falls back to TextIOWrapper replacement.

    Args:
        stream: The stream to fix (sys.stdout or sys.stderr)
        fd: File descriptor (1 for stdout, 2 for stderr)
    """
    if stream is None:
        return

    # Check if already UTF-8
    try:
        if hasattr(stream, 'encoding') and stream.encoding:
            if stream.encoding.lower() in ('utf-8', 'utf8'):
                return
    except (AttributeError, TypeError):
        pass

    # Prefer reconfigure() - modifies stream in-place (safer)
    try:
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
            return
    except (AttributeError, OSError, ValueError):
        pass

    # Fallback: create new TextIOWrapper
    try:
        if hasattr(stream, 'buffer'):
            buffer = stream.buffer
        else:
            buffer = io.BufferedWriter(io.FileIO(fd, 'w'))

        wrapper = io.TextIOWrapper(
            buffer,
            encoding='utf-8',
            errors='replace',  # Replace unencodable chars instead of crashing
            line_buffering=True
        )

        if fd == 1:
            sys.stdout = wrapper
        elif fd == 2:
            sys.stderr = wrapper

    except (AttributeError, OSError, ValueError):
        # If we can't fix it, just continue with original
        pass


def safe_print(*args: Any, **kwargs: Any) -> None:
    """
    Print with automatic encoding error handling.

    Like print(), but handles encoding errors gracefully by replacing
    unencodable characters instead of crashing.

    Args:
        *args: Arguments to print
        **kwargs: Keyword arguments passed to print()

    Example:
        safe_print("Japanese: \u65e5\u672c\u8a9e")
        safe_print("Mixed:", "English", "\u4e2d\u6587", sep=" | ")
    """
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Fallback: encode with replacement
        output = " ".join(str(arg) for arg in args)
        try:
            encoded = output.encode(sys.stdout.encoding or 'utf-8', errors='replace')
            decoded = encoded.decode(sys.stdout.encoding or 'utf-8', errors='replace')
            print(decoded, **kwargs)
        except Exception:
            # Last resort: ASCII only
            print(output.encode('ascii', errors='replace').decode('ascii'), **kwargs)


def suppress_dependency_warnings() -> None:
    """
    Suppress common noisy warnings from dependencies.

    Suppresses:
    - TensorFlow/oneDNN warnings
    - pkg_resources deprecation warnings
    - PyTorch dtype warnings
    - Various experimental feature warnings
    - requests chardet/urllib3 version mismatch (cosmetic)

    Should be called early in application startup.

    Example:
        from whisperjav.utils.console import suppress_dependency_warnings
        suppress_dependency_warnings()
    """
    # TensorFlow environment variables (must be set before TF import)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    # Suppress Python warnings
    _warning_patterns = [
        (".*pkg_resources is deprecated.*", DeprecationWarning),
        (".*pkg_resources.*", DeprecationWarning),
        (".*chunk_length_s.*is very experimental.*", None),
        (".*sparse_softmax_cross_entropy.*deprecated.*", None),
        (".*Pydantic V1.*deprecated.*", DeprecationWarning),
        (".*distutils.*deprecated.*", DeprecationWarning),
    ]

    for pattern, category in _warning_patterns:
        if category:
            warnings.filterwarnings("ignore", message=pattern, category=category)
        else:
            warnings.filterwarnings("ignore", message=pattern)

    # Suppress pkg_resources module-level warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="pkg_resources"
    )

    # Suppress requests' overly strict version check for chardet/urllib3.
    # requests 2.32.x caps chardet < 6.0.0 but pysrt pulls in chardet 7.x
    # which works fine — the check is just too conservative.
    # IMPORTANT: Must use message-based filter, NOT category-based.
    # Importing RequestsDependencyWarning triggers `import requests`
    # which fires the very warning we're trying to suppress.
    warnings.filterwarnings(
        "ignore",
        message=r".*urllib3.*or chardet.*doesn't match a supported version",
    )


def ensure_numba_cache_dir() -> None:
    """Ensure numba's JIT cache directory is writable.

    numba caches compiled functions in ``__pycache__`` directories inside
    installed packages (e.g. ``site-packages/librosa/__pycache__/``).
    On conda-constructor installs the site-packages directory may not be
    writable by the current user, causing numba to hang indefinitely
    when it tries to verify write access.  (#267)

    Fix: redirect the cache to a user-writable location via the
    ``NUMBA_CACHE_DIR`` environment variable.  If the user has already
    set this variable, we respect their choice.
    """
    import os
    if "NUMBA_CACHE_DIR" in os.environ:
        return

    # Choose a platform-appropriate user-writable directory
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get("XDG_CACHE_HOME", os.path.join(os.path.expanduser("~"), ".cache"))

    cache_dir = os.path.join(base, "whisperjav", "numba_cache")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = cache_dir
    except OSError:
        pass  # Best effort — if we can't create it, let numba try its default


def setup_console() -> None:
    """
    Complete console setup for WhisperJAV.

    Combines UTF-8 encoding fix, warning suppression, and environment
    setup.  Call this at the start of any entry point.

    Example:
        from whisperjav.utils.console import setup_console
        setup_console()

        # Now safe to import other modules
        from whisperjav.main import main
    """
    ensure_utf8_console()
    suppress_dependency_warnings()
    ensure_numba_cache_dir()


def check_extras_installed(extra_name: str, required_packages: list[str]) -> tuple[bool, list[str]]:
    """
    Check if packages for an extra are installed.

    Args:
        extra_name: Name of the extra (for error messages)
        required_packages: List of package names to check

    Returns:
        Tuple of (all_installed: bool, missing_packages: list[str])

    Example:
        installed, missing = check_extras_installed("gui", ["pywebview", "pythonnet"])
        if not installed:
            print(f"Missing packages: {missing}")
            print(f"Install with: pip install whisperjav[gui]")
    """
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)

    return len(missing) == 0, missing


def print_missing_extra_error(
    extra_name: str,
    missing_packages: list[str],
    feature_description: str = ""
) -> None:
    """
    Print a helpful error message for missing extra dependencies.

    Args:
        extra_name: Name of the extra (e.g., "gui", "translate")
        missing_packages: List of missing package names
        feature_description: Optional description of what the feature does

    Example:
        print_missing_extra_error(
            "gui",
            ["pywebview", "pythonnet"],
            "PyWebView GUI interface"
        )
    """
    print()
    print("=" * 60)
    print(f"  Missing Dependencies: [{extra_name}]")
    print("=" * 60)

    if feature_description:
        print(f"Feature: {feature_description}")
        print()

    print(f"Missing packages: {', '.join(missing_packages)}")
    print()
    print("To install, run:")
    print(f"  pip install whisperjav[{extra_name}]")
    print()
    print("Or install all features:")
    print("  pip install whisperjav[all]")
    print("=" * 60)
    print()


# Module-level exports
__all__ = [
    "relaunch_for_utf8",
    "ensure_utf8_console",
    "safe_print",
    "suppress_dependency_warnings",
    "setup_console",
    "check_extras_installed",
    "print_missing_extra_error",
]
