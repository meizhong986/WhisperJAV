"""
Import Scanner
==============

Scans code for imports not covered by the package registry.

WHY THIS CHECK:
--------------
"Ghost dependencies" are a common problem:
1. Developer adds import in code
2. Package is installed in their dev environment
3. Code works locally
4. User installs WhisperJAV
5. Import fails because package isn't declared

This scanner catches ghost dependencies by:
1. Scanning all .py files for import statements
2. Extracting top-level module names
3. Cross-referencing with registry
4. Reporting imports not in registry

IMPORT NAME MAPPING:
-------------------
Some packages have different pip and import names:
- opencv-python → cv2
- Pillow → PIL
- scikit-learn → sklearn
- PyYAML → yaml

The registry has an 'import_name' field for these mappings (Gemini feedback).
This scanner uses that mapping to avoid false positives.

Usage:
    from whisperjav.installer.validation.imports import scan_imports

    warnings = scan_imports()
    for warning in warnings:
        print(f"  Warning: {warning}")

Author: Senior Architect
Date: 2026-01-26
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Standard library modules (not external packages)
# This is a subset - we add to it as we scan
STDLIB_MODULES: Set[str] = {
    # Built-in modules
    "__future__", "abc", "aifc", "argparse", "array", "ast", "asynchat",
    "asyncio", "asyncore", "atexit", "audioop", "base64", "bdb", "binascii",
    "binhex", "bisect", "builtins", "bz2", "calendar", "cgi", "cgitb",
    "chunk", "cmath", "cmd", "code", "codecs", "codeop", "collections",
    "colorsys", "compileall", "concurrent", "configparser", "contextlib",
    "contextvars", "copy", "copyreg", "cProfile", "crypt", "csv", "ctypes",
    "curses", "dataclasses", "datetime", "dbm", "decimal", "difflib", "dis",
    "distutils", "doctest", "email", "encodings", "enum", "errno", "faulthandler",
    "fcntl", "filecmp", "fileinput", "fnmatch", "fractions", "ftplib", "functools",
    "gc", "getopt", "getpass", "gettext", "glob", "graphlib", "grp", "gzip",
    "hashlib", "heapq", "hmac", "html", "http", "idlelib", "imaplib", "imghdr",
    "imp", "importlib", "inspect", "io", "ipaddress", "itertools", "json",
    "keyword", "lib2to3", "linecache", "locale", "logging", "lzma", "mailbox",
    "mailcap", "marshal", "math", "mimetypes", "mmap", "modulefinder", "multiprocessing",
    "netrc", "nis", "nntplib", "numbers", "operator", "optparse", "os", "ossaudiodev",
    "pathlib", "pdb", "pickle", "pickletools", "pipes", "pkgutil", "platform",
    "plistlib", "poplib", "posix", "posixpath", "pprint", "profile", "pstats",
    "pty", "pwd", "py_compile", "pyclbr", "pydoc", "queue", "quopri", "random",
    "re", "readline", "reprlib", "resource", "rlcompleter", "runpy", "sched",
    "secrets", "select", "selectors", "shelve", "shlex", "shutil", "signal",
    "site", "smtpd", "smtplib", "sndhdr", "socket", "socketserver", "spwd",
    "sqlite3", "ssl", "stat", "statistics", "string", "stringprep", "struct",
    "subprocess", "sunau", "symtable", "sys", "sysconfig", "syslog", "tabnanny",
    "tarfile", "telnetlib", "tempfile", "termios", "test", "textwrap", "threading",
    "time", "timeit", "tkinter", "token", "tokenize", "tomllib", "trace", "traceback",
    "tracemalloc", "tty", "turtle", "turtledemo", "types", "typing", "unicodedata",
    "unittest", "urllib", "uu", "uuid", "venv", "warnings", "wave", "weakref",
    "webbrowser", "winreg", "winsound", "wsgiref", "xdrlib", "xml", "xmlrpc",
    "zipapp", "zipfile", "zipimport", "zlib", "zoneinfo",
    # Type checking
    "typing_extensions",
    # Common internal modules
    "_thread", "_collections", "_functools", "_io", "_weakref",
    # Platform-specific stdlib (Windows)
    "msvcrt", "winreg", "winsound", "_winapi",
    # Platform-specific stdlib (Unix)
    "fcntl", "grp", "pwd", "termios", "tty", "pty", "resource",
}


def _find_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Could not find project root")


def _get_import_map() -> Dict[str, str]:
    """
    Get mapping from import names to package names.

    WHY:
    Some packages have different pip and import names.
    This builds a reverse mapping from the registry.

    Returns:
        Dict mapping import name to package name
    """
    from ..core.registry import get_import_map
    return get_import_map()


def _extract_imports(source: str) -> Set[str]:
    """
    Extract top-level import names from Python source.

    WHY AST PARSING:
    - More reliable than regex
    - Handles all import forms correctly
    - Ignores imports in strings/comments

    Args:
        source: Python source code

    Returns:
        Set of top-level module names
    """
    imports = set()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # Get top-level module (e.g., "torch" from "torch.nn")
                top_level = alias.name.split(".")[0]
                imports.add(top_level)

        elif isinstance(node, ast.ImportFrom):
            # Skip relative imports (from . import X, from .foo import X)
            # These are internal module imports, not external packages
            if node.level > 0:
                continue
            if node.module:
                top_level = node.module.split(".")[0]
                imports.add(top_level)

    return imports


# Internal module names that look like external imports but are actually
# internal whisperjav submodules. These are imported without the full
# whisperjav.* prefix due to Python's import system.
KNOWN_INTERNAL_MODULES: Set[str] = {
    # Common internal module names used in whisperjav
    "api", "base", "base_registry", "builder", "components", "core",
    "factory", "helpers", "loader", "manager", "models", "pipeline",
    "registry", "resolver", "schemas", "settings", "tools", "utils",
    # WebView GUI internals
    "assets", "main", "webview_gui",
    # Config internals
    "v4", "v3", "v2", "v1", "ecosystems", "registries", "presets",
    # Speech enhancement internals
    "backends", "speech_enhancement",
    # Translation internals
    "translate", "providers", "instructions",
}


# Optional/environment-specific imports that are legitimately not in the registry
# These are typically inside try/except blocks for optional functionality
OPTIONAL_IMPORTS: Set[str] = {
    # Notebook environments (try/except guarded)
    "IPython",        # Jupyter/IPython detection
    "google",         # Google Colab detection (google.colab)

    # Alternative GUI frameworks (try/except guarded)
    "PyQt5",          # Optional Qt GUI

    # .NET interop (try/except guarded, Windows-only)
    "clr",            # pythonnet CLR module

    # NVIDIA/CUDA detection (try/except guarded)
    "nvidia",         # nvidia-ml-py for GPU detection
    "pynvml",         # Alternative NVIDIA ML bindings

    # Local LLM (installed separately with CUDA detection)
    "llama_cpp",      # llama-cpp-python is installed separately

    # Optional speech backends (try/except guarded)
    "nemo",           # NVIDIA NeMo speech
    "whisperx",       # WhisperX alternative

    # NeMo dependencies (optional)
    "omegaconf",      # OmegaConf for NeMo config
    "wget",           # wget for NeMo model downloads

    # Voice classifier dependencies (optional)
    "pyannote",       # pyannote-audio for speaker diarization

    # Python version backports (try/except guarded)
    "tomli",          # tomllib backport for Python < 3.11
}


def scan_imports(
    source_dir: Path = None,
    exclude_patterns: List[str] = None,
) -> List[str]:
    """
    Scan code for imports not covered by registry.

    This catches "ghost dependencies" - imports that work in dev
    but fail for users because the package isn't declared.

    WHY THIS IS A WARNING NOT ERROR:
    - Some imports may be optional (inside try/except)
    - Standard library detection isn't perfect
    - Better to warn than block legitimate code

    Args:
        source_dir: Directory to scan (default: whisperjav/)
        exclude_patterns: Patterns to exclude (default: tests, __pycache__)

    Returns:
        List of warning messages (imports not in registry)
    """
    if source_dir is None:
        try:
            source_dir = _find_project_root() / "whisperjav"
        except FileNotFoundError as e:
            return [str(e)]

    if not source_dir.exists():
        return [f"Source directory not found: {source_dir}"]

    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", "test_*", "*_test.py"]

    # Get import map from registry
    import_map = _get_import_map()

    # Get all registered import names
    registered_imports = set(import_map.keys())

    # Add the whisperjav package itself
    registered_imports.add("whisperjav")

    warnings = []
    untracked_imports: Dict[str, List[str]] = {}  # import -> files that use it

    # Scan all Python files
    for py_file in source_dir.rglob("*.py"):
        # Skip excluded patterns
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        try:
            source = py_file.read_text(encoding="utf-8")
        except Exception:
            continue

        # Extract imports
        file_imports = _extract_imports(source)

        # Check each import
        for imp in file_imports:
            # Skip stdlib
            if imp in STDLIB_MODULES:
                continue

            # Skip registered packages
            if imp in registered_imports:
                continue

            # Skip internal modules (relative imports resolve to full paths)
            if imp.startswith("_"):
                continue

            # Skip known internal module names
            # These are internal whisperjav submodules that may be imported
            # without the full whisperjav.* prefix in some contexts
            if imp in KNOWN_INTERNAL_MODULES:
                continue

            # Skip known optional imports
            # These are legitimately not in the registry - they're inside
            # try/except blocks for optional functionality
            if imp in OPTIONAL_IMPORTS:
                continue

            # Track untracked import
            rel_path = str(py_file.relative_to(source_dir.parent))
            if imp not in untracked_imports:
                untracked_imports[imp] = []
            if rel_path not in untracked_imports[imp]:
                untracked_imports[imp].append(rel_path)

    # Generate warnings
    for imp, files in sorted(untracked_imports.items()):
        if len(files) > 3:
            file_list = ", ".join(files[:3]) + f" (+{len(files) - 3} more)"
        else:
            file_list = ", ".join(files)
        warnings.append(f"Untracked import '{imp}' in: {file_list}")

    return warnings


__all__ = [
    "scan_imports",
    "STDLIB_MODULES",
    "KNOWN_INTERNAL_MODULES",
    "OPTIONAL_IMPORTS",
]
