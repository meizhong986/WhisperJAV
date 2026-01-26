"""
Package Registry - Single Source of Truth
==========================================

INSTITUTIONAL KNOWLEDGE - THIS FILE IS CRITICAL

This module defines ALL packages required by WhisperJAV. It is the SINGLE
SOURCE OF TRUTH for package definitions. All other files (pyproject.toml,
requirements.txt, install scripts) should be derived from this registry.

WHY THIS FILE EXISTS:
--------------------
Before this refactor, package definitions were scattered across:
- pyproject.toml (11 extras, ~80 packages)
- install_windows.bat (pip install commands)
- install_linux.sh (pip install commands)
- post_install.py.template (requirements section)
- requirements.txt.template (standalone installer)

This duplication caused:
1. Version drift (one file updated, others not)
2. Missing packages (added to pyproject.toml, forgotten in .bat)
3. "Ghost dependencies" (imports that work locally but fail for users)
4. Maintenance burden (4+ places to update for each change)

USAGE:
-----
1. Adding a new dependency:
   - Add Package(...) to PACKAGES list
   - Run: python -m whisperjav.installer.validation
   - Commit both registry.py and any generated changes

2. Removing a dependency:
   - Remove from PACKAGES
   - Run validation
   - Commit

3. Changing version:
   - Update version field in PACKAGES
   - Run validation
   - Commit

INSTALLATION ORDER (by order field):
-----------------------------------
The order field determines installation sequence. This is CRITICAL because:
- PyTorch MUST be installed first with --index-url for GPU support
- If Whisper (which depends on torch) installs first, pip gets CPU torch
- Scientific stack (numpy, numba) has binary compatibility requirements

Order ranges:
  10-19: PyTorch ecosystem (MUST BE FIRST - GPU lock-in)
  20-29: Scientific stack (numpy before numba - binary compat)
  30-39: Whisper packages (depend on torch being present)
  40-49: Audio/CLI packages
  50-59: GUI packages
  60-69: Translation packages
  70-79: Enhancement packages
  80-89: HuggingFace/optional
  90-99: Compatibility/dev

IMPORT_NAME FIELD (from Gemini Review):
--------------------------------------
Some packages have different pip names vs import names:
- opencv-python → cv2
- Pillow → PIL
- scikit-learn → sklearn

The import_name field enables the Import Scanner to correctly map imports
to packages, preventing false "ghost dependency" warnings.

Author: Senior Architect
Date: 2026-01-26
Gemini Review: 2026-01-26 (added import_name requirement)
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Set
import sys


# =============================================================================
# Enums
# =============================================================================


class Extra(Enum):
    """
    Maps to pyproject.toml [project.optional-dependencies] keys.

    WHY ENUMS:
    - Type safety (can't misspell "cli" as "cil")
    - IDE autocomplete
    - Easy iteration for validation

    NAMING:
    - Values match pyproject.toml keys exactly (lowercase)
    - CORE is special - these are base dependencies, not in extras
    """
    CORE = "core"           # Required for basic operation (in dependencies, not extras)
    CLI = "cli"             # Audio processing, VAD, scene detection
    GUI = "gui"             # PyWebView GUI
    TRANSLATE = "translate" # AI translation (pysubtrans, OpenAI, Gemini)
    LLM = "llm"             # Local LLM server (FastAPI, uvicorn)
    ENHANCE = "enhance"     # Speech enhancement (ClearVoice, BS-RoFormer)
    HUGGINGFACE = "huggingface"  # HuggingFace Transformers
    ANALYSIS = "analysis"   # Scientific analysis and visualization
    COMPATIBILITY = "compatibility"  # pyvideotrans compatibility
    DEV = "dev"             # Development dependencies


class InstallSource(Enum):
    """
    Where to install a package from.

    WHY THIS MATTERS:
    Different sources require different pip arguments:
    - PYPI: pip install <package>
    - GIT: pip install git+<url>
    - INDEX_URL: pip install <package> --index-url <url>
    - WHEEL_URL: pip install <direct-wheel-url>
    """
    PYPI = auto()           # Standard PyPI (default)
    GIT = auto()            # Git repository
    INDEX_URL = auto()      # PyPI-like index (e.g., PyTorch CUDA wheels)
    WHEEL_URL = auto()      # Direct wheel URL (e.g., HuggingFace llama-cpp)


class Platform(Enum):
    """
    Platform filter for packages.

    WHY PLATFORM FILTERING:
    Some packages are Windows-only (pywin32, pythonnet) or have
    platform-specific builds. Rather than installing and failing,
    we filter at the registry level.
    """
    ALL = "all"             # Install on all platforms
    WINDOWS = "windows"     # Windows only (sys.platform == 'win32')
    LINUX = "linux"         # Linux only (sys.platform.startswith('linux'))
    MACOS = "macos"         # macOS only (sys.platform == 'darwin')
    MACOS_SILICON = "macos_silicon"  # Apple Silicon only


# =============================================================================
# Package Dataclass
# =============================================================================


@dataclass
class Package:
    """
    A package dependency with full installation metadata.

    This dataclass captures everything needed to:
    1. Install the package correctly
    2. Validate it against pyproject.toml
    3. Generate requirements.txt
    4. Map imports to packages (for Import Scanner)

    REQUIRED FIELDS:
    ---------------
    - name: PyPI package name (e.g., "torch", "numpy")

    OPTIONAL FIELDS WITH DEFAULTS:
    -----------------------------
    - version: Version specifier (e.g., ">=1.26.0,<2.0")
    - extra: Which pyproject.toml extra this belongs to
    - source: Where to install from
    - order: Installation priority (lower = earlier)

    SOURCE-SPECIFIC FIELDS:
    ----------------------
    - git_url: For GIT source (e.g., "git+https://github.com/...")
    - index_url: For INDEX_URL source (pattern with {cuda} placeholder)
    - wheel_url: For WHEEL_URL source
    - hash: SHA-256 hash for security verification

    FILTERING:
    ---------
    - platforms: List of platforms, or None for all
    - required: If False, failure is non-fatal

    IMPORT MAPPING (from Gemini Review):
    -----------------------------------
    - import_name: If different from pip name (e.g., "cv2" for "opencv-python")

    DOCUMENTATION:
    -------------
    - reason: Why this package is needed (for humans and future developers)
    """
    # REQUIRED
    name: str

    # OPTIONAL - Version and categorization
    version: str = ""
    extra: Extra = Extra.CORE
    source: InstallSource = InstallSource.PYPI
    order: int = 100  # Default: install late

    # SOURCE-SPECIFIC
    git_url: Optional[str] = None
    index_url: Optional[str] = None
    wheel_url: Optional[str] = None
    hash: Optional[str] = None  # SHA-256 for security verification

    # FILTERING
    platforms: Optional[List[Platform]] = None  # None = all platforms
    required: bool = True  # If False, failure doesn't stop installation

    # IMPORT MAPPING (Gemini Review feedback)
    # WHY: Import Scanner needs to know that "import cv2" → "opencv-python"
    import_name: Optional[str] = None

    # DOCUMENTATION
    reason: str = ""  # WHY this package is needed

    def pyproject_spec(self) -> str:
        """
        Generate pyproject.toml dependency specification.

        Returns string like:
        - "numpy>=1.26.0,<2.0"
        - "openai-whisper @ git+https://github.com/openai/whisper@main"
        - "pywin32>=305; sys_platform == 'win32'"
        """
        # Base specification
        if self.source == InstallSource.GIT:
            spec = f"{self.name} @ {self.git_url}"
        elif self.version:
            spec = f"{self.name}{self.version}"
        else:
            spec = self.name

        # Platform marker
        if self.platforms and Platform.ALL not in self.platforms:
            markers = []
            if Platform.WINDOWS in self.platforms:
                markers.append("sys_platform == 'win32'")
            if Platform.LINUX in self.platforms:
                markers.append("sys_platform.startswith('linux')")
            if Platform.MACOS in self.platforms:
                markers.append("sys_platform == 'darwin'")
            if markers:
                spec = f"{spec}; {' or '.join(markers)}"

        return spec

    def pip_install_args(self, cuda_version: str = "cu118") -> List[str]:
        """
        Generate pip install arguments.

        Args:
            cuda_version: CUDA version for INDEX_URL packages (cu118, cu128, cpu)

        Returns:
            List of arguments for pip install command
        """
        args = []

        if self.source == InstallSource.GIT:
            args.append(self.git_url)
        elif self.source == InstallSource.INDEX_URL:
            index_url = self.index_url.replace("{cuda}", cuda_version)
            args.extend([self.name, "--index-url", index_url])
        elif self.source == InstallSource.WHEEL_URL:
            args.append(self.wheel_url)
        else:  # PYPI
            spec = f"{self.name}{self.version}" if self.version else self.name
            args.append(spec)

        return args

    def matches_platform(self, current_platform: str = None) -> bool:
        """
        Check if this package should be installed on the current platform.

        Args:
            current_platform: Platform string (defaults to sys.platform)

        Returns:
            True if package should be installed
        """
        if self.platforms is None or Platform.ALL in self.platforms:
            return True

        if current_platform is None:
            current_platform = sys.platform

        if Platform.WINDOWS in self.platforms and current_platform == "win32":
            return True
        if Platform.LINUX in self.platforms and current_platform.startswith("linux"):
            return True
        if Platform.MACOS in self.platforms and current_platform == "darwin":
            return True

        return False


# =============================================================================
# Package Registry - THE SINGLE SOURCE OF TRUTH
# =============================================================================
#
# MODIFICATION RULES:
# 1. Add packages in the correct order range
# 2. Always include 'reason' explaining WHY the package is needed
# 3. Set import_name if pip name != import name
# 4. Run validation after changes: python -m whisperjav.installer.validation
#

PACKAGES: List[Package] = [
    # =========================================================================
    # PHASE 2: PyTorch (Order 10-19) - MUST BE FIRST
    # =========================================================================
    #
    # WHY FIRST:
    # PyTorch must be installed with --index-url BEFORE any package that
    # depends on it. Otherwise, pip resolves torch from PyPI (CPU-only).
    #
    # THE GPU LOCK-IN PATTERN:
    # 1. Install torch with --index-url https://download.pytorch.org/whl/cu128
    # 2. torch is now "locked in" with GPU support
    # 3. When openai-whisper installs, it sees torch satisfied, skips reinstall
    # 4. User gets GPU inference (correct)
    #
    # WITHOUT THIS PATTERN:
    # 1. pip install openai-whisper
    # 2. pip resolves torch dependency from PyPI → CPU version
    # 3. User gets CPU inference (6-10x slower)
    #
    Package(
        name="torch",
        extra=Extra.CORE,
        source=InstallSource.INDEX_URL,
        index_url="https://download.pytorch.org/whl/{cuda}",
        order=10,
        required=True,
        reason="Deep learning framework - MUST install first with correct CUDA index",
    ),
    Package(
        name="torchaudio",
        extra=Extra.CORE,
        source=InstallSource.INDEX_URL,
        index_url="https://download.pytorch.org/whl/{cuda}",
        order=11,
        required=True,
        reason="Audio processing for PyTorch - must match torch version",
    ),

    # =========================================================================
    # PHASE 3: Scientific Stack (Order 20-29)
    # =========================================================================
    #
    # WHY ORDER MATTERS:
    # - numpy MUST be installed before numba
    # - numba builds against numpy's ABI; wrong order = import errors
    #
    Package(
        name="numpy",
        version=">=1.26.0,<2.0",
        extra=Extra.CLI,
        order=20,
        required=True,
        reason="NumPy 1.26.x for pyvideotrans compatibility - MUST be before numba",
    ),
    Package(
        name="scipy",
        version=">=1.10.1",
        extra=Extra.CLI,
        order=21,
        reason="Signal processing and scientific computing",
    ),
    Package(
        name="numba",
        version=">=0.58.0",
        extra=Extra.CLI,
        order=22,
        reason="JIT compilation for performance - 0.58.0+ supports NumPy 1.22-2.0",
    ),

    # =========================================================================
    # PHASE 4: Whisper Packages (Order 30-39)
    # =========================================================================
    #
    # WHY AFTER PYTORCH:
    # These packages depend on torch. By installing torch first with GPU,
    # these packages see torch as already satisfied and don't pull CPU torch.
    #
    Package(
        name="openai-whisper",
        extra=Extra.CORE,
        source=InstallSource.GIT,
        git_url="git+https://github.com/openai/whisper@main",
        order=30,
        required=True,
        import_name="whisper",  # pip name != import name
        reason="OpenAI Whisper ASR - main branch for latest fixes",
    ),
    Package(
        name="stable-ts",
        extra=Extra.CORE,
        source=InstallSource.GIT,
        git_url="git+https://github.com/meizhong986/stable-ts-fix-setup.git@main",
        order=31,
        required=True,
        import_name="stable_whisper",  # pip name != import name
        reason="Stable timestamps for Whisper - custom fork with setup.py fixes",
    ),
    Package(
        name="ffmpeg-python",
        extra=Extra.CORE,
        source=InstallSource.GIT,
        git_url="git+https://github.com/kkroening/ffmpeg-python.git",
        order=32,
        reason="FFmpeg bindings - PyPI tarball has build issues, use git",
    ),
    Package(
        name="faster-whisper",
        version=">=1.1.0",
        extra=Extra.CORE,
        order=33,
        required=True,
        reason="CTranslate2-based Whisper for 4x faster inference",
    ),

    # =========================================================================
    # PHASE 5: Audio/CLI Packages (Order 40-49)
    # =========================================================================
    Package(
        name="soundfile",
        extra=Extra.CLI,
        order=40,
        reason="Audio file I/O with libsndfile backend",
    ),
    Package(
        name="pydub",
        extra=Extra.CLI,
        order=41,
        reason="Audio segment manipulation for scene detection",
    ),
    Package(
        name="librosa",
        version=">=0.10.0",
        extra=Extra.CLI,
        order=42,
        reason="Audio analysis (mel spectrograms, etc.) - 0.10+ for numba compat",
    ),
    Package(
        name="pyloudnorm",
        extra=Extra.CLI,
        order=43,
        reason="Audio normalization (EBU R128 loudness)",
    ),
    Package(
        name="auditok",
        extra=Extra.CLI,
        order=44,
        reason="Voice Activity Detection for basic silence splitting",
    ),
    Package(
        name="silero-vad",
        version=">=6.0",
        extra=Extra.CLI,
        order=45,
        reason="Silero VAD for scene detection - 6.0+ has improved accuracy",
    ),
    Package(
        name="ten-vad",
        extra=Extra.CLI,
        order=46,
        reason="TEN Framework VAD for speech segmentation",
    ),
    Package(
        name="psutil",
        version=">=5.9.0",
        extra=Extra.CLI,
        order=47,
        reason="Process management and cleanup (kill stale processes)",
    ),
    Package(
        name="scikit-learn",
        version=">=1.3.0",
        extra=Extra.CLI,
        order=48,
        import_name="sklearn",  # pip name != import name
        reason="Clustering for semantic scene detection",
    ),

    # =========================================================================
    # PHASE 5: GUI Packages (Order 50-59)
    # =========================================================================
    Package(
        name="pywebview",
        version=">=5.0.0",
        extra=Extra.GUI,
        order=50,
        import_name="webview",  # pip name != import name
        reason="Modern web-based GUI framework",
    ),
    Package(
        name="pythonnet",
        version=">=3.0",
        extra=Extra.GUI,
        order=51,
        platforms=[Platform.WINDOWS],
        reason="Required for WebView2 backend on Windows",
    ),
    Package(
        name="pywin32",
        version=">=305",
        extra=Extra.GUI,
        order=52,
        platforms=[Platform.WINDOWS],
        import_name="win32com",  # pip name != import name (also win32api, etc.)
        reason="Windows COM support for desktop shortcuts",
    ),

    # =========================================================================
    # PHASE 5: Translation Packages (Order 60-69)
    # =========================================================================
    Package(
        name="pysubtrans",
        version=">=1.5.0",
        extra=Extra.TRANSLATE,
        order=60,
        import_name="PySubtrans",  # pip name != import name (case-sensitive)
        reason="AI-powered subtitle translation (requires Python 3.10+)",
    ),
    Package(
        name="openai",
        version=">=1.35.0",
        extra=Extra.TRANSLATE,
        order=61,
        reason="GPT provider for pysubtrans",
    ),
    Package(
        name="google-genai",
        version=">=1.39.0",
        extra=Extra.TRANSLATE,
        order=62,
        reason="Gemini provider for pysubtrans",
    ),

    # =========================================================================
    # PHASE 5: LLM Server Packages (Order 63-69)
    # =========================================================================
    #
    # NOTE: llama-cpp-python itself is installed separately with CUDA detection
    # These are just the server dependencies
    #
    Package(
        name="uvicorn",
        version=">=0.22.0",
        extra=Extra.LLM,
        order=63,
        reason="ASGI server for LLM API",
    ),
    Package(
        name="fastapi",
        version=">=0.100.0",
        extra=Extra.LLM,
        order=64,
        reason="API framework for LLM server",
    ),
    Package(
        name="pydantic-settings",
        version=">=2.0.1",
        extra=Extra.LLM,
        order=65,
        reason="Configuration management for LLM server",
    ),
    Package(
        name="sse-starlette",
        version=">=1.6.1",
        extra=Extra.LLM,
        order=66,
        reason="Server-Sent Events for streaming LLM responses",
    ),
    Package(
        name="starlette-context",
        version=">=0.3.6,<0.4",
        extra=Extra.LLM,
        order=67,
        reason="Request context for Starlette - pinned to 0.3.x for compatibility",
    ),

    # =========================================================================
    # PHASE 5: Enhancement Packages (Order 70-79)
    # =========================================================================
    Package(
        name="modelscope",
        version=">=1.20",
        extra=Extra.ENHANCE,
        order=70,
        reason="ModelScope framework for ZipEnhancer speech enhancement",
    ),
    Package(
        name="addict",
        extra=Extra.ENHANCE,
        order=71,
        reason="Dict subclass for modelscope configs",
    ),
    Package(
        name="datasets",
        version=">=2.14.0,<4.0",
        extra=Extra.ENHANCE,
        order=72,
        reason="HuggingFace datasets - pinned <4.0 due to modelscope compat",
    ),
    Package(
        name="simplejson",
        extra=Extra.ENHANCE,
        order=73,
        reason="JSON handling for modelscope",
    ),
    Package(
        name="sortedcontainers",
        extra=Extra.ENHANCE,
        order=74,
        reason="Sorted collections for modelscope",
    ),
    Package(
        name="packaging",
        extra=Extra.ENHANCE,
        order=75,
        reason="Version parsing for modelscope",
    ),
    Package(
        name="clearvoice",
        extra=Extra.ENHANCE,
        source=InstallSource.GIT,
        git_url="git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice",
        order=76,
        reason="ClearVoice speech enhancement - custom fork with relaxed librosa",
    ),
    Package(
        name="bs-roformer-infer",
        extra=Extra.ENHANCE,
        order=77,
        import_name="bs_roformer",  # pip name != import name
        reason="BS-RoFormer vocal isolation",
    ),
    Package(
        name="onnxruntime",
        version=">=1.16.0",
        extra=Extra.ENHANCE,
        order=78,
        reason="ONNX inference for speech enhancement models",
    ),

    # =========================================================================
    # PHASE 5: HuggingFace Packages (Order 80-84)
    # =========================================================================
    Package(
        name="huggingface-hub",
        version=">=0.25.0",
        extra=Extra.HUGGINGFACE,
        order=80,
        reason="HuggingFace cache management for model downloads",
    ),
    Package(
        name="transformers",
        version=">=4.40.0",
        extra=Extra.HUGGINGFACE,
        order=81,
        reason="HuggingFace Transformers for ASR pipeline (kotoba-whisper)",
    ),
    Package(
        name="accelerate",
        version=">=0.26.0",
        extra=Extra.HUGGINGFACE,
        order=82,
        reason="Efficient model loading for Transformers",
    ),
    Package(
        name="hf_xet",
        extra=Extra.HUGGINGFACE,
        order=83,
        reason="Faster HuggingFace downloads via XetHub",
    ),

    # =========================================================================
    # PHASE 5: Utility Packages (Order 85-89)
    # =========================================================================
    Package(
        name="tqdm",
        extra=Extra.CORE,
        order=85,
        reason="Progress bars for long operations",
    ),
    Package(
        name="colorama",
        extra=Extra.CORE,
        order=86,
        reason="Colored console output (cross-platform)",
    ),
    Package(
        name="requests",
        extra=Extra.CORE,
        order=87,
        reason="HTTP requests for model downloads and API calls",
    ),
    Package(
        name="aiofiles",
        extra=Extra.CORE,
        order=88,
        reason="Async file I/O for non-blocking operations",
    ),
    Package(
        name="regex",
        extra=Extra.CORE,
        order=89,
        reason="Advanced regex for Japanese text processing",
    ),

    # =========================================================================
    # PHASE 5: Subtitle Processing (Order 90-91)
    # =========================================================================
    Package(
        name="pysrt",
        extra=Extra.CORE,
        order=90,
        reason="SRT file parsing and manipulation",
    ),
    Package(
        name="srt",
        extra=Extra.CORE,
        order=91,
        reason="SRT utilities (complementary to pysrt)",
    ),

    # =========================================================================
    # PHASE 5: Configuration Packages (Order 92-94)
    # =========================================================================
    Package(
        name="pydantic",
        version=">=2.0,<3.0",
        extra=Extra.CORE,
        order=92,
        reason="Data validation for config schemas - pinned to 2.x",
    ),
    Package(
        name="PyYAML",
        version=">=6.0",
        extra=Extra.CORE,
        order=93,
        import_name="yaml",  # pip name != import name
        reason="YAML config file parsing",
    ),
    Package(
        name="jsonschema",
        extra=Extra.CORE,
        order=94,
        reason="JSON schema validation for config files",
    ),

    # =========================================================================
    # PHASE 5: Analysis Packages (Order 95-97)
    # =========================================================================
    #
    # Note: scikit-learn is in CLI (for scene detection clustering), not ANALYSIS.
    # Users needing clustering for analysis can install both: pip install .[cli,analysis]
    #
    Package(
        name="matplotlib",
        extra=Extra.ANALYSIS,
        order=95,
        reason="Plotting for diagnostic visualizations",
    ),
    Package(
        name="Pillow",
        extra=Extra.ANALYSIS,
        order=96,
        import_name="PIL",  # pip name != import name
        reason="Image processing library",
    ),

    # =========================================================================
    # PHASE 5: Compatibility Packages (Order 98-103)
    # =========================================================================
    #
    # WHY COMPATIBILITY:
    # These packages support pyvideotrans integration and other compatibility
    # features that some users need.
    #
    Package(
        name="av",
        version=">=13.0.0",
        extra=Extra.COMPATIBILITY,
        order=98,
        reason="Video container handling (PyAV) for pyvideotrans",
    ),
    Package(
        name="imageio",
        version=">=2.31.0",
        extra=Extra.COMPATIBILITY,
        order=99,
        reason="Image/video I/O for pyvideotrans",
    ),
    Package(
        name="imageio-ffmpeg",
        version=">=0.4.9",
        extra=Extra.COMPATIBILITY,
        order=100,
        reason="FFmpeg backend for imageio",
    ),
    Package(
        name="httpx",
        version=">=0.27.0",
        extra=Extra.COMPATIBILITY,
        order=101,
        reason="Modern async HTTP client for streaming APIs",
    ),
    Package(
        name="websockets",
        version=">=13.0",
        extra=Extra.COMPATIBILITY,
        order=102,
        reason="WebSocket support for streaming APIs",
    ),
    Package(
        name="soxr",
        version=">=0.3.0",
        extra=Extra.COMPATIBILITY,
        order=103,
        reason="High-quality audio resampling (SoX Resampler)",
    ),

    # =========================================================================
    # PHASE 5: Development Packages (Order 110-119)
    # =========================================================================
    Package(
        name="pytest",
        version=">=7.0",
        extra=Extra.DEV,
        order=110,
        reason="Testing framework",
    ),
    Package(
        name="pytest-cov",
        extra=Extra.DEV,
        order=111,
        reason="Coverage reporting for pytest",
    ),
    Package(
        name="ruff",
        version=">=0.1.0",
        extra=Extra.DEV,
        order=112,
        reason="Fast linter and formatter",
    ),
    Package(
        name="pre-commit",
        extra=Extra.DEV,
        order=113,
        reason="Git hooks for code quality",
    ),
]


# =============================================================================
# Public API Functions
# =============================================================================


def get_packages_in_install_order() -> List[Package]:
    """
    Return all packages sorted by order field.

    WHY SORTING:
    Installation order is critical for GPU PyTorch lock-in.
    See docstring at top of file for explanation.

    Returns:
        List of packages sorted by order (lowest first)
    """
    return sorted(PACKAGES, key=lambda p: p.order)


def get_packages_for_platform(platform: str = None) -> List[Package]:
    """
    Return packages applicable to given platform.

    Args:
        platform: Platform string (win32, linux, darwin) or None for current

    Returns:
        List of packages that should be installed on this platform
    """
    if platform is None:
        platform = sys.platform

    return [p for p in PACKAGES if p.matches_platform(platform)]


def get_packages_by_extra(extra: Extra) -> List[Package]:
    """
    Return packages for given extra.

    Args:
        extra: The Extra enum value

    Returns:
        List of packages belonging to that extra
    """
    return [p for p in PACKAGES if p.extra == extra]


def get_import_map() -> Dict[str, str]:
    """
    Build mapping from import names to package names.

    WHY THIS EXISTS (Gemini Review):
    The Import Scanner needs to know that 'import cv2' corresponds to
    package 'opencv-python'. This function builds that mapping from
    the registry, making it the SINGLE SOURCE OF TRUTH.

    NOTE: Returns both original case AND lowercase versions because
    Python imports are case-sensitive but we want forgiving matching.

    Returns:
        Dict mapping import name to pip package name
    """
    import_map = {}

    for pkg in PACKAGES:
        # Packages with explicit import_name
        if pkg.import_name:
            # Add both original case and lowercase for case-insensitive matching
            import_map[pkg.import_name] = pkg.name
            import_map[pkg.import_name.lower()] = pkg.name

        # Also add normalized package name (replace - with _)
        normalized = pkg.name.lower().replace("-", "_")
        import_map[normalized] = pkg.name

    return import_map


def generate_pyproject_extras() -> Dict[str, List[str]]:
    """
    Generate optional-dependencies for pyproject.toml.

    WHY GENERATE:
    By generating pyproject.toml sections from the registry, we ensure
    they can never drift out of sync. The registry is the source of truth;
    pyproject.toml is a derived artifact.

    Returns:
        Dict mapping extra name to list of dependency specifications
    """
    extras: Dict[str, List[str]] = {}

    for extra in Extra:
        if extra == Extra.CORE:
            continue  # Core deps go in [project].dependencies, not extras

        extra_packages = get_packages_by_extra(extra)
        if extra_packages:
            extras[extra.value] = [pkg.pyproject_spec() for pkg in extra_packages]

    return extras


def generate_core_dependencies() -> List[str]:
    """
    Generate core dependencies for pyproject.toml [project].dependencies.

    Returns:
        List of dependency specifications for core packages
    """
    core_packages = get_packages_by_extra(Extra.CORE)
    return [pkg.pyproject_spec() for pkg in core_packages]


def generate_requirements_txt(include_core: bool = True) -> str:
    """
    Generate requirements.txt content.

    WHY GENERATE:
    The standalone installer uses requirements.txt. By generating it from
    the registry, we ensure consistency with pyproject.toml and install scripts.

    Args:
        include_core: Whether to include core dependencies

    Returns:
        String content for requirements.txt file
    """
    lines = [
        "# WhisperJAV Requirements",
        "# =======================",
        "# Auto-generated from whisperjav/installer/core/registry.py",
        "# DO NOT EDIT MANUALLY - changes will be overwritten",
        "#",
        "# Note: PyTorch is installed separately with CUDA detection",
        "",
    ]

    # Group by extra for readability
    for extra in Extra:
        if extra == Extra.CORE and not include_core:
            continue

        packages = get_packages_by_extra(extra)
        if not packages:
            continue

        lines.append(f"# {extra.value.upper()} packages")
        for pkg in sorted(packages, key=lambda p: p.order):
            # Skip torch/torchaudio - installed separately with CUDA
            if pkg.name in ("torch", "torchaudio"):
                lines.append(f"# {pkg.name} - installed separately with CUDA detection")
                continue

            spec = pkg.pyproject_spec()
            if pkg.reason:
                lines.append(f"{spec}  # {pkg.reason}")
            else:
                lines.append(spec)

        lines.append("")

    return "\n".join(lines)


def get_package_by_name(name: str) -> Optional[Package]:
    """
    Find a package by name.

    Args:
        name: Package name (case-insensitive)

    Returns:
        Package if found, None otherwise
    """
    name_lower = name.lower()
    for pkg in PACKAGES:
        if pkg.name.lower() == name_lower:
            return pkg
    return None


def get_all_package_names() -> Set[str]:
    """
    Get set of all package names (normalized).

    Returns:
        Set of lowercase package names with hyphens replaced by underscores
    """
    names = set()
    for pkg in PACKAGES:
        names.add(pkg.name.lower())
        names.add(pkg.name.lower().replace("-", "_"))
        if pkg.import_name:
            names.add(pkg.import_name.lower())
    return names


# =============================================================================
# Validation
# =============================================================================


def validate_registry():
    """
    Validate registry consistency.

    Checks:
    1. No duplicate package names
    2. All git packages have git_url
    3. All index_url packages have index_url
    4. Order values are unique within same priority range

    Raises:
        ValueError: If validation fails
    """
    # Check for duplicates
    seen_names = set()
    for pkg in PACKAGES:
        if pkg.name.lower() in seen_names:
            raise ValueError(f"Duplicate package: {pkg.name}")
        seen_names.add(pkg.name.lower())

    # Check source-specific fields
    for pkg in PACKAGES:
        if pkg.source == InstallSource.GIT and not pkg.git_url:
            raise ValueError(f"Package {pkg.name} has GIT source but no git_url")
        if pkg.source == InstallSource.INDEX_URL and not pkg.index_url:
            raise ValueError(f"Package {pkg.name} has INDEX_URL source but no index_url")
        if pkg.source == InstallSource.WHEEL_URL and not pkg.wheel_url:
            raise ValueError(f"Package {pkg.name} has WHEEL_URL source but no wheel_url")


# Run validation at import time
validate_registry()
