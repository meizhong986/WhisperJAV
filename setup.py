from setuptools import setup, find_packages

# Read version from __version__.py without importing the package
version_file = {}
with open("whisperjav/__version__.py") as fp:
    exec(fp.read(), version_file)
__version__ = version_file['__version__']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Python version requirement (3.10+ required for pysubtrans translation dependency)
# See: https://github.com/meizhong986/WhisperJAV/issues/127
python_requires = ">=3.10,<3.13"

# ALL dependencies merged into a single flat list
# Note: Platform-specific markers (win32) are preserved so this setup.py 
# remains compatible with Linux/Mac without crashing.
install_requires = [
    # Core Dependencies
    "openai-whisper @ git+https://github.com/openai/whisper@main",
    "stable-ts @ git+https://github.com/meizhong986/stable-ts-fix-setup.git@main",
    "faster-whisper>=1.1.0",
    "ffmpeg-python @ git+https://github.com/kkroening/ffmpeg-python.git",  # PyPI tarball fails
    "soundfile",
    "auditok",
    "pydub",
    "numpy>=1.26.0,<2.0",  # NumPy 1.26.x for pyvideotrans compatibility (v1.8.0)
    "scipy>=1.10.1",       # Compatible with NumPy 1.26.x
    "tqdm",
    "pysrt",
    "srt",
    "aiofiles",
    "jsonschema",
    "colorama",
    "librosa>=0.10.0",        # v0.10.0+ works with NumPy 1.26; ClearVoice fork supports >=0.11.0
    "pyloudnorm",
    "requests",
    "regex",
    "pysubtrans>=1.5.0",  # Lowercase package name (PyPI), requires Python 3.10+
    "openai>=1.35.0",
    "google-genai>=1.39.0",
    "huggingface-hub>=0.25.0",
    "transformers>=4.40.0",  # HuggingFace Transformers for ASR pipeline
    "accelerate>=0.26.0",    # Efficient model loading for Transformers
    "silero-vad>=6.0",

    # NOTE: llama-cpp-python moved to [local-llm] extra (v1.8.0)
    # Building from source takes 7+ minutes on Colab/Linux
    # Install separately: pip install whisperjav[local-llm]

    # Speech segmentation backends (v1.7.2)
    "ten-vad",
    # NOTE: nemo_toolkit removed - causes resolution-too-deep errors, only 10% usage

    # Speech enhancement backends (v1.7.3)
    "modelscope>=1.20",       # ZipEnhancer (recommended, lightweight SOTA)
    "addict",                 # ModelScope dependency (dict with attribute access)
    "datasets>=2.14.0,<4.0",  # 4.x breaks modelscope
    "simplejson",             # ModelScope dependency (JSON parsing)
    "sortedcontainers",       # ModelScope dependency (sorted collections)
    "packaging",              # ModelScope dependency (version parsing)
    "clearvoice @ git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice",  # Fork with relaxed librosa constraint (>=0.11.0)
    "bs-roformer-infer",      # BS-RoFormer vocal isolation (44.1kHz)
    "onnxruntime>=1.16.0",    # ONNX inference for ZipEnhancer ONNX mode

    # Configuration system
    "pydantic>=2.0,<3.0",
    "PyYAML>=6.0",

    # GUI Dependencies (Previously in 'gui' extra)
    "pywebview>=5.0.0",
    "pythonnet>=3.0; sys_platform=='win32'", # Only installs on Windows
    "pywin32>=305; sys_platform=='win32'",   # Only installs on Windows

    # Speedup Dependencies (Previously in 'speedup' extra)
    "numba>=0.58.0",  # Supports NumPy 1.22-2.0
    "hf_xet",  # Faster HuggingFace downloads (Xet Storage)

    # Process Management (v1.7.4)
    "psutil>=5.9.0",  # Process tree termination for clean subprocess cleanup

    # Semantic Audio Clustering (v1.7.4)
    "scikit-learn>=1.3.0",  # Agglomerative clustering for texture-based scene detection

    # pyvideotrans compatibility prep (v1.8.0 Phase 1)
    # Pre-loading non-conflicting deps for future pyvideotrans integration
    "av>=13.0.0",             # Video container handling (PyAV)
    "imageio>=2.31.0",        # Image/video I/O
    "imageio-ffmpeg>=0.4.9",  # FFmpeg backend for imageio
    "httpx>=0.27.0",          # Modern async HTTP client
    "websockets>=13.0",       # WebSocket support for streaming APIs
    "soxr>=0.3.0",            # High-quality audio resampling (libsoxr bindings)
]

# Classifiers for supported Python versions (3.10+ only)
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
]

# Optional dependencies (extras)
# Install with: pip install whisperjav[local-llm]
extras_require = {
    # Local LLM translation - builds from source, takes 7+ minutes
    # Using JamePeng's fork for better CUDA support and active maintenance
    "local-llm": [
        "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git",
    ],
}

setup(
    name="whisperjav",
    version=__version__,
    author="MeiZhong",
    description="Japanese Adult Video Subtitle Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meizhong986/WhisperJAV",
    license="MIT",
    packages=find_packages(),
    classifiers=classifiers,
    python_requires=python_requires,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "whisperjav=whisperjav.main:main",
            "whisperjav-gui=whisperjav.webview_gui.main:main",
            "whisperjav-translate=whisperjav.translate.cli:main",
            "whisperjav-upgrade=whisperjav.upgrade:main",
        ],
    },
    include_package_data=True,
    package_data={
        "whisperjav": [
            "config/*.json",
            "config/v4/**/*.yaml",
            "instructions/*.txt",
            "translate/defaults/*.txt",
            "webview_gui/assets/*.html",
            "webview_gui/assets/*.css",
            "webview_gui/assets/*.js",
            "Menu/*.json",
        ],
    },
    zip_safe=False,
    ext_modules=[],
)