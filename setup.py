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
    "numpy>=2.0",          # NumPy 2.x (modelscope/zipenhancer compatible)
    "scipy>=1.14.0",       # Required for NumPy 2.0 compatibility (ABI change)
    "tqdm",
    "pysrt",
    "srt",
    "aiofiles",
    "jsonschema",
    "colorama",
    "librosa>=0.11.0",        # v0.11.0+ supports NumPy 2.0
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

    # Local LLM translation (v1.8.0)
    # Using JamePeng's fork for better CUDA support and active maintenance
    # [server] extra includes uvicorn for local LLM server
    # See: https://github.com/JamePeng/llama-cpp-python
    "llama-cpp-python[server] @ git+https://github.com/JamePeng/llama-cpp-python.git",

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
    "clearvoice @ git+https://github.com/meizhong986/ClearerVoice-Studio.git#subdirectory=clearvoice",  # Fork with NumPy 2.x support
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
    "numba>=0.60.0",  # NumPy 2.0 compatible
    "hf_xet",  # Faster HuggingFace downloads (Xet Storage)

    # Process Management (v1.7.4)
    "psutil>=5.9.0",  # Process tree termination for clean subprocess cleanup

    # Semantic Audio Clustering (v1.7.4)
    "scikit-learn>=1.5.0",  # Agglomerative clustering for texture-based scene detection (NumPy 2.0 compatible)
]

# Classifiers for supported Python versions (3.10+ only)
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Operating System :: OS Independent",
]

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