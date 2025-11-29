from setuptools import setup, find_packages

# Read version from __version__.py without importing the package
version_file = {}
with open("whisperjav/__version__.py") as fp:
    exec(fp.read(), version_file)
__version__ = version_file['__version__']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Relaxed Python version requirement (allow 3.9 to 3.12)
python_requires = ">=3.9,<3.13"

# ALL dependencies merged into a single flat list
# Note: Platform-specific markers (win32) are preserved so this setup.py 
# remains compatible with Linux/Mac without crashing.
install_requires = [
    # Core Dependencies
    "openai-whisper @ git+https://github.com/openai/whisper@main",
    "stable-ts @ git+https://github.com/meizhong986/stable-ts-fix-setup.git@main",
    "faster-whisper>=1.1.0",
    "ffmpeg-python",
    "soundfile",
    "auditok",
    "numpy",
    "scipy",
    "tqdm",
    "pysrt",
    "srt",
    "aiofiles",
    "jsonschema",
    "Pillow",
    "colorama",
    "librosa",
    "matplotlib",
    "pyloudnorm",
    "requests",
    "PySubtrans>=0.7.0",
    "openai>=1.35.0",
    "google-genai>=1.39.0",
    "huggingface-hub>=0.25.0",
    "silero-vad>=6.0",

    # Configuration system
    "pydantic>=2.0,<3.0",
    "PyYAML>=6.0",

    # GUI Dependencies (Previously in 'gui' extra)
    "pywebview>=5.0.0",
    "pythonnet>=3.0; sys_platform=='win32'", # Only installs on Windows
    "pywin32>=305; sys_platform=='win32'",   # Only installs on Windows

    # Speedup Dependencies (Previously in 'speedup' extra)
    "numba",
]

# Classifiers for supported Python versions
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
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
    # extras_require removed as they are now default
    entry_points={
        "console_scripts": [
            "whisperjav=whisperjav.main:main",
            "whisperjav-gui=whisperjav.webview_gui.main:main",
            "whisperjav-translate=whisperjav.translate.cli:main",
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