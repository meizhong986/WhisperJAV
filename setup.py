from setuptools import setup, find_packages
import sys

# Block installation on Python 3.13+
if sys.version_info >= (3, 13):
    raise RuntimeError(
        "whisperjav doesn't support Python 3.13 yet. "
        "Please use Python 3.8-3.12."
    )

# Read version from __version__.py without importing the package
version_file = {}
with open("whisperjav/__version__.py") as fp:
    exec(fp.read(), version_file)
__version__ = version_file['__version__']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()



setup(
    name="whisperjav",
    version=__version__,  # using simple version
    author="MeiZhong",
    description="Japanese Adult Video Subtitle Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meizhong986/WhisperJAV",
    license="MIT",  # Modern SPDX
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8,<3.13",
    install_requires=[
        "openai-whisper @ git+https://github.com/openai/whisper@v20250625",
        "stable-ts>=2.11.0", 
        "faster-whisper>=1.1.0",
        "torch>=2.0.0",
        "torchaudio",
        "ffmpeg-python",
        "soundfile",
        "auditok",
        "numpy<2.0",
        "scipy<2.0",
        "tqdm<5.0",
        "pysrt",
        "srt",
    ],
    # Removed setup_requires for setuptools_scm
    entry_points={
        "console_scripts": [
            "whisperjav=whisperjav.main:main",
            "whisperjav-gui=whisperjav.gui.whisperjav_gui:main",
        ],
    },
    include_package_data=True,
    package_data={
        "whisperjav": ["config/*.json"],
    },
    zip_safe=False,
    ext_modules=[],
)