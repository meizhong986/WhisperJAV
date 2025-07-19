from setuptools import setup, find_packages
import sys

# Read version from __version__.py without importing the package
version_file = {}
with open("whisperjav/__version__.py") as fp:
    exec(fp.read(), version_file)
__version__ = version_file['__version__']

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Relaxed Python version requirement (allow 3.8 to 3.12)
python_requires = ">=3.8,<3.13"

# Relaxed dependency versions to allow flexibility
install_requires = [
    "openai-whisper @ git+https://github.com/openai/whisper@main",  # Use 'main' branch for latest releases
    "stable-ts @ git+https://github.com/meizhong986/stable-ts-fix-setup.git@main",
    "faster-whisper>=1.1.0",
    "torch",  # No version constraint to allow Colab's pre-installed version
    "torchaudio",
    "ffmpeg-python",
    "soundfile",
    "auditok",
    "numpy",  # Removed <2.0 constraint
    "scipy",  # Removed <2.0 constraint
    "tqdm",   # Removed <5.0 constraint
    "pysrt",
    "srt",
]

# Classifiers for supported Python versions
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
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
