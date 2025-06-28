from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisperjav",
    version="1.0.0",
    author="MeiZhong",
    description="Japanese Adult Video Subtitle Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/meizhong986/WhisperJAV",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "openai-whisper",
        "stable-ts>=2.11.0", 
        "faster-whisper>=1.1.0",
        "torch>=2.0.0",
        "torchaudio",
        "ffmpeg-python",
        "soundfile",
        "auditok",
        "numpy",
        "scipy",
        "tqdm",
        "pysrt",
        "srt",
    ],
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
)
