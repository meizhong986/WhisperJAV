from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="whisperjav",
    version="1.0.0",
    author="WhisperJAV Team",
    description="Japanese Adult Video Subtitle Generator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/whisperjav",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "stable-ts>=2.0.0",
        "pysrt>=1.1.2",
        "numpy>=1.24.0",
        "requests>=2.31.0",
    ],
    entry_points={
        "console_scripts": [
            "whisperjav=whisperjav.main:main",
        ],
    },
)
