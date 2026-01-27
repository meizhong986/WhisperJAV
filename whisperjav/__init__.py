"""
WhisperJAV - Japanese Adult Video Subtitle Generator with AI-powered transcription.

WhisperJAV provides automatic transcription and subtitle generation optimized
for Japanese audio, using OpenAI Whisper and custom enhancements.

Installation
------------
WhisperJAV uses modular extras to allow customized installation:

    pip install whisperjav              # Core only (minimal)
    pip install whisperjav[cli]         # CLI with audio processing
    pip install whisperjav[gui]         # GUI support (Windows)
    pip install whisperjav[translate]   # AI subtitle translation
    pip install whisperjav[all]         # Everything
    pip install whisperjav[colab]       # Google Colab optimized

Available Extras
----------------
cli
    Audio processing, VAD, scene detection, and analysis tools.
    Includes: numpy, scipy, librosa, silero-vad, numba, scikit-learn

gui
    PyWebView GUI interface.
    Windows: WebView2 backend (pythonnet, pywin32)
    Linux/macOS: WebKit backend

translate
    AI-powered subtitle translation using PySubtrans.
    Includes: pysubtrans, openai, google-genai

llm
    Local LLM server for offline translation.
    Includes: uvicorn, fastapi, sse-starlette
    Note: llama-cpp-python must be installed separately with CUDA support.

enhance
    Speech enhancement (denoising, vocal isolation).
    Includes: modelscope, clearvoice, bs-roformer-infer

huggingface
    HuggingFace Transformers integration.
    Includes: transformers, accelerate, huggingface-hub

analysis
    Scientific analysis and visualization.
    Includes: scikit-learn, matplotlib, Pillow

compatibility
    pyvideotrans compatibility layer.
    Includes: av, imageio, httpx, websockets

all
    All of the above combined.

colab / kaggle
    Optimized for notebook environments (no GUI).
    Equivalent to: cli + translate + huggingface

Quick Start
-----------
CLI usage::

    whisperjav video.mp4 --mode balanced

GUI usage::

    whisperjav-gui

Translation::

    whisperjav-translate -i subtitles.srt --provider deepseek

API Usage
---------
For programmatic use::

    from whisperjav.pipelines.faster_pipeline import FasterPipeline

    pipeline = FasterPipeline()
    result = pipeline.transcribe("video.mp4")

For more information, see: https://github.com/meizhong986/whisperjav
"""

from whisperjav.__version__ import __version__, __version_info__

# TODO: Implement lazy imports for GUI startup performance
# Current Issue: These imports load torch, stable_whisper, librosa at import time,
# causing ~6-10 second delay when starting the GUI (which doesn't need these).
# Solution: Use __getattr__ for lazy loading or move to explicit imports in code.
# See: https://peps.python.org/pep-0562/ for module __getattr__
# Priority: Medium - affects user experience but not functionality

# Public API exports
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.utils.logger import setup_logger


__all__ = [
    # Version
    "__version__",
    "__version_info__",
    # Pipelines
    "FasterPipeline",
    "FastPipeline",
    "FidelityPipeline",
    # Utilities
    "MediaDiscovery",
    "setup_logger",
]


# Don't import main at top level - avoid premature loading
def _run():
    from .main import main
    main()


if __name__ == "__main__":
    _run()
