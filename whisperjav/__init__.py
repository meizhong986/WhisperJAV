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

# =============================================================================
# LAZY IMPORTS (PEP 562)
# =============================================================================
# Heavy modules (torch, whisper, librosa) are NOT imported at package load time.
# They are loaded on first access via __getattr__. This enables:
# - Fast GUI startup (~1 second instead of ~10 seconds)
# - Lighter memory footprint for utilities that don't need ML
#
# Usage remains unchanged:
#   from whisperjav import FasterPipeline  # Loads on first use
#   pipeline = FasterPipeline()            # Now torch/whisper are loaded
#
# See: https://peps.python.org/pep-0562/
# =============================================================================

from whisperjav.__version__ import __version__, __version_info__

# Lightweight imports only - these don't pull in torch/whisper
# (setup_logger is lazy-loaded too since it may pull in heavy deps indirectly)

__all__ = [
    # Version (always available)
    "__version__",
    "__version_info__",
    # Pipelines (lazy-loaded)
    "FasterPipeline",
    "FastPipeline",
    "FidelityPipeline",
    # Utilities (lazy-loaded)
    "MediaDiscovery",
    "setup_logger",
]

# Mapping of lazy attributes to their import paths
_LAZY_IMPORTS = {
    "FasterPipeline": "whisperjav.pipelines.faster_pipeline",
    "FastPipeline": "whisperjav.pipelines.fast_pipeline",
    "FidelityPipeline": "whisperjav.pipelines.fidelity_pipeline",
    "MediaDiscovery": "whisperjav.modules.media_discovery",
    "setup_logger": "whisperjav.utils.logger",
}


def __getattr__(name: str):
    """
    Lazy import handler (PEP 562).

    Called when an attribute is not found in the module namespace.
    Imports the requested class/function on first access.
    """
    if name in _LAZY_IMPORTS:
        import importlib
        module_path = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        attr = getattr(module, name)
        # Cache it in the module namespace for subsequent access
        globals()[name] = attr
        return attr

    raise AttributeError(f"module 'whisperjav' has no attribute '{name}'")


# Don't import main at top level - avoid premature loading
def _run():
    from .main import main
    main()


if __name__ == "__main__":
    _run()
