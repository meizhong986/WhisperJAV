"""
ASR (Automatic Speech Recognition) Components.

Each ASR engine is defined in its own module and auto-registered.
"""

# Import all ASR components to trigger registration
from .faster_whisper import FasterWhisperASR
from .stable_ts import StableTSASR

__all__ = [
    'FasterWhisperASR',
    'StableTSASR',
]
