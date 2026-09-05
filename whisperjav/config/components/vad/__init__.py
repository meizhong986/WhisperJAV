"""
VAD (Voice Activity Detection) Components.

Each VAD engine is defined in its own module and auto-registered.
"""

# Import all VAD components to trigger registration
from .silero import SileroVAD
from .faster_whisper_vad import FasterWhisperVAD

__all__ = [
    'SileroVAD',
    'FasterWhisperVAD',
]
