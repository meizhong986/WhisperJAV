"""
VAD (Voice Activity Detection) Components.

Each VAD engine is defined in its own module and auto-registered.
"""

# Import all VAD components to trigger registration
from .silero import SileroVAD

__all__ = [
    'SileroVAD',
]
