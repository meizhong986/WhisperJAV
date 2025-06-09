# In whisperjav/__init__.py
"""WhisperJAV - Japanese Adult Video Subtitle Generator"""

__version__ = "1.0.0"

# Public API exports
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.utils.logger import setup_logger

__all__ = [
    "FasterPipeline",
    "FastPipeline", 
    "BalancedPipeline",
    "MediaDiscovery",
    "setup_logger",
]