# In whisperjav/__init__.py
"""WhisperJAV - Japanese Adult Video Subtitle Generator"""

__version__ = "1.1.0"  # Match main.py version

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

# Don't import main at top level — avoid premature loading
def _run():
    from .main import main
    main()

if __name__ == "__main__":
    _run()
