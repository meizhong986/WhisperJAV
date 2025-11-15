# In whisperjav/__init__.py
"""WhisperJAV - Japanese Adult Video Subtitle Generator"""

"""WhisperJAV - Japanese Adult Video Subtitle Generator"""

from whisperjav.__version__ import __version__, __version_info__



# Public API exports
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.utils.logger import setup_logger



__all__ = [
    "FasterPipeline",
    "FastPipeline",
    "FidelityPipeline",
    "MediaDiscovery",
    "setup_logger",
]

# Don't import main at top level — avoid premature loading
def _run():
    from .main import main
    main()

if __name__ == "__main__":
    _run()
