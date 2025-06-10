#!/usr/bin/env python3
"""Base pipeline class for WhisperJAV."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
from whisperjav.utils.metadata_manager import MetadataManager
from whisperjav.utils.logger import logger


class BasePipeline(ABC):
    """Abstract base class for all WhisperJAV pipelines."""
    
    def __init__(self, 
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.keep_temp_files = keep_temp_files
        self.metadata_manager = MetadataManager(self.temp_dir, self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def get_mode_name(self) -> str:
        """Return the name of this pipeline mode."""
        pass
        
    # --- CONTRACT CHANGE: Accept media_info dict instead of string path ---
    @abstractmethod
    def process(self, media_info: Dict) -> Dict:
        """
        Process a single media file using its discovered metadata.
        
        Args:
            media_info: Dictionary containing discovered media information
                       including 'path', 'basename', 'type', 'duration', etc.
                       
        Returns:
            Dictionary containing processing metadata and results
        """
        pass
        
    def cleanup_temp_files(self, media_basename: str):
        """Clean up temporary files for a media file."""
        if not self.keep_temp_files:
            logger.info(f"Cleaning up temporary files for {media_basename}")
            # Implementation would clean up chunk files, temp audio, etc.
            # For now, we'll keep it simple
            pass