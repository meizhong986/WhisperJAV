#!/usr/bin/env python3
"""Base pipeline class for WhisperJAV."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
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
        
        # Ensure directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
    @abstractmethod
    def get_mode_name(self) -> str:
        """Return the name of this pipeline mode."""
        pass
        
    @abstractmethod
    def process(self, input_file: str) -> Dict:
        """Process a single media file through the pipeline."""
        pass
        
    def cleanup_temp_files(self, media_basename: str):
        """Clean up temporary files for a media file."""
        if not self.keep_temp_files:
            logger.info(f"Cleaning up temporary files for {media_basename}")
            # Implementation would remove temp files
            # For now, just log
            pass
            
    def get_media_basename(self, input_file: str) -> str:
        """Extract basename from input file path."""
        return Path(input_file).stem