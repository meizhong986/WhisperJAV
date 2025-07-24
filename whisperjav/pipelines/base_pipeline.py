#!/usr/bin/env python3
"""V3 Arch and new UI Base pipeline class for WhisperJAV."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict
from whisperjav.utils.metadata_manager import MetadataManager
from whisperjav.utils.logger import logger
import os
import shutil


class BasePipeline(ABC):
    """Abstract base class for all WhisperJAV pipelines."""
    
    def __init__(self, 
                 output_dir: str = "./output",
                 temp_dir: str = "./temp",
                 keep_temp_files: bool = False,
                 adaptive_classification: bool = False,
                 adaptive_audio_enhancement: bool = False,
                 smart_postprocessing: bool = False,
                 **kwargs  # --- FIX: Accept and ignore extra keyword arguments ---
                ):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.keep_temp_files = keep_temp_files
        self.metadata_manager = MetadataManager(self.temp_dir, self.output_dir)

        self.adaptive_classification = adaptive_classification
        self.adaptive_audio_enhancement = adaptive_audio_enhancement
        self.smart_postprocessing = smart_postprocessing
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Log any unused kwargs to help with debugging, but don't crash
        if kwargs:
            logger.debug(f"BasePipeline received unused arguments: {list(kwargs.keys())}")
        
    @abstractmethod
    def get_mode_name(self) -> str:
        """Return the name of this pipeline mode."""
        pass
        
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
        """Clean up temporary files for a specific media file."""
        if not self.keep_temp_files:
            logger.debug(f"Cleaning up temporary files for {media_basename}")
            try:
                # List of specific files to delete
                files_to_delete = [
                    self.temp_dir / f"{media_basename}_extracted.wav",
                    self.temp_dir / f"{media_basename}_raw.srt",
                    self.temp_dir / f"{media_basename}_stitched.srt",
                    self.temp_dir / f"{media_basename}_master.json"
                ]
                
                for file_path in files_to_delete:
                    if file_path.exists():
                        file_path.unlink()
                        logger.debug(f"Deleted temporary file: {file_path}")

                # Clean up scene-specific WAV files
                scenes_dir = self.temp_dir / "scenes"
                if scenes_dir.exists():
                    for scene_file in scenes_dir.glob(f"{media_basename}_scene_*.wav"):
                        scene_file.unlink()
                        logger.debug(f"Deleted temporary scene: {scene_file}")
                
                # Clean up scene-specific SRT files
                scene_srts_dir = self.temp_dir / "scene_srts"
                if scene_srts_dir.exists():
                    for srt_file in scene_srts_dir.glob(f"{media_basename}_scene_*.srt"):
                        srt_file.unlink()
                        logger.debug(f"Deleted temporary scene SRT: {srt_file}")

                # Clean up raw_subs in temp directory (only for current media)
                temp_raw_subs_dir = self.temp_dir / "raw_subs"
                if temp_raw_subs_dir.exists():
                    for raw_file in temp_raw_subs_dir.glob(f"{media_basename}*"):
                        raw_file.unlink()
                        logger.debug(f"Deleted temporary raw_subs file: {raw_file}")

            except Exception as e:
                logger.error(f"Error during temporary file cleanup for {media_basename}: {e}")
        else:
            logger.debug("Skipping cleanup of temporary files as requested by --keep-temp flag.")
