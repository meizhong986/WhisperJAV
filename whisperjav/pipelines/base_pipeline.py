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
                 save_metadata_json: bool = False,
                 adaptive_classification: bool = False,
                 adaptive_audio_enhancement: bool = False,
                 smart_postprocessing: bool = False,
                 **kwargs  # --- FIX: Accept and ignore extra keyword arguments ---
                ):
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.keep_temp_files = keep_temp_files
        self.save_metadata_json = save_metadata_json  # Preserve metadata JSON files (enabled by --debug)
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
        """Clean up temporary files for a specific media file.

        Metadata JSON files are preserved when save_metadata_json is True (via --debug flag).
        """
        if not self.keep_temp_files:
            logger.debug(f"Cleaning up temporary files for {media_basename}")
            try:
                # List of specific files to delete
                files_to_delete = [
                    self.temp_dir / f"{media_basename}_extracted.wav",
                    self.temp_dir / f"{media_basename}_raw.srt",
                    self.temp_dir / f"{media_basename}_stitched.srt",
                ]

                # Only delete metadata JSON if save_metadata_json is False
                if not self.save_metadata_json:
                    files_to_delete.append(self.temp_dir / f"{media_basename}_master.json")
                else:
                    logger.debug(f"Preserving metadata JSON: {media_basename}_master.json (--debug mode)")

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

    def cleanup(self):
        """
        Clean up pipeline resources including ASR model.

        This should be called when the pipeline is no longer needed,
        especially in batch processing scenarios where models need to be
        swapped between passes. This frees GPU memory.

        In subprocess workers (ensemble mode):
        - ASR cleanup IS called to release model references during controlled execution
          (prevents ctranslate2 destructor from running during interpreter shutdown)
        - CUDA cache clear is SKIPPED (torch.cuda.empty_cache() can crash on Windows)

        ROOT CAUSE FIX: The previous approach of skipping ALL cleanup in subprocess
        workers caused BrokenProcessPool crashes because ctranslate2's C++ destructor
        ran during Python interpreter shutdown - an unsafe phase where CUDA operations
        are unreliable on Windows. By calling asr.cleanup() during controlled execution,
        we trigger the destructor NOW (where we can catch errors) rather than during
        interpreter shutdown (where crashes terminate the process).
        """
        import os

        is_subprocess = os.environ.get('WHISPERJAV_SUBPROCESS_WORKER') == '1'

        # ALWAYS clean up GPU resources - even in subprocess workers
        # This triggers native destructors during controlled execution,
        # which is SAFER than letting them run during interpreter shutdown.

        # Clean up speech enhancer if present (may have GPU models)
        if hasattr(self, 'speech_enhancer') and self.speech_enhancer:
            try:
                if hasattr(self.speech_enhancer, 'cleanup'):
                    self.speech_enhancer.cleanup()
                self.speech_enhancer = None
            except Exception as e:
                logger.warning(f"Speech enhancer cleanup failed (non-fatal): {e}")

        # Clean up ASR model (ctranslate2, PyTorch models)
        if hasattr(self, 'asr') and hasattr(self.asr, 'cleanup'):
            try:
                self.asr.cleanup()
            except Exception as e:
                # Log but don't crash - this is best-effort cleanup
                logger.warning(f"ASR cleanup failed (non-fatal): {e}")

        # CUDA cache clear - skip in subprocess (known to crash on Windows)
        if is_subprocess:
            logger.debug(
                f"{self.__class__.__name__} CUDA cache clear skipped in subprocess"
            )
        else:
            from whisperjav.utils.gpu_utils import safe_cuda_cleanup
            safe_cuda_cleanup()

        logger.debug(f"{self.__class__.__name__} cleanup complete")

    def __enter__(self):
        """Context manager entry - returns self for use in with statement."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit - ensures cleanup is always called.

        This guarantees that GPU resources are released even if an exception
        occurs during processing, preventing VRAM leaks.

        Args:
            exc_type: Exception type if an exception was raised, None otherwise
            exc_val: Exception value if an exception was raised, None otherwise
            exc_tb: Exception traceback if an exception was raised, None otherwise

        Returns:
            False to propagate any exception that occurred
        """
        try:
            self.cleanup()
        except Exception as cleanup_error:
            logger.error(f"Error during pipeline cleanup: {cleanup_error}")
            # Don't suppress the original exception if one occurred
        return False  # Don't suppress exceptions
