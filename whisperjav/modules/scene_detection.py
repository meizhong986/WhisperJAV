#!/usr/bin/env python3
"""Audio scene detection module using a two-pass Auditok approach."""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import soundfile as sf
from tqdm import tqdm

from whisperjav.utils.logger import logger

# Try to import auditok
_AUDITOK_AVAILABLE = False
try:
    import auditok
    _AUDITOK_AVAILABLE = True
except ImportError:
    logger.warning("Auditok not available. Scene detection functionality will be limited.")


class SceneDetector:
    """Handles audio scene detection using a two-pass Auditok approach."""
    
    def __init__(self,
                 max_duration: float = 30.0,
                 min_duration: float = 0.2,
                 max_silence: float = 2.0,
                 energy_threshold: int = 50):
        """
        Initialize the audio scene detector.
        
        Args:
            max_duration: Maximum scene duration in seconds
            min_duration: Minimum scene duration in seconds
            max_silence: Maximum silence duration for splitting
            energy_threshold: Energy threshold for voice activity detection
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_silence = max_silence
        self.energy_threshold = energy_threshold
        
        # Two-pass parameters
        self.pass1_max_duration = 1599.0  # ~25 minutes
        self.pass1_min_duration = 0.1
        self.pass1_max_silence = max_silence
        self.pass1_energy_threshold = energy_threshold
        
        self.pass2_max_duration = max_duration
        self.pass2_min_duration = 0.5
        self.pass2_max_silence = 0.75
        self.pass2_energy_threshold = 60
        
    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str) -> List[Tuple[Path, float, float, float]]:
        """
        Split audio into scenes using a two-pass Auditok approach.
        
        Args:
            audio_path: Path to input audio file
            output_dir: Directory to save scene files
            media_basename: The base name of the original media file for naming scenes
            
        Returns:
            List of tuples: (scene_path, start_time, end_time, duration)
        """
        if not _AUDITOK_AVAILABLE:
            raise RuntimeError("Auditok is required for scene detection but not installed")
        
        logger.debug(f"Starting two-pass audio scene detection for: {audio_path}")
        
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)  # Convert to mono
        
        audio_duration = len(audio_data) / sample_rate
        logger.debug(f"Audio duration: {audio_duration:.1f}s, Sample rate: {sample_rate}Hz")
        
        scene_tuples = []
        scene_idx = 0
        
        # Pass 1: Coarse splitting
        logger.debug("Pass 1: Coarse splitting")
        normalized_audio = self._normalize_audio(audio_data.copy())
        audio_bytes = (normalized_audio * 32767).astype(np.int16).tobytes()
        
        pass1_params = {
            "sampling_rate": sample_rate,
            "channels": 1,
            "sample_width": 2,
            "min_dur": self.pass1_min_duration,
            "max_dur": self.pass1_max_duration,
            "max_silence": min(audio_duration * 0.95, self.pass1_max_silence),
            "energy_threshold": self.pass1_energy_threshold,
            "drop_trailing_silence": True
        }
        
        regions = list(auditok.split(audio_bytes, **pass1_params))
        logger.debug(f"Pass 1 found {len(regions)} regions")
        
        # Process each region - DISABLE tqdm here
        for region_idx, region in enumerate(regions):
            region_start = region.start
            region_end = region.end
            region_duration = region_end - region_start
            
            # Extract region audio
            start_sample = int(region_start * sample_rate)
            end_sample = int(region_end * sample_rate)
            region_audio = audio_data[start_sample:end_sample]
            
            # If region is short enough, save it
            if region_duration <= self.max_duration:
                if region_duration >= self.min_duration:
                    scene_path = self._save_scene(
                        region_audio, sample_rate, scene_idx, output_dir,
                        media_basename
                    )
                    scene_tuples.append((scene_path, region_start, region_end, region_duration))
                    scene_idx += 1
            else:
                # Pass 2: Split long regions
                logger.debug(f"Pass 2: Splitting long region {region_idx} ({region_duration:.1f}s)")
                
                # Try auditok pass 2 first
                normalized_region = self._normalize_audio(region_audio.copy())
                region_bytes = (normalized_region * 32767).astype(np.int16).tobytes()
                
                pass2_params = {
                    "sampling_rate": sample_rate,
                    "channels": 1,
                    "sample_width": 2,
                    "min_dur": self.pass2_min_duration,
                    "max_dur": self.pass2_max_duration,
                    "max_silence": min(region_duration * 0.95, self.pass2_max_silence),
                    "energy_threshold": self.pass2_energy_threshold,
                    "drop_trailing_silence": True
                }
                
                sub_regions = list(auditok.split(region_bytes, **pass2_params))
                
                if not sub_regions:
                    # Fallback: brute force splitting
                    logger.warning(f"Pass 2 found no sub-regions, using brute force splitting")
                    num_scenes = int(np.ceil(region_duration / self.max_duration))
                    for i in range(num_scenes):
                        sub_start = i * self.max_duration
                        sub_end = min((i + 1) * self.max_duration, region_duration)
                        
                        if sub_end - sub_start < self.min_duration:
                            continue
                        
                        sub_start_sample = int(sub_start * sample_rate)
                        sub_end_sample = int(sub_end * sample_rate)
                        sub_audio = region_audio[sub_start_sample:sub_end_sample]
                        
                        abs_start = region_start + sub_start
                        abs_end = region_start + sub_end
                        
                        scene_path = self._save_scene(
                            sub_audio, sample_rate, scene_idx, output_dir,
                            media_basename
                        )
                        scene_tuples.append((scene_path, abs_start, abs_end, sub_end - sub_start))
                        scene_idx += 1
                else:
                    # Process sub-regions from pass 2
                    logger.debug(f"Pass 2 split region into {len(sub_regions)} sub-scenes")
                    for sub_region in sub_regions:
                        sub_start = sub_region.start
                        sub_end = sub_region.end
                        sub_duration = sub_end - sub_start
                        
                        if sub_duration < self.min_duration:
                            continue
                        
                        sub_start_sample = int(sub_start * sample_rate)
                        sub_end_sample = int(sub_end * sample_rate)
                        sub_audio = region_audio[sub_start_sample:sub_end_sample]
                        
                        abs_start = region_start + sub_start
                        abs_end = region_start + sub_end
                        
                        scene_path = self._save_scene(
                            sub_audio, sample_rate, scene_idx, output_dir,
                            media_basename
                        )
                        scene_tuples.append((scene_path, abs_start, abs_end, sub_duration))
                        scene_idx += 1
        
        logger.info(f"Detected {len(scene_tuples)} final scenes")
        return scene_tuples
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio to have peak of 1.0."""
        logger.debug("Normalizing audio ....")
        peak = np.max(np.abs(audio_data))
        safe_peak = max(peak, 0.1)
        if safe_peak == 0:
            return audio_data
        return audio_data / safe_peak
    
    def _save_scene(self, audio_data: np.ndarray, sample_rate: int, 
                    scene_idx: int, output_dir: Path,
                    media_basename: str) -> Path:
        """Save audio scene to file with the new robust naming convention."""
        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename
        sf.write(str(scene_path), audio_data, sample_rate, subtype='PCM_16')
        return scene_path