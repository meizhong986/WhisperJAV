#!/usr/bin/env python3
"""
Audio scene detection module using a multi-feature, adaptive approach.
This version incorporates a two-pass strategy, Auditok coarse, STFT, DRC, onset-based
recovery, and metadata output.

==============================================================================
CLASS AND FUNCTION STATUS (for external reviewers)
==============================================================================

CLASSES:
┌─────────────────────────┬──────────┬─────────────────────────────────────────┐
│ Class                   │ Status   │ Notes                                   │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ DynamicSceneDetector    │ ACTIVE   │ Used by ALL pipelines (balanced, fast,  │
│ (line ~685)             │          │ fidelity, kotoba, transformers).        │
│                         │          │ Supports methods: auditok, silero,      │
│                         │          │ semantic. This is the PRIMARY class.    │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ SceneDetector           │ LEGACY   │ Original implementation. Imported but   │
│ (line ~77)              │          │ NOT instantiated by any active pipeline.│
│                         │          │ Only used in legacy tests. DO NOT USE.  │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ AdaptiveSceneDetector   │ LEGACY   │ Second-gen implementation. NOT used by  │
│ (line ~306)             │          │ any active pipeline. DO NOT USE.        │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ SileroRegion            │ INTERNAL │ Helper dataclass for Silero VAD output. │
│ (line ~946)             │          │ Used internally by DynamicSceneDetector.│
└─────────────────────────┴──────────┴─────────────────────────────────────────┘

TOP-LEVEL FUNCTIONS:
┌─────────────────────────┬──────────┬─────────────────────────────────────────┐
│ Function                │ Status   │ Notes                                   │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ load_audio_unified()    │ INTERNAL │ Used only within this module by         │
│ (line ~29)              │          │ DynamicSceneDetector.detect_scenes().   │
├─────────────────────────┼──────────┼─────────────────────────────────────────┤
│ analyze_scene_metadata()│ UNUSED   │ Utility function with no upstream usage.│
│ (line ~1357)            │          │ Candidate for removal.                  │
└─────────────────────────┴──────────┴─────────────────────────────────────────┘

DynamicSceneDetector METHODS (the active class):
┌─────────────────────────────┬──────────┬─────────────────────────────────────┐
│ Method                      │ Status   │ Notes                               │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ __init__()                  │ ACTIVE   │ Entry point. Accepts method param:  │
│                             │          │ "auditok" (default), "silero",      │
│                             │          │ "semantic"                          │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ detect_scenes()             │ ACTIVE   │ Main entry point called by all      │
│                             │          │ pipelines. Routes to appropriate    │
│                             │          │ backend based on method.            │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ get_detection_metadata()    │ ACTIVE   │ Called by pipelines for metadata    │
│                             │          │ (scene boundaries, VAD segments).   │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _init_silero_vad()          │ ACTIVE   │ Called when method="silero"         │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _init_semantic_adapter()    │ ACTIVE   │ Called when method="semantic"       │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _detect_pass2_silero()      │ ACTIVE   │ Silero VAD for fine scene splitting │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _apply_assistive_processing()│ ACTIVE  │ Audio preprocessing for auditok     │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _save_scene()               │ ACTIVE   │ Writes scene audio to disk          │
├─────────────────────────────┼──────────┼─────────────────────────────────────┤
│ _record_scene_metadata()    │ ACTIVE   │ Data contract: records scene info   │
└─────────────────────────────┴──────────┴─────────────────────────────────────┘

DEPRECATED PARAMETERS (v1.8.0+, Issue #129):
┌─────────────────────────┬─────────────────────────────────────────────────────┐
│ Parameter               │ Notes                                               │
├─────────────────────────┼─────────────────────────────────────────────────────┤
│ target_sr               │ No longer used. Auditok handles any sample rate     │
│                         │ natively. Kept for backward compatibility.          │
├─────────────────────────┼─────────────────────────────────────────────────────┤
│ preserve_original_sr    │ No longer used. Native SR always preserved.         │
│                         │ Kept for backward compatibility.                    │
└─────────────────────────┴─────────────────────────────────────────────────────┘

UPSTREAM USAGE (which pipelines use this module):
- balanced_pipeline.py  → DynamicSceneDetector (method from config, default: auditok)
- fast_pipeline.py      → DynamicSceneDetector (method from config, default: auditok)
- fidelity_pipeline.py  → DynamicSceneDetector (method from config, default: auditok)
- kotoba_faster_whisper_pipeline.py → DynamicSceneDetector (method from CLI arg)
- transformers_pipeline.py → DynamicSceneDetector (method from CLI arg)
- ensemble/pass_worker.py → DynamicSceneDetector (via pipeline instantiation)

Last updated: 2026-01-14 (Issue #129 fix)
==============================================================================
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Third-party libraries
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pydub import AudioSegment
from pydub.effects import compress_dynamic_range
from scipy.ndimage import binary_closing, binary_opening
from scipy.signal import medfilt, butter, filtfilt
from tqdm import tqdm

# Use the global WhisperJAV logger
from whisperjav.utils.logger import logger


def load_audio_unified(audio_path: Path, target_sr: Optional[int] = None, force_mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Unified audio loading function used by all scene detectors.
    Provides consistent audio format and sample rate handling.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (None = preserve original)
        force_mono: Convert to mono if stereo
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Use soundfile for consistent loading with optional resampling
        audio_data, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
        
        # Convert stereo to mono efficiently if needed
        if force_mono and audio_data.ndim > 1:
            logger.debug("Converting stereo to mono")
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if target sample rate specified
        if target_sr is not None and sample_rate != target_sr:
            logger.debug(f"Resampling from {sample_rate}Hz to {target_sr}Hz")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)
            sample_rate = target_sr
            
        logger.debug(f"Audio loaded: {len(audio_data)/sample_rate:.1f}s @ {sample_rate}Hz, {'mono' if audio_data.ndim == 1 else 'stereo'}")
        return audio_data, sample_rate
        
    except Exception as e:
        logger.error(f"Failed to load audio {audio_path}: {e}")
        raise

# Try to import auditok for the coarse-splitting pass
AUDITOK_AVAILABLE = False
try:
    import auditok
    AUDITOK_AVAILABLE = True
except ImportError:
    logger.warning("Auditok not available. Scene detection functionality will be limited.")

    
    
    
    

class SceneDetector:
    """Handles audio scene detection using a two-pass Auditok approach."""
    
    # Class-level LUFS meter cache to avoid recreation
    _lufs_meter_cache = {}
    
    def __init__(self,
                 max_duration: float = 29.0,
                 min_duration: float = 0.3,
                 max_silence: float = 1.8,
                 energy_threshold: int = 38):
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
        self.pass1_max_duration = 2700.0  # ~45 minutes
        self.pass1_min_duration = 0.3
        self.pass1_max_silence = 1.8
        self.pass1_energy_threshold = 38
        
        self.pass2_max_duration = 28
        self.pass2_min_duration = 0.3
        self.pass2_max_silence = 0.94
        self.pass2_energy_threshold = 50
        
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
        if not AUDITOK_AVAILABLE:
            raise RuntimeError("Auditok is required for scene detection but not installed")
        
        logger.debug(f"Starting two-pass audio scene detection for: {audio_path}")
        
        # Use unified audio loading for consistency
        audio_data, sample_rate = load_audio_unified(audio_path, target_sr=None, force_mono=True)
        
        audio_duration = len(audio_data) / sample_rate
        logger.info(f"Audio duration: {audio_duration:.1f}s, Sample rate: {sample_rate}Hz")
        
        scene_tuples = []
        scene_idx = 0
        
        # Pass 1: Coarse splitting
        logger.debug("Pass 1: Phase 1 splitting")
        
        normalized_audio = self._lufs_normalize(audio_data.copy(), sample_rate)
        #normalized_audio = self._normalize_audio(audio_data.copy())
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
                normalized_region = self._lufs_normalize(region_audio.copy(), sample_rate)
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
        
    def _lufs_normalize(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Normalize audio to target LUFS loudness"""
        # Use cached loudness meter (ITU-R BS.1770-4) to avoid expensive recreation
        target_lufs = -14.0
        if sample_rate not in self._lufs_meter_cache:
            self._lufs_meter_cache[sample_rate] = pyln.Meter(sample_rate)
        meter = self._lufs_meter_cache[sample_rate]
        
        # Measure current loudness (this is the expensive operation)
        try:
            loudness = meter.integrated_loudness(audio)
            logger.debug(f"Original loudness: {loudness:.1f} LUFS")
        except Exception as e:
            logger.warning(f"LUFS measurement failed: {e}, using original audio")
            return audio
        
        # Apply normalization if measurable
        if loudness > -70:  # Avoid silent segments
            try:
                normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
            except Exception as e:
                logger.warning(f"LUFS normalization failed: {e}, using original audio")
                return audio
            
            # Prevent clipping
            peak = np.max(np.abs(normalized))
            if peak > 1.0:
                normalized /= (peak * 1.05)  # Headroom
                logger.warning(f"Clipping prevented (post-LUFS peak: {peak:.2f})")
            return normalized
        return audio
        
    
    def _save_scene(self, audio_data: np.ndarray, sample_rate: int, 
                    scene_idx: int, output_dir: Path,
                    media_basename: str) -> Path:
        """Save audio scene to file with the new robust naming convention."""
        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename
        sf.write(str(scene_path), audio_data, sample_rate, subtype='PCM_16')
        return scene_path
        
        
   
class AdaptiveSceneDetector:
    """
    Handles audio scene detection using a multi-feature, adaptive thresholding
    approach, suitable for high dynamic range audio. It employs a two-pass
    strategy: a coarse pass to identify large 'story line' chunks based on
    silence, followed by a detailed processing pass on each chunk to conserve memory.
    """
    def __init__(self,
                 max_duration: float = 29.0,
                 min_duration: float = 0.3,
                 # Coarse pass parameters
                 story_line_silence_s: float = 1.8,
                 story_line_energy_threshold: int = 38,
                 # DRC parameters for speech pre-processing
                 use_drc: bool = True,
                 drc_threshold: float = -25.0,
                 drc_ratio: float = 5.0,
                 drc_attack: float = 5.0,
                 drc_release: float = 100.0,
                 # Librosa STFT parameters
                 n_fft: int = 2048,
                 hop_length: int = 256,
                 # Adaptive thresholding parameters
                 rms_threshold_factor: float = 1.5, #1.3,
                 zcr_threshold_factor: float = 1.0,
                 adaptive_filter_size_s: float = 4.0,
                 onset_recovery_threshold_p: float = 90.0, #80.0,
                 # Segment filtering parameters
                 min_silence_len_s: float = 2.3, #1.5, #0.7,
                 merge_short_speech_window_s: float = 0.1, # in seconds
                 remove_spurious_speech_window_s: float = 0.05, # in seconds
                 refinement_tolerance_s: float = 0.250, #0.1
                 min_avg_energy_db: Optional[float] = -50.0
                 ):
        """
        Initialize the robust audio scene detector with parameters tuned for Japanese audio.
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.story_line_silence_s = story_line_silence_s
        self.story_line_energy_threshold = story_line_energy_threshold
        self.use_drc = use_drc
        self.drc_threshold = drc_threshold
        self.drc_ratio = drc_ratio
        self.drc_attack = drc_attack
        self.drc_release = drc_release
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.rms_threshold_factor = rms_threshold_factor
        self.zcr_threshold_factor = zcr_threshold_factor
        self.adaptive_filter_size_s = adaptive_filter_size_s
        self.onset_recovery_threshold_p = onset_recovery_threshold_p
        self.min_silence_len_s = min_silence_len_s
        self.merge_short_speech_window_s = merge_short_speech_window_s
        self.remove_spurious_speech_window_s = remove_spurious_speech_window_s
        self.refinement_tolerance_s = refinement_tolerance_s
        self.min_avg_energy_db = min_avg_energy_db
        
    def _find_story_lines(self, audio_data: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """
        Performs a coarse split on the audio to find large 'story line' chunks
        separated by significant silence. Returns a list of (start_sec, end_sec).
        """
        duration_s = len(audio_data) / sample_rate
        if not AUDITOK_AVAILABLE:
            if duration_s > 600:
                logger.error("Auditok is not installed. Processing a long audio file (>10min) as a single chunk will be very slow and memory-intensive.")
            else:
                logger.warning("Auditok not available. Processing file as a single chunk.")
            return [(0.0, duration_s)]
        logger.info("Pass 1: Finding coarse 'story line' boundaries with Auditok...")
        peak = np.max(np.abs(audio_data))
        normalized_audio = audio_data / (peak + 1e-7)
        audio_bytes = (normalized_audio * 32767).astype(np.int16).tobytes()
        
        regions = auditok.split(
            audio_bytes, sampling_rate=sample_rate, sample_width=2, channels=1,
            min_dur=self.min_duration, max_dur=3600, max_silence=self.story_line_silence_s,
            energy_threshold=self.story_line_energy_threshold, drop_trailing_silence=True
        )
        
        story_chunks = [(r.start, r.end) for r in regions]
        logger.info(f"Pass 1: Found {len(story_chunks)} large story chunks.")
        return story_chunks
        
    def _preprocess_audio_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies Dynamic Range Compression (if enabled) to a NumPy audio chunk.
        """
        if not self.use_drc:
            return audio_chunk
        logger.debug("Applying DRC to audio chunk...")
        if np.max(np.abs(audio_chunk)) > 1.0:
             audio_chunk /= np.max(np.abs(audio_chunk))
        
        # Add dithering to prevent quantization artifacts
        dither = np.random.uniform(-0.5, 0.5, len(audio_chunk))
        samples_int16 = (audio_chunk * 32767 + dither).astype(np.int16)
        audio_segment = AudioSegment(
            samples_int16.tobytes(), frame_rate=sample_rate,
            sample_width=2, channels=1
        )
        compressed_segment = compress_dynamic_range(
            audio_segment, threshold=self.drc_threshold, ratio=self.drc_ratio,
            attack=self.drc_attack, release=self.drc_release
        )
        samples = np.array(compressed_segment.get_array_of_samples()).astype(np.float32)
        samples /= 32768.0
        return samples
        
    def _compute_all_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Compute all features with a single STFT for efficiency."""
        S = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        power = np.abs(S)**2
        
        # Get RMS and spectral centroid from STFT
        rms = librosa.feature.rms(S=S)[0]
        
        # Compute ZCR directly from audio
        zcr = librosa.feature.zero_crossing_rate(
            y=audio, 
            frame_length=self.n_fft, 
            hop_length=self.hop_length
        )[0]
        
        # Compute onset strength
        onset = librosa.onset.onset_strength(S=power, sr=sr)
        
        # CRITICAL FIX: Ensure all features have the same length
        # onset_strength often returns one fewer frame, so we need to align
        min_len = min(len(rms), len(zcr), len(onset))
        
        return {
            'rms': rms[:min_len],
            'zcr': zcr[:min_len],
            'onset': onset[:min_len],
        }
    
    def _adaptive_threshold(self, feature_series: np.ndarray, filter_size_frames: int, factor: float) -> np.ndarray:
        """Calculates an adaptive threshold using a moving median filter."""
        # Ensure minimum window size and odd number for proper median calculation
        filter_size_frames = max(1, filter_size_frames)
        if filter_size_frames % 2 == 0:
            filter_size_frames += 1
        
        if len(feature_series) < filter_size_frames:
             return np.median(feature_series) * np.ones_like(feature_series) * factor
        
        # CRITICAL FIX: Use explicit length calculation to avoid slicing issues
        pad_size = filter_size_frames // 2
        padded_series = np.pad(feature_series, (pad_size, pad_size), mode='edge')
        median_filtered = medfilt(padded_series, kernel_size=filter_size_frames)
        
        # Instead of negative slicing, use explicit indices
        start_idx = pad_size
        end_idx = len(median_filtered) - pad_size
        result = median_filtered[start_idx:end_idx]
        
        # CRITICAL: Ensure output has exactly the same length as input
        if len(result) != len(feature_series):
            logger.warning(f"Adaptive threshold length mismatch: input {len(feature_series)}, output {len(result)}")
            # Force to same length by padding or truncating
            if len(result) < len(feature_series):
                # Pad with the last value
                result = np.pad(result, (0, len(feature_series) - len(result)), mode='edge')
            else:
                # Truncate
                result = result[:len(feature_series)]
        
        return result * factor
    
    def _find_continuous_segments(self, is_speech: np.ndarray, frame_times: np.ndarray) -> List[Tuple[float, float]]:
        """Converts a boolean activity mask into a list of (start, end) time segments."""
        segments = []
        in_speech = False
        start_time = 0.0
        for i, active in enumerate(is_speech):
            if active and not in_speech:
                start_time = frame_times[i]
                in_speech = True
            elif not active and in_speech:
                segments.append((start_time, frame_times[i]))
                in_speech = False
        if in_speech:
            segments.append((start_time, frame_times[-1]))
        return segments
        
    def _merge_close_segments(self, segments: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merges speech segments that are separated by short silences."""
        if not segments:
            return []
        
        merged = [segments[0]]
        for next_start, next_end in segments[1:]:
            last_start, last_end = merged[-1]
            if (next_start - last_end) < self.min_silence_len_s:
                merged[-1] = (last_start, next_end)
            else:
                merged.append((next_start, next_end))
        return merged
        
    def _refine_and_split_segments(self, 
                                   segments: List[Tuple[float, float]], 
                                   onset_env: np.ndarray, 
                                   sample_rate: int, 
                                   chunk_start_offset_s: float) -> List[Tuple[float, float]]:
        """Refines start/end boundaries and splits segments that are too long."""
        final_scenes = []
        for start_sec, end_sec in segments:
            duration = end_sec - start_sec
            if duration < self.min_duration:
                continue
            start_frame = librosa.time_to_frames(start_sec, sr=sample_rate, hop_length=self.hop_length)
            # FIX: Convert to numpy array for numba compatibility
            refined_start_frame = librosa.onset.onset_backtrack(np.array([start_frame]), onset_env)[0]
            refined_start_sec = librosa.frames_to_time([refined_start_frame], sr=sample_rate, hop_length=self.hop_length)[0]
            
            end_frame = librosa.time_to_frames(end_sec, sr=sample_rate, hop_length=self.hop_length)
            # FIX: Clamp end_frame to avoid index errors in reversed tracking
            end_frame = min(end_frame, len(onset_env) - 1)
            
            offset_env = onset_env[::-1]
            reversed_end_frame = len(onset_env) - end_frame
            # FIX: Ensure reversed_end_frame is within bounds
            reversed_end_frame = min(reversed_end_frame, len(offset_env) - 1)
            # FIX: Convert to numpy array for numba compatibility
            refined_reversed_end_frame = librosa.onset.onset_backtrack(np.array([reversed_end_frame]), offset_env)[0]
            refined_end_frame = len(onset_env) - refined_reversed_end_frame
            refined_end_sec = librosa.frames_to_time([refined_end_frame], sr=sample_rate, hop_length=self.hop_length)[0]
            refined_start_sec = max(start_sec - self.refinement_tolerance_s, refined_start_sec)
            refined_start_sec = min(start_sec, refined_start_sec)
            
            refined_end_sec = min(end_sec + self.refinement_tolerance_s, refined_end_sec)
            refined_end_sec = max(end_sec, refined_end_sec)
            duration = refined_end_sec - refined_start_sec
            if duration > self.max_duration:
                num_chunks = int(np.ceil(duration / self.max_duration))
                chunk_dur = duration / num_chunks
                for i in range(num_chunks):
                    sub_start = refined_start_sec + i * chunk_dur
                    sub_end = min(refined_start_sec + (i + 1) * chunk_dur, refined_end_sec)
                    if (sub_end - sub_start) >= self.min_duration:
                        final_scenes.append((sub_start + chunk_start_offset_s, sub_end + chunk_start_offset_s))
            elif duration >= self.min_duration:
                final_scenes.append((refined_start_sec + chunk_start_offset_s, refined_end_sec + chunk_start_offset_s))
        
        return final_scenes
        
    def _process_story_chunk(self, original_audio_chunk: np.ndarray, sample_rate: int, chunk_start_offset_s: float) -> List[Tuple[float, float]]:
        processed_audio = self._preprocess_audio_chunk(original_audio_chunk, sample_rate)
        features = self._compute_all_features(processed_audio, sample_rate)
        
        # Features are already aligned in _compute_all_features
        rms = features['rms']
        zcr = features['zcr']
        onset = features['onset']
        
        # Compute adaptive thresholds
        # Note: adaptive_filter_frames calculation ensures minimum window size in _adaptive_threshold
        adaptive_filter_frames = int(self.adaptive_filter_size_s * sample_rate / self.hop_length)
        rms_threshold = self._adaptive_threshold(rms, adaptive_filter_frames, self.rms_threshold_factor)
        zcr_threshold = self._adaptive_threshold(zcr, adaptive_filter_frames, self.zcr_threshold_factor)
        
        # CRITICAL: Final safety check to ensure all arrays have the same length
        min_len = min(len(rms), len(zcr), len(onset), len(rms_threshold), len(zcr_threshold))
        if min_len < len(rms):
            logger.warning(f"Final length adjustment needed: {len(rms)} -> {min_len}")
            rms = rms[:min_len]
            zcr = zcr[:min_len]
            onset = onset[:min_len]
            rms_threshold = rms_threshold[:min_len]
            zcr_threshold = zcr_threshold[:min_len]
        
        # Compute onset threshold
        onset_threshold = np.percentile(onset, self.onset_recovery_threshold_p)
        
        # Now all arrays should have exactly the same length
        is_speech = ((rms > rms_threshold) & (zcr < zcr_threshold)) | (onset > onset_threshold)
        
        merge_window_frames = max(1, int(self.merge_short_speech_window_s * sample_rate / self.hop_length))
        remove_window_frames = max(1, int(self.remove_spurious_speech_window_s * sample_rate / self.hop_length))
        is_speech = binary_closing(is_speech, structure=np.ones(merge_window_frames))
        is_speech = binary_opening(is_speech, structure=np.ones(remove_window_frames))
        
        # Generate frame times with the correct length
        frame_times = librosa.frames_to_time(np.arange(len(is_speech)), sr=sample_rate, hop_length=self.hop_length)
        speech_segments = self._find_continuous_segments(is_speech, frame_times)
        
        if not speech_segments:
            return []
            
        merged_segments = self._merge_close_segments(speech_segments)
        return self._refine_and_split_segments(merged_segments, onset, sample_rate, chunk_start_offset_s)
        
    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str, metadata_path: Optional[Path] = None) -> List[Tuple[Path, float, float, float]]:
        """
        Detects audio scenes using a two-pass, memory-efficient adaptive algorithm.
        """
        logger.info(f"Loading full audio file: {audio_path}")
        # Use unified audio loading for consistency
        original_audio, sample_rate = load_audio_unified(audio_path, target_sr=None, force_mono=True)
        story_line_regions = self._find_story_lines(original_audio, sample_rate)
        all_final_scenes = []
        # Process story chunks with reduced progress spam
        for chunk_idx, (chunk_start_s, chunk_end_s) in enumerate(story_line_regions):
            # Only show progress for every 10th chunk or first/last chunk to reduce spam
            if chunk_idx % 10 == 0 or chunk_idx == len(story_line_regions) - 1:
                logger.debug(f"Processing story chunk {chunk_idx + 1}/{len(story_line_regions)}")
            start_sample = int(chunk_start_s * sample_rate)
            end_sample = int(chunk_end_s * sample_rate)
            
            original_audio_chunk = original_audio[start_sample:end_sample]
            
            chunk_scenes = self._process_story_chunk(
                original_audio_chunk, sample_rate, chunk_start_s
            )
            if chunk_scenes:
                all_final_scenes.extend(chunk_scenes)
        final_scene_tuples = []
        scene_metadata = []
        # FIX: Use a dedicated counter for saved scenes to ensure sequential numbering
        saved_scene_idx = 0
        logger.info(f"Validating and saving {len(all_final_scenes)} candidate scenes...")
        # Save scenes with reduced progress spam  
        for scene_idx_iter, (start_sec, end_sec) in enumerate(all_final_scenes):
            # Only show progress every 25 scenes or for first/last scene to reduce spam
            if scene_idx_iter % 25 == 0 or scene_idx_iter == len(all_final_scenes) - 1:
                logger.debug(f"Saving scene {scene_idx_iter + 1}/{len(all_final_scenes)}")
            scene_path, energy_db = self._save_scene(original_audio, sample_rate, saved_scene_idx, output_dir, media_basename, start_sec, end_sec)
            if scene_path:
                duration = end_sec - start_sec
                final_scene_tuples.append((scene_path, start_sec, end_sec, duration))
                if metadata_path:
                    scene_metadata.append({
                        'scene_index': saved_scene_idx, 'filename': scene_path.name,
                        'start_time_s': round(start_sec, 3), 'end_time_s': round(end_sec, 3), 'duration_s': round(duration, 3),
                        'energy_db': round(energy_db, 2)
                    })
                # Increment index only for successfully saved scenes
                saved_scene_idx += 1
        if metadata_path and scene_metadata:
            logger.info(f"Saving scene metadata to {metadata_path}")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(scene_metadata, f, indent=4)
        logger.info(f"Detected and saved {len(final_scene_tuples)} final scenes.")
        return final_scene_tuples
        
    def _save_scene(self, audio_data: np.ndarray, sample_rate: int, 
                      scene_idx: int, output_dir: Path,
                      media_basename: str, start_sec: float, end_sec: float) -> Optional[Tuple[Path, float]]:
        """Saves a segment of the audio data to a file after final validation."""
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)
        scene_audio = audio_data[start_sample:end_sample]
        
        # FIX: Perform energy check on the DRC-processed version for consistency with detection
        audio_for_check = self._preprocess_audio_chunk(scene_audio, sample_rate)
        
        rms_energy = np.sqrt(np.mean(audio_for_check**2))
        if rms_energy < 1e-9:
            return None, -np.inf
        db_energy = 20 * np.log10(rms_energy)
        if self.min_avg_energy_db is not None:
            if db_energy < self.min_avg_energy_db:
                logger.debug(f"Skipping scene {scene_idx} due to low energy ({db_energy:.2f} dB)")
                return None, db_energy
        # Save the original (non-DRC) audio segment, but normalized
        normalized_audio = librosa.util.normalize(scene_audio)
        
        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(scene_path), normalized_audio, sample_rate, subtype='PCM_16')
        return scene_path, db_energy



class DynamicSceneDetector:
    """
    Handles audio scene detection using a two-pass strategy with optional
    assistive processing for improved accuracy on challenging audio.

    This class is designed as a drop-in replacement for the original SceneDetector,
    maintaining the same public interface while offering more robust internal logic.
    """

    def __init__(self,
                 # Method selection (NEW)
                 method: str = "auditok",
                 # Core final segment bounds
                 max_duration: float = 29.0,
                 min_duration: float = 0.3,
                 # Legacy pass-1 defaults (kept for BC if *_s not provided)
                 max_silence: float = 1.8,
                 energy_threshold: int = 38,
                 # Assistive detection flags
                 assist_processing: bool = True,
                 verbose_summary: bool = True,
                 # DEPRECATED (v1.8.0+, Issue #129): These parameters have no effect.
                 # Auditok handles any sample rate natively. Kept for backward compatibility.
                 target_sr: int = 16000,  # DEPRECATED - no longer used
                 force_mono: bool = True,
                 preserve_original_sr: bool = True,  # DEPRECATED - no longer used
                 # New: pass-1 controls
                 pass1_min_duration_s: Optional[float] = None,
                 pass1_max_duration_s: Optional[float] = None,
                 pass1_max_silence_s: Optional[float] = None,
                 pass1_energy_threshold: Optional[int] = None,
                 # New: pass-2 controls
                 pass2_min_duration_s: float = 0.3,
                 pass2_max_duration_s: Optional[float] = None,
                 pass2_max_silence_s: float = 0.94,
                 pass2_energy_threshold: int = 50,
                 # New: assist processing shaping
                 bandpass_low_hz: int = 200,
                 bandpass_high_hz: int = 4000,
                 drc_threshold_db: float = -24.0,
                 drc_ratio: float = 4.0,
                 drc_attack_ms: float = 5.0,
                 drc_release_ms: float = 100.0,
                 skip_assist_on_loud_dbfs: float = -5.0,
                 # New: fallback and shaping
                 brute_force_fallback: bool = True,
                 brute_force_chunk_s: Optional[float] = None,
                 pad_edges_s: float = 0.0,
                 fade_ms: int = 0,
                 # Accept *_s aliases via kwargs for config flexibility
                 **kwargs):
        """
        Initialize the dynamic audio scene detector.
        """
        if not AUDITOK_AVAILABLE:
            raise ImportError("The 'auditok' library is required for DynamicSceneDetector.")

        # Store method selection
        self.method = str(kwargs.get("method", method))

        # Allow *_s aliases from config (e.g., max_duration_s)
        max_duration = float(kwargs.get("max_duration_s", max_duration))
        min_duration = float(kwargs.get("min_duration_s", min_duration))

        # Core parameters
        self.max_duration = max_duration
        self.min_duration = min_duration

        # Detection and saving audio handling
        self.target_sr = int(kwargs.get("target_sr", target_sr))
        self.force_mono = bool(kwargs.get("force_mono", force_mono))
        self.preserve_original_sr = bool(kwargs.get("preserve_original_sr", preserve_original_sr))

        # DEPRECATED (v1.8.0+, Issue #129): target_sr and preserve_original_sr are no longer used.
        # Auditok handles any sample rate natively; Silero Pass 2 resamples internally.
        # These parameters are kept for backward compatibility but have no effect.
        if "target_sr" in kwargs or target_sr != 16000:
            logger.debug("Note: target_sr parameter is deprecated (v1.8.0+). Auditok uses native sample rate.")
        if "preserve_original_sr" in kwargs:
            logger.debug("Note: preserve_original_sr parameter is deprecated (v1.8.0+). Native SR always preserved.")

        # --- Pass 1 Parameters (Coarse Splitting) ---
        # Defaults map to legacy args if explicit *_s not provided
        self.pass1_min_duration = float(kwargs.get("pass1_min_duration_s", pass1_min_duration_s if pass1_min_duration_s is not None else 0.3))
        self.pass1_max_duration = float(kwargs.get("pass1_max_duration_s", pass1_max_duration_s if pass1_max_duration_s is not None else 2700.0))
        self.pass1_max_silence = float(kwargs.get("pass1_max_silence_s", pass1_max_silence_s if pass1_max_silence_s is not None else max_silence))
        self.pass1_energy_threshold = int(kwargs.get("pass1_energy_threshold", pass1_energy_threshold if pass1_energy_threshold is not None else energy_threshold))

        # --- Pass 2 Parameters (Fine Splitting) ---
        # If pass2_max_duration_s not provided, derive from max_duration - 1.0
        derived_pass2_max = max(self.max_duration - 1.0, self.min_duration)
        self.pass2_min_duration = float(kwargs.get("pass2_min_duration_s", pass2_min_duration_s))
        self.pass2_max_duration = float(kwargs.get("pass2_max_duration_s", pass2_max_duration_s if pass2_max_duration_s is not None else derived_pass2_max))
        self.pass2_max_silence = float(kwargs.get("pass2_max_silence_s", pass2_max_silence_s))
        self.pass2_energy_threshold = int(kwargs.get("pass2_energy_threshold", pass2_energy_threshold))

        # --- Enhancement Flags and Shaping ---
        self.assist_processing = bool(kwargs.get("assist_processing", assist_processing))
        self.verbose_summary = bool(kwargs.get("verbose_summary", verbose_summary))

        self.bandpass_low_hz = int(kwargs.get("bandpass_low_hz", bandpass_low_hz))
        self.bandpass_high_hz = int(kwargs.get("bandpass_high_hz", bandpass_high_hz))
        self.drc_threshold_db = float(kwargs.get("drc_threshold_db", drc_threshold_db))
        self.drc_ratio = float(kwargs.get("drc_ratio", drc_ratio))
        self.drc_attack_ms = float(kwargs.get("drc_attack_ms", drc_attack_ms))
        self.drc_release_ms = float(kwargs.get("drc_release_ms", drc_release_ms))
        self.skip_assist_on_loud_dbfs = float(kwargs.get("skip_assist_on_loud_dbfs", skip_assist_on_loud_dbfs))

        self.brute_force_fallback = bool(kwargs.get("brute_force_fallback", brute_force_fallback))
        # default fallback chunk equals max_duration
        self.brute_force_chunk_s = float(kwargs.get("brute_force_chunk_s", brute_force_chunk_s if brute_force_chunk_s is not None else self.max_duration))
        self.pad_edges_s = float(kwargs.get("pad_edges_s", pad_edges_s))
        self.fade_ms = int(kwargs.get("fade_ms", fade_ms))

        # --- Data Contract: Scene metadata and VAD segments for visualization ---
        # Populated by detect_scenes() for external consumers (e.g., visualization tools)
        self.scenes_detected: List[Dict] = []  # Rich metadata per scene
        self.vad_segments: List[Dict] = []     # VAD speech regions (when using Silero)
        self.coarse_boundaries: List[Dict] = []  # Pass 1 coarse scene boundaries (before splitting)
        self.vad_method: Optional[str] = None  # 'silero' or None
        self.vad_params: Dict = {}             # VAD parameters used

        # Initialize Silero VAD if needed
        if self.method == "silero":
            self._init_silero_vad(kwargs)

        # Initialize Semantic Clustering adapter if needed (lazy loaded)
        self._semantic_adapter = None
        if self.method == "semantic":
            self._init_semantic_adapter(kwargs)

        logger.debug(
            "DynamicSceneDetector cfg: "
            f"method={self.method}, force_mono={self.force_mono}, "
            f"max_dur={self.max_duration}s, min_dur={self.min_duration}s, "
            f"pass1(max_dur={self.pass1_max_duration}, max_sil={self.pass1_max_silence}, thr={self.pass1_energy_threshold}), "
            f"pass2(max_dur={self.pass2_max_duration}, max_sil={self.pass2_max_silence}, thr={self.pass2_energy_threshold}), "
            f"assist={self.assist_processing}"
        )

    def _init_silero_vad(self, config: dict):
        """
        Initialize Silero VAD model and parameters for Pass 2.

        Uses pip-installed silero-vad package (v6.x) for scene detection.
        This is separate from the torch.hub v3.1 used by ASR in whisper_pro_asr.py.
        """
        # Read Silero-specific parameters from flat config structure
        # Default values are conservative for scene detection (not ultra-sensitive)
        self.silero_threshold = float(config.get('silero_threshold', 0.08))
        self.silero_neg_threshold = float(config.get('silero_neg_threshold', 0.15))
        self.silero_min_silence_ms = int(config.get('silero_min_silence_ms', 1500))
        self.silero_min_speech_ms = int(config.get('silero_min_speech_ms', 100))
        self.silero_max_speech_s = float(config.get('silero_max_speech_s', 600))
        self.silero_min_silence_at_max = int(config.get('silero_min_silence_at_max', 500))
        self.silero_speech_pad_ms = int(config.get('silero_speech_pad_ms', 200))

        # Load Silero VAD model (pip-installed version, latest v6.x)
        logger.info("Loading Silero VAD (pip package) for scene detection...")

        try:
            # Import pip-installed silero-vad package
            from silero_vad import load_silero_vad, get_speech_timestamps

            # Load the model (automatically uses latest version from pip)
            self.vad_model = load_silero_vad()

            # Store the get_speech_timestamps function
            self.get_speech_timestamps = get_speech_timestamps

            logger.info("Silero VAD (pip package v6.x) loaded successfully for scene detection")

        except Exception as e:
            logger.error(f"Failed to load Silero VAD: {e}")
            raise RuntimeError(
                "Silero VAD method selected but model failed to load. "
                "Fall back to --scene-detection-method auditok"
            ) from e

    def _init_semantic_adapter(self, config: dict):
        """
        Initialize Semantic Audio Clustering adapter for scene detection.

        Uses lazy import to avoid loading dependencies unless needed.
        """
        logger.info("Initializing Semantic Audio Clustering adapter...")

        try:
            # Lazy import to avoid loading dependencies unless method="semantic"
            from whisperjav.modules.scene_detection_backends.semantic_adapter import (
                SemanticClusteringAdapter,
                SemanticClusteringConfig,
            )

            # Build config from parameters
            adapter_config = SemanticClusteringConfig(
                min_duration=float(config.get("min_duration", 20.0)),
                max_duration=float(config.get("max_duration", 420.0)),
                snap_window=float(config.get("snap_window", 5.0)),
                clustering_threshold=float(config.get("clustering_threshold", 18.0)),
                sample_rate=int(config.get("sample_rate", 16000)),
                preserve_original_sr=bool(config.get("preserve_original_sr", True)),
                visualize=bool(config.get("visualize", False)),
            )

            # Create adapter instance
            self._semantic_adapter = SemanticClusteringAdapter(
                config=adapter_config,
                preset=str(config.get("preset", "default")),
            )

            logger.info("Semantic Audio Clustering adapter initialized")

        except ImportError as e:
            logger.error(f"Failed to load Semantic Audio Clustering: {e}")
            raise RuntimeError(
                "Semantic method selected but dependencies not available. "
                "Ensure 'semantic_audio_clustering' module is installed. "
                "Fall back to --scene-detection-method auditok"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Semantic adapter: {e}")
            raise

    def _detect_pass2_silero(self, region_audio: np.ndarray, sr: int, region_start_sec: float) -> List:
        """
        Use Silero VAD for Pass 2 fine scene splitting.

        Returns: List of timestamp tuples compatible with Auditok format
        """
        import torch
        import librosa

        # Resample to 16kHz for Silero VAD (if needed)
        if sr != 16000:
            audio_16k = librosa.resample(region_audio, orig_sr=sr, target_sr=16000)
        else:
            audio_16k = region_audio

        # Convert to torch tensor
        audio_tensor = torch.FloatTensor(audio_16k)

        # Get speech timestamps from Silero VAD
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=16000,
            threshold=self.silero_threshold,
            neg_threshold=self.silero_neg_threshold,
            min_silence_duration_ms=self.silero_min_silence_ms,
            min_speech_duration_ms=self.silero_min_speech_ms,
            max_speech_duration_s=self.silero_max_speech_s,
            min_silence_at_max_speech=self.silero_min_silence_at_max,
            speech_pad_ms=self.silero_speech_pad_ms,
            return_seconds=True
        )

        # Convert Silero format to Auditok-compatible format
        # Silero returns: [{'start': 0.5, 'end': 3.2}, {'start': 5.0, 'end': 8.5}, ...]
        # Need to return objects with .start and .end attributes (in seconds relative to region)

        class SileroRegion:
            """Wrapper to make Silero timestamps compatible with Auditok interface"""
            def __init__(self, start_sec, end_sec):
                self.start = start_sec  # In seconds
                self.end = end_sec      # In seconds

        sub_regions = [
            SileroRegion(seg['start'], seg['end'])
            for seg in speech_timestamps
        ]

        # Data contract: Store VAD segments with absolute timestamps
        for seg in speech_timestamps:
            self.vad_segments.append({
                "start_sec": round(region_start_sec + seg['start'], 3),
                "end_sec": round(region_start_sec + seg['end'], 3),
            })

        logger.debug(f"Silero VAD detected {len(sub_regions)} segments in Pass 2")

        return sub_regions

    def _apply_assistive_processing(self, audio_chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies a bandpass filter and DRC to enhance speech detection for Pass 2.
        """
        # Skip on loud chunks
        peak_dbfs = 20 * np.log10(np.max(np.abs(audio_chunk)) + 1e-9)
        if peak_dbfs > self.skip_assist_on_loud_dbfs:
            logger.debug(f"Peak {peak_dbfs:.2f} dBFS >= {self.skip_assist_on_loud_dbfs:.2f} dBFS; skipping assist.")
            return audio_chunk

        # Bandpass
        nyquist = 0.5 * sample_rate
        low = max(10.0, float(self.bandpass_low_hz)) / nyquist
        high = min(nyquist - 1.0, float(self.bandpass_high_hz)) / nyquist
        # Guard band limits
        high = min(max(high, low + 1e-4), 0.999)
        b, a = butter(5, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio_chunk.copy())

        # DRC via pydub
        audio_segment = AudioSegment(
            (filtered_audio * 32767).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        compressed_segment = compress_dynamic_range(
            audio_segment,
            threshold=self.drc_threshold_db,
            ratio=self.drc_ratio,
            attack=self.drc_attack_ms,
            release=self.drc_release_ms
        )
        processed_samples = np.array(compressed_segment.get_array_of_samples()).astype(np.float32) / 32768.0

        # Safety clip
        post_peak = np.max(np.abs(processed_samples))
        if post_peak > 1.0:
            processed_samples = np.clip(processed_samples, -1.0, 1.0)

        return processed_samples

    def _save_scene(self, audio_data: np.ndarray, sample_rate: int,
                    scene_idx: int, output_dir: Path,
                    media_basename: str) -> Path:
        """
        Save scene as PCM16 WAV.
        """
        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(scene_path), audio_data, sample_rate, subtype='PCM_16')
        return scene_path

    def _record_scene_metadata(
        self,
        scene_idx: int,
        start_sec: float,
        end_sec: float,
        detection_pass: int,
        scene_path: Optional[Path] = None,
    ) -> None:
        """
        Record rich scene metadata for the data contract.

        Args:
            scene_idx: Scene index (0-based)
            start_sec: Start time in seconds
            end_sec: End time in seconds
            detection_pass: 1 for Pass 1 (coarse/direct), 2 for Pass 2 (fine/split)
            scene_path: Optional path to the saved scene file
        """
        self.scenes_detected.append({
            "scene_index": scene_idx,
            "start_time_seconds": round(start_sec, 3),
            "end_time_seconds": round(end_sec, 3),
            "duration_seconds": round(end_sec - start_sec, 3),
            "detection_pass": detection_pass,
            "filename": scene_path.name if scene_path else None,
        })

    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str) -> List[Tuple[Path, float, float, float]]:
        """
        Splits audio into scenes using a robust two-pass approach.

        Side Effects:
            Populates self.scenes_detected with rich metadata per scene including:
            - scene_index, start_time_seconds, end_time_seconds, duration_seconds
            - detection_pass: 1 (coarse/direct) or 2 (fine/split)

            When using Silero VAD, also populates:
            - self.vad_segments: List of {'start_sec', 'end_sec'} speech regions
            - self.vad_method: 'silero'
            - self.vad_params: VAD parameters used
        """
        logger.info(f"Starting dynamic scene detection for: {audio_path}")

        # Reset data contract fields for fresh detection run
        self.scenes_detected = []
        self.vad_segments = []
        self.coarse_boundaries = []
        self.vad_method = self.method if self.method in ("silero", "semantic") else None
        if self.method == "silero":
            self.vad_params = {
                "threshold": self.silero_threshold,
                "neg_threshold": self.silero_neg_threshold,
                "min_silence_duration_ms": self.silero_min_silence_ms,
                "min_speech_duration_ms": self.silero_min_speech_ms,
                "max_speech_duration_s": self.silero_max_speech_s,
                "speech_pad_ms": self.silero_speech_pad_ms,
            }

        # --- Semantic Clustering: Delegate entirely to adapter ---
        if self.method == "semantic":
            if self._semantic_adapter is None:
                raise RuntimeError(
                    "Semantic adapter not initialized. "
                    "This should not happen if method='semantic' was set correctly."
                )

            # Delegate to semantic adapter
            scene_tuples = self._semantic_adapter.detect_scenes(
                audio_path=audio_path,
                output_dir=output_dir,
                media_basename=media_basename,
            )

            # Copy data contract fields from adapter to this instance
            self.scenes_detected = self._semantic_adapter.scenes_detected
            self.vad_segments = self._semantic_adapter.vad_segments
            self.coarse_boundaries = self._semantic_adapter.coarse_boundaries
            self.vad_method = self._semantic_adapter.vad_method
            self.vad_params = self._semantic_adapter.vad_params

            logger.info(f"Semantic scene detection complete: {len(scene_tuples)} scenes")
            return scene_tuples

        # --- Auditok/Silero: Original two-pass approach ---

        # Load audio once from disk - NO resampling needed for scene detection
        # Auditok handles any sample rate natively. Silero Pass 2 resamples internally.
        # This eliminates the 10-30 minute resample bottleneck for 2+ hour files (Issue #129).
        try:
            original_audio, orig_sr = load_audio_unified(audio_path, target_sr=None, force_mono=self.force_mono)
            # Use native sample rate - auditok is SR-agnostic, silero handles its own resampling
            detection_audio = original_audio
            det_sr = orig_sr
            det_duration = len(detection_audio) / det_sr
            logger.info(f"Audio loaded: {det_duration:.1f}s @ {det_sr}Hz (native SR, no resample needed)")

            # Warn user about long files (Issue #129 hardening)
            duration_hours = det_duration / 3600
            if duration_hours > 1.5:
                # Estimate memory usage: samples * 4 bytes (float32)
                estimated_gb = (len(detection_audio) * 4) / (1024 ** 3)
                logger.warning(
                    f"Large file detected: {duration_hours:.1f} hours (~{estimated_gb:.1f} GB memory). "
                    f"Processing may take several minutes. This is normal."
                )
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            return []

        # Single audio stream for both detection and saving (no dual storage)
        # After Issue #129 fix, detection_audio IS original_audio at native SR
        save_audio = detection_audio
        save_sr = det_sr
        total_duration_s = len(save_audio) / save_sr  # seconds baseline

        # Pass 1 on detection audio
        logger.info(f"Pass 1: Starting auditok scene detection on {det_duration:.1f}s audio @ {det_sr}Hz...")
        det_bytes = (detection_audio * 32767).astype(np.int16).tobytes()
        pass1_params = {
            "sampling_rate": det_sr,
            "channels": 1,
            "sample_width": 2,
            "min_dur": self.pass1_min_duration,
            "max_dur": self.pass1_max_duration,
            "max_silence": min(det_duration * 0.95, self.pass1_max_silence),
            "energy_threshold": self.pass1_energy_threshold,
            "drop_trailing_silence": True
        }
        story_lines = list(auditok.split(det_bytes, **pass1_params))
        logger.info(f"Pass 1: Found {len(story_lines)} coarse story line(s).")
        if len(story_lines) > 50:
            logger.info(f"Pass 2: Processing {len(story_lines)} story lines (this may take a moment for long files)...")

        # Data contract: Capture coarse boundaries BEFORE any splitting (for visualization)
        for idx, region in enumerate(story_lines):
            self.coarse_boundaries.append({
                "scene_index": idx,
                "start_time_seconds": round(region.start, 3),
                "end_time_seconds": round(region.end, 3),
                "duration_seconds": round(region.end - region.start, 3),
            })

        final_scene_tuples: List[Tuple[Path, float, float, float]] = []
        scene_idx = 0

        # Counters
        storyline_direct_saves = 0
        granular_segments_count = 0
        brute_force_segments_count = 0

        # Helper to clamp with optional edge padding
        def clamp_with_pad(s: float, e: float) -> Tuple[float, float]:
            s_pad = max(0.0, s - self.pad_edges_s)
            e_pad = min(total_duration_s, e + self.pad_edges_s)
            if e_pad < s_pad:
                e_pad = s_pad
            return s_pad, e_pad

        # Progress tracking for long file processing
        total_storylines = len(story_lines)
        last_logged_pct = -1

        for region_idx, region in enumerate(story_lines):
            # Log progress at 0%, 25%, 50%, 75%, 100% (or every 15 for small batches)
            current_pct = int((region_idx / max(total_storylines, 1)) * 100)
            if total_storylines > 20 and current_pct >= last_logged_pct + 25:
                logger.info(f"Pass 2: Processing story line {region_idx + 1}/{total_storylines} ({current_pct}%)")
                last_logged_pct = current_pct
            elif total_storylines <= 20 and (region_idx % 5 == 0 or region_idx == total_storylines - 1):
                logger.debug(f"Processing story line {region_idx + 1}/{total_storylines}")

            region_start_sec = region.start
            region_end_sec = region.end
            region_duration = region_end_sec - region_start_sec

            # Direct save if short enough
            if region_duration <= self.max_duration and region_duration >= self.min_duration:
                s_sec, e_sec = clamp_with_pad(region_start_sec, region_end_sec)
                start_sample = int(s_sec * save_sr)
                end_sample = int(e_sec * save_sr)
                scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                # Data contract: Record Pass 1 scene (coarse/direct)
                self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=1, scene_path=scene_path)
                scene_idx += 1
                storyline_direct_saves += 1
                continue

            # Pass 2 on detection audio chunk
            det_start = int(region_start_sec * det_sr)
            det_end = int(region_end_sec * det_sr)
            region_audio_det = detection_audio[det_start:det_end]

            # Pass 2: Method-dependent detection
            if self.method == "silero":
                # Use Silero VAD for Pass 2
                sub_regions = self._detect_pass2_silero(
                    region_audio_det,
                    det_sr,
                    region_start_sec
                )
            else:
                # Existing Auditok Pass 2 (keep as-is)
                audio_for_detection = region_audio_det
                if self.assist_processing:
                    audio_for_detection = self._apply_assistive_processing(region_audio_det, det_sr)

                region_bytes_for_detection = (audio_for_detection * 32767).astype(np.int16).tobytes()
                pass2_params = {
                    "sampling_rate": det_sr,
                    "channels": 1,
                    "sample_width": 2,
                    "min_dur": self.pass2_min_duration,
                    "max_dur": self.pass2_max_duration,
                    "max_silence": min(region_duration * 0.95, self.pass2_max_silence),
                    "energy_threshold": self.pass2_energy_threshold,
                    "drop_trailing_silence": True
                }
                sub_regions = list(auditok.split(region_bytes_for_detection, **pass2_params))

            if sub_regions:
                granular_segments_count += len(sub_regions)
                logger.debug(f"Pass 2 split region into {len(sub_regions)} sub-scenes.")
                for sub in sub_regions:
                    sub_start = region_start_sec + sub.start
                    sub_end = region_start_sec + sub.end
                    sub_dur = sub_end - sub_start
                    if sub_dur < self.min_duration:
                        continue
                    s_sec, e_sec = clamp_with_pad(sub_start, sub_end)
                    start_sample = int(s_sec * save_sr)
                    end_sample = int(e_sec * save_sr)
                    scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                    final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                    # Data contract: Record Pass 2 scene (fine/split)
                    self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=2, scene_path=scene_path)
                    scene_idx += 1
            else:
                if not self.brute_force_fallback:
                    logger.warning(f"Pass 2 found no sub-regions in region {region_idx}; skipping fallback.")
                    continue
                logger.warning(f"Pass 2 found no sub-regions in region {region_idx}, using brute-force splitting.")
                chunk_len = self.brute_force_chunk_s
                num_chunks = int(np.ceil(region_duration / max(chunk_len, self.min_duration)))
                brute_force_segments_count += num_chunks
                for i in range(num_chunks):
                    sub_start = region_start_sec + i * chunk_len
                    sub_end = min(region_start_sec + (i + 1) * chunk_len, region_end_sec)
                    sub_dur = sub_end - sub_start
                    if sub_dur < self.min_duration:
                        continue
                    s_sec, e_sec = clamp_with_pad(sub_start, sub_end)
                    start_sample = int(s_sec * save_sr)
                    end_sample = int(e_sec * save_sr)
                    scene_path = self._save_scene(save_audio[start_sample:end_sample], save_sr, scene_idx, output_dir, media_basename)
                    final_scene_tuples.append((scene_path, s_sec, e_sec, e_sec - s_sec))
                    # Data contract: Record brute-force scene (treated as Pass 2 fallback)
                    self._record_scene_metadata(scene_idx, s_sec, e_sec, detection_pass=2, scene_path=scene_path)
                    scene_idx += 1

        if self.verbose_summary:
            # Calculate duration statistics
            durations = [duration for _, _, _, duration in final_scene_tuples]
            if durations:
                min_duration = min(durations)
                max_duration = max(durations)
                mean_duration = sum(durations) / len(durations)
                
                # Calculate median duration
                sorted_durations = sorted(durations)
                n = len(sorted_durations)
                if n % 2 == 0:
                    median_duration = (sorted_durations[n//2 - 1] + sorted_durations[n//2]) / 2
                else:
                    median_duration = sorted_durations[n//2]
                
                total_scene_duration = sum(durations)
                
                summary_lines = [
                    "", "="*50,
                    "Dynamic Scene Detection Summary",
                    "="*50,
                    f"Total Story Lines Found: {len(story_lines)}",
                    f" - Segments saved directly: {storyline_direct_saves}",
                    f" - Segments from granular split (Pass 2): {granular_segments_count}",
                    f" - Segments from brute-force split: {brute_force_segments_count}",
                    "-" * 50,
                    f"Total Final Scenes Saved: {len(final_scene_tuples)}",
                    f"Scene Duration Statistics:",
                    f" - Shortest: {min_duration:.2f}s",
                    f" - Longest: {max_duration:.2f}s", 
                    f" - Mean: {mean_duration:.2f}s",
                    f" - Median: {median_duration:.2f}s",
                    f" - Total: {total_scene_duration:.1f}s ({total_scene_duration/60:.1f}m)",
                    "="*50, ""
                ]
            else:
                summary_lines = [
                    "", "="*50,
                    "Dynamic Scene Detection Summary",
                    "="*50,
                    f"Total Story Lines Found: {len(story_lines)}",
                    f" - Segments saved directly: {storyline_direct_saves}",
                    f" - Segments from granular split (Pass 2): {granular_segments_count}",
                    f" - Segments from brute-force split: {brute_force_segments_count}",
                    "-" * 50,
                    f"Total Final Scenes Saved: {len(final_scene_tuples)}",
                    "="*50, ""
                ]
            logger.info('\n'.join(summary_lines))

        logger.info(f"Detected and saved {len(final_scene_tuples)} final scenes.")
        return final_scene_tuples

    def get_detection_metadata(self) -> Dict:
        """
        Export scene detection metadata for the data contract.

        Returns a dictionary suitable for saving to the master metadata JSON file.
        Should be called after detect_scenes() to get populated data.

        Returns:
            Dict with 'scenes_detected', 'coarse_boundaries', 'vad_segments', 'vad_method', 'vad_params'
        """
        return {
            "scenes_detected": self.scenes_detected,
            "coarse_boundaries": self.coarse_boundaries if self.coarse_boundaries else None,
            "vad_segments": self.vad_segments if self.vad_segments else None,
            "vad_method": self.vad_method,
            "vad_params": self.vad_params if self.vad_params else None,
        }




def analyze_scene_metadata(metadata_path: Path, original_audio_path: Path):
    """Analyze scene detection results from metadata JSON file."""
    import matplotlib.pyplot as plt
    from scipy import stats
    
    logger.info(f"Analyzing scene metadata from: {metadata_path}")
    
    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not metadata:
        logger.error("No metadata found in file")
        return
    
    # Get original audio duration
    audio_info = sf.info(str(original_audio_path))
    total_duration = audio_info.duration
    
    # Extract segment info
    segments = [(item['start_time_s'], item['end_time_s'], item['duration_s'], item.get('energy_db', 0)) 
                for item in metadata]
    segments.sort(key=lambda x: x[0])  # Sort by start time
    
    # Calculate statistics
    durations = [seg[2] for seg in segments]
    energies = [seg[3] for seg in segments if seg[3] != -np.inf]
    
    print("\n" + "="*80)
    print("SCENE DETECTION ANALYSIS REPORT")
    print("="*80)
    
    print(f"\nOriginal Audio: {original_audio_path.name}")
    print(f"Total Duration: {total_duration:.2f}s ({total_duration/60:.2f} minutes)")
    print(f"Number of Scenes: {len(segments)}")
    
    # Coverage analysis
    total_scene_duration = sum(durations)
    coverage_percent = (total_scene_duration / total_duration) * 100
    print(f"\nCoverage:")
    print(f"  Total Scene Duration: {total_scene_duration:.2f}s")
    print(f"  Coverage: {coverage_percent:.1f}%")
    print(f"  Discarded Audio: {total_duration - total_scene_duration:.2f}s ({100-coverage_percent:.1f}%)")
    
    # Duration statistics
    print(f"\nDuration Statistics:")
    print(f"  Min: {np.min(durations):.2f}s")
    print(f"  Max: {np.max(durations):.2f}s")
    print(f"  Mean: {np.mean(durations):.2f}s")
    print(f"  Median: {np.median(durations):.2f}s")
    print(f"  Std Dev: {np.std(durations):.2f}s")
    
    # Energy statistics
    if energies:
        print(f"\nEnergy Statistics:")
        print(f"  Min: {np.min(energies):.2f} dB")
        print(f"  Max: {np.max(energies):.2f} dB")
        print(f"  Mean: {np.mean(energies):.2f} dB")
        print(f"  Median: {np.median(energies):.2f} dB")
    
    # Gap analysis
    gaps = []
    for i in range(1, len(segments)):
        gap_start = segments[i-1][1]
        gap_end = segments[i][0]
        gap_duration = gap_end - gap_start
        if gap_duration > 0.001:  # Ignore tiny floating point differences
            gaps.append((gap_start, gap_end, gap_duration))
    
    if gaps:
        gap_durations = [g[2] for g in gaps]
        print(f"\nGap Analysis:")
        print(f"  Number of Gaps: {len(gaps)}")
        print(f"  Total Gap Duration: {sum(gap_durations):.2f}s")
        print(f"  Min Gap: {np.min(gap_durations):.3f}s")
        print(f"  Max Gap: {np.max(gap_durations):.3f}s")
        print(f"  Mean Gap: {np.mean(gap_durations):.3f}s")
        
        # Top 5 longest gaps
        print(f"\n  Top 5 Longest Gaps:")
        sorted_gaps = sorted(gaps, key=lambda x: x[2], reverse=True)[:5]
        for i, (start, end, duration) in enumerate(sorted_gaps):
            print(f"    {i+1}. {start:.2f}s - {end:.2f}s ({duration:.3f}s)")
    
    # Check for overlaps
    overlaps = []
    for i in range(1, len(segments)):
        if segments[i][0] < segments[i-1][1]:
            overlap_duration = segments[i-1][1] - segments[i][0]
            overlaps.append((i-1, i, overlap_duration))
    
    if overlaps:
        print(f"\nWARNING: Found {len(overlaps)} overlapping segments!")
        for prev_idx, curr_idx, overlap in overlaps:
            print(f"  Segments {prev_idx} and {curr_idx} overlap by {overlap:.3f}s")
    
    # Distribution analysis
    print(f"\nDuration Distribution:")
    bins = [0, 1, 5, 10, 15, 20, 25, 30, float('inf')]
    bin_labels = ['0-1s', '1-5s', '5-10s', '10-15s', '15-20s', '20-25s', '25-30s', '30s+']
    hist, _ = np.histogram(durations, bins=bins)
    
    for label, count in zip(bin_labels, hist):
        if count > 0:
            percent = (count / len(durations)) * 100
            print(f"  {label}: {count} scenes ({percent:.1f}%)")
    
    # Create visualization if matplotlib is available
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Scene Detection Analysis - {original_audio_path.name}', fontsize=16)
        
        # Duration histogram
        ax = axes[0, 0]
        ax.hist(durations, bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Duration (s)')
        ax.set_ylabel('Count')
        ax.set_title('Scene Duration Distribution')
        ax.axvline(np.mean(durations), color='red', linestyle='--', label=f'Mean: {np.mean(durations):.2f}s')
        ax.axvline(np.median(durations), color='green', linestyle='--', label=f'Median: {np.median(durations):.2f}s')
        ax.legend()
        
        # Timeline visualization
        ax = axes[0, 1]
        for i, (start, end, duration, _) in enumerate(segments):
            ax.barh(0, duration, left=start, height=0.5, alpha=0.7, edgecolor='black')
        ax.set_xlim(0, total_duration)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time (s)')
        ax.set_yticks([])
        ax.set_title('Timeline Coverage')
        
        # Energy distribution
        ax = axes[1, 0]
        if energies:
            ax.hist(energies, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Energy (dB)')
            ax.set_ylabel('Count')
            ax.set_title('Scene Energy Distribution')
        else:
            ax.text(0.5, 0.5, 'No energy data available', ha='center', va='center')
            ax.set_title('Scene Energy Distribution')
        
        # Gap duration distribution
        ax = axes[1, 1]
        if gaps:
            gap_durations = [g[2] for g in gaps]
            ax.hist(gap_durations, bins=20, edgecolor='black', alpha=0.7)
            ax.set_xlabel('Gap Duration (s)')
            ax.set_ylabel('Count')
            ax.set_title('Gap Duration Distribution')
        else:
            ax.text(0.5, 0.5, 'No gaps found', ha='center', va='center')
            ax.set_title('Gap Duration Distribution')
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = metadata_path.parent / f"{metadata_path.stem}_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved to: {plot_path}")
        plt.close()
        
    except ImportError:
        logger.info("Matplotlib not available, skipping visualization")
    except Exception as e:
        logger.warning(f"Failed to create visualization: {e}")
    
    print("\n" + "="*80)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Robust Audio Scene Detector with CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_audio", type=Path, help="Path to the input audio file.")
    parser.add_argument("output_dir", type=Path, help="Directory to save the output scene files.")
    parser.add_argument("--media_basename", type=str, default=None, help="Basename for output files. Defaults to input file name.")
    parser.add_argument("--metadata_file", type=Path, default=None, help="Optional path to save a JSON metadata file.")
    parser.add_argument("--min_duration", type=float, default=0.3, help="Minimum scene duration in seconds.")
    parser.add_argument("--max_duration", type=float, default=29.0, help="Maximum scene duration in seconds.")
    parser.add_argument("--use_drc", action="store_true", help="Enable Dynamic Range Compression.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze existing metadata file without running detection.")
    
    args = parser.parse_args()
    # Set logger level from CLI
    logger.setLevel(args.log_level)
    
    if not args.input_audio.exists():
        logger.error(f"Input file not found: {args.input_audio}")
        sys.exit(1)
    
    # Use the input file's stem as the default basename if not provided
    if args.media_basename is None:
        args.media_basename = args.input_audio.stem
        
    # Construct metadata path if requested but not explicitly named
    if args.metadata_file is not None and args.metadata_file.is_dir():
         args.metadata_file = args.metadata_file / f"{args.media_basename}_metadata.json"
    elif args.metadata_file is None:
        # Default to saving metadata in the output directory
        args.metadata_file = args.output_dir / f"{args.media_basename}_metadata.json"
    
    if args.analyze_only:
        # Just run analysis on existing metadata
        if not args.metadata_file.exists():
            logger.error(f"Metadata file not found: {args.metadata_file}")
            sys.exit(1)
        analyze_scene_metadata(args.metadata_file, args.input_audio)
    else:
        # Initialize the detector with CLI arguments
        detector = AdaptiveSceneDetector(
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            use_drc=args.use_drc
            # Other parameters can be exposed here as needed
        )
        # Run detection
        detector.detect_scenes(
            audio_path=args.input_audio,
            output_dir=args.output_dir,
            media_basename=args.media_basename,
            metadata_path=args.metadata_file
        )
        
        # Run analysis after detection
        if args.metadata_file and args.metadata_file.exists():
            print("\nRunning post-detection analysis...")
            analyze_scene_metadata(args.metadata_file, args.input_audio)