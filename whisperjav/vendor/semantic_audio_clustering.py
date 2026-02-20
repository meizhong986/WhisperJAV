"""
Acoustic Scene Segmenter V7: Long Form & Silence Snapping
=========================================================
Changes from V6:
1. MIN_DUR increased to 20s (Segments are ~2x longer).
2. SPLIT LOGIC changed to 'Snap to Silence' (cuts at lowest energy).
3. FULL COVERAGE guaranteed (0.0 -> Duration).
4. Class naming fixed (SemanticSegmenter).
"""

import numpy as np
import librosa
import soundfile as sf
import json
import os
import argparse
import warnings
import shutil
import subprocess
import tempfile
import logging
import time
from typing import Optional, Callable, List, Dict
from scipy.ndimage import median_filter
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
from enum import Enum

warnings.filterwarnings('ignore')

__version__ = "7.0.0"

# ==========================================
# 0. EXCEPTIONS
# ==========================================

class SegmentationError(Exception):
    """Base exception for segmentation errors."""
    pass

class AudioLoadError(SegmentationError):
    """Failed to load audio file."""
    pass

class FeatureExtractionError(SegmentationError):
    """Feature extraction failed."""
    pass

# ==========================================
# 1. CONFIG & REGISTRY
# ==========================================

@dataclass
class SegmentationConfig:
    """
    Configuration for the Semantic Audio Clustering pipeline.
    
    Attributes:
        min_duration (float): Minimum duration of a segment in seconds. 
                              Segments shorter than this will be merged. Default: 20.0
        max_duration (float): Maximum duration of a segment in seconds.
                              Segments longer than this will be split. Default: 420.0 (7 mins)
        snap_window (float):  Window size in seconds for snapping boundaries to silence. Default: 5.0
        sample_rate (int):    Target sample rate for processing. Default: 16000
        chunk_duration (int): Duration of audio chunks for streaming in seconds. Default: 60
        viz_duration (int):   Maximum duration in seconds to visualize in the plot. Default: 300
    """
    min_duration: float = 20.0
    max_duration: float = 420.0
    snap_window: float = 5.0
    sample_rate: int = 16000
    chunk_duration: int = 60
    viz_duration: int = 300

    # Feature Extraction
    n_mfcc: int = 13
    hop_length: int = 512
    min_samples_multiplier: int = 10
    
    # Clustering
    fps: int = 31
    smoothing_window: int = 15
    clustering_threshold: float = 18.0
    rms_smoothing_window: int = 5
    
    # Classification
    silence_threshold_multiplier: float = 1.5
    std_rms_threshold: float = 0.02

class SceneType(Enum):
    QUIET_DIALOGUE = "quiet_dialogue"
    NOISY_DIALOGUE = "noisy_dialogue"  
    MUSIC_DOMINANT = "music_dominant"
    HIGH_ENERGY = "high_energy"
    SILENCE = "silence"
    MIXED = "mixed_content"            
    GAP_FILLER = "silence"
    AMBIENT = "ambient_noise"

class FeatureRegistry:
    """Central source of truth for feature indices."""
    MFCC = slice(0, 13)       
    DELTA = slice(13, 26)     
    RMS = 26                  
    ZCR = 27                  
    CONTRAST = slice(28, 35)  
    CHROMA_STD = 35           
    TOTAL_DIM = 36

@dataclass
class Segment:
    start: float
    end: float
    scene_type: SceneType
    confidence: float
    avg_db: float
    
    def to_dict(self):
        # Safe padding: buffer around each segment for ASR extraction.
        # 0.5s accommodates Japanese soft consonant onsets and trailing
        # particles/vowels that sit near the energy minimum.
        pad = 0.5
        safe_start = max(0.0, self.start - pad)
        # Note: We don't clamp the end to duration here because we don't always 
        # have the total duration handy in this class, but ffmpeg handles over-reading fine.
        safe_end = self.end + pad

        return {
            # 1. STRICT TIMESTAMPS (For SRT / Timeline / Database)
            # Continuous, no gaps, no overlap.
            "timestamps": {
                "start": round(self.start, 3),
                "end": round(self.end, 3),
                "duration": round(self.end - self.start, 3)
            },
            # 2. BUFFERED TIMESTAMPS (For ASR Processing)
            # Overlapping. Use THESE for ffmpeg extraction.
            "asr_processing": {
                "start": round(safe_start, 3),
                "end": round(safe_end, 3),
                "duration": round(safe_end - safe_start, 3),
                "padding_applied": pad
            },
            "context": {
                "label": self.scene_type.value,
                "confidence": round(self.confidence, 2),
                "loudness_db": round(self.avg_db, 1),
            },
            "asr_prompt": self._get_prompt()
        }

    def _get_prompt(self):
        prompts = {
            SceneType.NOISY_DIALOGUE: "Dialogue mixed with rhythmic background noise. Transcribe speech only.",
            SceneType.MUSIC_DOMINANT: "Music playing. Transcribe lyrics only if clearly audible.",
            SceneType.QUIET_DIALOGUE: "Clear dialogue. Transcribe accurately.",
            SceneType.HIGH_ENERGY: "Loud chaotic audio. Focus on speech.",
            SceneType.MIXED: "Dialogue with background music. Focus on the speech.",
            SceneType.SILENCE: "Silence."
        }
        return prompts.get(self.scene_type, "Transcribe speech.")

# ==========================================
# 2. STREAMING EXTRACTION
# ==========================================

def convert_to_temp_wav(input_path, target_sr=16000):
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found.")
    fd, temp_path = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    cmd = ['ffmpeg', '-y', '-i', input_path, '-ar', str(target_sr), '-ac', '1', '-vn', temp_path]
    try:
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_path
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise RuntimeError(f"ffmpeg conversion failed: {e}") from e

class StreamFeatureExtractor:
    def __init__(self, config: SegmentationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.sr = config.sample_rate
        self.chunk_dur = config.chunk_duration
        self.hop_length = config.hop_length
        self.logger = logger

    def _log(self, msg, level=logging.INFO):
        if self.logger: self.logger.log(level, msg)
        else: print(msg)

    def extract(self, file_path, progress_callback: Optional[Callable[[float, str], None]] = None) -> tuple:
        if progress_callback: progress_callback(0.0, "Initializing feature extraction...")
        self._log(f"--> [1/5] Streaming features ({self.chunk_dur}s chunks)...")
        temp_wav = None
        process_path = file_path
        
        try:
            sf.info(file_path)
        except Exception:
            try:
                if progress_callback: progress_callback(0.05, "Converting to WAV...")
                temp_wav = convert_to_temp_wav(file_path, self.sr)
                process_path = temp_wav
            except Exception as e:
                self._log(f"Error: {e}", logging.ERROR)
                raise AudioLoadError(f"Failed to convert/load audio: {e}")

        try:
            info = sf.info(process_path)
            total_dur = info.duration
            native_sr = info.samplerate
            block_size = int(self.chunk_dur * native_sr)
            feature_list = []
            
            # Estimate total blocks for progress
            total_blocks = int(np.ceil(info.frames / block_size))

            for i, block in enumerate(sf.blocks(process_path, blocksize=block_size, always_2d=True)):
                # Update progress (0.1 to 0.8 range reserved for extraction)
                if progress_callback:
                    p = 0.1 + (0.7 * (i / total_blocks))
                    progress_callback(p, f"Extracting features: Chunk {i+1}/{total_blocks}")

                if block.shape[1] > 1: y = np.mean(block, axis=1)
                else: y = block.flatten()
                
                if native_sr != self.sr:
                    y = librosa.resample(y, orig_sr=native_sr, target_sr=self.sr)

                min_samples = self.hop_length * self.config.min_samples_multiplier
                if len(y) < min_samples:
                    y = np.pad(y, (0, min_samples - len(y)), mode='constant')

                # Extract
                try:
                    mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=self.config.n_mfcc)
                    delta = librosa.feature.delta(mfcc)
                    rms = librosa.feature.rms(y=y)
                    zcr = librosa.feature.zero_crossing_rate(y=y)
                    contrast = librosa.feature.spectral_contrast(y=y, sr=self.sr)
                    chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
                    chroma_std = np.std(chroma, axis=0, keepdims=True)

                    feats = np.vstack([mfcc, delta, rms, zcr, contrast, chroma_std])
                    feature_list.append(feats)
                except Exception as e:
                    raise FeatureExtractionError(f"Extraction failed at chunk {i}: {e}")

            full_features = np.hstack(feature_list)
            times = librosa.frames_to_time(np.arange(full_features.shape[1]), sr=self.sr, hop_length=self.hop_length)
            return full_features, times, total_dur

        finally:
            if temp_wav and os.path.exists(temp_wav):
                try: 
                    os.remove(temp_wav)
                except Exception as e:
                    self._log(f"Warning: Failed to remove temp file {temp_wav}: {e}", logging.WARNING)

# ==========================================
# 3. SEMANTIC SEGMENTATION
# ==========================================

class SemanticSegmenter:
    def __init__(self, config: SegmentationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.min_dur = config.min_duration
        self.max_dur = config.max_duration
        self.logger = logger

    def _log(self, msg, level=logging.INFO):
        if self.logger: self.logger.log(level, msg)
        else: print(msg)

    def segment(self, features, times, duration):
        self._log("--> [2/5] Clustering, Merging & Snapping to Silence...")
        
        # 1. Pre-clustering Smoothing
        fps = self.config.fps 
        feats_smooth = median_filter(features, size=(1, self.config.smoothing_window)) 
        
        step = int(fps * 0.5) 
        X = feats_smooth[:, ::step].T
        X_times = times[::step]
        
        # 2. Dynamic Clustering
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=self.config.clustering_threshold, linkage='ward')
        labels = clusterer.fit_predict(X_scaled)
        
        # 3. Raw Boundaries
        boundaries = [0.0]
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                boundaries.append(X_times[i])
        boundaries.append(duration)
        
        # 4. Snap to SILENCE (Lowest Energy)
        # This replaces the old Flux snapping to prioritize speech safety
        boundaries = self._snap_to_silence(boundaries, features, times)
        
        # 5. Smart Merge (Semantic)
        segments = self._smart_merge(boundaries, features, times)
        
        # 6. Forced Cleanup
        segments = self._forced_cleanup(segments)
        
        # 7. ABSOLUTE COVERAGE CHECK
        segments = self._ensure_timeline_coverage(segments, duration)
        
        return segments

    def _snap_to_silence(self, boundaries, features, times):
        """Refines boundaries by sliding them to the Local Minimum Energy (RMS)."""
        self._log("    -> Snapping cut points to lowest energy (silence)...")
        
        # Extract RMS curve
        rms_curve = features[FeatureRegistry.RMS, :]
        rms_smooth = median_filter(rms_curve, size=self.config.rms_smoothing_window)

        refined = [0.0]
        search_window = self.config.snap_window # Look +/- window seconds
        
        for b in boundaries[1:-1]:
            # Search range
            start_idx = max(0, np.searchsorted(times, b - search_window))
            end_idx = min(len(rms_smooth), np.searchsorted(times, b + search_window))
            
            if end_idx > start_idx:
                # Find MINIMUM energy in this window
                window_slice = rms_smooth[start_idx:end_idx]
                local_min_idx = np.argmin(window_slice) + start_idx
                refined.append(times[local_min_idx])
            else:
                refined.append(b)
                
        refined.append(boundaries[-1])
        return sorted(list(set(refined)))

    def _smart_merge(self, boundaries, features, times):
        segments = []
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            mask = (times >= start) & (times < end)
            if not np.any(mask): continue
            
            seg_feat = np.mean(features[:, mask], axis=1)
            segments.append({"start": start, "end": end, "vec": seg_feat, "dur": end-start})

        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(segments):
                seg = segments[i]
                if seg["dur"] >= self.min_dur:
                    i += 1
                    continue
                
                left = segments[i-1] if i > 0 else None
                right = segments[i+1] if i < len(segments)-1 else None
                merge_target = None 
                
                if left and right:
                    sim_l = cosine_similarity([seg["vec"]], [left["vec"]])[0][0]
                    sim_r = cosine_similarity([seg["vec"]], [right["vec"]])[0][0]
                    if sim_l >= sim_r:
                        if (left["dur"] + seg["dur"]) <= self.max_dur: merge_target = -1
                        elif (right["dur"] + seg["dur"]) <= self.max_dur: merge_target = 1
                    else:
                        if (right["dur"] + seg["dur"]) <= self.max_dur: merge_target = 1
                        elif (left["dur"] + seg["dur"]) <= self.max_dur: merge_target = -1
                elif left and (left["dur"] + seg["dur"]) <= self.max_dur: merge_target = -1
                elif right and (right["dur"] + seg["dur"]) <= self.max_dur: merge_target = 1
                
                if merge_target == -1:
                    new_dur = left["dur"] + seg["dur"]
                    new_vec = (left["vec"]*left["dur"] + seg["vec"]*seg["dur"]) / new_dur
                    segments[i-1] = {"start": left["start"], "end": seg["end"], "dur": new_dur, "vec": new_vec}
                    del segments[i]
                    changed = True
                elif merge_target == 1:
                    new_dur = right["dur"] + seg["dur"]
                    new_vec = (right["vec"]*right["dur"] + seg["vec"]*seg["dur"]) / new_dur
                    segments[i+1] = {"start": seg["start"], "end": right["end"], "dur": new_dur, "vec": new_vec}
                    del segments[i]
                    changed = True
                else:
                    i += 1
        return segments

    def _forced_cleanup(self, segments):
        final_segments = []
        if not segments: return []
        curr = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            if curr["dur"] < self.min_dur:
                total_dur = curr["dur"] + next_seg["dur"]
                weighted_vec = (curr["vec"]*curr["dur"] + next_seg["vec"]*next_seg["dur"]) / total_dur
                curr = {"start": curr["start"], "end": next_seg["end"], "dur": total_dur, "vec": weighted_vec}
            else:
                final_segments.append(curr)
                curr = next_seg
        if curr["dur"] < self.min_dur and final_segments:
            last = final_segments.pop()
            total_dur = last["dur"] + curr["dur"]
            curr = {"start": last["start"], "end": curr["end"], "dur": total_dur, "vec": last["vec"]}
        final_segments.append(curr)
        return final_segments

    def _ensure_timeline_coverage(self, segments, duration):
        covered = []
        if not segments:
            return [{"start": 0.0, "end": duration, "dur": duration, "vec": np.zeros(36)}]

        if segments[0]["start"] > 0.001:
            covered.append({"start": 0.0, "end": segments[0]["start"], "dur": segments[0]["start"], "vec": np.zeros(36)})

        covered.append(segments[0])
        for i in range(1, len(segments)):
            prev = covered[-1]
            curr = segments[i]
            gap = curr["start"] - prev["end"]
            
            if gap > 0.001:
                covered.append({"start": prev["end"], "end": curr["start"], "dur": gap, "vec": np.zeros(36)})
            
            if curr["start"] < prev["end"]:
                curr["start"] = prev["end"]
                curr["dur"] = curr["end"] - curr["start"]
            
            covered.append(curr)

        last = covered[-1]
        if last["end"] < (duration - 0.001):
            covered.append({"start": last["end"], "end": duration, "dur": duration - last["end"], "vec": np.zeros(36)})
        elif last["end"] > duration:
            last["end"] = duration
            last["dur"] = last["end"] - last["start"]

        return covered

# ==========================================
# 4. ADAPTIVE CLASSIFIER
# ==========================================

class AdaptiveClassifier:
    def __init__(self, config: SegmentationConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.stats = {} 
        self.logger = logger

    def _log(self, msg, level=logging.INFO):
        if self.logger: self.logger.log(level, msg)
        else: print(msg)

    def calibrate(self, features):
        self._log("--> [3/5] Calibrating thresholds...")
        rms = features[FeatureRegistry.RMS, :]
        chroma_std = features[FeatureRegistry.CHROMA_STD, :]
        contrast = np.mean(features[FeatureRegistry.CONTRAST, :], axis=0)
        
        self.stats = {
            "rms_base": np.percentile(rms, 20),
            "rms_peak": np.percentile(rms, 85),
            "contrast_high": np.percentile(contrast, 75),
            "chroma_high": np.percentile(chroma_std, 75),
            "contrast_low": np.percentile(contrast, 25)
        }
        self.stats["rms_base"] = max(self.stats["rms_base"], 0.001)

    def classify(self, start, end, features, times):
        mask = (times >= start) & (times < end)
        if not np.any(mask): 
            return SceneType.SILENCE, 0.99, -100.0

        chunk = features[:, mask]
        
        avg_rms = np.mean(chunk[FeatureRegistry.RMS])
        std_rms = np.std(chunk[FeatureRegistry.RMS]) 
        avg_contrast = np.mean(chunk[FeatureRegistry.CONTRAST])
        avg_chroma = np.mean(chunk[FeatureRegistry.CHROMA_STD])
        
        db = 20 * np.log10(avg_rms + 1e-9)

        # 1. Silence
        if avg_rms <= self.stats["rms_base"] * self.config.silence_threshold_multiplier:
            return SceneType.SILENCE, 0.9, db

        # 2. Music 
        is_tonal = (avg_contrast > self.stats["contrast_high"] and 
                   avg_chroma > self.stats["chroma_high"])
        
        # 3. High Energy
        is_loud = avg_rms > self.stats["rms_peak"]
        
        if is_tonal:
            if is_loud: return SceneType.MIXED, 0.7, db
            return SceneType.MUSIC_DOMINANT, 0.85, db
            
        if is_loud:
            return SceneType.HIGH_ENERGY, 0.75, db

        if std_rms > self.config.std_rms_threshold: 
            if avg_contrast < self.stats["contrast_low"]:
                return SceneType.NOISY_DIALOGUE, 0.7, db
            return SceneType.QUIET_DIALOGUE, 0.8, db
        else:
            return SceneType.AMBIENT, 0.6, db

# ==========================================
# 5. VISUALIZATION
# ==========================================

def generate_visualization(audio_path, segments, output_path, config: SegmentationConfig, logger: Optional[logging.Logger] = None):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    msg = f"--> [5/5] Generating Plot: {output_path}"
    if logger: logger.info(msg)
    else: print(msg)

    SCENE_COLORS = {
        SceneType.QUIET_DIALOGUE: 'green',
        SceneType.NOISY_DIALOGUE: 'orange',
        SceneType.MUSIC_DOMINANT: 'purple',
        SceneType.HIGH_ENERGY: 'red',
        SceneType.SILENCE: 'gray',
        SceneType.MIXED: 'cyan',
        SceneType.AMBIENT: 'blue',
        SceneType.GAP_FILLER: 'black'
    }

    try:
        y, sr = librosa.load(audio_path, sr=config.sample_rate, duration=config.viz_duration) 
        fig, ax = plt.subplots(figsize=(15, 6))
        times = np.arange(len(y)) / sr
        ax.plot(times, y, color='black', alpha=0.5, linewidth=0.5)
        
        for seg in segments:
            if seg.end > config.viz_duration: break 
            c = SCENE_COLORS.get(seg.scene_type, 'black')
            ax.axvspan(seg.start, seg.end, color=c, alpha=0.3)
            ax.text((seg.start+seg.end)/2, 0.9, seg.scene_type.value, 
                    rotation=90, ha='center', fontsize=8, transform=ax.get_xaxis_transform())

        ax.set_title(f"Segmentation: {os.path.basename(audio_path)}")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        err_msg = f"Viz Error: {e}"
        if logger: logger.error(err_msg)
        else: print(err_msg)

# ==========================================
# 6. MAIN
# ==========================================

def process_movie_v7(
    file_path: str,
    output_path: str,
    config: Optional[SegmentationConfig] = None,
    visualize: bool = False,
    logger: Optional[logging.Logger] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> str:
    """
    Process media file and generate segmentation metadata.

    Returns:
        Path to the generated JSON file (same as output_path on success)

    Raises:
        FileNotFoundError: If input file doesn't exist
        RuntimeError: If ffmpeg is not available
        ValueError: If config parameters are invalid
        SegmentationError: Base class for other processing errors
    """
    start_time = time.time()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    if config is None:
        config = SegmentationConfig()

    # Helper for logging
    def log(msg, level=logging.INFO):
        if logger: logger.log(level, msg)
        else: print(msg)

    log(f"Starting processing for: {file_path}")

    # 1. Extraction
    extractor = StreamFeatureExtractor(config, logger)
    # Pass callback to extractor
    features, times, duration = extractor.extract(file_path, progress_callback)
    
    if features is None: 
        raise FeatureExtractionError("Feature extraction returned None")

    if progress_callback: progress_callback(0.8, "Calibrating classifier...")
    
    # 2. Calibration
    classifier = AdaptiveClassifier(config, logger)
    classifier.calibrate(features)

    if progress_callback: progress_callback(0.85, "Clustering and segmenting...")

    # 3. Segmentation
    segmenter = SemanticSegmenter(config, logger)
    raw_segments = segmenter.segment(features, times, duration)
    
    log(f"--> [4/5] Building Metadata...")
    if progress_callback: progress_callback(0.9, "Building metadata...")

    # 4. Metadata Construction
    processing_time = time.time() - start_time
    
    output = {
        "meta": {
            "filename": os.path.basename(file_path),
            "version": __version__,
            "algorithm": "agglomerative_clustering_ward",
            "processing_time_seconds": round(processing_time, 3),
            "total_duration_seconds": round(duration, 3),
            "config": {
                "min_duration": config.min_duration,
                "max_duration": config.max_duration,
                "snap_window": config.snap_window,
                "clustering_threshold": config.clustering_threshold
            }
        }, 
        "segments": []
    }
    
    final_objs = []
    for i, s_data in enumerate(raw_segments):
        s_type, conf, db = classifier.classify(s_data["start"], s_data["end"], features, times)
        
        seg = Segment(s_data["start"], s_data["end"], s_type, conf, db)
        final_objs.append(seg)
        
        # Add index to dict
        seg_dict = seg.to_dict()
        seg_dict["segment_index"] = i
        output["segments"].append(seg_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    log(f"Done! JSON at {output_path}")

    if visualize:
        if progress_callback: progress_callback(0.95, "Generating visualization...")
        generate_visualization(file_path, final_objs, output_path.replace('.json', '.png'), config, logger)

    if progress_callback: progress_callback(1.0, "Complete")
    
    return output_path

def extract_segments_to_wav(
    source_file: str,
    json_metadata_path: str,
    output_dir: str,
    use_asr_timestamps: bool = True,
    naming_pattern: str = "{basename}_scene_{index:04d}.wav",
    sample_rate: int = 16000,
    raise_on_error: bool = False
) -> List[Dict]:
    """
    Extract audio segments based on JSON metadata.

    Returns:
        List of dicts with 'path', 'start', 'end', 'duration', 'context'
    """
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"Source file not found: {source_file}")
    if not os.path.exists(json_metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {json_metadata_path}")
        
    with open(json_metadata_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    basename = os.path.splitext(os.path.basename(source_file))[0]
    results = []
    
    segments = data.get("segments", [])
    print(f"Extracting {len(segments)} segments to {output_dir}...")
    
    for seg in segments:
        idx = seg.get("segment_index", 0)
        
        # Choose timestamp source
        if use_asr_timestamps:
            ts = seg["asr_processing"]
        else:
            ts = seg["timestamps"]
            
        start = ts["start"]
        duration = ts["duration"]
        
        out_name = naming_pattern.format(basename=basename, index=idx)
        out_path = os.path.join(output_dir, out_name)
        
        # ffmpeg command
        # -ss before -i is faster seeking
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start),
            '-i', source_file,
            '-t', str(duration),
            '-ar', str(sample_rate),
            '-ac', '1',
            '-vn', # No video
            '-loglevel', 'error',
            out_path
        ]
        
        try:
            subprocess.check_call(cmd)
            results.append({
                "path": out_path,
                "start": start,
                "end": ts["end"],
                "duration": duration,
                "context": seg.get("context", {})
            })
        except subprocess.CalledProcessError as e:
            print(f"Failed to extract segment {idx}: {e}")
            if raise_on_error:
                raise
            
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Input audio/video")
    parser.add_argument("--out", help="Output JSON")
    parser.add_argument("--viz", "-v", action="store_true", help="Generate PNG plot")
    
    # Config overrides
    parser.add_argument("--min-dur", type=float, default=20.0, help="Min segment duration (s)")
    parser.add_argument("--max-dur", type=float, default=420.0, help="Max segment duration (s)")
    parser.add_argument("--snap-window", type=float, default=5.0, help="Silence snap window (s)")
    parser.add_argument("--cluster-threshold", type=float, default=18.0, help="Clustering distance threshold (lower=more segments)")
    
    args = parser.parse_args()
    
    # Create config from args
    config = SegmentationConfig(
        min_duration=args.min_dur,
        max_duration=args.max_dur,
        snap_window=args.snap_window,
        clustering_threshold=args.cluster_threshold
    )
    
    out = args.out if args.out else f"{os.path.splitext(args.file)[0]}_v7.json"
    process_movie_v7(args.file, out, config=config, visualize=args.viz)

