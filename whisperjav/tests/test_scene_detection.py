#!/usr/bin/env python3
"""
A comprehensive, self-contained test suite for the SceneDetector module.

This script runs multiple test passes with different parameter configurations on a
given audio file. It generates a detailed statistical report for each pass, including
a final summary table, to help users determine the optimal settings for their
specific audio content.
"""
import argparse
import logging
import shutil
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from tqdm import tqdm

# --- Setup Logger ---
# Configure a basic logger to see output from the detector
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Auditok Availability Check ---
_AUDITOK_AVAILABLE = False
try:
    import auditok
    _AUDITOK_AVAILABLE = True
except ImportError:
    logger.warning("Auditok not available. Scene detection functionality will be limited.")

class SceneDetector:
    """Handles audio scene detection using a two-pass Auditok approach."""

    def __init__(self,
                 max_duration: float = 29.0,
                 min_duration: float = 0.5,
                 max_silence: float = 2.0,
                 energy_threshold: int = 45):
        """
        Initialize the audio scene detector.
        (This is the original, user-provided __init__ contract)
        """
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_silence = max_silence
        self.energy_threshold = energy_threshold

        # Two-pass parameters (these will be dynamically overwritten by the test harness)
        self.pass1_max_duration = 1599.0
        self.pass1_min_duration = 0.3
        self.pass1_max_silence = 1.5
        self.pass1_energy_threshold = 38
        self.pass2_max_duration = 28
        self.pass2_min_duration = 0.5
        self.pass2_max_silence = 0.75
        self.pass2_energy_threshold = 40

        # Add target_lufs as an attribute that can be monkey-patched
        self.target_lufs = -20.0

    def detect_scenes(self, audio_path: Path, output_dir: Path, media_basename: str) -> Dict:
        """
        Split audio into scenes using a two-pass Auditok approach.
        Returns a dictionary containing detailed results for reporting.
        """
        if not _AUDITOK_AVAILABLE:
            raise RuntimeError("Auditok is required for scene detection but not installed")

        logger.debug(f"Starting two-pass audio scene detection for: {audio_path}")
        
        # --- Improved Audio Handling ---
        try:
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32', always_2d=False)
        except Exception as e:
            raise RuntimeError(f"Failed to read audio file: {e}")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        audio_duration = len(audio_data) / sample_rate
        logger.info(f"Processing '{audio_path.name}': Duration={audio_duration:.1f}s, Rate={sample_rate}Hz")

        # --- Handle cases where audio is too short ---
        if audio_duration < self.min_duration:
            logger.warning(f"Audio duration ({audio_duration:.1f}s) is shorter than minimum scene length ({self.min_duration}s)")
            return {
                "pass1_regions": [],
                "final_scenes": [],
                "brute_force_count": 0,
                "audio_metadata": {
                    "duration": audio_duration,
                    "sample_rate": sample_rate
                }
            }

        pass1_regions = []
        final_scenes = []
        brute_force_count = 0
        scene_idx = 0

        # Pass 1: Coarse splitting
        logger.debug("Pass 1: Coarse splitting")

        normalized_audio = self._lufs_normalize(audio_data.copy(), sample_rate, self.target_lufs)
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

        coarse_regions = list(auditok.split(audio_bytes, **pass1_params))
        logger.debug(f"Pass 1 found {len(coarse_regions)} coarse regions.")

        # Process coarse regions to generate final scenes
        for region in coarse_regions:
            region_start, region_end = region.start, region.end
            region_duration = region_end - region_start
            # Consistent return type for pass1: (start, end, duration)
            pass1_regions.append((region_start, region_end, region_duration))

            start_sample = int(region_start * sample_rate)
            end_sample = int(region_end * sample_rate)
            region_audio = audio_data[start_sample:end_sample]

            # If region is short enough, it's a final scene
            if self.min_duration <= region_duration <= self.max_duration:
                scene_path = self._save_scene(region_audio, sample_rate, scene_idx, output_dir, media_basename)
                final_scenes.append((scene_path, region_start, region_end, region_duration))
                scene_idx += 1
            # If region is too long, it needs Pass 2
            elif region_duration > self.max_duration:
                logger.debug(f"Pass 2: Splitting long region ({region_duration:.1f}s)")
                
                normalized_region = self._lufs_normalize(region_audio.copy(), sample_rate, self.target_lufs)
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
                    logger.warning("Pass 2 found no sub-regions, using brute force splitting.")
                    brute_force_count += 1
                    num_scenes = int(np.ceil(region_duration / self.max_duration))
                    for i in range(num_scenes):
                        sub_start = i * self.max_duration
                        sub_end = min((i + 1) * self.max_duration, region_duration)
                        if sub_end - sub_start >= self.min_duration:
                            sub_audio = region_audio[int(sub_start*sample_rate):int(sub_end*sample_rate)]
                            abs_start, abs_end = region_start + sub_start, region_start + sub_end
                            scene_path = self._save_scene(sub_audio, sample_rate, scene_idx, output_dir, media_basename)
                            final_scenes.append((scene_path, abs_start, abs_end, sub_end - sub_start))
                            scene_idx += 1
                else:
                    # Process sub-regions from pass 2
                    for sub_region in sub_regions:
                        sub_start, sub_end = sub_region.start, sub_region.end
                        if sub_end - sub_start >= self.min_duration:
                            sub_audio = region_audio[int(sub_start*sample_rate):int(sub_end*sample_rate)]
                            abs_start, abs_end = region_start + sub_start, region_start + sub_end
                            scene_path = self._save_scene(sub_audio, sample_rate, scene_idx, output_dir, media_basename)
                            final_scenes.append((scene_path, abs_start, abs_end, sub_end - sub_start))
                            scene_idx += 1

        logger.info(f"Detected {len(final_scenes)} final scenes.")
        
        # --- Consistent Return with Audio Metadata ---
        return {
            "pass1_regions": pass1_regions,
            "final_scenes": sorted(final_scenes, key=lambda x: x[1]),
            "brute_force_count": brute_force_count,
            "audio_metadata": {
                "duration": audio_duration,
                "sample_rate": sample_rate
            }
        }

    def _lufs_normalize(self, audio: np.ndarray, sample_rate: int, target_lufs: float) -> np.ndarray:
        """Normalize audio to target LUFS loudness with enhanced checks."""
        # --- Check for silence before normalization ---
        if np.max(np.abs(audio)) < 1e-6:
            logger.debug("Skipping normalization for silent audio")
            return audio

        try:
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            
            # --- Warn when audio is very quiet ---
            if loudness <= -70:
                warnings.warn(f"Audio is very quiet ({loudness:.1f} LUFS), normalization may not be effective")
            
            normalized = pyln.normalize.loudness(audio, loudness, target_lufs)
            peak = np.max(np.abs(normalized))
            
            # --- Apply gain reduction to prevent clipping ---
            if peak > 0.95:
                gain = 0.95 / peak
                logger.warning(f"Applying gain reduction ({gain:.2f}x) to prevent clipping")
                normalized = normalized * gain
                
            return normalized
        except Exception as e:
            logger.error(f"LUFS normalization failed: {e}. Returning original audio.")
            return audio

    def _save_scene(self, audio_data: np.ndarray, sample_rate: int, scene_idx: int, output_dir: Path, media_basename: str) -> Path:
        """Saves a single audio scene to a file."""
        scene_filename = f"{media_basename}_scene_{scene_idx:04d}.wav"
        scene_path = output_dir / scene_filename
        sf.write(str(scene_path), audio_data, sample_rate, subtype='PCM_16')
        return scene_path

# --- Test Harness Logic ---

class TestConfig:
    """
    Defines parameter configurations for the test suite.
    Users can add, remove, or modify dictionaries in the `CONFIGS` list
    to customize the test runs.
    """
    CONFIGS = [
        {
            "name": "Default Parameters",
            "params": {
                "pass1_energy_threshold": 50, "pass2_energy_threshold": 50,
                "pass1_max_silence": 1.5, "pass2_max_silence": 1.5,
                "max_duration": 29.0, "target_lufs": -14.0
            }
        },
        {
            "name": "High Sensitivity",
            "params": {
                "pass1_energy_threshold": 60, "pass2_energy_threshold": 50,
                "pass1_max_silence": 1.0, "pass2_max_silence": 1.0,
                "max_duration": 29.0, "target_lufs": -14.0
            }
        },
        {
            "name": "Low Sensitivity",
            "params": {
                "pass1_energy_threshold": 40, "pass2_energy_threshold": 45,
                "pass1_max_silence": 2.5, "pass2_max_silence": 2.0,
                "max_duration": 29.0, "target_lufs": -20.0
            }
        }
    ]

    @classmethod
    def get_configs(cls) -> List[Dict]:
        """Validates and returns the list of test configurations."""
        for config in cls.CONFIGS:
            params = config.get("params", {})
            name = config.get("name", "Unnamed")
            for key, (min_val, max_val) in {
                "max_duration": (1.0, 60.0),
                "target_lufs": (-30.0, -5.0),
                "pass1_energy_threshold": (20, 80),
                "pass2_energy_threshold": (20, 80),
            }.items():
                val = params.get(key)
                if val is not None and not (min_val <= val <= max_val):
                    raise ValueError(f"Parameter '{key}' ({val}) in '{name}' is out of range [{min_val}, {max_val}]")
        return cls.CONFIGS

# --- Enhanced Standalone Helper Functions ---

def _generate_stats_lines(scenes: List[Tuple], total_audio_duration: float, pass_name: str) -> List[str]:
    """Enhanced statistics generation with duration distribution and gap analysis."""
    lines = []
    lines.append(f"--- {pass_name} ---")

    if not scenes:
        lines.append("  No regions/scenes found in this pass.")
        return lines

    # Determine indices for start, end, and duration. This handles both
    # pass1 tuples (start, end, duration) and final scene tuples (path, start, end, duration).
    duration_idx = -1
    start_idx, end_idx = (1, 2) if isinstance(scenes[0][0], Path) else (0, 1)

    durations = np.array([s[duration_idx] for s in scenes])
    total_scenes_duration = np.sum(durations)
    coverage = (total_scenes_duration / total_audio_duration * 100) if total_audio_duration > 0 else 0

    # Basic stats
    lines.append(f"  - Regions/Scenes Found:   {len(scenes)}")
    lines.append(f"  - Total Audio Coverage:   {coverage:.2f}% ({total_scenes_duration:.2f}s / {total_audio_duration:.2f}s)")
    lines.append("\n  [Duration Statistics]")
    lines.append(f"  - Minimum: {np.min(durations):.2f}s")
    lines.append(f"  - Maximum: {np.max(durations):.2f}s")
    lines.append(f"  - Average: {np.mean(durations):.2f}s")
    lines.append(f"  - Std Dev: {np.std(durations):.2f}s")
    lines.append(f"  - Median:  {np.median(durations):.2f}s")

    # --- Enhanced Statistics: Duration Distribution Histogram ---
    lines.append("\n  [Duration Distribution]")
    bins = [0, 5, 10, 15, 20, 25, 30, 45, 60, float('inf')]
    hist, _ = np.histogram(durations, bins=bins)
    for i in range(len(bins)-1):
        lower, upper = bins[i], bins[i+1]
        count = hist[i]
        label = f">{lower}s" if upper == float('inf') else f"{lower}-{upper}s"
        percentage = (count / len(scenes) * 100) if len(scenes) > 0 else 0
        lines.append(f"  - {label:<8}: {count:>4} scenes ({percentage:.1f}%)")

    # Gap statistics
    gaps = []
    if len(scenes) > 1:
        # Scenes are pre-sorted by start time in the main detection function
        for i in range(len(scenes) - 1):
            gap_start = scenes[i][end_idx]
            gap_end = scenes[i + 1][start_idx]
            if gap_end > gap_start:
                gaps.append(gap_end - gap_start)

    if gaps:
        gaps = np.array(gaps)
        lines.append("\n  [Gap Statistics]")
        lines.append(f"  - Total Gap Duration: {np.sum(gaps):.2f}s")
        lines.append(f"  - Average Gap:        {np.mean(gaps):.2f}s")
        lines.append(f"  - Longest Gap:        {np.max(gaps):.2f}s")

    return lines

def calculate_and_print_stats(detection_results: Dict, test_name: str, params: Dict, exec_time: float) -> List[str]:
    """Calculates and prints a structured, multi-pass statistical report."""
    report_lines = []
    # --- Improved Formatting: Visual Separators ---
    separator = "━" * 80
    report_lines.append(separator)
    report_lines.append(f" 𝗧𝗘𝗦𝗧 𝗥𝗨𝗡: {test_name} ".center(80, "━"))
    report_lines.append(separator)

    report_lines.append("\n[Parameters Used]")
    # Get default values to show a complete parameter list for the run
    temp_detector = SceneDetector()
    full_params = {k: getattr(temp_detector, k) for k in vars(temp_detector)}
    full_params.update(params)
    for key, val in full_params.items():
        report_lines.append(f"  - {key:<25}: {val}")

    total_audio_duration = detection_results["audio_metadata"]["duration"]
    
    # Pass 1 Stats
    report_lines.append("")
    report_lines.extend(_generate_stats_lines(detection_results["pass1_regions"], total_audio_duration, "Pass 1: Coarse Region Detection"))
    
    # Final Stats
    report_lines.append("")
    report_lines.extend(_generate_stats_lines(detection_results["final_scenes"], total_audio_duration, "Pass 2: Final Scene Generation"))
    
    # Summary
    report_lines.append("\n--- Run Summary ---")
    report_lines.append(f"  - Final Scene Count:       {len(detection_results['final_scenes'])}")
    report_lines.append(f"  - Brute-Force Fallbacks:   {detection_results['brute_force_count']}")
    report_lines.append(f"  - Total Processing Time:   {exec_time:.2f} seconds")
    report_lines.append(separator + "\n\n")

    print("\n".join(report_lines))
    return report_lines


def run_test_suite(audio_path: Path, output_dir: Path):
    """Enhanced test suite runner with summary table."""
    if not audio_path.exists():
        logger.error(f"Error: Audio file not found at {audio_path}")
        return

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    master_output_dir = output_dir / f"test_run_{audio_path.stem}_{run_timestamp}"
    master_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nTest outputs will be saved in: {master_output_dir}")

    full_report = []
    test_results = [] # Store results for the final summary table

    test_configs = TestConfig.get_configs()
    for i, config in enumerate(tqdm(test_configs, desc="Running Test Configurations")):
        test_name = config["name"]
        params_to_set = config["params"]

        # Create a temporary directory for each pass's audio files
        pass_dir = master_output_dir / f"pass_{i+1}_{test_name.replace(' ', '_')}"
        pass_dir.mkdir(exist_ok=True)

        detector = SceneDetector()
        for param_name, param_value in params_to_set.items():
            setattr(detector, param_name, param_value)

        start_time = time.perf_counter()
        detection_results = detector.detect_scenes(
            audio_path=audio_path,
            output_dir=pass_dir,
            media_basename=audio_path.stem
        )
        end_time = time.perf_counter()

        report_lines = calculate_and_print_stats(
            detection_results,
            test_name,
            params_to_set,
            exec_time=end_time - start_time
        )
        full_report.extend(report_lines)
        
        shutil.rmtree(pass_dir) # Clean up scene files to save space

        # --- Store results for summary table ---
        final_scenes = detection_results["final_scenes"]
        total_duration = detection_results["audio_metadata"]["duration"]
        coverage = 0.0
        if final_scenes and total_duration > 0:
            coverage = (np.sum([s[-1] for s in final_scenes]) / total_duration) * 100

        test_results.append({
            "name": test_name,
            "final_scene_count": len(final_scenes),
            "coverage": coverage,
            "exec_time": end_time - start_time,
            "brute_force": detection_results["brute_force_count"]
        })

    # --- Improved Reporting: Final Summary Table ---
    if test_results:
        summary_lines = [
            "\n" + "=" * 80,
            " 📈 FINAL SUMMARY TABLE 📈 ".center(80, "="),
            "=" * 80,
            "\n{:<25} {:>7}  {:>9}  {:>9}  {:>12}".format("Configuration", "Scenes", "Coverage", "Time(s)", "Brute-Force"),
            "-" * 80
        ]
        for result in test_results:
            summary_lines.append(
                f"{result['name'][:25]:<25} {result['final_scene_count']:>7}  "
                f"{result['coverage']:>8.1f}% "
                f"{result['exec_time']:>9.2f} "
                f"{result['brute_force']:>12}"
            )
        summary_lines.append("=" * 80 + "\n")
        full_report.extend(summary_lines)
        
        print("\n".join(summary_lines))

    report_file = master_output_dir / "summary_report.txt"
    with open(report_file, "w", encoding='utf-8') as f:
        f.writelines(line + "\n" for line in full_report)

    logger.info(f"\nAll tests complete. A full summary has been saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A comprehensive test suite for the SceneDetector module.")
    parser.add_argument("--audio-file", type=Path, required=True, help="Path to the input audio file to be tested.")
    parser.add_argument("--output-dir", type=Path, default=Path("./test_results"), help="Directory to save test run outputs and the final report.")
    args = parser.parse_args()
    try:
        run_test_suite(args.audio_file, args.output_dir)
    except (RuntimeError, ValueError) as e:
        logger.error(f"A critical error occurred: {e}")

