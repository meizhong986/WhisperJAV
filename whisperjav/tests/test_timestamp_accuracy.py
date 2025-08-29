#!/usr/bin/env python3
"""
Comprehensive timestamp accuracy testing for WhisperJAV's segment-process-stitch pipeline.

This module verifies that timestamps in the final SRT output accurately match
the original audio, despite segmentation and stitching operations.
"""

import argparse
import json
import logging
import numpy as np
import soundfile as sf
import srt
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy.signal import correlate

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


@dataclass
class TimestampVerification:
    """Results of timestamp verification for a single subtitle."""
    subtitle_index: int
    subtitle_text: str
    expected_start: float
    expected_end: float
    actual_start: float
    actual_end: float
    start_drift: float  # Difference in seconds
    end_drift: float
    duration_error: float
    is_within_tolerance: bool
    confidence_score: float = 0.0  # Optional audio correlation score


class TimestampAccuracyTester:
    """
    Verifies timestamp accuracy through the segment-process-stitch pipeline.
    """
    
    def __init__(self, tolerance_ms: int = 100, verbose: bool = False):
        """
        Initialize the timestamp accuracy tester.
        
        Args:
            tolerance_ms: Maximum acceptable drift in milliseconds
            verbose: Enable detailed logging
        """
        self.tolerance_s = tolerance_ms / 1000.0
        self.verbose = verbose
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    def verify_scene_metadata(self, metadata_path: Path) -> Dict:
        """
        Verify scene detection metadata for continuity and coverage.
        
        Args:
            metadata_path: Path to scene metadata JSON file
            
        Returns:
            Dictionary containing verification results
        """
        logger.info(f"Verifying scene metadata: {metadata_path}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if not metadata:
            raise ValueError("No metadata found in file")
        
        # Sort scenes by start time
        scenes = sorted(metadata, key=lambda x: x['start_time_s'])
        
        # Check for continuity
        gaps = []
        overlaps = []
        
        for i in range(1, len(scenes)):
            prev_end = scenes[i-1]['end_time_s']
            curr_start = scenes[i]['start_time_s']
            
            if curr_start > prev_end + 0.001:  # Small tolerance for float precision
                gaps.append({
                    'between_scenes': (i-1, i),
                    'gap_start': prev_end,
                    'gap_end': curr_start,
                    'duration': curr_start - prev_end
                })
            elif curr_start < prev_end - 0.001:
                overlaps.append({
                    'between_scenes': (i-1, i),
                    'overlap_start': curr_start,
                    'overlap_end': prev_end,
                    'duration': prev_end - curr_start
                })
        
        # Calculate total coverage
        total_scene_duration = sum(s['duration_s'] for s in scenes)
        
        return {
            'num_scenes': len(scenes),
            'total_duration': total_scene_duration,
            'gaps': gaps,
            'overlaps': overlaps,
            'is_continuous': len(gaps) == 0 and len(overlaps) == 0,
            'scenes': scenes
        }
    
    def verify_stitching_accuracy(self,
                                 scene_srts: List[Tuple[Path, float]],
                                 final_srt_path: Path) -> List[TimestampVerification]:
        """
        Verify that stitching preserves timestamps correctly.
        
        Args:
            scene_srts: List of (srt_path, start_offset) tuples
            final_srt_path: Path to the final stitched SRT file
            
        Returns:
            List of verification results for each subtitle
        """
        logger.info("Verifying stitching accuracy...")
        
        # Read final SRT
        with open(final_srt_path, 'r', encoding='utf-8') as f:
            final_subs = list(srt.parse(f.read()))
        
        verifications = []
        final_sub_idx = 0
        
        # Process each scene SRT
        for srt_path, start_offset in sorted(scene_srts, key=lambda x: x[1]):
            if not srt_path.exists():
                logger.warning(f"Scene SRT not found: {srt_path}")
                continue
            
            with open(srt_path, 'r', encoding='utf-8') as f:
                scene_subs = list(srt.parse(f.read()))
            
            for scene_sub in scene_subs:
                # Calculate expected timestamps
                expected_start = scene_sub.start.total_seconds() + start_offset
                expected_end = scene_sub.end.total_seconds() + start_offset
                
                # Find corresponding subtitle in final SRT
                if final_sub_idx < len(final_subs):
                    final_sub = final_subs[final_sub_idx]
                    actual_start = final_sub.start.total_seconds()
                    actual_end = final_sub.end.total_seconds()
                    
                    # Verify text matches
                    if scene_sub.content.strip() == final_sub.content.strip():
                        start_drift = actual_start - expected_start
                        end_drift = actual_end - expected_end
                        duration_error = (actual_end - actual_start) - (expected_end - expected_start)
                        
                        verification = TimestampVerification(
                            subtitle_index=final_sub.index,
                            subtitle_text=final_sub.content[:50],  # First 50 chars
                            expected_start=expected_start,
                            expected_end=expected_end,
                            actual_start=actual_start,
                            actual_end=actual_end,
                            start_drift=start_drift,
                            end_drift=end_drift,
                            duration_error=duration_error,
                            is_within_tolerance=(
                                abs(start_drift) <= self.tolerance_s and
                                abs(end_drift) <= self.tolerance_s
                            )
                        )
                        verifications.append(verification)
                        final_sub_idx += 1
                    else:
                        logger.warning(f"Text mismatch at index {final_sub_idx}")
        
        return verifications
    
    def verify_audio_alignment(self,
                              original_audio_path: Path,
                              scene_audio_paths: List[Tuple[Path, float, float]],
                              sample_points: int = 10) -> Dict:
        """
        Verify audio alignment by comparing waveforms at boundary points.
        
        Args:
            original_audio_path: Path to original audio file
            scene_audio_paths: List of (scene_path, start_time, end_time) tuples
            sample_points: Number of boundary points to sample
            
        Returns:
            Dictionary containing alignment verification results
        """
        logger.info("Verifying audio alignment at scene boundaries...")
        
        # Load original audio
        original_audio, sr = sf.read(str(original_audio_path), dtype='float32')
        if original_audio.ndim > 1:
            original_audio = np.mean(original_audio, axis=1)
        
        alignment_results = []
        
        for scene_path, start_time, end_time in scene_audio_paths[:sample_points]:
            if not scene_path.exists():
                continue
            
            # Load scene audio
            scene_audio, scene_sr = sf.read(str(scene_path), dtype='float32')
            if scene_audio.ndim > 1:
                scene_audio = np.mean(scene_audio, axis=1)
            
            # Extract corresponding segment from original
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            original_segment = original_audio[start_sample:end_sample]
            
            # Resample if necessary
            if scene_sr != sr:
                from scipy import signal
                scene_audio = signal.resample(
                    scene_audio,
                    int(len(scene_audio) * sr / scene_sr)
                )
            
            # Compare lengths
            length_diff = len(scene_audio) - len(original_segment)
            
            # Calculate correlation for alignment verification
            if len(scene_audio) > 0 and len(original_segment) > 0:
                min_len = min(len(scene_audio), len(original_segment))
                correlation = np.corrcoef(
                    scene_audio[:min_len],
                    original_segment[:min_len]
                )[0, 1]
            else:
                correlation = 0.0
            
            alignment_results.append({
                'scene': scene_path.name,
                'start_time': start_time,
                'end_time': end_time,
                'length_diff_samples': length_diff,
                'length_diff_ms': (length_diff / sr) * 1000,
                'correlation': correlation,
                'is_aligned': abs(length_diff) < sr * 0.1  # Within 100ms
            })
        
        return {
            'sample_rate': sr,
            'alignments': alignment_results,
            'num_aligned': sum(1 for a in alignment_results if a['is_aligned']),
            'total_checked': len(alignment_results)
        }
    
    def generate_report(self,
                       scene_verification: Dict,
                       stitch_verifications: List[TimestampVerification],
                       audio_alignment: Optional[Dict] = None,
                       output_path: Optional[Path] = None) -> str:
        """
        Generate a comprehensive timestamp accuracy report.
        
        Args:
            scene_verification: Results from verify_scene_metadata
            stitch_verifications: Results from verify_stitching_accuracy
            audio_alignment: Optional results from verify_audio_alignment
            output_path: Optional path to save the report
            
        Returns:
            Report as a string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TIMESTAMP ACCURACY VERIFICATION REPORT")
        report_lines.append("=" * 80)
        
        # Scene Detection Analysis
        report_lines.append("\n1. SCENE DETECTION ANALYSIS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Scenes: {scene_verification['num_scenes']}")
        report_lines.append(f"Total Duration: {scene_verification['total_duration']:.2f}s")
        report_lines.append(f"Continuity: {'✓ Continuous' if scene_verification['is_continuous'] else '✗ Gaps/Overlaps Detected'}")
        
        if scene_verification['gaps']:
            report_lines.append(f"\nGaps Found: {len(scene_verification['gaps'])}")
            for gap in scene_verification['gaps'][:5]:  # Show first 5
                report_lines.append(
                    f"  - Between scenes {gap['between_scenes']}: "
                    f"{gap['duration']*1000:.1f}ms gap"
                )
        
        if scene_verification['overlaps']:
            report_lines.append(f"\nOverlaps Found: {len(scene_verification['overlaps'])}")
            for overlap in scene_verification['overlaps'][:5]:
                report_lines.append(
                    f"  - Between scenes {overlap['between_scenes']}: "
                    f"{overlap['duration']*1000:.1f}ms overlap"
                )
        
        # Stitching Accuracy Analysis
        report_lines.append("\n2. STITCHING ACCURACY ANALYSIS")
        report_lines.append("-" * 40)
        
        if stitch_verifications:
            total_subs = len(stitch_verifications)
            accurate_subs = sum(1 for v in stitch_verifications if v.is_within_tolerance)
            accuracy_rate = (accurate_subs / total_subs) * 100
            
            # Calculate statistics
            start_drifts = [v.start_drift * 1000 for v in stitch_verifications]
            end_drifts = [v.end_drift * 1000 for v in stitch_verifications]
            
            report_lines.append(f"Total Subtitles: {total_subs}")
            report_lines.append(f"Within Tolerance: {accurate_subs} ({accuracy_rate:.1f}%)")
            report_lines.append(f"Tolerance: ±{self.tolerance_s * 1000:.0f}ms")
            
            report_lines.append("\nDrift Statistics (ms):")
            report_lines.append(f"  Start Time Drift:")
            report_lines.append(f"    Mean: {np.mean(start_drifts):.1f}ms")
            report_lines.append(f"    Std Dev: {np.std(start_drifts):.1f}ms")
            report_lines.append(f"    Max: {np.max(np.abs(start_drifts)):.1f}ms")
            
            report_lines.append(f"  End Time Drift:")
            report_lines.append(f"    Mean: {np.mean(end_drifts):.1f}ms")
            report_lines.append(f"    Std Dev: {np.std(end_drifts):.1f}ms")
            report_lines.append(f"    Max: {np.max(np.abs(end_drifts)):.1f}ms")
            
            # Show worst offenders
            worst = sorted(stitch_verifications, 
                         key=lambda v: max(abs(v.start_drift), abs(v.end_drift)),
                         reverse=True)[:5]
            
            if any(not v.is_within_tolerance for v in worst):
                report_lines.append("\nSubtitles with Largest Drift:")
                for v in worst:
                    if not v.is_within_tolerance:
                        report_lines.append(
                            f"  #{v.subtitle_index}: "
                            f"Start drift: {v.start_drift*1000:.1f}ms, "
                            f"End drift: {v.end_drift*1000:.1f}ms"
                        )
        
        # Audio Alignment Analysis
        if audio_alignment:
            report_lines.append("\n3. AUDIO ALIGNMENT VERIFICATION")
            report_lines.append("-" * 40)
            report_lines.append(
                f"Scenes Checked: {audio_alignment['total_checked']}"
            )
            report_lines.append(
                f"Properly Aligned: {audio_alignment['num_aligned']} "
                f"({audio_alignment['num_aligned']/audio_alignment['total_checked']*100:.1f}%)"
            )
            
            # Show correlation statistics
            correlations = [a['correlation'] for a in audio_alignment['alignments']]
            report_lines.append(f"\nAudio Correlation:")
            report_lines.append(f"  Mean: {np.mean(correlations):.3f}")
            report_lines.append(f"  Min: {np.min(correlations):.3f}")
            report_lines.append(f"  Max: {np.max(correlations):.3f}")
        
        # Summary
        report_lines.append("\n4. SUMMARY")
        report_lines.append("-" * 40)
        
        issues = []
        if not scene_verification['is_continuous']:
            issues.append("Scene detection has gaps/overlaps")
        if stitch_verifications and accuracy_rate < 95:
            issues.append(f"Only {accuracy_rate:.1f}% of subtitles within tolerance")
        if audio_alignment and audio_alignment['num_aligned'] < audio_alignment['total_checked']:
            issues.append("Some audio segments misaligned")
        
        if issues:
            report_lines.append("⚠️  Issues Found:")
            for issue in issues:
                report_lines.append(f"  - {issue}")
        else:
            report_lines.append("✓ All timestamps verified successfully!")
        
        report_lines.append("\n" + "=" * 80)
        
        # Join and optionally save
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_path}")
        
        return report
    
    def visualize_drift(self,
                       verifications: List[TimestampVerification],
                       output_path: Optional[Path] = None):
        """
        Create visualization of timestamp drift over time.
        
        Args:
            verifications: List of timestamp verification results
            output_path: Optional path to save the plot
        """
        if not verifications:
            logger.warning("No verifications to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Timestamp Drift Analysis', fontsize=16)
        
        # Extract data
        indices = [v.subtitle_index for v in verifications]
        start_drifts_ms = [v.start_drift * 1000 for v in verifications]
        end_drifts_ms = [v.end_drift * 1000 for v in verifications]
        expected_starts = [v.expected_start for v in verifications]
        
        # Plot 1: Start time drift over subtitle index
        ax = axes[0, 0]
        ax.scatter(indices, start_drifts_ms, alpha=0.6, s=10)
        ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
        ax.axhline(y=self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Subtitle Index')
        ax.set_ylabel('Start Time Drift (ms)')
        ax.set_title('Start Time Drift by Subtitle Index')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: End time drift over subtitle index
        ax = axes[0, 1]
        ax.scatter(indices, end_drifts_ms, alpha=0.6, s=10, color='orange')
        ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
        ax.axhline(y=self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=-self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Subtitle Index')
        ax.set_ylabel('End Time Drift (ms)')
        ax.set_title('End Time Drift by Subtitle Index')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Drift over time
        ax = axes[1, 0]
        ax.scatter(expected_starts, start_drifts_ms, alpha=0.6, s=10, label='Start drift')
        ax.scatter(expected_starts, end_drifts_ms, alpha=0.6, s=10, color='orange', label='End drift')
        ax.axhline(y=0, color='g', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time in Audio (seconds)')
        ax.set_ylabel('Drift (ms)')
        ax.set_title('Timestamp Drift Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Drift histogram
        ax = axes[1, 1]
        all_drifts = start_drifts_ms + end_drifts_ms
        ax.hist(all_drifts, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='g', linestyle='-', alpha=0.5)
        ax.axvline(x=self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.axvline(x=-self.tolerance_s*1000, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Drift (ms)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Timestamp Drifts')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Visualization saved to: {output_path}")
        
        plt.show()


def run_timestamp_verification(
    original_audio: Path,
    scene_metadata: Path,
    scene_srts_dir: Path,
    final_srt: Path,
    output_dir: Path,
    tolerance_ms: int = 100,
    verify_audio: bool = False,
    visualize: bool = True
) -> Dict:
    """
    Run complete timestamp verification pipeline.
    
    Args:
        original_audio: Path to original audio file
        scene_metadata: Path to scene metadata JSON
        scene_srts_dir: Directory containing scene SRT files
        final_srt: Path to final stitched SRT file
        output_dir: Directory for output files
        tolerance_ms: Tolerance for timestamp drift in milliseconds
        verify_audio: Whether to perform audio alignment verification
        visualize: Whether to create visualization plots
    
    Returns:
        Dictionary containing all verification results
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tester = TimestampAccuracyTester(tolerance_ms=tolerance_ms, verbose=True)
    
    # Step 1: Verify scene metadata
    logger.info("Step 1: Verifying scene metadata...")
    scene_verification = tester.verify_scene_metadata(scene_metadata)
    
    # Step 2: Prepare scene SRT info
    logger.info("Step 2: Preparing scene SRT information...")
    scene_srts = []
    for scene in scene_verification['scenes']:
        scene_idx = scene['scene_index']
        srt_path = scene_srts_dir / f"{original_audio.stem}_scene_{scene_idx:04d}.srt"
        if srt_path.exists():
            scene_srts.append((srt_path, scene['start_time_s']))
    
    # Step 3: Verify stitching accuracy
    logger.info("Step 3: Verifying stitching accuracy...")
    stitch_verifications = tester.verify_stitching_accuracy(scene_srts, final_srt)
    
    # Step 4: Optional audio alignment verification
    audio_alignment = None
    if verify_audio and scene_verification['scenes']:
        logger.info("Step 4: Verifying audio alignment...")
        scene_audio_paths = []
        scenes_dir = scene_srts_dir.parent / "scenes"  # Assuming audio scenes are here
        for scene in scene_verification['scenes'][:10]:  # Sample first 10
            scene_idx = scene['scene_index']
            scene_path = scenes_dir / f"{original_audio.stem}_scene_{scene_idx:04d}.wav"
            if scene_path.exists():
                scene_audio_paths.append((
                    scene_path,
                    scene['start_time_s'],
                    scene['end_time_s']
                ))
        
        if scene_audio_paths:
            audio_alignment = tester.verify_audio_alignment(
                original_audio,
                scene_audio_paths
            )
    
    # Step 5: Generate report
    logger.info("Step 5: Generating verification report...")
    report = tester.generate_report(
        scene_verification,
        stitch_verifications,
        audio_alignment,
        output_path=output_dir / "timestamp_verification_report.txt"
    )
    print(report)
    
    # Step 6: Create visualizations
    if visualize and stitch_verifications:
        logger.info("Step 6: Creating visualizations...")
        tester.visualize_drift(
            stitch_verifications,
            output_path=output_dir / "timestamp_drift_analysis.png"
        )
    
    return {
        'scene_verification': scene_verification,
        'stitch_verifications': stitch_verifications,
        'audio_alignment': audio_alignment,
        'report': report
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify timestamp accuracy in WhisperJAV's segment-process-stitch pipeline"
    )
    parser.add_argument(
        "--original-audio",
        type=Path,
        required=True,
        help="Path to original audio file"
    )
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        required=True,
        help="Path to scene metadata JSON file"
    )
    parser.add_argument(
        "--scene-srts",
        type=Path,
        required=True,
        help="Directory containing scene SRT files"
    )
    parser.add_argument(
        "--final-srt",
        type=Path,
        required=True,
        help="Path to final stitched SRT file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./timestamp_verification"),
        help="Output directory for reports and visualizations"
    )
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=100,
        help="Tolerance for timestamp drift in milliseconds (default: 100ms)"
    )
    parser.add_argument(
        "--verify-audio",
        action="store_true",
        help="Also verify audio alignment (slower but more thorough)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Skip creating visualization plots"
    )
    
    args = parser.parse_args()
    
    try:
        results = run_timestamp_verification(
            original_audio=args.original_audio,
            scene_metadata=args.scene_metadata,
            scene_srts_dir=args.scene_srts,
            final_srt=args.final_srt,
            output_dir=args.output_dir,
            tolerance_ms=args.tolerance_ms,
            verify_audio=args.verify_audio,
            visualize=not args.no_visualize
        )
        
        # Print summary
        accuracy_rate = sum(
            1 for v in results['stitch_verifications'] 
            if v.is_within_tolerance
        ) / len(results['stitch_verifications']) * 100
        
        print(f"\n✓ Verification complete. Accuracy rate: {accuracy_rate:.1f}%")
        
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise