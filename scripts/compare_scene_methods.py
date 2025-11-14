#!/usr/bin/env python3
"""
Compare Auditok vs Silero scene detection methods.

Usage:
    python scripts/compare_scene_methods.py video.mp4 --reference reference.srt
"""

import argparse
import subprocess
import json
import sys
from pathlib import Path
import numpy as np


def run_whisperjav(video, method, output_dir):
    """Run whisperjav with specified scene detection method"""
    cmd = [
        'whisperjav',
        str(video),
        '--scene-detection-method', method,
        '--mode', 'balanced',
        '-o', str(output_dir)
    ]

    print(f"\nRunning with {method} method...")
    print(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running whisperjav with {method}: {e}")
        return False


def count_scenes(scenes_dir):
    """Count scene files"""
    if not scenes_dir.exists():
        return 0
    return len(list(scenes_dir.glob('scene_*.wav')))


def analyze_durations(scenes_dir):
    """Analyze scene duration distribution"""
    try:
        import soundfile as sf
    except ImportError:
        print("Warning: soundfile not available, skipping duration analysis")
        return {}

    durations = []
    if not scenes_dir.exists():
        return {}

    for scene_file in scenes_dir.glob('scene_*.wav'):
        try:
            info = sf.info(scene_file)
            durations.append(info.duration)
        except Exception as e:
            print(f"Warning: Could not read {scene_file}: {e}")

    if not durations:
        return {}

    return {
        'count': len(durations),
        'mean': float(np.mean(durations)),
        'median': float(np.median(durations)),
        'min': float(np.min(durations)),
        'max': float(np.max(durations)),
        'std': float(np.std(durations)),
        'under_2s': sum(1 for d in durations if d < 2.0),
        'over_600s': sum(1 for d in durations if d > 600.0),
    }


def compare_subtitles(srt1_path, srt2_path, reference_path=None):
    """Compare two subtitle files"""
    try:
        import pysrt
    except ImportError:
        print("Warning: pysrt not available, skipping subtitle comparison")
        return {}

    comparison = {}

    # Load subtitle files
    try:
        subs1 = pysrt.open(str(srt1_path), encoding='utf-8')
        subs2 = pysrt.open(str(srt2_path), encoding='utf-8')

        comparison['count_method1'] = len(subs1)
        comparison['count_method2'] = len(subs2)
        comparison['diff'] = len(subs2) - len(subs1)

        # Calculate total durations
        if subs1:
            total_dur1 = sum(
                (sub.end.ordinal - sub.start.ordinal) / 1000.0
                for sub in subs1
            )
            comparison['total_duration_method1'] = total_dur1

        if subs2:
            total_dur2 = sum(
                (sub.end.ordinal - sub.start.ordinal) / 1000.0
                for sub in subs2
            )
            comparison['total_duration_method2'] = total_dur2

    except Exception as e:
        print(f"Warning: Error comparing subtitles: {e}")
        return comparison

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare scene detection methods')
    parser.add_argument('video', type=Path, help='Input video file')
    parser.add_argument('--reference', type=Path, help='Reference subtitle file (optional)')
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    # Create output directories
    out_auditok = Path('output_comparison/auditok')
    out_silero = Path('output_comparison/silero')
    out_auditok.mkdir(parents=True, exist_ok=True)
    out_silero.mkdir(parents=True, exist_ok=True)

    # Run both methods
    print("="*60)
    print("SCENE DETECTION METHOD COMPARISON")
    print("="*60)
    print(f"Video: {args.video}")
    print()

    success_auditok = run_whisperjav(args.video, 'auditok', out_auditok)
    success_silero = run_whisperjav(args.video, 'silero', out_silero)

    if not (success_auditok and success_silero):
        print("\nError: One or both methods failed to complete.")
        sys.exit(1)

    # Analyze results
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)

    auditok_stats = analyze_durations(out_auditok / 'scenes')
    silero_stats = analyze_durations(out_silero / 'scenes')

    print(f"\nScene Count:")
    print(f"  Auditok: {auditok_stats.get('count', 0)}")
    print(f"  Silero:  {silero_stats.get('count', 0)}")

    if auditok_stats.get('count', 0) > 0 and silero_stats.get('count', 0) > 0:
        print(f"\nScene Duration Stats:")
        print(f"  Mean:    Auditok {auditok_stats.get('mean', 0):.1f}s  |  Silero {silero_stats.get('mean', 0):.1f}s")
        print(f"  Median:  Auditok {auditok_stats.get('median', 0):.1f}s  |  Silero {silero_stats.get('median', 0):.1f}s")

        print(f"\nFragmentation (<2s scenes):")
        aud_frag = auditok_stats.get('under_2s', 0)
        sil_frag = silero_stats.get('under_2s', 0)
        aud_pct = (aud_frag / auditok_stats['count'] * 100) if auditok_stats['count'] > 0 else 0
        sil_pct = (sil_frag / silero_stats['count'] * 100) if silero_stats['count'] > 0 else 0
        print(f"  Auditok: {aud_frag} ({aud_pct:.1f}%)")
        print(f"  Silero:  {sil_frag} ({sil_pct:.1f}%)")

        print(f"\nOversized (>600s scenes):")
        print(f"  Auditok: {auditok_stats.get('over_600s', 0)}")
        print(f"  Silero:  {silero_stats.get('over_600s', 0)}")

    # Compare subtitles if they exist
    auditok_srt = out_auditok / f"{args.video.stem}.srt"
    silero_srt = out_silero / f"{args.video.stem}.srt"

    if auditok_srt.exists() and silero_srt.exists():
        print(f"\nSubtitle Comparison:")
        sub_comparison = compare_subtitles(auditok_srt, silero_srt, args.reference)

        if sub_comparison:
            print(f"  Auditok subtitle count: {sub_comparison.get('count_method1', 'N/A')}")
            print(f"  Silero subtitle count:  {sub_comparison.get('count_method2', 'N/A')}")
            if 'diff' in sub_comparison:
                print(f"  Difference: {sub_comparison['diff']:+d}")

    # Save detailed report
    report = {
        'video': str(args.video),
        'auditok': auditok_stats,
        'silero': silero_stats,
    }

    report_path = Path('output_comparison/report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nDetailed report saved to: {report_path}")
    print("\nOutput directories:")
    print(f"  Auditok: {out_auditok}")
    print(f"  Silero:  {out_silero}")

    if auditok_srt.exists() and silero_srt.exists():
        print("\nSubtitle files:")
        print(f"  Auditok: {auditok_srt}")
        print(f"  Silero:  {silero_srt}")

    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)

    # Provide recommendation based on results
    if silero_stats.get('under_2s', 0) < auditok_stats.get('under_2s', 0):
        print("Silero produced fewer fragmented scenes (<2s).")

    if silero_stats.get('over_600s', 0) == 0 and auditok_stats.get('over_600s', 0) > 0:
        print("Silero respected the 10-minute max duration constraint better.")

    if silero_stats.get('count', 0) < auditok_stats.get('count', 0):
        print("Silero produced fewer total scenes (better merging).")

    print("\nManually review the subtitle files to assess transcription quality.")
    print("="*60)


if __name__ == '__main__':
    main()
