#!/usr/bin/env python3
"""
Audio Analysis Script for Scene Detection Tuning
Analyzes audio files to provide metrics for energy_threshold tuning
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from scipy import signal
import warnings
warnings.filterwarnings('ignore')


def analyze_audio_file(file_path):
    """Comprehensive audio analysis for scene detection tuning"""

    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(file_path).name}")
    print(f"{'='*80}\n")

    # Load audio
    y, sr = librosa.load(file_path, sr=None, mono=True)
    duration = len(y) / sr

    # Get file info using soundfile
    info = sf.info(file_path)

    print("## 1. BASIC INFORMATION")
    print(f"Duration:        {duration:.2f} seconds")
    print(f"Sample Rate:     {sr} Hz")
    print(f"Channels:        {info.channels}")
    print(f"Samples:         {len(y):,}")
    print(f"Format:          {info.format}, {info.subtype}")

    # RMS Analysis
    print("\n## 2. RMS ANALYSIS")

    # Overall RMS
    rms_overall = np.sqrt(np.mean(y**2))
    rms_db_overall = 20 * np.log10(rms_overall + 1e-10)

    # RMS over time windows (100ms windows)
    frame_length = int(0.1 * sr)  # 100ms
    hop_length = frame_length // 2
    rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db_frames = 20 * np.log10(rms_frames + 1e-10)

    print(f"Overall RMS:     {rms_overall:.6f}")
    print(f"Overall RMS dB:  {rms_db_overall:.2f} dB")
    print(f"Min RMS:         {np.min(rms_frames):.6f} ({np.min(rms_db_frames):.2f} dB)")
    print(f"Max RMS:         {np.max(rms_frames):.6f} ({np.max(rms_db_frames):.2f} dB)")
    print(f"Median RMS:      {np.median(rms_frames):.6f} ({np.median(rms_db_frames):.2f} dB)")
    print(f"Std RMS:         {np.std(rms_frames):.6f}")

    # Energy Analysis
    print("\n## 3. ENERGY ANALYSIS")

    # Frame-based energy
    energy_frames = rms_frames ** 2
    energy_db = 10 * np.log10(energy_frames + 1e-10)

    print(f"Mean Energy:     {np.mean(energy_frames):.8f}")
    print(f"Mean Energy dB:  {np.mean(energy_db):.2f} dB")
    print(f"Min Energy dB:   {np.min(energy_db):.2f} dB")
    print(f"Max Energy dB:   {np.max(energy_db):.2f} dB")
    print(f"Median Energy dB: {np.median(energy_db):.2f} dB")
    print(f"Std Energy dB:   {np.std(energy_db):.2f} dB")

    # Percentiles
    print("\n## 4. dB LEVEL PERCENTILES")
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = np.percentile(energy_db, p)
        print(f"{p:3d}th percentile: {val:6.2f} dB")

    # Peak Analysis
    print("\n## 5. PEAK ANALYSIS")
    peak_amplitude = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak_amplitude + 1e-10)
    avg_db = 20 * np.log10(np.mean(np.abs(y)) + 1e-10)
    dynamic_range = peak_db - np.min(rms_db_frames)

    print(f"Peak Amplitude:  {peak_amplitude:.6f}")
    print(f"Peak dB:         {peak_db:.2f} dB")
    print(f"Average dB:      {avg_db:.2f} dB")
    print(f"Dynamic Range:   {dynamic_range:.2f} dB")

    # Silence Detection Analysis
    print("\n## 6. SILENCE/QUIET REGION ANALYSIS")

    # Try different energy thresholds
    thresholds_db = [25, 28, 30, 32, 35, 38, 40, 42, 45, 48, 50]

    print("\nSilence detection at various thresholds:")
    print(f"{'Threshold (dB)':>15} | {'% Silent':>10} | {'# Regions':>10} | {'Interpretation':>20}")
    print("-" * 80)

    silence_stats = []

    for thresh_db in thresholds_db:
        # Count frames below threshold
        silent_frames = np.sum(energy_db < thresh_db)
        silent_percent = (silent_frames / len(energy_db)) * 100

        # Count contiguous silent regions (min 3 frames = ~150ms)
        is_silent = energy_db < thresh_db
        silent_regions = 0
        in_region = False
        region_length = 0

        for is_s in is_silent:
            if is_s:
                if not in_region:
                    in_region = True
                    region_length = 1
                else:
                    region_length += 1
            else:
                if in_region and region_length >= 3:
                    silent_regions += 1
                in_region = False
                region_length = 0

        if in_region and region_length >= 3:
            silent_regions += 1

        # Interpretation
        if silent_percent < 5:
            interp = "Very aggressive"
        elif silent_percent < 15:
            interp = "Aggressive"
        elif silent_percent < 30:
            interp = "Balanced"
        elif silent_percent < 50:
            interp = "Conservative"
        else:
            interp = "Very conservative"

        print(f"{thresh_db:>15.0f} | {silent_percent:>9.1f}% | {silent_regions:>10d} | {interp:>20}")
        silence_stats.append((thresh_db, silent_percent, silent_regions))

    # Energy Distribution Histogram
    print("\n## 7. ENERGY DISTRIBUTION (dB)")

    # Create histogram bins
    hist_bins = np.arange(-80, 0, 5)
    hist, _ = np.histogram(energy_db, bins=hist_bins)

    print("\nHistogram (Energy dB distribution):")
    print(f"{'Range (dB)':>15} | {'Count':>8} | {'%':>6} | {'Bar':>30}")
    print("-" * 80)

    for i in range(len(hist)):
        if hist[i] > 0:
            bin_start = hist_bins[i]
            bin_end = hist_bins[i+1] if i < len(hist_bins)-1 else hist_bins[i] + 5
            percent = (hist[i] / len(energy_db)) * 100
            bar_length = int((hist[i] / np.max(hist)) * 30)
            bar = '#' * bar_length
            print(f"{bin_start:>6.0f} to {bin_end:>5.0f} | {hist[i]:>8d} | {percent:>5.1f}% | {bar}")

    # Recommendations
    print("\n## 8. SCENE DETECTION THRESHOLD RECOMMENDATIONS")
    print("\nBased on energy distribution analysis:\n")

    # Find optimal thresholds
    p10 = np.percentile(energy_db, 10)
    p25 = np.percentile(energy_db, 25)
    median = np.percentile(energy_db, 50)

    print(f"AGGRESSIVE (catch most silences):    {p25:.0f} dB")
    print(f"  - Catches ~75% of quietest moments")
    print(f"  - Use when: Audio has clear voice/silence distinction")
    print(f"  - Risk: May over-split continuous speech")

    print(f"\nBALANCED (recommended starting point): {p10:.0f} dB")
    print(f"  - Catches ~90% of quietest moments")
    print(f"  - Use when: Mixed content, unsure of audio characteristics")
    print(f"  - Risk: Moderate - good middle ground")

    # Find where ~5% is silent
    conservative_thresh = None
    for thresh_db, silent_pct, _ in silence_stats:
        if silent_pct >= 5 and silent_pct <= 15:
            conservative_thresh = thresh_db
            break

    if conservative_thresh:
        print(f"\nCONSERVATIVE (avoid false splits):   {conservative_thresh:.0f} dB")
        print(f"  - Catches only clearest silence regions")
        print(f"  - Use when: Audio has background music, noise, or continuous speech")
        print(f"  - Risk: May miss subtle scene boundaries")

    # Auditok-specific recommendations
    print("\n## 9. AUDITOK CONFIGURATION MAPPING")
    print("\nFor whisperjav scene_detection.py (uses auditok):")
    print("Note: Auditok uses POSITIVE energy_threshold (higher = more sensitive)")
    print()

    # Auditok typically works in range 30-55
    # Lower threshold = more sensitive (detects more silence)
    auditok_aggressive = max(30, min(45, int(p25) + 50))
    auditok_balanced = max(35, min(50, int(p10) + 50))
    auditok_conservative = max(40, min(55, int(p10) + 55))

    print(f"sensitivity='aggressive':   energy_threshold={auditok_aggressive}")
    print(f"sensitivity='balanced':     energy_threshold={auditok_balanced}")
    print(f"sensitivity='conservative': energy_threshold={auditok_conservative}")

    print("\nNote: These are starting points. Fine-tune based on actual scene detection results.")

    return {
        'duration': duration,
        'sr': sr,
        'rms_overall': rms_overall,
        'rms_db_overall': rms_db_overall,
        'energy_db_mean': np.mean(energy_db),
        'energy_db_percentiles': {p: np.percentile(energy_db, p) for p in percentiles},
        'recommended_thresholds': {
            'aggressive': p25,
            'balanced': p10,
            'conservative': conservative_thresh
        }
    }


def main():
    """Analyze both audio files"""

    files = [
        r"C:\BIN\git\whisperJav_V1_Minami_Edition\tests\short_15_sec_test-966-00_01_45-00_01_59.wav",
        r"C:\BIN\git\whisperJav_V1_Minami_Edition\tests\MIAA-432.5sec.wav"
    ]

    results = {}

    for file_path in files:
        if Path(file_path).exists():
            results[Path(file_path).name] = analyze_audio_file(file_path)
        else:
            print(f"\nERROR: File not found: {file_path}")

    # Summary comparison
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("COMPARATIVE SUMMARY")
        print(f"{'='*80}\n")

        for name, data in results.items():
            print(f"{name}:")
            print(f"  Duration: {data['duration']:.2f}s")
            print(f"  Mean Energy: {data['energy_db_mean']:.2f} dB")
            print(f"  Recommended thresholds:")
            print(f"    Aggressive:    {data['recommended_thresholds']['aggressive']:.0f} dB")
            print(f"    Balanced:      {data['recommended_thresholds']['balanced']:.0f} dB")
            if data['recommended_thresholds']['conservative']:
                print(f"    Conservative:  {data['recommended_thresholds']['conservative']:.0f} dB")
            print()


if __name__ == "__main__":
    main()
