#!/usr/bin/env python3
"""
TEN VAD Diagnostic Script — Segment Granularity Analysis

Runs TEN VAD on an audio file and shows how parameter combinations
affect segment count and duration. Helps find the optimal settings
for anime-whisper / Kotoba backends on JAV content.

Usage:
    python tools/ten_vad_diagnostic.py <audio_file>
    python tools/ten_vad_diagnostic.py <audio_file> --scenarios
    python tools/ten_vad_diagnostic.py <audio_file> --dump-probs
"""

import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np


def load_audio(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file as float32 numpy array."""
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            from scipy import signal
            num_samples = int(len(audio) * target_sr / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)
            sr = target_sr
        return audio, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        sys.exit(1)


def run_ten_vad_raw(audio: np.ndarray, hop_size: int = 256, threshold: float = 0.20):
    """Run TEN VAD frame-by-frame and return raw probabilities and flags."""
    from ten_vad import TenVad

    model = TenVad(hop_size=hop_size, threshold=threshold)

    # Convert to int16
    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)

    flags = []
    probs = []
    for i in range(0, len(audio_int16) - hop_size, hop_size):
        frame = audio_int16[i:i + hop_size]
        model.process(frame)
        flags.append(model.out_flags.value)
        probs.append(model.out_probability.value)

    return flags, probs


def flags_to_segments(
    flags: List[int],
    probs: List[float],
    frame_duration: float,
    audio_duration: float,
    min_speech_duration_ms: int = 100,
    end_pad_ms: int = 200,
    start_pad_ms: int = 0,
    min_silence_duration_ms: int = 0,
) -> List[Dict[str, Any]]:
    """
    Convert frame-level flags to segments with configurable parameters.

    When min_silence_duration_ms > 0, implements hangover: requires N consecutive
    silence frames before ending a speech segment.
    """
    raw_segments = []
    in_speech = False
    speech_start = 0.0
    speech_start_idx = 0

    # Hangover state
    silence_start_idx = -1  # Frame index where silence started

    min_silence_frames = int(min_silence_duration_ms / (frame_duration * 1000))
    if min_silence_frames < 1:
        min_silence_frames = 1  # At least 1 frame to end speech

    for i, flag in enumerate(flags):
        time_sec = i * frame_duration

        if flag == 1 and not in_speech:
            # Speech started
            in_speech = True
            speech_start = time_sec
            speech_start_idx = i
            silence_start_idx = -1

        elif flag == 1 and in_speech:
            # Speech continues — reset any silence counter
            silence_start_idx = -1

        elif flag == 0 and in_speech:
            if silence_start_idx < 0:
                # First silence frame — start counting
                silence_start_idx = i

            consecutive_silence = i - silence_start_idx + 1
            silence_ms = consecutive_silence * frame_duration * 1000

            if silence_ms >= min_silence_duration_ms or min_silence_duration_ms == 0:
                # Enough consecutive silence — end the speech segment
                in_speech = False
                # End at the FIRST silence frame, not the current one
                speech_end = silence_start_idx * frame_duration
                silence_start_idx = -1

                segment_probs = probs[speech_start_idx:silence_start_idx] if silence_start_idx > speech_start_idx else probs[speech_start_idx:i]
                avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 1.0

                duration_ms = (speech_end - speech_start) * 1000
                if duration_ms >= min_speech_duration_ms:
                    raw_segments.append({
                        "start": speech_start,
                        "end": speech_end,
                        "confidence": avg_confidence,
                    })

    # Handle speech extending to end of audio
    if in_speech:
        speech_end = len(flags) * frame_duration
        segment_probs = probs[speech_start_idx:]
        avg_confidence = sum(segment_probs) / len(segment_probs) if segment_probs else 1.0
        duration_ms = (speech_end - speech_start) * 1000
        if duration_ms >= min_speech_duration_ms:
            raw_segments.append({
                "start": speech_start,
                "end": speech_end,
                "confidence": avg_confidence,
            })

    # Apply padding
    start_pad_sec = start_pad_ms / 1000.0
    end_pad_sec = end_pad_ms / 1000.0
    padded_segments = []
    for i, seg in enumerate(raw_segments):
        padded_start = max(0.0, seg["start"] - start_pad_sec)
        padded_end = min(audio_duration, seg["end"] + end_pad_sec)

        # Prevent overlap with previous segment
        if i > 0 and padded_segments:
            prev_end = padded_segments[-1]["end"]
            if padded_start < prev_end:
                padded_start = prev_end

        if padded_end > padded_start:
            padded_segments.append({
                "start": padded_start,
                "end": padded_end,
                "raw_start": seg["start"],
                "raw_end": seg["end"],
                "confidence": seg["confidence"],
            })

    return padded_segments


def group_segments(
    segments: List[Dict[str, Any]],
    chunk_threshold_s: float = 0.5,
    max_group_duration_s: float = 5.0,
) -> List[List[Dict[str, Any]]]:
    """Group segments by gap and max duration."""
    if not segments:
        return []

    groups: List[List[Dict[str, Any]]] = [[]]

    for i, seg in enumerate(segments):
        if i > 0:
            prev_end = segments[i - 1]["end"]
            gap = seg["start"] - prev_end

            would_exceed_max = False
            if groups[-1]:
                group_start = groups[-1][0]["start"]
                potential_duration = seg["end"] - group_start
                would_exceed_max = potential_duration > max_group_duration_s

            if gap > chunk_threshold_s or would_exceed_max:
                groups.append([])

        groups[-1].append(seg)

    return groups


def analyze_probability_stream(probs: List[float], frame_duration: float, threshold: float):
    """Analyze the raw probability stream for patterns."""
    total_frames = len(probs)
    duration = total_frames * frame_duration

    above = sum(1 for p in probs if p >= threshold)
    below = sum(1 for p in probs if p < threshold)

    print(f"\n{'='*70}")
    print(f"  PROBABILITY STREAM ANALYSIS")
    print(f"{'='*70}")
    print(f"  Total frames:      {total_frames} ({duration:.1f}s)")
    print(f"  Threshold:         {threshold}")
    print(f"  Frames >= thresh:  {above} ({100*above/total_frames:.1f}%)")
    print(f"  Frames <  thresh:  {below} ({100*below/total_frames:.1f}%)")
    print(f"  Prob min/max/mean: {min(probs):.4f} / {max(probs):.4f} / {np.mean(probs):.4f}")
    print(f"  Prob median:       {np.median(probs):.4f}")
    print(f"  Prob std:          {np.std(probs):.4f}")

    # Distribution buckets
    buckets = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5),
               (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    print(f"\n  Probability distribution:")
    for lo, hi in buckets:
        count = sum(1 for p in probs if lo <= p < hi)
        bar = "#" * int(50 * count / total_frames) if count else ""
        marker = " <-- threshold" if lo <= threshold < hi else ""
        print(f"    [{lo:.1f}-{hi:.1f}): {count:6d} ({100*count/total_frames:5.1f}%) {bar}{marker}")

    # Find silence runs (consecutive frames below threshold)
    silence_runs = []
    run_start = -1
    for i, p in enumerate(probs):
        if p < threshold:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0:
                run_len = i - run_start
                silence_runs.append((run_start, run_len))
                run_start = -1
    if run_start >= 0:
        silence_runs.append((run_start, len(probs) - run_start))

    print(f"\n  Silence runs (consecutive frames below {threshold}):")
    print(f"    Total runs:      {len(silence_runs)}")
    if silence_runs:
        run_durations_ms = [r[1] * frame_duration * 1000 for r in silence_runs]
        print(f"    Shortest:        {min(run_durations_ms):.0f}ms ({min(r[1] for r in silence_runs)} frames)")
        print(f"    Longest:         {max(run_durations_ms):.0f}ms ({max(r[1] for r in silence_runs)} frames)")
        print(f"    Mean:            {np.mean(run_durations_ms):.0f}ms")
        print(f"    Median:          {np.median(run_durations_ms):.0f}ms")

        # Duration buckets for silence runs
        dur_buckets = [(0, 16), (16, 32), (32, 64), (64, 100), (100, 200),
                       (200, 500), (500, 1000), (1000, float("inf"))]
        print(f"\n    Silence run duration distribution:")
        for lo, hi in dur_buckets:
            count = sum(1 for d in run_durations_ms if lo <= d < hi)
            hi_label = f"{hi:.0f}" if hi != float("inf") else "inf"
            bar = "#" * min(count, 50) if count else ""
            print(f"      [{lo:5.0f}-{hi_label:>5s}ms): {count:4d}  {bar}")

    # Find speech runs (consecutive frames above threshold)
    speech_runs = []
    run_start = -1
    for i, p in enumerate(probs):
        if p >= threshold:
            if run_start < 0:
                run_start = i
        else:
            if run_start >= 0:
                run_len = i - run_start
                speech_runs.append((run_start, run_len))
                run_start = -1
    if run_start >= 0:
        speech_runs.append((run_start, len(probs) - run_start))

    print(f"\n  Speech runs (consecutive frames above {threshold}):")
    print(f"    Total runs:      {len(speech_runs)}")
    if speech_runs:
        run_durations_s = [r[1] * frame_duration for r in speech_runs]
        print(f"    Shortest:        {min(run_durations_s)*1000:.0f}ms")
        print(f"    Longest:         {max(run_durations_s):.1f}s")
        print(f"    Mean:            {np.mean(run_durations_s):.1f}s")

        # Duration buckets for speech runs
        dur_buckets_s = [(0, 0.1), (0.1, 0.5), (0.5, 1), (1, 2), (2, 5),
                         (5, 10), (10, 30), (30, float("inf"))]
        print(f"\n    Speech run duration distribution:")
        for lo, hi in dur_buckets_s:
            count = sum(1 for d in run_durations_s if lo <= d < hi)
            hi_label = f"{hi:.0f}" if hi != float("inf") else "inf"
            print(f"      [{lo:5.1f}-{hi_label:>5s}s): {count:4d}")


def run_scenario(
    flags: List[int],
    probs: List[float],
    frame_duration: float,
    audio_duration: float,
    threshold: float,
    end_pad_ms: int,
    chunk_threshold_s: float,
    max_group_duration_s: float,
    min_silence_duration_ms: int = 0,
    label: str = "",
) -> Dict[str, Any]:
    """Run one parameter scenario and return results."""
    # Recompute flags at given threshold (probs are threshold-independent)
    flags_at_threshold = [1 if p >= threshold else 0 for p in probs]

    segments = flags_to_segments(
        flags_at_threshold, probs, frame_duration, audio_duration,
        min_speech_duration_ms=100,
        end_pad_ms=end_pad_ms,
        start_pad_ms=0,
        min_silence_duration_ms=min_silence_duration_ms,
    )
    groups = group_segments(segments, chunk_threshold_s, max_group_duration_s)

    group_durations = []
    for g in groups:
        if g:
            g_start = g[0]["start"]
            g_end = g[-1]["end"]
            group_durations.append(g_end - g_start)

    return {
        "label": label,
        "threshold": threshold,
        "end_pad_ms": end_pad_ms,
        "chunk_threshold_s": chunk_threshold_s,
        "max_group_duration_s": max_group_duration_s,
        "min_silence_duration_ms": min_silence_duration_ms,
        "n_raw_segments": len(segments),
        "n_groups": len(groups),
        "group_durations": group_durations,
        "max_group_dur": max(group_durations) if group_durations else 0,
        "mean_group_dur": np.mean(group_durations) if group_durations else 0,
        "segments_per_group": len(segments) / len(groups) if groups else 0,
    }


def print_scenario_result(r: Dict[str, Any]):
    """Print one scenario result."""
    label = r["label"]
    print(f"\n  {label}")
    print(f"  {'─'*60}")
    params = (
        f"  thresh={r['threshold']:.2f}  end_pad={r['end_pad_ms']}ms  "
        f"chunk_thr={r['chunk_threshold_s']:.2f}s  max_group={r['max_group_duration_s']:.0f}s  "
        f"min_sil={r['min_silence_duration_ms']}ms"
    )
    print(params)
    print(f"  Raw segments: {r['n_raw_segments']}  →  Groups: {r['n_groups']}  "
          f"(avg {r['segments_per_group']:.1f} segs/group)")

    if r["group_durations"]:
        durs = r["group_durations"]
        print(f"  Group durations: min={min(durs):.1f}s  max={max(durs):.1f}s  "
              f"mean={r['mean_group_dur']:.1f}s")
        over_5 = sum(1 for d in durs if d > 5.0)
        over_8 = sum(1 for d in durs if d > 8.0)
        over_15 = sum(1 for d in durs if d > 15.0)
        over_30 = sum(1 for d in durs if d > 30.0)
        print(f"  Groups > 5s: {over_5}   > 8s: {over_8}   > 15s: {over_15}   > 30s: {over_30}")

        # Show individual group durations if few enough
        if len(durs) <= 30:
            dur_strs = [f"{d:.1f}s" for d in durs]
            print(f"  All groups: [{', '.join(dur_strs)}]")


def dump_probability_timeline(probs: List[float], frame_duration: float, threshold: float):
    """Print a visual timeline of probabilities (downsampled to 1 row per second)."""
    frames_per_sec = int(1.0 / frame_duration)
    total_secs = len(probs) / frames_per_sec

    print(f"\n{'='*70}")
    print(f"  PROBABILITY TIMELINE (1 char = 1 second)")
    print(f"{'='*70}")
    print(f"  Legend: · <0.10  ░ 0.10-0.20  ▒ 0.20-0.35  ▓ 0.35-0.60  █ 0.60+")
    print(f"  Threshold: {threshold:.2f}\n")

    for sec in range(int(total_secs)):
        start_frame = sec * frames_per_sec
        end_frame = min(start_frame + frames_per_sec, len(probs))
        chunk = probs[start_frame:end_frame]
        avg_p = np.mean(chunk)
        min_p = min(chunk)
        max_p = max(chunk)
        below_count = sum(1 for p in chunk if p < threshold)

        if avg_p < 0.10:
            char = "·"
        elif avg_p < 0.20:
            char = "░"
        elif avg_p < 0.35:
            char = "▒"
        elif avg_p < 0.60:
            char = "▓"
        else:
            char = "█"

        below_pct = 100 * below_count / len(chunk)
        mm = sec // 60
        ss = sec % 60
        print(f"  {mm:02d}:{ss:02d}  {char}  avg={avg_p:.3f}  min={min_p:.3f}  max={max_p:.3f}  "
              f"below_thresh={below_count:3d}/{len(chunk)} ({below_pct:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="TEN VAD Diagnostic — Segment Granularity Analysis")
    parser.add_argument("audio", help="Path to audio file (WAV, MP3, etc.)")
    parser.add_argument("--hop-size", type=int, default=256, help="TEN VAD hop size (default: 256)")
    parser.add_argument("--threshold", type=float, default=0.20, help="TEN VAD threshold (default: 0.20)")
    parser.add_argument("--scenarios", action="store_true", help="Run parameter scenario comparison")
    parser.add_argument("--dump-probs", action="store_true", help="Dump per-second probability timeline")
    parser.add_argument("--all", action="store_true", help="Run everything (analysis + scenarios + timeline)")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"Error: {audio_path} not found")
        sys.exit(1)

    print(f"Loading audio: {audio_path}")
    audio, sr = load_audio(str(audio_path))
    audio_duration = len(audio) / sr
    print(f"Duration: {audio_duration:.1f}s  Sample rate: {sr}")

    print(f"\nRunning TEN VAD (hop_size={args.hop_size}, threshold={args.threshold})...")
    t0 = time.time()
    flags, probs = run_ten_vad_raw(audio, hop_size=args.hop_size, threshold=args.threshold)
    elapsed = time.time() - t0
    frame_duration = args.hop_size / sr
    print(f"Done in {elapsed:.2f}s  ({len(flags)} frames, {frame_duration*1000:.1f}ms/frame)")

    # Always run probability analysis
    analyze_probability_stream(probs, frame_duration, args.threshold)

    # Also show analysis at 0.26 if current threshold is different
    if abs(args.threshold - 0.26) > 0.01:
        print(f"\n  --- Comparison at threshold 0.26 ---")
        above_26 = sum(1 for p in probs if p >= 0.26)
        below_26 = sum(1 for p in probs if p < 0.26)
        silence_runs_26 = []
        run_start = -1
        for i, p in enumerate(probs):
            if p < 0.26:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0:
                    silence_runs_26.append((run_start, i - run_start))
                    run_start = -1
        if run_start >= 0:
            silence_runs_26.append((run_start, len(probs) - run_start))

        print(f"  Frames >= 0.26:    {above_26} ({100*above_26/len(probs):.1f}%)")
        print(f"  Frames <  0.26:    {below_26} ({100*below_26/len(probs):.1f}%)")
        print(f"  Silence runs:      {len(silence_runs_26)}")
        if silence_runs_26:
            run_durs = [r[1] * frame_duration * 1000 for r in silence_runs_26]
            runs_over_100 = sum(1 for d in run_durs if d >= 100)
            runs_over_200 = sum(1 for d in run_durs if d >= 200)
            runs_over_500 = sum(1 for d in run_durs if d >= 500)
            print(f"  Runs >= 100ms:     {runs_over_100}")
            print(f"  Runs >= 200ms:     {runs_over_200}")
            print(f"  Runs >= 500ms:     {runs_over_500}")

    if args.dump_probs or args.all:
        dump_probability_timeline(probs, frame_duration, args.threshold)

    if args.scenarios or args.all:
        print(f"\n{'='*70}")
        print(f"  SCENARIO COMPARISON")
        print(f"{'='*70}")

        scenarios = [
            # Current defaults for anime-whisper
            {"label": "A. CURRENT (anime-whisper defaults)",
             "threshold": 0.20, "end_pad_ms": 200, "chunk_threshold_s": 0.5,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Reduce end_pad only
            {"label": "B. Reduce end_pad to 50ms",
             "threshold": 0.20, "end_pad_ms": 50, "chunk_threshold_s": 0.5,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Reduce chunk_threshold only
            {"label": "C. Reduce chunk_threshold to 0.15s",
             "threshold": 0.20, "end_pad_ms": 200, "chunk_threshold_s": 0.15,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Reduce both pad and chunk
            {"label": "D. Reduce both: end_pad=50ms, chunk=0.15s",
             "threshold": 0.20, "end_pad_ms": 50, "chunk_threshold_s": 0.15,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Threshold 0.26 with current pads
            {"label": "E. Threshold 0.26 (current pads)",
             "threshold": 0.26, "end_pad_ms": 200, "chunk_threshold_s": 0.5,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Threshold 0.26 with reduced pads
            {"label": "F. Threshold 0.26 + reduced pads (end=50, chunk=0.15)",
             "threshold": 0.26, "end_pad_ms": 50, "chunk_threshold_s": 0.15,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Threshold 0.26, reduced pads, with min_silence hangover
            {"label": "G. Thresh 0.26 + reduced pads + min_silence=80ms",
             "threshold": 0.26, "end_pad_ms": 50, "chunk_threshold_s": 0.15,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 80},

            # Threshold 0.26, moderate pads, with min_silence hangover
            {"label": "H. Thresh 0.26 + end_pad=100 + chunk=0.25 + min_sil=80ms",
             "threshold": 0.26, "end_pad_ms": 100, "chunk_threshold_s": 0.25,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 80},

            # More aggressive threshold
            {"label": "I. Threshold 0.30 + end_pad=50 + chunk=0.15",
             "threshold": 0.30, "end_pad_ms": 50, "chunk_threshold_s": 0.15,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 0},

            # Threshold 0.30, with hangover
            {"label": "J. Thresh 0.30 + end_pad=80 + chunk=0.20 + min_sil=60ms",
             "threshold": 0.30, "end_pad_ms": 80, "chunk_threshold_s": 0.20,
             "max_group_duration_s": 5.0, "min_silence_duration_ms": 60},
        ]

        results = []
        for sc in scenarios:
            r = run_scenario(
                flags, probs, frame_duration, audio_duration, **sc
            )
            results.append(r)
            print_scenario_result(r)

        # Summary table
        print(f"\n{'='*70}")
        print(f"  SUMMARY TABLE")
        print(f"{'='*70}")
        header = f"  {'Scenario':<6} {'Thresh':>6} {'Pad':>5} {'Chunk':>6} {'MinSil':>6} │ {'Segs':>5} {'Grps':>5} {'MaxDur':>7} {'MeanDur':>8} {'>8s':>4} {'>15s':>4}"
        print(header)
        print(f"  {'─'*len(header)}")
        for r in results:
            label_short = r["label"].split(".")[0].strip() + "."
            over_8 = sum(1 for d in r["group_durations"] if d > 8.0)
            over_15 = sum(1 for d in r["group_durations"] if d > 15.0)
            print(
                f"  {label_short:<6} {r['threshold']:>6.2f} {r['end_pad_ms']:>5} "
                f"{r['chunk_threshold_s']:>6.2f} {r['min_silence_duration_ms']:>6} │ "
                f"{r['n_raw_segments']:>5} {r['n_groups']:>5} "
                f"{r['max_group_dur']:>7.1f}s {r['mean_group_dur']:>7.1f}s "
                f"{over_8:>4} {over_15:>4}"
            )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
