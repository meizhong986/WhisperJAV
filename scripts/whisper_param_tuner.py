#!/usr/bin/env python3
"""
Whisper Parameter Tuner — Standalone hypothesis testing utility.

Tests Whisper transcription parameters directly against ground truth SRT,
bypassing the full WhisperJAV pipeline. Enables rapid parameter optimization
by running Whisper model calls with configurable parameter combinations and
measuring timestamp coverage against a reference.

Usage:
    # Single parameter test
    python scripts/whisper_param_tuner.py \
        --audio scene_0004.wav \
        --ground-truth Ground-Truth.srt --gt-offset 163.874 \
        --backend openai --model large-v2 \
        --param compression_ratio_threshold=2.4

    # Sweep multiple values
    python scripts/whisper_param_tuner.py \
        --audio scene_0004.wav \
        --ground-truth Ground-Truth.srt --gt-offset 163.874 \
        --backend openai --model large-v2 \
        --gt-focus 43,47 \
        --sweep compression_ratio_threshold=2.2,2.4,2.6 \
        --sweep condition_on_previous_text=true,false
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import soundfile as sf
import srt

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GTSub:
    """Ground truth subtitle entry."""
    id: int
    start: float  # seconds (scene-local after offset applied)
    end: float
    text: str
    original_start: float  # seconds (global, before offset)
    original_end: float


@dataclass
class OutputSegment:
    """A segment from Whisper transcribe() output."""
    start: float
    end: float
    text: str
    temperature: Optional[float] = None
    avg_logprob: Optional[float] = None
    compression_ratio: Optional[float] = None
    no_speech_prob: Optional[float] = None


@dataclass
class GTResult:
    """Coverage result for a single GT subtitle."""
    gt_id: int
    gt_start: float
    gt_end: float
    gt_text: str
    hit: bool
    overlap_sec: float = 0.0
    nearest_gap_sec: float = 0.0  # distance to nearest output segment if miss
    matched_segment_text: str = ""


@dataclass
class RunResult:
    """Result of a single transcription run."""
    run_index: int
    params_changed: Dict[str, Any]
    full_params: Dict[str, Any]
    segments: List[OutputSegment]
    gt_results: List[GTResult]
    gt_coverage_count: int = 0
    gt_coverage_pct: float = 0.0
    gt_total: int = 0
    missing_ids: List[int] = field(default_factory=list)
    total_output_subs: int = 0
    elapsed_sec: float = 0.0
    temperature_distribution: Dict[str, int] = field(default_factory=dict)
    json_path: Optional[str] = None


# ---------------------------------------------------------------------------
# Default parameters (current aggressive presets)
# ---------------------------------------------------------------------------

OPENAI_DEFAULTS = {
    "task": "transcribe",
    "language": "ja",
    "beam_size": 5,
    "best_of": 3,
    "patience": 2.5,
    "temperature": [0.0, 0.15, 0.3, 0.5],
    "compression_ratio_threshold": 2.2,
    "logprob_threshold": -2.0,
    "no_speech_threshold": 0.60,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "fp16": True,
    "suppress_blank": False,
    "suppress_tokens": [],
    "verbose": None,
}

FASTER_DEFAULTS = {
    "task": "transcribe",
    "language": "ja",
    "beam_size": 4,
    "best_of": 3,
    "patience": 2.5,
    "temperature": [0.0, 0.15, 0.3, 0.5],
    "compression_ratio_threshold": 2.2,
    "log_prob_threshold": -2.0,
    "no_speech_threshold": 0.55,
    "condition_on_previous_text": False,
    "word_timestamps": True,
    "vad_filter": False,
    "suppress_blank": False,
    "suppress_tokens": [],
    "repetition_penalty": 1.3,
    "no_repeat_ngram_size": 2,
    "log_progress": False,
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_ground_truth(
    srt_path: Path,
    offset: float = 0.0,
    focus_ids: Optional[List[int]] = None,
    audio_duration: Optional[float] = None,
) -> List[GTSub]:
    """Load and filter ground truth SRT, adjusting for scene offset."""
    with open(srt_path, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    gt_subs = []
    for sub in subtitles:
        global_start = sub.start.total_seconds()
        global_end = sub.end.total_seconds()
        local_start = global_start - offset
        local_end = global_end - offset

        # Filter to subs that fall within the audio range
        if audio_duration is not None:
            if local_end < 0 or local_start > audio_duration:
                continue

        gt_subs.append(GTSub(
            id=sub.index,
            start=max(0.0, local_start),
            end=local_end,
            text=sub.content.replace("\n", " ").strip(),
            original_start=global_start,
            original_end=global_end,
        ))

    if focus_ids:
        gt_subs = [g for g in gt_subs if g.id in focus_ids]

    return gt_subs


def load_audio(audio_path: Path) -> Tuple[np.ndarray, int]:
    """Load audio file as mono float32 numpy array."""
    audio_data, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)
    return audio_data, sample_rate


def load_model(backend: str, model_name: str, device: str = "cuda"):
    """Load Whisper model. Returns the model object."""
    if backend == "openai":
        import whisper
        print(f"Loading OpenAI Whisper model: {model_name} on {device}...")
        return whisper.load_model(model_name, device=device)
    elif backend == "faster":
        from faster_whisper import WhisperModel
        print(f"Loading Faster-Whisper model: {model_name} on {device}...")
        return WhisperModel(
            model_size_or_path=model_name,
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
            cpu_threads=0,
            num_workers=1,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def run_transcribe(
    model, audio: np.ndarray, backend: str, params: Dict
) -> Tuple[List[OutputSegment], Any]:
    """Run Whisper transcribe() and return structured segments + raw result."""
    segments = []
    raw_result = None

    if backend == "openai":
        result = model.transcribe(audio, **params)
        raw_result = result
        for seg in result.get("segments", []):
            segments.append(OutputSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg.get("text", "").strip(),
                temperature=seg.get("temperature"),
                avg_logprob=seg.get("avg_logprob"),
                compression_ratio=seg.get("compression_ratio"),
                no_speech_prob=seg.get("no_speech_prob"),
            ))

    elif backend == "faster":
        segments_gen, info = model.transcribe(audio, **params)
        raw_segments = []
        for seg in segments_gen:
            segments.append(OutputSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text.strip() if seg.text else "",
                temperature=seg.temperature,
                avg_logprob=seg.avg_logprob,
                compression_ratio=seg.compression_ratio,
                no_speech_prob=seg.no_speech_prob,
            ))
            try:
                raw_segments.append(asdict(seg))
            except Exception:
                raw_segments.append({
                    "start": seg.start, "end": seg.end,
                    "text": seg.text, "avg_logprob": seg.avg_logprob,
                })
        raw_result = {"segments": raw_segments}

    return segments, raw_result


def compute_metrics(
    segments: List[OutputSegment], gt_subs: List[GTSub]
) -> Tuple[List[GTResult], Dict]:
    """Compute coverage metrics: which GT subs have timestamp overlap."""
    gt_results = []

    for gt in gt_subs:
        best_overlap = 0.0
        nearest_gap = float("inf")
        matched_text = ""

        for seg in segments:
            # Check overlap
            overlap_start = max(gt.start, seg.start)
            overlap_end = min(gt.end, seg.end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    matched_text = seg.text

            # Track nearest gap for misses
            if seg.end < gt.start:
                gap = gt.start - seg.end
            elif seg.start > gt.end:
                gap = seg.start - gt.end
            else:
                gap = 0.0
            nearest_gap = min(nearest_gap, gap)

        hit = best_overlap > 0.0
        gt_results.append(GTResult(
            gt_id=gt.id,
            gt_start=gt.start,
            gt_end=gt.end,
            gt_text=gt.text[:50],
            hit=hit,
            overlap_sec=round(best_overlap, 3),
            nearest_gap_sec=round(nearest_gap, 3) if not hit else 0.0,
            matched_segment_text=matched_text[:50],
        ))

    coverage_count = sum(1 for r in gt_results if r.hit)
    total = len(gt_results)

    summary = {
        "gt_coverage_count": coverage_count,
        "gt_coverage_pct": round(100.0 * coverage_count / total, 1) if total else 0.0,
        "gt_total": total,
        "missing_ids": [r.gt_id for r in gt_results if not r.hit],
        "total_output_subs": len(segments),
    }

    return gt_results, summary


def save_full_results(raw_result: Any, output_path: Path) -> None:
    """Save full Whisper transcribe() results as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(raw_result, f, ensure_ascii=False, indent=2, default=str)


def parse_param_value(value_str: str) -> Any:
    """Parse a parameter value string into the appropriate Python type."""
    lower = value_str.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "none":
        return None
    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        # Check for list syntax: [0.0,0.15,0.3]
        if value_str.startswith("[") and value_str.endswith("]"):
            items = value_str[1:-1].split(",")
            return [parse_param_value(i.strip()) for i in items]
        return value_str


def build_param_grid(
    base_params: Dict, overrides: Dict, sweeps: List[Tuple[str, List]]
) -> List[Dict]:
    """Build parameter grid from base + overrides + sweeps."""
    if not sweeps:
        merged = {**base_params, **overrides}
        return [merged]

    sweep_keys = [s[0] for s in sweeps]
    sweep_values = [s[1] for s in sweeps]

    grid = []
    for combo in product(*sweep_values):
        params = {**base_params, **overrides}
        for key, val in zip(sweep_keys, combo):
            params[key] = val
        grid.append(params)

    return grid


def get_temp_distribution(segments: List[OutputSegment]) -> Dict[str, int]:
    """Count segments by temperature used."""
    dist = {}
    for seg in segments:
        key = str(seg.temperature)
        dist[key] = dist.get(key, 0) + 1
    return dist


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------

def print_header(args, audio_duration: float, gt_subs: List[GTSub]):
    """Print the report header."""
    print()
    print("=" * 70)
    print("  Whisper Parameter Tuner")
    print("=" * 70)
    print(f"  Audio:  {Path(args.audio).name} ({audio_duration:.1f}s)")
    print(f"  GT:     {len(gt_subs)} subs in range", end="")
    if gt_subs:
        print(f" (GT#{gt_subs[0].id}–GT#{gt_subs[-1].id}, offset={args.gt_offset}s)")
    else:
        print()
    if args.gt_focus:
        print(f"  Focus:  GT#{', GT#'.join(str(i) for i in args.gt_focus)}")
    print(f"  Backend: {args.backend} ({args.model}, {args.device})")
    print("=" * 70)


def print_run_result(result: RunResult, focus_ids: Optional[List[int]] = None):
    """Print a single run's results."""
    changed_str = ", ".join(f"{k}={v}" for k, v in result.params_changed.items())
    print(f"\n--- Run {result.run_index}: {changed_str} ---")
    print(f"  Output: {result.total_output_subs} segments", end="")
    if result.temperature_distribution:
        temp_parts = [f"temp={k}: {v}" for k, v in sorted(result.temperature_distribution.items())]
        print(f" ({', '.join(temp_parts)})")
    else:
        print()
    print(f"  GT Coverage: {result.gt_coverage_count}/{result.gt_total} ({result.gt_coverage_pct}%)")

    if result.missing_ids:
        for gr in result.gt_results:
            if not gr.hit:
                print(f"  MISS: GT#{gr.gt_id} ({gr.gt_start:.2f}–{gr.gt_end:.2f}s) "
                      f"gap={gr.nearest_gap_sec:.2f}s | {gr.gt_text}")

    if focus_ids:
        print(f"  Focus results:")
        for gr in result.gt_results:
            if gr.gt_id in focus_ids:
                status = "HIT" if gr.hit else "MISS"
                detail = f"overlap={gr.overlap_sec:.2f}s" if gr.hit else f"gap={gr.nearest_gap_sec:.2f}s"
                print(f"    GT#{gr.gt_id}: {status} ({detail})")

    print(f"  Time: {result.elapsed_sec:.1f}s")
    if result.json_path:
        print(f"  Saved: {result.json_path}")


def print_sweep_summary(results: List[RunResult], focus_ids: Optional[List[int]] = None):
    """Print the sweep summary."""
    if len(results) <= 1:
        return

    print()
    print("=" * 70)
    print("  SWEEP SUMMARY")
    print("=" * 70)

    # Best overall coverage
    best = max(results, key=lambda r: (r.gt_coverage_count, -r.elapsed_sec))
    changed_str = ", ".join(f"{k}={v}" for k, v in best.params_changed.items())
    print(f"  Best Coverage: Run {best.run_index} ({changed_str}) "
          f"— {best.gt_coverage_count}/{best.gt_total} ({best.gt_coverage_pct}%)")

    # Best for each focus ID
    if focus_ids:
        for fid in focus_ids:
            hits = [r for r in results if any(
                gr.gt_id == fid and gr.hit for gr in r.gt_results
            )]
            if hits:
                run_strs = [str(r.run_index) for r in hits]
                print(f"  Best for GT#{fid}: Runs {', '.join(run_strs)}")
            else:
                print(f"  Best for GT#{fid}: NO run recovered this sub")

    # Fastest
    fastest = min(results, key=lambda r: r.elapsed_sec)
    changed_str = ", ".join(f"{k}={v}" for k, v in fastest.params_changed.items())
    print(f"  Fastest: Run {fastest.run_index} ({changed_str}) — {fastest.elapsed_sec:.1f}s")

    # Comparison table
    print()
    print("  Run | Coverage | Time   | Changed Parameters")
    print("  " + "-" * 60)
    for r in results:
        changed_str = ", ".join(f"{k}={v}" for k, v in r.params_changed.items())
        print(f"  {r.run_index:3d} | {r.gt_coverage_count:3d}/{r.gt_total:<3d}  "
              f"| {r.elapsed_sec:5.1f}s | {changed_str}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Whisper Parameter Tuner — test parameter hypotheses against ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Path to audio WAV file")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth SRT")
    parser.add_argument("--gt-offset", type=float, default=0.0,
                        help="Scene offset in seconds (subtracted from GT timestamps)")
    parser.add_argument("--gt-focus", type=str, default=None,
                        help="Comma-separated GT subtitle IDs to focus on (e.g., 43,47)")
    parser.add_argument("--backend", choices=["openai", "faster"], default="openai",
                        help="Whisper backend to use")
    parser.add_argument("--model", default="large-v2", help="Model name (default: large-v2)")
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument("--param", action="append", default=[],
                        help="Override a parameter: --param key=value")
    parser.add_argument("--sweep", action="append", default=[],
                        help="Sweep parameter values: --sweep key=val1,val2,val3")
    parser.add_argument("--output-dir", default=None,
                        help="Directory for result JSONs (default: alongside audio)")

    args = parser.parse_args()

    # Parse focus IDs
    focus_ids = None
    if args.gt_focus:
        focus_ids = [int(x.strip()) for x in args.gt_focus.split(",")]
    args.gt_focus = focus_ids

    # Load audio
    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    audio, sample_rate = load_audio(audio_path)
    audio_duration = len(audio) / sample_rate
    print(f"Loaded audio: {audio_path.name} ({audio_duration:.1f}s, {sample_rate}Hz)")

    # Load ground truth
    gt_path = Path(args.ground_truth)
    if not gt_path.exists():
        print(f"ERROR: Ground truth SRT not found: {gt_path}", file=sys.stderr)
        sys.exit(1)

    gt_subs = load_ground_truth(gt_path, args.gt_offset, focus_ids, audio_duration)
    if not gt_subs:
        print("WARNING: No GT subs in range after offset filtering.", file=sys.stderr)

    # Parse param overrides
    overrides = {}
    for p in args.param:
        key, val = p.split("=", 1)
        overrides[key] = parse_param_value(val)

    # Parse sweeps
    sweeps = []
    for s in args.sweep:
        key, vals = s.split("=", 1)
        values = [parse_param_value(v.strip()) for v in vals.split(",")]
        sweeps.append((key, values))

    # Build parameter grid
    base_params = OPENAI_DEFAULTS.copy() if args.backend == "openai" else FASTER_DEFAULTS.copy()
    param_grid = build_param_grid(base_params, overrides, sweeps)

    # Determine output directory
    output_dir = Path(args.output_dir) if args.output_dir else audio_path.parent / "tuner_results"

    # Load model (once)
    model = load_model(args.backend, args.model, args.device)

    # Print header
    print_header(args, audio_duration, gt_subs)

    # Run each parameter combination
    results = []
    for idx, params in enumerate(param_grid, 1):
        # Identify what changed from base
        changed = {}
        for key, val in params.items():
            if key in base_params and val != base_params[key]:
                changed[key] = val
        # Also include overrides that match base (user explicitly set them)
        for key, val in overrides.items():
            if key not in changed:
                changed[key] = val

        changed_str = ", ".join(f"{k}={v}" for k, v in changed.items()) or "(defaults)"
        print(f"\nRunning {idx}/{len(param_grid)}: {changed_str}...", flush=True)

        # Run transcription
        t0 = time.time()
        segments, raw_result = run_transcribe(model, audio, args.backend, params)
        elapsed = time.time() - t0

        # Save full results JSON
        json_path = output_dir / f"run_{idx:03d}.transcribe.json"
        save_full_results(raw_result, json_path)

        # Compute metrics
        gt_results, summary = compute_metrics(segments, gt_subs)

        run_result = RunResult(
            run_index=idx,
            params_changed=changed,
            full_params=params,
            segments=segments,
            gt_results=gt_results,
            gt_coverage_count=summary["gt_coverage_count"],
            gt_coverage_pct=summary["gt_coverage_pct"],
            gt_total=summary["gt_total"],
            missing_ids=summary["missing_ids"],
            total_output_subs=summary["total_output_subs"],
            elapsed_sec=round(elapsed, 1),
            temperature_distribution=get_temp_distribution(segments),
            json_path=str(json_path),
        )
        results.append(run_result)

        print_run_result(run_result, focus_ids)

    # Print sweep summary
    print_sweep_summary(results, focus_ids)
    print()


if __name__ == "__main__":
    main()
