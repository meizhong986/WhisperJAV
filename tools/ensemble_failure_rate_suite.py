"""
WhisperJAV Ensemble Failure-Rate Test Suite

Probes the v1.8.13 ensemble-mode catastrophe where pass 1 = fidelity + pass 2 =
balanced *intermittently* produces catastrophically truncated pass 2 SRT output.
Same configuration produces correct output on rerun, so the bug is non-deterministic.

The suite runs N iterations of each test configuration as independent subprocesses
and reports the catastrophic-failure rate per configuration. Designed to be:
  - Unattended (auto-runs, JSONL output for incremental capture)
  - Distributable (single file, stdlib only — depends on whisperjav being installed)
  - Cross-machine (auto-captures GPU/driver/OS info)

Usage:
    # Default: 10 runs of fidelity->balanced primary + 3 each of baselines on test media
    python tools/ensemble_failure_rate_suite.py --media path/to/clip.mkv

    # Just the primary config, 20 runs (overnight unattended)
    python tools/ensemble_failure_rate_suite.py --media clip.mkv --runs 20 --primary-only

    # Custom output directory
    python tools/ensemble_failure_rate_suite.py --media clip.mkv \
        --output-root ./failure_rate_results

    # See all options
    python tools/ensemble_failure_rate_suite.py --help

Output structure:
    <output-root>/
      results.jsonl          # one JSON line per run, append-only
      summary.txt            # human-readable aggregate stats per config
      system_info.json       # GPU, driver, OS, Python info (one-time capture)
      runs/
        001_A_fid_bal_<timestamp>/
          log.txt            # full whisperjav stdout/stderr
          *.pass1.srt        # produced subtitle files (if any)
          *.pass2.srt
          *.merged.whisperjav.srt
        002_A_fid_bal_<timestamp>/
        ...

A run is "catastrophic" if pass 2 SRT entry count is below CATASTROPHIC_THRESHOLD
(default 25 — half of healthy ~50-entry baseline). The threshold is configurable.

Test matrix (default):
    A) pass1=fidelity, pass2=balanced  (PRIMARY — the catastrophe trigger)
    B) pass1=balanced, pass2=balanced  (control — should always succeed)
    C) pass1=fast,     pass2=balanced  (lighter pass 1, ctranslate2-based)
    D) pass1=faster,   pass2=balanced  (lightest pass 1, ctranslate2-based)

The non-primary configs run with --baseline-runs (default 3) iterations to give
context. Skip them with --primary-only when you only want the failure-rate signal.

License: same as WhisperJAV.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# A pass 2 SRT with fewer than this many entries is considered catastrophic.
# Healthy baseline (T143) on the reference 293s clip is ~50 entries.
CATASTROPHIC_THRESHOLD_DEFAULT = 25

# Per-run timeout in seconds. Healthy fidelity+balanced on the 293s clip takes
# ~6 minutes; a degraded run can stretch to 10+ minutes. 40 minutes is a hard
# upper bound after which we treat the run as hung.
RUN_TIMEOUT_SECONDS = 2400

# Cool-down between runs (seconds). Lets the GPU return to a baseline state.
INTER_RUN_COOLDOWN_SECONDS = 5

# Default number of runs per configuration
DEFAULT_RUNS_PRIMARY = 10
DEFAULT_RUNS_BASELINE = 3

# --------------------------------------------------------------------------
# Test configuration definitions
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfigSpec:
    """A named test configuration. Mirrors the WhisperJAV ensemble GUI defaults."""
    name: str
    pass1_pipeline: str
    pass2_pipeline: str
    pass1_sensitivity: str = "aggressive"
    pass2_sensitivity: str = "aggressive"
    pass1_scene_detector: str = "semantic"   # GUI default for pass 1
    pass2_scene_detector: str = "auditok"    # GUI default for pass 2
    pass1_speech_segmenter: str = "whisperseg"
    pass2_speech_segmenter: str = "whisperseg"
    pass1_model: str = "large-v2"
    pass2_model: str = "large-v2"
    merge_strategy: str = "pass1_primary"
    subs_language: str = "native"
    language: str = "japanese"


# Built-in configs matching the user's T142/T143 GUI defaults plus v1.8.14 safety-cap validation
#
# Configs A-D mirror the original investigation's test matrix.
# Configs E-F are added in v1.8.14 to validate the ensemble safety cap fix
# (see docs/plans/V1814_T142_NONDETERMINISM_INVESTIGATION.md §15-§17).
#
# - A_fid_bal:        the catastrophic baseline (pass2 sensitivity stays "aggressive" by default)
# - B/C/D:            controls — different pass1, same pass2 = balanced + aggressive
# - E_fid_bal_BAL:    fidelity → balanced WITH pass2 sensitivity downgraded to "balanced".
#                     This is what the safety cap will produce. Expected: 0% catastrophic.
# - F_fid_bal_CON:    fidelity → balanced with pass2 sensitivity = "conservative" (extra-safe).
#                     Expected: 0% catastrophic.
DEFAULT_CONFIGS = {
    "A_fid_bal":      ConfigSpec(name="A_fid_bal",      pass1_pipeline="fidelity", pass2_pipeline="balanced"),
    "B_bal_bal":      ConfigSpec(name="B_bal_bal",      pass1_pipeline="balanced", pass2_pipeline="balanced"),
    "C_fast_bal":     ConfigSpec(name="C_fast_bal",     pass1_pipeline="fast",     pass2_pipeline="balanced"),
    "D_faster_bal":   ConfigSpec(name="D_faster_bal",   pass1_pipeline="faster",   pass2_pipeline="balanced"),
    "E_fid_bal_BAL":  ConfigSpec(name="E_fid_bal_BAL",  pass1_pipeline="fidelity", pass2_pipeline="balanced",
                                 pass2_sensitivity="balanced"),
    "F_fid_bal_CON":  ConfigSpec(name="F_fid_bal_CON",  pass1_pipeline="fidelity", pass2_pipeline="balanced",
                                 pass2_sensitivity="conservative"),
}

PRIMARY_CONFIG_NAME = "A_fid_bal"
SAFETY_CAP_VALIDATION_CONFIGS = ("E_fid_bal_BAL", "F_fid_bal_CON")


# --------------------------------------------------------------------------
# Result records
# --------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a single iteration."""
    config_name: str
    iteration: int
    started_at: str
    completed_at: str
    elapsed_seconds: float
    pass1_entry_count: int
    pass2_entry_count: int
    merged_entry_count: int
    pass1_srt_path: str
    pass2_srt_path: str
    merged_srt_path: str
    log_path: str
    return_code: int
    completed: bool          # subprocess returned (vs. timed out / launch-failed)
    catastrophic: bool       # pass2 entries < threshold but > 0
    success: bool            # completed AND not catastrophic AND pass2 entries > 0
    notes: list = field(default_factory=list)


# --------------------------------------------------------------------------
# System info capture
# --------------------------------------------------------------------------

def _safe_subprocess_text(cmd: list, timeout: int = 10) -> str | None:
    """Run a subprocess and return stripped stdout, or None on any error."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def gather_system_info() -> dict:
    """Collect machine + environment info. Best-effort; missing fields = unavailable."""
    info: dict = {
        "captured_at": datetime.now().isoformat(),
        "platform":     platform.platform(),
        "system":       platform.system(),
        "release":      platform.release(),
        "machine":      platform.machine(),
        "processor":    platform.processor(),
        "python_version": sys.version.replace("\n", " "),
        "python_executable": sys.executable,
    }

    gpu_csv = _safe_subprocess_text(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
         "--format=csv,noheader"],
    )
    if gpu_csv:
        info["nvidia_smi_gpus"] = [line.strip() for line in gpu_csv.splitlines() if line.strip()]
    else:
        info["nvidia_smi_gpus"] = "unavailable"

    info["whisperjav_version"] = _safe_subprocess_text(
        [sys.executable, "-c", "import whisperjav; print(whisperjav.__version__)"]
    ) or "unavailable"

    info["torch_info"] = _safe_subprocess_text(
        [sys.executable, "-c",
         "import torch; print('torch=' + torch.__version__ + ', cuda=' + str(torch.cuda.is_available()) + ', cuda_v=' + str(torch.version.cuda))"]
    ) or "unavailable"

    info["faster_whisper_info"] = _safe_subprocess_text(
        [sys.executable, "-c",
         "import faster_whisper, ctranslate2; print('faster_whisper=' + faster_whisper.__version__ + ', ctranslate2=' + ctranslate2.__version__)"]
    ) or "unavailable"

    return info


# --------------------------------------------------------------------------
# SRT entry counting
# --------------------------------------------------------------------------

def count_srt_entries(srt_path: Path) -> int:
    """Count SRT entries by counting numeric-only lines. Returns 0 if file missing."""
    if not srt_path.exists():
        return 0
    count = 0
    try:
        with open(srt_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                if line.strip().isdigit():
                    count += 1
    except Exception:
        return 0
    return count


def find_first_glob(root: Path, pattern: str) -> Path | None:
    """Return the first file matching the recursive glob pattern under root, or None."""
    matches = sorted(root.glob(f"**/{pattern}"))
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# Single-run execution
# --------------------------------------------------------------------------

def build_whisperjav_command(
    config: ConfigSpec,
    media_path: Path,
    run_dir: Path,
    temp_dir: Path,
) -> list[str]:
    """Construct the exact whisperjav.main CLI invocation matching the user's GUI test."""
    return [
        sys.executable, "-m", "whisperjav.main",
        str(media_path),
        "--ensemble",
        "--pass1-pipeline",         config.pass1_pipeline,
        "--pass1-sensitivity",      config.pass1_sensitivity,
        "--pass1-scene-detector",   config.pass1_scene_detector,
        "--pass1-speech-segmenter", config.pass1_speech_segmenter,
        "--pass1-model",            config.pass1_model,
        "--pass2-pipeline",         config.pass2_pipeline,
        "--pass2-sensitivity",      config.pass2_sensitivity,
        "--pass2-scene-detector",   config.pass2_scene_detector,
        "--pass2-speech-segmenter", config.pass2_speech_segmenter,
        "--pass2-model",            config.pass2_model,
        "--merge-strategy",         config.merge_strategy,
        "--subs-language",          config.subs_language,
        "--language",               config.language,
        "--output-dir",             str(run_dir),
        "--temp-dir",               str(temp_dir),
        "--keep-temp",
        "--debug",
    ]


def run_single_iteration(
    config: ConfigSpec,
    iteration_global: int,
    iteration_in_config: int,
    media_path: Path,
    output_root: Path,
    catastrophic_threshold: int,
) -> RunResult:
    """Execute one whisperjav ensemble run as a subprocess and capture results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / "runs" / f"{iteration_global:03d}_{config.name}_{timestamp}"
    temp_dir = run_dir / "temp"
    run_dir.mkdir(parents=True, exist_ok=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.txt"
    cmd = build_whisperjav_command(config, media_path, run_dir, temp_dir)

    # Persist the command for forensic clarity
    (run_dir / "command.txt").write_text(
        " ".join(f'"{a}"' if " " in a else a for a in cmd) + "\n",
        encoding="utf-8",
    )

    started_at = datetime.now().isoformat()
    t0 = time.time()
    timed_out = False
    return_code = -1
    notes: list[str] = []

    try:
        with open(log_path, 'w', encoding='utf-8', errors='replace') as logf:
            proc = subprocess.run(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
                timeout=RUN_TIMEOUT_SECONDS,
                check=False,
            )
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
        notes.append(f"timed out after {RUN_TIMEOUT_SECONDS}s")
    except FileNotFoundError as e:
        notes.append(f"launch failed: {e}")
    except Exception as e:
        notes.append(f"launch error: {e!r}")

    elapsed = time.time() - t0
    completed_at = datetime.now().isoformat()
    completed = (return_code == 0) and not timed_out

    # Locate produced SRT files
    pass1_srt  = find_first_glob(run_dir, "*.pass1.srt")           or run_dir / "_missing_pass1.srt"
    pass2_srt  = find_first_glob(run_dir, "*.pass2.srt")           or run_dir / "_missing_pass2.srt"
    merged_srt = find_first_glob(run_dir, "*.merged.whisperjav.srt") or run_dir / "_missing_merged.srt"

    p1_count = count_srt_entries(pass1_srt)
    p2_count = count_srt_entries(pass2_srt)
    pm_count = count_srt_entries(merged_srt)

    catastrophic = (0 < p2_count < catastrophic_threshold)
    success = completed and not catastrophic and (p2_count > 0)

    if timed_out:
        notes.append("treated as not-success due to timeout")

    return RunResult(
        config_name=config.name,
        iteration=iteration_global,
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=elapsed,
        pass1_entry_count=p1_count,
        pass2_entry_count=p2_count,
        merged_entry_count=pm_count,
        pass1_srt_path=str(pass1_srt) if pass1_srt.exists() else "",
        pass2_srt_path=str(pass2_srt) if pass2_srt.exists() else "",
        merged_srt_path=str(merged_srt) if merged_srt.exists() else "",
        log_path=str(log_path),
        return_code=return_code,
        completed=completed,
        catastrophic=catastrophic,
        success=success,
        notes=notes,
    )


# --------------------------------------------------------------------------
# Summary writing
# --------------------------------------------------------------------------

def _percentile(values: list[float], q: float) -> float:
    """Simple percentile without numpy. q in [0, 100]."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (q / 100)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def write_summary(results: list[RunResult], output_path: Path, threshold: int) -> str:
    """Emit a human-readable summary text. Returns the summary string."""
    lines: list[str] = []
    lines.append("WhisperJAV Ensemble Failure-Rate Test Suite — Summary")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now().isoformat()}")
    lines.append(f"Total runs: {len(results)}")
    lines.append(f"Catastrophic threshold: pass2 SRT entries < {threshold}")
    lines.append("")

    by_config: dict[str, list[RunResult]] = {}
    for r in results:
        by_config.setdefault(r.config_name, []).append(r)

    # Stable order: primary first, then alphabetical
    ordered = []
    if PRIMARY_CONFIG_NAME in by_config:
        ordered.append(PRIMARY_CONFIG_NAME)
    for name in sorted(by_config):
        if name not in ordered:
            ordered.append(name)

    for cfg_name in ordered:
        runs = by_config[cfg_name]
        n = len(runs)
        cat = sum(1 for r in runs if r.catastrophic)
        ok = sum(1 for r in runs if r.success)
        err = sum(1 for r in runs if not r.completed)
        p2_counts = [r.pass2_entry_count for r in runs if r.completed]
        elapsed = [r.elapsed_seconds for r in runs if r.completed]

        lines.append(f"Config: {cfg_name}")
        lines.append(f"  Runs:                  {n}")
        lines.append(f"  Successful:            {ok}")
        lines.append(f"  Catastrophic:          {cat}  ({(100 * cat / n if n else 0):.1f}%)")
        lines.append(f"  Errored / timed out:   {err}")
        if p2_counts:
            lines.append(f"  pass2 entries  min/p50/max:  {min(p2_counts)} / {_percentile(p2_counts, 50):.0f} / {max(p2_counts)}")
        if elapsed:
            lines.append(f"  Elapsed s     min/p50/max:  {min(elapsed):.0f} / {_percentile(elapsed, 50):.0f} / {max(elapsed):.0f}")
        # Per-iteration breakdown for the primary config (the failure-rate signal)
        if cfg_name == PRIMARY_CONFIG_NAME:
            lines.append("  Per-run:")
            for r in runs:
                tag = "CATASTROPHIC" if r.catastrophic else ("OK" if r.success else ("ERROR" if not r.completed else "DEGRADED"))
                lines.append(f"    iter {r.iteration:>3d}  pass1={r.pass1_entry_count:>3d} pass2={r.pass2_entry_count:>3d}  {r.elapsed_seconds:>5.0f}s  [{tag}]")
        lines.append("")

    summary_text = "\n".join(lines)
    output_path.write_text(summary_text, encoding="utf-8")
    return summary_text


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--media", type=Path, required=True,
                        help="Input media file (.mkv, .mp4, etc.)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS_PRIMARY,
                        help=f"Iterations of the PRIMARY config (default: {DEFAULT_RUNS_PRIMARY})")
    parser.add_argument("--baseline-runs", type=int, default=DEFAULT_RUNS_BASELINE,
                        help=f"Iterations of each baseline config (default: {DEFAULT_RUNS_BASELINE})")
    parser.add_argument("--output-root", type=Path,
                        default=Path("test_media") / "ensemble_failure_rate_results",
                        help="Output directory for results")
    parser.add_argument("--primary-only", action="store_true",
                        help="Run only the PRIMARY config (skip baseline B/C/D)")
    parser.add_argument("--config", type=str, default=None,
                        choices=list(DEFAULT_CONFIGS.keys()),
                        help="Run only the named config. Combine with --runs.")
    parser.add_argument("--validate-safety-cap", action="store_true",
                        help="Run the v1.8.14 safety-cap validation set: A_fid_bal "
                             "(catastrophic baseline) + E_fid_bal_BAL (capped) + F_fid_bal_CON "
                             "(extra-safe). Use --runs to set iterations per config "
                             "(default 10 each). Diff between A and E demonstrates the cap "
                             "is necessary AND sufficient.")
    parser.add_argument("--catastrophic-threshold", type=int,
                        default=CATASTROPHIC_THRESHOLD_DEFAULT,
                        help=f"pass 2 entry count below this = catastrophic (default {CATASTROPHIC_THRESHOLD_DEFAULT})")
    parser.add_argument("--cooldown-seconds", type=int, default=INTER_RUN_COOLDOWN_SECONDS,
                        help=f"Sleep between runs (default {INTER_RUN_COOLDOWN_SECONDS}s)")
    args = parser.parse_args()

    # --- Validation ---
    if not args.media.exists():
        print(f"ERROR: media file not found: {args.media}", file=sys.stderr)
        return 1
    args.media = args.media.resolve()

    # --- Output prep ---
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "runs").mkdir(parents=True, exist_ok=True)

    # --- System info ---
    print("Capturing system info...")
    system_info = gather_system_info()
    (args.output_root / "system_info.json").write_text(
        json.dumps(system_info, indent=2), encoding="utf-8",
    )
    for k in ("nvidia_smi_gpus", "whisperjav_version", "torch_info", "faster_whisper_info"):
        print(f"  {k}: {system_info.get(k)}")

    # --- Build the test plan ---
    if args.validate_safety_cap:
        # v1.8.14 safety-cap validation: A (baseline catastrophic) + E (cap'd) + F (extra-safe)
        configs_with_runs = [
            (DEFAULT_CONFIGS["A_fid_bal"],     args.runs),
            (DEFAULT_CONFIGS["E_fid_bal_BAL"], args.runs),
            (DEFAULT_CONFIGS["F_fid_bal_CON"], args.runs),
        ]
    elif args.config:
        configs_with_runs = [(DEFAULT_CONFIGS[args.config], args.runs)]
    else:
        configs_with_runs = [(DEFAULT_CONFIGS[PRIMARY_CONFIG_NAME], args.runs)]
        if not args.primary_only:
            for name in ("B_bal_bal", "C_fast_bal", "D_faster_bal"):
                configs_with_runs.append((DEFAULT_CONFIGS[name], args.baseline_runs))

    total_runs = sum(n for _, n in configs_with_runs)
    print()
    print(f"Plan: {total_runs} total runs across {len(configs_with_runs)} configurations")
    print(f"Media:        {args.media}")
    print(f"Output root:  {args.output_root}")
    print(f"Catastrophic threshold: pass2 entries < {args.catastrophic_threshold}")
    print()

    # --- Run loop ---
    results: list[RunResult] = []
    results_path = args.output_root / "results.jsonl"

    iter_global = 0
    suite_start = time.time()
    for config, n_runs in configs_with_runs:
        for i in range(1, n_runs + 1):
            iter_global += 1
            print(f"[{iter_global}/{total_runs}] {config.name} iter {i}/{n_runs} — running...")

            result = run_single_iteration(
                config=config,
                iteration_global=iter_global,
                iteration_in_config=i,
                media_path=args.media,
                output_root=args.output_root,
                catastrophic_threshold=args.catastrophic_threshold,
            )
            results.append(result)

            # Append to JSONL incrementally so partial results survive interrupts
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")

            tag = "CATASTROPHIC" if result.catastrophic else ("OK" if result.success else ("ERROR" if not result.completed else "DEGRADED"))
            print(f"    -> {tag} | pass1={result.pass1_entry_count} pass2={result.pass2_entry_count} merged={result.merged_entry_count} | {result.elapsed_seconds:.0f}s")

            if iter_global < total_runs and args.cooldown_seconds > 0:
                time.sleep(args.cooldown_seconds)

    suite_elapsed = time.time() - suite_start

    # --- Summary ---
    summary_path = args.output_root / "summary.txt"
    summary_text = write_summary(results, summary_path, args.catastrophic_threshold)

    print()
    print("=" * 60)
    print(summary_text)
    print(f"Suite total wall-time: {suite_elapsed:.0f}s ({suite_elapsed/60:.1f}min)")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
