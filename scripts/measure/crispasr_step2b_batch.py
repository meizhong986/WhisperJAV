"""CrispASR step 2b: the twelve scored runs one after another, with the GPU's memory sampled during each (CrispASR
master plan 4.5; U2).

For each configuration of crispasr_step2b.STEP_2B_CONFIGURATIONS and each film, in that order, the script runs
    python scripts/measure/crispasr_step2b.py <film> --pipeline <p> --model <m> --slice-form <s> --out <batch>
in its own process, and beside it, for the same time,
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits -lms 250
which samples the whole card every 0.25 s. Windows reports no per-program GPU memory on this card's driver mode, so the
program's own share is read from whole-card samples:
    before    the median memory in use in the 3 s before the CrispASR program started: WhisperJAV's PyTorch context
              and every other program on the card
    peak      the most in use while the CrispASR program ran
    program   peak - before
The program's start and end come from its run.log, which the runner writes when the program ends and which holds the
run's seconds (crispasr_step2b.py keeps a copy with its time).

Writes, beside each run's own files in <batch>/<model>_<slice form>/:
    <film>.console.log   everything the run printed
    <film>.gpu.csv       the samples taken while it ran
and <batch>/batch.json with, per run: the command, exit status, seconds, and the GPU figures above (MiB).
A run whose <film>.run.json exists is skipped, so an interrupted batch resumes. --dry-run lists the runs.

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2b_batch.py --out <batch> [--films <film> <film>] [--dry-run]
"""
import argparse
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEP2B = REPO / "scripts" / "measure" / "crispasr_step2b.py"
NETFLIX = REPO / "test_media" / "Ground_Truths" / "Netflix"
FILMS = (NETFLIX / "The.Naked.Director.S01E04.Scene.4.mkv",
         NETFLIX / "The.Naked.Director.S02E02.Scene.5.Bubble.More.mkv")
SAMPLE_MS = 250
BEFORE_S = 3.0


def step2b_module():
    spec = importlib.util.spec_from_file_location("crispasr_step2b", STEP2B)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_samples(path: Path) -> list:
    """(time, MiB used, MiB total, utilisation %) of each complete sample line."""
    samples = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return samples
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            samples.append((datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f"), int(parts[1]), int(parts[2]),
                            int(parts[3])))
        except ValueError:
            continue
    return samples


def gpu_figures(samples: list, run_log: Path, seconds) -> dict:
    figures = {"samples": len(samples), "whole_run_peak_mib": max((s[1] for s in samples), default=None),
               "first_sample_mib": samples[0][1] if samples else None,
               "card_mib": samples[0][2] if samples else None}
    if not samples or seconds is None or not run_log.is_file():
        figures["program"] = None
        figures["why_no_program_figures"] = "no samples, or no program run (run.log or its seconds missing)"
        return figures
    end = datetime.fromtimestamp(run_log.stat().st_mtime)
    start = end - timedelta(seconds=float(seconds))
    before = [s[1] for s in samples if start - timedelta(seconds=BEFORE_S) <= s[0] < start]
    during = [s for s in samples if start <= s[0] <= end]
    if not before or not during:
        figures["program"] = None
        figures["why_no_program_figures"] = f"{len(before)} samples before the program, {len(during)} while it ran"
        return figures
    peak = max(s[1] for s in during)
    base = statistics.median(before)
    figures["program"] = {"start": start.isoformat(timespec="milliseconds"), "end": end.isoformat(timespec="milliseconds"),
                          "seconds": float(seconds), "samples_while_running": len(during),
                          "before_mib": base, "peak_mib": peak, "program_mib": peak - base,
                          "utilisation_max_pct": max(s[3] for s in during),
                          "utilisation_median_pct": statistics.median(s[3] for s in during)}
    return figures


def run_one(cfg_label: str, folder: Path, film: Path, argv: list, env: dict) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    console, gpu_csv = folder / f"{film.stem}.console.log", folder / f"{film.stem}.gpu.csv"
    started = time.time()
    with open(gpu_csv, "w", encoding="utf-8") as gpu_out:
        sampler = subprocess.Popen(["nvidia-smi", "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
                                    "--format=csv,noheader,nounits", "-lms", str(SAMPLE_MS)],
                                   stdout=gpu_out, stderr=subprocess.DEVNULL)
        try:
            time.sleep(BEFORE_S + 1.0)  # the card before anything starts
            with open(console, "w", encoding="utf-8", errors="replace") as log:
                rc = subprocess.run(argv, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            time.sleep(1.0)
        finally:
            sampler.terminate()
            try:
                sampler.wait(timeout=10)
            except subprocess.TimeoutExpired:
                sampler.kill()
    run_json = folder / f"{film.stem}.run.json"
    record = json.loads(run_json.read_text(encoding="utf-8")) if run_json.is_file() else {}
    seconds = (record.get("program_run") or {}).get("seconds")
    return {"configuration": cfg_label, "film": film.name, "command": argv, "exit_status": rc,
            "seconds": round(time.time() - started, 1), "run_json": str(run_json) if run_json.is_file() else None,
            "error": record.get("error"), "gpu": gpu_figures(read_samples(gpu_csv), folder / f"{film.stem}.run.log",
                                                             seconds)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2b_batch.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="the batch folder")
    ap.add_argument("--films", nargs="+", type=Path, default=list(FILMS), help="default: the two Netflix reference films")
    ap.add_argument("--dry-run", action="store_true", help="list the runs; start nothing")
    a = ap.parse_args(argv)
    out = Path(os.path.abspath(a.out))
    step2b = step2b_module()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")

    runs = []
    for pipeline, model, slice_form in step2b.STEP_2B_CONFIGURATIONS:
        label = step2b.config_label(model, slice_form)
        for film in a.films:
            film = Path(os.path.abspath(film))
            runs.append((label, out / label, film, [sys.executable, str(STEP2B), str(film), "--pipeline", pipeline,
                                                     "--model", model, "--slice-form", slice_form, "--out", str(out)]))
    if a.dry_run:
        for label, folder, film, cmd in runs:
            done = (folder / f"{film.stem}.run.json").is_file()
            print(f"{label:32} {film.name:55} {'measured, skipped' if done else 'would run'}")
        return 0

    out.mkdir(parents=True, exist_ok=True)
    batch_path = out / "batch.json"
    batch = json.loads(batch_path.read_text(encoding="utf-8")) if batch_path.is_file() else {"runs": []}
    failed = 0
    for i, (label, folder, film, cmd) in enumerate(runs, 1):
        if (folder / f"{film.stem}.run.json").is_file():
            print(f"[{i}/{len(runs)}] {label} {film.name}: measured already, skipped", flush=True)
            continue
        print(f"[{i}/{len(runs)}] {label} {film.name}: running", flush=True)
        result = run_one(label, folder, film, cmd, env)
        batch["runs"] = [r for r in batch["runs"] if (r["configuration"], r["film"]) != (label, film.name)] + [result]
        batch_path.write_text(json.dumps(batch, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        program = result["gpu"].get("program") or {}
        print(f"[{i}/{len(runs)}] {label} {film.name}: exit {result['exit_status']}, {result['seconds']} s; "
              f"program {program.get('seconds', '-')} s, GPU before {program.get('before_mib', '-')} MiB, "
              f"peak {program.get('peak_mib', '-')} MiB, program {program.get('program_mib', '-')} MiB", flush=True)
        failed += result["exit_status"] != 0
    print(f"done: {len(runs)} runs, {failed} failed; {batch_path}", flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
