#!/usr/bin/env python3
"""Run the anime-whisper VAD hypothesis sweep (one whisperjav subprocess/config).

Each config runs the real CLI in a fresh process (production path, isolated), so
this driver never imports the heavy ASR stack itself. Resumable: a config whose
result SRT already exists is skipped.

Usage:
    python -m tools.vad_hypothesis_suite.run_sweep [--only NAME[,NAME...]] [--out DIR]
    python -m tools.vad_hypothesis_suite.run_sweep --list
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from tools.vad_hypothesis_suite import configs as C

DEFAULT_OUT = os.path.join(
    r"D:\Git\WhisperJav_V1_Minami_Edition\test_media\1815acceptance\T3", "vad_sweep"
)


def _find_srt(out_dir: str) -> str | None:
    hits = glob.glob(os.path.join(out_dir, "*.whisperjav.srt"))
    return hits[0] if hits else None


def run_one(name, cfg_args, note, base_out, python_exe) -> dict:
    results_dir = os.path.join(base_out, "results")
    os.makedirs(results_dir, exist_ok=True)
    result_srt = os.path.join(results_dir, f"{name}.srt")
    rec = {"name": name, "note": note}
    if os.path.exists(result_srt):
        rec["status"] = "cached"
        return rec

    cfg_out = os.path.join(base_out, "runs", name)
    cfg_tmp = os.path.join(base_out, "runs", name, "_tmp")
    os.makedirs(cfg_out, exist_ok=True)
    os.makedirs(cfg_tmp, exist_ok=True)

    cmd = C.build_command(python_exe, C.REFERENCE_MEDIA, cfg_args, cfg_out, cfg_tmp)
    log_path = os.path.join(cfg_out, "run.log")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    rec["seconds"] = round(time.time() - t0, 1)
    rec["returncode"] = proc.returncode

    srt = _find_srt(cfg_out)
    if proc.returncode == 0 and srt:
        shutil.copy2(srt, result_srt)
        rec["status"] = "ok"
        rec["result_srt"] = result_srt
    else:
        rec["status"] = "FAILED"
        rec["log"] = log_path
    # Free scene/temp WAVs to save disk; keep the log.
    shutil.rmtree(cfg_tmp, ignore_errors=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT, help="sweep output base dir")
    ap.add_argument("--only", default=None, help="comma-separated config names to run")
    ap.add_argument("--list", action="store_true", help="list configs and exit")
    ap.add_argument("--python", default=sys.executable, help="python exe for the subprocess")
    args = ap.parse_args(argv)

    if args.list:
        for name, hyp, cfg_args, note in C.CONFIGS:
            print(f"{name:28s} [{hyp}]  {note}")
        return 0

    only = set(args.only.split(",")) if args.only else None
    os.makedirs(args.out, exist_ok=True)
    print(f"Sweep out: {args.out}")
    print(f"Media:     {C.REFERENCE_MEDIA}")
    print(f"Python:    {args.python}\n")

    summary = []
    for name, hyp, cfg_args, note in C.CONFIGS:
        if only and name not in only:
            continue
        print(f">>> {name}  [{hyp}] ...", flush=True)
        rec = run_one(name, cfg_args, note, args.out, args.python)
        summary.append(rec)
        tag = rec.get("status")
        extra = f" {rec.get('seconds','')}s rc={rec.get('returncode','')}" if tag != "cached" else ""
        print(f"    {tag}{extra}", flush=True)

    print("\n=== sweep summary ===")
    for rec in summary:
        print(f"  {rec['name']:28s} {rec['status']:8s} {rec.get('seconds','')}")
    failed = [r for r in summary if r["status"] == "FAILED"]
    if failed:
        print(f"\n{len(failed)} FAILED — see run.log in runs/<name>/")
        return 1
    print("\nNext: python -m tools.vad_hypothesis_suite.report")
    return 0


if __name__ == "__main__":
    sys.exit(main())
