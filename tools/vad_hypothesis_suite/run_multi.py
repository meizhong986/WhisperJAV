#!/usr/bin/env python3
"""Cross-clip VAD sweep: run each config on every reference benchmark clip.

One whisperjav subprocess per (benchmark, config), fresh process, resumable.
Output layout:  <out>/<benchmark>/results/<config>.srt

Usage:
    python -m tools.vad_hypothesis_suite.run_multi [--configs NAME[,NAME]] [--benchmarks NAME[,NAME]]
    python -m tools.vad_hypothesis_suite.run_multi --list
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time

from tools.vad_hypothesis_suite import configs as C
from tools.vad_hypothesis_suite import benchmarks as B

DEFAULT_OUT = os.path.join(B.ROOT, "test_media", "reference_benchmarks", "_sweep")
_CONFIG_BY_NAME = {name: (hyp, args, note) for name, hyp, args, note in C.CONFIGS}


def _find_srt(d):
    hits = glob.glob(os.path.join(d, "*.whisperjav.srt"))
    return hits[0] if hits else None


def run_one(bench, cfg_name, base_out, python_exe):
    hyp, cfg_args, note = _CONFIG_BY_NAME[cfg_name]
    results_dir = os.path.join(base_out, bench["name"], "results")
    os.makedirs(results_dir, exist_ok=True)
    result_srt = os.path.join(results_dir, f"{cfg_name}.srt")
    if os.path.exists(result_srt):
        return "cached", 0.0

    run_dir = os.path.join(base_out, bench["name"], "runs", cfg_name)
    tmp_dir = os.path.join(run_dir, "_tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    cmd = C.build_command(python_exe, bench["media"], cfg_args, run_dir, tmp_dir)
    t0 = time.time()
    with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n"); log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    secs = time.time() - t0
    srt = _find_srt(run_dir)
    if proc.returncode == 0 and srt:
        shutil.copy2(srt, result_srt)
        status = "ok"
    else:
        status = "FAILED"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return status, secs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--configs", default=",".join(B.VALIDATION),
                    help="config names (default: the VALIDATION subset)")
    ap.add_argument("--benchmarks", default=None, help="benchmark names (default: all)")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)

    benches = B.load_benchmarks()
    if args.benchmarks:
        want = set(args.benchmarks.split(","))
        benches = [b for b in benches if b["name"] in want]
    cfg_names = args.configs.split(",")

    if args.list:
        print("Benchmarks:")
        for b in benches:
            print(f"  {b['name']:16s} {b.get('duration','?')}s  gt={b.get('gt_entries','?')}")
        print("Configs:", ", ".join(cfg_names))
        return 0

    total = len(benches) * len(cfg_names)
    print(f"{len(benches)} benchmarks x {len(cfg_names)} configs = {total} runs\n")
    fails = []
    for b in benches:
        for cfg in cfg_names:
            print(f">>> [{b['name']}] {cfg} ...", flush=True)
            status, secs = run_one(b, cfg, args.out, args.python)
            print(f"    {status} {round(secs,1)}s", flush=True)
            if status == "FAILED":
                fails.append((b["name"], cfg))
    if fails:
        print("\nFAILED:", fails)
        return 1
    print("\nDone. Next: python -m tools.vad_hypothesis_suite.report_multi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
