#!/usr/bin/env python3
"""Aggregate the cross-clip sweep: per-config mean across benchmarks + per-clip.

A config's recommendation strength = how it does AVERAGED over all clips (kills
single-clip overfitting). Also prints a per-clip breakdown of the headline
metrics so clip-specific effects stay visible.

Usage:
    python -m tools.vad_hypothesis_suite.report_multi [--sort time_recall|cer|del_rate]
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys

from tools.vad_hypothesis_suite import benchmarks as B
from tools.vad_hypothesis_suite.score import score_srt

DEFAULT_OUT = os.path.join(B.ROOT, "test_media", "reference_benchmarks", "_sweep")
AGG = ["cer", "sub_rate", "del_rate", "ins_rate", "char_recall", "seg_ratio",
       "mean_dur", "time_recall", "time_precision"]
LOWER_BETTER = {"cer", "sub_rate", "del_rate", "ins_rate"}


def _collect(out_dir):
    benches = B.load_benchmarks()
    # config -> list of per-benchmark metric dicts
    per_config = {}
    for b in benches:
        rdir = os.path.join(out_dir, b["name"], "results")
        if not os.path.isdir(rdir):
            continue
        for fn in os.listdir(rdir):
            if not fn.endswith(".srt"):
                continue
            cfg = fn[:-4]
            m = score_srt(b["gt"], os.path.join(rdir, fn), cfg)
            per_config.setdefault(cfg, {})[b["name"]] = m
    return benches, per_config


def _aggregate(per_config):
    rows = []
    for cfg, by_bench in per_config.items():
        agg = {"name": cfg, "n_clips": len(by_bench)}
        for c in AGG:
            vals = [m[c] for m in by_bench.values()]
            agg[c] = round(st.mean(vals), 4)
        rows.append(agg)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sort", default="time_recall", choices=AGG)
    args = ap.parse_args(argv)

    benches, per_config = _collect(args.out)
    if not per_config:
        print("No results. Run: python -m tools.vad_hypothesis_suite.run_multi")
        return 1
    rows = _aggregate(per_config)
    rows.sort(key=lambda r: r.get(args.sort, 9e9), reverse=args.sort not in LOWER_BETTER)

    best = {c: (min if c in LOWER_BETTER else max)(r[c] for r in rows) for c in AGG}
    print(f"AGGREGATE across clips (mean), ranked by {args.sort}  ('*' = best)\n")
    hdr = f"{'config':26s} {'clips':>5s} " + " ".join(f"{c:>11s}" for c in AGG)
    print(hdr); print("-" * len(hdr))
    for r in rows:
        cells = " ".join(f"{'*' if r[c]==best[c] else ' '}{r[c]:>10}" for c in AGG)
        print(f"{r['name']:26s} {r['n_clips']:>5d} {cells}")

    # Per-clip breakdown of the two headline metrics.
    for metric in ("time_recall", "cer"):
        print(f"\nper-clip {metric}:")
        bench_names = [b["name"] for b in benches]
        print(f"  {'config':26s} " + " ".join(f"{n[:12]:>13s}" for n in bench_names))
        for r in sorted(rows, key=lambda r: r["name"]):
            by = per_config[r["name"]]
            cells = " ".join(f"{by[n][metric]:>13}" if n in by else f"{'-':>13}" for n in bench_names)
            print(f"  {r['name']:26s} " + cells)

    csv_path = os.path.join(args.out, "report_multi.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["config", "n_clips"] + AGG)
        for r in rows:
            w.writerow([r["name"], r["n_clips"]] + [r[c] for c in AGG])
    print(f"\nWrote {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
