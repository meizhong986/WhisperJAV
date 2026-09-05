#!/usr/bin/env python3
"""Score every sweep result SRT vs the ground truth and emit a ranked report.

Also folds in the two shipped baselines (Balanced=pass1, Aggressive=pass2) from
the T3 folder when present, so the sweep is judged against what ships today.

Usage:
    python -m tools.vad_hypothesis_suite.report [--out DIR] [--sort cer|del_rate|seg_ratio]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

from tools.vad_hypothesis_suite import configs as C
from tools.vad_hypothesis_suite.score import score_srt

DEFAULT_OUT = os.path.join(
    r"D:\Git\WhisperJav_V1_Minami_Edition\test_media\1815acceptance\T3", "vad_sweep"
)
_T3 = os.path.dirname(C.GROUND_TRUTH)
_BASELINES = {
    "z_ship_BALANCED_p1": os.path.join(_T3, "293sec-The.Naked.Director.S01E04.Scene4.ja.pass1.srt"),
    "z_ship_AGGRESSIVE_p2": os.path.join(_T3, "293sec-The.Naked.Director.S01E04.Scene4.ja.pass2.srt"),
}

COLS = ["cer", "sub_rate", "del_rate", "ins_rate", "char_recall",
        "n_subs", "seg_ratio", "mean_dur", "time_recall", "time_precision"]
LOWER_BETTER = {"cer", "sub_rate", "del_rate", "ins_rate"}
# Gap regions the T3 analysis flagged as weak (localized recall check).
GAP_BINS = {"1:20-1:40", "1:40-2:00", "2:40-3:00", "4:00-4:20", "4:20-4:40"}


def _collect(base_out):
    rows = []
    for name, srt in _BASELINES.items():
        if os.path.exists(srt):
            rows.append(score_srt(C.GROUND_TRUTH, srt, name))
    for srt in sorted(glob.glob(os.path.join(base_out, "results", "*.srt"))):
        rows.append(score_srt(C.GROUND_TRUTH, srt, os.path.splitext(os.path.basename(srt))[0]))
    return rows


def _fmt_table(rows, sort_key):
    rows = sorted(rows, key=lambda r: r.get(sort_key, 9e9),
                  reverse=sort_key not in LOWER_BETTER)
    best = {}
    for c in COLS:
        vals = [r[c] for r in rows]
        best[c] = (min if c in LOWER_BETTER else max)(vals) if vals else None
    header = f"{'config':26s} " + " ".join(f"{c:>11s}" for c in COLS)
    lines = [header, "-" * len(header)]
    for r in rows:
        cells = []
        for c in COLS:
            v = r[c]
            mark = "*" if v == best[c] else " "
            cells.append(f"{mark}{v:>10}")
        lines.append(f"{r['name']:26s} " + " ".join(cells))
    return "\n".join(lines)


def _gap_table(rows):
    lines = [f"\nWeak-region recall (GT gaps):  " + "  ".join(sorted(GAP_BINS))]
    for r in rows:
        by_bin = {b["bin"]: b["recall"] for b in r.get("region_recall", [])}
        cells = "  ".join(f"{by_bin.get(b, float('nan')):.2f}" for b in sorted(GAP_BINS))
        lines.append(f"  {r['name']:26s} {cells}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--sort", default="cer", choices=COLS)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows = _collect(args.out)
    if not rows:
        print("No result SRTs found. Run the sweep first (run_sweep).")
        return 1

    table = _fmt_table(rows, args.sort)
    gaps = _gap_table(sorted(rows, key=lambda r: r["name"]))
    print(f"Ranked by {args.sort} ('*' = best in column, lower better for CER/rates)\n")
    print(table)
    print(gaps)

    # Persist CSV + JSON.
    csv_path = os.path.join(args.out, "report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name"] + COLS)
        for r in sorted(rows, key=lambda r: r.get(args.sort, 9e9),
                        reverse=args.sort not in LOWER_BETTER):
            w.writerow([r["name"]] + [r[c] for c in COLS])
    with open(os.path.join(args.out, "report.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {csv_path} and report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
