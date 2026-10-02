"""Score reference runs against the ground-truth subtitles, and compare repeated runs.

1. ChronosJAV batch (from run_chronosjav_reference_runs.py): for each pair, run1 and run2 are scored against the
   ground truth with whisperjav.bench.timing, on the ground-truth lines both runs matched: start and end errors of
   each run, and the lines matched by run1 but not run2.
2. Pipeline batch (from run_pipeline_reference_runs.py): for each config, run1 and run2 SRT files are compared line
   by line (identical text and times or not).

Usage (from the repository root, with a python that has whisperjav importable):
    python scripts/measure/score_reference_runs.py --ground-truth <folder of .srt> \\
        [--chronosjav <batch folder>] [--pipelines <batch folder>] [--json <out.json>]
"""
import argparse
import json
from pathlib import Path

from whisperjav.bench.timing import compare_clip, load_srt, summarize
from whisperjav.bench.timing_cli import clip_id_of, find_hyp


def srt_lines(p):
    return [(round(s["start"], 3), round(s["end"], 3), s["text"]) for s in load_srt(p)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ground-truth", required=True, type=Path)
    ap.add_argument("--chronosjav", type=Path)
    ap.add_argument("--pipelines", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    refs = sorted(a.ground_truth.glob("*.srt"))
    result = {"chronosjav": {}, "pipelines": {}}

    if a.chronosjav:
        print("ChronosJAV pairs, pooled over the clips, run1 vs run2 on common lines (seconds)")
        print(f"{'pair':<24}{'common':>7}{'lost':>5} | {'start|med| r1/r2':>18} {'end|med| r1/r2':>18} "
              f"{'end mean r1/r2':>18}")
        for pair_dir in sorted(p for p in a.chronosjav.iterdir() if p.is_dir()):
            r1, r2 = pair_dir / "run1", pair_dir / "run2"
            if not ((r1 / "DONE").exists() and (r2 / "DONE").exists()):
                continue
            e1, e2, common, lost, n_ref = [], [], 0, 0, 0
            for ref_path in refs:
                cid = clip_id_of(ref_path)
                c = compare_clip(load_srt(ref_path, reference=True), load_srt(find_hyp(r1, cid)),
                                 load_srt(find_hyp(r2, cid)))
                e1 += c.base_errors
                e2 += c.cand_errors
                common += c.n_common
                lost += c.n_lost
                n_ref += c.n_ref
            m1, m2 = summarize(e1).as_dict(), summarize(e2).as_dict()
            result["chronosjav"][pair_dir.name] = {"n_ref": n_ref, "n_common": common, "n_lost_run1_to_run2": lost,
                                                   "run1": m1, "run2": m2}
            print(f"{pair_dir.name:<24}{common:>7}{lost:>5} | {m1['start_abs_median']:>8.3f}/{m2['start_abs_median']:<8.3f} "
                  f"{m1['end_abs_median']:>8.3f}/{m2['end_abs_median']:<8.3f} "
                  f"{m1['end_signed_mean']:>+8.3f}/{m2['end_signed_mean']:<+8.3f}")

    if a.pipelines:
        print("\nPipelines, run1 vs run2 per config: files, lines, lines not identical (text+times)")
        for cfg in sorted(p for p in a.pipelines.iterdir() if p.is_dir()):
            r1, r2 = cfg / "run1", cfg / "run2"
            if not ((r1 / "DONE").exists() and (r2 / "DONE").exists()):
                continue
            files = sorted(p.name for p in r1.glob("*.srt"))
            n_lines = n_diff = 0
            missing = [f for f in files if not (r2 / f).exists()]
            for f in files:
                x, y = srt_lines(r1 / f), srt_lines(r2 / f) if (r2 / f).exists() else []
                n_lines += len(x)
                n_diff += sum(1 for p, q in zip(x, y) if p != q) + abs(len(x) - len(y))
            result["pipelines"][cfg.name] = {"files": len(files), "lines": n_lines, "lines_not_identical": n_diff,
                                             "missing_in_run2": missing}
            print(f"  {cfg.name:<22} files {len(files):>3}  lines {n_lines:>5}  not identical {n_diff:>5}"
                  + (f"  missing in run2: {missing}" if missing else ""))

    if a.json:
        a.json.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print("\nwritten", a.json)


if __name__ == "__main__":
    main()
