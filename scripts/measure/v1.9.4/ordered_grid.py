"""REQ2 ordering recovery (owner, 2026-10-08). For TEN and FireRedVAD, per model: a grid of threshold x "the other
settings of" each sensitivity (rows), with how much audio was let in (intake_scores.py) next to the text and timing
outcomes (strategy_scores.py, subeval). Then the ordering check of a proposed set.

Cell = one run whose settings are a sensitivity's settings with only the threshold changed.
Columns per cell: audio in (share of clip) | speech in | correct characters | precision | missed lines |
hallucination + misheard lines | CER | start / end median |error| over the run's own measured edges (s) and their
count. Own-edge medians are over different line sets per run; the paired comparison for the final proposals is in
the proposal table (subeval compare against today's setting).

Usage: python ordered_grid.py <levers folder> <baseline folder> [--propose proposals.json]
  proposals.json: {"<gen>_<seg>": {"conservative": "<run>", "balanced": ..., "aggressive": ...}}, where <run> is
  "baseline" (today's setting of that sensitivity) or a full run name "<combo>__<lever>" from run_seg_levers.py
  (it may belong to another sensitivity's settings). Each is compared with today's setting of the sensitivity it is
  proposed for. The check fails when audio in is not ordered
  conservative <= balanced <= aggressive, or when the thresholds are not ordered conservative >= balanced >= aggressive.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intake_scores import intake, reference_spans, threshold_of  # noqa: E402
from run_seg_levers import SEG, lever_table  # noqa: E402
from strategy_scores import REF, pooled, run_counts  # noqa: E402
from subeval.cli import pair_folder, score_files  # noqa: E402
from subeval.compare import compare  # noqa: E402
from subeval.evaluate import summarize  # noqa: E402

SENS = ("conservative", "balanced", "aggressive")


def run_dir(lev_root, base_root, combo, lever):
    return (base_root / combo / "run1") if lever == "baseline" else (lev_root / f"{combo}__{lever}" / "run1")


def cell(path, refs):
    per, res = run_counts(path)
    s = pooled(per)
    sm = summarize(res)
    s.update(intake(path, refs))
    s.update({k: sm[k] for k in ("start_abs_median", "end_abs_median", "start_measured", "end_measured",
                                 "start_within_share", "end_within_share")})
    return s, res


def fmt(s):
    return (f"{s['audio_share']:.0%} / {s['speech_share']:.1%} | {s['correct']} | {s['precision']:.1%} | {s['missed']} "
            f"| {s['halluc'] + s['misheard']} | {s['cer']:.3f} | {s['start_abs_median']:.3f} / "
            f"{s['end_abs_median']:.3f} ({s['end_measured']})")


def main():
    lev_root, base_root = Path(sys.argv[1]), Path(sys.argv[2])
    refs = reference_spans()
    table = lever_table()
    cache = {}

    def get(combo, lever):
        if (combo, lever) not in cache:
            cache[(combo, lever)] = cell(run_dir(lev_root, base_root, combo, lever), refs)
        return cache[(combo, lever)]

    for gen in ("qwen3", "anime"):
        for seg in ("ten", "firered-vad"):
            print(f"\n### {gen}, {seg}\n")
            print("| Other settings of | Threshold | Audio in / speech in | Correct | Precision | Missed | Halluc.+mish. "
                  "| CER | Start / end median (n ends) |")
            print("|---|---|---|---|---|---|---|---|---|")
            for sens in SENS:
                combo = f"{gen}_{seg}_{sens}"
                levers = ["baseline"] + sorted(p.parent.parent.name.split("__")[1]
                                               for p in lev_root.glob(f"{combo}__*/run1/DONE"))
                pts = sorted((threshold_of(combo, lv, table), lv) for lv in levers
                             if threshold_of(combo, lv, table) is not None)
                for th, lv in pts:
                    s, _ = get(combo, lv)
                    mark = " (today)" if lv == "baseline" else ""
                    print(f"| {sens} | {th:.2f}{mark} | {fmt(s)} |", flush=True)

    print("\n### Same threshold, different other settings (paired, clip by clip)\n")
    print("Each row compares two runs with the same threshold: A = one sensitivity's other settings, B = another's. "
          "Δ = B − A. Clips: better/worse for B.\n")
    print("| Model, segmenter | Threshold | A → B | Audio in A / B | Δ correct (clips) | Δ precision (clips) "
          "| Δ missed | Δ halluc.+mish. | Starts both (clips) | Ends both (clips) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    from strategy_scores import clips as clip_count, prec  # noqa: E402
    for gen in ("qwen3", "anime"):
        for seg in ("ten", "firered-vad"):
            by_thr = {}
            for sens in SENS:
                combo = f"{gen}_{seg}_{sens}"
                levers = ["baseline"] + [p.parent.parent.name.split("__")[1]
                                         for p in lev_root.glob(f"{combo}__*/run1/DONE")]
                for lv in levers:
                    th = threshold_of(combo, lv, table)
                    if th is not None:
                        by_thr.setdefault(round(th, 2), {})[sens] = (combo, lv)
            for th in sorted(by_thr):
                have = by_thr[th]
                for a, b in (("conservative", "aggressive"), ("balanced", "aggressive"),
                             ("conservative", "balanced")):
                    if a not in have or b not in have:
                        continue
                    (sa, ra), (sb, rb) = get(*have[a]), get(*have[b])
                    pa, _ = run_counts(run_dir(lev_root, base_root, *have[a]))
                    pb, _ = run_counts(run_dir(lev_root, base_root, *have[b]))
                    cc = clip_count(pa, pb, lambda c: c["correct"])
                    cp = clip_count(pa, pb, prec)
                    c = compare(ra, rb)
                    print(f"| {gen}, {seg} | {th:.2f} | {a} → {b} | {sa['audio_share']:.1%} / {sb['audio_share']:.1%} "
                          f"| {sb['correct'] - sa['correct']:+d} ({cc[0]}/{cc[1]}) "
                          f"| {100 * (sb['precision'] - sa['precision']):+.1f} pt ({cp[0]}/{cp[1]}) "
                          f"| {sb['missed'] - sa['missed']:+d} "
                          f"| {(sb['halluc'] + sb['misheard']) - (sa['halluc'] + sa['misheard']):+d} "
                          f"| {c['start_med_base']:.3f} → {c['start_med_cand']:.3f} ({c['start_clips_better']}/"
                          f"{c['start_clips_worse']}) | {c['end_med_base']:.3f} → {c['end_med_cand']:.3f} "
                          f"({c['end_clips_better']}/{c['end_clips_worse']}) |", flush=True)

    if "--propose" in sys.argv:
        props = json.loads(Path(sys.argv[sys.argv.index("--propose") + 1]).read_text(encoding="utf-8"))
        print("\n### Proposed ordered sets: ordering check and comparison with today\n")
        print("| Model, segmenter | Sensitivity | Run | Threshold | Audio in | Δ correct (clips +/−) "
              "| Δ precision | Δ missed | Δ halluc.+mish. | Starts both (clips +/−) | Ends both (clips +/−) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        bad = 0
        for gs, sets in props.items():
            gen, seg = gs.split(" ")[0].split("_", 1)  # "<gen>_<seg> <variant label>"
            audio, thr = [], []
            for sens in SENS:
                today = f"{gen}_{seg}_{sens}"
                lv = sets[sens]  # "baseline" (today's setting) or a run name "<combo>__<lever>"
                combo, lv = (today, "baseline") if lv == "baseline" else lv.split("__")
                s, res = get(combo, lv)
                b, rb = get(today, "baseline")
                pp = {} if lv == "baseline" else table[f"{combo}__{lv}"][3]
                th = pp.get("threshold", SEG[seg][combo.rsplit("_", 1)[1]][0] if seg in SEG else None)
                lv = sets[sens]
                audio.append(s["audio_in"])
                thr.append(th)
                c = compare(rb, res)
                print(f"| {gs} | {sens} | {lv} | {th} | {s['audio_share']:.1%} "
                      f"| {s['correct'] - b['correct']:+d} | {100 * (s['precision'] - b['precision']):+.1f} pt "
                      f"| {s['missed'] - b['missed']:+d} | {(s['halluc'] + s['misheard']) - (b['halluc'] + b['misheard']):+d} "
                      f"| {c['start_med_base']:.3f} → {c['start_med_cand']:.3f} ({c['start_clips_better']}/"
                      f"{c['start_clips_worse']}) | {c['end_med_base']:.3f} → {c['end_med_cand']:.3f} "
                      f"({c['end_clips_better']}/{c['end_clips_worse']}) |", flush=True)
            ok_a = audio[0] <= audio[1] <= audio[2]
            ok_t = None not in thr and thr[0] >= thr[1] >= thr[2]
            if not (ok_a and ok_t):
                bad += 1
            print(f"| {gs} | ordering | audio in {'ordered' if ok_a else '**NOT ordered**'}; threshold "
                  f"{'ordered' if ok_t else '**NOT ordered**'} | | | | | | | | |")
        print(f"\nOrdering check: {bad} set(s) not ordered.")


if __name__ == "__main__":
    main()
