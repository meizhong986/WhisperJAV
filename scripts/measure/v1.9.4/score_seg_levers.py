"""Score the step-3 runs (run_seg_levers.py) against their baselines, one block per combination: the baseline row,
then each changed run with the difference. Text first (character error rate and its parts, reference lines with no
text), then timing on the lines each run matched (score_segmenters.py measures), then a fair timing comparison on
the reference lines BOTH runs matched (compare_clip, as compare_pairs.py).
Usage: python score_seg_levers.py <levers folder> <baseline folder>"""
import sys
from pathlib import Path

from whisperjav.bench.timing import compare_clip, load_srt, summarize
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

from score_segmenters import REFS, measure

lev_root, base_root = Path(sys.argv[1]), Path(sys.argv[2])


def common(base, cand):
    eb, ec, n = [], [], 0
    for rp in REFS:
        cid = clip_id_of(rp)
        c = compare_clip(load_srt(rp, reference=True), load_srt(find_hyp(base, cid)), load_srt(find_hyp(cand, cid)))
        eb += c.base_errors
        ec += c.cand_errors
        n += c.n_common
    b, k = summarize(eb).as_dict(), summarize(ec).as_dict()
    return n, b["end_abs_median"], k["end_abs_median"], b["start_abs_median"], k["start_abs_median"]


runs = sorted(p.parent for p in lev_root.glob("*/run1/DONE"))
groups = {}
for r in runs:
    groups.setdefault(r.parent.name.split("__")[0], []).append(r)
for combo in sorted(groups):
    base = base_root / combo / "run1"
    b = measure(base)
    print(f"\n### {combo}")
    print("| Change | CER | Subst. | Missing | Extra | No-text lines | Our lines | Matched | Ends ≤ 0.5 s | End med (s) "
          "| End mean (s) | Start med (s) | Both matched: n, end med base → this, start med base → this |")
    print("|" + "---|" * 13)
    print(f"| baseline | {b['cer']:.3f} | {b['S']:.3f} | {b['D']:.3f} | {b['I']:.3f} | {b['gone']} | {b['lines']} "
          f"| {b['matched']} | {b['within']} | {b['en']:.3f} | {b['ens']:+.3f} | {b['st']:.3f} | |")
    for r in groups[combo]:
        m = measure(r)
        n, eb, ec, sb, sc = common(base, r)
        print(f"| {r.parent.name.split('__')[1]} | {m['cer']:.3f} ({m['cer'] - b['cer']:+.3f}) | {m['S']:.3f} "
              f"| {m['D']:.3f} | {m['I']:.3f} | {m['gone']} ({m['gone'] - b['gone']:+d}) | {m['lines']} "
              f"| {m['matched']} ({m['matched'] - b['matched']:+d}) | {m['within']} ({m['within'] - b['within']:+d}) "
              f"| {m['en']:.3f} | {m['ens']:+.3f} | {m['st']:.3f} | {n}: {eb:.3f} → {ec:.3f}, {sb:.3f} → {sc:.3f} |",
              flush=True)
