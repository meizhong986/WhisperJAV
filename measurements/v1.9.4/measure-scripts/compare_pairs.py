"""Compare pairs of runs on the ground-truth lines BOTH matched (fair timing comparison), pooled over the clips.
Usage: python compare_pairs.py <base run> <candidate run> [<base run> <candidate run> ...]"""
import sys
from pathlib import Path

from whisperjav.bench.timing import compare_clip, load_srt, summarize
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

REF = Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix"
args = [Path(a) for a in sys.argv[1:]]
print("| Base -> candidate | Common lines | Lost | Start median (s) | End median (s) | End mean (s) |")
print("|---|---|---|---|---|---|")
for base, cand in zip(args[0::2], args[1::2]):
    eb, ec, common, lost = [], [], 0, 0
    for rp in sorted(REF.glob("*.srt")):
        cid = clip_id_of(rp)
        c = compare_clip(load_srt(rp, reference=True), load_srt(find_hyp(base, cid)), load_srt(find_hyp(cand, cid)))
        eb += c.base_errors
        ec += c.cand_errors
        common += c.n_common
        lost += c.n_lost
    b, k = summarize(eb).as_dict(), summarize(ec).as_dict()
    name = f"{base.parent.name} -> {cand.parent.name}"
    print(f"| {name} | {common} | {lost} | {b['start_abs_median']:.3f} -> {k['start_abs_median']:.3f} "
          f"| {b['end_abs_median']:.3f} -> {k['end_abs_median']:.3f} "
          f"| {b['end_signed_mean']:+.3f} -> {k['end_signed_mean']:+.3f} |")
