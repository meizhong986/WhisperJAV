"""Score each lever run against its generator's baseline run1, pooled over the clips.
Two views:
  A. on the ground-truth lines BOTH runs matched (fair timing comparison): start/end |median|, end signed mean
  B. each run on its own: lines written, ground-truth lines matched, end within 0.5 s, our median line length
Usage: python score_levers.py <baseline batch> <lever batch>"""
import json
import statistics
import sys
from pathlib import Path

from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.timing import compare_clip, load_srt, summarize
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

REF = Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix"
base_root, lever_root = Path(sys.argv[1]), Path(sys.argv[2])
refs = sorted(REF.glob("*.srt"))
BASE = {"q": base_root / "qwen3_whisperseg" / "run1", "a": base_root / "anime_whisperseg" / "run1"}


import re
import unicodedata

import editdistance


def _norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)  # sound/speaker notes
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def cer(run):
    """Character error rate of the whole text against the ground truth's, pooled over the clips."""
    dist = total = 0
    for rp in refs:
        ref = _norm("".join(r["text"] for r in load_srt(rp, reference=True)))
        hyp = _norm("".join(h["text"] for h in load_srt(find_hyp(run, clip_id_of(rp)))))
        dist += editdistance.eval(ref, hyp)
        total += len(ref)
    return dist / total if total else 0.0


def own_view(run):
    n_lines = matched = within = 0
    lengths = []
    for rp in refs:
        hyps = load_srt(find_hyp(run, clip_id_of(rp)))
        n_lines += len(hyps)
        lengths += [h["end"] - h["start"] for h in hyps]
        for ref, hyp in match_subtitles(load_srt(rp, reference=True), hyps)["matched"]:
            matched += 1
            within += abs(hyp["end"] - ref["end"]) <= 0.5
    return n_lines, matched, within, statistics.median(lengths) if lengths else 0.0


def seg_view(run):
    f = run / "framing.jsonl"
    if not f.exists():
        return None
    n = forced = 0
    for line in f.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") != "frame_call":
            continue
        segs = d["segments"]
        n += len(segs)
        forced += sum(1 for i in range(len(segs) - 1) if abs(segs[i + 1]["s"] - segs[i]["e"]) < 0.001)
    return n, forced


rows = []
for name, run in [("q_BASELINE", BASE["q"]), ("a_BASELINE", BASE["a"])] + \
        [(p.name, p / "run1") for p in sorted(lever_root.iterdir()) if (p / "run1" / "DONE").exists()]:
    base = BASE[name[0]]
    e_b, e_c, common, lost = [], [], 0, 0
    for rp in refs:
        cid = clip_id_of(rp)
        c = compare_clip(load_srt(rp, reference=True), load_srt(find_hyp(base, cid)), load_srt(find_hyp(run, cid)))
        e_b += c.base_errors
        e_c += c.cand_errors
        common += c.n_common
        lost += c.n_lost
    mb, mc = summarize(e_b).as_dict(), summarize(e_c).as_dict()
    n_lines, matched, within, med_len = own_view(run)
    sv = seg_view(run)
    rows.append((name, common, lost, mb, mc, n_lines, matched, within, med_len, sv, cer(run)))

print("A. On ground-truth lines both runs matched (baseline -> lever), seconds")
print(f"{'run':<12}{'common':>7}{'lost':>5}  {'start|med|':>15}  {'end|med|':>15}  {'end mean':>17}")
for name, common, lost, mb, mc, *_ in rows:
    print(f"{name:<12}{common:>7}{lost:>5}  {mb['start_abs_median']:6.3f}->{mc['start_abs_median']:<6.3f}  "
          f"{mb['end_abs_median']:6.3f}->{mc['end_abs_median']:<6.3f}  {mb['end_signed_mean']:+6.3f}->{mc['end_signed_mean']:<+7.3f}")
print("\nB. Each run on its own")
print(f"{'run':<12}{'lines':>6}{'matched':>8}{'end<=0.5s':>10}{'med len':>8}{'CER':>7}{'segments':>9}{'cut by limit':>13}")
for name, _, _, _, _, n_lines, matched, within, med_len, sv, c in rows:
    seg = f"{sv[0]:>9}{sv[1]:>13}" if sv else ""
    print(f"{name:<12}{n_lines:>6}{matched:>8}{within:>10}{med_len:>8.2f}{c:>7.3f}{seg}")
