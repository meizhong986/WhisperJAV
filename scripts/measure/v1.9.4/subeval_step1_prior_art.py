"""Step 1 (subeval): how today's pairing code behaves on the 163 REQ2 runs. Read-only.
Counts, pooled over all runs x 7 clips:
 A. matcher (whisperjav.bench.matcher.match_subtitles): reference lines left unpaired although one of our lines that
    overlaps them in time (>= 0.1 s) contains their whole normalised text; split by cause (similarity < 0.2 / that line
    already taken by an earlier reference line).
 B. our lines the matcher calls "hallucinated" (unpaired) although they contain the whole text of an overlapping
    reference line, or at least half of its characters in order.
 C. pairs accepted by the matcher in which our line holds less than half of the reference line's characters in order.
 D. character alignment (rapidfuzz editops, as score_segmenters.py): "lines with no text" under the forward alignment
    and under the mirror alignment (both strings reversed) - how often and by how much they differ.
 E. for characters the plain alignment calls equal: time gap between the reference line and our line holding them
    (0 when the two lines overlap) - how far apart in time the alignment pairs characters.
"""
import difflib
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "D:/Git/WhisperJav_V1_Minami_Edition/docs/measurements/v1.9.4/measure-scripts")
from rapidfuzz.distance import Levenshtein

from score_segmenters import REFS, norm
from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.metrics import normalize_text
from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

R = Path("D:/WJ194_measure/req2_ten_firered")
runs = sorted(p.parent for p in R.glob("baseline/*/run*/DONE")) + sorted(p.parent for p in R.glob("levers/*/run1/DONE"))
REF = {clip_id_of(rp): load_srt(rp, reference=True) for rp in REFS}


def overlap(a, b):
    return min(a["end"], b["end"]) - max(a["start"], b["start"])


def lcs(a, b):
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())


def gap(a, b):
    return max(0.0, max(a["start"], b["start"]) - min(a["end"], b["end"]))


def gone(refs, hyps, mirror):
    r = norm("".join(x["text"] for x in refs))
    h = norm("".join(x["text"] for x in hyps))
    if mirror:
        dl = {len(r) - 1 - o.src_pos for o in Levenshtein.editops(r[::-1], h[::-1]) if o.tag == "delete"}
    else:
        dl = {o.src_pos for o in Levenshtein.editops(r, h) if o.tag == "delete"}
    out, pos = set(), 0
    for i, x in enumerate(refs):
        n = len(norm(x["text"]))
        if n and all(k in dl for k in range(pos, pos + n)):
            out.add(i)
        pos += n
    return out


def char_owner(lines):
    own = []
    for i, x in enumerate(lines):
        own += [i] * len(norm(x["text"]))
    return own


C = Counter()
gone_diff = []
gaps = []
examples = {}
for run in runs:
    for cid, refs in REF.items():
        hyps = load_srt(find_hyp(run, cid))
        res = match_subtitles(refs, hyps)
        paired_ref = {id(g) for g, _ in res["matched"]}
        used = {id(t) for _, t in res["matched"]}
        owner_of = {id(t): g for g, t in res["matched"]}
        C["ref lines"] += len(refs)
        C["our lines"] += len(hyps)
        C["pairs"] += len(res["matched"])
        # A
        for g in refs:
            if id(g) in paired_ref:
                continue
            gt = normalize_text(g["text"])
            if not gt:
                continue
            for t in hyps:
                if overlap(g, t) >= 0.1 and gt in normalize_text(t["text"]):
                    sim = difflib.SequenceMatcher(None, gt, normalize_text(t["text"])).ratio()
                    if id(t) in used:
                        C["A unpaired, text inside an overlapping line already taken"] += 1
                    elif sim < 0.2:
                        C["A unpaired, text inside an overlapping free line, similarity < 0.2"] += 1
                        examples.setdefault("A", (run.parent.name, cid, g["text"], t["text"], round(sim, 3)))
                    else:
                        C["A unpaired, other"] += 1
                    break
        # B
        for t in hyps:
            if id(t) in used:
                continue
            tt = normalize_text(t["text"])
            C["B our lines unpaired ('hallucinated')"] += 1
            for g in refs:
                gt = normalize_text(g["text"])
                if gt and overlap(g, t) >= 0.1 and lcs(gt, tt) >= 0.5 * len(gt):
                    C["B   ...of which holding >= half of an overlapping reference line"] += 1
                    examples.setdefault("B", (run.parent.name, cid, g["text"], t["text"]))
                    break
        # C
        for g, t in res["matched"]:
            gt, tt = normalize_text(g["text"]), normalize_text(t["text"])
            if gt and lcs(gt, tt) < 0.5 * len(gt):
                C["C pairs where our line holds < half of the reference line"] += 1
                examples.setdefault("C", (run.parent.name, cid, g["text"], t["text"]))
        # D
        f, m = gone(refs, hyps, False), gone(refs, hyps, True)
        C["D no-text lines, forward"] += len(f)
        C["D no-text lines, mirror"] += len(m)
        C["D lines no-text in one alignment only"] += len(f ^ m)
        gone_diff.append(len(f) - len(m))
        # E
        r = norm("".join(x["text"] for x in refs))
        h = norm("".join(x["text"] for x in hyps))
        ro, ho = char_owner(refs), char_owner(hyps)
        for op in Levenshtein.opcodes(r, h):
            if op.tag == "equal":
                for k in range(op.src_end - op.src_start):
                    gaps.append(gap(refs[ro[op.src_start + k]], hyps[ho[op.dest_start + k]]))

for k, v in C.items():
    print(f"{k}: {v}")
print("D per run-clip difference forward-mirror: mean abs", round(statistics.mean(abs(x) for x in gone_diff), 2),
      "max", max(abs(x) for x in gone_diff), "of", len(gone_diff), "run-clips;",
      sum(1 for x in gone_diff if x), "differ")
gaps.sort()
n = len(gaps)
print("E equal characters:", n, "| gap 0:", f"{sum(g == 0 for g in gaps) / n:.1%}",
      "| <=0.5 s:", f"{sum(g <= 0.5 for g in gaps) / n:.1%}", "| <=1 s:", f"{sum(g <= 1 for g in gaps) / n:.1%}",
      "| <=2 s:", f"{sum(g <= 2 for g in gaps) / n:.2%}", "| >5 s:", f"{sum(g > 5 for g in gaps) / n:.2%}",
      "| >10 s:", f"{sum(g > 10 for g in gaps) / n:.2%}", "| max", round(gaps[-1], 1))
for k, v in examples.items():
    print("example", k, v)
