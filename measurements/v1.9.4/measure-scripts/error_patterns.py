"""Patterns in character errors and timing errors (owner's questions, 2026-10-05).

For each run, pooled over the 7 reference clips:
  A. Character errors. Each clip's whole ground-truth text is aligned with our whole text (character edit
     operations, after the same normalisation as cer_trace.py). Every missing ground-truth character is traced to
     its ground-truth line: was the line matched at all; where in the line (first quarter / middle half / last
     quarter); and, from its position, an estimated time, and how far that is from the nearest edge of one of our
     lines. The same is computed for ALL ground-truth characters as the reference distribution.
     The most frequent missing runs (consecutive missing characters) are listed.
  B. Timing by line length: |start error| and |end error| medians by ground-truth line duration, and by whether
     our line was cut by the length limit (its end equals the next line's start, the mark of a forced split).
  C. Per matched line: Spearman correlation between the end error (signed and absolute) and that line's missing
     and wrong characters (rate per ground-truth character).
Usage: python error_patterns.py <run> [<run> ...]"""
import collections
import re
import statistics
import sys
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

REF = Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix"


def norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def med(v):
    return statistics.median(v) if v else float("nan")


for arg in sys.argv[1:]:
    run = Path(arg)
    name = run.parent.name if run.name.startswith("run") else run.name
    pos_all, pos_missing = collections.Counter(), collections.Counter()
    dist_all, dist_missing = [], []
    missing_in_unmatched = missing_total = 0
    runs_missing = collections.Counter()
    t_rows = []      # (gt duration, our duration, cut by limit, |start err|, |end err|)
    c_rows = []      # (signed end err, |end err|, missing rate, wrong rate)
    for rp in sorted(REF.glob("*.srt")):
        refs = sorted(load_srt(rp, reference=True), key=lambda r: r["start"])
        hyps = sorted(load_srt(find_hyp(run, clip_id_of(rp))), key=lambda h: h["start"])
        edges = sorted({round(h["start"], 3) for h in hyps} | {round(h["end"], 3) for h in hyps})
        matched = match_subtitles(refs, hyps)["matched"]
        matched_ids = {id(r) for r, _ in matched}
        # ground-truth characters with their line, position and estimated time
        g_chars, g_line, g_pos, g_time = [], [], [], []
        for li, r in enumerate(refs):
            t = norm(r["text"])
            for k, ch in enumerate(t):
                g_chars.append(ch)
                g_line.append(li)
                frac = (k + 0.5) / len(t)
                g_pos.append("first quarter" if frac < 0.25 else ("last quarter" if frac > 0.75 else "middle half"))
                g_time.append(r["start"] + frac * (r["end"] - r["start"]))
        h_text = "".join(norm(h["text"]) for h in hyps)
        g_text = "".join(g_chars)
        missing_idx, wrong_idx = set(), set()
        prev = None
        cur_run = ""
        for op in Levenshtein.editops(g_text, h_text):
            if op.tag == "delete":
                missing_idx.add(op.src_pos)
                if prev is not None and op.src_pos == prev + 1:
                    cur_run += g_text[op.src_pos]
                else:
                    if cur_run:
                        runs_missing[cur_run] += 1
                    cur_run = g_text[op.src_pos]
                prev = op.src_pos
            elif op.tag == "replace":
                wrong_idx.add(op.src_pos)
        if cur_run:
            runs_missing[cur_run] += 1

        def nearest_edge(t):
            return min((abs(t - e) for e in edges), default=float("nan"))

        for i in range(len(g_chars)):
            pos_all[g_pos[i]] += 1
            dist_all.append(nearest_edge(g_time[i]))
            if i in missing_idx:
                missing_total += 1
                pos_missing[g_pos[i]] += 1
                dist_missing.append(nearest_edge(g_time[i]))
                if id(refs[g_line[i]]) not in matched_ids:
                    missing_in_unmatched += 1
        # per-line rates for C
        per_line_missing = collections.Counter(g_line[i] for i in missing_idx)
        per_line_wrong = collections.Counter(g_line[i] for i in wrong_idx)
        line_index = {id(r): li for li, r in enumerate(refs)}
        hyp_starts = {round(h["start"], 3) for h in hyps}
        for r, h in matched:
            li = line_index[id(r)]
            n = max(1, len(norm(r["text"])))
            se, ee = h["start"] - r["start"], h["end"] - r["end"]
            cut = round(h["end"], 3) in hyp_starts
            t_rows.append((r["end"] - r["start"], h["end"] - h["start"], cut, abs(se), abs(ee)))
            c_rows.append((ee, abs(ee), per_line_missing[li] / n, per_line_wrong[li] / n))

    print(f"\n=== {name}")
    print(f"A. Missing characters: {missing_total}; in ground-truth lines we did not match at all: "
          f"{missing_in_unmatched} ({missing_in_unmatched / max(1, missing_total):.0%})")
    for p in ("first quarter", "middle half", "last quarter"):
        print(f"   {p:13}: {pos_missing[p] / max(1, missing_total):5.1%} of missing   "
              f"(share of all characters: {pos_all[p] / max(1, sum(pos_all.values())):5.1%})")
    print(f"   distance to the nearest edge of one of our lines, median: missing chars {med(dist_missing):.2f} s, "
          f"all chars {med(dist_all):.2f} s;  within 0.3 s: missing "
          f"{sum(d <= 0.3 for d in dist_missing) / max(1, len(dist_missing)):.0%}, all "
          f"{sum(d <= 0.3 for d in dist_all) / max(1, len(dist_all)):.0%}")
    print("   most frequent missing runs: " + ", ".join(f"{k}×{v}" for k, v in runs_missing.most_common(12)))
    print("B. Timing by ground-truth line duration (median |start| / |end| error, s):")
    for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
        sel = [r for r in t_rows if lo <= r[0] < hi]
        print(f"   {lo}-{hi if hi < 99 else '+'} s: n={len(sel):3}  start {med([r[3] for r in sel]):.2f}  "
              f"end {med([r[4] for r in sel]):.2f}")
    for cut in (True, False):
        sel = [r for r in t_rows if r[2] == cut]
        print(f"   our line {'cut by the length limit' if cut else 'ended at a pause      '}: n={len(sel):3}  "
              f"start {med([r[3] for r in sel]):.2f}  end {med([r[4] for r in sel]):.2f}  "
              f"our line length {med([r[1] for r in sel]):.2f} s")
    print("C. Per matched line, Spearman correlation:")
    print(f"   |end error| vs missing rate {spearman([r[1] for r in c_rows], [r[2] for r in c_rows]):+.2f}, "
          f"vs wrong rate {spearman([r[1] for r in c_rows], [r[3] for r in c_rows]):+.2f};  "
          f"signed end error vs missing {spearman([r[0] for r in c_rows], [r[2] for r in c_rows]):+.2f}")
    early = [r for r in c_rows if r[0] < -0.5]
    ok = [r for r in c_rows if abs(r[0]) <= 0.5]
    late = [r for r in c_rows if r[0] > 0.5]
    for label, sel in (("ends > 0.5 s early", early), ("ends within 0.5 s", ok), ("ends > 0.5 s late", late)):
        print(f"   {label:18}: n={len(sel):3}  mean missing rate {statistics.mean([r[2] for r in sel]) if sel else 0:.3f}"
              f"  mean wrong rate {statistics.mean([r[3] for r in sel]) if sel else 0:.3f}")
