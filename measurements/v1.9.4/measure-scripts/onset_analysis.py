"""Why is the first character of a line wrong so often? (owner, 2026-10-05: character accuracy first; step 1,
analysis only, no new runs.)

For each run (SRTs + framing.jsonl from the segmenter recorder), over the matched line pairs (ground-truth line,
our line) of the 7 reference clips:
  1. Line-level alignment: each pair aligned on its own (not the whole clip), so a character cannot be counted
     against a neighbouring line. Rates of wrong / missing for the first character, the last character, and all
     characters of the ground-truth line.
  2. What the first-character substitutions are; rates for lines whose ground truth starts like an interjection
     (a vowel kana, ん, っ, ー, small kana) vs others.
  3. Lines that start at a forced split (our line starts exactly where our previous line ended) vs after a pause.
  4. The audio window the ASR model received for the line, rebuilt from the recording: scene start + frame start
     (the frame is the padded group the model transcribed; for anime-whisper aggressive the displayed start can
     be later, at the "speech start"). Window start minus ground-truth start, in bins, against the first
     character's error rate. Window end minus ground-truth end against the last character's error rate.
  6. Lines whose window starts more than 0.3 s after the ground-truth start: how many start at a forced split,
     and whether the ground truth's first two characters appear at the end of our previous line (the words
     went into the previous line) or nowhere near (the onset was not heard).
Usage: python onset_analysis.py <run> [<run> ...]"""
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

from rapidfuzz.distance import Levenshtein

from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

REF = Path(__file__).resolve().parents[4] / "test_media" / "Ground_Truths" / "Netflix"
INTERJECTION_START = set("あいうえおアイウエオんンっッーぁぃぅぇぉゃゅょ")


def norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def char_states(g, h):
    st = ["correct"] * len(g)
    sub = {}
    for op in Levenshtein.editops(g, h):
        if op.tag == "delete":
            st[op.src_pos] = "missing"
        elif op.tag == "replace":
            st[op.src_pos] = "wrong"
            sub[op.src_pos] = h[op.dest_pos]
    return st, sub


def windows_by_clip(rec_path):
    """clip id -> list of (abs window start, abs window end, abs display start) for every frame."""
    out = collections.defaultdict(list)
    queue = []          # scenes waiting for their frame_call, in order
    for line in rec_path.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "scenes":
            for path, s0, s1, dur in d["scenes"]:
                clip = re.sub(r"_scene_\d+\.wav$", "", Path(path).name)
                queue.append((clip, s0, dur))
        elif d.get("kind") == "frame_call" and queue:
            # match this call to the next queued scene of the same duration
            for qi, (clip, s0, dur) in enumerate(queue):
                if abs(dur - d["dur"]) < 0.01:
                    queue.pop(qi)
                    break
            else:
                continue
            starts = d.get("speech_starts") or [None] * len(d["frames"])
            for (fs, fe), ss in zip(d["frames"], starts):
                out[clip].append((s0 + fs, s0 + fe, s0 + (ss if ss is not None else fs)))
    return out


def rate(c):
    return f"{c[1] / c[0]:.0%} of {c[0]}" if c[0] else "-"


def bins(v, edges, labels):
    for e, lab in zip(edges, labels):
        if v < e:
            return lab
    return labels[-1]


for arg in sys.argv[1:]:
    run = Path(arg)
    name = run.parent.name if run.name.startswith("run") else run.name
    wins = windows_by_clip(run / "framing.jsonl") if (run / "framing.jsonl").exists() else {}
    first = collections.Counter()
    last = collections.Counter()
    allc = collections.Counter()
    subs = collections.Counter()
    by_kind = {k: [0, 0] for k in ("interjection-like start", "other start")}
    by_split = {k: [0, 0] for k in ("starts at a forced split", "starts after a pause")}
    start_bins = collections.defaultdict(lambda: [0, 0])
    end_bins = collections.defaultdict(lambda: [0, 0])
    unmapped = 0
    late = collections.Counter()
    for rp in sorted(REF.glob("*.srt")):
        cid = clip_id_of(rp)
        refs = load_srt(rp, reference=True)
        hyps = sorted(load_srt(find_hyp(run, cid)), key=lambda x: x["start"])
        prev_end = {round(hyps[i]["start"], 3): round(hyps[i - 1]["end"], 3) for i in range(1, len(hyps))}
        frames = wins.get(cid, [])
        for r, h in match_subtitles(refs, hyps)["matched"]:
            g, o = norm(r["text"]), norm(h["text"])
            if not g:
                continue
            st, sub = char_states(g, o)
            for s in st:
                allc[s] += 1
            first[st[0]] += 1
            last[st[-1]] += 1
            bad_first = st[0] != "correct"
            bad_last = st[-1] != "correct"
            if 0 in sub:
                subs[f"{g[0]}→{sub[0]}"] += 1
            kind = "interjection-like start" if g[0] in INTERJECTION_START else "other start"
            by_kind[kind][0] += 1
            by_kind[kind][1] += bad_first
            split = "starts at a forced split" if prev_end.get(round(h["start"], 3)) == round(h["start"], 3) \
                else "starts after a pause"
            by_split[split][0] += 1
            by_split[split][1] += bad_first
            mid = (h["start"] + h["end"]) / 2
            fr = [f for f in frames if f[0] - 0.01 <= mid <= f[1] + 0.01]
            if not fr:
                unmapped += 1
                continue
            ws, we, _ = min(fr, key=lambda f: abs((f[0] + f[1]) / 2 - mid))
            sb = bins(ws - r["start"], (-0.3, 0.0, 0.1, 0.3),
                      ("window starts > 0.3 s before", "0-0.3 s before", "0-0.1 s after", "0.1-0.3 s after",
                       "> 0.3 s after"))
            start_bins[sb][0] += 1
            start_bins[sb][1] += bad_first
            if sb == "> 0.3 s after":
                late["lines"] += 1
                late["first char wrong or missing"] += bad_first
                late["start at a forced split"] += split == "starts at a forced split"
                i = next((k for k, x in enumerate(hyps) if x is h), None)
                prev_text = norm(hyps[i - 1]["text"]) if i else ""
                head = g[:2]
                if head and head in prev_text[-6:]:
                    late["opening found at the end of our previous line"] += 1
                elif head and head in o:
                    late["opening found inside this line (not at its start)"] += 1
                else:
                    late["opening not found nearby"] += 1
            eb = bins(we - r["end"], (-0.3, 0.0, 0.3), ("window ends > 0.3 s before", "0-0.3 s before",
                                                         "0-0.3 s after", "> 0.3 s after"))
            end_bins[eb][0] += 1
            end_bins[eb][1] += bad_last
    n = sum(allc.values())
    print(f"\n=== {name}  (matched pairs {first.total()}, characters in them {n})")
    print(f"1. per-pair alignment: first character wrong {first['wrong'] / first.total():.1%}, missing "
          f"{first['missing'] / first.total():.1%} | last character wrong {last['wrong'] / last.total():.1%}, missing "
          f"{last['missing'] / last.total():.1%} | all characters wrong {allc['wrong'] / n:.1%}, missing "
          f"{allc['missing'] / n:.1%}")
    print("2. first-character substitutions: " + ", ".join(f"{k}×{v}" for k, v in subs.most_common(12)))
    print("   first character wrong or missing: " + "; ".join(f"{k} {rate(v)}" for k, v in by_kind.items()))
    print("3. " + "; ".join(f"{k}: first char wrong or missing {rate(v)}" for k, v in by_split.items()))
    print(f"4. window start minus ground-truth start -> first char wrong or missing ({unmapped} lines not mapped):")
    for lab in ("window starts > 0.3 s before", "0-0.3 s before", "0-0.1 s after", "0.1-0.3 s after", "> 0.3 s after"):
        print(f"     {lab:30} {rate(start_bins[lab])}")
    print("6. lines whose window starts > 0.3 s after the ground truth: " +
          ", ".join(f"{k} {v}" for k, v in late.items()))
    print("5. window end minus ground-truth end -> last char wrong or missing:")
    for lab in ("window ends > 0.3 s before", "0-0.3 s before", "0-0.3 s after", "> 0.3 s after"):
        print(f"     {lab:30} {rate(end_bins[lab])}")
