"""REQ2 step 3, scored against the product strategy (owner, 2026-10-08):
  aggressive   = coverage: fewer missed lines, more of the speech shown, at the risk of errors and hallucinations;
  conservative = accuracy of the text shown;
  balanced     = in between.
Every count comes from subeval (forward and mirror alignment). Derived here, per run and per clip:
  correct  = reference characters paired with an identical test character (R - wrong - missing)
  shown    = test characters (normalised)
  shown_bad= shown - correct = wrong + extra (characters on screen that are not right)
  coverage = correct / R          precision = correct / shown
  bad_lines= suspected hallucination lines + misheard test lines (lines on screen with nothing right)
The wrong/missing/extra split can differ between equally good alignments; correct is given from the forward
alignment and the mirror alignment, and the report uses the lower.
Usage: python strategy_scores.py <levers folder> <baseline folder>   (markdown on stdout)"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from subeval.align import align  # noqa: E402
from subeval.cli import pair_folder  # noqa: E402
from subeval.compare import compare  # noqa: E402
from subeval.evaluate import _streams, evaluate_clip  # noqa: E402
from subeval.normalize import normalizer  # noqa: E402
from subeval.srt import read_srt  # noqa: E402

REF = REPO / "test_media" / "Ground_Truths" / "Netflix"
NORM = normalizer("ja-v1")


def clip_counts(ref_p, test_p, clip):
    rc, tc = read_srt(ref_p).cues, read_srt(test_p).cues
    res = evaluate_clip(rc, tc, clip)
    rcs, tcs = sorted(rc, key=lambda c: (c.start, c.end)), sorted(tc, key=lambda c: (c.start, c.end))
    rs, ro, rl = _streams(rcs, NORM)
    ts, to, tl = _streams(tcs, NORM)
    ops_m, _ = align(rs, ro, rl, ts, to, tl, 1.0, "mirror")
    correct_m = sum(1 for o in ops_m if o[0] == "hit")
    correct_f = res.n_ref_chars - res.wrong - res.missing
    return res, {"R": res.n_ref_chars, "shown": len(ts), "edits": res.edits, "correct": min(correct_f, correct_m),
            "correct_spread": abs(correct_f - correct_m),
            "missed": sum(1 for r in res.ref if r.missed and r.n > 0),
            "halluc": sum(1 for t in res.test if t.hallucination),
            "misheard": sum(1 for t in res.test if t.misheard)}


def run_counts(run):
    """(per-clip counts, subeval clip results)"""
    out = {clip: clip_counts(ref, test, clip) for clip, ref, test in pair_folder(REF, run)}
    return {k: v[1] for k, v in out.items()}, [v[0] for v in out.values()]


def prec(c):
    return c["correct"] / c["shown"] if c["shown"] else 0.0


def pooled(per):
    s = {k: sum(c[k] for c in per.values()) for k in next(iter(per.values()))}
    s["shown_bad"] = s["shown"] - s["correct"]
    s["coverage"] = s["correct"] / s["R"]
    s["precision"] = s["correct"] / s["shown"] if s["shown"] else 0.0
    s["cer"] = s["edits"] / s["R"]
    s["bad_lines"] = s["halluc"] + s["misheard"]
    return s


def clips(base, cand, key):
    """(better, worse) clip counts for a per-clip quantity; key(c) where higher is better."""
    b = w = 0
    for clip in base:
        d = key(cand[clip]) - key(base[clip])
        b += d > 0
        w += d < 0
    return b, w


def main():
    lev_root, base_root = Path(sys.argv[1]), Path(sys.argv[2])
    groups = {}
    for r in sorted(p.parent for p in lev_root.glob("*/run1/DONE")):
        groups.setdefault(r.parent.name.split("__")[0], []).append(r)
    combos = sorted({p.parent.parent.name for p in base_root.glob("*/run1/DONE")})
    head = ("| Run | CER | Correct chars (coverage) | Shown, not right (precision) | Missed lines | Halluc. + misheard "
            "lines | Δ correct (clips +/−) | Δ precision (clips +/−) | Δ missed (clips −/+) | Δ bad lines "
            "| Starts both: median (clips +/−) | Ends both: median (clips +/−) | Spread |")
    print(head)
    print("|---" * 13 + "|")
    today = {}
    for combo in combos:
        base, rb = run_counts(base_root / combo / "run1")
        sb = pooled(base)
        today[combo] = sb
        rows = [("baseline", base, rb, sb)]
        for run in groups.get(combo, []):
            per, rc = run_counts(run)
            rows.append((run.parent.name.split("__")[1], per, rc, pooled(per)))
        print(f"| **{combo}** |" + "|" * 12)
        for name, per, rc, s in rows:
            if name == "baseline":
                d = "| | | | | | |"
            else:
                cb = clips(base, per, lambda c: c["correct"])
                pb = clips(base, per, prec)
                ms = clips(base, per, lambda c: -c["missed"])
                c = compare(rb, rc)
                d = (f"| {s['correct'] - sb['correct']:+d} ({cb[0]}/{cb[1]}) "
                     f"| {100 * (s['precision'] - sb['precision']):+.1f} pt ({pb[0]}/{pb[1]}) "
                     f"| {s['missed'] - sb['missed']:+d} ({ms[0]}/{ms[1]}) "
                     f"| {s['bad_lines'] - sb['bad_lines']:+d} "
                     f"| {c['start_med_base']:.3f} → {c['start_med_cand']:.3f} "
                     f"({c['start_clips_better']}/{c['start_clips_worse']}) "
                     f"| {c['end_med_base']:.3f} → {c['end_med_cand']:.3f} "
                     f"({c['end_clips_better']}/{c['end_clips_worse']}) |")
            print(f"| {name} | {s['cer']:.3f} | {s['correct']} ({s['coverage']:.1%}) "
                  f"| {s['shown_bad']} ({s['precision']:.1%}) | {s['missed']} | {s['halluc']}+{s['misheard']} "
                  f"{d} {s['correct_spread']} |", flush=True)
    print("\n### Today's settings by sensitivity (conservative / balanced / aggressive)\n")
    print("| Model, segmenter | Correct chars | Missed lines | Precision | Halluc. + misheard lines | CER |")
    print("|---|---|---|---|---|---|")
    for gs in sorted({c.rsplit("_", 1)[0] for c in combos}):
        t = [today.get(f"{gs}_{v}") for v in ("conservative", "balanced", "aggressive")]
        if None in t:
            continue
        print(f"| {gs} | " + " / ".join(str(x["correct"]) for x in t) + " | "
              + " / ".join(str(x["missed"]) for x in t) + " | "
              + " / ".join(f"{x['precision']:.1%}" for x in t) + " | "
              + " / ".join(str(x["bad_lines"]) for x in t) + " | "
              + " / ".join(f"{x['cer']:.3f}" for x in t) + " |")


if __name__ == "__main__":
    main()
