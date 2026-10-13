"""TEN longest-segment split fix (owner, 2026-10-08: "FIX IT"), and the two open checks he approved.
Each row compares a run (B) with another run (A), clip by clip, scored with subeval as in strategy_scores.py:
  correct characters, precision, missed lines, hallucination + misheard lines, starts and ends measured in both runs
  (median |error| in seconds, clips better/worse), subtitle lines longer than 5 s and the median line, segments
  longer than the longest-segment limit (from the recorder, framing.jsonl), and audio sent to the ASR (intake).
Usage: python fix_scores.py <fix folder> <baseline folder>   (markdown on stdout)"""
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from intake_scores import intake, reference_spans  # noqa: E402
from strategy_scores import clips, pooled, prec, run_counts  # noqa: E402
from subeval.compare import compare  # noqa: E402
from subeval.srt import read_srt  # noqa: E402

LIMIT = {"conservative": 6.0, "balanced": 5.0, "aggressive": 4.0}


def lines(run):
    d = sorted(c.end - c.start for p in sorted(run.glob("*.srt")) for c in read_srt(p).cues)
    return sum(x > 5 for x in d), statistics.median(d)


def over_limit(run, lim):
    n = over = 0
    worst = 0.0
    for line in (run / "framing.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r.get("kind") == "frame_call":
            for s in r["segments"]:
                n += 1
                d = s["e"] - s["s"]
                worst = max(worst, d)
                over += d > lim + 0.001
    return over, n, worst


def main():
    fix, base = Path(sys.argv[1]), Path(sys.argv[2])
    rows = []
    for gen in ("qwen3", "anime"):
        for sens in ("conservative", "balanced", "aggressive"):
            rows.append((f"{gen} TEN {sens}: today → split fixed", base / f"{gen}_ten_{sens}" / "run1",
                         fix / f"{gen}_ten_{sens}__fix" / "run1", LIMIT[sens], LIMIT[sens]))
    rows += [
        ("qwen3 TEN aggressive, fixed: min speech 100 → 200 ms", fix / "qwen3_ten_aggressive__fix" / "run1",
         fix / "qwen3_ten_aggressive__fix_ms200" / "run1", 4.0, 4.0),
        ("qwen3 FireRedVAD aggressive: min speech 150 → 250 ms", base / "qwen3_firered-vad_aggressive" / "run1",
         fix / "qwen3_firered-vad_aggressive__ms250" / "run1", 5.0, 5.0),
        ("anime TEN, fixed: aggressive today → conservative's other settings at 0.22, 4 s",
         fix / "anime_ten_aggressive__fix" / "run1", fix / "anime_ten_conservative__fix_g_thr022_max4" / "run1",
         4.0, 4.0),
        ("anime TEN, fixed: aggressive today → same, end pad 100 → 30 ms",
         fix / "anime_ten_aggressive__fix" / "run1", fix / "anime_ten_conservative__fix_g_thr022_max4_ep30" / "run1",
         4.0, 4.0),
        ("anime TEN, fixed: conservative's settings at 0.22, 4 s: end pad 100 → 30 ms",
         fix / "anime_ten_conservative__fix_g_thr022_max4" / "run1",
         fix / "anime_ten_conservative__fix_g_thr022_max4_ep30" / "run1", 4.0, 4.0),
    ]
    # After the product change (owner, 2026-10-08): TEN and FireRedVAD 4 s in ChronosJAV, conservative.
    for gen in ("qwen3", "anime"):
        rows.append((f"{gen} TEN conservative, fixed split: 6 s → product 4 s", fix / f"{gen}_ten_conservative__fix" / "run1",
                     fix / f"{gen}_ten_conservative__product4s" / "run1", 6.0, 4.0))
        rows.append((f"{gen} FireRedVAD conservative: 7 s → product 4 s", base / f"{gen}_firered-vad_conservative" / "run1",
                     fix / f"{gen}_firered-vad_conservative__product4s" / "run1", 7.0, 4.0))
    refs = reference_spans()
    print("| A → B | Audio in A / B | Correct A → B (clips +/−) | Precision (clips +/−) | Missed | Halluc.+mish. "
          "| Starts both (clips +/−) | Ends both (clips +/−) | Lines > 5 s | Median line | Segments over limit |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for label, a, b, la, lb in rows:
        if not (a / "DONE").exists() or not (b / "DONE").exists():
            print(f"| {label} | not run | | | | | | | | | |")
            continue
        pa, ra = run_counts(a)
        pb, rb = run_counts(b)
        sa, sb = pooled(pa), pooled(pb)
        ia, ib = intake(a, refs), intake(b, refs)
        cc = clips(pa, pb, lambda c: c["correct"])
        cp = clips(pa, pb, prec)
        c = compare(ra, rb)
        (l5a, mda), (l5b, mdb) = lines(a), lines(b)
        (oa, na, wa), (ob, nb, wb) = over_limit(a, la), over_limit(b, lb)
        print(f"| {label} | {ia['audio_share']:.1%} / {ib['audio_share']:.1%} "
              f"| {sa['correct']} → {sb['correct']} ({sb['correct'] - sa['correct']:+d}; {cc[0]}/{cc[1]}) "
              f"| {100 * sa['precision']:.1f} → {100 * sb['precision']:.1f} % ({cp[0]}/{cp[1]}) "
              f"| {sa['missed']} → {sb['missed']} | {sa['bad_lines']} → {sb['bad_lines']} "
              f"| {c['start_med_base']:.3f} → {c['start_med_cand']:.3f} ({c['start_clips_better']}/"
              f"{c['start_clips_worse']}) | {c['end_med_base']:.3f} → {c['end_med_cand']:.3f} "
              f"({c['end_clips_better']}/{c['end_clips_worse']}) | {l5a} → {l5b} | {mda:.2f} → {mdb:.2f} s "
              f"| {oa}/{na} (longest {wa:.2f} s) → {ob}/{nb} (longest {wb:.2f} s) |", flush=True)


if __name__ == "__main__":
    main()
