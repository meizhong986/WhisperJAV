"""REQ2 step 3 runs scored with subeval (the standalone utility at the repository top level). WhisperJAV-specific
parts only: which run folders exist and which baseline each lever run belongs to. Every number comes from subeval.

  python score_runs_subeval.py regress <levers folder> <baseline folder>
      For every run: subeval's CER (plain) must equal score_segmenters.py's CER exactly (same normalisation, same
      edit distance), and subeval must read as many lines from every SRT as pysrt does. Prints any difference.
  python score_runs_subeval.py levers <levers folder> <baseline folder> [--window S]
      One block per combination: the baseline row, then each change with its differences, and the clip-by-clip
      comparison (subeval compare). Markdown on stdout.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from subeval.cli import pair_folder, score_files  # noqa: E402
from subeval.compare import compare  # noqa: E402
from subeval.evaluate import summarize  # noqa: E402

REF = REPO / "test_media" / "Ground_Truths" / "Netflix"


def runs(lev_root, base_root):
    out = sorted(p.parent for p in Path(base_root).glob("*/run1/DONE"))
    return out + sorted(p.parent for p in Path(lev_root).glob("*/run1/DONE"))


def regress(lev_root, base_root):
    from score_segmenters import measure
    from whisperjav.bench.timing import load_srt
    from subeval.srt import read_srt
    bad = n = 0
    for run in runs(lev_root, base_root):
        files = pair_folder(REF, run)
        res, problems = score_files(files)
        new = summarize(res)["cer_plain"]
        old = measure(run)["cer"]
        n += 1
        if abs(new - old) > 1e-12:
            bad += 1
            print("CER differs", run, old, new)
        for _, ref, test in files:
            for p, is_ref in ((ref, True), (test, False)):
                if len(read_srt(p).cues) != len(load_srt(p, reference=is_ref)):
                    bad += 1
                    print("line count differs", p)
    print(f"{n} runs checked, {bad} differences")


ROWS = [("CER", "cer", "{:.3f}"), ("CER plain", "cer_plain", "{:.3f}"), ("Missed", "missed_lines", "{}"),
        ("Halluc.", "hallucination_lines", "{}"), ("Misheard", "misheard_test_lines", "{}"),
        ("Start lost", "start_lost", "{}"),
        ("Start misplaced", "start_misplaced", "{}"), ("End lost", "end_lost", "{}"),
        ("End misplaced", "end_misplaced", "{}"), ("Start med", "start_abs_median", "{:.3f}"),
        ("End med", "end_abs_median", "{:.3f}"), ("Ends ≤0.5 s", "end_within_share", "{:.0%}"),
        ("End mean", "end_signed_mean", "{:+.3f}")]


def levers(lev_root, base_root, window):
    groups = {}
    for r in sorted(p.parent for p in Path(lev_root).glob("*/run1/DONE")):
        groups.setdefault(r.parent.name.split("__")[0], []).append(r)
    print(f"subeval window {window} s. Text columns: CER (time-aware), CER plain, missed lines, suspected "
          "hallucination lines, start/end classes. Timing: each run's own measured boundaries. Then, against the "
          "baseline, clip by clip: edits (text) and the end/start median on the boundaries measured in both runs.")
    for combo in sorted(groups):
        base = Path(base_root) / combo / "run1"
        rb, _ = score_files(pair_folder(REF, base), window)
        sb = summarize(rb)
        print(f"\n### {combo}\n")
        print("| Change | " + " | ".join(h for h, _, _ in ROWS) + " | Δ edits (clips better/worse) | Ends both: n, "
              "median base → this (clips better/worse) | Starts both: median base → this (clips better/worse) |")
        print("|---" * (len(ROWS) + 4) + "|")
        print("| baseline | " + " | ".join(f.format(sb[k]) for _, k, f in ROWS) + " | | | |")
        for run in groups[combo]:
            rc, _ = score_files(pair_folder(REF, run), window)
            sc = summarize(rc)
            c = compare(rb, rc)
            print(f"| {run.parent.name.split('__')[1]} | " + " | ".join(f.format(sc[k]) for _, k, f in ROWS)
                  + f" | {c['d_edits']:+d} ({c['text_clips_better']}/{c['text_clips_worse']}) "
                  f"| {c['end_n_both']}: {c['end_med_base']:.3f} → {c['end_med_cand']:.3f} "
                  f"({c['end_clips_better']}/{c['end_clips_worse']}) "
                  f"| {c['start_med_base']:.3f} → {c['start_med_cand']:.3f} "
                  f"({c['start_clips_better']}/{c['start_clips_worse']}) |", flush=True)


if __name__ == "__main__":
    mode, lev, base = sys.argv[1:4]
    w = float(sys.argv[sys.argv.index("--window") + 1]) if "--window" in sys.argv else 1.0
    regress(lev, base) if mode == "regress" else levers(lev, base, w)
