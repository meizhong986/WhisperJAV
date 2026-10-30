"""Score a batch from run_f0_baseline.py with subeval and put the results in one table.

For each pipeline folder: `python -m subeval score --ref-dir <batch>/refs --test-dir <batch>/<pipeline>
--out <batch>/scores/<pipeline>` (full report.md, result.json, pairs.txt), then one table row per pipeline and film
and one per pipeline over both films. Text columns first, timing after. The reference subtitles are copied into
<batch>/refs/ first, if they are not there yet.

Usage (from the repository root, any Python 3.10+; subeval uses the standard library only):
    python scripts/measure/summarize_f0.py --batch D:\\WJ110_measure\\F0_195 [--window 1.0]

Writes <batch>/summary.md and <batch>/summary.json.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from subeval import __version__ as SUBEVAL_VERSION  # noqa: E402
from subeval.cli import pair_folder, score_files  # noqa: E402
from subeval.evaluate import summarize  # noqa: E402

REF_DIR = REPO / "test_media" / "Ground_Truths" / "Netflix"
REFS = ["The.Naked.Director.S01E04.Scene.4.ja.srt", "The.Naked.Director.S02E02.Scene.5.Bubble.More.srt"]
PIPELINES = ["balanced", "fast", "faster", "fidelity", "chronos_qwen3", "chronos_anime"]

COLUMNS = [  # (header, summary key, kind)
    ("CER", "cer", "pct"), ("wrong", "wrong_rate", "pct"), ("missing", "missing_rate", "pct"),
    ("extra", "extra_rate", "pct"), ("missed lines", "missed_lines", "int"),
    ("susp. halluc. lines", "hallucination_lines", "int"), ("ref lines", "ref_lines", "int"),
    ("test lines", "test_lines", "int"), ("start |err| med s", "start_abs_median", "sec"),
    ("end |err| med s", "end_abs_median", "sec"), ("start measured", "start_measured", "int"),
    ("end measured", "end_measured", "int"),
]


def fmt(v, kind):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if kind == "pct" else f"{v:.3f}" if kind == "sec" else str(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True, type=Path)
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--pipelines", nargs="+", default=PIPELINES,
                    help="pipeline folders to score (default: the six baseline pipelines)")
    a = ap.parse_args()

    refs = a.batch / "refs"
    refs.mkdir(exist_ok=True)
    for name in REFS:
        if not (refs / name).exists():
            shutil.copy2(REF_DIR / name, refs / name)
    status_path = a.batch / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}

    rows, data = [], {"subeval": SUBEVAL_VERSION, "window": a.window, "pipelines": {}}
    for p in a.pipelines:
        test = a.batch / p
        if not test.is_dir():
            continue
        subprocess.run([sys.executable, "-m", "subeval", "score", "--ref-dir", str(refs), "--test-dir", str(test),
                        "--out", str(a.batch / "scores" / p), "--window", str(a.window)], cwd=REPO, check=True,
                       stdout=subprocess.DEVNULL)
        files = pair_folder(refs, test)
        results, problems = score_files(files, window=a.window)
        seconds = status.get(p, {}).get("seconds")
        entry = {"seconds": seconds, "reading_problems": problems, "all": summarize(results),
                 "films": {c.clip: summarize([c]) for c in results}}
        data["pipelines"][p] = entry
        for label, s in [*entry["films"].items(), ("both films", entry["all"])]:
            rows.append([p, label, *(fmt(s.get(k), kind) for _, k, kind in COLUMNS),
                         fmt(seconds, "int") if label == "both films" else "", str(len(problems))])

    head = ["pipeline", "film", *(h for h, _, _ in COLUMNS), "run s", "reading problems"]
    md = [f"# Baseline summary — {a.batch}", "",
          f"subeval {SUBEVAL_VERSION}, window {a.window} s. Rates are shares of reference characters. "
          f"Full definitions: scores/<pipeline>/report.md.", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (a.batch / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (a.batch / "summary.json").write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
