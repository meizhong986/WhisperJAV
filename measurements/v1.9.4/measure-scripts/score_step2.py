"""Score the character-accuracy step-2 runs against their option-B baselines.
For every run: the settings it actually used (recorder, incl. the experiment notes), then the same measures as
cer_trace.py, line_accounting.py, onset_analysis.py and compare_pairs.py, side by side with its baseline.
Usage: python score_step2.py <step2 folder> <levers folder> <confirm folder>"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
step2, levers, confirm = (Path(a) for a in sys.argv[1:4])
BASE = {"aa": levers / "a_floor015" / "run1", "q": levers / "q_maxseg4" / "run1", "ab": confirm / "a_bal_4" / "run1"}
runs = sorted(p.parent for p in step2.glob("*/run1/DONE"))


def py(script, *args):
    r = subprocess.run([sys.executable, str(HERE / script), *map(str, args)], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return r.stdout + (r.stderr[-2000:] if r.returncode else "")


print("## Settings each run used (from its recording)")
for run in runs:
    f = run / "framing.jsonl"
    seg, exp = None, []
    for line in f.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "segmenter" and seg is None:
            seg = d
        elif d.get("kind") == "experiment":
            exp.append({k: v for k, v in d.items() if k not in ("kind",)})
    a = seg["attrs"] if seg else {}
    print(f"- {run.parent.name}: start/end pad {a.get('start_pad_ms')}/{a.get('end_pad_ms')} ms, longest segment "
          f"{a.get('max_speech_duration_s')}, decoder {a.get('segmentation_decoder')}, grow floor {a.get('grow_floor')}; "
          f"experiment: {exp or 'none'}")

groups = {}
for run in runs:
    groups.setdefault(run.parent.name.split("_")[0], []).append(run)
for key, rs in groups.items():
    base = BASE[key]
    print(f"\n## Group {key}: baseline {base.parent.name}")
    print(py("cer_trace.py", base, *rs))
    acct = py("line_accounting.py", base, *rs)
    print(acct[acct.index("| Run | Our lines"):] if "| Run | Our lines" in acct else acct)
    onset = py("onset_analysis.py", base, *rs)
    print("\n".join(l for l in onset.splitlines() if l.startswith(("===", "1.", "3.", "6.")) or "> 0.3 s after" in l
                    or "0-0.3 s before" in l))
    pairs = []
    for r in rs:
        pairs += [base, r]
    print(py("compare_pairs.py", *pairs))
