"""Subtitle line durations per run (all 7 clips pooled): count of lines, lines longer than 5 s and 6 s, median and
90th percentile duration. Reference shown first. Read with subeval.srt (same parser as the scoring).
Usage: python line_lengths.py <run folder> [<run folder> ...]"""
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from subeval.srt import read_srt  # noqa: E402

REF = REPO / "test_media" / "Ground_Truths" / "Netflix"


def stats(paths):
    d = sorted(c.end - c.start for p in paths for c in read_srt(p).cues)
    return (f"{len(d)} lines | >5 s: {sum(x > 5 for x in d)} | >6 s: {sum(x > 6 for x in d)} | "
            f"median {statistics.median(d):.2f} s | p90 {d[int(0.9 * (len(d) - 1))]:.2f} s")


print("| Run | Lines |")
print("|---|---|")
print(f"| reference | {stats(sorted(REF.glob('*.srt')))} |")
for a in sys.argv[1:]:
    run = Path(a)
    print(f"| {run.parent.name} | {stats(sorted(run.glob('*.srt')))} |")
