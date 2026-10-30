"""Why do matched lines end late? For each matched pair (ground-truth line, our line):
  - 'spans next': our line runs past the start of the next ground-truth line (our line covers two of theirs)
  - 'tail': our end is late but stops before the next ground-truth line starts
  - 'ok/early': end error <= 0.5 s
Also the line-length comparison and how many of our lines there are per ground-truth line.
Usage: python late_ends.py <ref dir> <run dir> [<run dir> ...]"""
import statistics
import sys
from pathlib import Path

from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

ref_dir = Path(sys.argv[1])
for run in map(Path, sys.argv[2:]):
    spans = tail = ok = 0
    tail_amt, span_amt, hyp_len, ref_len = [], [], [], []
    n_hyp = n_ref = 0
    for ref_path in sorted(ref_dir.glob("*.srt")):
        refs = load_srt(ref_path, reference=True)
        hyps = load_srt(find_hyp(run, clip_id_of(ref_path)))
        n_hyp += len(hyps)
        n_ref += len(refs)
        refs_sorted = sorted(refs, key=lambda r: r["start"])
        for ref, hyp in match_subtitles(refs, hyps)["matched"]:
            e = hyp["end"] - ref["end"]
            nxt = [r["start"] for r in refs_sorted if r["start"] >= ref["end"] - 1e-6 and r is not ref]
            nxt_start = min(nxt) if nxt else None
            hyp_len.append(hyp["end"] - hyp["start"])
            ref_len.append(ref["end"] - ref["start"])
            if e <= 0.5:
                ok += 1
            elif nxt_start is not None and hyp["end"] > nxt_start + 0.2:
                spans += 1
                span_amt.append(e)
            else:
                tail += 1
                tail_amt.append(e)
    n = ok + spans + tail
    print(f"{run.parent.name}/{run.name}: our lines {n_hyp}, ground-truth lines {n_ref}, matched {n}")
    print(f"   end within 0.5 s: {ok}   late, covers next GT line: {spans} (median {statistics.median(span_amt) if span_amt else 0:.2f} s)"
          f"   late, tail only: {tail} (median {statistics.median(tail_amt) if tail_amt else 0:.2f} s)")
    print(f"   line length median: ours {statistics.median(hyp_len):.2f} s, ground truth {statistics.median(ref_len):.2f} s")
