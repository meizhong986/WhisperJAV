"""Step 1, part 2: wrong partners. An accepted matcher pair (g, t) has a wrong partner when another of our lines that
overlaps g in time by >= 0.1 s holds at least 2 more of g's characters in order (longest common subsequence) than t.
Also: how the timing errors of such pairs compare with those of the other pairs."""
import difflib
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "D:/Git/WhisperJav_V1_Minami_Edition/docs/measurements/v1.9.4/measure-scripts")
from score_segmenters import REFS
from whisperjav.bench.matcher import match_subtitles
from whisperjav.bench.metrics import normalize_text
from whisperjav.bench.timing import load_srt
from whisperjav.bench.timing_cli import clip_id_of, find_hyp

R = Path("D:/WJ194_measure/req2_ten_firered")
runs = sorted(p.parent for p in R.glob("baseline/*/run*/DONE")) + sorted(p.parent for p in R.glob("levers/*/run1/DONE"))
REF = {clip_id_of(rp): load_srt(rp, reference=True) for rp in REFS}


def lcs(a, b):
    return sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())


C = Counter()
end_ok, end_bad = [], []
ex = None
for run in runs:
    for cid, refs in REF.items():
        hyps = load_srt(find_hyp(run, cid))
        for g, t in match_subtitles(refs, hyps)["matched"]:
            gt = normalize_text(g["text"])
            mine = lcs(gt, normalize_text(t["text"]))
            best = max((lcs(gt, normalize_text(o["text"])) for o in hyps if o is not t and
                        min(g["end"], o["end"]) - max(g["start"], o["start"]) >= 0.1), default=0)
            C["pairs"] += 1
            e = abs(t["end"] - g["end"])
            if best >= mine + 2:
                C["wrong partner"] += 1
                end_bad.append(e)
                ex = ex or (run.parent.name, cid, g["text"], t["text"], mine, best)
            else:
                end_ok.append(e)
print(dict(C), f"{C['wrong partner'] / C['pairs']:.1%}")
print("end error median: other pairs", round(statistics.median(end_ok), 3), "| wrong-partner pairs",
      round(statistics.median(end_bad), 3))
print("example", ex)
