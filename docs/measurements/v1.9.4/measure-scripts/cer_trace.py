"""Character-error trace for every REQ2 run (1.9.4): which segmenter settings the run actually used (read from the
segmenter recorder, not from notes), its timing summary, and its character error rate split into substituted,
missing and extra characters, all pooled over the 7 reference clips.

Text normalisation, as score_levers.py: sound notes in brackets removed, NFKC, letters and digits only. Netflix
text is not word-for-word, so the rate compares runs; it is not an absolute accuracy.

Usage: python cer_trace.py <run folder> [<run folder> ...]   (each folder holds the SRTs and framing.jsonl)
Prints a Markdown table."""
import json
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
REFS = sorted(REF.glob("*.srt"))


def norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def settings(run):
    f = run / "framing.jsonl"
    cmd = json.loads((run / "command.json").read_text(encoding="utf-8")) if (run / "command.json").exists() else []
    if isinstance(cmd, dict):          # run_step2.py stores {"cmd": [...], "env": {...}}
        cmd = cmd.get("cmd", [])
    model = "anime-whisper" if any("anime-whisper" in c for c in cmd) else "Qwen3-ASR"
    sens = cmd[cmd.index("--pass1-sensitivity") + 1] if "--pass1-sensitivity" in cmd else "?"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") == "segmenter":
                a, fr = d["attrs"], d["framer"]
                dec = a.get("segmentation_decoder")
                return (model, sens, a.get("max_speech_duration_s"), dec,
                        a.get("grow_floor") if dec == "offline" else "-", a.get("gap_merge_ms") if dec == "offline" else "-",
                        a.get("neg_threshold") if dec != "offline" else "-", a.get("end_pad_ms"),
                        fr.get("_max_group"), fr.get("_chunk_threshold"))
    return (model, sens) + ("?",) * 8


def measure(run):
    S = D = I = R = H = 0
    matched = within = n_lines = 0
    end_err = []
    for rp in REFS:
        refs = load_srt(rp, reference=True)
        hyps = load_srt(find_hyp(run, clip_id_of(rp)))
        n_lines += len(hyps)
        r = norm("".join(x["text"] for x in refs))
        h = norm("".join(x["text"] for x in hyps))
        for op in Levenshtein.editops(r, h):
            if op.tag == "replace":
                S += 1
            elif op.tag == "delete":
                D += 1
            else:
                I += 1
        R += len(r)
        H += len(h)
        for ref, hyp in match_subtitles(refs, hyps)["matched"]:
            matched += 1
            e = hyp["end"] - ref["end"]
            within += abs(e) <= 0.5
            end_err.append(abs(e))
    return dict(cer=(S + D + I) / R, S=S / R, D=D / R, I=I / R, ref_chars=R, hyp_chars=H, lines=n_lines,
                matched=matched, within=within, end_med=statistics.median(end_err) if end_err else 0.0)


print("| Run | Model | Sensitivity | Longest segment (s) | Decoder | Grow floor | Gap merge (ms) | End level "
      "| End pad (ms) | Group cap / gap (s) | Lines | Matched | Ends ≤ 0.5 s | End error med (s) "
      "| CER | Substituted | Missing | Extra | Our chars |")
print("|" + "---|" * 19)
for arg in sys.argv[1:]:
    run = Path(arg)
    st = settings(run)
    m = measure(run)
    name = run.parent.name if run.name.startswith("run") else run.name
    print(f"| {name} | {st[0]} | {st[1]} | {st[2]} | {st[3]} | {st[4]} | {st[5]} | {st[6]} | {st[7]} | {st[8]} / {st[9]} "
          f"| {m['lines']} | {m['matched']} | {m['within']} | {m['end_med']:.2f} | {m['cer']:.3f} | {m['S']:.3f} "
          f"| {m['D']:.3f} | {m['I']:.3f} | {m['hyp_chars']} |", flush=True)
print(f"\nGround-truth characters (after normalisation): {m['ref_chars']}")
