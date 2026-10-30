"""REQ2 for every speech segmenter (1.9.4, 2026-10-08): one row per run, text first, timing second.
Settings are read from the run's recording (framing.jsonl), not from the YAML. Text: character error rate split into
substituted / missing / extra (cer_trace.py normalisation), and ground-truth lines with no character produced.
Timing: on the ground-truth lines the run matched, start and end error (median of the absolute error, and the signed
mean of the end error: + = our line ends late). All pooled over the 7 reference clips (305 lines).

Usage: python score_segmenters.py <run folder> [<run folder> ...]   prints a Markdown table."""
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

REF = Path(__file__).resolve().parents[3] / "test_media" / "Ground_Truths" / "Netflix"
REFS = sorted(REF.glob("*.srt"))
SHORT = {"TenSpeechSegmenter": "TEN", "FireRedVadSpeechSegmenter": "FireRedVAD"}


def norm(t):
    t = re.sub(r"[（(][^）)]*[）)]", "", t)
    t = unicodedata.normalize("NFKC", t)
    return "".join(ch for ch in t if unicodedata.category(ch)[0] in "LN")


def settings(run):
    cmd = json.loads((run / "command.json").read_text(encoding="utf-8"))
    if isinstance(cmd, dict):
        cmd = cmd.get("cmd", [])
    model = "anime-whisper" if any("anime-whisper" in c for c in cmd) else "Qwen3-ASR"
    sens = cmd[cmd.index("--pass1-sensitivity") + 1] if "--pass1-sensitivity" in cmd else "?"
    seg = cmd[cmd.index("--pass1-speech-segmenter") + 1] if "--pass1-speech-segmenter" in cmd else "?"
    f = run / "framing.jsonl"
    if f.exists():
        for line in f.open(encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") == "segmenter":
                a, fr = d["attrs"], d["framer"]
                level = a.get("threshold", a.get("neg_threshold", "?"))
                extra = f", smooth {a['smooth_window_size']}" if "smooth_window_size" in a else ""
                if "grow_floor" in a and a.get("segmentation_decoder") == "offline":
                    extra = f", floor {a['grow_floor']}"
                s = (f"lvl {level}, min sp/sil {a.get('min_speech_duration_ms', '-')}/"
                     f"{a.get('min_silence_duration_ms', '-')}, max {a.get('max_speech_duration_s')}, "
                     f"pad {a.get('start_pad_ms')}/{a.get('end_pad_ms')}, grp {fr.get('_max_group')}/"
                     f"{fr.get('_chunk_threshold')}{extra}")
                return model, sens, seg, s
    return model, sens, seg, "not recorded"


def measure(run):
    S = D = I = R = 0
    lines = matched = gone = 0
    st, en, en_signed = [], [], []
    for rp in REFS:
        refs = load_srt(rp, reference=True)
        hyps = load_srt(find_hyp(run, clip_id_of(rp)))
        lines += len(hyps)
        r = norm("".join(x["text"] for x in refs))
        h = norm("".join(x["text"] for x in hyps))
        ops = Levenshtein.editops(r, h)
        deleted = set()
        for op in ops:
            if op.tag == "replace":
                S += 1
            elif op.tag == "delete":
                D += 1
                deleted.add(op.src_pos)
            else:
                I += 1
        R += len(r)
        pos = 0
        for x in refs:  # ground-truth lines none of whose characters appear in our text
            n = len(norm(x["text"]))
            if n and all(k in deleted for k in range(pos, pos + n)):
                gone += 1
            pos += n
        for ref, hyp in match_subtitles(refs, hyps)["matched"]:
            matched += 1
            st.append(abs(hyp["start"] - ref["start"]))
            en.append(abs(hyp["end"] - ref["end"]))
            en_signed.append(hyp["end"] - ref["end"])
    med = lambda v: statistics.median(v) if v else float("nan")
    return dict(cer=(S + D + I) / R, S=S / R, D=D / R, I=I / R, lines=lines, gone=gone, matched=matched,
                within=sum(e <= 0.5 for e in en), st=med(st), en=med(en),
                ens=statistics.mean(en_signed) if en_signed else float("nan"))


if __name__ == "__main__":
    print("| Run | Model | Sens. | Segmenter | Settings used (recorded) | CER | Subst. | Missing | Extra "
          "| Lines with no text | Our lines | Matched | Ends ≤ 0.5 s | Start err med (s) | End err med (s) "
          "| End mean (s) |")
    print("|" + "---|" * 16)
    for arg in sys.argv[1:]:
        run = Path(arg)
        model, sens, seg, s = settings(run)
        m = measure(run)
        name = run.parent.name if run.name.startswith("run") else run.name
        print(f"| {name} | {model} | {sens} | {seg} | {s} | {m['cer']:.3f} | {m['S']:.3f} | {m['D']:.3f} "
              f"| {m['I']:.3f} | {m['gone']} | {m['lines']} | {m['matched']} | {m['within']} | {m['st']:.3f} "
              f"| {m['en']:.3f} | {m['ens']:+.3f} |", flush=True)
