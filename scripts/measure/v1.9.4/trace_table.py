"""Table of threshold and padding values as traced at the calls that use them (scripts/measure/call_tracer), one row
per model x segmenter x sensitivity, cross-checked against the settings recorded in the 7-clip baseline batch.
Usage: python trace_table.py <trace batch folder> <baseline batch folder>"""
import json
import sys
from pathlib import Path

trace_root, base_root = Path(sys.argv[1]), Path(sys.argv[2])
MODELS = {"qwen3": "Qwen3-ASR", "anime": "anime-whisper"}
SEGS = {"whisperseg": "WhisperSeg", "ten": "TEN", "firered-vad": "FireRedVAD"}


def calls(run):
    return [json.loads(l) for l in (run / "calls.jsonl").open(encoding="utf-8")]


def recorded(name):
    f = base_root / name / "run1" / "framing.jsonl"
    for line in f.open(encoding="utf-8"):
        d = json.loads(line)
        if d.get("kind") == "segmenter":
            return d["attrs"]
    return {}


def one(values, label):
    v = sorted(set(values), key=str)
    return v[0] if len(v) == 1 else f"VARIES {v} ({label})"


print("| Model | Segmenter | Sensitivity | Threshold at the final call | Offset threshold used | Start pad read (ms) "
      "| Start pad applied, largest (ms) | End pad read (ms) | End pad applied, largest (ms) | Same as 7-clip recording |")
print("|" + "---|" * 10)
for g, gname in MODELS.items():
    for s, sname in SEGS.items():
        for sens in ("conservative", "balanced", "aggressive"):
            name = f"{g}_{s}_{sens}"
            run = trace_root / name / "run1"
            if not (run / "calls.jsonl").exists():
                print(f"| {gname} | {sname} | {sens} | not traced |||||||")
                continue
            c = calls(run)
            pad = [x for x in c if x["call"].endswith(("_apply_padding", "_pad_and_convert"))]
            if s == "ten":
                thr = one([x["threshold"] for x in c if x["call"] == "ten_vad.TenVad.__init__"], "TenVad")
                off = "-"
            elif s == "firered-vad":
                pp = [x["values"] for x in c if x["call"].startswith("fireredvad.FireRedVad.from_pretrained")]
                thr = one([v.get("prob_threshold") for v in pp], "postprocessor prob_threshold")
                off = "-"
            else:
                dec = [x for x in c if x["call"].startswith("WhisperSegSpeechSegmenter._probs")]
                thr = one([x["threshold"] for x in dec], "decoder")
                off = one([x["offset_threshold_used"] if x["decoder"] != "offline" else f"grow floor {x['grow_floor']}"
                           for x in dec], "decoder")
            sp = one([x["start_pad_ms_read"] for x in pad], "pad")
            ep = one([x["end_pad_ms_read"] for x in pad], "pad")
            sa = max((x["largest_start_shift_applied_ms"] or 0) for x in pad) if pad else None
            ea = max((x["largest_end_shift_applied_ms"] or 0) for x in pad) if pad else None
            rec = recorded(name)
            same = (rec.get("threshold") == thr and rec.get("start_pad_ms") == sp and rec.get("end_pad_ms") == ep)
            print(f"| {gname} | {sname} | {sens} | {thr} | {off} | {sp} | {sa} | {ep} | {ea} | "
                  f"{'yes' if same else 'NO: ' + json.dumps({k: rec.get(k) for k in ('threshold', 'start_pad_ms', 'end_pad_ms')})} |")
