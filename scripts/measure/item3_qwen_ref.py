"""Item 3 (transformers 5): Qwen3-ASR + Qwen3-ForcedAligner reference outputs straight through the `qwen-asr` package.

Run once on the stack before item 3 (transformers 4.57.6, PyPI qwen-asr 0.0.6) and once after (transformers 5.20, the
patched qwen-asr fork); `--compare` then says whether the text is identical and how far the aligner timestamps moved.

    python item3_qwen_ref.py run --out D:\\WJ110_measure\\F2_ref_qwen\\before
    python item3_qwen_ref.py run --out D:\\WJ110_measure\\F2_ref_qwen\\after
    python item3_qwen_ref.py compare D:\\WJ110_measure\\F2_ref_qwen\\before D:\\WJ110_measure\\F2_ref_qwen\\after

The clips are 60 seconds of each reference film from second 30, 16 kHz mono, cut with ffmpeg into <out>/clips/.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO / "test_media" / "Ground_Truths" / "Netflix"
FILMS = ["The.Naked.Director.S01E04.Scene.4.mkv", "The.Naked.Director.S02E02.Scene.5.Bubble.More.mkv"]
START, LENGTH = 30, 60


def cut_clips(out):
    clips = []
    (out / "clips").mkdir(parents=True, exist_ok=True)
    for f in FILMS:
        wav = out / "clips" / (Path(f).stem + ".wav")
        if not wav.exists():
            subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", str(START), "-t", str(LENGTH),
                            "-i", str(MEDIA_DIR / f), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
        clips.append(wav)
    return clips


def run(out):
    import importlib.metadata as md
    import torch
    import qwen_asr
    from qwen_asr import Qwen3ASRModel

    out.mkdir(parents=True, exist_ok=True)
    clips = cut_clips(out)
    info = {n: md.version(n) for n in ("transformers", "qwen-asr", "torch", "accelerate", "huggingface-hub")}
    info["qwen_asr_path"] = str(Path(qwen_asr.__file__).parent)
    try:
        info["qwen_asr_direct_url"] = md.distribution("qwen-asr").read_text("direct_url.json")
    except Exception:
        info["qwen_asr_direct_url"] = None
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-1.7B", dtype=torch.bfloat16, device_map="cuda:0", max_inference_batch_size=1,
        max_new_tokens=1024, forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
        forced_aligner_kwargs={"dtype": torch.bfloat16, "device_map": "cuda:0"})
    results = {}
    for wav in clips:
        r = model.transcribe(audio=str(wav), language="Japanese", return_time_stamps=True)[0]
        items = [{"text": i.text, "start": i.start_time, "end": i.end_time} for i in (r.time_stamps or [])]
        results[wav.name] = {"language": r.language, "text": r.text, "items": items}
        print(wav.name, len(r.text), "chars,", len(items), "aligned items", flush=True)
    (out / "result.json").write_text(json.dumps({"info": info, "results": results}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")
    print(json.dumps(info, indent=1))


def compare(a, b):
    ra = json.loads((a / "result.json").read_text(encoding="utf-8"))
    rb = json.loads((b / "result.json").read_text(encoding="utf-8"))
    print("before:", {k: v for k, v in ra["info"].items() if k != "qwen_asr_direct_url"})
    print("after: ", {k: v for k, v in rb["info"].items() if k != "qwen_asr_direct_url"})
    print("after direct_url:", rb["info"].get("qwen_asr_direct_url"))
    ok = True
    for name, x in ra["results"].items():
        y = rb["results"].get(name)
        if y is None:
            print(name, "MISSING in after"); ok = False; continue
        same_text = x["text"] == y["text"]
        same_tokens = [i["text"] for i in x["items"]] == [i["text"] for i in y["items"]]
        diffs = [max(abs(p["start"] - q["start"]), abs(p["end"] - q["end"]))
                 for p, q in zip(x["items"], y["items"])] if same_tokens else []
        print(f"{name}: text identical={same_text}; aligned items {len(x['items'])} vs {len(y['items'])}, "
              f"same tokens={same_tokens}; max timestamp difference={max(diffs) if diffs else 'n/a'} s; "
              f"items differing by >0.08 s: {sum(d > 0.08 for d in diffs)}")
        if not same_text:
            print("  before:", x["text"]); print("  after: ", y["text"])
        ok = ok and same_text and same_tokens
    print("RESULT:", "same text and tokens" if ok else "DIFFERENT")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--out", required=True, type=Path)
    c = sub.add_parser("compare"); c.add_argument("a", type=Path); c.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args.out)
    else:
        sys.exit(compare(args.a, args.b))


if __name__ == "__main__":
    main()
