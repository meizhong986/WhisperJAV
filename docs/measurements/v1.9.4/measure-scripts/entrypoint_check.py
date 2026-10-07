"""Run every ChronosJAV entry point on a short clip with the segmenter recorder, and print the settings the
speech segmenter and the framer were actually built with. Usage: python entrypoint_check.py"""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = sys.executable  # the Python running this script (the WhisperJAV environment)
S = Path(r"<work folder>")
CLIP = str(S / "e2e" / "film_A_clean_clip.mp4")
OUT = S / "entrypoints"
RECORDER = REPO / "scripts" / "measure" / "segmenter_recorder"

RUNS = {}
for sens in ("conservative", "balanced", "aggressive"):
    RUNS[f"cli_qwen3_{sens}"] = ["--mode", "qwen", "--qwen-sensitivity", sens]
    RUNS[f"cli_anime_{sens}"] = ["--mode", "qwen", "--qwen-generator", "anime-whisper", "--qwen-sensitivity", sens]
    RUNS[f"ens_qwen3_{sens}"] = ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-sensitivity", sens,
                                 "--pass1-model", "Qwen/Qwen3-ASR-1.7B",
                                 "--pass1-qwen-params", json.dumps({"framer": "vad-grouped"}),
                                 "--pass1-speech-segmenter", "whisperseg"]
    RUNS[f"ens_anime_{sens}"] = ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-sensitivity", sens,
                                 "--pass1-model", "litagin/anime-whisper",
                                 "--pass1-qwen-params", json.dumps({"framer": "vad-grouped",
                                                                    "generator_backend": "anime-whisper"}),
                                 "--pass1-speech-segmenter", "whisperseg"]
RUNS["cli_qwen3_user_5.5"] = ["--mode", "qwen", "--qwen-max-speech-duration", "5.5"]
RUNS["ens_anime_aggr_user_5.5"] = ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-sensitivity", "aggressive",
                                   "--pass1-model", "litagin/anime-whisper",
                                   "--pass1-qwen-params", json.dumps({"framer": "vad-grouped",
                                                                      "generator_backend": "anime-whisper",
                                                                      "max_speech_duration": 5.5}),
                                   "--pass1-speech-segmenter", "whisperseg"]

for name, args in RUNS.items():
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    rec = out / "framing.jsonl"
    rec.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=str(RECORDER), WJ_RECORD_OUT=str(rec))
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run([PY, "-m", "whisperjav.main", CLIP, *args, "--output-dir", str(out),
                             "--language", "japanese"], stdout=log, stderr=subprocess.STDOUT, env=env,
                            cwd=str(REPO)).returncode
    seg = None
    if rec.exists():
        for line in rec.open(encoding="utf-8"):
            d = json.loads(line)
            if d.get("kind") == "segmenter":
                seg = d
                break
    if seg is None:
        print(f"{name:26} exit {rc}  NO SEGMENTER RECORD", flush=True)
        continue
    a, f = seg["attrs"], seg["framer"]
    print(f"{name:26} exit {rc}  longest segment {a.get('max_speech_duration_s')}  decoder {a.get('segmentation_decoder')}"
          f"  grow floor {a.get('grow_floor')}  threshold {a.get('threshold')}  min silence {a.get('min_silence_duration_ms')}"
          f"  | group cap {f.get('_max_group')}  group gap {f.get('_chunk_threshold')}", flush=True)
