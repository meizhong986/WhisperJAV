"""REQ2 lever runs: the baseline command of run_chronosjav_reference_runs.py (WhisperSeg, default timestamp mode)
with ONE setting changed per run. Same clips, same recorder. Resumable (DONE marker).
Usage: python run_levers.py <out folder> [lever ...]"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = sys.executable  # the Python running this script (the WhisperJAV environment)
CLIPS = sorted(str(p) for p in (REPO / "test_media" / "Ground_Truths" / "Netflix").glob("*.mkv"))
RECORDER = REPO / "scripts" / "measure" / "segmenter_recorder"

BASE = {
    "qwen3": ({"framer": "vad-grouped"}, ["--pass1-sensitivity", "balanced", "--pass1-model", "Qwen/Qwen3-ASR-1.7B"]),
    "anime": ({"framer": "vad-grouped", "generator_backend": "anime-whisper", "timestamp_mode": "vad_only",
               "assembly_cleaner": "passthrough", "stepdown": False},
              ["--pass1-sensitivity", "aggressive", "--pass1-model", "litagin/anime-whisper"]),
}
# lever name -> (generator, extra qwen-params, extra --pass1-params)
LEVERS = {
    "q_neg020": ("qwen3", {}, {"neg_threshold": 0.20}),
    "q_maxseg3": ("qwen3", {"max_speech_duration": 3.0}, {}),
    "q_endpad0": ("qwen3", {"vad_end_pad": 0}, {}),
    "a_floor010": ("anime", {"vad_grow_floor": 0.10}, {}),
    "a_floor015": ("anime", {"vad_grow_floor": 0.15}, {}),
    "a_gap150": ("anime", {"vad_gap_merge_ms": 150}, {}),
    "a_maxseg3": ("anime", {"max_speech_duration": 3.0}, {}),
    "a_endpad0": ("anime", {"vad_end_pad": 0}, {}),
    "q_maxseg25": ("qwen3", {"max_speech_duration": 2.5}, {}),
    "q_maxseg2": ("qwen3", {"max_speech_duration": 2.0}, {}),
    "a_maxseg25": ("anime", {"max_speech_duration": 2.5}, {}),
    "a_maxseg2": ("anime", {"max_speech_duration": 2.0}, {}),
    "q_maxseg4": ("qwen3", {"max_speech_duration": 4.0}, {}),
    "a_maxseg3_floor015": ("anime", {"max_speech_duration": 3.0, "vad_grow_floor": 0.15}, {}),
}

out_root = Path(sys.argv[1])
names = sys.argv[2:] or list(LEVERS)
for name in names:
    gen, qx, px = LEVERS[name]
    qparams, args = BASE[gen]
    out = out_root / name / "run1"
    if (out / "DONE").exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "whisperjav.main", *CLIPS, "--ensemble", "--pass1-pipeline", "qwen", *args,
           "--pass1-qwen-params", json.dumps({**qparams, **qx})]
    if px:
        cmd += ["--pass1-params", json.dumps(px)]
    cmd += ["--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", "whisperseg",
            "--merge-strategy", "pass1_primary", "--output-dir", str(out), "--subs-language", "native",
            "--language", "japanese"]
    (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
    rec = out / "framing.jsonl"
    rec.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=str(RECORDER), WJ_RECORD_OUT=str(rec))
    t0 = time.monotonic()
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
    status = {"lever": name, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
              "n_srt": len(list(out.glob("*.srt")))}
    print(json.dumps(status), flush=True)
    if rc == 0:
        (out / "DONE").write_text(json.dumps(status), encoding="utf-8")
print("finished", flush=True)
