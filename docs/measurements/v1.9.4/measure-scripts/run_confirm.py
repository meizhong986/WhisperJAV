"""REQ2 confirming runs (2026-10-05): anime-whisper at conservative and balanced, which run the hysteresis decoder
and were not measured before the 1.9.4 longest-segment change. Each sensitivity runs at its 1.9.3 value
(passed explicitly; an explicit value wins over the table) and at the 1.9.4 default (3.0 s, from the table).
Same clips, command shape and recorder as run_levers.py. Resumable (DONE marker).
Usage: python run_confirm.py <out folder> [names]"""
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
ANIME = {"framer": "vad-grouped", "generator_backend": "anime-whisper", "timestamp_mode": "vad_only",
         "assembly_cleaner": "passthrough", "stepdown": False}
RUNS = {
    "a_cons_old6": ("conservative", {"max_speech_duration": 6.0}),
    "a_cons_new3": ("conservative", {}),
    "a_bal_old5": ("balanced", {"max_speech_duration": 5.0}),
    "a_bal_new3": ("balanced", {}),
    # owner, 2026-10-05: try the middle value, 4 s, for the two hysteresis rows
    "a_cons_4": ("conservative", {"max_speech_duration": 4.0}),
    "a_bal_4": ("balanced", {"max_speech_duration": 4.0}),
}

out_root = Path(sys.argv[1])
for name in (sys.argv[2:] or list(RUNS)):
    sens, extra = RUNS[name]
    out = out_root / name / "run1"
    if (out / "DONE").exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "whisperjav.main", *CLIPS, "--ensemble", "--pass1-pipeline", "qwen",
           "--pass1-sensitivity", sens, "--pass1-model", "litagin/anime-whisper",
           "--pass1-qwen-params", json.dumps({**ANIME, **extra}),
           "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", "whisperseg",
           "--merge-strategy", "pass1_primary", "--output-dir", str(out), "--subs-language", "native",
           "--language", "japanese"]
    (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
    rec = out / "framing.jsonl"
    rec.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=str(RECORDER), WJ_RECORD_OUT=str(rec))
    t0 = time.monotonic()
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
    status = {"run": name, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
              "n_srt": len(list(out.glob("*.srt")))}
    print(json.dumps(status), flush=True)
    if rc == 0:
        (out / "DONE").write_text(json.dumps(status), encoding="utf-8")
print("finished", flush=True)
