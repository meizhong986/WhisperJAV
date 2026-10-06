"""Character-accuracy step 2 (owner, 2026-10-05): start pads, added silence, forced-split rule. Same 7 clips, command
shape and recorder as run_levers.py; experiment settings through experiment_hooks/ (no product change).
Baselines (option B defaults, already on disk): anime-whisper aggressive = req2_levers/a_floor015, Qwen3-ASR =
req2_levers/q_maxseg4, anime-whisper balanced = req2_confirm/a_bal_4.
Usage: python run_step2.py <out folder> [names]"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = sys.executable
CLIPS = sorted(str(p) for p in (REPO / "test_media" / "Ground_Truths" / "Netflix").glob("*.mkv"))
HOOKS = Path(__file__).resolve().parent / "experiment_hooks"
ANIME = {"framer": "vad-grouped", "generator_backend": "anime-whisper", "timestamp_mode": "vad_only",
         "assembly_cleaner": "passthrough", "stepdown": False}
CONFIG = {
    "aa": ("aggressive", "litagin/anime-whisper", ANIME),
    "ab": ("balanced", "litagin/anime-whisper", ANIME),
    "q": ("balanced", "Qwen/Qwen3-ASR-1.7B", {"framer": "vad-grouped"}),
    "ac": ("conservative", "litagin/anime-whisper", ANIME),
}
# name -> (config, extra qwen-params, experiment environment)
RUNS = {
    "aa_sp100": ("aa", {"vad_start_pad": 100}, {}),
    "aa_sp200": ("aa", {"vad_start_pad": 200}, {}),
    "aa_sp300": ("aa", {"vad_start_pad": 300}, {}),
    "q_sp0": ("q", {"vad_start_pad": 0}, {}),
    "q_sp200": ("q", {"vad_start_pad": 200}, {}),
    "q_sp300": ("q", {"vad_start_pad": 300}, {}),
    "aa_lead200": ("aa", {}, {"WJ_EXP_LEADIN_MS": "200"}),
    "aa_leadtail200": ("aa", {}, {"WJ_EXP_LEADIN_MS": "200", "WJ_EXP_TAIL_MS": "200"}),
    "q_lead200": ("q", {}, {"WJ_EXP_LEADIN_MS": "200"}),
    "q_leadtail200": ("q", {}, {"WJ_EXP_LEADIN_MS": "200", "WJ_EXP_TAIL_MS": "200"}),
    "q_dip30": ("q", {}, {"WJ_EXP_DIP_FROM": "0.3", "WJ_EXP_DIP_ACCEPT_ALL": "1"}),
    "ab_dip30": ("ab", {}, {"WJ_EXP_DIP_FROM": "0.3", "WJ_EXP_DIP_ACCEPT_ALL": "1"}),
    # owner, 2026-10-06: the model's window reaches 300 ms back; the subtitle keeps its time
    "aa_reach300": ("aa", {}, {"WJ_EXP_REACHBACK_MS": "300"}),
    "q_reach300": ("q", {}, {"WJ_EXP_REACHBACK_MS": "300"}),
    # product verification after the 2026-10-06 changes (anime silence 200 ms; split rule for anime
    # conservative/balanced): no experiment variables, product defaults only
    "f_aa": ("aa", {}, {}),
    "f_ab": ("ab", {}, {}),
    "f_ac": ("ac", {}, {}),
    "f_q": ("q", {}, {}),
    # f_ac started 11 s before the lead-in was moved from the generator default to the qwen pipeline and
    # mixed the two file versions: it ran WITHOUT the lead-in (split rule only). Re-run on the final code;
    # f_aa2 repeats f_aa on the final code (must equal aa_lead200 byte for byte).
    "f_ac2": ("ac", {}, {}),
    "f_aa2": ("aa", {}, {}),
    # off switch (owner, 2026-10-06): 0 must reproduce the pre-lead-in output (req2_levers/a_floor015);
    # the default must still reproduce aa_lead200
    "f_aa_off": ("aa", {"leading_silence_ms": 0}, {}),
    "f_aa3": ("aa", {}, {}),
}

out_root = Path(sys.argv[1])
for name in (sys.argv[2:] or list(RUNS)):
    cfg, qx, envx = RUNS[name]
    sens, model, qparams = CONFIG[cfg]
    out = out_root / name / "run1"
    if (out / "DONE").exists():
        continue
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "whisperjav.main", *CLIPS, "--ensemble", "--pass1-pipeline", "qwen",
           "--pass1-sensitivity", sens, "--pass1-model", model,
           "--pass1-qwen-params", json.dumps({**qparams, **qx}),
           "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", "whisperseg",
           "--merge-strategy", "pass1_primary", "--output-dir", str(out), "--subs-language", "native",
           "--language", "japanese"]
    (out / "command.json").write_text(json.dumps({"cmd": cmd, "env": envx}, indent=1), encoding="utf-8")
    rec = out / "framing.jsonl"
    rec.unlink(missing_ok=True)
    env = dict(os.environ, PYTHONPATH=str(HOOKS), WJ_RECORD_OUT=str(rec), **envx)
    t0 = time.monotonic()
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=str(REPO)).returncode
    status = {"run": name, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
              "n_srt": len(list(out.glob("*.srt")))}
    print(json.dumps(status), flush=True)
    if rc == 0:
        (out / "DONE").write_text(json.dumps(status), encoding="utf-8")
print("finished", flush=True)
