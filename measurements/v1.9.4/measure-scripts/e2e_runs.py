"""End-to-end runs of the wired audio check (REQ1) on two 3-minute stream-copied clips of film A:
one with a 2.0 s hidden hole at 0:53, one without. Each run: its log, exit status, the 'Audio check' lines,
the run-summary lines, any traceback, and the manifest's per-file state and detail.
Usage: python e2e_runs.py [run names]"""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
PY = sys.executable  # the Python running this script (the WhisperJAV environment)
E2E = Path(r"<work folder>\e2e")
GAP = str(E2E / "film_A_gap_clip.mp4")
CLEAN = str(E2E / "film_A_clean_clip.mp4")
ENS = ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-sensitivity", "aggressive",
       "--pass1-model", "litagin/anime-whisper",
       "--pass1-qwen-params", json.dumps({"framer": "vad-grouped", "generator_backend": "anime-whisper"}),
       "--pass2-pipeline", "qwen", "--pass2-model", "Qwen/Qwen3-ASR-1.7B"]
RUNS = {
    "1_qwen_gap": [GAP, "--mode", "qwen"],
    "2_qwen_clean": [CLEAN, "--mode", "qwen"],
    "3_balanced_gap": [GAP, "--mode", "balanced"],
    "4_fidelity_turbo_gap": [GAP, "--mode", "fidelity", "--model", "turbo"],
    "5_ensemble_gap": [GAP, *ENS],
    "6_qwen_gap_failon": [GAP, "--mode", "qwen", "--fail-on", "suspect"],
    "7_ensemble_gap_failon": [GAP, *ENS, "--fail-on", "suspect"],
    "8_balanced_async_gap": [GAP, "--mode", "balanced", "--async-processing"],
}

for name in (sys.argv[1:] or list(RUNS)):
    out = E2E / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    cmd = [PY, "-m", "whisperjav.main", *RUNS[name], "--output-dir", str(out), "--language", "japanese"]
    t0 = time.monotonic()
    with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=str(REPO)).returncode
    text = (out / "run.log").read_text(encoding="utf-8", errors="replace")
    print(f"\n===== {name}: exit {rc}, {time.monotonic() - t0:.0f}s", flush=True)
    for line in text.splitlines():
        if ("Audio check" in line or "audio integrity" in line or "Traceback" in line
                or "stopped before transcription" in line):
            print("  | " + line.strip()[:400], flush=True)
    manifests = list(out.rglob("whisperjav_run.json"))
    for m in manifests:
        try:
            d = json.loads(m.read_text(encoding="utf-8"))
            for f in d.get("files", []):
                print(f"  manifest: state={f.get('state')} detail={str(f.get('detail'))[:300]}", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  manifest {m.name}: unreadable ({exc})")
    if not manifests:
        print("  (no manifest found under the output folder)")
