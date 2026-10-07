"""Reference runs: Balanced, Fidelity and the GUI's default two-pass run, each at its defaults, run N times over a
folder of reference clips. Used to record what an installed WhisperJAV version produces, so a later version can be
compared with it file by file.

Usage:
    python run_pipeline_reference_runs.py --python <install>\\python.exe --media-dir <clips> --out <folder>
        [--runs 2] [--configs balanced fidelity twopass_gui_default]

Each run goes to <out>/<config>/run<k>/ with command.json, run.log and the SRTs; a DONE marker lets an interrupted
batch resume. A summary is written to <out>/status.json.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

CONFIGS = {m: ["--mode", m] for m in ("balanced", "fidelity")}
CONFIGS["twopass_gui_default"] = [
    "--ensemble",
    "--pass1-pipeline", "qwen", "--pass1-sensitivity", "aggressive", "--pass1-scene-detector", "semantic",
    "--pass1-speech-segmenter", "whisperseg", "--pass1-model", "litagin/anime-whisper",
    "--pass1-qwen-params", json.dumps({"framer": "vad-grouped", "generator_backend": "anime-whisper",
                                       "timestamp_mode": "vad_only", "assembly_cleaner": "passthrough",
                                       "stepdown": False}),
    "--pass2-pipeline", "qwen", "--pass2-sensitivity", "balanced", "--pass2-scene-detector", "semantic",
    "--pass2-speech-segmenter", "ten", "--pass2-model", "Qwen/Qwen3-ASR-1.7B",
    "--pass2-qwen-params", json.dumps({"framer": "vad-grouped"}),
    "--merge-strategy", "pass1_primary", "--subs-language", "native", "--language", "japanese",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="python.exe of the WhisperJAV install to run")
    ap.add_argument("--media-dir", required=True, type=Path, help="folder of reference clips (*.mkv)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS), choices=list(CONFIGS))
    a = ap.parse_args()

    media = [str(p) for p in sorted(a.media_dir.glob("*.mkv"))]
    status = []
    for run in range(1, a.runs + 1):
        for name in a.configs:
            out = a.out / name / f"run{run}"
            if (out / "DONE").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [a.python, "-m", "whisperjav.main", *media, *CONFIGS[name], "--output-dir", str(out)]
            (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
            t0 = time.monotonic()
            with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            rec = {"config": name, "run": run, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
                   "files": sorted(p.name for p in out.glob("*.srt"))}
            status.append(rec)
            print(json.dumps(rec), flush=True)
            if rc == 0:
                (out / "DONE").write_text(json.dumps(rec), encoding="utf-8")
    (a.out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
