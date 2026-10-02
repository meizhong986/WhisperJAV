"""ChronosJAV reference runs: Qwen3-ASR (balanced) and anime-whisper (aggressive), each with every speech segmenter
(WhisperSeg, TEN, FireRedVAD, Silero), run N times over a folder of reference clips. The command lines are the ones
the GUI's Ensemble tab builds for a pass-1-only run in the default timestamp mode (no aligner).

Optional recording: with --record, the speech segmenter recorder (segmenter_recorder/sitecustomize.py) is put on
PYTHONPATH, and each run also writes <run folder>/framing.jsonl: the scene list, the speech segmenter's speech
segments and groups per scene, and WhisperSeg's per-frame speech probabilities. The recorder only reads values.
With --compare-with <folder of an earlier batch>, each run's SRTs are compared byte for byte with that batch's run1,
which shows whether the recorder left the output unchanged.

Usage:
    python run_chronosjav_reference_runs.py --python <install>\\python.exe --media-dir <clips> --out <folder>
        [--runs 2] [--pairs qwen3_whisperseg anime_ten ...] [--record] [--compare-with <folder>]

Each run goes to <out>/<pair>/run<k>/ with command.json, run.log and the SRTs; a DONE marker lets an interrupted
batch resume. A summary is written to <out>/status.json.
"""
import argparse
import filecmp
import json
import os
import subprocess
import time
from pathlib import Path

GENERATORS = {
    "qwen3": ["--pass1-sensitivity", "balanced", "--pass1-model", "Qwen/Qwen3-ASR-1.7B",
              "--pass1-qwen-params", json.dumps({"framer": "vad-grouped"})],
    "anime": ["--pass1-sensitivity", "aggressive", "--pass1-model", "litagin/anime-whisper",
              "--pass1-qwen-params", json.dumps({"framer": "vad-grouped", "generator_backend": "anime-whisper",
                                                 "timestamp_mode": "vad_only", "assembly_cleaner": "passthrough",
                                                 "stepdown": False})],
}
SEGMENTERS = ["whisperseg", "ten", "firered-vad", "silero"]
PAIRS = [f"{g}_{s}" for g in GENERATORS for s in SEGMENTERS]
RECORDER = Path(__file__).resolve().parent / "segmenter_recorder"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="python.exe of the WhisperJAV install to run")
    ap.add_argument("--media-dir", required=True, type=Path, help="folder of reference clips (*.mkv)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--pairs", nargs="+", default=PAIRS, choices=PAIRS)
    ap.add_argument("--record", action="store_true", help="also record speech segmenter data (framing.jsonl)")
    ap.add_argument("--compare-with", type=Path, help="earlier batch folder; compare SRTs with its run1")
    a = ap.parse_args()

    media = [str(p) for p in sorted(a.media_dir.glob("*.mkv"))]
    status = []
    for run in range(1, a.runs + 1):
        for pair in a.pairs:
            gen, seg = pair.split("_", 1)
            out = a.out / pair / f"run{run}"
            if (out / "DONE").exists():
                continue
            out.mkdir(parents=True, exist_ok=True)
            cmd = [a.python, "-m", "whisperjav.main", *media, "--ensemble", "--pass1-pipeline", "qwen",
                   *GENERATORS[gen], "--pass1-scene-detector", "semantic", "--pass1-speech-segmenter", seg,
                   "--merge-strategy", "pass1_primary", "--output-dir", str(out),
                   "--subs-language", "native", "--language", "japanese"]
            (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
            env = dict(os.environ)
            if a.record:
                rec_file = out / "framing.jsonl"
                rec_file.unlink(missing_ok=True)
                env.update(PYTHONPATH=str(RECORDER), WJ_RECORD_OUT=str(rec_file))
            t0 = time.monotonic()
            with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
            text = (out / "run.log").read_text(encoding="utf-8", errors="replace")
            rec = {"pair": pair, "run": run, "exit": rc, "seconds": round(time.monotonic() - t0, 1),
                   "backend_seen": f"backend={seg}" in text, "no_aligner": "0.0% aligner" in text,
                   "n_srt": len(list(out.glob("*.srt")))}
            if a.compare_with:
                ref = a.compare_with / pair / "run1"
                names = sorted(p.name for p in ref.glob("*.srt"))
                rec["srts_identical"] = sum((out / n).exists() and filecmp.cmp(ref / n, out / n, shallow=False)
                                            for n in names)
                rec["srts_expected"] = len(names)
            status.append(rec)
            print(json.dumps(rec), flush=True)
            if rc == 0:
                (out / "DONE").write_text(json.dumps(rec), encoding="utf-8")
    (a.out / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")
    print("finished", flush=True)


if __name__ == "__main__":
    main()
