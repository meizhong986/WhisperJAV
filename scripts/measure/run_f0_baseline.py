"""1.10.0 baseline (item 1, F0): six pipelines at their defaults, sensitivity aggressive, over the two reference
films, one run each. Later batches (new stack, transformers 5) run the same commands in another install and are
compared with this one.

Pipelines (folder names): balanced, fast, faster, fidelity, chronos_qwen3, chronos_anime. The two ChronosJAV command
lines are what the 1.9.5 GUI builds for a pass-1-only Ensemble run (webview_gui/api.py _build_twopass_args):
Qwen3-ASR with the GUI's pass-1 defaults (FireRedVAD), anime-whisper with its pass-2 defaults (TEN) run as pass 1.

Usage:
    python run_f0_baseline.py --python D:\\195\\python.exe --out D:\\WJ110_measure\\F0_195 [--pipelines balanced ...]

Each command runs with its output folder as working folder, so `python -m whisperjav.main` loads the WhisperJAV
installed in --python, never a repository checkout. Each pipeline goes to <out>/<pipeline>/ with command.json,
run.log and the SRTs; a DONE marker lets an interrupted batch resume. <out>/env.txt records the install's package
versions, where its whisperjav was loaded from, and the GPU; <out>/status.json the exit code, seconds and SRT files
of each command.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MEDIA_DIR = REPO / "test_media" / "Ground_Truths" / "Netflix"
FILMS = ["The.Naked.Director.S01E04.Scene.4.mkv", "The.Naked.Director.S02E02.Scene.5.Bubble.More.mkv"]

_CHRONOS = ["--ensemble", "--pass1-pipeline", "qwen", "--pass1-sensitivity", "aggressive"]
_CHRONOS_TAIL = ["--pass1-scene-detector", "semantic"]
_COMMON_TAIL = ["--merge-strategy", "pass1_primary", "--subs-language", "native", "--language", "japanese"]
PIPELINES = {m: ["--mode", m, "--sensitivity", "aggressive"] for m in ("balanced", "fast", "faster", "fidelity")}
PIPELINES["chronos_qwen3"] = [
    *_CHRONOS, "--pass1-qwen-params", json.dumps({"framer": "vad-grouped"}), *_CHRONOS_TAIL,
    "--pass1-speech-segmenter", "firered-vad", "--pass1-model", "Qwen/Qwen3-ASR-1.7B", *_COMMON_TAIL,
]
PIPELINES["chronos_anime"] = [
    *_CHRONOS, "--pass1-qwen-params", json.dumps({"framer": "vad-grouped", "generator_backend": "anime-whisper",
                                                  "timestamp_mode": "vad_only", "assembly_cleaner": "passthrough",
                                                  "stepdown": False}),
    *_CHRONOS_TAIL, "--pass1-speech-segmenter", "ten", "--pass1-model", "litagin/anime-whisper", *_COMMON_TAIL,
]
BASELINE = list(PIPELINES)

# Item 3 (transformers 5) extra comparisons, run only when named with --pipelines: the parts transformers 5 and
# huggingface-hub 1.x touch that the six baseline commands do not (WhisperSeg, Transformers mode, the enhancers).
_QWEN_FIRERED = PIPELINES["chronos_qwen3"]
PIPELINES["chronos_qwen3_whisperseg"] = [a if a != "firered-vad" else "whisperseg" for a in _QWEN_FIRERED]
PIPELINES["transformers"] = ["--mode", "transformers", "--sensitivity", "aggressive"]
for _name, _enh in (("zipenhancer", "zipenhancer"), ("zipenhancer_onnx", "zipenhancer:onnx"),
                    ("htdemucs", "htdemucs"), ("bs_roformer", "bs-roformer")):
    PIPELINES[f"chronos_qwen3_{_name}"] = [*_QWEN_FIRERED, "--pass1-speech-enhancer", _enh]

PROBE = r"""
import importlib.metadata as md, importlib.util, sys
print("python", sys.version.split()[0], sys.executable)
for n in ("whisperjav", "torch", "torchaudio", "transformers", "ctranslate2", "faster-whisper", "openai-whisper",
          "stable-ts", "qwen-asr", "llama-cpp-python", "huggingface-hub"):
    try:
        print(f"{n:18} {md.version(n)}")
    except md.PackageNotFoundError:
        print(f"{n:18} (not installed)")
print("whisperjav loaded from", list(importlib.util.find_spec("whisperjav").submodule_search_locations))
print("whisperjav direct_url.json", md.distribution("whisperjav").read_text("direct_url.json"))
"""


def write_env(python, out):
    r = subprocess.run([python, "-c", PROBE], cwd=out, capture_output=True, text=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True)
    text = r.stdout + r.stderr + "gpu " + gpu.stdout
    (out / "env.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="python.exe of the WhisperJAV install to run")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--media-dir", type=Path, default=MEDIA_DIR, help=f"folder holding the two films (default {MEDIA_DIR})")
    ap.add_argument("--pipelines", nargs="+", default=BASELINE, choices=list(PIPELINES))
    a = ap.parse_args()

    media = [a.media_dir / f for f in FILMS]
    missing = [str(m) for m in media if not m.exists()]
    if missing:
        raise SystemExit(f"films not found: {missing}")
    a.out.mkdir(parents=True, exist_ok=True)
    write_env(a.python, a.out)

    status_path = a.out / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {}
    for name in a.pipelines:
        out = a.out / name
        if (out / "DONE").exists():
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [a.python, "-m", "whisperjav.main", *map(str, media), *PIPELINES[name], "--output-dir", str(out)]
        (out / "command.json").write_text(json.dumps(cmd, indent=1), encoding="utf-8")
        t0 = time.monotonic()
        with open(out / "run.log", "w", encoding="utf-8", errors="replace") as log:
            rc = subprocess.run(cmd, cwd=out, stdout=log, stderr=subprocess.STDOUT).returncode
        srts = sorted(p.name for p in out.glob("*.srt"))
        per_film = {f: sum(1 for s in srts if s.startswith(Path(f).stem + ".")) for f in FILMS}
        rec = {"exit": rc, "seconds": round(time.monotonic() - t0, 1), "srt_files": srts, "srt_per_film": per_film}
        status[name] = rec
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
        print(json.dumps({name: rec}), flush=True)
        if rc == 0 and all(n == 1 for n in per_film.values()):
            (out / "DONE").write_text(json.dumps(rec), encoding="utf-8")


if __name__ == "__main__":
    main()
