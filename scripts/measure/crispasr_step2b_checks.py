"""CrispASR step 2b: the two checks that run through WhisperJAV's command line (CrispASR master plan 4.5, section 6;
U6, U12).

  cpu        Parakeet-CrispASR through `python -m whisperjav.main --ensemble`, with CUDA_VISIBLE_DEVICES=-1, so PyTorch
             sees no GPU and WhisperJAV's own build choice picks the CPU build, as on a PC without an NVIDIA card.
             --accept-cpu-mode answers the start-up GPU question in advance.
  japanese   Cohere-CrispASR (cohere-asr-ja-v0.1) through `python -m whisperjav.main --ensemble`, with the film, the
             output folder, the temporary folder and the model folder named in Japanese. The model reaches the run
             through WHISPERJAV_CRISPASR_MODEL_DIR: a Japanese-named folder holding a hard link to the file in the
             Hugging Face cache. Its path then has characters outside ASCII, and the plan 1.7 rule must hard-link it
             into the program's working folder. The temporary and model folders go under --c-folder, which must be on
             the drive of the Hugging Face cache, so both hard links are possible there, as on a PC whose user name is
             Japanese.

Both checks use one reference film (default: the shorter one) and keep their temporary files. Each is compared with
the scored run of the same configuration and film in the batch: its own SRT and its cut points.

Writes <batch>/checks/<check>/:
    console.log      everything the run printed
    check.json       command, environment changes, exit status, seconds, the files found, the comparison
    output/          the run's SRT (for "japanese", in the Japanese-named output folder named in check.json)
    run.log, cuts.json, out.json   copies of the program's own files from the kept temporary folder

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2b_checks.py cpu --batch <batch>
    python scripts/measure/crispasr_step2b_checks.py japanese --batch <batch> --c-folder <folder on the cache's drive>
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NETFLIX = REPO / "test_media" / "Ground_Truths" / "Netflix"
FILM = NETFLIX / "The.Naked.Director.S02E02.Scene.5.Bubble.More.mkv"

CHECKS = {
    "cpu": {"pipeline": "parakeet-crispasr", "model": None, "config": "parakeet-tdt-0.6b-ja_speech"},
    "japanese": {"pipeline": "cohere-crispasr", "model": "cohere-asr-ja-v0.1", "config": "cohere-asr-ja-v0.1_speech"},
}
JP_FOLDER = "日本語フォルダ"          # "Japanese folder"
JP_FILM = "裸の監督_S02E02_場面5"     # "The Naked Director S02E02 scene 5"
JP_OUTPUT = "出力"                   # "output"
JP_TEMP = "一時ファイル"              # "temporary files"
JP_MODELS = "モデル"                  # "models"


def srt_lines(path: Path) -> list:
    import srt
    try:
        return [(s.start.total_seconds(), s.end.total_seconds(), s.content)
                for s in srt.parse(path.read_text(encoding="utf-8", errors="replace"))]
    except Exception:
        return []


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def program_files(temp: Path) -> dict:
    """The program's own files in the kept temporary folder: every crispasr_out/<film>/ found under it."""
    found = {}
    for run_log in temp.rglob("run.log"):
        if run_log.parent.parent.name == "crispasr_out":
            for name in ("run.log", "cuts.json", "out.json"):
                if (run_log.parent / name).is_file():
                    found.setdefault(name, []).append(run_log.parent / name)
    return found


def compare(check_srt: Path, check_cuts, scored_folder: Path, film: Path) -> dict:
    scored_srt = scored_folder / f"{film.stem}.ja.whisperjav.srt"
    scored_cuts = scored_folder / f"{film.stem}.cuts.json"
    mine, theirs = srt_lines(check_srt), srt_lines(scored_srt) if scored_srt.is_file() else []
    same_text = [m[2] for m in mine] == [t[2] for t in theirs]
    return {"scored_run_srt": str(scored_srt) if scored_srt.is_file() else None,
            "subtitles": len(mine), "scored_run_subtitles": len(theirs),
            "same_subtitles": mine == theirs, "same_text_in_order": same_text,
            "same_cut_points": (read_json(Path(check_cuts)) == read_json(scored_cuts))
            if check_cuts and scored_cuts.is_file() else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2b_checks.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=list(CHECKS))
    ap.add_argument("--batch", required=True, type=Path, help="the step 2b batch folder")
    ap.add_argument("--film", type=Path, default=FILM, help=f"default: {FILM.name}")
    ap.add_argument("--c-folder", type=Path,
                    help="japanese only: a folder on the Hugging Face cache's drive for the temporary and model folders")
    a = ap.parse_args(argv)
    spec = CHECKS[a.check]
    batch, film = Path(os.path.abspath(a.batch)), Path(os.path.abspath(a.film))
    here = batch / "checks" / a.check
    if (here / "check.json").exists():
        print(f"refused: {here / 'check.json'} exists", file=sys.stderr)
        return 2
    here.mkdir(parents=True, exist_ok=True)
    env_changes = {"PYTHONIOENCODING": "utf-8"}

    if a.check == "cpu":
        run_film, output, temp = film, here / "output", here / "temp"
        env_changes["CUDA_VISIBLE_DEVICES"] = "-1"
        extra = ["--accept-cpu-mode"]
    else:
        if a.c_folder is None:
            ap.error("japanese needs --c-folder")
        from whisperjav.crispasr import install, manifest
        model = manifest.MODELS[spec["model"]]
        status = install.model_status(model)
        if not status.downloaded:
            print(f"refused: {model.filename} is not downloaded yet (run the scored runs first)", file=sys.stderr)
            return 2
        jp = here / JP_FOLDER
        jp.mkdir(parents=True, exist_ok=True)
        run_film = jp / f"{JP_FILM}{film.suffix}"
        if not run_film.exists():
            os.link(film, run_film)  # same drive as test_media
        output = here / JP_OUTPUT
        c_root = Path(os.path.abspath(a.c_folder)) / JP_FOLDER
        temp, models = c_root / JP_TEMP, c_root / JP_MODELS
        models.mkdir(parents=True, exist_ok=True)
        linked = models / model.filename
        if not linked.exists():
            os.link(os.path.realpath(status.path), linked)
        env_changes["WHISPERJAV_CRISPASR_MODEL_DIR"] = str(models)
        extra = []

    output.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "whisperjav.main", str(run_film), "--ensemble", "--pass1-pipeline", spec["pipeline"],
           *(["--pass1-model", spec["model"]] if spec["model"] else []), "--output-dir", str(output),
           "--temp-dir", str(temp), "--keep-temp", *extra]
    started = time.time()
    with open(here / "console.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, **env_changes), stdout=log,
                            stderr=subprocess.STDOUT).returncode
    seconds = round(time.time() - started, 1)

    srts = sorted(output.rglob("*.srt"))
    found = program_files(temp)
    copied = {}
    for name, paths in found.items():
        if len(paths) == 1:
            shutil.copy2(paths[0], here / name)
            copied[name] = str(here / name)
    record = {"plan": "CrispASR master plan 4.5, step 2b, check " + a.check, "command": cmd, "environment": env_changes,
              "exit_status": rc, "seconds": seconds, "film": str(run_film), "output_folder": str(output),
              "temporary_folder": str(temp), "srt_files": [str(p) for p in srts],
              "program_files_found": {k: [str(p) for p in v] for k, v in found.items()}, "copied": copied}
    if a.check == "japanese":
        record["model_folder"] = env_changes["WHISPERJAV_CRISPASR_MODEL_DIR"]
        record["model_path_is_ascii"] = env_changes["WHISPERJAV_CRISPASR_MODEL_DIR"].isascii()
        staged = [str(p) for p in temp.rglob("*.gguf")]
        record["model_files_in_temporary_folder"] = staged
        record["staged_model_link_count"] = {p: os.stat(p).st_nlink for p in staged}
    if len(srts) == 1:
        record["comparison"] = compare(srts[0], copied.get("cuts.json"), batch / spec["config"], film)
    (here / "check.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("exit_status", "seconds", "srt_files", "copied")}, ensure_ascii=False))
    print(json.dumps(record.get("comparison"), ensure_ascii=False))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
