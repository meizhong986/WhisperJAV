"""CrispASR step 2b: the checks that run through WhisperJAV's command line (CrispASR master plan 4.5, section 6;
U6, U12), and the follow-up checks of the step 2 report's item 10.

  cpu        Parakeet-CrispASR through `python -m whisperjav.main --ensemble`, with CUDA_VISIBLE_DEVICES=-1, so PyTorch
             sees no GPU and WhisperJAV's own build choice picks the CPU build, as on a PC without an NVIDIA card.
             --accept-cpu-mode answers the start-up GPU question in advance.
  cpu-cohere Cohere-CrispASR (cohere-asr-ja-v0.1) on the CPU build, the same way as cpu (item 10 (d)).
  japanese   Cohere-CrispASR (cohere-asr-ja-v0.1) through `python -m whisperjav.main --ensemble`, with the film, the
             output folder, the temporary folder and the model folder named in Japanese. The model reaches the run
             through WHISPERJAV_CRISPASR_MODEL_DIR: a Japanese-named folder holding a hard link to the file in the
             Hugging Face cache. Its path then has characters outside ASCII, and the plan 1.7 rule must hard-link it
             into the program's working folder. The temporary and model folders go under --c-folder, which must be on
             the drive of the Hugging Face cache, so both hard links are possible there, as on a PC whose user name is
             Japanese.
  jpuser     Parakeet-CrispASR through `python -m whisperjav.main --ensemble` on both reference films in one pass, with
             the user folder (USERPROFILE) moved to --user-folder, a Japanese-named folder, and --temp-dir inside it
             (item 10 (a)). The program, the Hugging Face cache and the temporary folder then all sit under Japanese
             names, as for a Windows user whose user name is Japanese; the Python environment stays where it is, and
             what the run downloads goes into that folder. Run it twice with the same --user-folder and another
             --label (before and after a change): the second run meets what the first left in the temporary folder.

The single-film checks use one reference film (default: the shorter one); every check keeps its temporary files. Each
film is compared with the scored run of the same configuration and film in the batch: its own SRT and its cut points.

Writes <batch>/checks/<check>/ (with --label: <check>-<label>/):
    console.log      everything the run printed
    check.json       command, environment changes, exit status, seconds, the files found, the comparison
    output/          the run's SRT (for "japanese", in the Japanese-named output folder named in check.json)
    run.log, cuts.json, out.json   copies of the program's own files from the kept temporary folder
                     (jpuser: one folder per film, named as the film's run folder)
jpuser's check.json also names the user folder, the Hugging Face cache and the program folder the run used, the -m
each film's program run was given, and, for every model file in the cache and in the temporary folder, whether it is a
link, where it leads and whether it can be read.

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2b_checks.py cpu --batch <batch>
    python scripts/measure/crispasr_step2b_checks.py cpu-cohere --batch <batch>
    python scripts/measure/crispasr_step2b_checks.py japanese --batch <batch> --c-folder <folder on the cache's drive>
    python scripts/measure/crispasr_step2b_checks.py jpuser --batch <batch> --user-folder <folder> --label before
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
FILMS = (NETFLIX / "The.Naked.Director.S01E04.Scene.4.mkv", FILM)  # jpuser: both, in one pass

CHECKS = {
    "cpu": {"pipeline": "parakeet-crispasr", "model": None, "config": "parakeet-tdt-0.6b-ja_speech"},
    "cpu-cohere": {"pipeline": "cohere-crispasr", "model": "cohere-asr-ja-v0.1", "config": "cohere-asr-ja-v0.1_speech"},
    "japanese": {"pipeline": "cohere-crispasr", "model": "cohere-asr-ja-v0.1", "config": "cohere-asr-ja-v0.1_speech"},
    "jpuser": {"pipeline": "parakeet-crispasr", "model": None, "config": "parakeet-tdt-0.6b-ja_speech"},
}
CPU_CHECKS = ("cpu", "cpu-cohere")
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


def run_log_head(path: Path) -> dict:
    """The 'key: value' lines that open the program's run.log (command, working folder, exit code, outcome, seconds)."""
    head = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("-----"):
                break
            key, sep, value = line.partition(": ")
            if sep:
                head[key.strip()] = value.strip()
    except OSError:
        pass
    return head


def model_argument(command: str):
    """What the program's -m was given, read from run.log's command line."""
    parts = command.split()
    return parts[parts.index("-m") + 1] if "-m" in parts[:-1] else None


def model_files(root: Path) -> list:
    """Every *.gguf entry under root, a link that leads nowhere included."""
    found = []
    for folder, _, names in os.walk(root):
        found += [Path(folder) / n for n in names if n.lower().endswith(".gguf")]
    return sorted(found)


def file_facts(path: Path) -> dict:
    """Whether a model file is a link, where it leads, its link count, and whether its bytes can be read."""
    facts = {"path": str(path), "is_symbolic_link": os.path.islink(path)}
    if facts["is_symbolic_link"]:
        try:
            facts["link_target"] = os.readlink(path)
        except OSError as e:
            facts["link_target"] = f"unreadable: {e}"
        facts["leads_to"] = os.path.realpath(path)
        facts["leads_to_a_file"] = os.path.isfile(facts["leads_to"])
    try:
        facts["hard_links"] = os.lstat(path).st_nlink
    except OSError as e:
        facts["hard_links"] = f"unreadable: {e}"
    try:
        with open(path, "rb") as fh:
            facts["first_bytes"] = fh.read(4).decode("ascii", "replace")  # a GGUF file begins with "GGUF"
        facts["readable"], facts["size"] = True, os.path.getsize(path)
    except OSError as e:
        facts["readable"], facts["read_error"] = False, f"{type(e).__name__}: {e}"
    return facts


def film_runs(temp: Path, films: list, here: Path, since: float) -> list:
    """Per film: its run folder, what the program's run.log says, and copies of the program's files."""
    from whisperjav.crispasr.runner import film_run_dir

    runs = []
    for film in films:
        candidates = [film_run_dir(out.parent, film.stem) for out in temp.rglob("crispasr_out")]
        run_dir = next((c for c in candidates if c.is_dir()), candidates[0] if candidates else None)
        entry = {"film": film.name, "run_folder": str(run_dir) if run_dir else None, "copied": {},
                 "written_by_this_run": {}}
        if run_dir and run_dir.is_dir():
            dest = here / run_dir.name
            dest.mkdir(exist_ok=True)
            for name in ("run.log", "cuts.json", "out.json"):
                src = run_dir / name
                if src.is_file():
                    shutil.copy2(src, dest / name)
                    entry["copied"][name] = str(dest / name)
                    entry["written_by_this_run"][name] = src.stat().st_mtime >= since - 2.0
            head = run_log_head(run_dir / "run.log")
            entry.update({"exit_code": head.get("exit code"), "outcome": head.get("outcome"),
                          "seconds": head.get("seconds"), "model_argument": model_argument(head.get("command", "")),
                          "program": (head.get("command", "").split() or [None])[0]})
        runs.append(entry)
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2b_checks.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("check", choices=list(CHECKS))
    ap.add_argument("--batch", required=True, type=Path, help="the step 2b batch folder")
    ap.add_argument("--film", type=Path, default=FILM, help=f"default: {FILM.name} (jpuser: both reference films)")
    ap.add_argument("--c-folder", type=Path,
                    help="japanese only: a folder on the Hugging Face cache's drive for the temporary and model folders")
    ap.add_argument("--user-folder", type=Path,
                    help="jpuser only: the run's user folder (USERPROFILE), with a name outside ASCII")
    ap.add_argument("--label", help="added to the check's folder name: checks/<check>-<label>/")
    a = ap.parse_args(argv)
    spec = CHECKS[a.check]
    batch, film = Path(os.path.abspath(a.batch)), Path(os.path.abspath(a.film))
    here = batch / "checks" / (f"{a.check}-{a.label}" if a.label else a.check)
    if (here / "check.json").exists():
        print(f"refused: {here / 'check.json'} exists", file=sys.stderr)
        return 2
    env_changes = {"PYTHONIOENCODING": "utf-8"}
    films, user = [film], None

    if a.check in CPU_CHECKS:
        here.mkdir(parents=True, exist_ok=True)
        run_films, output, temp = [film], here / "output", here / "temp"
        env_changes["CUDA_VISIBLE_DEVICES"] = "-1"
        extra = ["--accept-cpu-mode"]
    elif a.check == "jpuser":
        if a.user_folder is None:
            ap.error("jpuser needs --user-folder")
        user = Path(os.path.abspath(a.user_folder))
        if user.name.isascii():
            ap.error("--user-folder must have a name outside ASCII")
        here.mkdir(parents=True, exist_ok=True)
        user.mkdir(parents=True, exist_ok=True)
        films = [Path(os.path.abspath(f)) for f in FILMS]
        run_films, output = films, here / "output"
        temp = user / "AppData" / "Local" / "Temp" / "whisperjav"  # where a real user's %TEMP% would put it
        env_changes["USERPROFILE"] = str(user)
        extra = []
    else:
        if a.c_folder is None:
            ap.error("japanese needs --c-folder")
        from whisperjav.crispasr import install, manifest
        model = manifest.MODELS[spec["model"]]
        status = install.model_status(model)
        if not status.downloaded:
            print(f"refused: {model.filename} is not downloaded yet (run the scored runs first)", file=sys.stderr)
            return 2
        here.mkdir(parents=True, exist_ok=True)
        jp = here / JP_FOLDER
        jp.mkdir(parents=True, exist_ok=True)
        run_film = jp / f"{JP_FILM}{film.suffix}"
        if not run_film.exists():
            os.link(film, run_film)  # same drive as test_media
        run_films = [run_film]
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
    cmd = [sys.executable, "-m", "whisperjav.main", *map(str, run_films), "--ensemble",
           "--pass1-pipeline", spec["pipeline"], *(["--pass1-model", spec["model"]] if spec["model"] else []),
           "--output-dir", str(output), "--temp-dir", str(temp), "--keep-temp", *extra]
    started = time.time()
    with open(here / "console.log", "w", encoding="utf-8", errors="replace") as log:
        rc = subprocess.run(cmd, cwd=REPO, env=dict(os.environ, **env_changes), stdout=log,
                            stderr=subprocess.STDOUT).returncode
    seconds = round(time.time() - started, 1)

    srts = sorted(output.rglob("*.srt"))
    record = {"plan": "CrispASR master plan 4.5, step 2b, check " + a.check, "command": cmd, "environment": env_changes,
              "exit_status": rc, "seconds": seconds, "films": [str(p) for p in run_films],
              "output_folder": str(output), "temporary_folder": str(temp), "srt_files": [str(p) for p in srts]}
    if a.check == "jpuser":
        hub = user / ".cache" / "huggingface" / "hub"
        record["user_folder"] = str(user)
        record["hugging_face_cache"] = str(hub)
        record["program_folder"] = str(user / ".cache" / "whisperjav" / "crispasr")
        record["film_runs"] = film_runs(temp, films, here, started)
        record["model_files_in_the_cache"] = [file_facts(p) for p in model_files(hub)]
        record["model_files_in_the_temporary_folder"] = [file_facts(p) for p in model_files(temp)]
        record["comparison"] = {}
        for f, run in zip(films, record["film_runs"]):
            mine = [p for p in srts if p.name.startswith(f.stem + ".")]
            if len(mine) == 1:
                cuts = run["copied"].get("cuts.json") if run["written_by_this_run"].get("cuts.json") else None
                record["comparison"][f.name] = compare(mine[0], cuts, batch / spec["config"], f)
    else:
        found = program_files(temp)
        copied = {}
        for name, paths in found.items():
            if len(paths) == 1:
                shutil.copy2(paths[0], here / name)
                copied[name] = str(here / name)
        record["program_files_found"] = {k: [str(p) for p in v] for k, v in found.items()}
        record["copied"] = copied
        if a.check == "japanese":
            record["model_folder"] = env_changes["WHISPERJAV_CRISPASR_MODEL_DIR"]
            record["model_path_is_ascii"] = env_changes["WHISPERJAV_CRISPASR_MODEL_DIR"].isascii()
            staged = [str(p) for p in temp.rglob("*.gguf")]
            record["model_files_in_temporary_folder"] = staged
            record["staged_model_link_count"] = {p: os.stat(p).st_nlink for p in staged}
        if len(srts) == 1:
            record["comparison"] = compare(srts[0], copied.get("cuts.json"), batch / spec["config"], film)
    (here / "check.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("exit_status", "seconds", "srt_files")}, ensure_ascii=False))
    print(json.dumps(record.get("comparison"), ensure_ascii=False))
    for run in record.get("film_runs", []):
        print(json.dumps({k: run.get(k) for k in ("film", "outcome", "exit_code", "model_argument")}, ensure_ascii=False))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
