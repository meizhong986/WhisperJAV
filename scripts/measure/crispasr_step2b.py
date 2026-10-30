"""CrispASR step 2b: one film, one configuration, one CrispASR program run, and every line-building and
post-processing variant built from that same run (CrispASR master plan 3.3, 3.6, 3.7, 4.5).

A configuration is a pipeline, a model and a slice form. The six of plan 4.5:

    --pipeline parakeet-crispasr --model parakeet-tdt-0.6b-ja --slice-form speech
    --pipeline parakeet-crispasr --model parakeet-tdt-0.6b-ja --slice-form windows
    --pipeline parakeet-crispasr --model parakeet-ctc-1.1b-ja --slice-form speech
    --pipeline cohere-crispasr   --model cohere-asr-ja-v0.1   --slice-form speech
    --pipeline cohere-crispasr   --model cohere-asr-ja-v0.1   --slice-form windows
    --pipeline cohere-crispasr   --model cohere-asr-ja        --slice-form speech

"speech" is each pipeline's own slice rule: speech groups of at most 12 s (Parakeet) or 30 s (Cohere). "windows"
cuts each scene into equal windows of at most that length. The Parakeet CTC 1.1B is measured here but not offered
to users: only a pipeline built in code reaches it, never WhisperJAV's command line. --max-slice-s gives the pipeline
another longest slice than its own, for one comparison of the step 2 report (item 10 (c): Cohere v0.1 on speech groups
of at most 12 s); the configuration's folder name then ends in _max<seconds>s.

What the script does:
  1. builds the pipeline as an Ensemble pass does, with the model and slice form above and its temporary files
     kept. Every other setting is the pipeline's own default; the script adds no option to WhisperJAV;
  2. calls process() once: the program and the model are made ready (downloaded if missing), and the CrispASR
     program runs once over the film;
  3. from that same run (the pipeline's last_run), builds the lines again with the pipeline's own line building and
     post-processing, through the constructor arguments the pipelines have for step 2b:
       lines            regroup (REGROUP_JAV over the words); segments (Parakeet's own segments, Parakeet only)
       post-processing  none; text (the non-verbal line filter and the non-linguistic utterance filter);
                        text+overlap (those two, then the scene-overlap resolver)

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2b.py <film> --pipeline <p> --model <m> --slice-form <s> --out <batch>

Output, in <batch>/<model>_<slice form>/ (both films of a configuration share it; every file name starts with the
film's name):
    <film>.ja.whisperjav.srt          the pipeline's own SRT, with its default settings
    lines-<l>_post-<p>/<film>.lines-<l>_post-<p>.srt
                                      one folder per variant, so `python -m subeval score --test-dir <folder>`
                                      finds exactly one SRT per film in a variant folder or in the configuration
                                      folder itself
    <film>.run.json                   the configuration, the pinned model, the program build, the program run
                                      (outcome, exit code, seconds, warnings), the slices, the variants written,
                                      WhisperJAV's git commit and its uncommitted files
    <film>.run.log, <film>.cuts.json  copies of the program's own log and of the cut points it was given
    temp/                             the pipeline's temporary files, kept: the audio, the scenes, and the
                                      program's run folder with its JSON output

--dry-run makes the same checks and prints what would run; it starts, downloads and writes nothing. An existing
<film>.run.json stops the run unless --overwrite is given. Exit status: 0 measured; 1 the run failed (the reason is
in run.json); 2 refused (arguments, a missing film, already measured).

The program runs are step 2b of the plan, which needs the owner's go (plan 4.5).
"""
import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from whisperjav.crispasr import manifest

# The two 1.10.0 CrispASR pipelines. Imported only for a real run: they load stable-ts and PyTorch, which --help
# and --dry-run do not need.
PIPELINE_CLASSES = {
    "parakeet-crispasr": ("whisperjav.pipelines.parakeet_crispasr_pipeline", "ParakeetCrispASRPipeline"),
    "cohere-crispasr": ("whisperjav.pipelines.cohere_crispasr_pipeline", "CohereCrispASRPipeline"),
}

# The pipelines' own SLICE_FORMS and LINE_MODES, written here so that --help and --dry-run import no pipeline.
# A real run checks them against the classes before it starts.
SLICE_FORMS = ("speech", "windows")
LINE_MODES = {"parakeet-crispasr": ("regroup", "segments"), "cohere-crispasr": ("regroup",)}

# The post-processing candidates of plan 3.6, by plain name; each is a value of the pipelines' postprocess argument.
POSTPROCESS_SETS = {
    "none": (),
    "text": ("nonverbal", "nonlinguistic"),
    "text+overlap": ("nonverbal", "nonlinguistic", "overlap"),
}

# Plan 4.5: (pipeline, model, slice form) of the twelve Q12 runs, each over the two reference films.
STEP_2B_CONFIGURATIONS = (
    ("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "speech"),
    ("parakeet-crispasr", "parakeet-tdt-0.6b-ja", "windows"),
    ("parakeet-crispasr", "parakeet-ctc-1.1b-ja", "speech"),
    ("cohere-crispasr", "cohere-asr-ja-v0.1", "speech"),
    ("cohere-crispasr", "cohere-asr-ja-v0.1", "windows"),
    ("cohere-crispasr", "cohere-asr-ja", "speech"),
)

SLICE_WORDS = {"speech": "speech groups from the speech segmenter (the pipeline's own slice rule)",
               "windows": "equal windows over the scenes, at most the engine's window long"}
LINE_WORDS = {"regroup": "lines built with REGROUP_JAV over the words",
              "segments": "one line per segment the engine returned"}
POSTPROCESS_WORDS = {"none": "no post-processing",
                     "text": "the non-verbal line filter and the non-linguistic utterance filter",
                     "text+overlap": "the non-verbal line filter, the non-linguistic utterance filter, "
                                     "then the scene-overlap resolver"}


class Refused(Exception):
    """The run does not start; nothing has been started or written."""


@dataclass(frozen=True)
class Config:
    pipeline: str
    model: manifest.ModelFile
    slice_form: str
    film: Path
    out: Path
    overwrite: bool = False
    dry_run: bool = False
    argv: Tuple[str, ...] = ()
    max_slice_s: Optional[float] = None  # --max-slice-s; None: the pipeline's own longest slice

    @property
    def label(self) -> str:
        return config_label(self.model.name, self.slice_form, self.max_slice_s)

    @property
    def basename(self) -> str:
        return self.film.stem  # as WhisperJAV's media discovery names a film

    @property
    def folder(self) -> Path:
        return self.out / self.label

    @property
    def temp_dir(self) -> Path:
        return self.folder / "temp"


def config_label(model_name: str, slice_form: str, max_slice_s: Optional[float] = None) -> str:
    """The configuration's folder name: the model (which names its pipeline), the slice form, and a longest slice
    other than the pipeline's own when one is given."""
    return f"{model_name}_{slice_form}" + ("" if max_slice_s is None else f"_max{max_slice_s:g}s")


def variant_name(line_mode: str, post: str) -> str:
    return f"lines-{line_mode}_post-{post}"


def variants_for(pipeline: str) -> List[Tuple[str, str, str, Tuple[str, ...]]]:
    """(name, line mode, post-processing name, post-processing steps) of every variant, lines first."""
    return [(variant_name(line_mode, post), line_mode, post, steps)
            for line_mode in LINE_MODES[pipeline] for post, steps in POSTPROCESS_SETS.items()]


def planned_files(cfg: Config) -> Dict:
    folder, film = cfg.folder, cfg.basename
    return {
        "pipeline_srt": folder / f"{film}.ja.whisperjav.srt",  # the pipeline's own name for it
        "variants": {name: folder / name / f"{film}.{name}.srt" for name, _, _, _ in variants_for(cfg.pipeline)},
        "run_json": folder / f"{film}.run.json",
        "run_log": folder / f"{film}.run.log",
        "cuts": folder / f"{film}.cuts.json",
    }


def constructor_arguments(cfg: Config) -> Dict:
    """What the script gives the pipeline's constructor; everything else is the pipeline's own default."""
    return {"output_dir": str(cfg.folder), "temp_dir": str(cfg.temp_dir), "keep_temp_files": True,
            "model_name": cfg.model.name, "slice_form": cfg.slice_form}


def say(text: str = "", stream=None) -> None:
    stream = stream or sys.stdout
    try:
        print(text, file=stream, flush=True)
    except UnicodeEncodeError:  # a console that cannot show a Japanese file name
        encoding = getattr(stream, "encoding", None) or "ascii"
        print(text.encode(encoding, "replace").decode(encoding, "replace"), file=stream, flush=True)


# ---------------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------------

def _model_help() -> str:
    parts = []
    for pipeline in manifest.CRISPASR_PIPELINES:
        default = manifest.PIPELINE_DEFAULT_MODEL[pipeline]
        names = [f"{m} (default)" if m == default else m for m in manifest.PIPELINE_MODELS[pipeline]]
        names += [f"{m} (measured only, not offered to users)" for m in manifest.MEASUREMENT_MODELS[pipeline]]
        parts.append(f"{pipeline}: {', '.join(names)}" + ("" if default else "; no default, name one"))
    return "the model's short name. " + ". ".join(parts)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crispasr_step2b.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("film", type=Path, help="the film: one media file")
    ap.add_argument("--pipeline", required=True, choices=list(PIPELINE_CLASSES), help="the CrispASR pipeline")
    ap.add_argument("--model", help=_model_help())
    ap.add_argument("--slice-form", required=True, choices=SLICE_FORMS,
                    help="speech: the pipeline's own speech groups; windows: equal windows over the scenes")
    ap.add_argument("--max-slice-s", type=float, metavar="SECONDS",
                    help="the longest slice instead of the pipeline's own (12 s Parakeet, 30 s Cohere); the folder "
                         "name then ends in _max<seconds>s")
    ap.add_argument("--out", required=True, type=Path,
                    help="the batch folder; this configuration's files go into <out>/<model>_<slice form>/")
    ap.add_argument("--overwrite", action="store_true",
                    help="measure again although <film>.run.json exists; this film's earlier files in the "
                         "configuration folder are deleted first")
    ap.add_argument("--dry-run", action="store_true",
                    help="make the checks and print what would run; start, download and write nothing")
    return ap


def parse_args(argv: Optional[List[str]] = None) -> Config:
    """The configuration, or exit status 2 with the reason (argparse)."""
    ap = build_parser()
    a = ap.parse_args(argv)
    allowed = manifest.PIPELINE_MODELS[a.pipeline] + manifest.MEASUREMENT_MODELS[a.pipeline]
    try:
        model = manifest.model_for(a.pipeline, a.model, include_measurement=True)
    except ValueError:
        if a.model:
            ap.error(f"'{a.model}' is not a model of {a.pipeline}; choose one of: {', '.join(allowed)}")
        ap.error(f"{a.pipeline} has no default model; name one with --model: {', '.join(allowed)}")
    film = Path(os.path.abspath(a.film))
    if not film.is_file():
        ap.error(f"the film was not found: {film}")
    if a.max_slice_s is not None and not a.max_slice_s > 0:
        ap.error(f"--max-slice-s must be more than 0 seconds, not {a.max_slice_s:g}")
    return Config(pipeline=a.pipeline, model=model, slice_form=a.slice_form, film=film,
                  out=Path(os.path.abspath(a.out)), overwrite=a.overwrite, dry_run=a.dry_run,
                  argv=tuple(sys.argv[1:] if argv is None else argv), max_slice_s=a.max_slice_s)


def refuse_if_measured(cfg: Config) -> None:
    run_json = planned_files(cfg)["run_json"]
    if run_json.exists() and not cfg.overwrite:
        raise Refused(f"{cfg.film.name} is already measured in this configuration ({run_json} exists); "
                      f"give --overwrite to measure it again")


def clear_previous(cfg: Config) -> None:
    """With --overwrite: this film's earlier files in the configuration folder, so none can pass as this run's."""
    files = planned_files(cfg)
    for path in (files["pipeline_srt"], *files["variants"].values(), files["run_json"], files["run_log"],
                 files["cuts"]):
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------
# --dry-run
# ---------------------------------------------------------------------------------------------------

def describe(cfg: Config) -> List[str]:
    """What a run would do, from the arguments and the manifest alone."""
    m, files = cfg.model, planned_files(cfg)
    class_name = PIPELINE_CLASSES[cfg.pipeline][1]
    offered = m.name in manifest.PIPELINE_MODELS[cfg.pipeline]
    builds = ", ".join(b.label for b in manifest.builds_for_this_machine()) or "none for this computer"

    def rel(path: Path) -> str:
        return os.path.relpath(path, cfg.folder)

    def shown(key: str, value) -> str:  # the two folders by name; they are listed in full under "would write"
        if key in ("output_dir", "temp_dir"):
            return f"{key}=<configuration folder>" + ("" if key == "output_dir" else f"{os.sep}{rel(value)}")
        return f"{key}={value!r}"

    arguments = ", ".join(shown(k, v) for k, v in constructor_arguments(cfg).items())

    lines = [
        "Dry run: nothing is started, downloaded or written.",
        "",
        f"configuration  {cfg.label}",
        f"pipeline       {cfg.pipeline} ({class_name})",
        f"model          {m.name}: {m.label}, back end {m.backend}, {m.size / 1e6:,.0f} MB"
        + ("" if offered else "; measured in step 2b only, not offered to users"),
        f"               {m.repo_id}, file {m.filename}, revision {m.revision}",
        f"slices         {cfg.slice_form}: {SLICE_WORDS[cfg.slice_form]}"
        + ("" if cfg.max_slice_s is None else f"; at most {cfg.max_slice_s:g} s (--max-slice-s), not the "
                                              f"pipeline's own longest slice"),
        f"film           {cfg.film}",
        f"program        CrispASR {manifest.CRISPASR_VERSION}; builds for this computer: {builds}. The pipeline",
        "               chooses the build when it runs, and downloads the program and the model if they are missing.",
        "",
        f"would build    {class_name}({arguments})",
        "               every other setting is the pipeline's own default",
        "would run      process() once: one CrispASR program run over the film",
        f"would rebuild  {len(files['variants'])} variants from that one run:",
    ]
    for name, line_mode, post, _ in variants_for(cfg.pipeline):
        lines.append(f"                 {name}: {LINE_WORDS[line_mode]}; {POSTPROCESS_WORDS[post]}")
    lines += ["", f"would write    {cfg.folder}{os.sep}",
              f"                 {rel(files['pipeline_srt'])}   (the pipeline's own SRT)"]
    lines += [f"                 {rel(path)}" for path in files["variants"].values()]
    lines += [f"                 {rel(files['run_json'])}, {rel(files['run_log'])}, {rel(files['cuts'])}",
              f"                 {rel(cfg.temp_dir)}{os.sep}   (the pipeline's temporary files, kept)"]
    if cfg.overwrite and files["run_json"].exists():
        lines.append(f"would delete   this film's earlier files in {cfg.folder} first (--overwrite)")
    return lines


# ---------------------------------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------------------------------

def pipeline_class(pipeline: str):
    module, name = PIPELINE_CLASSES[pipeline]
    return getattr(importlib.import_module(module), name)


def check_against(cls, pipeline: str) -> None:
    """The names this script writes must still be the pipeline's own; otherwise nothing starts."""
    problems = []
    if cls.PIPELINE_NAME != pipeline:
        problems.append(f"{cls.__name__} is '{cls.PIPELINE_NAME}', not '{pipeline}'")
    if tuple(cls.SLICE_FORMS) != SLICE_FORMS:
        problems.append(f"slice forms {tuple(cls.SLICE_FORMS)}, the script has {SLICE_FORMS}")
    if tuple(cls.LINE_MODES) != LINE_MODES[pipeline]:
        problems.append(f"line modes {tuple(cls.LINE_MODES)}, the script has {LINE_MODES[pipeline]}")
    unknown = sorted({s for steps in POSTPROCESS_SETS.values() for s in steps} - set(cls.POSTPROCESS_STEPS))
    if unknown:
        problems.append(f"post-processing steps {unknown} are not the pipeline's")
    if problems:
        raise Refused(f"this script no longer matches {cls.__name__}: " + "; ".join(problems))


def build_pipeline(cfg: Config):
    """The measured pipeline: the constructor arguments above, the pipeline's defaults for everything else."""
    cls = pipeline_class(cfg.pipeline)
    check_against(cls, cfg.pipeline)
    if cfg.max_slice_s is not None:  # the same pipeline with another longest slice; the class itself is not changed
        cls = type(cls.__name__, (cls,), {"MAX_SLICE_S": float(cfg.max_slice_s), "__module__": cls.__module__})
    return cls(**constructor_arguments(cfg))


def _twin(pipe, line_mode: str, steps: Tuple[str, ...]):
    """A pipeline built as the measured one, but with these line-building and post-processing settings.

    Only its _build_lines and _postprocess are used, on the measured pipeline's last_run: it never prepares,
    downloads or runs anything.
    """
    return type(pipe)(output_dir=str(pipe.output_dir), temp_dir=str(pipe.temp_dir), keep_temp_files=True,
                      model_name=pipe.model.name, speech_segmenter=pipe.speech_segmenter,
                      slice_form=pipe.slice_form, line_mode=line_mode, postprocess=tuple(steps))


def read_lines(path: Path) -> List[Tuple[str, float, float]]:
    """(text, start, end) of each subtitle, read as the pipelines count them; [] for an empty or unreadable file."""
    import srt
    try:
        return [(s.content, s.start.total_seconds(), s.end.total_seconds())
                for s in srt.parse(Path(path).read_text(encoding="utf-8", errors="replace"))]
    except Exception:
        return []


def write_variants(cfg: Config, pipe, files: Dict, pipeline_srt: Path) -> Tuple[Dict, List[Dict]]:
    """Every variant's SRT, from the measured run: lines built once per line mode, then each post-processing set."""
    run = pipe.last_run
    if not run or run.get("basename") != cfg.basename:
        raise RuntimeError("the pipeline kept no record of this film's run (last_run)")
    own = read_lines(pipeline_srt)
    line_building: Dict[str, Dict] = {}
    records: List[Dict] = []
    stitched = None
    for name, line_mode, post, steps in variants_for(cfg.pipeline):
        twin = _twin(pipe, line_mode, steps)
        if line_mode not in line_building:
            md = {"stages": {}}
            stitched = twin._build_lines(run["scenes"], run["cuts"], run["cut_scene"], run["segments"],
                                         run["basename"], md)
            line_building[line_mode] = dict(md["stages"].get("lines", {}), notes=list(twin.notes))
        dest = files["variants"][name]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(stitched, dest)  # the lines before post-processing; the stitched file is not changed
        stats = twin._postprocess(dest)
        lines = read_lines(dest)
        if not lines:
            dest.write_text("", encoding="utf-8")  # as process() writes an SRT without subtitles
        records.append({"name": name, "line_mode": line_mode, "line_building": LINE_WORDS[line_mode],
                        "postprocess": post, "steps": list(steps), "postprocessing": POSTPROCESS_WORDS[post],
                        "file": str(dest), "subtitles": len(lines), "postprocess_stats": stats,
                        "same_lines_as_pipeline_srt": lines == own})
    return line_building, records


def copy_evidence(cfg: Config, pipe, files: Dict, since: float) -> Dict:
    """Copies of the program's run.log and cuts.json, when this run wrote them."""
    from whisperjav.crispasr.runner import film_run_dir

    run_dir = film_run_dir(pipe.temp_dir, cfg.basename)  # the pipeline's own rule for the folder
    found = {"run_folder": str(run_dir), "run_log": None, "cuts": None, "program_output": None}

    def from_this_run(path: Path) -> bool:  # never a file left by an earlier run (--overwrite)
        return path.is_file() and path.stat().st_mtime >= since - 2.0

    for key, name in (("run_log", "run.log"), ("cuts", "cuts.json")):
        if from_this_run(run_dir / name):
            shutil.copy2(run_dir / name, files[key])
            found[key] = str(files[key])
    if from_this_run(run_dir / "out.json"):
        found["program_output"] = str(run_dir / "out.json")
    return found


def run_log_head(path) -> Optional[Dict[str, str]]:
    """The 'key: value' lines that open the program's run.log (command, working folder, exit code, outcome, seconds)."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    head = {}
    for line in text.splitlines():
        if line.startswith("-----"):
            break
        key, sep, value = line.partition(": ")
        if sep:
            head[key.strip()] = value.strip()
    return head


def git_state(repo: Path, run: Callable = subprocess.run) -> Dict:
    """The commit of the WhisperJAV checkout, and its uncommitted files under whisperjav/.

    Read-only: --no-optional-locks keeps 'git status' from refreshing the index.
    """
    def git(*args) -> Optional[str]:
        try:
            p = run(["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True,
                    encoding="utf-8", errors="replace", timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        return p.stdout if p.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "checkout": str(repo), "reason": "not a git checkout, or git is not available"}
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    changes = git("status", "--porcelain", "--", "whisperjav")
    return {"commit": commit.strip(), "branch": branch.strip() if branch else None, "checkout": str(repo),
            "uncommitted_in_whisperjav": None if changes is None
            else [line for line in changes.splitlines() if line.strip()]}


def whisperjav_record() -> Dict:
    import whisperjav
    from whisperjav.__version__ import __version__

    package = Path(whisperjav.__file__).resolve().parent
    return {"version": __version__, "path": str(package), "git": git_state(package.parent)}


def _failure(stage: str, e: BaseException) -> Dict:
    return {"stage": stage, "type": type(e).__name__, "message": str(e),
            "traceback": "".join(traceback.format_exception(type(e), e, e.__traceback__))}


def _program_record(pipe, md: Optional[Dict]) -> Dict:
    """The program build the pipeline chose. Its ProgramChoice (pipe._program) is kept even when the run failed."""
    record = {"version": manifest.CRISPASR_VERSION, "git": manifest.CRISPASR_GIT, "build": None}
    choice = getattr(pipe, "_program", None)
    if choice is not None:
        b, check = choice.build, choice.check
        record.update({"build": b.key, "build_label": b.label, "asset": b.asset, "size": b.size,
                       "sha256": b.sha256, "exe": str(choice.exe),
                       "start_check": {"version": check.version, "git": check.git,
                                       "compiled_backends": list(check.compiled_backends),
                                       "gpu_devices": list(check.gpu_devices)},
                       "cpu_build_reason": choice.note, "reason_in_summary": choice.note_in_summary})
    elif md:
        crisp = md.get("config", {}).get("crispasr", {})
        record.update({"build": crisp.get("build"), "exe": crisp.get("program")})
    return record


def _model_record(cfg: Config, pipe, md: Optional[Dict]) -> Dict:
    m = cfg.model
    local = (md or {}).get("config", {}).get("crispasr", {}).get("model_path") or getattr(pipe, "_model_path", None)
    return {"name": m.name, "label": m.label, "backend": m.backend, "repo": m.repo_id, "file": m.filename,
            "revision": m.revision, "size": m.size, "sha256": m.sha256, "licence": m.licence,
            "extra_args": list(m.extra_args), "offered_to_users": m.name in manifest.PIPELINE_MODELS[cfg.pipeline],
            "local_path": str(local) if local else None}


def _settings_variant(cfg: Config, pipe) -> Optional[str]:
    """The variant with the measured pipeline's own settings, if they are among the candidates."""
    for name, line_mode, _, steps in variants_for(cfg.pipeline):
        if line_mode == pipe.line_mode and set(steps) == set(pipe.postprocess):
            return name
    return None


def run_record(cfg: Config, pipe, md: Optional[Dict], error: Optional[Dict], evidence: Dict,
               line_building: Dict, variants: List[Dict], started: float, started_at: str) -> Dict:
    stages = (md or {}).get("stages", {})
    program_run = dict(stages["crispasr"]) if "crispasr" in stages else None
    if program_run is None and evidence.get("run_log"):
        program_run = {"read_from_run_log": run_log_head(evidence["run_log"])}
    elif program_run is None and md is not None:
        program_run = {"not_run": "no slices, so the CrispASR program was not run"}
    summary = (md or {}).get("summary", {})
    pipeline_srt = (md or {}).get("output_files", {}).get("final_srt")
    return {
        "plan": "CrispASR master plan 4.5, step 2b",
        "script": str(Path(__file__).resolve()),
        "command": list(cfg.argv),
        "started": started_at,
        "finished": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.time() - started, 1),
        "configuration": {
            "label": cfg.label, "pipeline": cfg.pipeline, "pipeline_class": type(pipe).__name__,
            "model": cfg.model.name, "slice_form": cfg.slice_form, "slices": SLICE_WORDS[cfg.slice_form],
            "max_slice_s": getattr(pipe, "MAX_SLICE_S", None), "max_slice_s_option": cfg.max_slice_s,
            "speech_segmenter": pipe.speech_segmenter, "speech_segmenter_used": cfg.slice_form == "speech",
            "line_mode": pipe.line_mode, "postprocess": list(pipe.postprocess),
            "constructor_arguments": constructor_arguments(cfg),
            "film": str(cfg.film), "basename": cfg.basename, "folder": str(cfg.folder)},
        "model": _model_record(cfg, pipe, md),
        "program": _program_record(pipe, md),
        "program_run": program_run,
        "slices": stages.get("slices"),
        "scenes": stages.get("scene_detection", {}).get("scenes"),
        "audio_seconds": (md or {}).get("input_info", {}).get("audio_duration_seconds"),
        "pipeline_srt": None if pipeline_srt is None else {
            "file": pipeline_srt, "subtitles": len(read_lines(Path(pipeline_srt))),
            "settings_variant": _settings_variant(cfg, pipe),
            "same_lines_as": [v["name"] for v in variants if v["same_lines_as_pipeline_srt"]]},
        "line_building": line_building,
        "variants": variants,
        "notes": summary.get("notes", []),
        "degradations": summary.get("degradations", []),
        "evidence": evidence,
        "whisperjav": whisperjav_record(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "error": error,
        "pipeline_metadata": md,
    }


def report(cfg: Config, record: Dict, files: Dict) -> None:
    say("")
    say(f"{cfg.label}: {cfg.film.name}")
    program, run, slices = record["program"], record["program_run"] or {}, record["slices"] or {}
    if program.get("build"):
        devices = (program.get("start_check") or {}).get("gpu_devices") or []
        say(f"  program: CrispASR {program['version']} {program.get('build_label', program['build'])}"
            + (f" on {', '.join(devices)}" if devices else ""))
    if "outcome" in run:
        say(f"  program run: {run['outcome']} (exit code {run['exit_code']}), {run['seconds']} s, "
            f"{run['warnings']} warning(s)")
    if slices:
        say(f"  slices: {slices.get('count')} ({slices.get('seconds')} s of {record['audio_seconds']} s of audio), "
            f"{record['scenes']} scenes")
    if record["pipeline_srt"]:
        say(f"  the pipeline's own SRT: {record['pipeline_srt']['subtitles']} subtitles")
    for v in record["variants"]:
        say(f"  {v['name']}: {v['subtitles']} subtitles")
    if record["error"]:
        say(f"  FAILED in {record['error']['stage']}: {record['error']['type']}: {record['error']['message']}")
    say(f"  {record['seconds']} s in all; {files['run_json']}")


def measure(cfg: Config, pipe) -> int:
    """One process() call -- one CrispASR program run -- then every variant from it, and run.json."""
    files = planned_files(cfg)
    started = time.time()
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    say(f"{cfg.label}: {cfg.film.name}: one CrispASR program run, then {len(files['variants'])} variants from it")
    md, error = None, None
    try:
        md = pipe.process({"path": str(cfg.film), "basename": cfg.basename})
    except Exception as e:  # recorded in run.json with its traceback, never hidden
        error = _failure("the pipeline run", e)
    evidence = copy_evidence(cfg, pipe, files, since=started)
    line_building, variants = {}, []
    if error is None:
        try:
            line_building, variants = write_variants(cfg, pipe, files, Path(md["output_files"]["final_srt"]))
        except Exception as e:
            error = _failure("the variants", e)
    record = run_record(cfg, pipe, md, error, evidence, line_building, variants, started, started_at)
    files["run_json"].write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n",
                                 encoding="utf-8")
    report(cfg, record, files)
    return 0 if error is None else 1


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        refuse_if_measured(cfg)
        if cfg.dry_run:
            for line in describe(cfg):
                say(line)
            return 0
        if cfg.overwrite:
            clear_previous(cfg)
        pipe = build_pipeline(cfg)
    except Refused as e:
        say(f"refused: {e}", stream=sys.stderr)
        return 2
    return measure(cfg, pipe)


if __name__ == "__main__":
    sys.exit(main())
