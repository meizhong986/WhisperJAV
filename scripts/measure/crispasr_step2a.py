"""CrispASR step 2a: the program downloaded through WhisperJAV's own code, and what was unpacked (CrispASR master
plan 1.2, 4.5; U17, U18).

What the script does:
  1. for every build manifest.builds_for_this_machine() lists, install.ensure_program(build): the HTTPS download from
     CrispASR's GitHub release, the size and SHA-256 check against the manifest, the unpacking into
     ~/.cache/whisperjav/crispasr/<version>/<build>/, and the .verified record. A build already unpacked is not
     downloaded again (the record says which);
  2. per build: the .verified record and every unpacked file with its size;
  3. the GPU build's start check before PyTorch's three CUDA 13 files are put beside it (U18): install.start_check,
     which starts `crispasr --diagnostics` once, with the program's whole output kept. Skipped, with the reason, when
     any of the three files is already in the folder. The program also finds CUDA files through PATH: on a PC with
     NVIDIA's CUDA 13 toolkit installed, --start-check-path (for example "C:\\Windows\\System32;C:\\Windows") runs
     the check with that PATH instead, as on a PC without the toolkit;
  4. the pin refusing a download that does not match it: install.ensure_program on the CPU build with its SHA-256
     changed, into a separate cache folder under <out>, never the real one. It must end with CrispASRSetupError and
     install nothing (the CPU build is the smaller file: about 9 MB, fetched twice).

Usage (from the repository root, with WJ110's python):
    python scripts/measure/crispasr_step2a.py --out <folder>

Writes <out>/step2a.json and prints a summary. Exit status: 0 when every build is ready and the wrong pin was refused;
1 otherwise.
"""
import argparse
import dataclasses
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from whisperjav.crispasr import install, manifest


def unpacked_files(folder: Path) -> list:
    return [{"file": str(p.relative_to(folder)), "bytes": p.stat().st_size}
            for p in sorted(folder.rglob("*")) if p.is_file()]


@contextmanager
def recorded_runs():
    """Every subprocess.run call made inside, with its full output (start_check keeps only what it parses)."""
    calls, real = [], subprocess.run

    def recording(*args, **kwargs):
        p = real(*args, **kwargs)
        calls.append(p)
        return p

    subprocess.run = recording
    try:
        yield calls
    finally:
        subprocess.run = real


def text(data) -> str:
    return data.decode("utf-8", "replace") if isinstance(data, bytes) else (data or "")


def download_builds() -> list:
    out = []
    for build in manifest.builds_for_this_machine():
        before = install.program_status(build).downloaded
        started = time.time()
        entry = {"build": build.key, "label": build.label, "asset": build.asset, "url": build.url,
                 "pinned_size": build.size, "pinned_sha256": build.sha256, "already_unpacked": before}
        try:
            exe = install.ensure_program(build)
        except install.CrispASRSetupError as e:
            entry.update({"ready": False, "error": str(e)})
        else:
            folder = exe.parent
            entry.update({"ready": True, "seconds": round(time.time() - started, 1), "exe": str(exe),
                          "folder": str(folder),
                          "verified_record": json.loads((folder / ".verified").read_text(encoding="utf-8")),
                          "files": unpacked_files(folder)})
        out.append(entry)
    return out


def cuda_files_on_path(path: str) -> dict:
    """Where each CUDA 13 runtime file would be found through this PATH (the first folder that holds it)."""
    found = {}
    for name in manifest.CUDA13_RUNTIME_FILES:
        found[name] = next((str(Path(d) / name) for d in path.split(os.pathsep) if d and (Path(d) / name).is_file()),
                           None)
    return found


def start_check_record(build_key: str = "cuda13", path: str = None, place_files: bool = False) -> dict:
    """One start check of a build. For the GPU build, without PyTorch's CUDA files beside the program unless
    place_files, which puts them there first with install.place_cuda_runtime, as a run does."""
    build = manifest.build_by_key(build_key)
    if build is None:
        return {"skipped": f"this computer has no {build_key} build"}
    status = install.program_status(build)
    if not status.downloaded:
        return {"skipped": f"the {build.label} is not unpacked"}
    placed = None
    if build.gpu == "cuda13":
        if place_files:
            placed = install.place_cuda_runtime(status.exe.parent, install.torch_lib_dir())
        present = [n for n in manifest.CUDA13_RUNTIME_FILES if (status.exe.parent / n).exists()]
        if present and not place_files:
            return {"skipped": f"already beside the program: {', '.join(present)}"}
    else:
        present = []
    used_path = os.environ.get("PATH", "") if path is None else path
    with mock.patch.dict(os.environ, {} if path is None else {"PATH": path}), recorded_runs() as calls:
        check = install.start_check(status.exe, build)
    p = calls[-1] if calls else None
    return {"build": build.key, "exe": str(status.exe), "cuda_files_beside_program": present,
            "placed_by_place_cuda_runtime": placed, "path": used_path,
            "path_replaced": path is not None, "cuda_files_on_path": cuda_files_on_path(used_path),
            "check": dataclasses.asdict(check),
            "program_exit_code": None if p is None else p.returncode,
            "program_stdout": None if p is None else text(p.stdout),
            "program_stderr": None if p is None else text(p.stderr)}


def wrong_pin_is_refused(out: Path) -> dict:
    cpu = next((b for b in manifest.builds_for_this_machine() if b.gpu is None), None)
    if cpu is None:
        return {"skipped": "this computer has no CPU build"}
    wrong = dataclasses.replace(cpu, sha256="0" * 64)
    cache = out / "wrong_pin_cache"
    with mock.patch.object(install, "whisperjav_cache_root", return_value=cache):
        try:
            install.ensure_program(wrong)
        except install.CrispASRSetupError as e:
            refused, message = True, str(e)
        else:
            refused, message = False, "the download was accepted"
        left = [str(p.relative_to(cache)) for p in sorted(cache.rglob("*")) if p.is_file()] if cache.exists() else []
        installed = install.program_status(wrong).downloaded
    return {"build": cpu.key, "asset": cpu.asset, "pin_used": wrong.sha256, "cache": str(cache), "refused": refused,
            "message": message, "installed": installed, "files_left": left}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="crispasr_step2a.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder for the record and the wrong-pin cache")
    ap.add_argument("--start-check-path", help="run the start check of step 3 with this PATH instead of this one")
    ap.add_argument("--start-check-build", default="cuda13", choices=["cuda13", "cpu"],
                    help="the build step 3 checks (default cuda13)")
    ap.add_argument("--place-cuda-files", action="store_true",
                    help="step 3 on the GPU build with PyTorch's three CUDA 13 files put beside it first")
    ap.add_argument("--skip-wrong-pin", action="store_true", help="leave out step 4")
    ap.add_argument("--record", default="step2a.json", help="the record's file name in --out (default step2a.json)")
    a = ap.parse_args(argv)
    out = Path(os.path.abspath(a.out))
    out.mkdir(parents=True, exist_ok=True)

    record = {"plan": "CrispASR master plan 4.5, step 2a", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "command": sys.argv[1:] if argv is None else list(argv),
              "crispasr_version": manifest.CRISPASR_VERSION, "program_root": str(install.program_root()),
              "platform": platform.platform(), "python": sys.version.split()[0]}
    record["builds"] = download_builds()
    record["start_check_without_cuda_files"] = start_check_record(a.start_check_build, a.start_check_path,
                                                                  a.place_cuda_files)
    record["wrong_pin"] = {"skipped": "--skip-wrong-pin"} if a.skip_wrong_pin else wrong_pin_is_refused(out)
    (out / a.record).write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    for b in record["builds"]:
        if b.get("ready"):
            print(f"{b['label']}: ready in {b['folder']} ({'already unpacked' if b['already_unpacked'] else 'downloaded'}"
                  f", {len(b['files'])} files); .verified sha256 {b['verified_record']['sha256']}")
        else:
            print(f"{b['label']}: NOT READY: {b.get('error')}")
    u18 = record["start_check_without_cuda_files"]
    if "check" in u18:
        c = u18["check"]
        print(f"start check of the {u18['build']} build, CUDA files beside it: {u18['cuda_files_beside_program']}: "
              f"ok={c['ok']}, reason={c['reason']!r}, exit code {u18['program_exit_code']}, "
              f"devices {c['gpu_devices']}")
    else:
        print(f"start check: skipped ({u18['skipped']})")
    if "cuda_files_on_path" in u18:
        print(f"  PATH {'replaced' if u18['path_replaced'] else 'as inherited'}; CUDA files found through it: "
              f"{u18['cuda_files_on_path']}")
    w = record["wrong_pin"]
    print(f"wrong pin: refused={w.get('refused')}, installed={w.get('installed')}, files left={w.get('files_left')}; "
          f"{w.get('message', w.get('skipped'))}")
    print(f"written {out / a.record}")
    ok = all(b.get("ready") for b in record["builds"]) and (
        a.skip_wrong_pin or (w.get("refused") is True and not w.get("installed")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
