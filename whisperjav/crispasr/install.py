"""Getting CrispASR's program and model files ready, and saying whether they are (master plan 1.2-1.7).

The program
- WhisperJAV downloads it itself the first time a CrispASR pipeline runs, in every environment
  (installer, WJ110, source installs). Nothing is downloaded with --offline.
- Plain HTTPS from CrispASR's GitHub release; size and SHA-256 checked against the manifest; a
  damaged download is deleted and fetched again once, then the run stops with a message.
- Unpacked to ~/.cache/whisperjav/crispasr/<version>/<build>/, with a ``.verified`` record.
- Which build runs is decided before the film run, never by re-running (plan 1.4):
    1. the CUDA 13 build, when PyTorch sees an NVIDIA GPU, is built for CUDA 13, has the three CUDA 13
       runtime files, and the build's start check finds a CUDA device;
    2. otherwise the CPU build, with one console line and a note in the end-of-run summary;
    3. an x86-64 processor without AVX2 cannot run any build: the CrispASR pipelines stop (Q23).
- The start check runs ``crispasr --diagnostics``. ``--version`` is not enough: the backend list it
  prints is fixed when the program is compiled (crispasr_diagnostics.cpp:48-73), so it names CUDA even
  where CUDA cannot load. ``--diagnostics`` lists the devices that actually registered (:279-320).
- The CUDA 13 runtime files come from the user's own PyTorch (torch/lib) and are put beside the
  program as hard links (no extra disk), or copied when a link is impossible (plan 1.3).

The models
- WhisperJAV downloads them, never the program: each file by repository, name and pinned commit,
  into the Hugging Face cache, at the start of the run, before audio extraction (plan 1.5).
- Network-free status (cache first) feeds ``whisperjav --check``.
- A folder the user fills by hand can be named in WHISPERJAV_CRISPASR_MODEL_DIR (Q14).
- Each file's SHA-256 is checked once and remembered by size and modification time.
- A model path with characters outside ASCII is hard-linked, else copied, into the run folder,
  because the program's model loader opens files in the PC's code page (plan 1.7).
"""

import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from whisperjav.crispasr import manifest
from whisperjav.crispasr.manifest import CRISPASR_VERSION, CUDA13_RUNTIME_FILES, ModelFile, ProgramBuild
from whisperjav.crispasr.runner import exit_code_meaning
from whisperjav.utils.logger import logger
from whisperjav.utils.offline_mode import hub_offline

MODEL_DIR_ENV = "WHISPERJAV_CRISPASR_MODEL_DIR"
_MB = 1_000_000
_DOWNLOAD_TIMEOUT = (30, 120)  # seconds: connect, and between bytes
# Room for the archive and its unpacked files. The CUDA 13 build is 168 MB zipped and about 212 MB
# unpacked; the CPU build 9 MB and 22 MB.
_PROGRAM_SPACE_FACTOR = 2.5
_MODEL_SPACE_FACTOR = 1.1  # the file size plus 10% (plan 1.5)


class CrispASRSetupError(RuntimeError):
    """The program or a model cannot be made ready. The run stops before audio extraction (plan 2.6)."""


def mb(n_bytes: int) -> str:
    return f"{n_bytes / _MB:,.0f} MB"


# ---------------------------------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------------------------------

def whisperjav_cache_root() -> Path:
    """WhisperJAV's own cache folder (~/.cache/whisperjav)."""
    return Path.home() / ".cache" / "whisperjav"


def program_root() -> Path:
    return whisperjav_cache_root() / "crispasr" / CRISPASR_VERSION


def program_dir(build: ProgramBuild) -> Path:
    return program_root() / build.key


def _verified_models_record() -> Path:
    return whisperjav_cache_root() / "crispasr" / "verified-models.json"


def _check_free_space(folder: Path, needed: int, what: str) -> None:
    folder = Path(folder)
    probe = folder
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # cannot tell; the download itself will fail clearly if the disk fills
    if free < needed:
        raise CrispASRSetupError(
            f"Not enough disk space for {what}: {mb(needed)} needed in {probe}, {mb(free)} free.")


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------
# The processor
# ---------------------------------------------------------------------------------------------------

def has_avx2() -> Optional[bool]:
    """Whether this x86-64 processor has AVX2: True, False, or None when it cannot be told.

    None is not a stop: a program that then cannot run ends with "illegal instruction", which the
    runner reports in plain words.
    """
    if platform.machine().lower() not in ("amd64", "x86_64", "x64"):
        return None
    if sys.platform == "win32":
        try:
            import ctypes
            if ctypes.windll.kernel32.IsProcessorFeaturePresent(40):  # PF_AVX2_INSTRUCTIONS_AVAILABLE
                return True
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
                if line.startswith("flags"):
                    return "avx2" in line.split()
        except OSError:
            pass
    elif sys.platform == "darwin":
        try:
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.leaf7_features"], capture_output=True,
                                 text=True, timeout=10).stdout
            if out.strip():
                return "AVX2" in out.upper().split()
        except (OSError, subprocess.SubprocessError):
            pass
    # Second opinion: PyTorch's CPU dispatch level. Windows builds older than the AVX2 feature flag
    # answer "no" above even on a processor that has it.
    try:
        import torch
        level = str(torch.backends.cpu.get_cpu_capability()).upper()
        return level.startswith(("AVX2", "AVX512"))
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------------
# The program: status, download, unpack
# ---------------------------------------------------------------------------------------------------

@dataclass
class ProgramStatus:
    build: ProgramBuild
    downloaded: bool
    exe: Optional[Path] = None


def program_status(build: ProgramBuild) -> ProgramStatus:
    """Whether this build is downloaded and unpacked. Reads files only: no network, no program start."""
    folder = program_dir(build)
    exe = folder / build.exe_name
    try:
        record = json.loads((folder / ".verified").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ProgramStatus(build, False)
    if (record.get("version") == CRISPASR_VERSION and record.get("asset") == build.asset
            and record.get("sha256") == build.sha256 and exe.is_file()):
        return ProgramStatus(build, True, exe)
    return ProgramStatus(build, False)


def _download(url: str, dest: Path, size: int, sha256: str, what: str) -> Path:
    """Download ``url`` to ``dest``, checking size and SHA-256. A damaged file is fetched again once."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in (1, 2):
        logger.info(f"CrispASR: downloading {what} ({mb(size)}) from {url}")
        h = hashlib.sha256()
        got = 0
        next_report = 0.1
        try:
            with requests.get(url, stream=True, timeout=_DOWNLOAD_TIMEOUT) as r:
                r.raise_for_status()
                with open(part, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        h.update(chunk)
                        got += len(chunk)
                        if size and got / size >= next_report:
                            logger.info(f"CrispASR: {what}: {min(100, int(got * 100 / size))}% "
                                        f"({mb(got)} of {mb(size)})")
                            next_report += 0.1
        except (requests.RequestException, OSError) as e:
            part.unlink(missing_ok=True)
            raise CrispASRSetupError(f"Downloading {what} failed: {e}") from e
        if got == size and h.hexdigest() == sha256:
            os.replace(part, dest)
            return dest
        part.unlink(missing_ok=True)
        logger.warning(f"CrispASR: the download of {what} is damaged ({got:,} bytes, SHA-256 "
                       f"{h.hexdigest()[:12]}..., expected {size:,} bytes, {sha256[:12]}...)"
                       + ("; downloading it again" if attempt == 1 else ""))
    raise CrispASRSetupError(f"The download of {what} was damaged twice; nothing was installed.")


def _safe_extract(archive: Path, into: Path) -> None:
    """Unpack a release archive; a member that would land outside ``into`` stops it (both formats)."""
    into.mkdir(parents=True, exist_ok=True)
    root = into.resolve()
    try:
        if archive.name.endswith(".zip"):
            with zipfile.ZipFile(archive) as zf:
                for member in zf.namelist():
                    target = (into / member).resolve()
                    if target != root and root not in target.parents:
                        raise CrispASRSetupError(f"Refusing to unpack {archive.name}: '{member}' leaves the folder")
                zf.extractall(into)
        else:
            with tarfile.open(archive, "r:*") as tf:
                tf.extractall(into, filter="data")
    except (tarfile.TarError, zipfile.BadZipFile) as e:
        raise CrispASRSetupError(f"Refusing to unpack {archive.name}: {e}") from e


def ensure_program(build: ProgramBuild, offline: Optional[bool] = None) -> Path:
    """The program file of this build, downloading and unpacking it if needed.

    ``offline`` defaults to WhisperJAV's --offline switch (HF_HUB_OFFLINE), which does not reach a
    plain HTTPS download on its own.
    """
    status = program_status(build)
    if status.downloaded:
        return status.exe
    offline = hub_offline() if offline is None else offline
    what = f"the CrispASR {CRISPASR_VERSION} {build.label}"
    if offline:
        raise CrispASRSetupError(f"{what[0].upper()}{what[1:]} is not downloaded yet ({mb(build.size)}), "
                                 f"and --offline allows no download.")
    root = program_root()
    _check_free_space(root, int(build.size * _PROGRAM_SPACE_FACTOR), what)
    # The download and unpacking places carry this process's id, so two WhisperJAV runs fetching the
    # same build at once never write into each other's files.
    tag = os.getpid()
    archive = _download(build.url, root / ".download" / f"{tag}-{build.asset}", build.size, build.sha256, what)

    staging = root / f".{build.key}.staging-{tag}"
    final = program_dir(build)
    shutil.rmtree(staging, ignore_errors=True)
    try:
        _safe_extract(archive, staging)
        found = [p for p in staging.rglob(build.exe_name) if p.is_file()]
        if len(found) != 1:
            raise CrispASRSetupError(f"{build.asset} does not hold exactly one {build.exe_name} "
                                     f"(found {len(found)}).")
        finished = program_status(build)
        if finished.downloaded:
            # Another run finished the same download first: use its copy, which is in use already.
            return finished.exe
        if final.exists():
            shutil.rmtree(final)
        shutil.move(str(found[0].parent), str(final))
        exe = final / build.exe_name
        if sys.platform != "win32":
            exe.chmod(exe.stat().st_mode | 0o111)
        record = {"version": CRISPASR_VERSION, "asset": build.asset, "size": build.size,
                  "sha256": build.sha256, "unpacked": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (final / ".verified").write_text(json.dumps(record, indent=1), encoding="utf-8")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        archive.unlink(missing_ok=True)
    logger.info(f"CrispASR: {what} is ready in {final}")
    return exe


# ---------------------------------------------------------------------------------------------------
# The CUDA 13 runtime files and the start check
# ---------------------------------------------------------------------------------------------------

def torch_lib_dir() -> Optional[Path]:
    """PyTorch's library folder, found without importing torch."""
    try:
        spec = importlib.util.find_spec("torch")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0]) / "lib"


def place_cuda_runtime(program_folder: Path, source_dir: Path) -> str:
    """Put PyTorch's three CUDA 13 runtime files beside the program (plan 1.3).

    Done before the first run and again whenever a file's size no longer matches PyTorch's. A hard
    link takes no extra disk and is the same file PyTorch has; a copy is made where a link is
    impossible (another drive, or not NTFS). Returns "ready", "linked" or "copied".
    """
    how = "ready"
    for name in CUDA13_RUNTIME_FILES:
        src, dst = Path(source_dir) / name, Path(program_folder) / name
        if not src.is_file():
            raise CrispASRSetupError(f"PyTorch's {name} was not found in {source_dir}")
        if dst.is_file() and dst.stat().st_size == src.stat().st_size:
            continue
        if dst.exists():
            dst.unlink()
        try:
            os.link(src, dst)
            how = "linked" if how == "ready" else how
        except OSError:
            shutil.copy2(src, dst)
            how = "copied"
    if how == "copied":
        total = sum((Path(source_dir) / n).stat().st_size for n in CUDA13_RUNTIME_FILES)
        logger.info(f"CrispASR: PyTorch's CUDA 13 runtime files were copied beside the program "
                    f"({mb(total)}); a hard link was not possible there.")
    return how


@dataclass
class StartCheck:
    ok: bool
    reason: Optional[str]  # why not, in plain words
    version: Optional[str] = None
    git: Optional[str] = None
    compiled_backends: list = field(default_factory=list)
    gpu_devices: list = field(default_factory=list)  # "CUDA0: NVIDIA GeForce RTX 3060"
    exit_code: Optional[int] = None


_DIAG_VERSION = re.compile(r"^\s*version\s*:\s*(\S+)", re.MULTILINE)
_DIAG_GIT = re.compile(r"^\s*git sha\s*:\s*(\S+)", re.MULTILINE)
_DIAG_COMPILED = re.compile(r"^\s*ggml backends\s*:\s*(\S+)", re.MULTILINE)
_DIAG_DEVICE = re.compile(r"^\s*\[\d+\]\s+(gpu|igpu)\s+name=(\S+)\s+desc=(.*?)\s+mem=", re.MULTILINE)


def parse_diagnostics(text: str) -> StartCheck:
    version = _DIAG_VERSION.search(text)
    git = _DIAG_GIT.search(text)
    compiled = _DIAG_COMPILED.search(text)
    return StartCheck(
        ok=False, reason=None,
        version=version.group(1) if version else None,
        git=git.group(1) if git else None,
        compiled_backends=compiled.group(1).split(",") if compiled else [],
        gpu_devices=[f"{m.group(2)}: {m.group(3).strip()}" for m in _DIAG_DEVICE.finditer(text)])


def start_check(exe: Path, build: ProgramBuild, timeout_s: float = 120) -> StartCheck:
    """Start the program once with ``--diagnostics`` and judge whether this build can run here."""
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    try:
        p = subprocess.run([str(exe), "--diagnostics"], cwd=str(Path(exe).parent), capture_output=True,
                           timeout=timeout_s, creationflags=creationflags)
    except subprocess.TimeoutExpired:
        return StartCheck(False, f"it did not answer within {timeout_s:.0f} s")
    except OSError as e:
        return StartCheck(False, f"it could not be started ({e})")
    text = (p.stdout or b"").decode("utf-8", "replace") + "\n" + (p.stderr or b"").decode("utf-8", "replace")
    check = parse_diagnostics(text)
    check.exit_code = p.returncode
    if p.returncode != 0:
        check.reason = exit_code_meaning(p.returncode)
    elif check.version != CRISPASR_VERSION:
        check.reason = f"it reports version {check.version or 'unknown'}, not {CRISPASR_VERSION}"
    elif check.git and not (check.git.startswith(manifest.CRISPASR_GIT)
                            or manifest.CRISPASR_GIT.startswith(check.git)):
        check.reason = f"it reports build {check.git}, not {manifest.CRISPASR_GIT}"
    elif build.gpu == "cuda13" and not any(d.upper().startswith("CUDA") for d in check.gpu_devices):
        check.reason = "it found no CUDA device"
    else:
        check.ok = True
    return check


def cuda13_unavailable_reason() -> Optional[str]:
    """Why the CUDA 13 build cannot be used on this PC, or None when it can be tried."""
    unavailable = cuda13_unavailable()
    return unavailable.reason if unavailable else None


@dataclass(frozen=True)
class CudaUnavailable:
    reason: str     # plain words
    expected: bool  # a normal set-up for the CPU build (no NVIDIA GPU, a PyTorch without CUDA 13)


def cuda13_unavailable() -> Optional[CudaUnavailable]:
    """Why the CUDA 13 build cannot be used on this PC, or None when it can be tried.

    PyTorch's own CUDA build is asked first, so a PC with an NVIDIA card but a PyTorch without CUDA
    is not told that it has no NVIDIA GPU.
    """
    lib = torch_lib_dir()
    try:
        import torch
    except Exception:
        return CudaUnavailable("PyTorch could not be loaded to look for an NVIDIA GPU", expected=False)
    cuda = str(getattr(torch.version, "cuda", "") or "")
    if not cuda:
        return CudaUnavailable("this install's PyTorch has no CUDA support", expected=True)
    if not cuda.startswith("13."):
        return CudaUnavailable(f"this install's PyTorch is built for CUDA {cuda}, not CUDA 13", expected=True)
    try:
        available = torch.cuda.is_available()
    except Exception:
        available = False
    if not available:
        return CudaUnavailable("no NVIDIA GPU was found", expected=True)
    missing = [n for n in CUDA13_RUNTIME_FILES if lib is None or not (lib / n).is_file()]
    if missing:
        return CudaUnavailable(f"PyTorch's CUDA 13 runtime files are missing ({', '.join(missing)})",
                               expected=False)
    return None


@dataclass
class ProgramChoice:
    build: ProgramBuild
    exe: Path
    check: StartCheck
    note: Optional[str] = None  # why the CPU build runs: one console line per pass
    # Plan 2.6: the note also goes into each file's summary only when the fallback was not expected
    # (the GPU build's start check failed, PyTorch's CUDA files were missing or could not be placed).
    # No NVIDIA GPU, or a PyTorch without CUDA 13, is the normal CPU set-up: console line only.
    note_in_summary: bool = False


def prepare_program(offline: Optional[bool] = None) -> ProgramChoice:
    """Choose the build for this PC, make it ready, and check that it starts (plan 1.4).

    Raises CrispASRSetupError when no build can run here; the run then stops before audio extraction.
    """
    builds = manifest.builds_for_this_machine()
    if not builds:
        raise CrispASRSetupError(f"CrispASR has no program build for this computer "
                                 f"({sys.platform}, {platform.machine()}).")
    if any(b.needs_avx2 for b in builds) and has_avx2() is False:
        raise CrispASRSetupError("This processor lacks AVX2, which CrispASR's program needs "
                                 "(Intel since 2013, AMD since 2015). The CrispASR pipelines cannot run here.")
    gpu = next((b for b in builds if b.gpu == "cuda13"), None)
    cpu = next((b for b in builds if b.gpu is None), None)
    in_summary = False
    if gpu is not None:
        unavailable = cuda13_unavailable()
        if unavailable is None:
            in_summary = True  # from here on, a fallback to the CPU build was not expected
            exe = ensure_program(gpu, offline)
            try:
                place_cuda_runtime(exe.parent, torch_lib_dir())
            except (CrispASRSetupError, OSError) as e:
                reason = f"PyTorch's CUDA 13 runtime files could not be placed beside the GPU build ({e})"
            else:
                check = start_check(exe, gpu)
                if check.ok:
                    return ProgramChoice(gpu, exe, check)
                reason = f"the GPU build's start check failed: {check.reason}"
        else:
            reason, in_summary = unavailable.reason, not unavailable.expected
        note = f"CrispASR runs on the processor (CPU build), which is slower: {reason}."
    else:
        note = "CrispASR runs on the processor (CPU build), which is slower."
    if cpu is None:
        raise CrispASRSetupError(f"CrispASR has no CPU build for this computer ({sys.platform}).")
    exe = ensure_program(cpu, offline)
    check = start_check(exe, cpu)
    if not check.ok:
        raise CrispASRSetupError(f"CrispASR's {cpu.label} cannot start on this computer: {check.reason}.")
    return ProgramChoice(cpu, exe, check, note, note_in_summary=in_summary)


# ---------------------------------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------------------------------

@dataclass
class ModelStatus:
    model: ModelFile
    downloaded: bool
    path: Optional[Path] = None
    source: str = ""  # "folder" (WHISPERJAV_CRISPASR_MODEL_DIR) or "cache" (Hugging Face cache)


def model_status(model: ModelFile) -> ModelStatus:
    """Whether a model file is on this PC. No network: the hand-filled folder, then the cache."""
    folder = os.environ.get(MODEL_DIR_ENV, "").strip()
    if folder:
        p = Path(folder) / model.filename
        if p.is_file() and p.stat().st_size == model.size:
            return ModelStatus(model, True, p, "folder")
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(model.repo_id, model.filename, revision=model.revision)
    except Exception:
        hit = None
    if isinstance(hit, str):
        p = Path(hit)
        if p.is_file() and p.stat().st_size == model.size:
            return ModelStatus(model, True, p, "cache")
    return ModelStatus(model, False)


def _load_verified() -> dict:
    try:
        return json.loads(_verified_models_record().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_verified(path: Path, sha256: str) -> None:
    record = _load_verified()
    st = path.stat()
    record[os.path.realpath(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256}
    target = _verified_models_record()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=1), encoding="utf-8")
    except OSError as e:
        logger.debug(f"CrispASR: could not record the model check ({e})")


def _already_verified(path: Path, model: ModelFile) -> bool:
    entry = _load_verified().get(os.path.realpath(path))
    if not entry:
        return False
    st = path.stat()
    return (entry.get("sha256") == model.sha256 and entry.get("size") == st.st_size
            and entry.get("mtime_ns") == st.st_mtime_ns)


def _hf_download(model: ModelFile, force: bool = False) -> Path:
    """Fetch one pinned file into the Hugging Face cache."""
    from huggingface_hub import hf_hub_download

    logger.info(f"CrispASR: downloading the model {model.label} ({mb(model.size)}) "
                f"from {model.repo_id}")
    try:
        return Path(hf_hub_download(model.repo_id, model.filename, revision=model.revision,
                                    force_download=force))
    except Exception as e:
        raise CrispASRSetupError(f"Downloading the model {model.label} failed: {e}") from e


def _hf_cache_dir() -> Path:
    try:
        from huggingface_hub import constants
        return Path(constants.HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def ensure_model(model: ModelFile, offline: Optional[bool] = None) -> Path:
    """The model file on this PC, downloading it if needed, with its SHA-256 checked once.

    A damaged file in the cache is deleted and downloaded once more; a damaged file in the user's own
    folder is reported and never deleted.
    """
    offline = hub_offline() if offline is None else offline
    status = model_status(model)
    if not status.downloaded:
        if offline:
            raise CrispASRSetupError(f"The model {model.label} ({model.filename}, {mb(model.size)}) is not "
                                     f"downloaded yet, and --offline allows no download.")
        _check_free_space(_hf_cache_dir(), int(model.size * _MODEL_SPACE_FACTOR), f"the model {model.label}")
        path = _hf_download(model)
        source = "cache"
    else:
        path, source = status.path, status.source
    if _already_verified(path, model):
        return path

    logger.info(f"CrispASR: checking the model file {model.filename} ({mb(model.size)})")
    if _sha256_of(path) == model.sha256:
        _remember_verified(path, model.sha256)
        return path
    if source == "folder":
        raise CrispASRSetupError(f"The model file {path} (from {MODEL_DIR_ENV}) is damaged: its SHA-256 "
                                 f"does not match {model.repo_id} at the pinned revision.")
    if offline:
        raise CrispASRSetupError(f"The model file {path} is damaged, and --offline allows no new download.")
    logger.warning(f"CrispASR: the model file {path} is damaged; downloading it again")
    real = Path(os.path.realpath(path))
    for p in {path, real}:
        try:
            p.unlink()
        except OSError:
            pass
    path = _hf_download(model, force=True)
    if _sha256_of(path) != model.sha256:
        raise CrispASRSetupError(f"The model {model.label} was damaged twice; it cannot be used.")
    _remember_verified(path, model.sha256)
    return path


def link_or_copy(src: Path, dst: Path, what: Optional[str] = None) -> str:
    """Make ``dst`` the same file as ``src``: a hard link (no extra disk), else a copy.

    Returns "linked" or "copied". An existing ``dst`` is replaced, a symbolic link that leads nowhere
    included. With ``what`` (plain words for the file), a copy is checked against the free disk space first.
    """
    dst = Path(dst)
    if os.path.lexists(dst):  # exists() is False for a link that leads nowhere, and os.link cannot replace it
        os.unlink(dst)
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        if what:
            _check_free_space(dst.parent, int(Path(src).stat().st_size * _MODEL_SPACE_FACTOR), what)
        shutil.copy2(src, dst)
        return "copied"


def stage_model(model_path: Path, run_dir: Path, stage_dir: Optional[Path] = None) -> str:
    """What ``-m`` gets: the model's full path when it is all ASCII, else a short relative ASCII name.

    The program opens model files with the PC's code page (CreateFileA, gguf_loader.cpp:205), so a path
    with Chinese or Japanese characters may not open. Such a file is hard-linked into ``stage_dir``
    (default: the run folder), or copied when a link is impossible (plan 1.7), and named relative to the
    run folder, the program's working folder. A pipeline passes the folder that holds every film's run
    folder, so one staged file serves the whole pass instead of a copy per film. A symbolic link, as the
    Hugging Face cache holds its files with Windows Developer Mode on, is followed first: the staged file
    is the model file itself.
    """
    model_path = Path(model_path)
    if str(model_path).isascii():
        return str(model_path)
    stage_dir = Path(stage_dir) if stage_dir is not None else Path(run_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    dst = stage_dir / (model_path.name if model_path.name.isascii() else "model.gguf")
    size = model_path.stat().st_size
    if not (dst.is_file() and dst.stat().st_size == size):
        # On Windows os.link links a symbolic link itself, not the file it leads to, and the cache's links are
        # relative: from stage_dir such a link leads nowhere. Link the file the path leads to.
        if link_or_copy(os.path.realpath(model_path), dst, what="a copy of the model") == "copied":
            logger.info(f"CrispASR: the model was copied to {stage_dir} ({mb(size)}), because its path "
                        f"has characters the program may not open")
    return os.path.relpath(dst, run_dir)
