"""Making CrispASR and its model files ready, and saying whether they are (1.10.0, route B).

CrispASR
- It is the Python package ``crispasr``, installed with WhisperJAV from the locked list (route B,
  owner W176): the CUDA build (``+cuda``, its own CUDA 12.8 libraries inside) in the cu130 and cu128
  variants, the CPU build otherwise (Metal inside on macOS arm64). Nothing is downloaded for it at run
  time.
- ``prepare_runtime(device, model)`` says, before any audio is read, whether it can run here and on
  which device (route B design 4.2, 9): a build for this computer, the package, AVX2, the model's back
  end in this build, then the device by the --device rule of owner W194 1 (A + Metal).

The models
- WhisperJAV downloads them, never CrispASR (W130, W133 3): each file by repository, name and pinned
  commit, into the Hugging Face cache, at the start of the run, before audio extraction.
- Network-free status (cache first) feeds ``whisperjav --check``.
- A folder the user fills by hand can be named in WHISPERJAV_CRISPASR_MODEL_DIR.
- Each file's SHA-256 is checked once and remembered by size and modification time.
- CrispASR cannot open a model path with Japanese characters on Windows (binding checks of
  2026-10-24, item 2), so such a file is staged under a plain-ASCII name (``stage_model``).
"""

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from whisperjav.crispasr.errors import CrispASRSetupError
from whisperjav.crispasr.manifest import ModelFile
from whisperjav.utils.logger import logger
from whisperjav.utils.offline_mode import hub_offline

__all__ = ["CrispASRSetupError", "hub_offline"]  # re-exported for the start-up check

MODEL_DIR_ENV = "WHISPERJAV_CRISPASR_MODEL_DIR"
STAGE_FOLDER = "crispasr_stage"  # the name WhisperJAV's leftover handling knows (utils/media_leftovers.py)
_MB = 1_000_000
_MODEL_SPACE_FACTOR = 1.1  # the file size plus 10%

# (sys.platform, platform.machine()) pairs CrispASR publishes a wheel for (design 11.1).
CRISPASR_PLATFORMS = frozenset({
    ("win32", "AMD64"), ("win32", "x86_64"),
    ("linux", "x86_64"), ("linux", "aarch64"),
    ("darwin", "arm64"),
})

# --- Texts (owner-approved wording, W193 5 / W203 I2; route B design 12) -------------------------
NOT_INSTALLED_TEXT = "CrispASR is not installed in this WhisperJAV. Run the WhisperJAV installer again."
NO_BUILD_TEXT = ("CrispASR has no build for this computer ({platform}, {machine}); "
                 "the CrispASR pipelines cannot run here.")
NO_AVX2_TEXT = ("This processor lacks AVX2, which CrispASR needs (Intel since 2013, AMD since 2015). "
                "The CrispASR pipelines cannot run here.")
NO_BACKEND_TEXT = "This CrispASR build ({version}) cannot run {model}: it was built without '{backend}'."
NOT_LOADED_TEXT = "CrispASR could not be loaded on this computer: {reason}"
# T10j, the Vulkan build (design 22.10) [inferred cause].
NOT_LOADED_VULKAN_HINT = (" With the GPU-through-Vulkan or Own PyTorch install on Windows this usually means the "
                          "Vulkan runtime (vulkan-1.dll) is missing: update the graphics driver, or run the "
                          "WhisperJAV installer again and choose CPU only.")
DEVICE_CUDA_TEXT = ("--device cuda: CrispASR cannot use an NVIDIA GPU here: {reason}. "
                    "Use --device auto or --device cpu.")
# T10g, the Vulkan build (design 22.3.2).
DEVICE_CUDA_VULKAN_TEXT = ("--device cuda needs CrispASR's CUDA build; this install has its Vulkan build. "
                           "Use --device auto to let the Vulkan build use this PC's GPU.")
NO_STAGE_TEXT = ("CrispASR cannot open the model's folder (its name has non-ASCII characters) and no "
                 "plain-named folder was found for a link or copy. Put {filename} in a plain-named "
                 "folder and set " + MODEL_DIR_ENV + " to it.")
STAGED_LINK_TEXT = "CrispASR: the model was linked to {folder}, because CrispASR cannot open its folder's name"
STAGED_COPY_TEXT = ("CrispASR: the model was copied to {folder} ({size}), because CrispASR cannot open "
                    "its folder's name{kept}")
STAGED_COPY_KEPT = "; the copy is kept for later runs"
# The device line, once per pass (T6, revised in design 22.10).
DEVICE_LINE_NVIDIA = "{label} asks CrispASR to use the NVIDIA GPU (CUDA) (CrispASR {version})."
DEVICE_LINE_METAL = ("{label} asks CrispASR to use the Apple GPU (Metal) (CrispASR {version}); "
                     "not yet tested on a Mac.")
DEVICE_LINE_VULKAN = ("{label} asks CrispASR's Vulkan build to use this PC's GPU (CrispASR {version}). "
                      "GPUs through Vulkan are experimental in 1.10.0 and have not been tested on AMD or Intel "
                      "graphics cards.")
DEVICE_LINE_CPU = "{label} runs on the CPU (not accelerated), several times slower than on a GPU: {reason}."
REASON_NO_NVIDIA = "no NVIDIA GPU was found"
REASON_NO_GPU = "no GPU was found"
REASON_CPU_BUILD = "this install has CrispASR's CPU build"
REASON_DEVICE_CPU = "--device cpu was given"


def mb(n_bytes: int) -> str:
    return f"{n_bytes / _MB:,.0f} MB"


# ---------------------------------------------------------------------------------------------------
# Places
# ---------------------------------------------------------------------------------------------------

def whisperjav_cache_root() -> Path:
    """WhisperJAV's own cache folder (~/.cache/whisperjav)."""
    return Path.home() / ".cache" / "whisperjav"


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
        return  # cannot tell; the copy or download itself will fail clearly if the disk fills
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
# CrispASR on this computer
# ---------------------------------------------------------------------------------------------------

def has_avx2() -> Optional[bool]:
    """Whether this x86-64 processor has AVX2: True, False, or None when it cannot be told (not a stop).

    CrispASR's Windows libraries are built with -DGGML_AVX2=ON (binding checks of 2026-10-24, item 1).
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


def platform_supported(sys_platform: Optional[str] = None, machine: Optional[str] = None) -> bool:
    return ((sys_platform or sys.platform), (machine or platform.machine())) in CRISPASR_PLATFORMS


def crispasr_installed() -> bool:
    try:
        return importlib.util.find_spec("crispasr") is not None
    except (ImportError, ValueError):
        return False


def not_installed_text() -> str:
    """Why CrispASR is missing (T10a). Every 1.10 install of a supported computer has CrispASR
    (design 22.3.2), so a missing one means a damaged install."""
    return NOT_INSTALLED_TEXT


def installed_build() -> Tuple[str, str]:
    """The installed crispasr's version and build: ("0.8.44+cuda", "cuda"), ("0.8.44", "cpu"), ...

    The build is read from the version's local label (+cuda, +vulkan); no label is the CPU build.
    Raises CrispASRSetupError when the package is not installed.
    """
    from importlib import metadata

    try:
        version = metadata.version("crispasr")
    except metadata.PackageNotFoundError as e:
        raise CrispASRSetupError(not_installed_text()) from e
    local = version.partition("+")[2].lower()
    if local.startswith("cuda"):
        return version, "cuda"
    if local.startswith("vulkan"):
        return version, "vulkan"
    return version, "cpu"


def available_backends() -> List[str]:
    """The model families the installed CrispASR library was built with. This loads its libraries."""
    try:
        import crispasr
        return list(crispasr.Session.available_backends())
    except (ImportError, OSError) as e:
        hint = ""
        try:
            hint = NOT_LOADED_VULKAN_HINT if installed_build()[1] == "vulkan" else ""
        except CrispASRSetupError:
            pass
        raise CrispASRSetupError(NOT_LOADED_TEXT.format(reason=e) + hint) from e


@dataclass(frozen=True)
class RuntimeChoice:
    use_gpu: bool
    device: str              # "cuda", "metal", "vulkan" or "cpu"
    reason: Optional[str]    # why the processor, in plain words; None on a GPU
    note_in_summary: bool    # the processor was not expected: say so in each file's summary too
    version: str             # the installed crispasr, e.g. "0.8.44+cuda"
    build: str               # "cuda", "cpu" or "vulkan"

    def device_line(self, label: str) -> str:
        if self.device == "cuda":
            return DEVICE_LINE_NVIDIA.format(label=label, version=self.version)
        if self.device == "metal":
            return DEVICE_LINE_METAL.format(label=label, version=self.version)
        if self.device == "vulkan":
            return DEVICE_LINE_VULKAN.format(label=label, version=self.version)
        return DEVICE_LINE_CPU.format(label=label, reason=self.reason)


def _best_device() -> Tuple[str, Optional[str]]:
    """WhisperJAV's own device check: ("cuda" | "mps" | "cpu", why an NVIDIA card PyTorch sees is unusable)."""
    from whisperjav.utils import device_detector

    best = device_detector.get_best_device()
    return best, device_detector.CUDA_UNUSABLE_REASON


def choose_device(device: Optional[str], build: str, version: str,
                  sys_platform: Optional[str] = None) -> RuntimeChoice:
    """The --device rule (owner W194 1, A + Metal; route B design 9.1).

    ``use_gpu=True`` is a request to CrispASR: it sends the same value as leaving it out, and with no
    usable CUDA the library may run on the processor without an error (design 4.2 step 5).
    """
    device = (device or "auto").strip().lower()
    sys_platform = sys_platform or sys.platform
    if device == "cpu":
        return RuntimeChoice(False, "cpu", REASON_DEVICE_CPU, False, version, build)
    if build == "cuda":
        best, unusable = _best_device()
        if best == "cuda":
            return RuntimeChoice(True, "cuda", None, False, version, build)
        reason = unusable or REASON_NO_NVIDIA
        if device == "cuda":
            raise CrispASRSetupError(DEVICE_CUDA_TEXT.format(reason=reason))
        # A card PyTorch sees but cannot use is not the normal set-up: the note also goes in the summary.
        return RuntimeChoice(False, "cpu", reason, unusable is not None, version, build)
    if build == "vulkan":
        # The GPU-through-Vulkan install (design 22.3.2, owner W234 D-A): WhisperJAV cannot see which
        # GPU Vulkan takes, so auto asks the build for its GPU (text rule 2, W217 3).
        if device == "cuda":
            raise CrispASRSetupError(DEVICE_CUDA_VULKAN_TEXT)
        return RuntimeChoice(True, "vulkan", None, False, version, build)
    reason = REASON_CPU_BUILD
    if device == "cuda":
        raise CrispASRSetupError(DEVICE_CUDA_TEXT.format(reason=reason))
    if build == "cpu" and sys_platform == "darwin":
        best, _ = _best_device()
        if best == "mps":
            return RuntimeChoice(True, "metal", None, False, version, build)
        reason = REASON_NO_GPU
    return RuntimeChoice(False, "cpu", reason, False, version, build)


def _installed_here() -> Tuple[str, str]:
    """A build for this computer, the package, AVX2 -- in that order; the installed version and build."""
    if not platform_supported():
        raise CrispASRSetupError(NO_BUILD_TEXT.format(platform=sys.platform, machine=platform.machine()))
    if not crispasr_installed():
        raise CrispASRSetupError(not_installed_text())
    version, build = installed_build()
    if has_avx2() is False:
        raise CrispASRSetupError(NO_AVX2_TEXT)
    return version, build


def runtime_status(device: Optional[str] = None) -> RuntimeChoice:
    """What prepare_runtime would choose, without loading CrispASR's libraries: for --check and the GUI."""
    version, build = _installed_here()
    return choose_device(device, build, version)


def prepare_runtime(device: Optional[str], model: ModelFile) -> RuntimeChoice:
    """Whether CrispASR can run ``model`` here, and on which device (route B design 4.2).

    In this order: a build for this computer, the package, AVX2, the model's back end in this build
    (this loads CrispASR's libraries), then the device. Raises CrispASRSetupError with the reason.
    """
    version, build = _installed_here()
    backends = available_backends()
    if model.backend not in backends:
        raise CrispASRSetupError(NO_BACKEND_TEXT.format(version=version, model=model.label,
                                                        backend=model.backend))
    return choose_device(device, build, version)


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
    st = Path(path).stat()
    record[os.path.realpath(path)] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": sha256}
    target = _verified_models_record()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=1), encoding="utf-8")
    except OSError as e:
        logger.debug(f"CrispASR: could not record the model check ({e})")


def _already_verified(path: Path, sha256: str) -> bool:
    entry = _load_verified().get(os.path.realpath(path))
    if not entry:
        return False
    st = Path(path).stat()
    return (entry.get("sha256") == sha256 and entry.get("size") == st.st_size
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
    folder is reported and never deleted. CrispASR's own downloader is never used (W130, W133 3).
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
    if _already_verified(path, model.sha256):
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


# ---------------------------------------------------------------------------------------------------
# A plain-ASCII path for the model (route B design 4.2, owner O14)
# ---------------------------------------------------------------------------------------------------

def _short_path(path: str) -> Optional[str]:
    """Windows' 8.3 short form of an existing path, or None."""
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        fn = ctypes.windll.kernel32.GetShortPathNameW
        fn.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
        fn.restype = wintypes.DWORD
        size = fn(path, None, 0)
        if not size:
            return None
        buf = ctypes.create_unicode_buffer(size)
        if not fn(path, buf, size):
            return None
        return buf.value
    except Exception:
        return None


def stage_candidates(temp_dir: Path) -> List[Tuple[Path, bool]]:
    """Folders a staged model may go in, in order, each with whether a copy there outlives the run:
    %PUBLIC%\\WhisperJAV\\crispasr and %ProgramData%\\WhisperJAV\\crispasr on Windows, then
    <temp>\\crispasr_stage. Only folders whose own path is all ASCII."""
    out: List[Tuple[Path, bool]] = []
    if sys.platform == "win32":
        for env in ("PUBLIC", "ProgramData"):
            base = os.environ.get(env, "").strip()
            if base:
                out.append((Path(base) / "WhisperJAV" / "crispasr", True))
    out.append((Path(temp_dir) / STAGE_FOLDER, False))
    return [(folder, keeps) for folder, keeps in out if str(folder.resolve()).isascii()]


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def stage_model(model_path: Path, temp_dir: Path, sha256: str) -> Path:
    """The model's path as CrispASR can open it: itself when all ASCII, else a plain-ASCII name.

    1. An all-ASCII path is used as it is; on Windows the file's 8.3 short path is tried next.
    2. Otherwise a hard link (no extra disk) is tried in every candidate folder, and only if no link
       can be made anywhere is a copy made, in the first candidate with room.
    3. The staged file is named after the model's SHA-256, so the two Cohere models (the same size)
       never share it. A link is reused only if it is the same file as the model; a copy only if its
       own SHA-256 check is on record. The file the path leads to is staged (os.path.realpath: the
       Hugging Face cache holds its files as symbolic links with Windows Developer Mode on).
    Raises CrispASRSetupError when neither works.
    """
    model_path = Path(model_path).absolute()
    if str(model_path).isascii():
        return model_path
    real = Path(os.path.realpath(model_path))
    short = _short_path(str(real))
    if short and short.isascii() and Path(short).is_file():
        return Path(short)

    name = f"{sha256[:16]}.gguf"
    candidates = stage_candidates(temp_dir)
    usable: List[Tuple[Path, bool]] = []
    for folder, keeps in candidates:
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        usable.append((folder, keeps))
        dst = folder / name
        if _same_file(dst, real):
            return dst
        if dst.is_file() and _already_verified(dst, sha256):
            return dst  # a copy made by an earlier run, checked when it was made
    for folder, _keeps in usable:
        dst = folder / name
        try:
            if os.path.lexists(dst):  # a stale link or a copy without a check record: replace it
                os.unlink(dst)
            os.link(real, dst)
        except OSError:
            continue
        logger.info(STAGED_LINK_TEXT.format(folder=folder))
        return dst
    size = real.stat().st_size
    for folder, keeps in usable:
        dst = folder / name
        part = folder / (name + ".part")
        try:
            _check_free_space(folder, int(size * _MODEL_SPACE_FACTOR), "a copy of the model")
            shutil.copy2(real, part)
            os.replace(part, dst)
        except (OSError, CrispASRSetupError) as e:
            logger.debug(f"CrispASR: no copy of the model in {folder} ({e})")
            try:
                part.unlink()
            except OSError:
                pass
            continue
        _remember_verified(dst, sha256)
        logger.info(STAGED_COPY_TEXT.format(folder=folder, size=mb(size), kept=STAGED_COPY_KEPT if keeps else ""))
        return dst
    logger.error(f"CrispASR: no plain-named folder for the model {model_path} (tried: "
                 f"{', '.join(str(f) for f, _k in candidates) or 'none'})")
    raise CrispASRSetupError(NO_STAGE_TEXT.format(filename=model_path.name))
