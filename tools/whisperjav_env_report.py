#!/usr/bin/env python3
"""
whisperjav_env_report.py — collect the technology stack of a WhisperJAV installation.

Run it with the SAME Python that runs WhisperJAV, from any folder:

    Windows installer:  %LOCALAPPDATA%\\WhisperJAV\\python.exe whisperjav_env_report.py
    source install:     python whisperjav_env_report.py      (inside the WhisperJAV environment)
    Colab / Kaggle:     !python whisperjav_env_report.py

or double-click / right-click "Run with PowerShell" on whisperjav_env_report.ps1, which finds
that Python for you.

It writes whisperjav_env_report.md next to itself and prints the same text, ready to paste
into a GitHub comment. Optional:

    --probe     also load the smallest Whisper model on the GPU, report which compute type
                "auto" resolves to, transcribe 5 s of generated audio, and report whether
                releasing the model is clean (the destructor abort WhisperJAV's nuclear exit
                works around; issue #125). Downloads ~75 MB once (Systran/faster-whisper-tiny)
                unless already cached; loads the model three times. About a minute.
    --out PATH  write the report somewhere else.

What it collects: operating system, CPU, RAM, GPU name / driver / VRAM, Python, WhisperJAV
version and install type, versions of the packages that matter for the balanced pipeline,
which CUDA / cuDNN / cuBLAS libraries actually load and from where, the Silero VAD model file
faster-whisper bundles, and the CUDA-related environment variables and PATH entries.

What it does NOT collect: file names, media, subtitles, API keys. Your home folder and your
account name are replaced by "~" in paths; other folder names (for example where you cloned the
repository) are shown as they are. Read the report before posting it and delete any line you
would rather not share.

No changes are made to your installation. Nothing is sent anywhere; you post the text yourself.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPORT_VERSION = "1.2 (2026-09-09)"
_WHISPERJAV_FOUND = False

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
_HOME = str(Path.home())
_USER = os.environ.get("USERNAME") or os.environ.get("USER") or ""


def redact(text: Any) -> str:
    """Replace the home folder and the account name (as a whole path segment) with '~'."""
    s = str(text)
    if _HOME:
        s = re.sub(re.escape(_HOME), "~", s, flags=re.IGNORECASE)
    if _USER:
        s = re.sub(r"(?<![A-Za-z0-9_.-])" + re.escape(_USER) + r"(?![A-Za-z0-9_.-])", "~", s, flags=re.IGNORECASE)
    return s.replace("|", "/")  # keep Markdown table cells intact


def run(cmd: List[str], timeout: int = 20) -> Optional[str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
        return None
    except Exception:
        return None


def dist_version(name: str) -> Optional[str]:
    try:
        from importlib import metadata

        return metadata.version(name)
    except Exception:
        return None


def md5_of(path: Path, limit: Optional[int] = None) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(limit) if limit else f.read())
    return h.hexdigest()


class Section:
    def __init__(self, title: str):
        self.title = title
        self.rows: List[str] = []
        self.errors: List[str] = []

    def add(self, key: str, value: Any) -> None:
        self.rows.append(f"| {key} | {redact(value)} |")

    def err(self, what: str, exc: BaseException) -> None:
        self.errors.append(f"{what}: {type(exc).__name__}: {redact(exc)}")

    def render(self) -> str:
        out = [f"### {self.title}", "", "| item | value |", "|---|---|", *self.rows]
        if self.errors:
            out += ["", "Not available: " + "; ".join(self.errors)]
        return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def sec_system() -> Section:
    s = Section("System")
    s.add("report version", REPORT_VERSION)
    s.add("date (UTC)", time.strftime("%Y-%m-%d %H:%M", time.gmtime()))
    s.add("OS", platform.platform())
    s.add("machine", platform.machine())
    try:
        cpu = platform.processor() or ""
        if os.name == "nt":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip() or cpu
        s.add("CPU", cpu)
    except Exception as e:
        s.err("CPU", e)
    try:
        if os.name == "nt":
            class _MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            ms = _MemStatus()
            ms.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            s.add("RAM (GB)", round(ms.ullTotalPhys / 2**30, 1))
        elif hasattr(os, "sysconf"):
            s.add("RAM (GB)", round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1))
    except Exception as e:
        s.err("RAM", e)
    return s


def sec_gpu() -> Section:
    s = Section("GPU (nvidia-smi)")
    q = run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap", "--format=csv,noheader"], 20)
    if q is None:
        s.errors.append("nvidia-smi not found or failed (no NVIDIA GPU, or not on PATH)")
        return s
    for i, line in enumerate(q.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            s.add(f"GPU {i}", f"{parts[0]} | driver {parts[1]} | {parts[2]}" + (f" | compute capability {parts[3]}" if len(parts) > 3 else ""))
    head = run(["nvidia-smi"], 20)
    if head:
        m = re.search(r"CUDA Version:\s*([\d.]+)", head)
        if m:
            s.add("CUDA version the driver reports", m.group(1))
    return s


def sec_python_install() -> Section:
    s = Section("Python and WhisperJAV")
    s.add("Python", sys.version.split()[0] + f" ({platform.architecture()[0]})")
    s.add("Python executable", sys.executable)
    s.add("sys.prefix", sys.prefix)
    s.add("WhisperJAV (installed package)", dist_version("whisperjav") or "not installed in this Python")
    try:
        import whisperjav  # type: ignore

        globals()["_WHISPERJAV_FOUND"] = True
        s.add("WhisperJAV import path", Path(whisperjav.__file__).parent)
        try:
            from whisperjav.__version__ import __version__ as v  # type: ignore

            s.add("WhisperJAV __version__", v)
        except Exception:
            pass
    except Exception as e:
        s.err("import whisperjav", e)
    # install type heuristics
    prefix = Path(sys.prefix)
    kind = []
    if any(prefix.glob("post_install_v*.py")) or any(prefix.glob("install_log_v*.txt")):
        kind.append("Windows conda-constructor installer")
    if (prefix / "conda-meta").exists():
        kind.append("conda environment")
    if (prefix / "pyvenv.cfg").exists():
        kind.append("venv")
    if os.environ.get("COLAB_RELEASE_TAG") or Path("/content").exists():
        kind.append("Google Colab")
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        kind.append("Kaggle")
    try:
        from importlib import metadata

        d = metadata.distribution("whisperjav")
        direct = d.read_text("direct_url.json")
        if direct:
            j = json.loads(direct)
            if j.get("dir_info", {}).get("editable"):
                kind.append("editable source install")
            if j.get("vcs_info"):
                kind.append(f"git install ({j['vcs_info'].get('requested_revision') or j['vcs_info'].get('commit_id', '')[:8]})")
    except Exception:
        pass
    s.add("install type (heuristic)", ", ".join(kind) or "unknown")
    return s


PACKAGES = [
    "whisperjav", "torch", "torchaudio", "faster-whisper", "faster-whisper2", "ctranslate2",
    "onnxruntime", "onnxruntime-gpu", "onnxruntime-directml", "numpy", "av", "stable-ts",
    "openai-whisper", "huggingface-hub", "tokenizers", "transformers", "silero-vad",
    "ten-vad", "fireredvad", "whisperseg", "qwen-asr", "auditok", "pydub", "soundfile",
    "nvidia-cudnn-cu12", "nvidia-cudnn-cu13", "nvidia-cublas-cu12", "nvidia-cublas-cu13",
    "nvidia-cuda-runtime-cu12", "nvidia-cuda-runtime", "triton", "setuptools",
]


def sec_packages() -> Section:
    s = Section("Packages (importlib.metadata)")
    for name in PACKAGES:
        v = dist_version(name)
        if v is not None:
            s.add(name, v)
    return s


def sec_torch() -> Section:
    s = Section("PyTorch and CUDA runtime")
    try:
        import torch  # type: ignore

        s.add("torch", torch.__version__)
        s.add("torch.version.cuda", getattr(torch.version, "cuda", None))
        try:
            s.add("torch cuDNN version", torch.backends.cudnn.version())
        except Exception as e:
            s.err("cudnn version", e)
        s.add("torch.cuda.is_available", torch.cuda.is_available())
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                cap = torch.cuda.get_device_capability(i)
                s.add(f"torch device {i}", f"{torch.cuda.get_device_name(i)} | sm_{cap[0]}{cap[1]} | "
                                          f"{round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1)} GB")
            try:
                s.add("torch arch list", ", ".join(torch.cuda.get_arch_list()))
            except Exception:
                pass
    except Exception as e:
        s.err("import torch", e)
    return s


def sec_ct2_fw() -> Section:
    s = Section("CTranslate2 and faster-whisper")
    try:
        import ctranslate2  # type: ignore

        s.add("ctranslate2", ctranslate2.__version__)
        s.add("ctranslate2 file", Path(ctranslate2.__file__).parent)
        try:
            s.add("CUDA devices seen by CTranslate2", ctranslate2.get_cuda_device_count())
            s.add("CTranslate2 supported compute types (cuda)", ", ".join(sorted(ctranslate2.get_supported_compute_types("cuda"))))
        except Exception as e:
            s.err("ct2 cuda query", e)
        try:
            s.add("CTranslate2 supported compute types (cpu)", ", ".join(sorted(ctranslate2.get_supported_compute_types("cpu"))))
        except Exception:
            pass
    except Exception as e:
        s.err("import ctranslate2", e)
    try:
        import faster_whisper  # type: ignore

        s.add("faster_whisper import version", getattr(faster_whisper, "__version__", "?"))
        s.add("faster_whisper file", Path(faster_whisper.__file__).parent)
        s.add("distribution providing it", ", ".join(n for n in ("faster-whisper", "faster-whisper2") if dist_version(n)) or "?")
        try:
            from faster_whisper.utils import get_assets_path  # type: ignore

            assets = Path(get_assets_path())
            for p in sorted(assets.glob("*.onnx")):
                s.add(f"bundled VAD asset {p.name}", f"{p.stat().st_size} bytes, md5 {md5_of(p)}")
        except Exception as e:
            s.err("VAD asset", e)
        try:
            from faster_whisper import BatchedInferencePipeline  # noqa: F401

            s.add("BatchedInferencePipeline available", True)
        except Exception:
            s.add("BatchedInferencePipeline available", False)
    except Exception as e:
        s.err("import faster_whisper", e)
    try:
        import onnxruntime as ort  # type: ignore

        s.add("onnxruntime providers", ", ".join(ort.get_available_providers()))
    except Exception as e:
        s.err("onnxruntime", e)
    return s


_CUDA_DLL_NAMES = [
    "cudart64_12.dll", "cudart64_13.dll", "cublas64_12.dll", "cublas64_13.dll",
    "cublasLt64_12.dll", "cublasLt64_13.dll", "cudnn64_8.dll", "cudnn64_9.dll",
    "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll", "cudnn_ops_infer64_8.dll",
]


def sec_cuda_libraries() -> Section:
    """Which CUDA libraries the process can actually load, and from where.

    Imports torch first when present (WhisperJAV does the same before faster-whisper), then
    asks Windows to resolve each library by name. The path answers the question "whose
    cuDNN / cuBLAS does CTranslate2 get in this installation?".
    """
    s = Section("CUDA libraries the process resolves (after importing torch, if present)")
    try:
        importlib.import_module("torch")
    except Exception:
        pass
    if os.name == "nt":
        k32 = ctypes.windll.kernel32
        k32.GetModuleHandleW.restype = ctypes.c_void_p
        k32.LoadLibraryW.restype = ctypes.c_void_p
        k32.GetModuleFileNameW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint)
        for name in _CUDA_DLL_NAMES:
            try:
                h = k32.GetModuleHandleW(name) or k32.LoadLibraryW(name)
                if not h:
                    s.add(name, "not found")
                    continue
                buf = ctypes.create_unicode_buffer(1024)
                k32.GetModuleFileNameW(h, buf, 1024)
                s.add(name, buf.value)
            except Exception as e:
                s.err(name, e)
    else:
        try:
            import psutil  # type: ignore

            paths = sorted({m.path for m in psutil.Process().memory_maps()
                            if re.search(r"libcudnn|libcublas|libcudart|ctranslate2", m.path)})
            for p in paths:
                s.add(Path(p).name, p)
            if not paths:
                s.add("note", "no CUDA libraries mapped yet (psutil listing)")
        except Exception as e:
            s.err("library listing", e)
    return s


def sec_env() -> Section:
    s = Section("Environment variables and PATH (CUDA-related)")
    for k in ("CUDA_PATH", "CUDA_HOME", "CUDNN_PATH", "CT2_CUDA_ALLOCATOR", "CT2_VERBOSE",
              "HF_HUB_OFFLINE", "HF_HOME", "OMP_NUM_THREADS", "KMP_DUPLICATE_LIB_OK", "LD_LIBRARY_PATH"):
        if k in os.environ:
            s.add(k, os.environ[k])
    entries = [p for p in os.environ.get("PATH", "").split(os.pathsep)
               if re.search(r"cuda|cudnn|nvidia|whisperjav|torch", p, re.I)]
    for i, p in enumerate(entries[:12]):
        s.add(f"PATH[{i}]", p)
    # NVIDIA pip runtime packages present in this Python (a second possible source of cuDNN/cuBLAS)
    try:
        import site

        found = []
        for sp in {*site.getsitepackages(), site.getusersitepackages()}:
            nv = Path(sp) / "nvidia"
            if nv.is_dir():
                found += [d.name for d in nv.iterdir() if d.is_dir() and ((d / "bin").is_dir() or (d / "lib").is_dir())]
        s.add("nvidia/* pip packages with binaries", ", ".join(sorted(set(found))) or "none")
    except Exception as e:
        s.err("nvidia pip dirs", e)
    return s


def sec_probe() -> Section:
    """Optional: load faster-whisper `tiny` on CUDA, transcribe 5 s of generated audio, then
    release the model in a child process and report the child's exit code."""
    s = Section("Probe (--probe): CTranslate2 on this GPU")
    try:
        import numpy as np  # type: ignore
        from faster_whisper import WhisperModel  # type: ignore

        t0 = time.perf_counter()
        model = WhisperModel("tiny", device="cuda", compute_type="float16")
        s.add("tiny model load, compute_type=float16 (s)", round(time.perf_counter() - t0, 2))
        s.add("effective compute type when float16 is requested", getattr(model.model, "compute_type", "?"))
        try:
            m_auto = WhisperModel("tiny", device="cuda", compute_type="auto")
            s.add("effective compute type when 'auto' is requested", getattr(m_auto.model, "compute_type", "?"))
            globals().setdefault("_KEEP", []).append(m_auto)  # never destroyed in this process
        except Exception as e:
            s.err("auto compute type", e)
        globals().setdefault("_KEEP", []).append(model)
        sr = 16000
        t = np.arange(5 * sr) / sr
        audio = (0.05 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
        t0 = time.perf_counter()
        segs, info = model.transcribe(audio, language="ja", beam_size=2, vad_filter=False)
        n = len(list(segs))
        s.add("5 s synthetic transcribe (s)", round(time.perf_counter() - t0, 2))
        s.add("segments returned (any number is fine)", n)
    except Exception as e:
        s.err("model load / transcribe", e)
        return s
    # destructor behaviour, in a child so this report survives either way
    code = "import faster_whisper,gc,sys\nm=faster_whisper.WhisperModel('tiny',device='cuda',compute_type='float16')\ndel m; gc.collect()\nprint('released ok'); sys.stdout.flush()\n"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
        rc = r.returncode & 0xFFFFFFFF
        s.add("child process: release model, then exit", f"exit code {r.returncode} (0x{rc:08X}); stdout: {r.stdout.strip()[:60]}")
        if rc == 0xC0000409:
            s.add("interpretation", "the CTranslate2 model destructor aborts the process on this stack (0xC0000409, the abort WhisperJAV's nuclear exit in main.py works around; issue #125)")
        elif r.returncode == 0 and "released ok" in r.stdout:
            s.add("interpretation", "model release is clean on this stack")
        else:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
            s.add("interpretation", "the child did not complete the release test; last output lines: " + " / ".join(tail)[:300])
    except Exception as e:
        s.err("release probe", e)
    return s


# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Collect the WhisperJAV technology stack into a Markdown report.")
    ap.add_argument("--probe", action="store_true", help="also load the tiny model on the GPU and test model release")
    ap.add_argument("--out", type=Path, default=None, help="report path (default: whisperjav_env_report.md next to this script)")
    args = ap.parse_args(argv)

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    sections = [sec_system(), sec_gpu(), sec_python_install(), sec_packages(), sec_torch(),
                sec_ct2_fw(), sec_cuda_libraries(), sec_env()]
    if args.probe:
        sections.append(sec_probe())

    body = "\n".join(sec.render() for sec in sections)
    warn = ("" if _WHISPERJAV_FOUND else
            "**WARNING: this Python cannot import whisperjav; the report may describe the wrong interpreter.**\n\n")
    text = (
        "<details><summary>WhisperJAV environment report</summary>\n\n"
        + warn + body
        + "\n</details>\n"
    )
    out = args.out or (Path(__file__).resolve().parent / "whisperjav_env_report.md")
    try:
        out.write_text(text, encoding="utf-8")
        where = str(out)
    except Exception as e:
        where = f"(could not write file: {e})"

    print(text)
    print(f"\nSaved to: {redact(where)}")
    print("Please read it, then paste the whole block (including the <details> lines) into the GitHub issue.")
    if not _WHISPERJAV_FOUND:
        print("\n*** WARNING: this Python cannot import whisperjav. The report describes the WRONG interpreter. ***\n"
              "*** Run the script with the python.exe inside your WhisperJAV installation (see the top of the file). ***")
    sys.stdout.flush()
    sys.stderr.flush()
    # CTranslate2 4.6.x aborts at interpreter exit once a CUDA model existed (--probe); the
    # report is already written and printed, so leave without running destructors.
    os._exit(0)


if __name__ == "__main__":
    main()
