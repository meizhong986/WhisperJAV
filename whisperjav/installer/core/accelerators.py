"""
The accelerators of WhisperJAV 1.10 -- one table for every installer, upgrade and run.

WHY THIS EXISTS (design section 22.0 and 22.6.8; owner W234, W238)
    Every install of 1.10 is made for ONE accelerator, chosen once when installing and
    recorded in <install>/whisperjav_stack.txt. Every run, every upgrade and every second
    run of an installer follows that record. Before 1.10 the same facts (record values,
    labels, lock extras, PyTorch indexes) were spread over about eight tables in
    install.py, upgrade.py, cublas12.py and the Windows installer's post-install script.
    This module is the one place for them.

IMPORT RULES
    Standard library only, no imports from whisperjav.*: install.py imports it before
    WhisperJAV is installed, and installer/build_release.py writes this table into the
    Windows installer's post-install script (which cannot import WhisperJAV before its
    phase 5). tests/test_accelerators_v110.py checks that the copy equals this file.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

STACK_FILE = "whisperjav_stack.txt"

CU130_INDEX = "https://download.pytorch.org/whl/cu130"
CU128_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"


class Accelerator(NamedTuple):
    value: str            # in whisperjav_stack.txt; never shown to users (text rule 4, W217 6)
    name: str             # what users see (design 22.10, T27)
    extra: str            # the uv.lock extra; Windows may differ (windows_extra)
    windows_extra: str    # the extra on Windows
    pytorch_index: str    # "" = WhisperJAV installs no PyTorch here (Own PyTorch)
    nvidia: bool          # made for an NVIDIA GPU: the one start-up stop applies (design 22.5.4)
    runs: str             # where the parts run, for the install summary (T29b)


ACCELERATORS: Tuple[Accelerator, ...] = (
    Accelerator("cu130", "NVIDIA GPU (CUDA 13.0)", "cu130", "cu130", CU130_INDEX, True,
                "Every pipeline runs on the NVIDIA GPU (CUDA); the WhisperSeg speech segmenter on the CPU."),
    Accelerator("cu128", "NVIDIA GPU (CUDA 12.8)", "cu128", "cu128", CU128_INDEX, True,
                "Every pipeline runs on the NVIDIA GPU (CUDA); the WhisperSeg speech segmenter on the CPU."),
    Accelerator("vulkan", "GPU through Vulkan (AMD, Intel or older NVIDIA card)", "vulkan", "vulkan", CPU_INDEX, False,
                "The GGUF pipelines (Parakeet-CrispASR, Cohere-CrispASR) ask for your graphics card through "
                "Vulkan (experimental, not tested); all other pipelines run on the CPU (not accelerated)."),
    Accelerator("metal", "Apple GPU (Metal)", "cpu", "cpu", CPU_INDEX, False,
                "The transformers pipeline asks for the Apple GPU (Metal); the GGUF pipelines and fidelity ask "
                "for it too (not yet tested on a Mac); balanced, fast, faster and ChronosJAV run on the CPU "
                "(not accelerated)."),
    Accelerator("own-pytorch", "Own PyTorch", "cpu", "vulkan", "", False,
                "fidelity, transformers, ChronosJAV and the speech enhancers ask for the GPU your PyTorch "
                "reports (not tested); balanced, fast and faster run on the CPU (not accelerated) unless "
                "CTranslate2 sees an NVIDIA GPU."),
    Accelerator("cpu", "CPU only (not accelerated)", "cpu", "cpu", CPU_INDEX, False,
                "Every pipeline runs on the CPU (not accelerated), several times slower than on a GPU."),
)

VALUES = tuple(a.value for a in ACCELERATORS)
# The uv.lock extras, one exported requirements file each (installer/build_release.py).
LOCK_EXTRAS = ("cu130", "cu128", "vulkan", "cpu")

_BY_VALUE = {a.value: a for a in ACCELERATORS}


def get(value: Optional[str]) -> Optional[Accelerator]:
    """The accelerator for a record value, or None for an unknown or missing one."""
    return _BY_VALUE.get(value or "")


def lock_extra(value: str, platform: str = sys.platform) -> str:
    """The uv.lock extra an install of this accelerator uses on this platform."""
    acc = _BY_VALUE[value]
    return acc.windows_extra if platform == "win32" else acc.extra


def requirements_file(value: str, platform: str = sys.platform) -> str:
    """The exported locked file of this accelerator: requirements_<extra>.txt."""
    return f"requirements_{lock_extra(value, platform)}.txt"


def installs_pytorch(value: str) -> bool:
    """False for Own PyTorch: WhisperJAV never installs, replaces or removes PyTorch there (D-B)."""
    return bool(_BY_VALUE[value].pytorch_index)


def record_path(prefix: Optional[str] = None) -> Path:
    return Path(prefix or sys.prefix) / STACK_FILE


def read_record(prefix: Optional[str] = None) -> Tuple[bool, Optional[str]]:
    """(the record file exists, its value if it names a known accelerator).

    (False, None): no record (an install made some other way, or before records).
    (True, None): a record this release does not know (damaged, or from a newer release).
    """
    try:
        raw = record_path(prefix).read_text(encoding="utf-8").strip()
    except OSError:
        return False, None
    return True, (raw if raw in _BY_VALUE else None)


def write_record(value: str, prefix: Optional[str] = None) -> None:
    if value not in _BY_VALUE:
        raise ValueError(f"unknown accelerator {value!r}; expected one of {VALUES}")
    record_path(prefix).write_text(value + "\n", encoding="utf-8")
