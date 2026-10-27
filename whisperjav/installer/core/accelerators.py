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


# ---------------------------------------------------------------------------------------------
# The installers' texts (design 22.10; agreed W234 D-H, corrected W238): the same question in
# the same words in the Windows installer's window and in install.py's console.
# ---------------------------------------------------------------------------------------------

ISSUES_URL = "https://github.com/meizhong986/WhisperJAV/issues"
RELEASE_195_URL = "https://github.com/meizhong986/WhisperJAV/releases/tag/v1.9.5"
NVIDIA_DRIVER_URL = "https://www.nvidia.com/drivers"

WINDOW_TITLE = "WhisperJAV — no supported NVIDIA GPU found"                                   # T25
WINDOW_INTRO = ("WhisperJAV found no NVIDIA graphics card that version 1.10 supports{found}. "
                "What should this install be made for?")
WINDOW_FOUND = " (graphics found: {names})"
# (answer, label, accelerator); the first three give the GPU-through-Vulkan install (22.6.2).
WINDOW_CHOICES = (
    ("1", "AMD graphics card", "vulkan"),
    ("2", "Intel graphics card", "vulkan"),
    ("3", "NVIDIA graphics card older than WhisperJAV 1.10 supports", "vulkan"),
    ("4", "I have my own PyTorch for my GPU (advanced)", "own-pytorch"),
    ("5", "None, or use the CPU only", "cpu"),
)
WINDOW_EXPLANATION = (
    "With an AMD, Intel or older NVIDIA card, the GGUF pipelines (Parakeet-CrispASR and Cohere-CrispASR) "
    "use your graphics card through Vulkan. This is experimental in 1.10.0: it has not been tested, and "
    "we would like to hear how it works for you. All other pipelines run on the CPU (not accelerated), "
    "several times slower than on a supported NVIDIA GPU. With your own PyTorch, WhisperJAV installs "
    "everything except PyTorch, and you add your PyTorch build afterwards (the command is shown at the "
    "end). With the CPU only, every pipeline runs on the CPU.")
WINDOW_FOOTER = "Without an answer, the CPU-only version is installed in 20 seconds."
WINDOW_TIMEOUT_SECONDS = 20
NOT_REACHABLE_TEXT = (                                                                        # T25b
    "Windows lists an NVIDIA graphics card ({name}), but WhisperJAV cannot reach it: its driver may be "
    "missing or not working, or, on a laptop with two graphics chips, the NVIDIA chip may be switched off. "
    "Install or repair the NVIDIA driver (580 or newer, " + NVIDIA_DRIVER_URL + "), restart the PC, and "
    "run this installer again to get the NVIDIA GPU version. If the driver is fine, please tell us: "
    + ISSUES_URL + ". Or choose below.")
OLD_CARD_TEXT = ("Your graphics card, {name} (compute capability {cc}), is older than WhisperJAV 1.10 "   # T29, W43
                 "supports. 1.10 needs a GTX 16-series, RTX 20-series or newer card.")
OLD_CARD_195_TEXT = "WhisperJAV 1.9.5 still supports this card: " + RELEASE_195_URL + "."
IDENTIFIED_WRONGLY = "My card is identified wrongly"
REPORT_IT_TEXT = "Please report it here: " + ISSUES_URL + " (attach the install log)."
OLD_DRIVER_TEXT = ("Your NVIDIA driver ({version}) is older than WhisperJAV 1.10 needs ({minimum} or newer). "  # T47w
                   "Update it from " + NVIDIA_DRIVER_URL + ", restart the PC, and run this installer again "
                   "to get the NVIDIA GPU version. Or choose below.")
SECOND_RUN_TITLE = "WhisperJAV — this install's accelerator"                                  # T25r
SECOND_RUN_TEXT = "This install is made for: {name}. Keep it, or choose another."
SECOND_RUN_FOOTER = "Without an answer, the install is kept as it is ({name}) in 20 seconds."
NO_VULKAN_TEXT = (                                                                            # T26
    "Vulkan (vulkan-1.dll) was not found on this PC, so WhisperJAV is installed as the CPU-only version: "
    "every pipeline runs on the CPU (not accelerated). To use your graphics card with the GGUF pipelines, "
    "update its driver and run the WhisperJAV installer again.")
CONSOLE_PROMPT_WINDOWS = ("Type 1 = AMD, 2 = Intel, 3 = older NVIDIA card, 4 = my own PyTorch, 5 = CPU only, "  # T28
                          "6 = stop{extra}, then Enter: ")
CONSOLE_PROMPT_LINUX = ("No supported NVIDIA GPU found. What should this install be made for? 1 = my own "    # T28L
                        "PyTorch (already installed in this Python environment, for example ROCm for an AMD "
                        "GPU); 2 = CPU only (not accelerated); 3 = stop{extra}. Type 1, 2 or 3{more} and press "
                        "Enter: ")
CONSOLE_WRONGLY_EXTRA = ", {n} = my card is identified wrongly (show where to report it, then stop)"
MAC_CPU_TEXT = ("On an Apple silicon Mac, WhisperJAV is installed for the Apple GPU (Metal), which includes "   # T35c
                "the CPU. To run one job on the CPU, add --device cpu to that run.")
VULKAN_NOT_HERE_TEXT = ("--accelerator vulkan is for Windows PCs only: CrispASR has no Vulkan package for "    # T35b
                        "{system}. Nothing was installed.")
FORCE_NVIDIA_HINT = ("If this PC has an NVIDIA GPU that was not found (for example a laptop with two GPUs), "  # T36
                     "run again with --accelerator nvidia.")
APPLE_INSTALL_TEXT = ("Apple silicon Mac: installing for the Apple GPU (Metal) — PyTorch with Apple GPU "      # T30
                      "support, and CrispASR's macOS build.")
OWN_PYTORCH_NEXT_STEP = (                                                                     # T42a
    "WhisperJAV is installed without PyTorch, as you chose. Before it can run, install your PyTorch build "
    "(2.11 recommended) into WhisperJAV's Python: \"{python}\" -m pip install torch torchaudio --index-url "
    "{{the address of your PyTorch build}}. WhisperJAV will never replace it. This command is also saved "
    "in {file}.")
OWN_PYTORCH_NEXT_STEP_FILE = "OWN_PYTORCH_NEXT_STEP.txt"
OWN_PYTORCH_NONE_TEXT = ("--accelerator own-pytorch needs your PyTorch and torchaudio already installed in "  # T42c
                         "this Python environment; none was found. Nothing was installed.")
OWN_PYTORCH_OTHER_VERSION = ("This environment has PyTorch {version}. WhisperJAV 1.10.0 is built for "
                             "PyTorch 2.11; other versions have not been tested.")
OWN_PYTORCH_DONE = ("Installed without PyTorch: WhisperJAV uses your PyTorch {version} ({where}). "
                    "whisperjav-upgrade will keep it.")
MADE_FOR_TEXT = "Made for: {name}."                                                           # T29b
