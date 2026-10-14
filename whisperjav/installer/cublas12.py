"""
cuBLAS 12 for CTranslate2 on a CUDA 13 install (v1.10.0, REQ2; item 2 spec C and D).

WHY THIS EXISTS
    The 1.10 stack is PyTorch 2.11.0 for CUDA 13. CTranslate2 -- the engine under
    faster-whisper, so Balanced, Fast, Faster, Kotoba and the whisper-vad segmenter --
    has no CUDA 13 build. Its Windows DLL needs exactly one CUDA library,
    cublas64_12.dll (which needs cublasLt64_12.dll), and nothing on a CUDA 13 PC
    provides it. The installer therefore installs nvidia-cublas-cu12 and places its two
    DLLs in CTranslate2's own package folder: CTranslate2 loads every DLL in that folder
    when it is imported, then asks Windows for cublas64_12.dll by name, and gets the one
    already loaded. No DLL-loading code is added to WhisperJAV itself.

WHAT IS HERE (standard library only; nothing is imported from torch or ctranslate2)
    place()        put the two DLLs in CTranslate2's folder (hard link, copy if that
                   fails) and remove CTranslate2's unused cudnn64_9.dll
    load_check()   load cublas64_12.dll from CTranslate2's folder, as CTranslate2 does,
                   and create and destroy a cuBLAS handle -- run it in a fresh process
    missing_files()  the placed files that are not there (for `whisperjav --check`)
    read_stack() / write_stack()  the stack this PC was installed with: cu130, cu128
                   or cpu, in <install>/whisperjav_stack.txt

    Command line (the installers call it in a fresh process):
        python -m whisperjav.installer.cublas12 place
        python -m whisperjav.installer.cublas12 check
    Exit 0 = done / passed, 1 = failed; one line of explanation on stdout.

TEST SWITCH
    WHISPERJAV_TEST_FORCE_CUBLAS12_FAIL=1 makes load_check() fail, so the CUDA 12.8
    fallback can be shown on a PC where the check passes (spec D, check S4).
"""

from __future__ import annotations

import ctypes
import importlib.util
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

CUBLAS12_PACKAGE = "nvidia-cublas-cu12==12.8.4.1"
CUBLAS12_DLLS = ("cublas64_12.dll", "cublasLt64_12.dll")
UNUSED_CT2_DLLS = ("cudnn64_9.dll",)  # CTranslate2 is built without cuDNN (research F1.3)

STACK_FILE = "whisperjav_stack.txt"
STACKS = ("cu130", "cu128", "cpu")
TEST_FAIL_ENV = "WHISPERJAV_TEST_FORCE_CUBLAS12_FAIL"


def _package_dir(name: str) -> Optional[Path]:
    """The folder of an installed package, found without importing it."""
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def ctranslate2_dir() -> Optional[Path]:
    return _package_dir("ctranslate2")


def cublas12_source_dir() -> Optional[Path]:
    """Where nvidia-cublas-cu12 put its Windows DLLs (site-packages/nvidia/cublas/bin)."""
    base = _package_dir("nvidia.cublas")
    if base is None:
        return None
    return base / "bin"


def place() -> Tuple[bool, str]:
    """Place the cuBLAS 12 DLLs in CTranslate2's folder; remove its unused cuDNN DLL.

    Windows only (elsewhere nothing to do). Safe to run again: an existing placed
    file is replaced, so a newer nvidia-cublas-cu12 is picked up.
    """
    if sys.platform != "win32":
        return True, "not Windows: nothing to place"
    ct2 = ctranslate2_dir()
    if ct2 is None:
        return False, "ctranslate2 is not installed"
    src = cublas12_source_dir()
    if src is None or not src.is_dir():
        return False, f"{CUBLAS12_PACKAGE.split('==')[0]} is not installed"

    actions = []
    for name in CUBLAS12_DLLS:
        source = src / name
        target = ct2 / name
        if not source.is_file():
            return False, f"{source} is missing"
        try:
            if target.exists():
                target.unlink()
        except OSError as e:
            return False, f"could not replace {target}: {e}"
        try:
            os.link(source, target)
            actions.append(f"linked {name}")
        except OSError:
            try:
                shutil.copy2(source, target)
                actions.append(f"copied {name}")
            except OSError as e:
                return False, f"could not place {name} in {ct2}: {e}"
    for name in UNUSED_CT2_DLLS:
        stray = ct2 / name
        if stray.exists():
            try:
                stray.unlink()
                actions.append(f"removed {name}")
            except OSError as e:
                actions.append(f"could not remove {name}: {e}")
    return True, f"{ct2}: " + ", ".join(actions)


def missing_files() -> List[str]:
    """Placed files that are not in CTranslate2's folder (Windows); [] elsewhere."""
    if sys.platform != "win32":
        return []
    ct2 = ctranslate2_dir()
    if ct2 is None:
        return list(CUBLAS12_DLLS)
    return [name for name in CUBLAS12_DLLS if not (ct2 / name).is_file()]


def load_check() -> Tuple[bool, str]:
    """Load cublas64_12.dll from CTranslate2's folder and create a cuBLAS handle.

    Run in a fresh process: a process that has already loaded another
    cublas64_12.dll would test that one instead.
    """
    if os.environ.get(TEST_FAIL_ENV) == "1":
        return False, f"failed on purpose ({TEST_FAIL_ENV}=1)"
    if sys.platform != "win32":
        return True, "not Windows: not checked"
    ct2 = ctranslate2_dir()
    if ct2 is None:
        return False, "ctranslate2 is not installed"
    path = ct2 / CUBLAS12_DLLS[0]
    if not path.is_file():
        return False, f"{path} is missing"
    try:
        lib = ctypes.WinDLL(str(path))
    except OSError as e:
        return False, f"could not load {path}: {e}"
    handle = ctypes.c_void_p()
    try:
        status = lib.cublasCreate_v2(ctypes.byref(handle))
    except Exception as e:  # noqa: BLE001 -- any failure here means "cannot use it"
        return False, f"cublasCreate_v2 failed: {e}"
    if status != 0:
        return False, f"cublasCreate_v2 returned status {status}"
    lib.cublasDestroy_v2(handle)
    return True, f"loaded {path} and created a cuBLAS handle"


def stack_file(prefix: Optional[str] = None) -> Path:
    return Path(prefix or sys.prefix) / STACK_FILE


def read_stack(prefix: Optional[str] = None) -> Optional[str]:
    """'cu130', 'cu128' or 'cpu' as the installer wrote it; None if unknown."""
    try:
        value = stack_file(prefix).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value if value in STACKS else None


def write_stack(stack: str, prefix: Optional[str] = None) -> None:
    if stack not in STACKS:
        raise ValueError(f"unknown stack {stack!r}; expected one of {STACKS}")
    stack_file(prefix).write_text(stack + "\n", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["place"]:
        ok, detail = place()
    elif argv == ["check"]:
        ok, detail = load_check()
    else:
        print("usage: python -m whisperjav.installer.cublas12 place|check")
        return 2
    print(("OK: " if ok else "FAILED: ") + detail)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
