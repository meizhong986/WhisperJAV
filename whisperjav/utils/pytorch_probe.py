"""
Is PyTorch there? -- asked at the very top of whisperjav.main, before anything loads it.

WHY THIS EXISTS (design section 22.5.4; review finding 1 of W236 I2)
    whisperjav.main imports the pipelines when it loads, and their ASR modules import
    torch at once. Without PyTorch every command -- a run, --check, --help, --version --
    ended in a ModuleNotFoundError trace before any check could say what was wrong. On an
    Own PyTorch install (owner W232 3, D-B of W234) that is the normal state until the
    user has added their PyTorch build, so the message with the command must show.

    Standard library only (it runs before torch, and before the pipelines are imported).
"""

import importlib.util
import sys

from whisperjav.installer.core import accelerators

# Exit status of a run that did not start because PyTorch is missing (design 22.5.4; 22.13 detail 2).
EXIT_PYTORCH_MISSING = 4

OWN_PYTORCH_MISSING_TEXT = (
    "This install uses your own PyTorch, and none is installed in WhisperJAV's Python ({prefix}). "
    "Install it with: \"{python}\" -m pip install torch torchaudio --index-url "
    "{{the address of your PyTorch build}}"
)
# New (review P7 finding 10; no draft in design 22.10): neither the installer nor whisperjav-upgrade
# touches PyTorch on an Own PyTorch install, so T42's advice would lead nowhere there.
OWN_PYTORCH_DAMAGED_TEXT = (
    "This install uses your own PyTorch, and the one in WhisperJAV's Python ({prefix}) does not load{detail}. "
    "Reinstall it with: \"{python}\" -m pip install --force-reinstall torch torchaudio --index-url "
    "{{the address of your PyTorch build}}"
)
PYTORCH_MISSING_TEXT = (
    "PyTorch, which WhisperJAV needs, is missing or damaged in this install{detail}. "
    "Run the WhisperJAV installer again (or whisperjav-upgrade for a source install)."
)


def pytorch_problem() -> str:
    """'' when torch imports; otherwise the text to show (T42, or T42b on an Own PyTorch install)."""
    detail = ""
    if importlib.util.find_spec("torch") is None:
        missing = True
    else:
        try:
            import torch  # noqa: F401 -- the import itself is the check
            return ""
        except Exception as exc:  # noqa: BLE001 -- any failure means PyTorch cannot be used
            missing = False
            detail = f" ({type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''})"
    if accelerators.read_record()[1] == "own-pytorch":
        text = OWN_PYTORCH_MISSING_TEXT if missing else OWN_PYTORCH_DAMAGED_TEXT
        return text.format(prefix=sys.prefix, python=sys.executable, detail=detail)
    return PYTORCH_MISSING_TEXT.format(detail=detail)


def stop_if_pytorch_missing() -> None:
    """Print the text and exit with status 4 when PyTorch cannot be imported; else return."""
    text = pytorch_problem()
    if text:
        line = "=" * 72
        print(f"\n{line}\n{text}\nNothing was processed.\n{line}\n", file=sys.stderr)
        sys.exit(EXIT_PYTORCH_MISSING)
