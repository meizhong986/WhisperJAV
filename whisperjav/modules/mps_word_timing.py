"""openai-whisper's word-timing step on the Apple GPU (1.10.3, owner W750 A4; #515).

openai-whisper's ``whisper/timing.py`` ``dtw()`` converts the alignment table to float64 while it is still on the
device (``x.double().cpu()``), which the Apple GPU (MPS) cannot do: the fidelity pipeline then stops with "Cannot
convert a MPS Tensor to float64 dtype" when word timestamps are on. The fix moves the table to the CPU first, then
converts it (``x.cpu().double()``): only the order of two calls changes, and the decoding stays on the Apple GPU. The
table is the model's attention between the text and the audio of one window, used only to find word times.

The same one-line change is openai-whisper's pull requests #2704 and #2825 (not merged on 2026-11-23). This
replacement is installed only when fidelity runs on the Apple GPU (modules/whisper_pro_asr.py), and only for tensors on
the Apple GPU; every other tensor takes openai-whisper's own ``dtw()``. Keep it until openai-whisper fixes it.
"""

from typing import Any


def _on_apple_gpu(x: Any) -> bool:
    device = getattr(x, "device", None)
    return getattr(device, "type", None) == "mps"


def install() -> bool:
    """Replace ``whisper.timing.dtw`` once per process with the Apple-GPU-safe order. Returns True when in place."""
    import whisper.timing as timing

    if getattr(timing.dtw, "_whisperjav_apple_gpu", False):
        return True
    original = timing.dtw

    def dtw(x):
        if _on_apple_gpu(x):
            return timing.dtw_cpu(x.cpu().double().numpy())   # to the CPU first, then float64
        return original(x)

    dtw._whisperjav_apple_gpu = True
    dtw._whisperjav_original = original
    timing.dtw = dtw
    return True
