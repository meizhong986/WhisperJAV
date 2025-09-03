"""
Audio enhancement orchestrator.

Provides a simple registry and runner for enhancement chains
such as denoise, dereverb, separation, normalization, etc.

This module is intentionally minimal; implement steps with
preferred libraries in your environment.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

EnhancementFn = Callable[[Any, int], Tuple[Any, int]]


class EnhancementRegistry:
    def __init__(self) -> None:
        self._fns: Dict[str, EnhancementFn] = {}

    def register(self, name: str, fn: EnhancementFn) -> None:
        self._fns[name] = fn

    def get(self, name: str) -> EnhancementFn | None:
        return self._fns.get(name)


registry = EnhancementRegistry()


def run_chain(waveform: Any, sample_rate: int, steps: List[str]) -> Tuple[Any, int]:
    """
    Run a sequence of enhancement steps registered in the registry.
    Each step is a function: (waveform, sample_rate) -> (waveform, sample_rate)
    """
    out_wav, out_sr = waveform, sample_rate
    for name in steps:
        fn = registry.get(name)
        if fn is None:
            # skip unknown steps, allowing config-driven experimentation
            continue
        out_wav, out_sr = fn(out_wav, out_sr)
    return out_wav, out_sr


# Example placeholder steps (no-ops) — replace with real implementations
def _noop(wav: Any, sr: int) -> Tuple[Any, int]:
    return wav, sr


# Pre-register common names to avoid KeyErrors during experimentation
for _name in ["denoise", "dereverb", "separate", "norm", "agc"]:
    registry.register(_name, _noop)
