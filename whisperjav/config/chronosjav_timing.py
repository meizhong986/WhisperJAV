#!/usr/bin/env python3
"""
Which timestamp mode a ChronosJAV run with Qwen3-ASR or anime-whisper uses when it names none (1.10.1).

ONE decision, used by every place that needs it, so they cannot disagree:

    - whisperjav/main.py                    (standalone --mode qwen, both models)
    - whisperjav/ensemble/pass_worker.py    (Ensemble passes; the GUI's ChronosJAV rows)
    - tests/test_chronosjav_timing_wiring.py

The rule (owner W498 D1, W501 1): the default is the timestamp mode vad_only with no timing model -- no aligner is
loaded and each subtitle takes its VAD group's edges, ChronosJAV's timing of 1.9.5. A run that names an aligner
(--qwen-aligner, or an Ensemble pass's aligner_id / aligner_backend crispasr or qwen3) but no timestamp mode gets
aligner_vad_fallback instead, so the aligner it names is the aligner that runs, as in 1.10.0. The timing model stays
off unless it is named too. A timestamp mode the run names always wins.

The GUI is not affected: a customized ChronosJAV row sends its timestamp mode with its aligner, and an uncustomized
row sends no aligner. The dormant Cohere-in-ChronosJAV is not covered (it keeps 1.10.0's values).

Light: no imports.
"""

from typing import Any, Optional

DEFAULT_TIMESTAMP_MODE = "vad_only"          # owner W498 D1
DEFAULT_TIMING_MODEL = "none"                # owner W498 D1
ALIGNER_TIMESTAMP_MODE = "aligner_vad_fallback"   # what a named aligner brings (owner W501 1)
ALIGNER_BACKENDS = ("crispasr", "qwen3")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip().lower()


def aligner_named(aligner_id: Optional[Any] = None, aligner_backend: Optional[Any] = None,
                  use_aligner: Optional[Any] = None) -> bool:
    """True when the run names an aligner to use: an aligner model other than "none", or the aligner backend
    crispasr or qwen3. False when it turns the aligner off (aligner backend "none", use_aligner false)."""
    if use_aligner is False or _text(aligner_backend) == "none":
        return False
    if _text(aligner_backend) in ALIGNER_BACKENDS:
        return True
    return _text(aligner_id) not in ("", "none")


def timestamp_mode_for(named_mode: Optional[str], aligner_is_named: bool) -> str:
    """The run's timestamp mode: the one it names, else aligner_vad_fallback when it names an aligner, else
    vad_only."""
    if named_mode:
        return named_mode
    return ALIGNER_TIMESTAMP_MODE if aligner_is_named else DEFAULT_TIMESTAMP_MODE


def timestamp_mode_for_pass(qwen_params: Optional[dict]) -> Optional[str]:
    """For an Ensemble pass's ChronosJAV values (--passN-qwen-params, both key forms): the mode to set when the pass
    names none, or None when it names one (then its own value stands)."""
    p = qwen_params or {}
    if p.get("timestamp_mode") or p.get("qwen_timestamp_mode"):
        return None
    named = aligner_named(p.get("aligner_id", p.get("qwen_aligner")),
                          p.get("aligner_backend", p.get("qwen_aligner_backend")),
                          p.get("use_aligner", p.get("qwen_use_aligner")))
    return timestamp_mode_for(None, named)
