"""Helper utilities for ensemble orchestration."""
from __future__ import annotations

from typing import Any, Dict, Optional


def resolve_language_code(pass_config: Optional[Dict[str, Any]], subs_language: str) -> str:
    """Return the effective subtitle language code for a pass.

    Args:
        pass_config: Pipeline configuration dict for the pass (may be None).
        subs_language: High-level subtitle option from CLI/GUI.

    Returns:
        Two-character language code inferred from overrides, params, or defaults.
    """
    if subs_language == "direct-to-english":
        return "en"

    config = pass_config or {}
    overrides = (config.get("overrides") or {})
    params = (config.get("params") or {})
    hf_params = (config.get("hf_params") or {})

    candidates = [
        overrides.get("language"),
        params.get("language"),
        hf_params.get("language"),
        config.get("language"),
    ]

    for value in candidates:
        if value:
            return str(value)

    return "ja"
