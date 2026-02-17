"""
Passthrough cleaner — returns text unchanged.

Used for ASR models whose output needs no mid-pipeline cleaning,
or for debugging/comparison to measure the effect of cleaning.
"""

from typing import Any


class PassthroughCleaner:
    """No-op TextCleaner that returns input unchanged."""

    def clean(self, text: str, **kwargs: Any) -> str:
        """Return text unchanged."""
        return text

    def clean_batch(self, texts: list[str], **kwargs: Any) -> list[str]:
        """Return texts unchanged."""
        return list(texts)
