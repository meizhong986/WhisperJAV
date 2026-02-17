"""
No-op aligner for aligner-free workflows.

Returns empty alignment results, leaving all timestamp resolution
to the TemporalFramer and hardening stages.
"""

from pathlib import Path
from typing import Any

from whisperjav.modules.subtitle_pipeline.types import AlignmentResult


class NoneAligner:
    """
    No-op TextAligner that returns empty alignment results.

    Used in workflows where timestamps come entirely from the TemporalFramer
    (e.g., SRT-source framer, manual framer) and no forced alignment is needed.
    """

    def align(
        self,
        audio_path: Path,
        text: str,
        language: str = "ja",
        **kwargs: Any,
    ) -> AlignmentResult:
        """Return empty alignment (no words)."""
        return AlignmentResult(words=[], metadata={"skipped": True})

    def align_batch(
        self,
        audio_paths: list[Path],
        texts: list[str],
        language: str = "ja",
        **kwargs: Any,
    ) -> list[AlignmentResult]:
        """Return empty alignments for each input."""
        return [self.align(p, t, language, **kwargs) for p, t in zip(audio_paths, texts)]

    def load(self) -> None:
        """No-op — nothing to load."""

    def unload(self) -> None:
        """No-op — nothing to unload."""

    def cleanup(self) -> None:
        """No-op — nothing to clean up."""
