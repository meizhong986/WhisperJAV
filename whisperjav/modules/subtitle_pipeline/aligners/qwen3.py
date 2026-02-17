"""
Qwen3 ForcedAligner TextAligner adapter.

Wraps QwenASR's standalone aligner (load_aligner_only / align_standalone)
behind the TextAligner protocol.  Includes the Qwen3-specific
merge_master_with_timestamps() step that reconciles punctuated ASR text
with unpunctuated aligner timestamps.

Lifecycle design (audit C1 resolution):
    Fresh QwenASR instance per load()/unload() cycle — no stale closures.

VRAM cleanup (audit M5 resolution):
    unload() uses safe_cuda_cleanup() from whisperjav.utils.gpu_utils.
"""

from pathlib import Path
from typing import Any

from whisperjav.modules.subtitle_pipeline.types import AlignmentResult, WordTimestamp
from whisperjav.utils.logger import logger


class Qwen3ForcedAlignerAdapter:
    """
    TextAligner backed by Qwen3's standalone ForcedAligner.

    Produces word-level timestamps by aligning pre-existing text against audio.
    The aligner is a 0.6B parameter non-autoregressive model (~1.2GB VRAM).

    The alignment result is post-processed through merge_master_with_timestamps()
    which reconciles the punctuated ASR text (master) with the unpunctuated
    aligner timestamps.  This step is Qwen3-specific — it understands Qwen3's
    particular text/punctuation format.
    """

    def __init__(
        self,
        aligner_id: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "Japanese",
    ):
        """
        Store configuration for deferred aligner construction.

        Args:
            aligner_id: HuggingFace model ID for the ForcedAligner.
            device: Device ('auto', 'cuda', 'cuda:0', 'cpu').
            dtype: Data type ('auto', 'float16', 'bfloat16', 'float32').
            language: Language for alignment tokenization (canonical name).
        """
        self._config = {
            "aligner_id": aligner_id,
            "device": device,
            "dtype": dtype,
            "language": language,
        }
        self._asr = None  # QwenASR instance, created in load()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the aligner is currently loaded in memory."""
        return self._loaded

    def load(self) -> None:
        """
        Create QwenASR instance and load the standalone aligner into GPU.

        Fresh instance each time — prevents stale closure (audit C1).
        """
        if self._loaded:
            logger.debug("[Qwen3ForcedAligner] Already loaded")
            return

        from whisperjav.modules.qwen_asr import QwenASR

        cfg = self._config
        self._asr = QwenASR(
            device=cfg["device"],
            dtype=cfg["dtype"],
            aligner_id=cfg["aligner_id"],
            use_aligner=True,
            language=cfg["language"],
        )
        self._asr.load_aligner_only()
        self._loaded = True
        logger.info("[Qwen3ForcedAligner] Aligner loaded")

    def unload(self) -> None:
        """Unload the aligner and release VRAM."""
        if not self._loaded:
            return

        if self._asr is not None:
            self._asr.unload_model()
            self._asr = None

        from whisperjav.utils.gpu_utils import safe_cuda_cleanup

        safe_cuda_cleanup()
        self._loaded = False
        logger.info("[Qwen3ForcedAligner] Aligner unloaded")

    def align(
        self,
        audio_path: Path,
        text: str,
        language: str = "ja",
        **kwargs: Any,
    ) -> AlignmentResult:
        """
        Align text to a single audio file.

        Args:
            audio_path: Path to the audio file.
            text: Pre-cleaned text to align (may contain punctuation).
            language: Language code (e.g., 'ja').

        Returns:
            AlignmentResult with word-level timestamps.
        """
        results = self.align_batch(
            audio_paths=[audio_path],
            texts=[text],
            language=language,
            **kwargs,
        )
        return results[0]

    def align_batch(
        self,
        audio_paths: list[Path],
        texts: list[str],
        language: str = "ja",
        **kwargs: Any,
    ) -> list[AlignmentResult]:
        """
        Align text to a batch of audio files.

        Calls QwenASR.align_standalone() then merge_master_with_timestamps()
        for each scene to produce word dicts with punctuation-aware timestamps.

        Args:
            audio_paths: Paths to audio files (one per scene).
            texts: Pre-cleaned text strings (one per scene).
            language: Language code applied to all scenes.

        Returns:
            List of AlignmentResults (one per audio file).
        """
        if not self._loaded:
            raise RuntimeError("Qwen3ForcedAlignerAdapter.align_batch() called before load(). Call load() first.")

        from whisperjav.modules.qwen_asr import merge_master_with_timestamps

        # Extract audio durations if provided in kwargs (for progress display)
        audio_durations = kwargs.get("audio_durations")

        # Call the batch aligner
        raw_results = self._asr.align_standalone(
            audio_paths=list(audio_paths),
            texts=list(texts),
            language=language,
            audio_durations=audio_durations,
        )

        # Post-process each result: merge punctuation from master text
        alignment_results = []
        for i, (raw, master_text) in enumerate(zip(raw_results, texts)):
            if raw is None or not master_text.strip():
                # No alignment for empty/silent scenes
                alignment_results.append(
                    AlignmentResult(
                        words=[],
                        metadata={"scene_index": i, "skipped": True},
                    )
                )
                continue

            # Extract timestamps from aligner result
            timestamps = getattr(raw, "timestamps", None)
            if timestamps is None:
                timestamps = raw if isinstance(raw, list) else []

            # Merge master text (with punctuation) + aligner timestamps
            word_dicts = merge_master_with_timestamps(master_text, timestamps)

            # Convert raw dicts to WordTimestamp objects
            words = [
                WordTimestamp(
                    word=wd["word"],
                    start=wd["start"],
                    end=wd["end"],
                )
                for wd in word_dicts
            ]

            alignment_results.append(
                AlignmentResult(
                    words=words,
                    metadata={
                        "scene_index": i,
                        "aligner": "qwen3",
                        "raw_word_count": len(timestamps) if timestamps else 0,
                        "merged_word_count": len(words),
                    },
                )
            )

        return alignment_results

    def cleanup(self) -> None:
        """Final cleanup — unload if still loaded."""
        self.unload()
