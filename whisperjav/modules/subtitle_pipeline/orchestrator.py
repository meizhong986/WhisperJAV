"""
Decoupled Subtitle Pipeline orchestrator.

Composes TemporalFramer, TextGenerator, TextCleaner, and TextAligner
protocol implementations into a working pipeline.  Handles the 9-step
processing flow defined in ADR-006 Section 9.2:

    1. Temporal framing (per scene)
    2. Audio slicing (per frame → temp WAV)
    3. Text generation (batch, with VRAM lifecycle)
    4. Text cleaning (batch, lightweight)
    5-7. Alignment (batch, with VRAM lifecycle)
    8. Word merging (frame-relative → scene-relative)
    9. Sentinel + Reconstruction + Hardening (per scene)

VRAM swap pattern:
    generator.load() → generate → generator.unload()
    → safe_cuda_cleanup()
    → aligner.load() → align → aligner.unload()
    → safe_cuda_cleanup()
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.hardening import harden_scene_result
from whisperjav.modules.subtitle_pipeline.protocols import (
    TemporalFramer,
    TextAligner,
    TextCleaner,
    TextGenerator,
)
from whisperjav.modules.subtitle_pipeline.reconstruction import reconstruct_from_words
from whisperjav.modules.subtitle_pipeline.types import (
    HardeningConfig,
    SceneDiagnostics,
    StepDownConfig,
    TemporalFrame,
    TimestampMode,
)
from whisperjav.utils.logger import logger

try:
    import soundfile as sf
except ImportError:
    sf = None  # type: ignore[assignment]


class DecoupledSubtitlePipeline:
    """
    Model-agnostic subtitle generation pipeline.

    Composes protocol-based components into a working pipeline with
    explicit VRAM lifecycle management, alignment sentinel integration,
    and per-scene diagnostics.
    """

    def __init__(
        self,
        framer: TemporalFramer,
        generator: TextGenerator,
        cleaner: TextCleaner,
        aligner: Optional[TextAligner],
        hardening_config: HardeningConfig,
        artifacts_dir: Optional[Path] = None,
        language: str = "ja",
        context: str = "",
        stepdown_config: Optional[StepDownConfig] = None,
    ):
        """
        Initialize the pipeline with protocol components.

        Args:
            framer: Produces temporal frames from scene audio.
            generator: Produces text from audio.
            cleaner: Cleans raw transcription text.
            aligner: Aligns text to audio for word-level timestamps.
                     None for aligner-free workflows.
            hardening_config: Timestamp resolution and boundary config.
            artifacts_dir: Directory for debug artifacts (None = no artifacts).
            language: Language code for generation and alignment.
            context: User-provided context for ASR (cast names, terminology).
            stepdown_config: Optional step-down retry config. When enabled
                and alignment collapses on a scene, the orchestrator re-frames
                with tighter grouping and retries generation + alignment.
        """
        self.framer = framer
        self.generator = generator
        self.cleaner = cleaner
        self.aligner = aligner
        self.hardening_config = hardening_config
        self.artifacts_dir = artifacts_dir
        self.language = language
        self.context = context
        self.stepdown_config = stepdown_config

        # Sentinel stats accumulated across all scenes
        self.sentinel_stats: dict[str, Any] = {
            "total_scenes": 0,
            "collapsed_scenes": 0,
            "recovered_scenes": 0,
            "recovery_strategies": {"vad_guided": 0, "proportional": 0},
        }

        # Temp files for cleanup
        self._temp_files: list[Path] = []

    def process_scenes(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Process all scenes through the decoupled pipeline.

        When step-down retry is enabled, collapsed scenes are automatically
        retried with tighter temporal framing (Pass 2).

        Args:
            scene_audio_paths: Paths to per-scene audio files (WAV, 16kHz).
            scene_durations: Duration of each scene in seconds.
            scene_speech_regions: Optional per-scene VAD speech regions
                (from Phase 4 or VadGroupedFramer metadata).

        Returns:
            List of (WhisperResult_or_None, diagnostics_dict) per scene.
        """
        n_scenes = len(scene_audio_paths)
        if n_scenes != len(scene_durations):
            raise ValueError(f"scene_audio_paths ({n_scenes}) and scene_durations ({len(scene_durations)}) must match")

        logger.info(
            "[DecoupledPipeline] Processing %d scenes (aligner=%s, step-down=%s)",
            n_scenes,
            "yes" if self.aligner else "none",
            "enabled" if (self.stepdown_config and self.stepdown_config.enabled) else "disabled",
        )

        # --- Pass 1: Normal processing ---
        results = self._run_pass(scene_audio_paths, scene_durations, scene_speech_regions)

        # --- Identify collapsed scenes ---
        collapsed_indices = [
            i for i, (_result, diag) in enumerate(results)
            if self._is_collapsed(diag)
        ]

        if not collapsed_indices:
            return results

        # --- Step-down decision ---
        can_reframe = hasattr(self.framer, "reframe")
        stepdown_enabled = (
            self.stepdown_config is not None
            and self.stepdown_config.enabled
            and can_reframe
        )

        if stepdown_enabled:
            logger.info(
                "[DecoupledPipeline] Step-down: %d/%d scenes collapsed, "
                "retrying with tighter framing (%.1fs max group)",
                len(collapsed_indices), n_scenes,
                self.stepdown_config.fallback_max_group_s,
            )
            retry_results = self._run_stepdown_pass(
                collapsed_indices, scene_audio_paths,
                scene_durations, scene_speech_regions,
            )
            # Replace Pass 1 results for retried scenes with Pass 2 results
            for idx, retry_result in zip(collapsed_indices, retry_results):
                _pass1_diag = results[idx][1]
                _pass2_result, _pass2_diag = retry_result
                # Annotate step-down outcome
                improved = not self._is_collapsed(_pass2_diag)
                _pass2_diag["stepdown"] = {
                    "attempted": True,
                    "enabled": True,
                    "improved": improved,
                    "pass1_sentinel": _pass1_diag.get("sentinel_status", "N/A"),
                    "pass2_sentinel": _pass2_diag.get("sentinel_status", "N/A"),
                    "fallback_max_group_s": self.stepdown_config.fallback_max_group_s,
                }
                if improved:
                    results[idx] = retry_result
                    logger.info(
                        "[DecoupledPipeline] Step-down: Scene %d improved (was COLLAPSED → %s)",
                        idx, _pass2_diag.get("sentinel_status", "?"),
                    )
                else:
                    # Pass 2 also collapsed — keep Pass 2 result anyway
                    # (proportional recovery was already applied in _step9)
                    results[idx] = retry_result
                    logger.warning(
                        "[DecoupledPipeline] Step-down: Scene %d still collapsed after retry",
                        idx,
                    )
        else:
            # Step-down disabled or not available
            if self.stepdown_config and not self.stepdown_config.enabled:
                reason = "disabled by user configuration"
            elif not can_reframe:
                reason = "framer does not support reframing"
            else:
                reason = "not configured"
            logger.warning(
                "[DecoupledPipeline] %d/%d scenes collapsed but step-down retry is %s. "
                "Proportional recovery applied.",
                len(collapsed_indices), n_scenes, reason,
            )
            # Annotate diagnostics for collapsed scenes
            for idx in collapsed_indices:
                results[idx][1]["stepdown"] = {
                    "attempted": False,
                    "enabled": False if (self.stepdown_config and not self.stepdown_config.enabled) else None,
                    "improved": False,
                }

        return results

    # -----------------------------------------------------------------------
    # Pass execution (shared by Pass 1 and step-down Pass 2)
    # -----------------------------------------------------------------------

    def _run_pass(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
        framer_override_max_group: Optional[float] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Execute a single pass of the pipeline (framing → generation → alignment → hardening).

        Args:
            scene_audio_paths: Per-scene audio file paths.
            scene_durations: Per-scene durations.
            scene_speech_regions: Optional per-scene VAD speech regions.
            framer_override_max_group: When set, calls framer.reframe()
                with this max group duration instead of framer.frame().
        """
        try:
            scene_frames, frame_audio_paths, frame_speech_regions = self._step1_frame_and_slice(
                scene_audio_paths, scene_durations,
                framer_override_max_group=framer_override_max_group,
            )
            scene_texts = self._step2_4_generate_and_clean(scene_frames, frame_audio_paths, scene_durations)
            scene_alignments = self._step5_7_align(scene_frames, frame_audio_paths, scene_texts, scene_durations)
            results = self._step9_reconstruct_and_harden(
                scene_frames, scene_texts, scene_alignments,
                scene_audio_paths, scene_durations,
                frame_speech_regions, scene_speech_regions,
            )
        finally:
            self._cleanup_temp_files()
        return results

    def _run_stepdown_pass(
        self,
        collapsed_indices: list[int],
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Re-process collapsed scenes with tighter framing (step-down retry)."""
        retry_audio_paths = [scene_audio_paths[i] for i in collapsed_indices]
        retry_durations = [scene_durations[i] for i in collapsed_indices]
        retry_speech_regions = (
            [scene_speech_regions[i] for i in collapsed_indices]
            if scene_speech_regions else None
        )
        return self._run_pass(
            retry_audio_paths, retry_durations, retry_speech_regions,
            framer_override_max_group=self.stepdown_config.fallback_max_group_s,
        )

    @staticmethod
    def _is_collapsed(diag: dict[str, Any]) -> bool:
        """Check if a scene's diagnostics indicate alignment collapse."""
        return diag.get("sentinel_status") == "COLLAPSED"

    # -----------------------------------------------------------------------
    # Step 1: Temporal framing + audio slicing
    # -----------------------------------------------------------------------

    def _step1_frame_and_slice(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        framer_override_max_group: Optional[float] = None,
    ) -> tuple[
        list[list[TemporalFrame]],
        list[list[Path]],
        list[Optional[list[list[tuple[float, float]]]]],
    ]:
        """
        Frame each scene and slice audio per frame into temp WAV files.

        Args:
            framer_override_max_group: When set and the framer supports
                ``reframe()``, uses tighter grouping (step-down retry).

        Returns:
            scene_frames: Per-scene list of TemporalFrame objects.
            frame_audio_paths: Per-scene, per-frame temp WAV paths.
            frame_speech_regions: Per-scene speech regions from framer metadata.
        """
        n_scenes = len(scene_audio_paths)
        logger.info(
            "[DecoupledPipeline] Step 1: Framing %d scenes%s", n_scenes,
            f" (reframe override={framer_override_max_group}s)" if framer_override_max_group else "",
        )

        scene_frames: list[list[TemporalFrame]] = []
        frame_audio_paths: list[list[Path]] = []
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]] = []

        for scene_idx, audio_path in enumerate(scene_audio_paths):
            # Load scene audio
            audio, sr = self._load_audio(audio_path)

            # Run framer (or reframe for step-down)
            if framer_override_max_group is not None and hasattr(self.framer, "reframe"):
                framing_result = self.framer.reframe(
                    audio, sr, max_group_duration_s=framer_override_max_group,
                )
            else:
                framing_result = self.framer.frame(audio, sr)
            frames = framing_result.frames
            scene_frames.append(frames)

            # Extract speech regions from framer metadata (VadGroupedFramer provides these)
            regions = framing_result.metadata.get("speech_regions")
            frame_speech_regions.append(regions)

            # Slice audio per frame → temp WAV
            frame_paths = []
            for frame_idx, frame in enumerate(frames):
                if len(frames) == 1 and frame.start == 0.0:
                    # Full-scene frame — use original audio path (no slicing needed)
                    frame_paths.append(audio_path)
                else:
                    # Slice and write temp WAV
                    start_sample = int(frame.start * sr)
                    end_sample = int(frame.end * sr)
                    frame_audio = audio[start_sample:end_sample]
                    temp_path = self._write_temp_wav(frame_audio, sr, scene_idx, frame_idx)
                    frame_paths.append(temp_path)

            frame_audio_paths.append(frame_paths)

            logger.debug(
                "[DecoupledPipeline] Scene %d: %d frames (%.1fs)",
                scene_idx,
                len(frames),
                scene_durations[scene_idx],
            )

        total_frames = sum(len(f) for f in scene_frames)
        logger.info(
            "[DecoupledPipeline] Step 1: Complete — %d scenes, %d total frames",
            n_scenes, total_frames,
        )

        return scene_frames, frame_audio_paths, frame_speech_regions

    # -----------------------------------------------------------------------
    # Steps 2-4: Text generation + cleaning
    # -----------------------------------------------------------------------

    def _step2_4_generate_and_clean(
        self,
        scene_frames: list[list[TemporalFrame]],
        frame_audio_paths: list[list[Path]],
        scene_durations: list[float],
    ) -> list[list[str]]:
        """
        Generate text for each frame, then clean.

        VRAM lifecycle: generator.load() → generate all → generator.unload()

        Returns:
            scene_texts: Per-scene, per-frame cleaned text strings.
        """
        import time as _time

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Steps 2-4: Generating + cleaning text for %d scenes",
            n_scenes,
        )
        step24_start = _time.monotonic()

        # Collect frames that need generation (no pre-existing text)
        needs_generation = False
        for frames in scene_frames:
            for frame in frames:
                if frame.text is None:
                    needs_generation = True
                    break
            if needs_generation:
                break

        # Phase 1: Generation
        scene_raw_texts: list[list[str]] = []

        if needs_generation:
            self.generator.load()

        try:
            for scene_idx in range(n_scenes):
                frames = scene_frames[scene_idx]
                audio_paths = frame_audio_paths[scene_idx]
                raw_texts = []

                logger.info(
                    "[DecoupledPipeline] Generating scene %d/%d (%.1fs audio)...",
                    scene_idx + 1, n_scenes, scene_durations[scene_idx],
                )

                # Separate framer-provided and needs-generation frames
                gen_indices = []
                gen_audio_paths = []
                for frame_idx, frame in enumerate(frames):
                    if frame.text is not None:
                        raw_texts.append(frame.text)
                    else:
                        raw_texts.append(None)  # placeholder
                        gen_indices.append(frame_idx)
                        gen_audio_paths.append(audio_paths[frame_idx])

                # Batch generate for frames without text
                if gen_indices:
                    try:
                        gen_contexts = [self.context] * len(gen_audio_paths) if self.context else None
                        gen_results = self.generator.generate_batch(
                            audio_paths=gen_audio_paths,
                            language=self.language,
                            contexts=gen_contexts,
                            audio_durations=[frames[i].duration for i in gen_indices],
                        )
                        for i, gen_idx in enumerate(gen_indices):
                            raw_texts[gen_idx] = gen_results[i].text
                    except Exception:
                        # Batch failed — fall back to per-frame
                        logger.warning(
                            "[DecoupledPipeline] Batch generation failed for scene %d, falling back to per-frame",
                            scene_idx,
                            exc_info=True,
                        )
                        for i, gen_idx in enumerate(gen_indices):
                            try:
                                result = self.generator.generate(
                                    audio_path=gen_audio_paths[i],
                                    language=self.language,
                                    context=self.context if self.context else None,
                                )
                                raw_texts[gen_idx] = result.text
                            except Exception:
                                logger.error(
                                    "[DecoupledPipeline] Generation failed for scene %d frame %d",
                                    scene_idx,
                                    gen_idx,
                                    exc_info=True,
                                )
                                raw_texts[gen_idx] = ""

                # Replace any remaining None with empty string
                raw_texts = [t if t is not None else "" for t in raw_texts]
                scene_raw_texts.append(raw_texts)

                # Per-scene generation result
                scene_chars = sum(len(t) for t in raw_texts)
                if scene_chars == 0:
                    logger.info(
                        "[DecoupledPipeline]   Scene %d/%d: empty (no text generated)",
                        scene_idx + 1, n_scenes,
                    )
                else:
                    logger.debug(
                        "[DecoupledPipeline]   Scene %d/%d: %d chars",
                        scene_idx + 1, n_scenes, scene_chars,
                    )

                # Save raw text artifacts
                if self.artifacts_dir:
                    self._save_artifact(scene_idx, "raw", "\n---\n".join(raw_texts))

        finally:
            if needs_generation:
                self.generator.unload()
                self._safe_cuda_cleanup()

        # Phase 2: Cleaning
        logger.info("[DecoupledPipeline] Cleaning %d scenes", n_scenes)
        scene_texts: list[list[str]] = []
        for scene_idx, raw_texts in enumerate(scene_raw_texts):
            clean_texts = self.cleaner.clean_batch(raw_texts)
            scene_texts.append(clean_texts)

            # Save clean text artifacts
            if self.artifacts_dir:
                self._save_artifact(scene_idx, "clean", "\n---\n".join(clean_texts))

        # Step summary
        total_raw_chars = sum(len(t) for texts in scene_raw_texts for t in texts)
        total_clean_chars = sum(len(t) for texts in scene_texts for t in texts)
        n_empty = sum(1 for texts in scene_texts if all(not t.strip() for t in texts))
        elapsed = _time.monotonic() - step24_start
        logger.info(
            "[DecoupledPipeline] Steps 2-4: Complete — %d scenes, %d chars (%d removed by cleaning), %d empty (%.1fs)",
            n_scenes, total_clean_chars, total_raw_chars - total_clean_chars, n_empty, elapsed,
        )

        return scene_texts

    # -----------------------------------------------------------------------
    # Steps 5-7: Alignment
    # -----------------------------------------------------------------------

    def _step5_7_align(
        self,
        scene_frames: list[list[TemporalFrame]],
        frame_audio_paths: list[list[Path]],
        scene_texts: list[list[str]],
        scene_durations: list[float],
    ) -> Optional[list[list[list[dict[str, Any]]]]]:
        """
        Align text to audio for word-level timestamps.

        VRAM lifecycle: aligner.load() → align all → aligner.unload()

        Returns:
            None if no aligner, otherwise:
            scene_alignments[scene_idx][frame_idx] = list of word dicts
            Each word dict: {'word': str, 'start': float, 'end': float}
        """
        if self.aligner is None:
            return None

        import time as _time

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Steps 5-7: Aligning %d scenes", n_scenes,
        )
        step57_start = _time.monotonic()
        scene_alignments: list[list[list[dict[str, Any]]]] = []

        self.aligner.load()
        try:
            for scene_idx in range(n_scenes):
                frames = scene_frames[scene_idx]
                audio_paths = frame_audio_paths[scene_idx]
                texts = scene_texts[scene_idx]

                # Separate frames that need alignment from empty ones
                batch_indices: list[int] = []
                batch_audio_paths: list[Path] = []
                batch_texts: list[str] = []
                batch_durations: list[float] = []

                for frame_idx, (frame, audio_path, text) in enumerate(zip(frames, audio_paths, texts)):
                    if text.strip():
                        batch_indices.append(frame_idx)
                        batch_audio_paths.append(audio_path)
                        batch_texts.append(text)
                        batch_durations.append(frame.duration)

                scene_chars = sum(len(t) for t in texts if t.strip())
                logger.info(
                    "[DecoupledPipeline] Aligning scene %d/%d (%.1fs audio, %d chars)...",
                    scene_idx + 1, n_scenes, scene_durations[scene_idx], scene_chars,
                )

                # Initialize all frames as empty
                frame_alignments: list[list[dict[str, Any]]] = [[] for _ in frames]

                if batch_indices:
                    try:
                        # Batch align all non-empty frames for this scene
                        batch_results = self.aligner.align_batch(
                            audio_paths=batch_audio_paths,
                            texts=batch_texts,
                            language=self.language,
                            audio_durations=batch_durations,
                        )
                        # Unpack results back to per-frame positions
                        for i, frame_idx in enumerate(batch_indices):
                            word_dicts = [
                                {
                                    "word": w.word,
                                    "start": w.start,
                                    "end": w.end,
                                }
                                for w in batch_results[i].words
                            ]
                            frame_alignments[frame_idx] = word_dicts
                    except Exception:
                        # Batch failed — fall back to per-frame alignment
                        logger.warning(
                            "[DecoupledPipeline] Batch alignment failed for scene %d, falling back to per-frame",
                            scene_idx,
                            exc_info=True,
                        )
                        for i, frame_idx in enumerate(batch_indices):
                            try:
                                align_result = self.aligner.align(
                                    audio_path=batch_audio_paths[i],
                                    text=batch_texts[i],
                                    language=self.language,
                                    audio_durations=[batch_durations[i]],
                                )
                                word_dicts = [
                                    {
                                        "word": w.word,
                                        "start": w.start,
                                        "end": w.end,
                                    }
                                    for w in align_result.words
                                ]
                                frame_alignments[frame_idx] = word_dicts
                            except Exception:
                                logger.error(
                                    "[DecoupledPipeline] Alignment failed for scene %d frame %d",
                                    scene_idx,
                                    frame_idx,
                                    exc_info=True,
                                )

                scene_alignments.append(frame_alignments)

                scene_words = sum(len(fa) for fa in frame_alignments)
                logger.debug(
                    "[DecoupledPipeline]   Scene %d/%d: %d words aligned",
                    scene_idx + 1, n_scenes, scene_words,
                )

                # Save alignment artifacts
                if self.artifacts_dir:
                    self._save_artifact(
                        scene_idx,
                        "aligned",
                        json.dumps(frame_alignments, ensure_ascii=False, indent=2),
                        ext=".json",
                    )

        finally:
            self.aligner.unload()
            self._safe_cuda_cleanup()

        total_words = sum(
            len(w) for fa_list in scene_alignments for w in fa_list
        )
        elapsed = _time.monotonic() - step57_start
        logger.info(
            "[DecoupledPipeline] Steps 5-7: Complete — %d scenes aligned, %d total words (%.1fs)",
            n_scenes, total_words, elapsed,
        )

        return scene_alignments

    # -----------------------------------------------------------------------
    # Step 9: Reconstruction + Sentinel + Hardening
    # -----------------------------------------------------------------------

    def _step9_reconstruct_and_harden(
        self,
        scene_frames: list[list[TemporalFrame]],
        scene_texts: list[list[str]],
        scene_alignments: Optional[list[list[list[dict[str, Any]]]]],
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]],
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Per-scene: merge words → sentinel → reconstruct → harden.

        Returns list of (WhisperResult_or_None, diagnostics) per scene.
        """
        from whisperjav.modules.alignment_sentinel import (
            assess_alignment_quality,
            redistribute_collapsed_words,
        )

        n_scenes = len(scene_frames)
        logger.info(
            "[DecoupledPipeline] Step 9: Reconstructing %d scenes", n_scenes,
        )

        results: list[tuple[Any, dict[str, Any]]] = []
        total_segments = 0
        total_collapses = 0

        for scene_idx in range(n_scenes):
            frames = scene_frames[scene_idx]
            texts = scene_texts[scene_idx]
            duration = scene_durations[scene_idx]
            audio_path = scene_audio_paths[scene_idx]

            self.sentinel_stats["total_scenes"] += 1

            try:
                word_count = 0
                assessment = None
                recovery_info = None

                if scene_alignments is not None:
                    # Branch A: Aligned workflow — merge frame-relative → scene-relative
                    all_words = self._merge_frame_words(frames, scene_alignments[scene_idx])
                    word_count = len(all_words)

                    # Sentinel assessment (always run for diagnostics visibility)
                    assessment = assess_alignment_quality(all_words, duration)
                    sentinel_status = assessment["status"]

                    # G3 fix: aligner_only mode wants raw aligner output —
                    # assess for diagnostics but skip recovery even if collapsed.
                    skip_recovery = (
                        self.hardening_config.timestamp_mode == TimestampMode.ALIGNER_ONLY
                    )

                    if sentinel_status == "COLLAPSED" and not skip_recovery:
                        # Standard recovery path (aligner_interpolation, aligner_vad_fallback)
                        self.sentinel_stats["collapsed_scenes"] += 1
                        total_collapses += 1
                        logger.warning(
                            "[SENTINEL] Scene %d/%d: Alignment Collapse — "
                            "coverage=%.1f%%, CPS=%.1f, %d chars in %.3fs span",
                            scene_idx + 1, n_scenes,
                            assessment["coverage_ratio"] * 100,
                            assessment["aggregate_cps"],
                            assessment["char_count"],
                            assessment["word_span_sec"],
                        )

                        # Get speech regions for recovery
                        regions = self._get_speech_regions(scene_idx, frame_speech_regions, scene_speech_regions)

                        corrected_words = redistribute_collapsed_words(all_words, duration, regions)
                        self.sentinel_stats["recovered_scenes"] += 1
                        strategy = "vad_guided" if regions else "proportional"
                        self.sentinel_stats["recovery_strategies"][strategy] += 1
                        recovery_info = {
                            "strategy": strategy,
                            "words_redistributed": len(corrected_words),
                        }

                        # Reconstruct with suppress_silence=False (H3 fix)
                        result = reconstruct_from_words(corrected_words, audio_path, suppress_silence=False)

                    elif sentinel_status == "COLLAPSED" and skip_recovery:
                        # aligner_only: log collapse but keep raw aligner timestamps
                        self.sentinel_stats["collapsed_scenes"] += 1
                        total_collapses += 1
                        logger.info(
                            "[SENTINEL] Scene %d/%d: COLLAPSED but aligner_only mode "
                            "— keeping raw aligner timestamps (no recovery)",
                            scene_idx + 1, n_scenes,
                        )
                        result = reconstruct_from_words(all_words, audio_path, suppress_silence=False)
                    else:
                        # OK path — ForcedAligner timestamps are already accurate.
                        # Don't let stable-ts's crude loudness quantizer shrink them.
                        result = reconstruct_from_words(all_words, audio_path, suppress_silence=False)

                else:
                    # Branch B: Aligner-free — build word dicts from frame boundaries
                    words = []
                    for frame, text in zip(frames, texts):
                        if text.strip():
                            words.append(
                                {
                                    "word": text,
                                    "start": frame.start,
                                    "end": frame.end,
                                }
                            )
                    word_count = len(words)
                    # G1 complement: VAD_ONLY preserves exact frame boundaries —
                    # suppress_silence=False prevents stable-ts silence detection
                    # from shifting VAD group timing. Other aligner-free modes
                    # (future) may want stable-ts adjustment.
                    suppress = (
                        self.hardening_config.timestamp_mode != TimestampMode.VAD_ONLY
                    )
                    result = reconstruct_from_words(words, audio_path, suppress_silence=suppress)
                    sentinel_status = "N/A"

                # Hardening (shared by all paths)
                # Resolve per-scene speech regions for VAD_ONLY mode
                per_scene_regions = self._get_speech_regions(
                    scene_idx, frame_speech_regions, scene_speech_regions,
                )
                config = HardeningConfig(
                    timestamp_mode=self.hardening_config.timestamp_mode,
                    scene_duration_sec=duration,
                    speech_regions=per_scene_regions,
                )
                hardening_diag = harden_scene_result(result, config)

                segment_count = len(result.segments) if result and result.segments else 0
                total_segments += segment_count

                # Per-scene progress
                logger.info(
                    "[DecoupledPipeline] Scene %d/%d: %d words → %d segments (sentinel: %s)",
                    scene_idx + 1, n_scenes, word_count, segment_count, sentinel_status,
                )

                # Canonical diagnostics (SceneDiagnostics v2.0.0)
                aligner_native_count = max(
                    0,
                    segment_count - hardening_diag.interpolated_count - hardening_diag.fallback_count,
                )

                # VAD regions for this scene (if available)
                vad_regions = None
                if per_scene_regions:
                    vad_regions = [
                        {"start": round(s, 3), "end": round(e, 3)}
                        for s, e in per_scene_regions
                    ]

                scene_diag = SceneDiagnostics(
                    schema_version="2.0.0",
                    scene_index=scene_idx,
                    scene_duration_sec=duration,
                    framer_backend=frames[0].source if frames else "",
                    frame_count=len(frames),
                    word_count=word_count,
                    segment_count=segment_count,
                    sentinel_status=sentinel_status,
                    sentinel_triggers=assessment.get("triggers", []) if assessment else [],
                    sentinel_recovery=recovery_info,
                    timing_aligner_native=aligner_native_count,
                    timing_interpolated=hardening_diag.interpolated_count,
                    timing_vad_fallback=hardening_diag.fallback_count,
                    timing_total_segments=segment_count,
                    hardening_clamped=hardening_diag.clamped_count,
                    hardening_sorted=hardening_diag.sorted,
                    vad_regions=vad_regions,
                )
                diagnostics = asdict(scene_diag)

                # Save diagnostics artifact
                if self.artifacts_dir:
                    self._save_artifact(
                        scene_idx,
                        "diag",
                        json.dumps(diagnostics, ensure_ascii=False, indent=2),
                        ext=".json",
                    )

                results.append((result, diagnostics))

            except Exception as e:
                logger.error(
                    "[DecoupledPipeline] Scene %d failed: %s",
                    scene_idx,
                    e,
                    exc_info=True,
                )
                error_diag = SceneDiagnostics(
                    scene_index=scene_idx,
                    scene_duration_sec=scene_durations[scene_idx] if scene_idx < len(scene_durations) else 0.0,
                    error=str(e),
                )
                results.append((None, asdict(error_diag)))

        logger.info(
            "[DecoupledPipeline] Step 9: Complete — %d scenes, %d total segments, %d collapses",
            n_scenes, total_segments, total_collapses,
        )

        return results

    # -----------------------------------------------------------------------
    # Word merging: frame-relative → scene-relative
    # -----------------------------------------------------------------------

    @staticmethod
    def _merge_frame_words(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
    ) -> list[dict[str, Any]]:
        """
        Merge per-frame word lists into a single scene-relative word list.

        Each frame's words are in frame-relative coordinates.  We offset
        them by frame.start to convert to scene-relative coordinates.
        """
        all_words: list[dict[str, Any]] = []

        for frame, word_list in zip(frames, frame_word_lists):
            offset = frame.start
            for w in word_list:
                all_words.append(
                    {
                        "word": w["word"],
                        "start": w["start"] + offset,
                        "end": w["end"] + offset,
                    }
                )

        return all_words

    # -----------------------------------------------------------------------
    # Speech regions resolution
    # -----------------------------------------------------------------------

    @staticmethod
    def _get_speech_regions(
        scene_idx: int,
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]],
    ) -> Optional[list[tuple[float, float]]]:
        """
        Get speech regions for a scene, preferring explicit over framer-derived.

        Priority:
            1. scene_speech_regions (passed by caller, e.g., from Phase 4 VAD)
            2. frame_speech_regions (from VadGroupedFramer metadata)
        """
        # Priority 1: Explicit scene-level regions
        if scene_speech_regions and scene_idx < len(scene_speech_regions):
            regions = scene_speech_regions[scene_idx]
            if regions:
                return regions

        # Priority 2: Framer-derived regions (flatten per-frame regions)
        if scene_idx < len(frame_speech_regions):
            per_frame_regions = frame_speech_regions[scene_idx]
            if per_frame_regions:
                flat: list[tuple[float, float]] = []
                for frame_regions in per_frame_regions:
                    flat.extend(frame_regions)
                return flat if flat else None

        return None

    # -----------------------------------------------------------------------
    # Audio I/O helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _load_audio(path: Path) -> tuple[np.ndarray, int]:
        """Load audio from file as numpy array."""
        if sf is None:
            raise ImportError("soundfile is required for audio loading")
        audio, sr = sf.read(str(path), dtype="float32")
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr

    def _write_temp_wav(
        self,
        audio: np.ndarray,
        sample_rate: int,
        scene_idx: int,
        frame_idx: int,
    ) -> Path:
        """Write audio slice to a temporary WAV file."""
        if sf is None:
            raise ImportError("soundfile is required for audio writing")

        temp_dir = self.artifacts_dir or Path(tempfile.gettempdir())
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"dsp_s{scene_idx:03d}_f{frame_idx:03d}.wav"
        sf.write(str(temp_path), audio, sample_rate)
        self._temp_files.append(temp_path)
        return temp_path

    def _cleanup_temp_files(self) -> None:
        """Delete temporary audio files."""
        cleaned = 0
        for path in self._temp_files:
            try:
                if path.exists():
                    path.unlink()
                    cleaned += 1
            except OSError:
                pass
        if cleaned > 0:
            logger.debug("[DecoupledPipeline] Cleaned %d temp files", cleaned)
        self._temp_files.clear()

    # -----------------------------------------------------------------------
    # Artifact saving
    # -----------------------------------------------------------------------

    def _save_artifact(
        self,
        scene_idx: int,
        name: str,
        content: str,
        ext: str = ".txt",
    ) -> None:
        """Save a debug artifact to artifacts_dir."""
        if not self.artifacts_dir:
            return
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"scene{scene_idx:03d}_{name}{ext}"
        path.write_text(content, encoding="utf-8")

    # -----------------------------------------------------------------------
    # CUDA cleanup
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_cuda_cleanup() -> None:
        """Centralized CUDA cache cleanup."""
        from whisperjav.utils.gpu_utils import safe_cuda_cleanup

        safe_cuda_cleanup()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self) -> None:
        """Release all component resources."""
        self.framer.cleanup()
        self.generator.cleanup()
        if self.aligner:
            self.aligner.cleanup()
        self._cleanup_temp_files()
