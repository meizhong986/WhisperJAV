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
    8. Timing model (optional, 1.10.0 item 12 C3): where speech starts in each
       frame and where a new sentence starts (timing_rules.py)
    9. Word merging (frame-relative → scene-relative), Sentinel +
       Reconstruction + Hardening (per scene)

VRAM swap pattern:
    generator.load() → generate → generator.unload()
    → safe_cuda_cleanup()
    → aligner.load() → align → aligner.unload()
    → safe_cuda_cleanup()
    → timer.load() → time → timer.unload()
"""

import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from whisperjav.modules.subtitle_pipeline.hardening import harden_scene_result
from whisperjav.modules.subtitle_pipeline import timing_rules
from whisperjav.modules.subtitle_pipeline.protocols import (
    SpeechTimer,
    TemporalFramer,
    TextAligner,
    TextCleaner,
    TextGenerator,
)
from whisperjav.modules.subtitle_pipeline.reconstruction import (
    REGROUP_VAD_ONLY,
    reconstruct_frame_native,
    reconstruct_from_words,
    resolve_regroup,
    split_frame_to_words,
)
from whisperjav.modules.subtitle_pipeline.types import (
    HardeningConfig,
    RegroupMode,
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


def line_filter_test(ellipsis_filter: bool = False, nonverbal_filter: bool = True) -> Callable[[str], bool]:
    """Rule T9 (owner W296 2, W300 3): a test of whether a line filter the run applies after this pipeline
    would remove a subtitle with this text, so a split never hands one a part to delete. The ChronosJAV
    Phase 8 filters' own tests, as the run has them: anime-whisper's ellipsis filter (``ellipsis_filter``),
    the non-verbal line filter with its punctuation-only rule (``nonverbal_filter``; off with
    --no-qwen-drop-nonverbal-lines) and the non-linguistic utterance filter (always). The defaults are
    Qwen3-ASR's."""
    from whisperjav.modules.subtitle_pipeline.cleaners.anime_whisper import AnimeWhisperCleaner
    from whisperjav.modules.subtitle_pipeline.cleaners.nonlinguistic_utterance_filter import (
        NonlinguisticUtteranceFilter,
    )
    from whisperjav.modules.subtitle_pipeline.cleaners.nonverbal_line_filter import (
        NonverbalLineFilter,
        _normalizes_to_nothing,
    )

    def removed(text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        if ellipsis_filter and AnimeWhisperCleaner._is_ellipsis_only(text):
            return True
        if nonverbal_filter and (_normalizes_to_nothing(text) or NonverbalLineFilter.is_nonverbal_line(text)):
            return True
        return NonlinguisticUtteranceFilter.is_nonlinguistic(text)

    return removed

# The run-summary lines when a part could not be loaded (#436; item 12). Texts
# approved by the owner as drafted (W268); design note 2026-10-28, section 9. The timing model's line departs from the section 9
# draft ("keep their VAD group edges"), which is untrue when the aligner worked.
ALIGNER_FALLBACK_TEXT = "subtitle times come from the VAD groups, and no subtitle was split; the text is kept."
ALIGNER_NOT_LOADED_TEXT = "ChronosJAV's aligner ({name}) could not be loaded ({error}): " + ALIGNER_FALLBACK_TEXT
TIMING_NOT_LOADED_TEXT = ("ChronosJAV's timing model ({name}) could not be loaded ({error}): no subtitle was "
                          "split, and subtitle times come from the aligner or the VAD groups; the text is kept.")
GROUPS_NOT_ALIGNED_TEXT = ("ChronosJAV's aligner ({name}) gave no times for {n} of {m} VAD group(s): each keeps "
                           "its text as one unsplit subtitle, timed from its VAD group (the timing model may "
                           "still move its start).")


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
        timer: Optional[SpeechTimer] = None,
        line_filter: Optional[Callable[[str], bool]] = None,
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
            timer: Optional timing model (SpeechTimer). When given, step 8
                splits a frame where a new sentence starts and, where there
                are no aligner times, moves its start to where the speech
                starts (timing_rules.py).
                None = no timing step (the behaviour before 1.10.0).
            line_filter: The test of the line filters the caller runs after
                this pipeline (line_filter_test()); a split never leaves a
                part it would remove (rule T9). None = line_filter_test().
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
        self.timer = timer
        self.line_filter = line_filter if line_filter is not None else line_filter_test()

        # Sentinel stats accumulated across all scenes
        self.sentinel_stats: dict[str, Any] = {
            "total_scenes": 0,
            "collapsed_scenes": 0,
            "recovered_scenes": 0,
            "recovery_strategies": {"vad_guided": 0, "proportional": 0, "vad_fallback": 0},
        }

        # Set when the aligner could not be loaded for the current file and the
        # times came from the aligner-free path instead (#436). Reset per file.
        self.alignment_unavailable: Optional[str] = None
        # Set when the aligner loaded but gave no times for some VAD groups with text
        # (first pass only). Reset per file.
        self.alignment_partial: Optional[str] = None
        self._last_align_counts = (0, 0)
        # Set when the timing model could not be loaded for the current file: the
        # subtitles keep their times and no frame is split. Reset per file.
        self.timing_unavailable: Optional[str] = None

        # Temp files for cleanup
        self._temp_files: list[Path] = []

    def process_scenes(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
        scene_speech_regions: Optional[list[list[tuple[float, float]]]] = None,
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Process all scenes through the decoupled pipeline.

        When step-down retry is enabled, collapsed scenes are automatically
        retried with tighter temporal framing (Pass 2).

        Args:
            scene_audio_paths: Paths to per-scene audio files (WAV, 16kHz).
                Used for ASR generation and alignment.
            scene_durations: Duration of each scene in seconds.
            scene_speech_regions: Optional per-scene VAD speech regions
                (from Phase 4 or VadGroupedFramer metadata).
            vad_audio_paths: Optional per-scene audio files for VAD/framing
                only (dual-track ``--enhance-for-vad`` mode).  When provided,
                the framer uses these (enhanced) files for temporal framing
                while ASR slicing reads from *scene_audio_paths* (original).
                When None (default), framing and slicing use the same files.

        Returns:
            List of (WhisperResult_or_None, diagnostics_dict) per scene.
        """
        n_scenes = len(scene_audio_paths)
        if n_scenes != len(scene_durations):
            raise ValueError(f"scene_audio_paths ({n_scenes}) and scene_durations ({len(scene_durations)}) must match")
        if vad_audio_paths is not None and len(vad_audio_paths) != n_scenes:
            raise ValueError(f"vad_audio_paths ({len(vad_audio_paths)}) and scene_audio_paths ({n_scenes}) must match")

        self.alignment_unavailable = None
        self.alignment_partial = None
        self.timing_unavailable = None

        logger.info(
            "[DecoupledPipeline] Processing %d scenes (aligner=%s, timer=%s, step-down=%s%s)",
            n_scenes,
            "yes" if self.aligner else "none",
            "yes" if self.timer else "none",
            "enabled" if (self.stepdown_config and self.stepdown_config.enabled) else "disabled",
            ", dual-track=yes" if vad_audio_paths else "",
        )

        # --- Pass 1: Normal processing ---
        self._last_align_counts = (0, 0)
        results = self._run_pass(
            scene_audio_paths, scene_durations, scene_speech_regions,
            vad_audio_paths=vad_audio_paths,
        )
        # From the first pass only, so a step-down retry does not count a VAD group twice.
        with_text, unaligned = self._last_align_counts
        if unaligned:
            self.alignment_partial = GROUPS_NOT_ALIGNED_TEXT.format(
                name=getattr(self.aligner, "display_name", "aligner"), n=unaligned, m=with_text)

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
                vad_audio_paths=vad_audio_paths,
            )
            # The aligner loaded for pass 1 but not for the retry (#436 fallback):
            # the retry ran without alignment, which is not an improvement. Keep
            # the pass-1 results (already recovered) and say what happened.
            if self.alignment_unavailable:
                self.alignment_unavailable = self.alignment_unavailable.replace(
                    "could not be loaded",
                    "could not be reloaded for the step-down retry", 1,
                ).replace(
                    ALIGNER_FALLBACK_TEXT,
                    f"{len(collapsed_indices)} collapsed scene(s) keep their first-pass timing.", 1,
                )
                return results
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
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Execute a single pass of the pipeline (framing → generation → alignment → hardening).

        Args:
            scene_audio_paths: Per-scene audio file paths (used for ASR).
            scene_durations: Per-scene durations.
            scene_speech_regions: Optional per-scene VAD speech regions.
            framer_override_max_group: When set, calls framer.reframe()
                with this max group duration instead of framer.frame().
            vad_audio_paths: Optional per-scene audio for framing/VAD
                (dual-track mode).  When None, framing uses scene_audio_paths.
        """
        try:
            (
                scene_frames, frame_audio_paths, frame_speech_regions,
                frame_speech_starts,
            ) = self._step1_frame_and_slice(
                scene_audio_paths, scene_durations,
                framer_override_max_group=framer_override_max_group,
                vad_audio_paths=vad_audio_paths,
            )
            scene_texts = self._step2_4_generate_and_clean(scene_frames, frame_audio_paths, scene_durations)
            scene_alignments = self._step5_7_align(scene_frames, frame_audio_paths, scene_texts, scene_durations)
            scene_timings = self._step8_time(scene_frames, frame_audio_paths, scene_texts)
            results = self._step9_reconstruct_and_harden(
                scene_frames, scene_texts, scene_alignments,
                scene_audio_paths, scene_durations,
                frame_speech_regions, scene_speech_regions,
                frame_speech_starts=frame_speech_starts,
                scene_timings=scene_timings,
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
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """Re-process collapsed scenes with tighter framing (step-down retry)."""
        retry_audio_paths = [scene_audio_paths[i] for i in collapsed_indices]
        retry_durations = [scene_durations[i] for i in collapsed_indices]
        retry_speech_regions = (
            [scene_speech_regions[i] for i in collapsed_indices]
            if scene_speech_regions else None
        )
        retry_vad_paths = (
            [vad_audio_paths[i] for i in collapsed_indices]
            if vad_audio_paths else None
        )
        return self._run_pass(
            retry_audio_paths, retry_durations, retry_speech_regions,
            framer_override_max_group=self.stepdown_config.fallback_max_group_s,
            vad_audio_paths=retry_vad_paths,
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
        vad_audio_paths: Optional[list[Path]] = None,
    ) -> tuple[
        list[list[TemporalFrame]],
        list[list[Path]],
        list[Optional[list[list[tuple[float, float]]]]],
        list[Optional[list[Optional[float]]]],
    ]:
        """
        Frame each scene and slice audio per frame into temp WAV files.

        Args:
            framer_override_max_group: When set and the framer supports
                ``reframe()``, uses tighter grouping (step-down retry).
            vad_audio_paths: Optional per-scene audio for framing/VAD
                (dual-track ``--enhance-for-vad`` mode).  When provided,
                the framer runs on these (enhanced) files while audio
                slicing reads from *scene_audio_paths* (original quality).

        Returns:
            scene_frames: Per-scene list of TemporalFrame objects.
            frame_audio_paths: Per-scene, per-frame temp WAV paths.
            frame_speech_regions: Per-scene speech regions from framer metadata.
            frame_speech_starts: Per-scene, per-frame "3a" refined display
                starts from framer metadata (None entries fall back to the
                frame start; None per scene when the framer doesn't provide
                them). Audio slicing above always uses the RAW frame
                boundaries — the refined start affects display timing only.
        """
        n_scenes = len(scene_audio_paths)
        dual_track = vad_audio_paths is not None
        logger.info(
            "[DecoupledPipeline] Step 1: Framing %d scenes%s%s", n_scenes,
            f" (reframe override={framer_override_max_group}s)" if framer_override_max_group else "",
            " (dual-track: enhanced→framer, original→ASR)" if dual_track else "",
        )

        scene_frames: list[list[TemporalFrame]] = []
        frame_audio_paths: list[list] = []  # Path entries, or np.ndarray in pathless mode
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]] = []
        frame_speech_starts: list[Optional[list[Optional[float]]]] = []

        # v1.9.0 pathless mode: when the generator can consume in-memory arrays
        # AND there is no aligner (e.g. anime-whisper, which is vad_only), skip the
        # per-VAD-group temp WAV write entirely and pass numpy slices downstream.
        # Fail-safe: if either condition is unmet, fall back to the WAV path.
        pathless = getattr(self.generator, "accepts_array", False) and (
            self.aligner is None or getattr(self.aligner, "accepts_array", False)
        ) and (self.timer is None or getattr(self.timer, "accepts_array", False))
        if pathless:
            logger.info(
                "[DecoupledPipeline] Step 1: pathless mode — in-memory frame audio, no temp WAVs"
            )

        for scene_idx, audio_path in enumerate(scene_audio_paths):
            # Dual-track: framer runs on enhanced audio, slicing on original
            framing_path = vad_audio_paths[scene_idx] if dual_track else audio_path
            framing_audio, sr = self._load_audio(framing_path)

            # Run framer (or reframe for step-down)
            if framer_override_max_group is not None and hasattr(self.framer, "reframe"):
                framing_result = self.framer.reframe(
                    framing_audio, sr, max_group_duration_s=framer_override_max_group,
                )
            else:
                framing_result = self.framer.frame(framing_audio, sr)
            frames = framing_result.frames
            scene_frames.append(frames)

            # Extract speech regions from framer metadata (VadGroupedFramer provides these)
            regions = framing_result.metadata.get("speech_regions")
            frame_speech_regions.append(regions)
            # "3a" refined display starts (VadGroupedFramer; None elsewhere)
            frame_speech_starts.append(framing_result.metadata.get("speech_starts"))

            if pathless:
                # In-memory slicing — NO temp WAV. ASR always reads the ORIGINAL
                # audio (load it explicitly under dual-track, since framing_audio
                # is the enhanced track in that mode).
                if dual_track:
                    asr_audio, sr = self._load_audio(audio_path)
                else:
                    asr_audio = framing_audio
                frame_entries: list = []
                for frame in frames:
                    if len(frames) == 1 and frame.start == 0.0:
                        frame_entries.append(asr_audio)  # full-scene frame
                    else:
                        start_sample = int(frame.start * sr)
                        end_sample = int(frame.end * sr)
                        # .copy() so the parent scene array can be freed after the
                        # scene loop — bounds peak RAM to the speech regions only.
                        frame_entries.append(asr_audio[start_sample:end_sample].copy())
                frame_audio_paths.append(frame_entries)
            else:
                # Load ASR audio (only needed if dual-track and we have frames to slice)
                if dual_track and not (len(frames) == 1 and frames[0].start == 0.0):
                    asr_audio, sr = self._load_audio(audio_path)
                else:
                    asr_audio = framing_audio  # same source, no extra load

                # Slice audio per frame → temp WAV (always from ASR audio)
                frame_paths = []
                for frame_idx, frame in enumerate(frames):
                    if len(frames) == 1 and frame.start == 0.0:
                        # Full-scene frame — use original ASR audio path (no slicing needed)
                        frame_paths.append(audio_path)
                    else:
                        # Slice and write temp WAV from ASR audio
                        start_sample = int(frame.start * sr)
                        end_sample = int(frame.end * sr)
                        frame_audio = asr_audio[start_sample:end_sample]
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

        return scene_frames, frame_audio_paths, frame_speech_regions, frame_speech_starts

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
        frames_with_text = frames_unaligned = 0

        # The aligner loads only now, after all text is generated. A failed load
        # (first use with an interrupted model download, out of memory) used to
        # end the job and throw that text away (#436). Keep the text and take the
        # aligner-free path (Branch B: times from the speech segments).
        try:
            self.aligner.load()
        except Exception as load_err:
            first_line = (str(load_err).strip().splitlines() or [""])[0][:200]
            self.alignment_unavailable = ALIGNER_NOT_LOADED_TEXT.format(
                name=getattr(self.aligner, "display_name", "aligner"),
                error=f"{type(load_err).__name__}: {first_line}",
            )
            logger.warning("[DecoupledPipeline] %s", self.alignment_unavailable, exc_info=True)
            try:
                self.aligner.unload()
            except Exception:
                pass
            self._safe_cuda_cleanup()
            return None
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
                # VAD groups with text that the aligner gave no times (it returns nothing when the
                # library cannot open the model, or the call fails): they keep their text over the
                # VAD group's time, and the run summary says how many (item 12).
                frames_with_text += len(batch_indices)
                frames_unaligned += sum(1 for i in batch_indices if not frame_alignments[i])

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
        self._last_align_counts = (frames_with_text, frames_unaligned)
        if frames_unaligned:
            logger.warning(
                "[DecoupledPipeline] The aligner gave no times for %d of %d VAD group(s)",
                frames_unaligned, frames_with_text,
            )

        return scene_alignments

    # -----------------------------------------------------------------------
    # Step 8: Timing model (1.10.0, item 12 C3)
    # -----------------------------------------------------------------------

    def _step8_time(
        self,
        scene_frames: list[list[TemporalFrame]],
        frame_audio_paths: list[list[Path]],
        scene_texts: list[list[str]],
    ) -> Optional[list[list[Optional[list[timing_rules.TimedSegment]]]]]:
        """
        Run the timing model on every frame that has text.

        VRAM lifecycle: timer.load() → time all → timer.unload(), after the
        aligner is unloaded, so the three models never share memory.

        Returns:
            None if no timer (or it could not be loaded), otherwise
            scene_timings[scene_idx][frame_idx] = list of TimedSegment
            (onsets relative to the frame), or None for a frame without text.
        """
        if self.timer is None:
            return None

        import time as _time

        step8_start = _time.monotonic()
        try:
            self.timer.load()
        except Exception as load_err:
            first_line = (str(load_err).strip().splitlines() or [""])[0][:200]
            self.timing_unavailable = TIMING_NOT_LOADED_TEXT.format(
                name=getattr(self.timer, "display_name", "timing model"),
                error=f"{type(load_err).__name__}: {first_line}",
            )
            logger.warning("[DecoupledPipeline] %s", self.timing_unavailable, exc_info=True)
            try:
                self.timer.unload()
            except Exception:
                pass
            return None

        scene_timings: list[list[Optional[list[timing_rules.TimedSegment]]]] = []
        try:
            for scene_idx, frames in enumerate(scene_frames):
                frame_timings: list[Optional[list[timing_rules.TimedSegment]]] = []
                for frame_idx, (audio, text) in enumerate(zip(frame_audio_paths[scene_idx], scene_texts[scene_idx])):
                    if not text.strip():
                        frame_timings.append(None)
                        continue
                    try:
                        frame_timings.append(list(self.timer.time(audio)))
                    except Exception:
                        logger.warning(
                            "[DecoupledPipeline] Timing failed for scene %d frame %d; its times are kept",
                            scene_idx, frame_idx, exc_info=True,
                        )
                        frame_timings.append(None)
                scene_timings.append(frame_timings)
                if self.artifacts_dir:
                    self._save_artifact(
                        scene_idx,
                        "timed",
                        json.dumps(
                            [None if ft is None else [asdict(s) for s in ft] for ft in frame_timings],
                            ensure_ascii=False, indent=2,
                        ),
                        ext=".json",
                    )
        finally:
            self.timer.unload()

        logger.info(
            "[DecoupledPipeline] Step 8: Timing model done — %d scenes (%.1fs)",
            len(scene_timings), _time.monotonic() - step8_start,
        )
        return scene_timings

    def _timing_of(
        self,
        scene_timings: Optional[list[list[Optional[list[timing_rules.TimedSegment]]]]],
        scene_idx: int,
        frame_idx: int,
    ) -> Optional[list[timing_rules.TimedSegment]]:
        """The timed segments of one frame, or None."""
        if scene_timings is None or scene_idx >= len(scene_timings):
            return None
        frame_timings = scene_timings[scene_idx]
        return frame_timings[frame_idx] if frame_idx < len(frame_timings) else None

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
        frame_speech_starts: Optional[list[Optional[list[Optional[float]]]]] = None,
        scene_timings: Optional[list[list[Optional[list[timing_rules.TimedSegment]]]]] = None,
    ) -> list[tuple[Any, dict[str, Any]]]:
        """
        Per-scene: merge words → sentinel → reconstruct → harden.

        With ``scene_timings`` (step 8), each subtitle starts where the speech
        starts, and with word timestamps and regrouping off a frame is split
        where a new sentence starts (timing_rules.py).

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

        vad_only_mode = self.hardening_config.timestamp_mode == TimestampMode.VAD_ONLY

        for scene_idx in range(n_scenes):
            frames = scene_frames[scene_idx]
            texts = scene_texts[scene_idx]
            duration = scene_durations[scene_idx]
            audio_path = scene_audio_paths[scene_idx]

            # "3a" refined display starts (vad_only display timing only). The
            # RAW frame boundaries already fed the ASR audio slices in Step 1;
            # this moves each subtitle's start to the first confident-speech
            # frame the segmenter recorded. Starts only move LATER, so this
            # cannot create overlaps or lose captured content.
            scene_starts = (
                frame_speech_starts[scene_idx]
                if (
                    vad_only_mode
                    and frame_speech_starts
                    and scene_idx < len(frame_speech_starts)
                )
                else None
            )
            display_starts: list[float] = []
            for fi, fr in enumerate(frames):
                ds = fr.start
                if scene_starts and fi < len(scene_starts):
                    s = scene_starts[fi]
                    if s is not None and fr.start < s < fr.end:
                        ds = s
                display_starts.append(ds)

            self.sentinel_stats["total_scenes"] += 1
            frame_timings = scene_timings[scene_idx] if scene_timings and scene_idx < len(scene_timings) else None
            timing_info: Optional[dict[str, Any]] = None

            try:
                word_count = 0
                assessment = None
                recovery_info = None
                vad_fallback_scene = False
                not_aligner = 0  # subtitles whose times are not the aligner's (the run summary)

                if scene_alignments is not None:
                    # Branch A: Aligned workflow — merge frame-relative → scene-relative
                    # Resolve regroup mode for Branch A
                    regroup_a = resolve_regroup(self.hardening_config.regroup_mode, is_branch_b=False)

                    if regroup_a is False:
                        # Frame-native path: one segment per frame (or per timed piece,
                        # step 8), skip sentinel recovery.
                        # User explicitly set regroup_mode=OFF — they want raw frame output.
                        # Sentinel assessment for diagnostics only (no recovery); a
                        # collapsed scene is not split by the timing model (rule T5).
                        flat_words = self._merge_frame_words(frames, scene_alignments[scene_idx], texts)
                        assessment = assess_alignment_quality(flat_words, duration)
                        sentinel_status = assessment["status"]
                        if sentinel_status == "COLLAPSED":
                            self.sentinel_stats["collapsed_scenes"] += 1
                            total_collapses += 1

                        if (sentinel_status == "COLLAPSED"
                                and self.hardening_config.timestamp_mode == TimestampMode.ALIGNER_WITH_VAD_FALLBACK):
                            # Owner W266 1: a collapsed scene takes the aligner-free times
                            # (Branch B below: VAD-group edges, the start moved by the timing
                            # model, no splits), as "Aligner + VAD Fallback" says; no retry.
                            vad_fallback_scene = True
                            self.sentinel_stats["recovered_scenes"] += 1
                            self.sentinel_stats["recovery_strategies"]["vad_fallback"] += 1
                            recovery_info = {"strategy": "vad_fallback", "words_redistributed": 0}
                            logger.warning(
                                "[SENTINEL] Scene %d/%d: Alignment Collapse — its subtitles take "
                                "the VAD groups' times instead of the aligner's",
                                scene_idx + 1, n_scenes,
                            )
                        else:
                            frame_word_groups, timing_info = self._timed_frame_groups(
                                frames, scene_alignments[scene_idx], texts, frame_timings,
                                allow_split=sentinel_status != "COLLAPSED",
                                aligner_start=sentinel_status != "COLLAPSED",
                                removable=self.line_filter,
                            )
                            not_aligner = self._text_only_frames(frames, scene_alignments[scene_idx], texts)
                            word_count = sum(len(g) for g in frame_word_groups)

                            result = reconstruct_frame_native(frame_word_groups, audio_path)
                    else:
                        # Standard merge-and-regroup path. The splits are the regrouping's;
                        # the start is the aligner's, so the timing model moves only the start
                        # of a frame without word times (T2; owner W296 1, W300 4).
                        frame_word_groups, timing_info = self._timed_frame_groups(
                            frames, scene_alignments[scene_idx], texts, frame_timings, allow_split=False,
                        )
                        # At most one subtitle per text-only frame; regrouping may join it to another.
                        not_aligner = self._text_only_frames(frames, scene_alignments[scene_idx], texts)
                        all_words = [w for g in frame_word_groups for w in g]
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
                            result = reconstruct_from_words(
                                corrected_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )

                        elif sentinel_status == "COLLAPSED" and skip_recovery:
                            # aligner_only: log collapse but keep raw aligner timestamps
                            self.sentinel_stats["collapsed_scenes"] += 1
                            total_collapses += 1
                            logger.info(
                                "[SENTINEL] Scene %d/%d: COLLAPSED but aligner_only mode "
                                "— keeping raw aligner timestamps (no recovery)",
                                scene_idx + 1, n_scenes,
                            )
                            result = reconstruct_from_words(
                                all_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )
                        else:
                            # OK path — ForcedAligner timestamps are already accurate.
                            # Don't let stable-ts's crude loudness quantizer shrink them.
                            result = reconstruct_from_words(
                                all_words, audio_path, suppress_silence=False, regroup=regroup_a,
                            )

                if scene_alignments is None or vad_fallback_scene:
                    # Branch B: Aligner-free (also a collapsed scene under aligner_vad_fallback
                    # with regrouping off, owner W266 1)
                    # The timing model (step 8) moves each frame's start to where the
                    # speech starts; without word times it splits nothing (rule T5).
                    if frame_timings is not None:
                        timing_info = {"frames_timed": 0, "starts_moved": 0, "splits": 0, "split_times": []}
                        for fi, frame in enumerate(frames):
                            segs = frame_timings[fi] if fi < len(frame_timings) else None
                            if segs is None:
                                continue
                            timing_info["frames_timed"] += 1
                            start = timing_rules.speech_start(segs, frame.duration)
                            if start is not None and frame.start + start > display_starts[fi]:
                                display_starts[fi] = frame.start + start
                                timing_info["starts_moved"] += 1
                    # Resolve regroup mode for Branch B (aligner-free)
                    regroup = resolve_regroup(self.hardening_config.regroup_mode, is_branch_b=True)

                    if regroup is False:
                        # Frame-native: one segment per frame, each frame = one word entry.
                        # User explicitly set regroup_mode=OFF — they want raw frame output.
                        frame_word_groups = []
                        for fi, (frame, text) in enumerate(zip(frames, texts)):
                            if text.strip():
                                frame_word_groups.append([{
                                    "word": text.strip(),
                                    "start": display_starts[fi],
                                    "end": frame.end,
                                }])
                        word_count = len(frame_word_groups)
                        result = reconstruct_frame_native(frame_word_groups, audio_path)
                        if not vad_fallback_scene:
                            sentinel_status = "N/A"
                    else:
                        # Standard split-and-regroup path
                        words = []
                        for fi, (frame, text) in enumerate(zip(frames, texts)):
                            if text.strip():
                                words.extend(
                                    split_frame_to_words(text, display_starts[fi], frame.end)
                                )
                        word_count = len(words)
                        # G1 complement: VAD_ONLY preserves exact frame boundaries —
                        # suppress_silence=False prevents stable-ts silence detection
                        # from shifting VAD group timing.
                        suppress = (
                            self.hardening_config.timestamp_mode != TimestampMode.VAD_ONLY
                        )
                        try:
                            result = reconstruct_from_words(
                                words, audio_path, suppress_silence=suppress, regroup=regroup,
                            )
                        except Exception as regroup_err:
                            logger.warning(
                                "[DecoupledPipeline] Scene %d: regrouping failed (%s), "
                                "retrying without regrouping",
                                scene_idx, regroup_err,
                            )
                            result = reconstruct_from_words(
                                words, audio_path, suppress_silence=suppress, regroup=False,
                            )
                        if not vad_fallback_scene:
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

                # Canonical diagnostics (SceneDiagnostics v2.0.0). Aligner native counts
                # the subtitles whose times came from the aligner (owner W300 1): not those
                # of an aligner-free scene (vad_only, an aligner that could not be loaded,
                # a collapsed scene's VAD-group times) or of any collapsed scene, not a
                # text-only frame's (rule T3), not those hardening had to time.
                hardened = hardening_diag.interpolated_count + hardening_diag.fallback_count
                if scene_alignments is None or vad_fallback_scene or sentinel_status == "COLLAPSED":
                    not_aligner = segment_count
                not_aligner = min(max(0, segment_count - hardened), not_aligner)
                aligner_native_count = max(0, segment_count - hardened - not_aligner)

                # VAD regions for this scene (if available)
                vad_regions = None
                if per_scene_regions:
                    vad_regions = [
                        {"start": round(s, 3), "end": round(e, 3)}
                        for s, e in per_scene_regions
                    ]

                # "3a" audit trail: raw VAD frame boundaries (the audio fed to the ASR),
                # written in every run (F7, owner W307 1: before, only in vad_only runs),
                # and the refined display start where it is used (vad_only; else None).
                # internal_breaks reserved for the future "3b" splitter.
                frame_timing = [
                    {
                        "vad_start": round(fr.start, 3),
                        "vad_end": round(fr.end, 3),
                        "speech_start": round(display_starts[fi], 3) if scene_starts is not None else None,
                        "internal_breaks": [],
                    }
                    for fi, fr in enumerate(frames)
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
                    timing_vad_fallback=hardening_diag.fallback_count + not_aligner,
                    timing_total_segments=segment_count,
                    hardening_clamped=hardening_diag.clamped_count,
                    hardening_sorted=hardening_diag.sorted,
                    vad_regions=vad_regions,
                    frame_timing=frame_timing,
                    timing_model=timing_info,
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
    def _frame_words_or_text(
        frame: TemporalFrame,
        word_list: list[dict[str, Any]],
        text: Optional[str],
    ) -> list[dict[str, Any]]:
        """A frame's aligned words; a frame whose text got no word times keeps its text as one
        word spanning the frame (frame-relative), so no text is lost (rule T3, 1.10.0)."""
        if word_list or not text or not text.strip():
            return list(word_list)
        return [{"word": text.strip(), "start": 0.0, "end": frame.duration}]

    @staticmethod
    def _merge_frame_words(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
        texts: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        """
        Merge per-frame word lists into a single scene-relative word list.

        Each frame's words are in frame-relative coordinates.  We offset
        them by frame.start to convert to scene-relative coordinates.
        With ``texts``, a frame with text but no word times keeps its text.
        """
        all_words: list[dict[str, Any]] = []

        for fi, (frame, word_list) in enumerate(zip(frames, frame_word_lists)):
            offset = frame.start
            text = texts[fi] if texts and fi < len(texts) else None
            for w in DecoupledSubtitlePipeline._frame_words_or_text(frame, word_list, text):
                all_words.append(
                    {
                        "word": w["word"],
                        "start": w["start"] + offset,
                        "end": w["end"] + offset,
                    }
                )

        return all_words

    @staticmethod
    def _group_frame_words(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        """
        Group per-frame word lists into scene-relative word groups (one per frame).

        Like _merge_frame_words but keeps frame grouping instead of flattening.
        Each frame's words are offset by frame.start to convert from frame-relative
        to scene-relative coordinates.

        Returns:
            List of per-frame word groups (empty frames omitted).
        """
        groups: list[list[dict[str, Any]]] = []
        for frame, word_list in zip(frames, frame_word_lists):
            offset = frame.start
            frame_words = [
                {"word": w["word"], "start": w["start"] + offset, "end": w["end"] + offset}
                for w in word_list
            ]
            if frame_words:
                groups.append(frame_words)
        return groups

    @staticmethod
    def _text_only_frames(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
        texts: Optional[list[str]],
    ) -> int:
        """The frames with text but no word times (rule T3): each is one subtitle not timed by the aligner."""
        return sum(
            1 for fi, (_frame, word_list) in enumerate(zip(frames, frame_word_lists))
            if not word_list and texts and fi < len(texts) and texts[fi] and texts[fi].strip()
        )

    @staticmethod
    def _timed_frame_groups(
        frames: list[TemporalFrame],
        frame_word_lists: list[list[dict[str, Any]]],
        texts: Optional[list[str]],
        frame_timings: Optional[list[Optional[list[timing_rules.TimedSegment]]]],
        allow_split: bool = True,
        aligner_start: bool = True,
        removable: Optional[Callable[[str], bool]] = None,
    ) -> tuple[list[list[dict[str, Any]]], Optional[dict[str, Any]]]:
        """
        Scene-relative word groups: one per frame, or one per timed piece (step 8).

        A frame with text but no word times keeps its text (rule T3). With the
        frame's timed segments, timing_rules.pieces() splits it where a new
        sentence starts (when ``allow_split`` and the frame has word times). Each
        part starts at its first character's aligner time (T2, owner W296 1); a
        frame without word times, or with ``aligner_start`` off (a collapsed
        scene), moves its start to the Whisper onset instead. A split never leaves
        a part ``removable`` says a later line filter would remove (T9, W296 2).
        Without timings this is _group_frame_words() plus the kept text.

        Returns the groups and the timing counts for the diagnostics (None
        without timings).
        """
        groups: list[list[dict[str, Any]]] = []
        info: Optional[dict[str, Any]] = None
        if frame_timings is not None:
            info = {"frames_timed": 0, "starts_moved": 0, "splits": 0, "split_times": []}
        for fi, (frame, word_list) in enumerate(zip(frames, frame_word_lists)):
            text = texts[fi] if texts and fi < len(texts) else None
            words = DecoupledSubtitlePipeline._frame_words_or_text(frame, word_list, text)
            segs = frame_timings[fi] if frame_timings is not None and fi < len(frame_timings) else None
            if not words:
                continue
            if segs is None:
                parts = [[dict(w) for w in words]]
            else:
                parts = timing_rules.pieces(words, segs, frame.duration,
                                            allow_split=allow_split and bool(word_list),
                                            aligner_start=aligner_start and bool(word_list),
                                            removable=removable)
                info["frames_timed"] += 1
                if parts and parts[0][0]["start"] > words[0]["start"] + 1e-9:
                    info["starts_moved"] += 1
                info["splits"] += max(0, len(parts) - 1)
                info["split_times"].extend(round(frame.start + p[0]["start"], 3) for p in parts[1:])
            for part in parts:
                if part:
                    groups.append([
                        {"word": w["word"], "start": w["start"] + frame.start, "end": w["end"] + frame.start}
                        for w in part
                    ])
        return groups, info

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
