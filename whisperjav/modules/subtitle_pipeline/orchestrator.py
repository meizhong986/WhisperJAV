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
    TemporalFrame,
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
        """
        self.framer = framer
        self.generator = generator
        self.cleaner = cleaner
        self.aligner = aligner
        self.hardening_config = hardening_config
        self.artifacts_dir = artifacts_dir
        self.language = language

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
            "[DecoupledPipeline] Processing %d scenes (aligner=%s)",
            n_scenes,
            "yes" if self.aligner else "none",
        )

        try:
            # Step 1: Temporal framing + audio slicing
            scene_frames, frame_audio_paths, frame_speech_regions = self._step1_frame_and_slice(
                scene_audio_paths, scene_durations
            )

            # Steps 2-4: Text generation + cleaning (with VRAM lifecycle)
            scene_texts = self._step2_4_generate_and_clean(scene_frames, frame_audio_paths, scene_durations)

            # Steps 5-7: Alignment (with VRAM lifecycle), if aligner present
            scene_alignments = self._step5_7_align(scene_frames, frame_audio_paths, scene_texts, scene_durations)

            # Step 9: Per-scene reconstruction + sentinel + hardening
            results = self._step9_reconstruct_and_harden(
                scene_frames,
                scene_texts,
                scene_alignments,
                scene_audio_paths,
                scene_durations,
                frame_speech_regions,
                scene_speech_regions,
            )

        finally:
            self._cleanup_temp_files()

        return results

    # -----------------------------------------------------------------------
    # Step 1: Temporal framing + audio slicing
    # -----------------------------------------------------------------------

    def _step1_frame_and_slice(
        self,
        scene_audio_paths: list[Path],
        scene_durations: list[float],
    ) -> tuple[
        list[list[TemporalFrame]],
        list[list[Path]],
        list[Optional[list[list[tuple[float, float]]]]],
    ]:
        """
        Frame each scene and slice audio per frame into temp WAV files.

        Returns:
            scene_frames: Per-scene list of TemporalFrame objects.
            frame_audio_paths: Per-scene, per-frame temp WAV paths.
            frame_speech_regions: Per-scene speech regions from framer metadata.
        """
        scene_frames: list[list[TemporalFrame]] = []
        frame_audio_paths: list[list[Path]] = []
        frame_speech_regions: list[Optional[list[list[tuple[float, float]]]]] = []

        for scene_idx, audio_path in enumerate(scene_audio_paths):
            # Load scene audio
            audio, sr = self._load_audio(audio_path)

            # Run framer
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
        n_scenes = len(scene_frames)

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
                        gen_results = self.generator.generate_batch(
                            audio_paths=gen_audio_paths,
                            language=self.language,
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

                # Save raw text artifacts
                if self.artifacts_dir:
                    self._save_artifact(scene_idx, "raw", "\n---\n".join(raw_texts))

        finally:
            if needs_generation:
                self.generator.unload()
                self._safe_cuda_cleanup()

        # Phase 2: Cleaning
        scene_texts: list[list[str]] = []
        for scene_idx, raw_texts in enumerate(scene_raw_texts):
            clean_texts = self.cleaner.clean_batch(raw_texts)
            scene_texts.append(clean_texts)

            # Save clean text artifacts
            if self.artifacts_dir:
                self._save_artifact(scene_idx, "clean", "\n---\n".join(clean_texts))

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

        n_scenes = len(scene_frames)
        scene_alignments: list[list[list[dict[str, Any]]]] = []

        self.aligner.load()
        try:
            for scene_idx in range(n_scenes):
                frames = scene_frames[scene_idx]
                audio_paths = frame_audio_paths[scene_idx]
                texts = scene_texts[scene_idx]

                frame_alignments: list[list[dict[str, Any]]] = []

                # Align each frame individually (frame-relative coordinates)
                for frame_idx, (frame, audio_path, text) in enumerate(zip(frames, audio_paths, texts)):
                    if not text.strip():
                        frame_alignments.append([])
                        continue

                    try:
                        align_result = self.aligner.align(
                            audio_path=audio_path,
                            text=text,
                            language=self.language,
                            audio_durations=[frame.duration],
                        )
                        # Convert WordTimestamp objects to dicts
                        word_dicts = [
                            {
                                "word": w.word,
                                "start": w.start,
                                "end": w.end,
                            }
                            for w in align_result.words
                        ]
                        frame_alignments.append(word_dicts)
                    except Exception:
                        logger.error(
                            "[DecoupledPipeline] Alignment failed for scene %d frame %d",
                            scene_idx,
                            frame_idx,
                            exc_info=True,
                        )
                        frame_alignments.append([])

                scene_alignments.append(frame_alignments)

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

        results: list[tuple[Any, dict[str, Any]]] = []

        for scene_idx in range(len(scene_frames)):
            frames = scene_frames[scene_idx]
            texts = scene_texts[scene_idx]
            duration = scene_durations[scene_idx]
            audio_path = scene_audio_paths[scene_idx]

            self.sentinel_stats["total_scenes"] += 1

            try:
                if scene_alignments is not None:
                    # Aligned workflow: merge frame-relative → scene-relative
                    all_words = self._merge_frame_words(frames, scene_alignments[scene_idx])

                    # Sentinel assessment
                    assessment = assess_alignment_quality(all_words, duration)
                    sentinel_status = assessment["status"]

                    if sentinel_status == "COLLAPSED":
                        self.sentinel_stats["collapsed_scenes"] += 1

                        # Get speech regions for recovery
                        regions = self._get_speech_regions(scene_idx, frame_speech_regions, scene_speech_regions)

                        corrected_words = redistribute_collapsed_words(all_words, duration, regions)
                        self.sentinel_stats["recovered_scenes"] += 1
                        strategy = "vad_guided" if regions else "proportional"
                        self.sentinel_stats["recovery_strategies"][strategy] += 1

                        # Reconstruct with suppress_silence=False (H3 fix)
                        result = reconstruct_from_words(corrected_words, audio_path, suppress_silence=False)
                    else:
                        result = reconstruct_from_words(all_words, audio_path, suppress_silence=True)

                else:
                    # Aligner-free: build word dicts from frame boundaries
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
                    result = reconstruct_from_words(words, audio_path)
                    sentinel_status = "N/A"

                # Hardening (shared by all paths)
                config = HardeningConfig(
                    timestamp_mode=self.hardening_config.timestamp_mode,
                    scene_duration_sec=duration,
                    speech_regions=self.hardening_config.speech_regions,
                )
                hardening_diag = harden_scene_result(result, config)

                diagnostics: dict[str, Any] = {
                    "scene_idx": scene_idx,
                    "frame_count": len(frames),
                    "sentinel_status": sentinel_status,
                    "hardening": asdict(hardening_diag),
                    "segment_count": (len(result.segments) if result and result.segments else 0),
                }

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
                results.append((None, {"scene_idx": scene_idx, "error": str(e)}))

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
