#!/usr/bin/env python3
"""
HuggingFace Transformers Pipeline for WhisperJAV.

Uses the HuggingFace Transformers pipeline with chunked long-form algorithm
for Japanese audio transcription. Designed as a drop-in mode with dedicated
CLI arguments (--hf-*) that don't affect other pipelines.

Key Features:
- Uses HuggingFace Transformers ASR (default: kotoba-whisper-bilingual-v1.0)
- Optional scene detection (none, auditok, silero)
- No external VAD needed (HF pipeline handles chunking internally)
- Supports any HuggingFace whisper model via --hf-model-id
- Supports Qwen3-ASR backend via asr_backend="qwen" (v1.8.3+)
"""

import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import srt
import stable_whisper

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules.transformers_asr import TransformersASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor, normalize_language_code
from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger
from whisperjav.utils.progress_display import DummyProgress

from whisperjav.modules.speech_enhancement import (
    create_enhancer_direct,
    enhance_scenes,
    enhance_single_audio,
    get_extraction_sample_rate,
    is_passthrough_backend,
)


class TransformersPipeline(BasePipeline):
    """
    HuggingFace Transformers-based ASR pipeline.

    This pipeline uses the HuggingFace Transformers library with chunked
    long-form transcription. It's optimized for Japanese with kotoba-whisper
    as the default model, but supports any HuggingFace whisper model.

    Scene detection is optional (default: none). When enabled, audio is
    split into scenes before transcription, with timestamps adjusted
    during stitching.
    """

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        progress_display=None,
        # ASR backend selection (v1.8.3+)
        asr_backend: str = "hf",  # "hf" (HuggingFace/Transformers) or "qwen" (Qwen3-ASR)
        # HF Transformers specific config (when asr_backend="hf")
        hf_model_id: str = "kotoba-tech/kotoba-whisper-bilingual-v1.0",
        hf_chunk_length: int = 15,
        hf_stride: Optional[float] = None,
        hf_batch_size: int = 16,
        hf_scene: str = "none",
        hf_beam_size: int = 5,
        hf_temperature: float = 0.0,
        hf_attn: str = "sdpa",
        hf_timestamps: str = "segment",
        hf_language: str = "ja",
        hf_task: str = "transcribe",
        hf_device: str = "auto",
        hf_dtype: str = "auto",
        # Qwen3-ASR specific config (when asr_backend="qwen")
        qwen_model_id: str = "Qwen/Qwen3-ASR-1.7B",
        qwen_device: str = "auto",
        qwen_dtype: str = "auto",
        qwen_batch_size: int = 1,  # batch_size=1 for accuracy
        qwen_max_tokens: int = 4096,  # Supports ~10 min audio
        qwen_language: Optional[str] = None,  # None = auto-detect
        qwen_timestamps: str = "word",  # "word" or "none"
        qwen_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        qwen_scene: str = "none",
        qwen_context: str = "",  # Context string for ASR accuracy
        qwen_context_file: Optional[str] = None,  # Path to context/glossary text file
        qwen_attn: str = "auto",  # Attention: auto, sdpa, flash_attention_2, eager
        # Speech enhancement (default: none = skip enhancement)
        hf_speech_enhancer: str = "none",
        hf_speech_enhancer_model: Optional[str] = None,
        qwen_enhancer: str = "none",  # Speech enhancement for Qwen mode
        qwen_enhancer_model: Optional[str] = None,  # Enhancer model variant for Qwen
        qwen_segmenter: str = "none",  # Post-ASR VAD filter for Qwen mode
        # Japanese post-processing options for Qwen mode (v1.8.4+)
        qwen_japanese_postprocess: bool = True,  # Apply Japanese regrouping
        qwen_postprocess_preset: str = "high_moan",  # Preset: high_moan (default for JAV), default, narrative
        # Standard options
        subs_language: str = "native",
        **kwargs
    ):
        """
        Initialize HuggingFace Transformers Pipeline.

        Args:
            output_dir: Output directory for final subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            progress_display: Progress display object
            asr_backend: ASR backend to use ('hf' for HuggingFace/Transformers, 'qwen' for Qwen3-ASR)
            hf_model_id: HuggingFace model ID (when asr_backend='hf')
            hf_chunk_length: Chunk length in seconds
            hf_stride: Overlap between chunks (None = chunk_length/6)
            hf_batch_size: Batch size for parallel processing
            hf_scene: Scene detection method ('none', 'auditok', 'silero')
            hf_beam_size: Beam size for beam search
            hf_temperature: Sampling temperature
            hf_attn: Attention implementation
            hf_timestamps: Timestamp granularity ('segment' or 'word')
            hf_language: Language code
            hf_task: Task type ('transcribe' or 'translate')
            hf_device: Device to use
            hf_dtype: Data type
            qwen_model_id: Qwen3-ASR model ID (when asr_backend='qwen')
            qwen_device: Device to use for Qwen
            qwen_dtype: Data type for Qwen
            qwen_batch_size: Maximum inference batch size for Qwen
            qwen_max_tokens: Maximum new tokens per utterance for Qwen
            qwen_language: Language code for Qwen (None = auto-detect)
            qwen_timestamps: Timestamp granularity for Qwen ('word' or 'none')
            qwen_aligner: ForcedAligner model ID for Qwen
            qwen_scene: Scene detection method for Qwen mode
            qwen_context: Context string to improve transcription accuracy
            qwen_attn: Attention implementation ('auto', 'sdpa', 'flash_attention_2', 'eager')
            hf_speech_enhancer: Speech enhancement backend for HF mode ('none', 'zipenhancer', 'clearvoice', 'bs-roformer'). Default: 'none'
            hf_speech_enhancer_model: Optional model variant for HF mode enhancer
            qwen_enhancer: Speech enhancement backend for Qwen mode ('none', 'clearvoice', 'bs-roformer', 'zipenhancer', 'ffmpeg-dsp'). Default: 'none'
            qwen_enhancer_model: Optional model variant for Qwen mode enhancer
            qwen_segmenter: Post-ASR VAD filter for Qwen mode ('none', 'silero', 'silero-v4.0', 'silero-v3.1', 'nemo', 'nemo-lite', 'whisper-vad', 'ten'). Filters out ASR segments in non-speech regions. Default: 'none'
            qwen_japanese_postprocess: Apply Japanese-specific subtitle regrouping (default: True). Improves quality by handling sentence particles, aizuchi removal, and natural segmentation.
            qwen_postprocess_preset: Japanese post-processing preset: 'default' (conversational), 'high_moan' (adult content), 'narrative' (longer passages)
            subs_language: Subtitle language ('native' or 'direct-to-english')
            **kwargs: Additional parameters for base class
        """
        super().__init__(
            output_dir=output_dir,
            temp_dir=temp_dir,
            keep_temp_files=keep_temp_files,
            **kwargs
        )

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Store ASR backend selection (v1.8.3+)
        self.asr_backend = asr_backend
        if asr_backend not in ("hf", "qwen"):
            raise ValueError(f"Invalid ASR backend: {asr_backend}. Must be 'hf' or 'qwen'")

        # Store HF config (used for metadata even when qwen backend is active)
        self.hf_config = {
            "model_id": hf_model_id,
            "chunk_length_s": hf_chunk_length,
            "stride_length_s": hf_stride,
            "batch_size": hf_batch_size,
            "beam_size": hf_beam_size,
            "temperature": hf_temperature,
            "attn_implementation": hf_attn,
            "timestamps": hf_timestamps,
            "language": hf_language,
            "task": hf_task,
            "device": hf_device,
            "dtype": hf_dtype,
        }

        # Store Qwen config (v1.8.3+)
        self.qwen_config = {
            "model_id": qwen_model_id,
            "device": qwen_device,
            "dtype": qwen_dtype,
            "batch_size": qwen_batch_size,
            "max_new_tokens": qwen_max_tokens,
            "language": qwen_language,
            "timestamps": qwen_timestamps,
            "use_aligner": qwen_timestamps == "word",
            "aligner_id": qwen_aligner,
            "context": self._resolve_context(qwen_context, qwen_context_file),
            "attn_implementation": qwen_attn,
            # Japanese post-processing (v1.8.4+)
            "japanese_postprocess": qwen_japanese_postprocess,
            "postprocess_preset": qwen_postprocess_preset,
        }

        # Scene detection config - use qwen_scene for qwen backend, hf_scene otherwise
        self.scene_method = qwen_scene if asr_backend == "qwen" else hf_scene
        if self.scene_method not in ("none", "auditok", "silero", "semantic"):
            raise ValueError(f"Invalid scene method: {self.scene_method}. Must be 'none', 'auditok', 'silero', or 'semantic'")

        # Determine output language code (JP-004 fix: normalize language codes)
        # Accepts both ISO codes ('ja') and full names ('Japanese')
        if subs_language == 'direct-to-english':
            self.lang_code = 'en'
        elif asr_backend == "qwen":
            # For Qwen, use detected language or default to 'ja'
            self.lang_code = normalize_language_code(qwen_language or 'ja')
        else:
            self.lang_code = normalize_language_code(hf_language)

        # =================================================================
        # SCOPE-BASED RESOURCE MANAGEMENT (v1.7.3+)
        # GPU models are NOT created in __init__. We store CONFIGS only.
        # Models are created as LOCAL VARIABLES inside process() and
        # explicitly destroyed after use to prevent VRAM overlap.
        # =================================================================

        # Speech enhancement CONFIG (model created in process())
        # Use qwen_enhancer when asr_backend="qwen", hf_speech_enhancer otherwise
        if asr_backend == "qwen":
            self._enhancer_config = {
                'backend': qwen_enhancer,
                'model': qwen_enhancer_model
            }
        else:
            self._enhancer_config = {
                'backend': hf_speech_enhancer,
                'model': hf_speech_enhancer_model
            }

        # Speech segmenter CONFIG for post-ASR VAD filtering (Phase 2)
        # Only used when asr_backend="qwen" and qwen_segmenter != "none"
        self._segmenter_backend = qwen_segmenter if asr_backend == "qwen" else "none"

        # v1.8.5+: Extract at 16kHz when enhancer is "none", 48kHz for real enhancers
        self._enhancer_is_passthrough = is_passthrough_backend(self._enhancer_config.get('backend'))
        extraction_sr = get_extraction_sample_rate(self._enhancer_config.get('backend'))
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)

        # Scene detector (only if enabled)
        self.scene_detector = None
        if self.scene_method != "none":
            from whisperjav.modules.scene_detection_backends import SceneDetectorFactory
            self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(method=self.scene_method)

        # ASR CONFIG based on backend (model created in process() after enhancement cleanup)
        if asr_backend == "qwen":
            self._asr_config = {
                'model_id': qwen_model_id,
                'device': qwen_device,
                'dtype': qwen_dtype,
                'batch_size': qwen_batch_size,
                'max_new_tokens': qwen_max_tokens,
                'language': qwen_language,
                'timestamps': qwen_timestamps,
                'use_aligner': qwen_timestamps == "word",
                'aligner_id': qwen_aligner,
                'context': self._resolve_context(qwen_context, qwen_context_file),
                'attn_implementation': qwen_attn,
                # Japanese post-processing (v1.8.4+)
                'japanese_postprocess': qwen_japanese_postprocess,
                'postprocess_preset': qwen_postprocess_preset,
            }
        else:
            self._asr_config = {
                'model_id': hf_model_id,
                'device': hf_device,
                'dtype': hf_dtype,
                'attn_implementation': hf_attn,
                'batch_size': hf_batch_size,
                'chunk_length_s': hf_chunk_length,
                'stride_length_s': hf_stride,
                'language': hf_language,
                'task': hf_task,
                'timestamps': hf_timestamps,
                'beam_size': hf_beam_size,
                'temperature': hf_temperature,
            }
        # NOTE: self.asr is NOT created here - it's a local variable in process()

        # SRT stitcher (for multi-scene)
        self.stitcher = SRTStitcher()

        # Post-processor
        self.postprocessor = SRTPostProcessor(language=self.lang_code)

        import os
        model_name = qwen_model_id if asr_backend == "qwen" else hf_model_id
        logger.info(f"TransformersPipeline initialized")
        logger.info(f"  ASR backend: {asr_backend}")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Scene detection: {self.scene_method}")
        logger.info(f"  Speech enhancer: {self._enhancer_config.get('backend', 'none')}")
        enh_label = "passthrough" if self._enhancer_is_passthrough else "enhancement enabled"
        logger.info(f"  Extraction SR: {extraction_sr}Hz ({enh_label})")

        # Diagnostic: Log full config for Pass 2 debugging
        if asr_backend == "qwen":
            logger.debug(
                "[TransformersPipeline PID %s] Initialized with asr_backend=%s, model_id=%s, language=%s, scene=%s",
                os.getpid(), asr_backend, qwen_model_id, qwen_language, self.scene_method
            )
        else:
            logger.debug(
                "[TransformersPipeline PID %s] Initialized with asr_backend=%s, model_id=%s, task=%s, language=%s, scene=%s",
                os.getpid(), asr_backend, hf_model_id, hf_task, hf_language, self.scene_method
            )

    @staticmethod
    def _resolve_context(context: str, context_file: Optional[str]) -> str:
        """Resolve context from inline string and/or file path.

        If both are provided, they are concatenated with a newline separator.
        """
        parts = []
        if context:
            parts.append(context)
        if context_file:
            try:
                file_content = Path(context_file).read_text(encoding='utf-8').strip()
                if file_content:
                    parts.append(file_content)
                    logger.debug(f"Loaded context from file: {context_file} ({len(file_content)} chars)")
            except Exception as e:
                logger.warning(f"Failed to load context file '{context_file}': {e}")
        return "\n".join(parts)

    def get_mode_name(self) -> str:
        """Return pipeline mode name."""
        if self.asr_backend == "qwen":
            return "qwen"
        return "transformers"

    def _convert_asr_result_to_segments(
        self,
        result: Union[stable_whisper.WhisperResult, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """
        Convert ASR result to unified segment format.

        Handles both:
        - WhisperResult from QwenASR (stable-ts format)
        - List[Dict] from TransformersASR

        Args:
            result: ASR output (WhisperResult or List[Dict])

        Returns:
            List of segment dicts with 'text', 'start', 'end' keys
        """
        # If already a list, return as-is (TransformersASR format)
        if isinstance(result, list):
            return result

        # Convert WhisperResult to List[Dict]
        if isinstance(result, stable_whisper.WhisperResult):
            segments = []
            for seg in result.segments:
                text = seg.text.strip() if hasattr(seg, 'text') else ''
                if not text:
                    continue

                start = float(seg.start) if hasattr(seg, 'start') else 0.0
                end = float(seg.end) if hasattr(seg, 'end') else start + 2.0

                segments.append({
                    'text': text,
                    'start': start,
                    'end': end,
                })
            return segments

        # Fallback: try to use as-is
        logger.warning(f"Unknown ASR result type: {type(result)}, attempting direct use")
        return result if result else []

    def _filter_segments_by_vad(
        self,
        segments: List[Dict[str, Any]],
        audio_path: Path,
        min_overlap_ratio: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Filter ASR segments using VAD to remove hallucinations in non-speech regions.

        This is a POST-ASR filter that removes segments that don't overlap sufficiently
        with detected speech regions. This helps reduce hallucinations that occur in
        silence, music, or background noise.

        Args:
            segments: List of ASR segment dicts with 'text', 'start', 'end' keys
            audio_path: Path to the audio file to run VAD on
            min_overlap_ratio: Minimum overlap with speech regions to keep segment (0.0-1.0)
                              Default 0.3 means at least 30% of segment must be speech

        Returns:
            Filtered list of segments that overlap with detected speech
        """
        if self._segmenter_backend == "none" or not segments:
            return segments

        try:
            import soundfile as sf
            import numpy as np
            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory

            logger.info(f"Running post-ASR VAD filter with {self._segmenter_backend}...")

            # Load audio for VAD
            audio_data, sample_rate = sf.read(str(audio_path), dtype='float32')

            # Create segmenter
            segmenter = SpeechSegmenterFactory.create(self._segmenter_backend)

            # Run VAD
            vad_result = segmenter.segment(audio_data, sample_rate=sample_rate)

            # Cleanup segmenter immediately (VRAM management)
            segmenter.cleanup()
            del segmenter

            if not vad_result.segments:
                logger.warning("VAD detected no speech - keeping all ASR segments")
                return segments

            # Build list of speech regions [(start_sec, end_sec), ...]
            speech_regions = [(seg.start_sec, seg.end_sec) for seg in vad_result.segments]

            logger.debug(
                f"VAD found {len(speech_regions)} speech regions, "
                f"coverage: {vad_result.speech_coverage_ratio:.1%}"
            )

            # Filter segments
            filtered_segments = []
            removed_count = 0

            for seg in segments:
                seg_start = seg.get('start', 0.0)
                seg_end = seg.get('end', seg_start + 0.1)
                seg_duration = max(seg_end - seg_start, 0.001)  # Avoid division by zero

                # Calculate overlap with speech regions
                overlap_duration = 0.0
                for speech_start, speech_end in speech_regions:
                    # Calculate intersection
                    intersect_start = max(seg_start, speech_start)
                    intersect_end = min(seg_end, speech_end)
                    if intersect_end > intersect_start:
                        overlap_duration += intersect_end - intersect_start

                overlap_ratio = overlap_duration / seg_duration

                if overlap_ratio >= min_overlap_ratio:
                    filtered_segments.append(seg)
                else:
                    removed_count += 1
                    logger.debug(
                        f"VAD filter removed segment [{seg_start:.2f}-{seg_end:.2f}]: "
                        f"overlap={overlap_ratio:.1%} < {min_overlap_ratio:.1%}"
                    )

            if removed_count > 0:
                logger.info(
                    f"VAD filter: kept {len(filtered_segments)}/{len(segments)} segments "
                    f"(removed {removed_count} likely hallucinations)"
                )

            return filtered_segments

        except Exception as e:
            logger.warning(f"VAD filtering failed, keeping original segments: {e}")
            return segments

    def _segments_to_srt(self, segments: List[Dict[str, Any]], offset: float = 0.0) -> str:
        """
        Convert ASR segments to SRT format string.

        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
            offset: Time offset to add to all timestamps (for scene stitching)

        Returns:
            SRT formatted string
        """
        subtitles = []

        for idx, seg in enumerate(segments, 1):
            text = seg.get("text", "").strip()
            if not text:
                continue

            start_sec = seg.get("start", 0.0) + offset
            end_sec = seg.get("end", start_sec + 2.0) + offset

            # Create timedelta objects for srt library
            start_td = timedelta(seconds=start_sec)
            end_td = timedelta(seconds=end_sec)

            # Ensure end is after start
            if end_td <= start_td:
                end_td = start_td + timedelta(milliseconds=100)

            subtitle = srt.Subtitle(
                index=idx,
                start=start_td,
                end=end_td,
                content=text
            )
            subtitles.append(subtitle)

        return srt.compose(subtitles)

    def _write_srt(self, srt_content: str, output_path: Path) -> int:
        """
        Write SRT content to file.

        Args:
            srt_content: SRT formatted string
            output_path: Path to write

        Returns:
            Number of subtitles written
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(srt_content, encoding="utf-8")

        # Count subtitles
        count = srt_content.count("\n\n")
        return count

    def process(self, media_info: Dict) -> Dict:
        """
        Process media file through HuggingFace Transformers pipeline.

        Steps:
        1. Extract audio (if video)
        2. Optionally detect scenes
        3. Transcribe (per scene or full file)
        4. Generate SRT from segments
        5. Stitch (if scenes)
        6. Post-process
        7. Return metadata

        Args:
            media_info: Media information dict

        Returns:
            Master metadata dict
        """
        import os
        start_time = time.time()

        input_file = media_info['path']
        media_basename = media_info['basename']

        # Diagnostic: Log process start for Pass 2 debugging
        logger.debug(
            "[TransformersPipeline PID %s] process() started for: %s (model=%s)",
            os.getpid(), media_basename, self.hf_config.get("model_id")
        )

        # Create master metadata
        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        master_metadata["config"]["asr_backend"] = self.asr_backend
        if self.asr_backend == "qwen":
            master_metadata["config"]["qwen_config"] = self.qwen_config
        else:
            master_metadata["config"]["hf_config"] = self.hf_config
        master_metadata["config"]["scene_detection"] = self.scene_method
        master_metadata["config"]["post_asr_vad_filter"] = self._segmenter_backend

        try:
            # Step 1: Extract audio
            self.progress.set_current_step("Extracting audio", 1, 5)
            logger.info("Step 1/5: Extracting audio...")

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration
            )

            # Step 2: Optional scene detection
            scene_paths = None
            if self.scene_method != "none":
                self.progress.set_current_step(f"Detecting scenes ({self.scene_method})", 2, 5)
                logger.info(f"Step 2/5: Detecting scenes ({self.scene_method})...")

                scenes_dir = self.temp_dir / "scenes"
                scenes_dir.mkdir(exist_ok=True)
                detection_result = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
                scene_paths = detection_result.to_legacy_tuples()

                if not scene_paths:
                    logger.warning("Scene detection produced zero scenes. Processing full audio.")
                    scene_paths = None
                else:
                    master_metadata["scenes_detected"] = []
                    for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                        master_metadata["scenes_detected"].append({
                            "scene_index": idx,
                            "filename": scene_path.name,
                            "start_time_seconds": round(start_sec, 3),
                            "end_time_seconds": round(end_sec, 3),
                            "duration_seconds": round(dur_sec, 3),
                            "path": str(scene_path)
                        })
                    master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "completed",
                    scene_count=len(scene_paths) if scene_paths else 0,
                    method=self.scene_method
                )
            else:
                self.progress.set_current_step("Skipping scene detection", 2, 5)
                logger.info("Step 2/5: Skipping scene detection (disabled)")
                self.metadata_manager.update_processing_stage(
                    master_metadata, "scene_detection", "skipped"
                )

            # =================================================================
            # PHASE 1: SPEECH ENHANCEMENT (Exclusive VRAM Block)
            # When enhancer is "none" (passthrough), scenes are already at
            # 16kHz from extraction — skip this phase entirely.
            # When a real enhancer is configured, it runs as a LOCAL variable
            # created, used, and DESTROYED before ASR loads (VRAM Sandwich prevention).
            # =================================================================
            import gc
            try:
                import torch
                _torch_available = torch.cuda.is_available()
            except ImportError:
                _torch_available = False

            if self._enhancer_is_passthrough:
                # v1.8.5+: Audio already at 16kHz — skip enhancement entirely
                enhancer_name = "none"
                logger.info("Speech enhancer is passthrough — skipping enhancement")
                master_metadata["config"]["speech_enhancement"] = {
                    "backend": "none",
                    "skipped": True,
                }
            else:
                # A. Load Enhancer (always succeeds - "none" backend is fallback)
                enhancer = create_enhancer_direct(**self._enhancer_config)
                enhancer_name = enhancer.name
                logger.info(f"Step 2.5: Preparing audio with {enhancer.display_name}...")

                if scene_paths:
                    # B. Process Enhancement - Enhance each scene (includes 48kHz→16kHz resampling)
                    scene_paths = enhance_scenes(
                        scene_paths,
                        enhancer,
                        self.temp_dir,
                        progress_callback=lambda n, t, name: logger.debug(
                            f"Enhancing scene {n}/{t}: {name}"
                        )
                    )
                    master_metadata["config"]["speech_enhancement"] = {
                        "backend": enhancer_name,
                        "enhanced_scenes": len(scene_paths)
                    }
                else:
                    # Enhance full audio file (includes 48kHz→16kHz resampling)
                    enhanced_path = self.temp_dir / f"{media_basename}_enhanced.wav"
                    extracted_audio = enhance_single_audio(
                        extracted_audio,
                        enhancer,
                        output_path=enhanced_path
                    )
                    master_metadata["config"]["speech_enhancement"] = {
                        "backend": enhancer_name,
                        "enhanced_full_audio": True
                    }

                # C. DESTROY Enhancer - This is the "JIT Unload"
                # We must confirm VRAM is near-zero before loading ASR
                logger.debug("Destroying enhancer to free VRAM before ASR load")
                enhancer.cleanup()
                del enhancer
                gc.collect()
                if _torch_available:
                    torch.cuda.empty_cache()
                    logger.debug("GPU memory cleared after enhancement - VRAM should be near-zero")

                logger.info(f"Audio preparation complete, GPU memory released")

            self.metadata_manager.update_processing_stage(
                master_metadata, "speech_enhancement", "completed",
                backend=enhancer_name
            )

            # =================================================================
            # PHASE 2: ASR TRANSCRIPTION (Exclusive VRAM Block)
            # ASR is a LOCAL variable - created after enhancer is destroyed,
            # and destroyed before function returns.
            # =================================================================

            # A. Load ONLY the ASR (LOCAL variable, not self.asr)
            # Backend selection: "hf" uses TransformersASR, "qwen" uses QwenASR
            logger.info("Initializing ASR model (exclusive VRAM block)")
            if self.asr_backend == "qwen":
                from whisperjav.modules.qwen_asr import QwenASR
                asr = QwenASR(**self._asr_config)
                asr_display_name = "Qwen3-ASR"
            else:
                asr = TransformersASR(**self._asr_config)
                asr_display_name = "HF Transformers"

            # Step 3: Transcription
            self.progress.set_current_step(f"Transcribing with {asr_display_name}", 3, 5)
            logger.info(f"Step 3/5: Transcribing with {asr_display_name}...")

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)

            if scene_paths:
                # Multi-scene transcription
                scene_srt_info = []
                total_scenes = len(scene_paths)

                print(f"\nTranscribing {total_scenes} scenes with HF Transformers:")
                transcription_start = time.time()

                for idx, (scene_path, start_sec, end_sec, dur_sec) in enumerate(scene_paths):
                    scene_num = idx + 1

                    # Progress display
                    progress_pct = (scene_num / total_scenes) * 100
                    bar_width = 30
                    filled = int(bar_width * scene_num / total_scenes)
                    bar = '=' * filled + '-' * (bar_width - filled)

                    # ETA calculation
                    eta_text = ""
                    if scene_num > 2:
                        elapsed = time.time() - transcription_start
                        avg_per_scene = elapsed / scene_num
                        remaining = (total_scenes - scene_num) * avg_per_scene
                        if remaining > 60:
                            eta_text = f" | ETA: {remaining/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {remaining:.0f}s"

                    scene_name = scene_path.name[:25] + "..." if len(scene_path.name) > 25 else scene_path.name
                    print(f"\rTranscribing: [{bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_name}{eta_text}", end='', flush=True)

                    try:
                        # Transcribe scene
                        # For Qwen backend, pass artifacts_dir to save debug artifacts
                        if self.asr_backend == "qwen":
                            asr_result = asr.transcribe(scene_path, artifacts_dir=scene_srts_dir)
                        else:
                            asr_result = asr.transcribe(scene_path)
                        # Convert to unified segment format (handles both WhisperResult and List[Dict])
                        segments = self._convert_asr_result_to_segments(asr_result)

                        # Post-ASR VAD filtering (removes hallucinations in non-speech regions)
                        if self._segmenter_backend != "none" and segments:
                            segments = self._filter_segments_by_vad(segments, scene_path)

                        if segments:
                            # Convert to SRT (timestamps relative to scene start)
                            srt_content = self._segments_to_srt(segments)
                            scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                            self._write_srt(srt_content, scene_srt_path)

                            scene_srt_info.append((scene_srt_path, start_sec))
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                            master_metadata["scenes_detected"][idx]["segment_count"] = len(segments)
                        else:
                            master_metadata["scenes_detected"][idx]["transcribed"] = True
                            master_metadata["scenes_detected"][idx]["no_speech_detected"] = True

                    except Exception as e:
                        logger.error(f"Scene {scene_num} transcription failed: {e}")
                        master_metadata["scenes_detected"][idx]["transcribed"] = False
                        master_metadata["scenes_detected"][idx]["error"] = str(e)

                print(f"\n[DONE] Completed transcription of {total_scenes} scenes")

                # Step 4: Stitch scene SRTs
                self.progress.set_current_step("Stitching scene transcriptions", 4, 5)
                logger.info("Step 4/5: Stitching scene transcriptions...")

                stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "transcription", "completed",
                    scenes_transcribed=len(scene_srt_info)
                )
                self.metadata_manager.update_processing_stage(
                    master_metadata, "stitching", "completed",
                    subtitle_count=num_subtitles
                )

            else:
                # Full-file transcription (no scenes)
                logger.info("Transcribing full audio file...")

                # For Qwen backend, pass artifacts_dir to save debug artifacts
                if self.asr_backend == "qwen":
                    asr_result = asr.transcribe(extracted_audio, artifacts_dir=scene_srts_dir)
                else:
                    asr_result = asr.transcribe(extracted_audio)
                # Convert to unified segment format (handles both WhisperResult and List[Dict])
                segments = self._convert_asr_result_to_segments(asr_result)

                # Post-ASR VAD filtering (removes hallucinations in non-speech regions)
                if self._segmenter_backend != "none" and segments:
                    segments = self._filter_segments_by_vad(segments, extracted_audio)

                # Convert to SRT
                srt_content = self._segments_to_srt(segments)
                stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
                num_subtitles = self._write_srt(srt_content, stitched_srt_path)

                self.metadata_manager.update_processing_stage(
                    master_metadata, "transcription", "completed",
                    segment_count=len(segments)
                )
                self.metadata_manager.update_processing_stage(
                    master_metadata, "stitching", "skipped"
                )

                print(f"[DONE] Transcription complete: {len(segments)} segments")

            # Step 5: Post-process
            self.progress.set_current_step("Post-processing subtitles", 5, 5)
            logger.info("Step 5/5: Post-processing subtitles...")

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.postprocessor.process(stitched_srt_path, final_srt_path)

            # Ensure final SRT is in output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)

            # Move raw_subs folder to output directory
            temp_raw_subs = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs.exists():
                final_raw_subs = self.output_dir / "raw_subs"
                final_raw_subs.mkdir(exist_ok=True)
                for file in temp_raw_subs.glob(f"{media_basename}*"):
                    shutil.copy2(file, final_raw_subs / file.name)

            # Copy Qwen debug artifacts to raw_subs (when using Qwen backend)
            if self.asr_backend == "qwen" and scene_srts_dir.exists():
                final_raw_subs = self.output_dir / "raw_subs"
                final_raw_subs.mkdir(exist_ok=True)
                # Copy JSON and TXT artifacts (_qwen_master.txt, _qwen_timestamps.json, _qwen_merged.json)
                for pattern in ["*_qwen_master.txt", "*_qwen_timestamps.json", "*_qwen_merged.json"]:
                    for artifact in scene_srts_dir.glob(pattern):
                        shutil.copy2(artifact, final_raw_subs / artifact.name)
                        logger.debug(f"Copied artifact: {artifact.name}")

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed",
                statistics=stats, output_path=str(final_srt_path)
            )

            # Update metadata
            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0),
                "cps_filtered": stats.get('cps_filtered', 0)
            }

            # C. DESTROY ASR - Trigger C++ destructor while interpreter is STABLE
            # This prevents the "Zone of Death" crash during Python shutdown
            logger.debug("Destroying ASR to free VRAM and trigger safe destructor")
            asr.cleanup()
            del asr
            gc.collect()
            if _torch_available:
                torch.cuda.empty_cache()
                logger.debug("GPU memory cleared after ASR - VRAM should be near-zero")

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"

            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            self.cleanup_temp_files(media_basename)

            logger.info(f"Output saved to: {final_srt_path}")
            logger.info(f"Total processing time: {total_time:.1f}s")

            return master_metadata

        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e)
            )
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            raise

    def cleanup(self) -> None:
        """Clean up pipeline resources.

        With Scope-Based Resource Management (v1.7.3+), GPU models are local variables
        inside process() and are destroyed immediately after use. This cleanup() method
        only handles non-GPU resources and delegates to parent.
        """
        # Delegate to parent for any remaining cleanup (scene detector, CUDA cache clear, etc.)
        super().cleanup()
