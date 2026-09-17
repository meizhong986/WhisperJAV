#!/usr/bin/env python3
"""V3 Architecture. Balanced pipeline implementation - scene detection with FasterWhisperPro ASR."""

import shutil
from pathlib import Path
from typing import Dict, List
import time
from datetime import datetime

from whisperjav.pipelines.base_pipeline import BasePipeline
from whisperjav.modules.audio_extraction import AudioExtractor
from whisperjav.modules import analytics
from whisperjav.modules.faster_whisper_pro_asr import FasterWhisperProASR
from whisperjav.modules.srt_postprocessing import SRTPostProcessor as StandardPostProcessor

from whisperjav.modules.scene_detection_backends import SceneDetectorFactory

from whisperjav.modules.srt_stitching import SRTStitcher
from whisperjav.utils.logger import logger
from whisperjav.utils.asr_telemetry import AsrTelemetry, resolve_telemetry_path
from whisperjav.utils.model_refresh import DEFAULT_MODEL_REFRESH_AUDIO_MINUTES

from whisperjav.utils.progress_display import DummyProgress
from whisperjav.utils.progress_aggregator import AsyncProgressReporter
from whisperjav.utils.parameter_tracer import NullTracer

from whisperjav.modules.speech_enhancement import (
    create_enhancer_from_config,
    enhance_scenes,
    get_extraction_sample_rate,
    is_passthrough_backend,
    resample_scenes,
)

# =============================================================================
# IMMORTAL OBJECT PATTERN - Prevents ctranslate2 Destructor Crash
# =============================================================================
# The ctranslate2 C++ destructor crashes with Access Violation (0xC0000005) or
# Stack Buffer Overrun (0xC0000409) on Windows when garbage collected during
# Python shutdown. This is a known upstream issue in ctranslate2/faster-whisper.
#
# Solution: Store ASR reference here to prevent garbage collection during normal
# execution. The nuclear exit (os._exit(0)) in main() terminates the process
# without running Python's shutdown sequence — the OS reclaims all memory.
#
# References:
# - https://github.com/SYSTRAN/faster-whisper/issues/1293
# - https://github.com/SYSTRAN/faster-whisper/issues/71
# - https://github.com/OpenNMT/CTranslate2/issues/1782
# =============================================================================
_IMMORTAL_ASR_REFERENCE = None


def safe_cleanup_immortal_asr() -> bool:
    """
    Clear the immortal ASR reference. Does NOT trigger destructors.

    IMPORTANT: This function intentionally does NOT call cleanup(), del, or gc.collect()
    on the ASR model. The ctranslate2 C++ destructor crashes with 0xC0000409
    (STATUS_STACK_BUFFER_OVERRUN) on Windows — a native structured exception that
    Python's try/except CANNOT catch. Even "controlled" cleanup triggers this crash.

    The correct approach is to skip all destructor-triggering operations and rely on
    os._exit(0) (nuclear exit) to terminate the process. The OS kernel reclaims all
    GPU memory when the process dies. No resource leak, no crash.

    This function exists for backwards compatibility. Callers should use os._exit(0)
    instead of attempting ASR cleanup.

    See: https://github.com/meizhong986/WhisperJAV/issues/125

    Returns:
        True if reference was held, False if already None.
    """
    global _IMMORTAL_ASR_REFERENCE

    if _IMMORTAL_ASR_REFERENCE is None:
        return False

    # Do NOT call cleanup(), del, or gc.collect() — these trigger the native crash.
    # Just clear the Python reference. The nuclear exit (os._exit) handles the rest.
    logger.debug(
        "Immortal ASR reference cleared (destructor skipped — nuclear exit will handle)"
    )
    _IMMORTAL_ASR_REFERENCE = None
    return True


class BalancedPipeline(BasePipeline):
    """Balanced pipeline using scene detection with FasterWhisperPro ASR (faster-whisper via stable-ts, VAD-enhanced)."""

    def __init__(self,
                 output_dir: str,
                 temp_dir: str,
                 keep_temp_files: bool,
                 subs_language: str,
                 resolved_config: Dict,
                 progress_display=None,
                 **kwargs):
        """
        Initializes the BalancedPipeline using V3 structured configuration.

        Args:
            output_dir: Output directory for subtitles
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep temporary files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            resolved_config: V3 structured configuration from TranscriptionTunerV3
            progress_display: Progress display object
            **kwargs: Additional parameters for base class
        """
        super().__init__(output_dir=output_dir, temp_dir=temp_dir, keep_temp_files=keep_temp_files, **kwargs)

        self.progress = progress_display or DummyProgress()
        self.subs_language = subs_language

        # Extract progress reporter and parameter tracer from kwargs
        self.progress_reporter = kwargs.get('progress_reporter', None)
        self.tracer = kwargs.get('parameter_tracer', NullTracer())

        # --- V3 STRUCTURED CONFIG UNPACKING ---
        # #394 per-scene ASR telemetry. On by default; the file goes to
        # raw_subs/ next to the outputs unless a path is given. The sync path
        # and the ensemble pass worker overwrite these attributes after
        # construction; the async path only has resolved_config to carry them.
        self.asr_telemetry_enabled = bool(resolved_config.get("asr_telemetry_enabled", True))
        self.asr_telemetry_path = resolved_config.get("asr_telemetry")
        self.asr_telemetry_tag = None  # e.g. "pass1" inside an ensemble run

        model_cfg = resolved_config["model"]
        params = resolved_config["params"]
        features = resolved_config["features"]
        task = resolved_config["task"]

        # Set the ASR task based on the chosen output language
        self.asr_task = task  # Use the task from resolved config directly

        # Extract feature configurations
        scene_opts = features.get("scene_detection", {})
        post_proc_opts = features.get("post_processing", {})

        # Store params for metadata logging
        self.scene_detection_params = scene_opts
        self.vad_params = params.get("vad", {})

        # Implement the smart model-switching logic (preserved from V2)
        effective_model_cfg = model_cfg.copy()
        if self.subs_language == 'direct-to-english' and model_cfg.get("model_name") == 'turbo':
            logger.info("Direct translation requested. Switching to 'large-v2' to perform translation.")
            effective_model_cfg["model_name"] = 'large-v2'

        # Store full pipeline options for diagnostic metadata (after model switching)
        self.pipeline_options = {
            "model": effective_model_cfg,
            "decoder": params.get("decoder", {}),
            "provider": params.get("provider", {}),
            "vad": self.vad_params,
            "task": task
        }
        # --- END V3 CONFIG UNPACKING ---

        # =================================================================
        # SCOPE-BASED RESOURCE MANAGEMENT (v1.7.3+)
        # GPU models are NOT created in __init__. We store CONFIGS only.
        # Models are created as LOCAL VARIABLES inside process() and
        # explicitly destroyed after use to prevent VRAM overlap.
        # =================================================================

        # Speech enhancement CONFIG (model created in process())
        self._enhancer_config = resolved_config  # Store full config for enhancer creation

        # Read enhancer backend to determine extraction sample rate
        enhancer_params = params.get("speech_enhancer", {})
        self._enhancer_backend_name = enhancer_params.get("backend", "none") or "none"
        self._enhancer_is_passthrough = is_passthrough_backend(self._enhancer_backend_name)
        self._enhance_for_vad = kwargs.get("enhance_for_vad", False)
        if self._enhance_for_vad and not self._enhancer_is_passthrough:
            # v1.9.3: this should no longer be reachable. Balanced finds the speech
            # inside faster-whisper's own transcribe() call, on the same audio it
            # transcribes, so there is no second track to hand the cleaned-up audio
            # to; the owner's decision of 2026-09-17 is that balanced does not offer
            # the setting at all. main.validate_balanced_vad_options refuses the flag
            # for a balanced pass and the GUI hides the box. Kept as a guard, and
            # honest about what it does if some other caller still sets it.
            logger.warning(
                "Enhance-for-VAD was set for a balanced pass, which cannot split the "
                "two: the cleaned-up audio is used for speech detection AND for "
                "transcription. Use the fidelity or qwen pipeline for the split.")

        # v1.8.5+: Extract at 16kHz when enhancer is "none" (skip enhancement entirely)
        # Extract at 48kHz when a real enhancer is configured (enhancer needs high-SR)
        extraction_sr = get_extraction_sample_rate(self._enhancer_backend_name)
        self.audio_extractor = AudioExtractor(sample_rate=extraction_sr)
        self.scene_detector = SceneDetectorFactory.safe_create_from_legacy_kwargs(**scene_opts)

        # v1.9.2 (owner CFF1 / D2): unload and reload the recogniser after this
        # many minutes of SCENE audio, at a scene boundary. 0 = never. The sync
        # path overwrites the attribute after construction (like telemetry);
        # async carries it in resolved_config; ensemble passes it as a kwarg.
        self.model_refresh_audio_minutes = float(
            kwargs.get(
                "model_refresh_audio_minutes",
                resolved_config.get("model_refresh_audio_minutes", DEFAULT_MODEL_REFRESH_AUDIO_MINUTES),
            ) or 0.0
        )

        # ASR CONFIG (model created lazily on first process() call)
        self._asr_config = {
            'model_config': effective_model_cfg,
            'params': params,
            'task': task,
            'tracer': self.tracer
        }
        # ASR instance - created once, reused for all files in batch
        # Named with underscore to prevent base_pipeline.cleanup() from touching it
        # (base cleanup looks for self.asr, not self._asr)
        self._asr = None

        self.stitcher = SRTStitcher()

        # Language code for post-processor and output filenames
        # Use 'en' for direct-to-english translation, otherwise use the selected source language
        if self.subs_language == 'direct-to-english':
            self.lang_code = 'en'
        else:
            # Get language from decoder params (set by CLI --language)
            self.lang_code = params["decoder"].get("language", "ja")
        self.standard_postprocessor = StandardPostProcessor(language=self.lang_code, **post_proc_opts)

    def _ensure_asr(self):
        """
        Lazy ASR initialization - create once, reuse for all files.

        This implements the Model Reuse pattern to prevent ctranslate2 destructor
        crashes during multi-file batch processing. The ASR model is created on
        first call and stored in self._asr. Subsequent calls return the same instance.

        The _IMMORTAL_ASR_REFERENCE global is set once to prevent garbage collection.
        Nuclear Exit (os._exit(0)) in pass_worker.py handles final cleanup.

        Returns:
            FasterWhisperProASR: The shared ASR instance
        """
        global _IMMORTAL_ASR_REFERENCE

        if self._asr is None:
            if self.model_refresh_audio_minutes > 0:
                # v1.9.2 (CFF1 / D5): the CTranslate2 model lives in a child
                # process so it can be replaced with a fresh one when the refresh
                # budget is spent — the only destructor-free way to "unload and
                # reload" it (see asr_worker_proxy.py). Same seven-member surface
                # as FasterWhisperProASR; the immortal reference is not needed.
                from whisperjav.modules.asr_worker_proxy import RemoteFasterWhisperASR
                from whisperjav.utils.model_refresh import ModelRefreshPolicy
                logger.info(
                    "Initializing ASR model in a worker process "
                    "(fresh instance after every %.0f min of scene audio)",
                    self.model_refresh_audio_minutes,
                )
                self._asr = RemoteFasterWhisperASR(
                    self._asr_config,
                    ModelRefreshPolicy.from_minutes(self.model_refresh_audio_minutes),
                )
                return self._asr

            logger.info("Initializing ASR model (exclusive VRAM block)")
            self._asr = FasterWhisperProASR(**self._asr_config)

            # Store in immortal reference ONCE - this reference never changes
            _IMMORTAL_ASR_REFERENCE = self._asr
            logger.debug("ASR stored in _IMMORTAL_ASR_REFERENCE (one-time, destructor prevention)")
        else:
            logger.debug("Reusing existing ASR model instance")

        return self._asr

    def cleanup(self):
        """End the ASR worker (refresh mode) before the base cleanup.

        The in-process immortal instance is deliberately NOT destroyed here
        (see the module header); the worker process simply exits.
        """
        asr = self._asr
        if asr is not None and hasattr(asr, "shutdown"):
            try:
                asr.shutdown()
            except Exception as e:  # noqa: BLE001 - cleanup is best-effort
                logger.warning(f"ASR worker shutdown failed (non-fatal): {e}")
            self._asr = None
        super().cleanup()

    def process(self, media_info: Dict) -> Dict:
        """Process media file through balanced pipeline with scene detection and VAD-enhanced ASR."""
        start_time = time.time()
        # Cross-cutting rule of the agreed error-handling table (2026-09-17):
        # anything that quietly fell short is reported in the run summary rather
        # than only in the log. Reset per file -- the pipeline object is reused
        # across the whole run.
        self.degradations = []

        input_file = media_info['path']
        media_basename = media_info['basename']

        # Report file start if async reporter available
        if self.progress_reporter:
            self.progress_reporter.report_file_start(
                filename=media_basename,
                file_number=media_info.get('file_number', 1),
                total_files=media_info.get('total_files', 1)
            )

        # Trace file start
        self.tracer.emit_file_start(
            filename=media_basename,
            file_number=media_info.get('file_number', 1),
            total_files=media_info.get('total_files', 1),
            media_info=media_info
        )

        master_metadata = self.metadata_manager.create_master_metadata(
            input_file=input_file,
            mode=self.get_mode_name(),
            media_info=media_info
        )

        # NOTE: reset_statistics moved to after ASR initialization (deferred loading)

        master_metadata["config"]["scene_detection_params"] = self.scene_detection_params
        master_metadata["config"]["vad_params"] = self.vad_params
        master_metadata["config"]["pipeline_options"] = self.pipeline_options

        try:
            # Step 1: Extract audio
            if self.progress_reporter:
                self.progress_reporter.report_step("Transforming audio", 1, 6)
            self.progress.set_current_step("Transforming audio", 1, 6)

            audio_path = self.temp_dir / f"{media_basename}_extracted.wav"
            extracted_audio, duration = self.audio_extractor.extract(input_file, audio_path)
            master_metadata["input_info"]["processed_audio_file"] = str(extracted_audio)
            master_metadata["input_info"]["audio_duration_seconds"] = duration
            self.metadata_manager.update_processing_stage(
                master_metadata, "audio_extraction", "completed",
                output_path=str(audio_path), duration_seconds=duration)

            # Trace audio extraction
            self.tracer.emit_audio_extraction(str(audio_path), duration)

            # Step 2: Detect scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Detecting audio scenes", 2, 6)
            self.progress.set_current_step("Detecting audio scenes", 2, 6)

            scenes_dir = self.temp_dir / "scenes"
            scenes_dir.mkdir(exist_ok=True)
            detection_result = self.scene_detector.detect_scenes(extracted_audio, scenes_dir, media_basename)
            scene_paths = detection_result.to_legacy_tuples()

            # Extract structured metadata from detection result
            detection_meta = detection_result.to_metadata_dict()
            master_metadata["scenes_detected"] = detection_meta["scenes_detected"]
            # Include VAD segments if available (Silero method)
            if detection_meta.get("vad_segments"):
                master_metadata["vad_segments"] = detection_meta["vad_segments"]
                master_metadata["vad_method"] = detection_meta.get("vad_method")
                master_metadata["vad_params"] = detection_meta.get("vad_params")
            # Include coarse boundaries (Pass 1 scene boundaries before splitting)
            if detection_meta.get("coarse_boundaries"):
                master_metadata["coarse_boundaries"] = detection_meta["coarse_boundaries"]
            master_metadata["summary"]["total_scenes_detected"] = len(scene_paths)
            self.metadata_manager.update_processing_stage(
                master_metadata, "scene_detection", "completed",
                scene_count=len(scene_paths), scenes_dir=str(scenes_dir))

            # Trace scene detection
            self.tracer.emit_scene_detection(
                method=self.scene_detector.name,
                params=self.scene_detection_params,
                scenes_found=len(scene_paths),
                scene_stats={
                    "total_duration": sum(d for _, _, _, d in scene_paths),
                    "shortest": min((d for _, _, _, d in scene_paths), default=0),
                    "longest": max((d for _, _, _, d in scene_paths), default=0),
                }
            )

            # Tell the user which scenes look likely to lose speech, before they
            # spend the run finding out. Reads and prints only; never raises.
            _analytics = analytics.report(
                extracted_audio,
                [(i, s.start_sec, s.end_sec)
                 for i, s in enumerate(detection_result.scenes)],
                scene_method=self.scene_detector.name,
                vad_threshold=self.vad_params.get("threshold", 0.40),
            )
            if _analytics is not None:
                master_metadata["audio_analytics"] = _analytics.to_dict()

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

            self.progress.set_current_step("Preparing audio for ASR", 3, 6)

            if self._enhancer_is_passthrough:
                # v1.8.5+: Scenes already at 16kHz — skip enhancement entirely
                logger.info(
                    "Speech enhancer is passthrough — %d scenes at 16kHz, skipping enhancement",
                    len(scene_paths),
                )
                enhancer_name = "none"
                enhancer_display = "None (passthrough)"
            else:
                # A. Load Enhancer (always succeeds - "none" backend is fallback)
                enhancer = create_enhancer_from_config(self._enhancer_config)
                enhancer_name = enhancer.name
                enhancer_display = enhancer.display_name
                logger.info(f"Processing {len(scene_paths)} scenes with {enhancer_display}")

                def enhancement_progress(scene_num, total, name):
                    if scene_num == 1 or scene_num % 5 == 0 or scene_num == total:
                        pct = (scene_num / total) * 100
                        print(f"\rProcessing: [{scene_num}/{total}] {pct:.0f}%", end='', flush=True)

                # B. Process scenes (enhancement + 48kHz→16kHz resampling)
                scene_paths = enhance_scenes(
                    scene_paths,
                    enhancer,
                    self.temp_dir,
                    progress_callback=enhancement_progress,
                    degradations=self.degradations,
                )
                print()  # Newline after progress

                # C. DESTROY Enhancer - This is the "JIT Unload"
                # We must confirm VRAM is near-zero before loading ASR
                logger.debug("Destroying enhancer to free VRAM before ASR load")
                enhancer.cleanup()
                del enhancer
                gc.collect()
                if _torch_available:
                    try:
                        torch.cuda.empty_cache()
                        logger.debug("GPU memory cleared after enhancement - VRAM should be near-zero")
                    except Exception as e:
                        # CUDA context may be corrupted from prior OOM during enhancement
                        # Log and continue - ASR phase will either work (fresh allocation) or fail explicitly
                        logger.warning(f"CUDA cache clear failed after enhancement: {e}")

            master_metadata["config"]["speech_enhancement"] = {
                "enabled": not self._enhancer_is_passthrough,
                "backend": enhancer_name,
            }

            # =================================================================
            # PHASE 2: ASR TRANSCRIPTION (Model Reuse Pattern)
            # ASR is created ONCE and reused for all files in batch.
            # This prevents ctranslate2 destructor crashes that occurred when
            # multiple ASR instances were created and the old ones were GC'd.
            # The _ensure_asr() method handles lazy initialization and stores
            # the reference in _IMMORTAL_ASR_REFERENCE (one-time).
            # Nuclear Exit (os._exit(0)) in pass_worker.py handles final cleanup.
            # =================================================================

            # Get or create the shared ASR instance
            asr = self._ensure_asr()

            # Reset per-file statistics (safe - just Python dict assignment)
            if hasattr(asr, "reset_statistics"):
                asr.reset_statistics()
            # v1.9.2 (CFF1): the proxy's refresh counter spans the batch; report per file.
            _refreshes_at_file_start = int(getattr(asr, "refresh_count", 0) or 0)

            # Trace ASR config before transcription
            self.tracer.emit_asr_config(
                model=asr.model_name,
                backend="faster-whisper",
                params=self.pipeline_options.get("decoder", {})
            )

            # Step 4: Transcribe scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Transcribing scenes with VAD", 4, 6)
            self.progress.set_current_step("Transcribing scenes with VAD", 4, 6)

            scene_srts_dir = self.temp_dir / "scene_srts"
            scene_srts_dir.mkdir(exist_ok=True)
            scene_srt_info = []

            # Start scene transcription with unified progress management
            self.progress.start_subtask("Transcribing scenes", len(scene_paths))

            # Check if we have access to unified progress manager for external library suppression
            unified_manager = getattr(self.progress, 'unified_manager', None)

            # Scene-level progress tracking variables
            last_update_time = time.time()
            transcription_start_time = time.time()
            update_interval = 30  # seconds
            batch_update_size = 5  # scenes
            total_scenes = len(scene_paths)

            # Print initial scene transcription header (always visible)
            print(f"\nTranscribing {total_scenes} scenes with VAD-enhanced processing:")

            # Accumulate VAD segments across all scenes for visualization data contract
            all_vad_segments = []

            # v1.9.2 (owner Part C): name the mechanism actually in use, so the
            # terminal states what ran instead of leaving the user to infer it.
            _seg_name = asr.get_segmenter_name() if hasattr(asr, 'get_segmenter_name') else "none"
            _detection_label = (
                "Internal FW Silero VAD" if _seg_name == "none"
                else f"speech segmenter '{_seg_name}'"
            )

            # v1.9.2 (owner V1/V3): the consecutive-empty-scene tracker is gone.
            # Its signal was "the detector reported speech and nothing came back",
            # which is produced identically by two opposite situations it cannot
            # separate -- a recogniser that has stopped working, and a scene with
            # no intelligible speech in it (a long continuous action scene, or the
            # music performance in #324). It was also inert under the built-in VAD,
            # so it recorded a reassuring zero for exactly the runs that failed.
            # The per-scene telemetry below records the facts instead.

            # #394 diagnostics: per-scene record of decode behaviour and
            # memory, so the *approach* to a failure is visible and not only its
            # aftermath. On by default (raw_subs/ next to the outputs);
            # --asr-telemetry moves it, --no-asr-telemetry switches it off.
            telemetry = None
            _tp = resolve_telemetry_path(
                getattr(self, 'asr_telemetry_path', None),
                getattr(self, 'asr_telemetry_enabled', True),
                self.output_dir,
                media_basename,
                getattr(self, 'asr_telemetry_tag', None),
            )
            if _tp is not None:
                telemetry = AsrTelemetry(_tp, media_basename)
            # Reachable from the error handler below, which cannot see this local.
            self._active_telemetry = telemetry

            for idx, (scene_path, start_time_sec, _, _) in enumerate(scene_paths):
                scene_srt_path = scene_srts_dir / f"{scene_path.stem}.srt"
                scene_num = idx + 1

                # Show scene-level progress for user feedback (bypass adapter filtering)
                should_show_update = (
                    scene_num == 1 or  # Always show first scene
                    scene_num % batch_update_size == 0 or  # Every 5 scenes
                    time.time() - last_update_time > update_interval or  # Every 30 seconds
                    scene_num == len(scene_paths)  # Always show last scene
                )

                if should_show_update:
                    # Create tqdm-style progress bar (ASCII-compatible for Windows)
                    progress_pct = (scene_num / total_scenes) * 100
                    bar_width = 30
                    filled_width = int(bar_width * scene_num / total_scenes)
                    progress_bar = '=' * filled_width + '-' * (bar_width - filled_width)

                    # Calculate ETA (only after processing a few scenes)
                    eta_text = ""
                    if scene_num > 3:
                        elapsed = time.time() - transcription_start_time
                        avg_time_per_scene = elapsed / scene_num
                        remaining_scenes = total_scenes - scene_num
                        eta_seconds = remaining_scenes * avg_time_per_scene

                        if eta_seconds > 60:
                            eta_text = f" | ETA: {eta_seconds/60:.1f}m"
                        else:
                            eta_text = f" | ETA: {eta_seconds:.0f}s"

                    # Format scene filename (truncate if needed)
                    scene_filename = scene_path.name
                    if len(scene_filename) > 25:
                        scene_filename = scene_filename[:22] + "..."

                    # Direct console output (bypasses adapter filtering)
                    progress_line = f"\rTranscribing: [{progress_bar}] {scene_num}/{total_scenes} [{progress_pct:.1f}%] | {scene_filename}{eta_text}"
                    print(progress_line, end='', flush=True)

                    last_update_time = time.time()

                _scene_wall = 0.0
                try:
                    _scene_t0 = time.time()
                    # Use unified progress manager's external suppression if available
                    if unified_manager:
                        with unified_manager.suppress_external_progress():
                            asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
                    else:
                        asr.transcribe_to_srt(scene_path, scene_srt_path, task=self.asr_task)
                    _scene_wall = time.time() - _scene_t0

                    # v1.9.2 (owner Part C): report what this scene actually yielded.
                    # A scene that cost real time and returned nothing is called out,
                    # because that is the condition worth noticing.
                    _yield = 0
                    try:
                        if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                            _yield = scene_srt_path.read_text(encoding='utf-8').count(' --> ')
                    except Exception:  # noqa: BLE001 - reporting must never break a run
                        _yield = -1
                    # One line per scene, newline-terminated. No carriage return and
                    # no padding: the progress bar above draws in place with \r and
                    # redraws on its next update, so a plain line cannot tear it.
                    print(
                        f"  Scene {scene_num}/{total_scenes} "
                        f"({scene_paths[idx][3]:.0f}s, {_detection_label}): "
                        f"{_yield} subtitle(s) in {_scene_wall:.0f}s"
                        f"{'  <-- NO OUTPUT' if _yield == 0 else ''}",
                        flush=True,
                    )

                    # Process results - simplified to reduce message spam
                    if scene_srt_path.exists() and scene_srt_path.stat().st_size > 0:
                        scene_srt_info.append((scene_srt_path, start_time_sec))
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["srt_path"] = str(scene_srt_path)
                    else:
                        master_metadata["scenes_detected"][idx]["transcribed"] = True
                        master_metadata["scenes_detected"][idx]["no_speech_detected"] = True

                    # Speech regions, when an external detector actually produced
                    # any. Under the built-in VAD this is empty by design: the
                    # recogniser does its own detection and does not report the
                    # regions it used, so there is nothing honest to record here.
                    if hasattr(asr, 'get_last_vad_segments'):
                        for seg in asr.get_last_vad_segments():
                            all_vad_segments.append({
                                "start_sec": round(start_time_sec + seg["start_sec"], 3),
                                "end_sec": round(start_time_sec + seg["end_sec"], 3),
                            })

                    _scene_produced = bool(
                        scene_srt_path.exists() and scene_srt_path.stat().st_size > 0
                    )
                    self.progress.update_subtask(1)

                except Exception as e:
                    # The time was really spent, so record it. Leaving it at 0.0
                    # would make the telemetry report rtf 0.0 for a scene that may
                    # have burned minutes before failing.
                    _scene_wall = time.time() - _scene_t0
                    # Show errors with scene context
                    self.progress.show_message(f"Scene {scene_num}/{len(scene_paths)} failed: {str(e)}", "error", 2.0)
                    master_metadata["scenes_detected"][idx]["transcribed"] = False
                    master_metadata["scenes_detected"][idx]["error"] = str(e)
                    _scene_produced = False
                    print(
                        (f"  Scene {scene_num}/{total_scenes} "
                         f"({scene_paths[idx][3]:.0f}s, {_detection_label}): "
                         f"FAILED after {_scene_wall:.0f}s -- {e}")[:160],
                        flush=True,
                    )
                    self.progress.update_subtask(1)

                # v1.9.2 (owner in1): the recorder sits OUTSIDE the try above. It
                # only observes, so it must never be able to mark a scene failed --
                # which it could when it lived inside the same block.
                if telemetry is not None:
                    try:
                        telemetry.record_scene(
                            index=scene_num,
                            audio_duration_s=scene_paths[idx][3],
                            wall_s=_scene_wall,
                            segments=(asr.get_last_decode_stats()
                                      if hasattr(asr, 'get_last_decode_stats') else []),
                            produced_output=_scene_produced,
                            model_epoch=getattr(asr, "epoch", None),
                        )
                    except Exception as _te:  # noqa: BLE001 - an observer never fails a run
                        logger.debug("Telemetry record failed for scene %s: %s", scene_num, _te)

                # v1.9.2 (CFF1 / D2): every scene handed to the recogniser counts
                # toward the refresh budget (failed ones included, same as
                # Fidelity); the worker is replaced before the next scene once
                # the budget is spent.
                if hasattr(asr, "record_audio"):
                    asr.record_audio(scene_paths[idx][3])
                master_metadata["scenes_detected"][idx]["model_epoch"] = getattr(asr, "epoch", 1)

            self.progress.finish_subtask()

            # Save accumulated ASR-level VAD segments to metadata for visualization
            # Only recorded when an external detector actually reported regions.
            # This used to write vad_method "silero" and a set of Silero parameters
            # for every run, including runs where no external detector ran at all.
            if all_vad_segments:
                master_metadata["vad_segments"] = all_vad_segments
                master_metadata["vad_method"] = asr.get_segmenter_name() if hasattr(
                    asr, 'get_segmenter_name') else "unknown"
                master_metadata["vad_params"] = self.vad_params

            # Print completion message for scene transcription (always visible)
            print(f"\n[DONE] Completed transcription of {total_scenes} scenes")

            # Step 5: Stitch scenes
            if self.progress_reporter:
                self.progress_reporter.report_step("Combining scene transcriptions", 5, 6)
            self.progress.set_current_step("Combining scene transcriptions", 5, 6)

            stitched_srt_path = self.temp_dir / f"{media_basename}_stitched.srt"
            num_subtitles = self.stitcher.stitch(scene_srt_info, stitched_srt_path)
            self.metadata_manager.update_processing_stage(
                master_metadata, "stitching", "completed",
                subtitle_count=num_subtitles, output_path=str(stitched_srt_path))

            # Step 6: Post-process
            if self.progress_reporter:
                self.progress_reporter.report_step("Post-processing subtitles", 6, 6)
            self.progress.set_current_step("Post-processing subtitles", 6, 6)

            final_srt_path = self.output_dir / f"{media_basename}.{self.lang_code}.whisperjav.srt"
            processed_srt_path, stats = self.standard_postprocessor.process(stitched_srt_path, final_srt_path)

            # Ensure the final SRT is in the output directory
            if processed_srt_path != final_srt_path:
                shutil.copy2(processed_srt_path, final_srt_path)
                logger.debug(f"Copied final SRT from {processed_srt_path} to {final_srt_path}")

            # Move raw_subs folder to output directory
            temp_raw_subs_path = stitched_srt_path.parent / "raw_subs"
            if temp_raw_subs_path.exists():
                final_raw_subs_path = self.output_dir / "raw_subs"
                # Create raw_subs directory if it doesn't exist
                final_raw_subs_path.mkdir(exist_ok=True)

                # Copy only files related to current media_basename to avoid ghost files
                for file in temp_raw_subs_path.glob(f"{media_basename}*"):
                    dest_file = final_raw_subs_path / file.name
                    shutil.copy2(file, dest_file)
                    logger.debug(f"Copied {file.name} to raw_subs")

                logger.debug(f"Copied relevant raw_subs files to: {final_raw_subs_path}")

            self.metadata_manager.update_processing_stage(
                master_metadata, "postprocessing", "completed", statistics=stats, output_path=str(final_srt_path))

            master_metadata["output_files"]["final_srt"] = str(final_srt_path)
            master_metadata["output_files"]["stitched_srt"] = str(stitched_srt_path)
            master_metadata["summary"]["final_subtitles_refined"] = stats.get('total_subtitles', 0) - stats.get('empty_removed', 0)
            master_metadata["summary"]["final_subtitles_raw"] = num_subtitles
            master_metadata["summary"]["quality_metrics"] = {
                "hallucinations_removed": stats.get('removed_hallucinations', 0),
                "repetitions_removed": stats.get('removed_repetitions', 0),
                "duration_adjustments": stats.get('duration_adjustments', 0),
                "empty_removed": stats.get('empty_removed', 0)
            }

            logprob_filtered = 0
            nonverbal_filtered = 0
            if hasattr(asr, "get_filter_statistics"):
                filter_stats = asr.get_filter_statistics() or {}
                logprob_filtered = filter_stats.get('logprob_filtered', 0)
                nonverbal_filtered = filter_stats.get('nonverbal_filtered', 0)

            # C. ASR CLEANUP INTENTIONALLY SKIPPED (Model Reuse Pattern)
            # ================================================================
            # DO NOT call asr.cleanup(), del asr, or gc.collect() here!
            #
            # The ASR model (self._asr) is REUSED across all files in the batch.
            # This prevents ctranslate2 C++ destructor crashes that occurred when
            # multiple ASR instances were created and old ones were garbage collected.
            #
            # The model is stored in both self._asr and _IMMORTAL_ASR_REFERENCE.
            # Nuclear Exit (os._exit(0)) in pass_worker.py terminates the process
            # without calling Python destructors - the OS reclaims all memory.
            #
            # VRAM stays allocated (~3GB) until process exit, but this is safe
            # because the same model is reused for all files.
            # ================================================================
            logger.debug("ASR cleanup skipped (Model Reuse Pattern - Nuclear Exit will handle)")

            master_metadata["summary"]["final_subtitles_raw"] += logprob_filtered + nonverbal_filtered
            master_metadata["summary"]["quality_metrics"].update({
                "logprob_filtered": logprob_filtered,
                "nonverbal_filtered": nonverbal_filtered,
                "cps_filtered": stats.get('cps_filtered', 0)
            })

            total_time = time.time() - start_time
            master_metadata["summary"]["total_processing_time_seconds"] = round(total_time, 2)
            # Carried out with the rest of the summary so every caller sees it
            # the same way: the plain path, the async path, and a pass running in
            # its own process (agreed error-handling table, 2026-09-17).
            master_metadata["summary"]["degradations"] = list(getattr(self, "degradations", None) or [])
            # #394: every scene is already on disk; this logs the trend line.
            if telemetry is not None:
                telemetry.finalize()
            self._active_telemetry = None

            # v1.9.2 (CFF1): how often the recogniser was replaced during this file
            # (the proxy counts across the batch; the difference is this file's share).
            master_metadata["summary"]["model_refresh_audio_minutes"] = self.model_refresh_audio_minutes
            master_metadata["summary"]["model_refreshes"] = (
                int(getattr(asr, "refresh_count", 0) or 0) - _refreshes_at_file_start
            )
            master_metadata["metadata_master"]["updated_at"] = datetime.now().isoformat() + "Z"

            self.metadata_manager.save_master_metadata(master_metadata, media_basename)
            self.cleanup_temp_files(media_basename)

            # Trace postprocessing
            self.tracer.emit_postprocessing(stats)

            # Trace completion
            self.tracer.emit_completion(
                success=True,
                final_subtitles=master_metadata["summary"]["final_subtitles_refined"],
                total_duration=total_time,
                output_path=str(final_srt_path)
            )

            # Report completion
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=True,
                    stats={
                        'subtitles': master_metadata["summary"]["final_subtitles_refined"],
                        'duration': total_time,
                        'scenes': len(scene_paths)
                    }
                )

            return master_metadata

        except Exception as e:
            self.progress.show_message(f"Pipeline error: {str(e)}", "error", 0)
            logger.error(f"Pipeline error: {e}", exc_info=True)
            # #394: the scenes recorded so far are already on disk; say where.
            _t = getattr(self, '_active_telemetry', None)
            if _t is not None:
                _t.finalize()
                self._active_telemetry = None
            self.metadata_manager.update_processing_stage(
                master_metadata, "error", "failed", error_message=str(e))
            self.metadata_manager.save_master_metadata(master_metadata, media_basename)

            # Trace failure
            self.tracer.emit_completion(
                success=False,
                final_subtitles=0,
                total_duration=time.time() - start_time,
                output_path="",
                error=str(e)
            )

            # Report failure
            if self.progress_reporter:
                self.progress_reporter.report_completion(
                    success=False,
                    stats={'error': str(e)}
                )

            raise

    def get_mode_name(self) -> str:
        return "balanced"

