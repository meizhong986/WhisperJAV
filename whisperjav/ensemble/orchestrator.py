"""Ensemble Orchestrator for two-pass pipeline processing."""

import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from whisperjav.config.legacy import resolve_legacy_pipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.pipelines.kotoba_faster_whisper_pipeline import KotobaFasterWhisperPipeline
from whisperjav.pipelines.transformers_pipeline import TransformersPipeline
from whisperjav.utils.logger import logger
from whisperjav.utils.metadata_manager import MetadataManager

from .merge import MergeEngine


# Pipeline class mapping
PIPELINE_CLASSES = {
    'balanced': BalancedPipeline,
    'fast': FastPipeline,
    'faster': FasterPipeline,
    'fidelity': FidelityPipeline,
    'kotoba-faster-whisper': KotobaFasterWhisperPipeline,
    'transformers': TransformersPipeline,
}


class EnsembleOrchestrator:
    """Orchestrates two-pass ensemble processing with result merging."""

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        subs_language: str = 'native',
        progress_display=None,
        **kwargs
    ):
        """
        Initialize the ensemble orchestrator.

        Args:
            output_dir: Output directory for final results
            temp_dir: Temporary directory for processing
            keep_temp_files: Whether to keep intermediate files
            subs_language: Language for subtitles ('native' or 'direct-to-english')
            progress_display: Progress display object
            **kwargs: Additional parameters passed to pipelines
        """
        self.output_dir = Path(output_dir)
        self.temp_dir = Path(temp_dir)
        self.keep_temp_files = keep_temp_files
        self.subs_language = subs_language
        self.progress_display = progress_display
        self.extra_kwargs = kwargs

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_manager = MetadataManager(self.temp_dir, self.output_dir)
        self.merge_engine = MergeEngine()

    def process(
        self,
        media_info: Dict,
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]] = None,
        merge_strategy: str = 'smart_merge'
    ) -> Dict:
        """
        Process media file through two-pass ensemble.

        Args:
            media_info: Dictionary containing media file information
            pass1_config: Configuration for pass 1
                - pipeline: Pipeline name ('balanced', 'fast', 'faster', 'fidelity')
                - sensitivity: Sensitivity level ('conservative', 'balanced', 'aggressive')
                - overrides: Optional parameter overrides dict
            pass2_config: Configuration for pass 2 (same structure as pass1)
                         If None, only pass 1 is executed
            merge_strategy: Merge strategy name ('smart_merge', 'full_merge', 'pass1_primary', etc.)

        Returns:
            Dictionary containing processing metadata and results
        """
        start_time = time.time()
        media_basename = media_info['basename']

        logger.info(f"Starting ensemble processing for {media_basename}")

        # Create ensemble metadata
        ensemble_metadata = self._create_ensemble_metadata(
            media_info, pass1_config, pass2_config, merge_strategy
        )

        try:
            # Execute Pass 1
            logger.info(f"Pass 1: {pass1_config['pipeline']} ({pass1_config.get('sensitivity', 'balanced')})")
            pass1_result = self._execute_pass(
                media_info=media_info,
                pass_config=pass1_config,
                pass_number=1
            )
            pass1_srt = Path(pass1_result['output_files']['final_srt'])

            # Store pass 1 result with distinct name
            pass1_output = self.output_dir / f"{media_basename}_pass1.srt"
            if pass1_srt.exists():
                import shutil
                shutil.copy2(pass1_srt, pass1_output)

            ensemble_metadata['pass1'] = {
                'status': 'completed',
                'srt_path': str(pass1_output),
                'subtitles': pass1_result['summary'].get('final_subtitles_refined', 0),
                'processing_time': pass1_result['summary'].get('total_processing_time_seconds', 0)
            }

            # Execute Pass 2 if enabled
            if pass2_config:
                logger.info(f"Pass 2: {pass2_config['pipeline']} ({pass2_config.get('sensitivity', 'balanced')})")
                pass2_result = self._execute_pass(
                    media_info=media_info,
                    pass_config=pass2_config,
                    pass_number=2
                )
                pass2_srt = Path(pass2_result['output_files']['final_srt'])

                # Store pass 2 result with distinct name
                pass2_output = self.output_dir / f"{media_basename}_pass2.srt"
                if pass2_srt.exists():
                    import shutil
                    shutil.copy2(pass2_srt, pass2_output)

                ensemble_metadata['pass2'] = {
                    'status': 'completed',
                    'srt_path': str(pass2_output),
                    'subtitles': pass2_result['summary'].get('final_subtitles_refined', 0),
                    'processing_time': pass2_result['summary'].get('total_processing_time_seconds', 0)
                }

                # Merge results
                logger.info(f"Merging results using {merge_strategy} strategy")
                merged_srt = self.output_dir / f"{media_basename}.merged.srt"

                merge_stats = self.merge_engine.merge(
                    srt1_path=pass1_output,
                    srt2_path=pass2_output,
                    output_path=merged_srt,
                    strategy=merge_strategy
                )

                ensemble_metadata['merge'] = {
                    'status': 'completed',
                    'strategy': merge_strategy,
                    'output_path': str(merged_srt),
                    'statistics': merge_stats
                }

                final_output = merged_srt
            else:
                # Single pass mode - use pass 1 result directly
                final_output = pass1_output
                ensemble_metadata['pass2'] = {'status': 'skipped'}
                ensemble_metadata['merge'] = {'status': 'skipped'}

            # Determine language code for final output naming
            if self.subs_language == 'direct-to-english':
                lang_code = 'en'
            else:
                # Check params first (customized mode), then overrides (legacy)
                if pass1_config.get('params'):
                    lang_code = pass1_config['params'].get('language', 'ja')
                else:
                    lang_code = pass1_config.get('overrides', {}).get('language', 'ja')

            # Create final output with standard naming
            final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
            if final_output.exists() and final_output != final_srt_path:
                import shutil
                shutil.copy2(final_output, final_srt_path)

            # Update metadata
            total_time = time.time() - start_time
            ensemble_metadata['summary'] = {
                'total_processing_time_seconds': round(total_time, 2),
                'final_output': str(final_srt_path),
                'passes_completed': 2 if pass2_config else 1
            }
            ensemble_metadata['output_files'] = {
                'final_srt': str(final_srt_path),
                'pass1_srt': str(pass1_output) if pass1_output.exists() else None,
                'pass2_srt': str(ensemble_metadata.get('pass2', {}).get('srt_path')) if pass2_config else None
            }

            # Clean up intermediate files if not keeping
            if not self.keep_temp_files:
                self._cleanup_intermediate(media_basename)

            logger.info(f"Ensemble processing completed in {total_time:.1f}s")
            return ensemble_metadata

        except Exception as e:
            logger.error(f"Ensemble processing failed: {e}", exc_info=True)
            ensemble_metadata['error'] = str(e)
            ensemble_metadata['status'] = 'failed'
            raise

    def process_batch(
        self,
        media_files: List[Dict],
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]] = None,
        merge_strategy: str = 'smart_merge'
    ) -> List[Dict]:
        """
        Process multiple media files through two-pass ensemble with optimized memory usage.

        This method loads each model only once for all files, significantly reducing
        VRAM usage compared to processing files individually.

        Args:
            media_files: List of media info dictionaries
            pass1_config: Configuration for pass 1
            pass2_config: Configuration for pass 2 (optional)
            merge_strategy: Merge strategy name

        Returns:
            List of ensemble metadata dictionaries, one per file
        """
        if not media_files:
            return []

        batch_start_time = time.time()
        total_files = len(media_files)

        logger.info(f"Starting batch ensemble processing for {total_files} files")

        # Storage for intermediate results
        pass1_results = {}  # basename -> result metadata
        pass1_srts = {}     # basename -> Path to pass1 SRT
        pass2_results = {}  # basename -> result metadata
        pass2_srts = {}     # basename -> Path to pass2 SRT
        failed_files = []   # List of failed basenames
        all_metadata = []   # Final results

        # ========== PASS 1 PROCESSING ==========
        logger.info(f"=== Pass 1: {pass1_config['pipeline']} ({pass1_config.get('sensitivity', 'balanced')}) ===")

        # Create Pass 1 pipeline once
        pass1_pipeline = self._create_pipeline(pass1_config, pass_number=1)

        try:
            for i, media_info in enumerate(media_files, 1):
                media_basename = media_info['basename']
                logger.info(f"Pass 1 - File {i}/{total_files}: {media_basename}")

                try:
                    result = pass1_pipeline.process(media_info)
                    pass1_srt = Path(result['output_files']['final_srt'])

                    # Store result with distinct name
                    pass1_output = self.output_dir / f"{media_basename}_pass1.srt"
                    if pass1_srt.exists():
                        import shutil
                        shutil.copy2(pass1_srt, pass1_output)

                    pass1_results[media_basename] = result
                    pass1_srts[media_basename] = pass1_output

                except Exception as e:
                    logger.error(f"Pass 1 failed for {media_basename}: {e}")
                    failed_files.append(media_basename)
                    continue

        finally:
            # Always cleanup Pass 1 pipeline to free VRAM
            logger.info("Cleaning up Pass 1 pipeline...")
            try:
                pass1_pipeline.cleanup()
            except Exception as e:
                logger.warning(f"Pass 1 cleanup error (non-fatal): {e}")
            finally:
                del pass1_pipeline
                # Give CUDA time to release resources before next model load
                time.sleep(0.5)

        # ========== PASS 2 PROCESSING ==========
        if pass2_config:
            logger.info(f"=== Pass 2: {pass2_config['pipeline']} ({pass2_config.get('sensitivity', 'balanced')}) ===")

            # Create Pass 2 pipeline
            pass2_pipeline = self._create_pipeline(pass2_config, pass_number=2)

            try:
                for i, media_info in enumerate(media_files, 1):
                    media_basename = media_info['basename']

                    # Skip files that failed in Pass 1
                    if media_basename in failed_files:
                        logger.debug(f"Skipping {media_basename} (failed in Pass 1)")
                        continue

                    logger.info(f"Pass 2 - File {i}/{total_files}: {media_basename}")

                    try:
                        result = pass2_pipeline.process(media_info)
                        pass2_srt = Path(result['output_files']['final_srt'])

                        # Store result with distinct name
                        pass2_output = self.output_dir / f"{media_basename}_pass2.srt"
                        if pass2_srt.exists():
                            import shutil
                            shutil.copy2(pass2_srt, pass2_output)

                        pass2_results[media_basename] = result
                        pass2_srts[media_basename] = pass2_output

                    except Exception as e:
                        logger.error(f"Pass 2 failed for {media_basename}: {e}")
                        # Don't add to failed_files - we still have Pass 1 result
                        continue

            finally:
                # Always cleanup Pass 2 pipeline to free VRAM
                logger.info("Cleaning up Pass 2 pipeline...")
                try:
                    pass2_pipeline.cleanup()
                except Exception as e:
                    logger.warning(f"Pass 2 cleanup error (non-fatal): {e}")
                finally:
                    del pass2_pipeline
                    # Give CUDA time to release resources before merge
                    time.sleep(0.5)

        # ========== MERGE PHASE ==========
        logger.info("=== Merging results ===")

        for media_info in media_files:
            media_basename = media_info['basename']

            # Create ensemble metadata for this file
            ensemble_metadata = self._create_ensemble_metadata(
                media_info, pass1_config, pass2_config, merge_strategy
            )

            # Handle failed files
            if media_basename in failed_files:
                ensemble_metadata['status'] = 'failed'
                ensemble_metadata['error'] = 'Failed in Pass 1'
                all_metadata.append(ensemble_metadata)
                continue

            # Get Pass 1 result
            if media_basename in pass1_results:
                pass1_result = pass1_results[media_basename]
                ensemble_metadata['pass1'] = {
                    'status': 'completed',
                    'srt_path': str(pass1_srts[media_basename]),
                    'subtitles': pass1_result['summary'].get('final_subtitles_refined', 0),
                    'processing_time': pass1_result['summary'].get('total_processing_time_seconds', 0)
                }
            else:
                ensemble_metadata['pass1'] = {'status': 'failed'}
                ensemble_metadata['status'] = 'failed'
                all_metadata.append(ensemble_metadata)
                continue

            # Get Pass 2 result and merge
            if pass2_config and media_basename in pass2_results:
                pass2_result = pass2_results[media_basename]
                ensemble_metadata['pass2'] = {
                    'status': 'completed',
                    'srt_path': str(pass2_srts[media_basename]),
                    'subtitles': pass2_result['summary'].get('final_subtitles_refined', 0),
                    'processing_time': pass2_result['summary'].get('total_processing_time_seconds', 0)
                }

                # Merge results
                merged_srt = self.output_dir / f"{media_basename}.merged.srt"
                try:
                    merge_stats = self.merge_engine.merge(
                        srt1_path=pass1_srts[media_basename],
                        srt2_path=pass2_srts[media_basename],
                        output_path=merged_srt,
                        strategy=merge_strategy
                    )

                    ensemble_metadata['merge'] = {
                        'status': 'completed',
                        'strategy': merge_strategy,
                        'output_path': str(merged_srt),
                        'statistics': merge_stats
                    }
                    final_output = merged_srt
                except Exception as e:
                    logger.error(f"Merge failed for {media_basename}: {e}")
                    ensemble_metadata['merge'] = {'status': 'failed', 'error': str(e)}
                    final_output = pass1_srts[media_basename]
            else:
                # Single pass or Pass 2 failed
                if pass2_config:
                    ensemble_metadata['pass2'] = {'status': 'failed'}
                else:
                    ensemble_metadata['pass2'] = {'status': 'skipped'}
                ensemble_metadata['merge'] = {'status': 'skipped'}
                final_output = pass1_srts[media_basename]

            # Determine language code for final output naming
            if self.subs_language == 'direct-to-english':
                lang_code = 'en'
            else:
                if pass1_config.get('params'):
                    lang_code = pass1_config['params'].get('language', 'ja')
                else:
                    lang_code = pass1_config.get('overrides', {}).get('language', 'ja')

            # Create final output with standard naming
            final_srt_path = self.output_dir / f"{media_basename}.{lang_code}.whisperjav.srt"
            if final_output.exists() and final_output != final_srt_path:
                import shutil
                shutil.copy2(final_output, final_srt_path)

            # Update metadata
            ensemble_metadata['summary'] = {
                'final_output': str(final_srt_path),
                'passes_completed': 2 if (pass2_config and media_basename in pass2_results) else 1
            }
            ensemble_metadata['output_files'] = {
                'final_srt': str(final_srt_path),
                'pass1_srt': str(pass1_srts.get(media_basename)) if media_basename in pass1_srts else None,
                'pass2_srt': str(pass2_srts.get(media_basename)) if media_basename in pass2_srts else None
            }

            all_metadata.append(ensemble_metadata)

        # Cleanup intermediate files if not keeping
        if not self.keep_temp_files:
            for media_info in media_files:
                self._cleanup_intermediate(media_info['basename'])

        batch_time = time.time() - batch_start_time
        logger.info(f"Batch ensemble processing completed in {batch_time:.1f}s for {total_files} files")

        return all_metadata

    def _apply_custom_params(
        self,
        resolved_config: Dict[str, Any],
        custom_params: Dict[str, Any],
        pass_number: int
    ) -> None:
        """
        Apply custom parameters to a resolved config (M1: extracted from duplicate code).

        Handles both V3 config structure (kotoba with params.asr) and legacy structure
        (params.decoder/provider/vad). Modifies resolved_config in place.

        Args:
            resolved_config: Resolved pipeline configuration to modify
            custom_params: Custom parameters to apply
            pass_number: Pass number for logging
        """
        model_config = resolved_config['model']

        # Detect config structure: V3 uses params.asr, legacy uses params.decoder
        is_v3_config = 'asr' in resolved_config['params']

        # Parameters to skip (handled elsewhere or not applicable)
        SKIP_PARAMS = {'scene_detection_method'}

        if is_v3_config:
            # V3 config (kotoba): all params go to params.asr
            asr_params = resolved_config['params']['asr']

            for key, value in custom_params.items():
                if key in SKIP_PARAMS:
                    continue
                elif key == 'model_name':
                    model_config['model_name'] = value
                    logger.debug(f"Pass {pass_number}: Overriding model to {value}")
                elif key == 'device':
                    model_config['device'] = value
                    logger.debug(f"Pass {pass_number}: Overriding device to {value}")
                else:
                    # All other params go to asr_params for V3 config
                    asr_params[key] = value
                    logger.debug(f"Pass {pass_number}: Set asr param '{key}' = {value}")
        else:
            # Legacy config: params split into decoder/provider/vad
            decoder_params = resolved_config['params']['decoder']
            provider_params = resolved_config['params']['provider']
            vad_params = resolved_config['params'].get('vad', {})

            # Define which params belong to VAD
            VAD_PARAM_NAMES = {
                'threshold', 'neg_threshold', 'min_speech_duration_ms',
                'max_speech_duration_s', 'min_silence_duration_ms', 'speech_pad_ms'
            }

            for key, value in custom_params.items():
                if key in SKIP_PARAMS:
                    continue
                elif key == 'model_name':
                    model_config['model_name'] = value
                    logger.debug(f"Pass {pass_number}: Overriding model to {value}")
                elif key == 'device':
                    model_config['device'] = value
                    logger.debug(f"Pass {pass_number}: Overriding device to {value}")
                elif key in VAD_PARAM_NAMES:
                    vad_params[key] = value
                elif key in decoder_params:
                    decoder_params[key] = value
                elif key in provider_params:
                    provider_params[key] = value
                else:
                    # Unknown params go to provider (safer than decoder)
                    provider_params[key] = value
                    logger.debug(f"Pass {pass_number}: Unknown param '{key}' added to provider_params")

        logger.debug(f"Pass {pass_number}: Custom params applied")

    def _create_pipeline(
        self,
        pass_config: Dict[str, Any],
        pass_number: int
    ):
        """
        Create a pipeline instance for the given configuration.

        Args:
            pass_config: Pass configuration
            pass_number: Pass number (1 or 2)

        Returns:
            Pipeline instance
        """
        pipeline_name = pass_config['pipeline']
        sensitivity = pass_config.get('sensitivity', 'balanced')

        # Get pipeline class
        pipeline_class = PIPELINE_CLASSES.get(pipeline_name)
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Create temp directory for this pass
        pass_temp_dir = self.temp_dir / f"pass{pass_number}"
        pass_temp_dir.mkdir(parents=True, exist_ok=True)

        # Handle Transformers pipeline separately (uses hf_* params, not resolved_config)
        if pipeline_name == 'transformers':
            return self._create_transformers_pipeline(pass_config, pass_number, pass_temp_dir)

        # Legacy pipeline handling
        # Model compatibility validation (faster-whisper doesn't support turbo)
        PIPELINE_MODEL_COMPATIBILITY = {
            'balanced': ['large-v2', 'large-v3'],
            'faster': ['large-v2', 'large-v3'],
            'fast': ['large-v2', 'large-v3'],
            'fidelity': ['turbo', 'large-v2', 'large-v3'],
            'kotoba-faster-whisper': ['kotoba-tech/kotoba-whisper-v2.0-faster', 'RoachLin/kotoba-whisper-v2.2-faster']
        }

        # Check if custom params have an incompatible model
        if pass_config.get('params'):
            custom_model = pass_config['params'].get('model_name')
            if custom_model:
                allowed_models = PIPELINE_MODEL_COMPATIBILITY.get(pipeline_name, ['large-v2', 'large-v3'])
                if custom_model not in allowed_models:
                    raise ValueError(
                        f"Model '{custom_model}' is not compatible with pipeline '{pipeline_name}'. "
                        f"Allowed models: {allowed_models}"
                    )

        # Resolve configuration
        resolved_config = resolve_legacy_pipeline(
            pipeline_name=pipeline_name,
            sensitivity=sensitivity,
            task='transcribe',
            overrides=pass_config.get('overrides')
        )

        # Apply custom params if provided (M1: use shared helper)
        if pass_config.get('params'):
            self._apply_custom_params(resolved_config, pass_config['params'], pass_number)

        # Instantiate legacy pipeline
        pipeline = pipeline_class(
            output_dir=str(self.output_dir),
            temp_dir=str(pass_temp_dir),
            keep_temp_files=True,
            subs_language=self.subs_language,
            resolved_config=resolved_config,
            progress_display=self.progress_display,
            **self.extra_kwargs
        )

        return pipeline

    def _create_transformers_pipeline(
        self,
        pass_config: Dict[str, Any],
        pass_number: int,
        pass_temp_dir: Path
    ):
        """
        Create a TransformersPipeline instance.

        TransformersPipeline uses dedicated hf_* parameters instead of
        resolved_config. If hf_params are provided (customized), use them.
        Otherwise, use model defaults (minimal args pattern).

        Args:
            pass_config: Pass configuration with optional 'hf_params'
            pass_number: Pass number (1 or 2)
            pass_temp_dir: Temporary directory for this pass

        Returns:
            TransformersPipeline instance
        """
        # Default HF parameters (model uses its tuned defaults)
        hf_defaults = {
            'hf_model_id': 'kotoba-tech/kotoba-whisper-v2.2',
            'hf_chunk_length': 15,
            'hf_stride': None,
            'hf_batch_size': 16,
            'hf_scene': 'none',
            'hf_beam_size': 5,
            'hf_temperature': 0.0,
            'hf_attn': 'sdpa',
            'hf_timestamps': 'segment',
            'hf_language': 'ja',
            'hf_task': 'transcribe',
            'hf_device': 'auto',
            'hf_dtype': 'auto',
        }

        # If customized HF params provided, merge with defaults
        hf_params = pass_config.get('hf_params', {})
        if hf_params:
            logger.debug(f"Pass {pass_number}: Using customized Transformers parameters")
            # Map GUI param names to pipeline param names if needed
            param_mapping = {
                'model_id': 'hf_model_id',
                'chunk_length_s': 'hf_chunk_length',
                'stride_length_s': 'hf_stride',
                'batch_size': 'hf_batch_size',
                'scene': 'hf_scene',
                'beam_size': 'hf_beam_size',
                'temperature': 'hf_temperature',
                'attn_implementation': 'hf_attn',
                'timestamps': 'hf_timestamps',
                'language': 'hf_language',
                'device': 'hf_device',
                'dtype': 'hf_dtype',
            }
            for gui_key, hf_key in param_mapping.items():
                if gui_key in hf_params:
                    hf_defaults[hf_key] = hf_params[gui_key]
            # Also accept direct hf_* keys
            for key, value in hf_params.items():
                if key.startswith('hf_'):
                    hf_defaults[key] = value
        else:
            logger.debug(f"Pass {pass_number}: Using default Transformers parameters (minimal args)")

        # Create TransformersPipeline with HF parameters
        pipeline = TransformersPipeline(
            output_dir=str(self.output_dir),
            temp_dir=str(pass_temp_dir),
            keep_temp_files=True,
            progress_display=self.progress_display,
            subs_language=self.subs_language,
            **hf_defaults
        )

        return pipeline

    def _execute_pass(
        self,
        media_info: Dict,
        pass_config: Dict[str, Any],
        pass_number: int
    ) -> Dict:
        """
        Execute a single pass using the specified pipeline configuration.

        Args:
            media_info: Media file information
            pass_config: Pass configuration with either:
                - params: Full configuration snapshot (when customized)
                - pipeline + sensitivity: Resolve from presets (when default)
                - hf_params: HuggingFace params (for transformers pipeline)
            pass_number: Pass number (1 or 2)

        Returns:
            Pipeline processing result metadata
        """
        pipeline_name = pass_config['pipeline']

        # Create pass-specific temp directory to avoid conflicts
        pass_temp_dir = self.temp_dir / f"pass{pass_number}"
        pass_temp_dir.mkdir(parents=True, exist_ok=True)

        # Handle Transformers pipeline separately
        if pipeline_name == 'transformers':
            pipeline = self._create_transformers_pipeline(pass_config, pass_number, pass_temp_dir)
            result = pipeline.process(media_info)
            return result

        # Legacy pipeline handling
        sensitivity = pass_config.get('sensitivity', 'balanced')

        # Always resolve base config to get proper structure
        resolved_config = resolve_legacy_pipeline(
            pipeline_name=pipeline_name,
            sensitivity=sensitivity,
            task='transcribe',
            overrides=pass_config.get('overrides')
        )

        # If custom params provided, merge them into the resolved config (M1: use shared helper)
        if pass_config.get('params'):
            logger.debug(f"Pass {pass_number}: Applying custom parameters to {pipeline_name}/{sensitivity}")
            self._apply_custom_params(resolved_config, pass_config['params'], pass_number)
        else:
            logger.debug(f"Pass {pass_number}: Using defaults for {pipeline_name}/{sensitivity}")

        # Get pipeline class
        pipeline_class = PIPELINE_CLASSES.get(pipeline_name)
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Instantiate legacy pipeline
        pipeline = pipeline_class(
            output_dir=str(self.output_dir),
            temp_dir=str(pass_temp_dir),
            keep_temp_files=True,  # Keep until merge is done
            subs_language=self.subs_language,
            resolved_config=resolved_config,
            progress_display=self.progress_display,
            **self.extra_kwargs
        )

        # Process
        result = pipeline.process(media_info)
        return result

    def _create_ensemble_metadata(
        self,
        media_info: Dict,
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]],
        merge_strategy: str
    ) -> Dict:
        """Create initial ensemble metadata structure."""
        return {
            'metadata_type': 'ensemble',
            'created_at': datetime.now().isoformat() + 'Z',
            'input': {
                'file': str(media_info['path']),
                'basename': media_info['basename']
            },
            'configuration': {
                'pass1': pass1_config,
                'pass2': pass2_config,
                'merge_strategy': merge_strategy
            },
            'pass1': {'status': 'pending'},
            'pass2': {'status': 'pending' if pass2_config else 'disabled'},
            'merge': {'status': 'pending' if pass2_config else 'disabled'}
        }

    def _cleanup_intermediate(self, media_basename: str):
        """Clean up intermediate pass files."""
        try:
            # Clean pass-specific temp directories
            for pass_num in [1, 2]:
                pass_temp = self.temp_dir / f"pass{pass_num}"
                if pass_temp.exists():
                    import shutil
                    shutil.rmtree(pass_temp)
                    logger.debug(f"Cleaned up pass {pass_num} temp directory")

            # Optionally remove intermediate SRT files
            # Keep them for now for debugging - user can delete manually
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def get_mode_name(self) -> str:
        """Return the mode name for this orchestrator."""
        return "ensemble"
