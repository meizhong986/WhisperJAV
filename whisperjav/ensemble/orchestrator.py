"""Ensemble Orchestrator for two-pass pipeline processing."""

import time
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from whisperjav.config.legacy import resolve_legacy_pipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
from whisperjav.utils.logger import logger
from whisperjav.utils.metadata_manager import MetadataManager

from .merge import MergeEngine


# Pipeline class mapping
PIPELINE_CLASSES = {
    'balanced': BalancedPipeline,
    'fast': FastPipeline,
    'faster': FasterPipeline,
    'fidelity': FidelityPipeline,
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
        merge_strategy: str = 'confidence'
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
            merge_strategy: Merge strategy name ('confidence', 'union', 'intersection', 'timing')

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
            pass_number: Pass number (1 or 2)

        Returns:
            Pipeline processing result metadata
        """
        pipeline_name = pass_config['pipeline']

        # Check if full params provided (customized mode)
        if pass_config.get('params'):
            # Use full configuration snapshot directly
            resolved_config = pass_config['params']
            logger.debug(f"Pass {pass_number}: Using custom configuration snapshot")
        else:
            # Resolve from pipeline + sensitivity presets (default mode)
            sensitivity = pass_config.get('sensitivity', 'balanced')
            overrides = pass_config.get('overrides')

            resolved_config = resolve_legacy_pipeline(
                pipeline_name=pipeline_name,
                sensitivity=sensitivity,
                task='transcribe',
                overrides=overrides
            )
            logger.debug(f"Pass {pass_number}: Resolved config from {pipeline_name}/{sensitivity}")

        # Get pipeline class
        pipeline_class = PIPELINE_CLASSES.get(pipeline_name)
        if not pipeline_class:
            raise ValueError(f"Unknown pipeline: {pipeline_name}")

        # Create pass-specific temp directory to avoid conflicts
        pass_temp_dir = self.temp_dir / f"pass{pass_number}"
        pass_temp_dir.mkdir(parents=True, exist_ok=True)

        # Instantiate pipeline
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
