"""Ensemble Orchestrator for two-pass pipeline processing."""

import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperjav.utils.logger import logger
from whisperjav.utils.metadata_manager import MetadataManager

from .merge import MergeEngine
from .pass_worker import WorkerPayload, run_pass_worker


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
        self.worker_kwargs = self._filter_picklable_kwargs(kwargs)

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
        """Process a single file by delegating to batch processing."""
        results = self.process_batch(
            media_files=[media_info],
            pass1_config=pass1_config,
            pass2_config=pass2_config,
            merge_strategy=merge_strategy,
        )
        if not results:
            raise RuntimeError("Ensemble processing produced no results")
        return results[0]

    def process_batch(
        self,
        media_files: List[Dict],
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]] = None,
        merge_strategy: str = 'smart_merge'
    ) -> List[Dict]:
        if not media_files:
            return []

        batch_start = time.time()
        serialized_media = self._serialize_media_files(media_files)

        logger.info(
            "Starting batch ensemble processing for %s files (Pass 1: %s)",
            len(serialized_media),
            pass1_config['pipeline'],
        )

        pass1_results = self._run_pass_in_subprocess(
            pass_number=1,
            media_files=serialized_media,
            pass_config=pass1_config,
        )

        pass2_results: Dict[str, Dict[str, Any]] = {}
        if pass2_config:
            logger.info(
                "Pass 2 is enabled (%s). Launching worker...",
                pass2_config['pipeline'],
            )
            pass2_results = self._run_pass_in_subprocess(
                pass_number=2,
                media_files=serialized_media,
                pass_config=pass2_config,
            )

        all_metadata: List[Dict[str, Any]] = []
        table_rows: List[Dict[str, str]] = []

        for media_info in serialized_media:
            basename = media_info['basename']
            ensemble_metadata = self._create_ensemble_metadata(
                media_info, pass1_config, pass2_config, merge_strategy
            )

            pass1 = pass1_results.get(basename)
            pass2 = pass2_results.get(basename) if pass2_config else None

            if not pass1 or pass1['status'] != 'completed':
                error_msg = pass1.get('error') if pass1 else 'Pass 1 worker returned no result'
                ensemble_metadata['pass1'] = {'status': 'failed', 'error': error_msg}
                ensemble_metadata['status'] = 'failed'
                table_rows.append(
                    {
                        'file': basename,
                        'pass1': self._format_status(pass1),
                        'pass2': self._format_status(pass2) if pass2_config else 'n/a',
                        'merge': 'n/a',
                        'output': 'n/a',
                    }
                )
                all_metadata.append(ensemble_metadata)
                continue

            pass1_srt = Path(pass1['srt_path']) if pass1.get('srt_path') else None
            ensemble_metadata['pass1'] = {
                'status': pass1['status'],
                'srt_path': pass1.get('srt_path'),
                'subtitles': pass1.get('subtitles', 0),
                'processing_time': pass1.get('processing_time', 0.0),
            }

            final_output = pass1_srt
            merge_info: Dict[str, Any] = {'status': 'skipped'}

            if pass2_config:
                if pass2 and pass2['status'] == 'completed' and pass2.get('srt_path'):
                    pass2_srt = Path(pass2['srt_path'])
                    ensemble_metadata['pass2'] = {
                        'status': pass2['status'],
                        'srt_path': pass2['srt_path'],
                        'subtitles': pass2.get('subtitles', 0),
                        'processing_time': pass2.get('processing_time', 0.0),
                    }

                    merged_srt = self.output_dir / f"{basename}.merged.srt"
                    try:
                        merge_stats = self.merge_engine.merge(
                            srt1_path=pass1_srt,
                            srt2_path=pass2_srt,
                            output_path=merged_srt,
                            strategy=merge_strategy,
                        )
                        merge_info = {
                            'status': 'completed',
                            'strategy': merge_strategy,
                            'output_path': str(merged_srt),
                            'statistics': merge_stats,
                        }
                        final_output = merged_srt
                    except Exception as merge_error:
                        logger.error("Merge failed for %s: %s", basename, merge_error)
                        merge_info = {'status': 'failed', 'error': str(merge_error)}
                else:
                    ensemble_metadata['pass2'] = (
                        {'status': pass2['status'], 'error': pass2.get('error')}
                        if pass2
                        else {'status': 'failed', 'error': 'Pass 2 did not return a result'}
                    )
            else:
                ensemble_metadata['pass2'] = {'status': 'skipped'}

            ensemble_metadata['merge'] = merge_info

            # Determine final naming
            lang_code = 'en' if self.subs_language == 'direct-to-english' else (
                pass1_config.get('params', {}).get('language')
                or pass1_config.get('overrides', {}).get('language')
                or 'ja'
            )

            final_srt_path = self.output_dir / f"{basename}.{lang_code}.whisperjav.srt"
            if final_output and final_output.exists() and final_output != final_srt_path:
                shutil.copy2(final_output, final_srt_path)

            ensemble_metadata['summary'] = {
                'final_output': str(final_srt_path),
                'passes_completed': 2
                if pass2_config and pass2 and pass2.get('status') == 'completed'
                else 1,
                'total_processing_time_seconds': pass1.get('processing_time', 0.0)
                + (pass2.get('processing_time', 0.0) if pass2 else 0.0),
            }
            ensemble_metadata['output_files'] = {
                'final_srt': str(final_srt_path),
                'pass1_srt': pass1.get('srt_path'),
                'pass2_srt': pass2.get('srt_path') if pass2 else None,
            }

            table_rows.append(
                {
                    'file': basename,
                    'pass1': self._format_status(pass1),
                    'pass2': self._format_status(pass2) if pass2_config else 'n/a',
                    'merge': self._format_status(merge_info),
                    'output': final_srt_path.name,
                }
            )

            all_metadata.append(ensemble_metadata)

        if not self.keep_temp_files:
            for media_info in serialized_media:
                self._cleanup_intermediate(media_info['basename'])

        self._print_summary_table(table_rows, time.time() - batch_start)
        return all_metadata

    def _run_pass_in_subprocess(
        self,
        pass_number: int,
        media_files: List[Dict[str, Any]],
        pass_config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Execute a pass inside an isolated worker process."""

        payload = WorkerPayload(
            pass_number=pass_number,
            media_files=media_files,
            pass_config=pass_config,
            output_dir=str(self.output_dir),
            temp_dir=str(self.temp_dir),
            keep_temp_files=self.keep_temp_files,
            subs_language=self.subs_language,
            extra_kwargs=self.worker_kwargs,
        )

        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_pass_worker, payload)
                worker_output = future.result()
        except Exception as exc:  # pragma: no cover - catastrophic worker failure
            logger.exception("Pass %s worker crashed", pass_number)
            return {
                info['basename']: {
                    'status': 'failed',
                    'error': f'Worker crash: {exc}',
                }
                for info in media_files
            }

        worker_error = worker_output.get('worker_error')
        if worker_error:
            logger.error("Pass %s worker reported error:\n%s", pass_number, worker_error)

        results = worker_output.get('results') or []
        if not results:
            fallback_error = worker_error or 'Worker produced no results'
            return {
                info['basename']: {
                    'status': 'failed',
                    'error': fallback_error,
                }
                for info in media_files
            }

        formatted: Dict[str, Dict[str, Any]] = {}
        for item in results:
            formatted[item['basename']] = item

        # Ensure every media file has an entry even if the worker skipped it
        for info in media_files:
            formatted.setdefault(
                info['basename'],
                {
                    'status': 'failed',
                    'error': 'Worker returned no result',
                },
            )

        return formatted

    def _serialize_media_files(self, media_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert media descriptors to picklable dicts before multiprocessing."""

        def normalize(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: normalize(v) for k, v in value.items()}
            if isinstance(value, list):
                return [normalize(v) for v in value]
            return value

        serialized = []
        for info in media_files:
            normalized = {k: normalize(v) for k, v in info.items()}
            if 'path' in normalized and normalized['path'] is not None:
                normalized['path'] = str(normalized['path'])
            serialized.append(normalized)
        return serialized

    def _format_status(self, result: Optional[Dict[str, Any]]) -> str:
        """Render status for summary table."""
        if not result:
            return 'n/a'

        status = result.get('status', 'unknown')
        if status == 'completed':
            return 'completed'

        error = result.get('error')
        if not error:
            return status

        first_line = str(error).strip().splitlines()[0]
        if len(first_line) > 40:
            first_line = first_line[:37] + '...'
        return f"{status}: {first_line}"

    def _print_summary_table(self, rows: List[Dict[str, str]], duration: float) -> None:
        """Pretty-print a compact summary table for the batch run."""
        if not rows:
            logger.info("No files processed (elapsed %.1fs)", duration)
            return

        columns = [
            ('File', 'file'),
            ('Pass1', 'pass1'),
            ('Pass2', 'pass2'),
            ('Merge', 'merge'),
            ('Output', 'output'),
        ]

        widths = {header: len(header) for header, _ in columns}
        for row in rows:
            for header, key in columns:
                widths[header] = max(widths[header], len(str(row.get(key, ''))))

        header_line = ' | '.join(header.ljust(widths[header]) for header, _ in columns)
        divider = '-+-'.join('-' * widths[header] for header, _ in columns)
        table_lines = [header_line, divider]
        for row in rows:
            table_lines.append(
                ' | '.join(str(row.get(key, '')).ljust(widths[header]) for header, key in columns)
            )

        table_output = '\n'.join(table_lines)
        logger.info("Ensemble summary (%.1fs)\n%s", duration, table_output)

    def _filter_picklable_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove objects that cannot be pickled before sending to subprocesses."""
        safe_kwargs = {}
        for key, value in kwargs.items():
            try:
                pickle.dumps(value)
            except Exception:
                logger.debug("Dropping non-picklable kwarg '%s' (%s)", key, type(value))
                continue
            safe_kwargs[key] = value
        return safe_kwargs

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
            'created_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
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
            # Clean pass-specific temp directories created by orchestrator and workers
            for suffix in ["pass1", "pass2", "pass1_worker", "pass2_worker"]:
                pass_temp = self.temp_dir / suffix
                if pass_temp.exists():
                    shutil.rmtree(pass_temp)
                    logger.debug(f"Cleaned up temp directory: {pass_temp.name}")

            # Optionally remove intermediate SRT files
            # Keep them for now for debugging - user can delete manually
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def get_mode_name(self) -> str:
        """Return the mode name for this orchestrator."""
        return "ensemble"
