"""Ensemble Orchestrator for two-pass pipeline processing."""

import json
import multiprocessing as mp
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperjav.utils.logger import logger
from whisperjav.utils.metadata_manager import MetadataManager
from whisperjav.utils.parameter_tracer import NullTracer

from .merge import MergeEngine
from .pass_worker import WorkerPayload, run_pass_worker
from .utils import resolve_language_code


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

        # Extract parameter tracer for orchestrator-level tracing
        # Note: Cannot pass to subprocesses (not picklable), so trace at orchestrator level only
        self.tracer = kwargs.get('parameter_tracer', NullTracer())

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
        pass_languages = {
            1: resolve_language_code(pass1_config, self.subs_language),
        }
        if pass2_config:
            pass_languages[2] = resolve_language_code(pass2_config, self.subs_language)

        # Trace ensemble configuration
        self.tracer.emit("ensemble_batch_start", {
            "files_count": len(serialized_media),
            "pass1_pipeline": pass1_config.get('pipeline'),
            "pass2_pipeline": pass2_config.get('pipeline') if pass2_config else None,
            "merge_strategy": merge_strategy,
            "subs_language": self.subs_language,
        })

        logger.info(
            "Starting batch ensemble processing for %s files (Pass 1: %s)",
            len(serialized_media),
            pass1_config['pipeline'],
        )

        # Trace pass 1 start
        self.tracer.emit("ensemble_pass_start", {
            "pass_number": 1,
            "pipeline": pass1_config.get('pipeline'),
            "config": pass1_config,
        })

        pass1_results = self._run_pass_in_subprocess(
            pass_number=1,
            media_files=serialized_media,
            pass_config=pass1_config,
            language_code=pass_languages[1],
        )

        # Trace pass 1 complete
        pass1_completed = sum(1 for r in pass1_results.values() if r.get('status') == 'completed')
        pass1_failed = sum(1 for r in pass1_results.values() if r.get('status') == 'failed')
        pass1_subs = sum(r.get('subtitles', 0) for r in pass1_results.values() if r.get('status') == 'completed')

        self.tracer.emit("ensemble_pass_complete", {
            "pass_number": 1,
            "results_count": len(pass1_results),
            "completed": pass1_completed,
            "failed": pass1_failed,
        })

        # Log pass 1 completion for user feedback
        logger.info(
            "Pass 1 completed: %d/%d file(s) successful, %d subtitles total",
            pass1_completed, len(pass1_results), pass1_subs
        )

        pass2_results: Dict[str, Dict[str, Any]] = {}
        if pass2_config:
            logger.info(
                "Pass 2 is enabled (%s). Launching worker...",
                pass2_config['pipeline'],
            )

            # Trace pass 2 start
            self.tracer.emit("ensemble_pass_start", {
                "pass_number": 2,
                "pipeline": pass2_config.get('pipeline'),
                "config": pass2_config,
            })

            pass2_results = self._run_pass_in_subprocess(
                pass_number=2,
                media_files=serialized_media,
                pass_config=pass2_config,
                language_code=pass_languages[2],
            )

            # Trace pass 2 complete
            pass2_completed = sum(1 for r in pass2_results.values() if r.get('status') == 'completed')
            pass2_failed = sum(1 for r in pass2_results.values() if r.get('status') == 'failed')
            pass2_subs = sum(r.get('subtitles', 0) for r in pass2_results.values() if r.get('status') == 'completed')

            self.tracer.emit("ensemble_pass_complete", {
                "pass_number": 2,
                "results_count": len(pass2_results),
                "completed": pass2_completed,
                "failed": pass2_failed,
            })

            # Log pass 2 completion for user feedback
            logger.info(
                "Pass 2 completed: %d/%d file(s) successful, %d subtitles total",
                pass2_completed, len(pass2_results), pass2_subs
            )

        all_metadata: List[Dict[str, Any]] = []
        table_rows: List[Dict[str, str]] = []

        for media_info in serialized_media:
            basename = media_info['basename']
            lang_code = pass_languages[1]
            ensemble_metadata = self._create_ensemble_metadata(
                media_info, pass1_config, pass2_config, merge_strategy
            )

            pass1 = pass1_results.get(basename)
            pass2 = pass2_results.get(basename) if pass2_config else None

            row_display = {
                'file': basename,
                'pass1': self._format_status(pass1),
                'pass2': self._format_status(pass2) if pass2_config else 'n/a',
                'merge': 'n/a' if not pass2_config else 'NOK',
                'final': f"{basename}: missing",
            }

            if not pass1 or pass1['status'] != 'completed':
                error_msg = pass1.get('error') if pass1 else 'Pass 1 worker returned no result'
                ensemble_metadata['pass1'] = {'status': 'failed', 'error': error_msg}
                if pass2_config:
                    ensemble_metadata['pass2'] = (
                        pass2 if pass2 else {'status': 'skipped', 'error': 'Pass 2 not executed'}
                    )
                else:
                    ensemble_metadata['pass2'] = {'status': 'skipped'}
                ensemble_metadata['status'] = 'failed'
                ensemble_metadata['merge'] = {'status': 'skipped'}
                all_metadata.append(ensemble_metadata)
                table_rows.append(row_display)
                continue

            pass1_srt = Path(pass1['srt_path']) if pass1.get('srt_path') else None

            # Defensive check: verify pass1 SRT file actually exists
            if pass1_srt and not pass1_srt.exists():
                logger.warning(
                    "Pass 1 reported completed but SRT file not found: %s",
                    pass1_srt,
                )
                pass1_srt = None  # Treat as if no SRT was produced

            ensemble_metadata['pass1'] = {
                'status': pass1['status'],
                'srt_path': str(pass1_srt) if pass1_srt else None,
                'subtitles': pass1.get('subtitles', 0),
                'processing_time': pass1.get('processing_time', 0.0),
            }

            final_output_path: Optional[Path] = None
            merge_info: Dict[str, Any] = {'status': 'skipped' if not pass2_config else 'pending'}

            if pass2_config:
                if pass2 and pass2['status'] == 'completed' and pass2.get('srt_path'):
                    pass2_srt = Path(pass2['srt_path'])

                    # Defensive check: verify pass2 SRT file actually exists
                    if not pass2_srt.exists():
                        logger.warning(
                            "Pass 2 reported completed but SRT file not found: %s",
                            pass2_srt,
                        )
                        # Treat pass2 as failed
                        ensemble_metadata['pass2'] = {
                            'status': 'failed',
                            'error': f'SRT file not found: {pass2_srt}',
                            'processing_time': pass2.get('processing_time', 0.0),
                        }
                        merge_info = {'status': 'skipped', 'reason': 'pass2_file_missing'}
                        row_display['merge'] = 'NOK'
                        row_display['pass2'] = 'NOK'
                        if pass1_srt:
                            row_display['final'] = f"fallback to pass1 ({pass1_srt.name})"
                    else:
                        # pass2_srt exists - record pass2 metadata
                        ensemble_metadata['pass2'] = {
                            'status': pass2['status'],
                            'srt_path': pass2['srt_path'],
                            'subtitles': pass2.get('subtitles', 0),
                            'processing_time': pass2.get('processing_time', 0.0),
                        }

                        # Check if pass1_srt is valid before attempting merge
                        if not pass1_srt:
                            logger.warning(
                                "Cannot merge: Pass 1 SRT file missing or invalid for %s",
                                basename,
                            )
                            merge_info = {'status': 'skipped', 'reason': 'pass1_file_missing'}
                            row_display['merge'] = 'NOK'
                            # Use pass2 as final output since pass1 is unavailable
                            final_output_path = pass2_srt
                            row_display['final'] = f"pass2 only ({pass2_srt.name})"
                        else:
                            # Both pass1 and pass2 SRT files exist - proceed with merge
                            tmp_merge = self.temp_dir / f"{basename}.merge.tmp.srt"
                            tmp_merge.parent.mkdir(parents=True, exist_ok=True)
                            final_candidate = self.output_dir / f"{basename}.{lang_code}.merged.whisperjav.srt"
                            try:
                                merge_stats = self.merge_engine.merge(
                                    srt1_path=pass1_srt,
                                    srt2_path=pass2_srt,
                                    output_path=tmp_merge,
                                    strategy=merge_strategy,
                                )
                                if final_candidate.exists():
                                    final_candidate.unlink()
                                shutil.move(tmp_merge, final_candidate)
                                merge_info = {
                                    'status': 'completed',
                                    'strategy': merge_strategy,
                                    'output_path': str(final_candidate),
                                    'statistics': merge_stats,
                                }
                                final_output_path = final_candidate
                                row_display['merge'] = 'OK'
                                row_display['final'] = final_candidate.name
                            except Exception as merge_error:
                                if tmp_merge.exists():
                                    tmp_merge.unlink()
                                logger.error("Merge failed for %s: %s", basename, merge_error)
                                merge_info = {'status': 'failed', 'error': str(merge_error)}
                                row_display['merge'] = 'NOK'
                                if pass1_srt:
                                    row_display['final'] = f"fallback to pass1 ({pass1_srt.name})"
                else:
                    ensemble_metadata['pass2'] = (
                        {'status': pass2['status'], 'error': pass2.get('error')}
                        if pass2
                        else {'status': 'failed', 'error': 'Pass 2 did not return a result'}
                    )
                    merge_info = {'status': 'skipped', 'reason': 'pass2_failed'}
                    row_display['merge'] = 'NOK'
                    row_display['pass2'] = self._format_status(ensemble_metadata['pass2'])
                    if pass1_srt:
                        row_display['final'] = f"fallback to pass1 ({pass1_srt.name})"
            else:
                ensemble_metadata['pass2'] = {'status': 'skipped'}

            if not pass2_config and pass1_srt:
                final_output_path = pass1_srt
                row_display['final'] = pass1_srt.name

            ensemble_metadata['merge'] = merge_info

            passes_completed = 2 if pass2_config and pass2 and pass2.get('status') == 'completed' else 1
            total_time = pass1.get('processing_time', 0.0) + (pass2.get('processing_time', 0.0) if pass2 else 0.0)
            ensemble_metadata['summary'] = {
                'final_output': str(final_output_path) if final_output_path else None,
                'passes_completed': passes_completed,
                'total_processing_time_seconds': total_time,
            }
            ensemble_metadata['output_files'] = {
                'final_srt': str(final_output_path) if final_output_path else None,
                'pass1_srt': pass1.get('srt_path'),
                'pass2_srt': pass2.get('srt_path') if pass2 else None,
            }

            ensemble_metadata.setdefault('status', 'completed')

            all_metadata.append(ensemble_metadata)
            table_rows.append(row_display)

        if not self.keep_temp_files:
            for media_info in serialized_media:
                self._cleanup_intermediate(media_info['basename'])

        self._print_summary_table(table_rows, time.time() - batch_start)
        summary_path = self._write_batch_summary(all_metadata)
        if summary_path:
            logger.info("Batch summary saved to %s", summary_path)

        # Trace batch completion
        batch_duration = time.time() - batch_start
        completed_count = sum(1 for m in all_metadata if m.get('status') == 'completed')
        self.tracer.emit("ensemble_batch_complete", {
            "files_processed": len(all_metadata),
            "completed": completed_count,
            "failed": len(all_metadata) - completed_count,
            "total_duration_seconds": round(batch_duration, 2),
            "summary_path": str(summary_path) if summary_path else None,
        })

        return all_metadata

    def _run_pass_in_subprocess(
        self,
        pass_number: int,
        media_files: List[Dict[str, Any]],
        pass_config: Dict[str, Any],
        language_code: str,
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
            language_code=language_code,
        )

        # Use 'spawn' start method for GPU compatibility across all platforms.
        # - Linux defaults to 'fork' which breaks CUDA/MPS in child processes
        #   (CUDA context cannot be re-initialized after fork)
        # - Windows and macOS ARM already use 'spawn' by default
        # - Explicitly setting 'spawn' ensures consistent, safe behavior everywhere
        # - Performance impact is negligible (<1% of total processing time)
        mp_context = mp.get_context('spawn')

        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=mp_context) as executor:
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
            ('Pass1', 'pass1'),
            ('Pass2', 'pass2'),
            ('Merge', 'merge'),
            ('Final', 'final'),
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

    def _write_batch_summary(self, metadata_list: List[Dict[str, Any]]) -> Optional[Path]:
        """Persist a lightweight JSON summary for GUI/CLI consumers."""
        if not metadata_list:
            return None

        summary = {
            'generated_at': datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            'files': [],
        }

        for item in metadata_list:
            summary['files'].append(
                {
                    'basename': item['input']['basename'],
                    'input_file': item['input']['file'],
                    'pass1': item.get('pass1'),
                    'pass2': item.get('pass2'),
                    'merge': item.get('merge'),
                    'final_output': item.get('summary', {}).get('final_output'),
                    'status': item.get('status', 'completed'),
                }
            )

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = self.output_dir / f"ensemble_summary_{timestamp}.json"
        try:
            summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        except OSError as exc:
            logger.warning("Failed to write batch summary: %s", exc)
            return None

        return summary_path

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
