"""Ensemble Orchestrator for two-pass pipeline processing."""

import json
import multiprocessing as mp
import pickle
import shutil
import time
# ProcessPoolExecutor removed - using raw mp.Process with Drop-Box pattern
# from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from whisperjav.modules.qwen_errors import NOT_STARTED, PREFIX as UNUSABLE_PREFIX
from whisperjav.utils.logger import logger, pass_log_file
from whisperjav.utils.metadata_manager import MetadataManager
from whisperjav.utils.parameter_tracer import NullTracer
from whisperjav.utils.output_naming import ensemble_srt_name

from .merge import MergeEngine
from .pass_worker import WorkerPayload, partial_result_file, run_pass_worker
from .utils import resolve_language_code


_KEPT_KEYS = ('savepoint_srt', 'savepoint_stage', 'savepoint_subtitles', 'kept_dir', 'save_points')


def _model_unusable(result: Optional[Dict[str, Any]]) -> bool:
    """A pass result that stopped because the pass's ASR model cannot be used (R-QFT, #494; modules/qwen_errors.py)."""
    return bool(result) and str(result.get('error') or '').startswith(UNUSABLE_PREFIX)


def _first_unusable(media_files: List[Dict[str, Any]], results: Dict[str, Dict[str, Any]]) -> Optional[int]:
    """The position of the first file whose pass stopped because its ASR model cannot be used, or None. The pass
    worker stops on that file and gives every later file of its list the same message without starting it."""
    return next((k for k, info in enumerate(media_files) if _model_unusable(results.get(info['basename']))), None)


def _not_started() -> Dict[str, Any]:
    """The result of a file no pass started after the stop (owner W501: shown as "skipped")."""
    return {'status': 'skipped', 'error': NOT_STARTED}


def _kept_work(result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a pass result that did not finish left behind (owner W462, W463 C1), as recorded by
    EnsembleOrchestrator._keep_failed_pass_work."""
    return {k: result[k] for k in _KEPT_KEYS if result and result.get(k) not in (None, "")}


# How far a failed pass's unfinished subtitles got, by the stage its pipeline recorded with the save-point
# (subtitle_pipeline/orchestrator.py, qwen_pipeline.py); owner W485: the plainer wording.
_STAGE_WORDS = {
    "after the ASR": "text only, timed by the VAD groups; the aligner did not run",
    "after the ASR, before the timing model": "timed by the VAD groups; the timing model did not run",
    "after the aligner, before the timing model": "timed by the aligner; the timing model did not run",
}


def _stage_words(stage: Optional[str]) -> str:
    """The user's words for a save-point stage; "" for a stage this table does not know."""
    stage = str(stage or "")
    if stage.startswith("before the clean-up"):
        return "timed, but not cleaned up"
    return _STAGE_WORDS.get(stage, "")


def _kept_work_sentences(pass_number: int, result: Optional[Dict[str, Any]]) -> List[str]:
    """What a pass that did not finish left behind, in plain words (owner W462, W463 C1: the user is told clearly;
    W485: the plainer wording). Only what is true: the unfinished subtitles with their count and how far they got,
    or, for a pipeline that writes save-points, that none were ready; the kept working folder. A pipeline without
    save-points gets no word about them."""
    kept = _kept_work(result)
    sentences = []
    if kept.get('savepoint_srt'):
        count = kept.get('savepoint_subtitles')
        counted = f"{count} subtitle{'' if count == 1 else 's'}" if isinstance(count, int) else ""
        details = ", ".join(part for part in (counted,
                                              _stage_words(kept.get('savepoint_stage'))) if part)
        sentences.append(f"pass {pass_number}'s unfinished subtitles are saved as {kept['savepoint_srt']}"
                         + (f" ({details})" if details else ""))
    elif kept.get('save_points'):
        sentences.append(f"pass {pass_number} stopped before any of its subtitles were ready, so none could be saved")
    if kept.get('kept_dir'):
        sentences.append(f"pass {pass_number}'s working files are kept in {kept['kept_dir']}")
    return sentences


class EnsembleOrchestrator:
    """Orchestrates two-pass ensemble processing with result merging."""

    def __init__(
        self,
        output_dir: str,
        temp_dir: str,
        keep_temp_files: bool = False,
        subs_language: str = 'native',
        progress_display=None,
        log_level: str = "INFO",
        serial_file_processing: bool = False,
        log_file: Optional[str] = None,
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
            log_level: Log level to propagate to subprocess workers
            serial_file_processing: If True, each file completes its full cycle
                (Pass 1 → Pass 2 → Merge) before the next file begins. Slower
                (reloads models per file) but delivers results incrementally.
            **kwargs: Additional parameters passed to pipelines
        """
        # "source" sentinel means each file's SRT goes next to its input file
        self.source_output_mode = (
            isinstance(output_dir, str) and output_dir.lower().strip() == "source"
        )
        if self.source_output_mode:
            # In source mode, there's no single output dir — per-file dirs are resolved later.
            # Use temp_dir as the fallback for metadata/summary files.
            self.output_dir = Path(temp_dir)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.temp_dir = Path(temp_dir)
        self.keep_temp_files = keep_temp_files
        self.subs_language = subs_language
        self.progress_display = progress_display
        self.log_level = log_level
        self.log_file = log_file
        self.serial_file_processing = serial_file_processing
        self.extra_kwargs = kwargs
        self.worker_kwargs = self._filter_picklable_kwargs(kwargs)

        self.temp_dir.mkdir(parents=True, exist_ok=True)
        # Pass folders kept in place because they could not be moved to kept/ (owner W462, W463 C1).
        self._kept_in_place: set = set()

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
        merge_strategy: str = 'pass1_primary'
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
        merge_strategy: str = 'pass1_primary'
    ) -> List[Dict]:
        if not media_files:
            return []

        # Serial mode: each file completes Pass1→Pass2→Merge before the next starts.
        # Single-file batches always take the batch path (identical result, no overhead).
        if self.serial_file_processing and len(media_files) > 1:
            return self._process_batch_serial(
                media_files, pass1_config, pass2_config, merge_strategy,
            )

        batch_start = time.time()
        serialized_media = self._serialize_media_files(media_files)

        # Annotate each file with its resolved output directory.
        # In source mode, each file's SRT goes next to its input file.
        for info in serialized_media:
            info['output_dir'] = str(self._resolve_file_output_dir(info))

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

        logger.debug(
            "Starting Pass 1 for %d files with pipeline=%s, sensitivity=%s",
            len(serialized_media),
            pass1_config.get('pipeline'), pass1_config.get('sensitivity')
        )

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

        logger.debug(
            "Pass 1 results: %d completed, %d failed, basenames=%s",
            pass1_completed, pass1_failed, list(pass1_results.keys())
        )

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

        # The Ensemble stop (owner W501): once pass 1 stops because its ASR model cannot be used, pass 2 is skipped
        # for that file and no further file starts; the later files show as "skipped" (not started).
        pass2_media = serialized_media
        stop_at = _first_unusable(serialized_media, pass1_results)
        if stop_at is not None:
            stopped = serialized_media[stop_at]['basename']
            for info in serialized_media[stop_at + 1:]:
                pass1_results[info['basename']] = _not_started()
            pass2_media = serialized_media[:stop_at]
            later = len(serialized_media) - stop_at - 1
            logger.warning(
                "Pass 1 stopped on %s: its ASR model cannot be used.%s%s", stopped,
                " Pass 2 is skipped for it." if pass2_config else "",
                f" The {later} file(s) after it are not started." if later else "")

        pass2_results: Dict[str, Dict[str, Any]] = {}
        if pass2_config and pass2_media:
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

            logger.debug(
                "Starting Pass 2 for %d files with pipeline=%s, sensitivity=%s",
                len(pass2_media),
                pass2_config.get('pipeline'), pass2_config.get('sensitivity')
            )

            pass2_results = self._run_pass_in_subprocess(
                pass_number=2,
                media_files=pass2_media,
                pass_config=pass2_config,
                language_code=pass_languages[2],
            )

            # Trace pass 2 complete
            pass2_completed = sum(1 for r in pass2_results.values() if r.get('status') == 'completed')
            pass2_failed = sum(1 for r in pass2_results.values() if r.get('status') == 'failed')
            pass2_subs = sum(r.get('subtitles', 0) for r in pass2_results.values() if r.get('status') == 'completed')

            logger.debug(
                "Pass 2 results: %d completed, %d failed, basenames=%s",
                pass2_completed, pass2_failed, list(pass2_results.keys())
            )

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
            metadata, row = self._process_single_file_merge(
                media_info, pass1_results, pass2_results,
                pass1_config, pass2_config, merge_strategy, pass_languages,
            )
            all_metadata.append(metadata)
            table_rows.append(row)

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
        degraded_count = sum(1 for m in all_metadata if m.get('status') == 'degraded')
        failed_count = sum(1 for m in all_metadata if m.get('status') == 'failed')
        self.tracer.emit("ensemble_batch_complete", {
            "files_processed": len(all_metadata),
            "completed": completed_count,
            "degraded": degraded_count,
            "failed": failed_count,
            "skipped": sum(1 for m in all_metadata if m.get('status') == 'skipped'),
            "total_duration_seconds": round(batch_duration, 2),
            "summary_path": str(summary_path) if summary_path else None,
        })

        return all_metadata

    def _process_single_file_merge(
        self,
        media_info: Dict[str, Any],
        pass1_results: Dict[str, Dict],
        pass2_results: Dict[str, Dict],
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]],
        merge_strategy: str,
        pass_languages: Dict[int, str],
    ) -> tuple:
        """Merge results for a single file and produce metadata + display row.

        This is the shared merge/metadata logic used by both batch and serial modes.

        Returns:
            (ensemble_metadata, row_display) tuple.
        """
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

        if pass1 and pass1.get('status') == 'skipped':
            # Not started (owner W501): an earlier file's pass stopped because its ASR model cannot be used.
            ensemble_metadata['pass1'] = dict(pass1)
            ensemble_metadata['pass2'] = {'status': 'skipped'}
            ensemble_metadata['merge'] = {'status': 'skipped'}
            ensemble_metadata['status'] = 'skipped'
            ensemble_metadata['error'] = pass1.get('error') or NOT_STARTED
            row_display['pass2'] = 'skipped' if pass2_config else 'n/a'
            row_display['merge'] = 'n/a'
            row_display['final'] = f"{basename}: not started"
            return ensemble_metadata, row_display

        if not pass1 or pass1['status'] != 'completed':
            error_msg = pass1.get('error') if pass1 else 'Pass 1 worker returned no result'
            ensemble_metadata['pass1'] = {'status': 'failed', 'error': error_msg,
                                          **_kept_work(pass1)}
            # What the passes that did not finish left behind (owner W462, W463 C1), for main.py's line.
            ensemble_metadata['kept_work'] = (
                _kept_work_sentences(1, pass1)
                + (_kept_work_sentences(2, pass2) if pass2_config and pass2 and pass2.get('status') != 'completed'
                   else []))
            if pass2_config:
                ensemble_metadata['pass2'] = (
                    pass2 if pass2 else {'status': 'skipped', 'error': 'Pass 2 not executed'}
                )
                if not pass2:
                    row_display['pass2'] = 'skipped'
            else:
                ensemble_metadata['pass2'] = {'status': 'skipped'}
            ensemble_metadata['status'] = 'failed'
            ensemble_metadata['merge'] = {'status': 'skipped'}
            return ensemble_metadata, row_display

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
                    ensemble_metadata['status'] = 'degraded'
                    if pass1_srt:
                        final_output_path = pass1_srt
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
                        file_output_dir = Path(media_info.get('output_dir', str(self.output_dir)))
                        final_candidate = file_output_dir / ensemble_srt_name(basename, lang_code)
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
                            ensemble_metadata['status'] = 'degraded'
                            if pass1_srt:
                                final_output_path = pass1_srt
                                row_display['final'] = f"fallback to pass1 ({pass1_srt.name})"
            else:
                ensemble_metadata['pass2'] = (
                    {'status': pass2['status'], 'error': pass2.get('error'), **_kept_work(pass2)}
                    if pass2
                    else {'status': 'failed', 'error': 'Pass 2 did not return a result'}
                )
                merge_info = {'status': 'skipped', 'reason': 'pass2_failed'}
                row_display['merge'] = 'NOK'
                row_display['pass2'] = self._format_status(ensemble_metadata['pass2'])
                ensemble_metadata['status'] = 'degraded'
                if pass1_srt:
                    final_output_path = pass1_srt
                    row_display['final'] = f"fallback to pass1 ({pass1_srt.name})"
        else:
            ensemble_metadata['pass2'] = {'status': 'skipped'}

        if not pass2_config and pass1_srt:
            final_output_path = pass1_srt
            row_display['final'] = pass1_srt.name

        ensemble_metadata['merge'] = merge_info

        passes_completed = 2 if pass2_config and pass2 and pass2.get('status') == 'completed' else 1
        total_time = pass1.get('processing_time', 0.0) + (pass2.get('processing_time', 0.0) if pass2 else 0.0)

        # Anything that quietly fell short, carried back from the worker
        # processes so the run summary can name it (agreed error-handling table,
        # 2026-09-17, cross-cutting rule 1). Labelled by pass, because "2 of 40
        # scenes" means something different depending on which pass produced the
        # audio the user is reading. Only a pass whose output is actually used
        # is reported: a pass 2 that failed contributed nothing to complain
        # about, and the file is already marked degraded for that.
        degradations: List[str] = []
        for _pass_number, _pass in ((1, pass1), (2, pass2)):
            if not _pass or _pass.get('status') != 'completed':
                continue
            for _note in _pass.get('degradations') or []:
                degradations.append(f"pass {_pass_number}: {_note}")

        # Routine notes (owner Q21, CrispASR master plan 2.6): printed in the run
        # summary, and they change nothing -- no state, no exit status. From the same
        # passes as the degradations above, but not labelled by pass: each note names
        # its pipeline itself where that matters, and a note both passes make (the
        # CrispASR program running on the CPU build, say) is printed once, in the
        # order the passes made them.
        notes: List[str] = []
        for _pass in (pass1, pass2):
            if not _pass or _pass.get('status') != 'completed':
                continue
            for _note in _pass.get('notes') or []:
                if _note not in notes:
                    notes.append(_note)

        ensemble_metadata['summary'] = {
            'final_output': str(final_output_path) if final_output_path else None,
            'passes_completed': passes_completed,
            'total_processing_time_seconds': total_time,
            'degradations': degradations,
            'notes': notes,
        }
        # Owner W462, W463 C1: what a pass that did not finish left behind, in plain words for the run summary.
        ensemble_metadata['kept_work'] = (_kept_work_sentences(2, pass2)
                                          if pass2_config and pass2 and pass2.get('status') != 'completed' else [])
        ensemble_metadata['output_files'] = {
            'final_srt': str(final_output_path) if final_output_path else None,
            'pass1_srt': pass1.get('srt_path'),
            'pass2_srt': pass2.get('srt_path') if pass2 else None,
        }

        ensemble_metadata.setdefault('status', 'completed')

        return ensemble_metadata, row_display

    def _process_batch_serial(
        self,
        media_files: List[Dict],
        pass1_config: Dict[str, Any],
        pass2_config: Optional[Dict[str, Any]],
        merge_strategy: str,
    ) -> List[Dict]:
        """Process files serially: each file completes Pass1→Pass2→Merge before the next starts.

        Slower than batch mode (reloads models per file) but delivers results incrementally —
        each file's final SRT is on disk and usable before the next file begins processing.
        """
        batch_start = time.time()
        serialized_media = self._serialize_media_files(media_files)

        for info in serialized_media:
            info['output_dir'] = str(self._resolve_file_output_dir(info))

        pass_languages = {
            1: resolve_language_code(pass1_config, self.subs_language),
        }
        if pass2_config:
            pass_languages[2] = resolve_language_code(pass2_config, self.subs_language)

        # Trace serial ensemble start
        self.tracer.emit("ensemble_batch_start", {
            "files_count": len(serialized_media),
            "pass1_pipeline": pass1_config.get('pipeline'),
            "pass2_pipeline": pass2_config.get('pipeline') if pass2_config else None,
            "merge_strategy": merge_strategy,
            "subs_language": self.subs_language,
            "serial_mode": True,
        })

        total_files = len(serialized_media)
        logger.info(
            "Starting SERIAL ensemble: %d files (each file completes fully before next)",
            total_files,
        )

        all_metadata: List[Dict[str, Any]] = []
        table_rows: List[Dict[str, str]] = []
        # The Ensemble stop (owner W501): once pass 1 stops because its ASR model cannot be used, no further file
        # starts; the later files show as "skipped" (not started).
        run_stopped = False
        # Once pass 2 stops because its ASR model cannot be used (owner W505 1): the later files get pass 1 alone,
        # pass 2 is not tried again, and each is marked as with the box unticked -- pass 2 failed with this reason,
        # output pass 1 alone (state "suspect"; the exit status is not changed by it).
        pass2_stop: Optional[str] = None

        for file_idx, media_info in enumerate(serialized_media, 1):
            basename = media_info['basename']
            file_start = time.time()

            if run_stopped:
                metadata, row = self._process_single_file_merge(
                    media_info, {basename: _not_started()}, {},
                    pass1_config, pass2_config, merge_strategy, pass_languages,
                )
                all_metadata.append(metadata)
                table_rows.append(row)
                logger.warning("[%d/%d] NOT STARTED: %s — the run stopped because its ASR model cannot be used",
                               file_idx, total_files, basename)
                continue

            # --- Pass 1 ---
            logger.info(
                "[%d/%d] %s — Pass 1 (%s)...",
                file_idx, total_files, basename, pass1_config['pipeline'],
            )
            pass1_results = self._run_pass_in_subprocess(
                pass_number=1,
                media_files=[media_info],
                pass_config=pass1_config,
                language_code=pass_languages[1],
            )
            pass1_stopped = _model_unusable(pass1_results.get(basename))

            # --- Pass 2 (if enabled; skipped when pass 1's ASR model cannot be used, owner W501) ---
            pass2_results: Dict[str, Dict[str, Any]] = {}
            pass2_ran = False
            if pass2_config and pass1_stopped:
                logger.warning("[%d/%d] %s — Pass 2 skipped: pass 1's ASR model cannot be used",
                               file_idx, total_files, basename)
            elif pass2_config and pass2_stop is not None:
                # The same result the pass worker gives the files after its stop in the normal mode.
                logger.warning("[%d/%d] %s — Pass 2 not run: its ASR model cannot be used; output is pass 1 alone",
                               file_idx, total_files, basename)
                pass2_results = {basename: {'status': 'failed', 'error': pass2_stop}}
            elif pass2_config:
                pass2_ran = True
                logger.info(
                    "[%d/%d] %s — Pass 2 (%s)...",
                    file_idx, total_files, basename, pass2_config['pipeline'],
                )
                pass2_results = self._run_pass_in_subprocess(
                    pass_number=2,
                    media_files=[media_info],
                    pass_config=pass2_config,
                    language_code=pass_languages[2],
                )

            # --- Merge + metadata (reuses shared method) ---
            if pass2_ran:
                logger.info(
                    "[%d/%d] %s — Merge (%s)...",
                    file_idx, total_files, basename, merge_strategy,
                )
            metadata, row = self._process_single_file_merge(
                media_info, pass1_results, pass2_results,
                pass1_config, pass2_config, merge_strategy, pass_languages,
            )
            all_metadata.append(metadata)
            table_rows.append(row)
            if pass1_stopped:
                run_stopped = True
                if file_idx < total_files:
                    logger.warning("Pass 1 stopped on %s: its ASR model cannot be used. The %d file(s) after it "
                                   "are not started.", basename, total_files - file_idx)
            elif pass2_ran and _model_unusable(pass2_results.get(basename)):
                pass2_stop = str(pass2_results[basename].get('error') or '')
                if file_idx < total_files:
                    logger.warning("Pass 2 stopped on %s: its ASR model cannot be used. The %d file(s) after it "
                                   "get pass 1 alone.", basename, total_files - file_idx)

            # --- Per-file cleanup ---
            if not self.keep_temp_files:
                self._cleanup_intermediate(basename)

            # --- Per-file completion banner ---
            file_elapsed = time.time() - file_start
            file_status = metadata.get('status', 'unknown')
            final_output = metadata.get('summary', {}).get('final_output')
            if file_status == 'failed' or not final_output:
                logger.warning(
                    "[%d/%d] FAILED: %s (%.0fs)",
                    file_idx, total_files, basename, file_elapsed,
                )
            elif file_status == 'degraded':
                final_name = Path(final_output).name
                subs_count = metadata.get('pass1', {}).get('subtitles', '?')
                logger.warning(
                    "[%d/%d] PARTIAL: %s -> %s (%s subs, %.0fs) — pass 2 failed, fell back to pass 1",
                    file_idx, total_files, basename, final_name,
                    subs_count, file_elapsed,
                )
            else:
                final_name = Path(final_output).name
                subs_count = metadata.get('pass1', {}).get('subtitles', '?')
                logger.info(
                    "[%d/%d] DONE: %s -> %s (%s subs, %.0fs)",
                    file_idx, total_files, basename, final_name,
                    subs_count, file_elapsed,
                )

        # --- Batch summary (same as batch mode) ---
        self._print_summary_table(table_rows, time.time() - batch_start)
        summary_path = self._write_batch_summary(all_metadata)
        if summary_path:
            logger.info("Batch summary saved to %s", summary_path)

        batch_duration = time.time() - batch_start
        completed_count = sum(1 for m in all_metadata if m.get('status') == 'completed')
        degraded_count = sum(1 for m in all_metadata if m.get('status') == 'degraded')
        failed_count = sum(1 for m in all_metadata if m.get('status') == 'failed')
        self.tracer.emit("ensemble_batch_complete", {
            "files_processed": len(all_metadata),
            "completed": completed_count,
            "degraded": degraded_count,
            "failed": failed_count,
            "skipped": sum(1 for m in all_metadata if m.get('status') == 'skipped'),
            "total_duration_seconds": round(batch_duration, 2),
            "summary_path": str(summary_path) if summary_path else None,
            "serial_mode": True,
        })

        return all_metadata

    def _run_pass_in_subprocess(
        self,
        pass_number: int,
        media_files: List[Dict[str, Any]],
        pass_config: Dict[str, Any],
        language_code: str,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Execute a pass inside an isolated worker process.

        Uses the "Drop-Box + Nuclear Exit" pattern:
        - Worker writes results to a pickle file (Drop-Box)
        - Worker exits via os._exit(0) to skip Python shutdown
        - Parent reads results from Drop-Box file

        This prevents ctranslate2 C++ destructor crashes on Windows.
        """
        import pickle

        # Get trace file path if tracer is active (ParameterTracer has output_path, NullTracer doesn't)
        trace_file_path = str(self.tracer.output_path) if hasattr(self.tracer, 'output_path') else None

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
            log_level=self.log_level,
            trace_file_path=trace_file_path,
            log_file=pass_log_file(self.log_file, pass_number),
        )

        # Setup Drop-Box path (unique per pass)
        result_file = str(self.temp_dir / f"worker_result_pass{pass_number}.pkl")
        partial_file = Path(partial_result_file(result_file))
        partial_file.unlink(missing_ok=True)  # a stale one from an earlier run must not count
        # No earlier save-point may pass for this run's (owner W462, W463 C1); the pipeline clears its own too.
        for stale in (self.temp_dir / f"pass{pass_number}_worker").glob(f"*{self.SAVE_POINT_SUFFIX}"):
            stale.unlink(missing_ok=True)
        for stale in (self.temp_dir / f"pass{pass_number}_worker").glob("*.savepoint.json"):
            stale.unlink(missing_ok=True)

        # Use 'spawn' start method for GPU compatibility across all platforms.
        # - Linux defaults to 'fork' which breaks CUDA/MPS in child processes
        #   (CUDA context cannot be re-initialized after fork)
        # - Windows and macOS ARM already use 'spawn' by default
        # - Explicitly setting 'spawn' ensures consistent, safe behavior everywhere
        # - Performance impact is negligible (<1% of total processing time)
        mp_context = mp.get_context('spawn')

        # Spawn "One-Shot" Process (not using ProcessPoolExecutor)
        # This gives us precise control over the lifecycle and avoids
        # pool poisoning when the worker calls os._exit(0)
        p = mp_context.Process(target=run_pass_worker, args=(payload, result_file))
        p.start()

        # Wait for process to die (Nuclear Exit or crash)
        p.join()

        # Analyze the aftermath
        if p.exitcode != 0:
            logger.error(
                "Pass %s worker died with exit code %s",
                pass_number, p.exitcode
            )

        # Read result from Drop-Box
        worker_output = None
        if Path(result_file).exists():
            try:
                with open(result_file, 'rb') as f:
                    worker_output = pickle.load(f)
                # Clean up Drop-Box file
                Path(result_file).unlink()
            except Exception as e:
                logger.error("Pass %s: Failed to read Drop-Box: %s", pass_number, e)

        # Handle missing or invalid Drop-Box
        if worker_output is None:
            logger.error("Pass %s: Worker produced no Drop-Box (crashed before writing?)", pass_number)
            # Step 3 (owner W462, W463): the files the worker finished before it stopped keep their results.
            done: Dict[str, Dict[str, Any]] = {}
            if partial_file.exists():
                try:
                    with open(partial_file, 'rb') as f:
                        done = {r['basename']: r for r in (pickle.load(f).get('results') or [])}
                except Exception as e:
                    logger.error("Pass %s: Failed to read the per-file results: %s", pass_number, e)
            formatted_crash = {
                info['basename']: done.get(info['basename']) or {
                    'status': 'failed',
                    'error': f'Worker crash: exit code {p.exitcode}, no result file',
                }
                for info in media_files
            }
            partial_file.unlink(missing_ok=True)
            if done:
                logger.warning("Pass %s: the worker stopped after %d of %d file(s); their results are kept",
                               pass_number, len(done), len(media_files))
            self._keep_failed_pass_work(pass_number, media_files, formatted_crash, language_code,
                                        pass_config.get('pipeline'))
            return formatted_crash
        partial_file.unlink(missing_ok=True)

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
            basename = info['basename']
            if basename not in formatted:
                # Log when a file has no matching result - helps diagnose Unicode/encoding issues
                logger.warning(
                    "Pass %s: No result for basename (len=%d chars, %d UTF-8 bytes). "
                    "This may indicate a Unicode encoding mismatch between processes.",
                    pass_number,
                    len(basename),
                    len(basename.encode('utf-8')),
                )
                formatted[basename] = {
                    'status': 'failed',
                    'error': 'Worker returned no result for this file',
                }

        self._keep_failed_pass_work(pass_number, media_files, formatted, language_code, pass_config.get('pipeline'))
        return formatted

    # Owner W462, W463 (step 1 and C1): a pass that did not finish a file keeps what it made.
    SAVE_POINT_SUFFIX = ".savepoint.srt"   # what the pipeline writes in its temp folder (qwen_pipeline.py)
    UNFINISHED_SUFFIX = ".unfinished.srt"  # the copy beside the video: <basename>.<lang>.pass<N>.unfinished.srt (W485)
    KEPT_DIR = "kept"
    # A kept folder keeps the subtitles, records and logs, not the audio (owner W485): the extracted audio and the scene
    # files are about 0.5 GB for a 2-hour film and can be made again. WhisperJAV's own working files only.
    KEPT_AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".opus")

    # The pipelines that write save-points (qwen_pipeline.py; ChronosJAV's three models all run as "qwen").
    SAVE_POINT_PIPELINES = ("qwen",)

    def _keep_failed_pass_work(self, pass_number: int, media_files: List[Dict[str, Any]],
                               results: Dict[str, Dict[str, Any]], language_code: str,
                               pipeline: Optional[str] = None) -> None:
        """For each file this pass did not finish: the pass's working folder is kept under
        ``<temp>/kept/<time>_pass<N>`` (the next pass or file would reuse its name), without its audio (W485), then the
        file's last save-point, if the pipeline wrote one, is copied beside the video as
        ``<basename>.<lang>.pass<N>.unfinished.srt``. Both are named in the file's result (``savepoint_srt``,
        ``savepoint_stage``, ``kept_dir``, ``save_points``) for the run summary, and in a warning now."""
        failed = [info for info in media_files
                  if (results.get(info['basename']) or {}).get('status') != 'completed']
        # After a stop because the pass's ASR model cannot be used, the worker started no later file of its list
        # (owner W501): those have no work to keep and get no "stopped before any subtitles" warning.
        stop_at = _first_unusable(media_files, results)
        if stop_at is not None:
            never_started = {info['basename'] for info in media_files[stop_at + 1:]}
            failed = [info for info in failed if info['basename'] not in never_started]
        if not failed:
            return
        work_dir = self.temp_dir / f"pass{pass_number}_worker"
        kept = work_dir
        if work_dir.is_dir():
            target_dir = self.temp_dir / self.KEPT_DIR / f"{time.strftime('%Y%m%d_%H%M%S')}_pass{pass_number}"
            try:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(work_dir), str(target_dir))
                kept = target_dir
            except OSError as e:
                logger.error("Pass %s: its working files could not be moved (%s); they stay in %s",
                             pass_number, e, work_dir)
                self.__dict__.setdefault("_kept_in_place", set()).add(work_dir.name)  # cleanup leaves it
            self._drop_kept_audio(pass_number, kept)
            logger.warning("Pass %s: its working files are kept in %s", pass_number, kept)
        writes_save_points = str(pipeline or "") in self.SAVE_POINT_PIPELINES
        for info in failed:
            basename = info['basename']
            entry = results.setdefault(basename, {'status': 'failed'})
            entry['save_points'] = writes_save_points
            if kept.is_dir():
                entry['kept_dir'] = str(kept)
            saved = kept / f"{basename}{self.SAVE_POINT_SUFFIX}"
            if not saved.is_file():
                if writes_save_points:
                    logger.warning("Pass %s stopped on %s before any of its subtitles were ready, so none could be "
                                   "saved.", pass_number, basename)
                continue
            stage, count = "", None
            try:
                meta = json.loads((kept / f"{basename}.savepoint.json").read_text(encoding="utf-8"))
                stage, count = str(meta.get("stage") or ""), meta.get("subtitles")
            except (OSError, ValueError):
                pass
            entry['savepoint_stage'] = stage
            entry['savepoint_subtitles'] = count
            target = (self._resolve_file_output_dir(info)
                      / f"{basename}.{language_code}.pass{pass_number}{self.UNFINISHED_SUFFIX}")
            try:
                shutil.copy2(saved, target)
                entry['savepoint_srt'] = str(target)
            except OSError as e:
                logger.error("Pass %s: the unfinished subtitles of %s could not be copied beside the video (%s)",
                             pass_number, basename, e)
                entry['savepoint_srt'] = str(saved)   # inside the kept folder, which stays
            words = _stage_words(stage)
            logger.warning("Pass %s did not finish %s. Its unfinished subtitles are saved as %s%s",
                           pass_number, basename, entry['savepoint_srt'], f" ({words})" if words else "")

    def _drop_kept_audio(self, pass_number: int, folder: Path) -> None:
        """Owner W485: a kept folder keeps the subtitles, records and logs and drops the audio files (WhisperJAV's own
        extracted audio and scene files in the pass's working folder; never a user's media). A file that cannot be
        removed stays; nothing here stops the run."""
        if not folder.is_dir():
            return
        removed, freed = 0, 0
        for path in folder.rglob("*"):
            if path.is_file() and path.suffix.lower() in self.KEPT_AUDIO_SUFFIXES:
                try:
                    size = path.stat().st_size
                    path.unlink()
                    removed, freed = removed + 1, freed + size
                except OSError as e:
                    logger.debug("Pass %s: kept audio file %s not removed: %s", pass_number, path, e)
        if removed:
            logger.info("Pass %s: %d audio file(s) left out of the kept working files (%.0f MB)",
                        pass_number, removed, freed / 2**20)

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

    def _resolve_file_output_dir(self, media_info: Dict[str, Any]) -> Path:
        """Return the output directory for a specific file.

        In source mode, SRTs go next to each input file.
        Otherwise, all SRTs go to the single configured output_dir.
        """
        if self.source_output_mode:
            return Path(media_info['path']).parent
        return self.output_dir

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
        summary_path = self.temp_dir / f"ensemble_summary_{timestamp}.json"
        try:
            summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
        except OSError as exc:
            logger.warning("Failed to write batch summary: %s", exc)
            return None

        return summary_path

    def _filter_picklable_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Remove objects that cannot be pickled before sending to subprocesses.

        Also excludes 'parameter_tracer' which is handled separately via
        trace_file_path in the WorkerPayload. The worker creates its own
        tracer from the file path, so passing the tracer object would cause
        'got multiple values for keyword argument' errors.
        """
        # Keys to always exclude (handled separately or cause duplicate kwarg errors)
        excluded_keys = {'parameter_tracer', 'progress_display'}

        safe_kwargs = {}
        for key, value in kwargs.items():
            if key in excluded_keys:
                logger.debug("Excluding kwarg '%s' from worker (handled separately)", key)
                continue
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
                if suffix in getattr(self, "_kept_in_place", ()):
                    continue  # a failed pass's work that could not be moved to kept/
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
