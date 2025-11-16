#!/usr/bin/env python3
"""Asynchronous pipeline processor for WhisperJAV.

This module implements background processing using ThreadPoolExecutor
to prevent UI freezing and enable responsive applications.
"""

import os
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from queue import Queue, Empty
import traceback
from enum import Enum

from whisperjav.utils.logger import logger
from whisperjav.utils.progress_aggregator import (
    AsyncProgressReporter, VerbosityLevel, create_progress_handler
)


class ProcessingStatus(Enum):
    """Status of async processing task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProcessingTask:
    """Represents a single processing task."""
    task_id: str
    media_info: Dict
    pipeline_args: Dict
    future: Optional[Future] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    result: Optional[Dict] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AsyncPipelineProcessor:
    """Manages asynchronous pipeline processing with progress reporting."""
    
    def __init__(self, 
                 max_workers: int = 1,
                 progress_callback: Optional[Callable] = None,
                 verbosity: VerbosityLevel = VerbosityLevel.SUMMARY):
        """
        Initialize async processor.
        
        Args:
            max_workers: Maximum concurrent processing threads (1 recommended for memory)
            progress_callback: Function to call with progress updates
            verbosity: Console output verbosity level
        """
        self.max_workers = max_workers
        self.progress_callback = progress_callback
        self.verbosity = verbosity
        
        # Thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="WhisperJAV-Worker"
        )
        
        # Task tracking
        self.tasks: Dict[str, ProcessingTask] = {}
        self.task_lock = threading.RLock()
        
        # Progress queue for inter-thread communication
        self.progress_queue = Queue()
        
        # Cancellation support
        self._shutdown = False
        self._cancellation_tokens = {}
        
        # Start progress monitor thread
        self.progress_monitor = threading.Thread(
            target=self._monitor_progress,
            name="WhisperJAV-ProgressMonitor",
            daemon=True
        )
        self.progress_monitor.start()
    
    def submit_task(self, 
                   media_info: Dict,
                   pipeline_class: type,
                   pipeline_args: Dict) -> str:
        """
        Submit a media file for async processing.
        
        Args:
            media_info: Media file information
            pipeline_class: Pipeline class to use (e.g., FasterPipeline)
            pipeline_args: Arguments for pipeline initialization
            
        Returns:
            Task ID for tracking
        """
        # Generate task ID
        task_id = f"{Path(media_info['path']).stem}_{int(time.time()*1000)}"
        
        # Create cancellation token
        cancellation_token = threading.Event()
        self._cancellation_tokens[task_id] = cancellation_token
        
        # Create task
        task = ProcessingTask(
            task_id=task_id,
            media_info=media_info,
            pipeline_args=pipeline_args
        )
        
        # Submit to executor
        future = self.executor.submit(
            self._process_media,
            task_id,
            media_info,
            pipeline_class,
            pipeline_args,
            cancellation_token
        )
        
        # Add done callback
        future.add_done_callback(lambda f: self._on_task_complete(task_id, f))
        
        # Store task
        task.future = future
        task.status = ProcessingStatus.RUNNING
        task.start_time = time.time()
        
        with self.task_lock:
            self.tasks[task_id] = task
        
        # Report task start
        self._report_progress('task_start', task_id=task_id, media_info=media_info)
        
        return task_id
    

    def _process_media(self,
                      task_id: str,
                      media_info: Dict,
                      pipeline_class: type,
                      pipeline_args: Dict,
                      cancellation_token: threading.Event) -> Dict:
        """
        Process media in background thread. This is the core worker function.
        """
        progress_reporter = AsyncProgressReporter(
            self.progress_queue,
            self.verbosity
        )
        
        # Add the reporter to the pipeline args, as it's an expected kwarg
        pipeline_args_with_reporter = pipeline_args.copy()
        pipeline_args_with_reporter['progress_reporter'] = progress_reporter
        
        try:
            # CORRECTLY report progress using metadata from media_info
            progress_reporter.report_file_start(
                filename=media_info['basename'],
                file_number=media_info.get('file_number', 1),
                total_files=media_info.get('total_files', 1)
            )
            
            # Create pipeline instance with a clean set of args
            pipeline = pipeline_class(**pipeline_args_with_reporter)
            
            if cancellation_token.is_set():
                raise InterruptedError("Task cancelled before processing started")
            
            result = pipeline.process(media_info)
            
            progress_reporter.report_completion(
                success=True,
                stats={
                    'subtitles': result.get('summary', {}).get('final_subtitles_refined', 0),
                    'duration': time.time() - self.tasks[task_id].start_time
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            progress_reporter.report_completion(
                success=False,
                stats={'error': str(e)}
            )
            raise
            
        except Exception as e:
            # Report error
            logger.error(f"Task {task_id} failed: {e}", exc_info=True)
            progress_reporter.report_completion(
                success=False,
                stats={'error': str(e)}
            )
            raise
    
    def _process_with_cancellation(self,
                                 pipeline: Any,
                                 media_info: Dict,
                                 cancellation_token: threading.Event,
                                 progress_reporter: AsyncProgressReporter) -> Dict:
        """
        Process media with periodic cancellation checks.
        
        Note: This would require modifying the pipeline classes to accept
        and check a cancellation token. For now, this is a placeholder.
        """
        # TODO: Modify pipeline.process() to accept cancellation_token
        # For now, just process normally
        return pipeline.process(media_info)
    
    def _on_task_complete(self, task_id: str, future: Future):
        """Handle task completion."""
        with self.task_lock:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            task.end_time = time.time()
            
            try:
                # Get result
                result = future.result()
                task.status = ProcessingStatus.COMPLETED
                task.result = result
                
                # Report success
                self._report_progress('task_complete', 
                                    task_id=task_id,
                                    success=True,
                                    result=result)
                
            except Exception as e:
                # Handle cancellation vs other errors
                if isinstance(e, InterruptedError):
                    task.status = ProcessingStatus.CANCELLED
                else:
                    task.status = ProcessingStatus.FAILED
                    task.error = e
                
                # Report failure
                self._report_progress('task_complete',
                                    task_id=task_id,
                                    success=False,
                                    error=str(e))
            
            # Clean up cancellation token
            if task_id in self._cancellation_tokens:
                del self._cancellation_tokens[task_id]
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a running task.
        
        Returns:
            True if cancellation was initiated, False if task not found/running
        """
        with self.task_lock:
            if task_id not in self.tasks:
                return False
            
            task = self.tasks[task_id]
            if task.status != ProcessingStatus.RUNNING:
                return False
            
            # Set cancellation token
            if task_id in self._cancellation_tokens:
                self._cancellation_tokens[task_id].set()
                
                # Note: The actual cancellation depends on the pipeline
                # checking this token periodically
                self._report_progress('task_cancelled', task_id=task_id)
                return True
            
            return False
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Get current status of a task."""
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> ProcessingTask:
        """
        Wait for a task to complete.
        
        Args:
            task_id: Task to wait for
            timeout: Maximum time to wait in seconds
            
        Returns:
            Final task status
            
        Raises:
            TimeoutError if timeout exceeded
        """
        with self.task_lock:
            if task_id not in self.tasks:
                raise ValueError(f"Unknown task: {task_id}")
            
            task = self.tasks[task_id]
        
        if task.future:
            try:
                task.future.result(timeout=timeout)
            except Exception:
                pass  # Already handled in callback
        
        return self.get_task_status(task_id)
    
    def process_batch(self,
                     media_files: List[Dict],
                     pipeline_class: type,
                     pipeline_args: Dict,
                     wait: bool = True) -> List:
        """
        Process a batch of media files.
        """
        task_ids = []
        # --- FIX (a): Create a copy of the media_info dict to prevent side-effects ---
        # and correctly add file numbers for progress reporting.
        for i, mi in enumerate(media_files):
            media_info_copy = mi.copy()
            media_info_copy['file_number'] = i + 1
            media_info_copy['total_files'] = len(media_files)
            task_id = self.submit_task(media_info_copy, pipeline_class, pipeline_args)
            task_ids.append(task_id)
        
        if not wait:
            return task_ids
        
        futures = [self.tasks[tid].future for tid in task_ids if self.tasks[tid].future]
        as_completed(futures)
        
        return [self.tasks[tid] for tid in task_ids]

    
    def _monitor_progress(self):
        """Monitor progress queue and dispatch to callback."""
        while not self._shutdown:
            try:
                # Get progress message with timeout
                message = self.progress_queue.get(timeout=0.1)
                
                # Dispatch to callback if provided
                if self.progress_callback:
                    try:
                        self.progress_callback(message)
                    except Exception as e:
                        logger.error(f"Progress callback error: {e}")
                        
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Progress monitor error: {e}")
    
    def _report_progress(self, event_type: str, **kwargs):
        """Report progress event."""
        message = {
            'type': event_type,
            'timestamp': time.time(),
            **kwargs
        }
        self.progress_queue.put(message)
    
    def shutdown(self, wait: bool = True):
        """Shutdown the processor."""
        self._shutdown = True
        
        # Cancel all running tasks
        with self.task_lock:
            for task_id in list(self._cancellation_tokens.keys()):
                self.cancel_task(task_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=wait)
        
        # Wait for progress monitor to stop
        if self.progress_monitor.is_alive():
            self.progress_monitor.join(timeout=1.0)


class AsyncPipelineManager:
    """High-level manager for async pipeline processing with UI integration."""
    
    def __init__(self, ui_update_callback: Callable, verbosity: VerbosityLevel = VerbosityLevel.SUMMARY):
        """
        Initialize pipeline manager.
        
        Args:
            ui_update_callback: Function to call for UI updates
            verbosity: Output verbosity level
        """
        self.ui_update_callback = ui_update_callback
        self.verbosity = verbosity
        
        # Create processor
        self.processor = AsyncPipelineProcessor(
            max_workers=1,  # Single worker to prevent memory issues
            progress_callback=self._handle_progress,
            verbosity=verbosity
        )
        
        # Progress aggregators for each file
        self.progress_aggregators = {}
    
    def _handle_progress(self, message: Dict):
        """Handle progress messages from worker threads."""
        msg_type = message.get('type')
        
        if msg_type == 'file_start':
            # Create progress aggregator for this file
            # We'd need to know total scenes - this would come from pipeline
            pass
        
        elif msg_type == 'scene_progress':
            # Update progress aggregator
            pass
        
        # Forward to UI callback
        self.ui_update_callback(message)
    
    def process_files(self,
                     media_files: List[Dict],
                     pipeline_mode: str,
                     resolved_config: Dict) -> List[str]:
        """
        Process files asynchronously, returning task IDs immediately.
        """
        # Import pipeline classes
        if pipeline_mode == "faster":
            from whisperjav.pipelines.faster_pipeline import FasterPipeline
            pipeline_class = FasterPipeline
        elif pipeline_mode == "fast":
            from whisperjav.pipelines.fast_pipeline import FastPipeline
            pipeline_class = FastPipeline
        elif pipeline_mode == "balanced":
            from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
            pipeline_class = BalancedPipeline
        else:  # fidelity
            from whisperjav.pipelines.fidelity_pipeline import FidelityPipeline
            pipeline_class = FidelityPipeline
        
        # Prepare a CLEAN set of pipeline args, containing only what the pipeline expects.
        pipeline_args = {
            'output_dir': resolved_config.get('output_dir', './output'),
            'temp_dir': resolved_config.get('temp_dir', './temp'),
            'keep_temp_files': resolved_config.get('keep_temp_files', False),
            'subs_language': resolved_config.get('subs_language', 'japanese'),
            'resolved_config': resolved_config
        }
        
        # Submit the batch for processing. This call is non-blocking.
        task_ids = self.processor.process_batch(
            media_files,
            pipeline_class,
            pipeline_args,
            wait=False
        )
        
        return task_ids
    
    def shutdown(self):
        """Shutdown the manager."""
        self.processor.shutdown()



# Asyncio support for I/O operations (as suggested by reviewer)
import asyncio
import aiofiles


async def async_read_file(filepath: Path) -> bytes:
    """Asynchronously read a file."""
    async with aiofiles.open(filepath, 'rb') as f:
        return await f.read()


async def async_write_file(filepath: Path, data: bytes):
    """Asynchronously write a file."""
    async with aiofiles.open(filepath, 'wb') as f:
        await f.write(data)


async def async_copy_file(src: Path, dst: Path):
    """Asynchronously copy a file."""
    data = await async_read_file(src)
    await async_write_file(dst, data)