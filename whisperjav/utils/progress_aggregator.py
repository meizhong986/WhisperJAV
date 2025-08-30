#!/usr/bin/env python3
"""Smart progress aggregation system to reduce console output overwhelm.

This module provides intelligent batching and summarization of progress updates
to create a balance between keeping users informed and avoiding spam.
"""

import time
import sys
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from queue import Queue, Empty
import threading
from contextlib import contextmanager

from whisperjav.utils.logger import logger


class VerbosityLevel(Enum):
    """Console output verbosity levels."""
    QUIET = "quiet"          # Only major milestones
    SUMMARY = "summary"      # Scene batch summaries  
    NORMAL = "normal"        # Default behavior
    VERBOSE = "verbose"      # Debug level output


@dataclass
class SceneProgress:
    """Progress information for a single scene."""
    scene_index: int
    start_time: float
    end_time: Optional[float] = None
    status: str = "pending"
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ProgressBatch:
    """A batch of scene progress updates."""
    batch_id: int
    scenes: List[SceneProgress]
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        if not self.scenes:
            return 0.0
        successful = sum(1 for s in self.scenes if s.status == "complete")
        return successful / len(self.scenes)


class ProgressAggregator:
    """Intelligent progress aggregation to reduce console spam."""
    
    def __init__(self, 
                 total_scenes: int,
                 verbosity: VerbosityLevel = VerbosityLevel.SUMMARY,
                 batch_size: int = 10,
                 output_fn: Optional[Callable[[str], None]] = None):
        """
        Initialize the progress aggregator.
        
        Args:
            total_scenes: Total number of scenes to process
            verbosity: Output verbosity level
            batch_size: Number of scenes to batch before reporting
            output_fn: Custom output function (defaults to print)
        """
        self.total_scenes = total_scenes
        self.verbosity = verbosity
        self.batch_size = batch_size
        self.output_fn = output_fn or print
        
        # Progress tracking
        self.completed_scenes = 0
        self.failed_scenes = 0
        self.current_batch = []
        self.batch_count = 0
        self.start_time = time.time()
        
        # Scene timing statistics
        self.scene_times = []
        
        # Current processing state
        self.current_file = ""
        self.current_step = ""
        
        # Thread safety
        self._lock = threading.RLock()
        
    def set_file_info(self, filename: str, file_number: int, total_files: int):
        """Set current file being processed."""
        with self._lock:
            self.current_file = filename
            if self.verbosity != VerbosityLevel.QUIET:
                self.output_fn(f"\n[{file_number}/{total_files}] Processing: {filename}")
    
    def set_step_info(self, step_name: str, step_number: int, total_steps: int):
        """Set current processing step."""
        with self._lock:
            self.current_step = step_name
            if self.verbosity == VerbosityLevel.VERBOSE:
                self.output_fn(f"  Step {step_number}/{total_steps}: {step_name}")
            elif self.verbosity == VerbosityLevel.NORMAL:
                # Show only major steps
                if step_number in [1, 3, 5]:  # Audio extraction, transcription, post-processing
                    self.output_fn(f"  {step_name}...")
    
    def start_scene(self, scene_index: int) -> SceneProgress:
        """Start tracking a scene."""
        with self._lock:
            progress = SceneProgress(
                scene_index=scene_index,
                start_time=time.time()
            )
            self.current_batch.append(progress)
            
            # Immediate output only in verbose mode
            if self.verbosity == VerbosityLevel.VERBOSE:
                self.output_fn(f"    Scene {scene_index + 1}/{self.total_scenes}: Starting...")
            
            return progress
    
    def update_scene(self, scene_progress: SceneProgress, status: str, details: Optional[Dict] = None):
        """Update scene progress."""
        with self._lock:
            scene_progress.status = status
            scene_progress.end_time = time.time()
            if details:
                scene_progress.details.update(details)
            
            # Check if batch is complete
            if len(self.current_batch) >= self.batch_size:
                self._flush_batch()
    
    def complete_scene(self, scene_progress: SceneProgress, success: bool = True, error: Optional[str] = None):
        """Mark a scene as complete."""
        with self._lock:
            scene_progress.end_time = time.time()
            scene_progress.status = "complete" if success else "failed"
            scene_progress.error = error
            
            # Update counters
            if success:
                self.completed_scenes += 1
                self.scene_times.append(scene_progress.end_time - scene_progress.start_time)
            else:
                self.failed_scenes += 1
            
            # Verbose mode: immediate feedback
            if self.verbosity == VerbosityLevel.VERBOSE:
                duration = scene_progress.end_time - scene_progress.start_time
                if success:
                    self.output_fn(f"    Scene {scene_progress.scene_index + 1}: ✓ Complete ({duration:.1f}s)")
                else:
                    self.output_fn(f"    Scene {scene_progress.scene_index + 1}: ✗ Failed - {error}")
            
            # Check for batch completion
            if len(self.current_batch) >= self.batch_size:
                self._flush_batch()
    
    def _flush_batch(self):
        """Flush the current batch and display summary."""
        if not self.current_batch:
            return
        
        batch = ProgressBatch(
            batch_id=self.batch_count,
            scenes=self.current_batch[:],
            start_time=self.current_batch[0].start_time,
            end_time=time.time()
        )
        
        self._display_batch_summary(batch)
        
        self.batch_count += 1
        self.current_batch.clear()
    
    def _display_batch_summary(self, batch: ProgressBatch):
        """Display a summary for a batch of scenes."""
        if self.verbosity == VerbosityLevel.QUIET:
            # Only show percentage milestones
            progress_pct = (self.completed_scenes / self.total_scenes) * 100
            if progress_pct in [25, 50, 75, 100]:
                self.output_fn(f"  Progress: {progress_pct:.0f}%")
            return
        
        # Calculate batch statistics
        successful = sum(1 for s in batch.scenes if s.status == "complete")
        failed = sum(1 for s in batch.scenes if s.status == "failed")
        
        # Scene range
        scene_start = batch.scenes[0].scene_index + 1
        scene_end = batch.scenes[-1].scene_index + 1
        
        # Average time per scene
        avg_time = batch.duration / len(batch.scenes) if batch.scenes else 0
        
        # ETA calculation
        remaining_scenes = self.total_scenes - self.completed_scenes
        eta_seconds = remaining_scenes * (sum(self.scene_times) / len(self.scene_times) if self.scene_times else avg_time)
        eta = timedelta(seconds=int(eta_seconds))
        
        # Progress bar
        progress_pct = (self.completed_scenes / self.total_scenes) * 100
        bar_width = 30
        filled = int(bar_width * self.completed_scenes / self.total_scenes)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        if self.verbosity == VerbosityLevel.SUMMARY:
            # Compact summary
            self.output_fn(
                f"  [{bar}] {progress_pct:3.0f}% | "
                f"Scenes {scene_start}-{scene_end}: "
                f"✓{successful} ✗{failed} | "
                f"Avg: {avg_time:.1f}s | "
                f"ETA: {eta}"
            )
        else:  # NORMAL
            # More detailed summary
            self.output_fn(f"\n  Scene Batch {scene_start}-{scene_end} Summary:")
            self.output_fn(f"    Successful: {successful}/{len(batch.scenes)}")
            if failed > 0:
                self.output_fn(f"    Failed: {failed}")
            self.output_fn(f"    Batch time: {batch.duration:.1f}s (avg {avg_time:.1f}s/scene)")
            self.output_fn(f"    Overall progress: [{bar}] {progress_pct:.1f}%")
            self.output_fn(f"    ETA: {eta}")
    
    def finalize(self):
        """Finalize processing and show final summary."""
        with self._lock:
            # Flush any remaining scenes
            if self.current_batch:
                self._flush_batch()
            
            # Final summary
            total_time = time.time() - self.start_time
            
            if self.verbosity != VerbosityLevel.QUIET:
                self.output_fn("\n" + "="*50)
                self.output_fn(f"Transcription Complete")
                self.output_fn(f"  Total scenes: {self.total_scenes}")
                self.output_fn(f"  Successful: {self.completed_scenes}")
                if self.failed_scenes > 0:
                    self.output_fn(f"  Failed: {self.failed_scenes}")
                self.output_fn(f"  Total time: {timedelta(seconds=int(total_time))}")
                if self.scene_times:
                    avg_scene_time = sum(self.scene_times) / len(self.scene_times)
                    self.output_fn(f"  Average per scene: {avg_scene_time:.1f}s")
                self.output_fn("="*50)
    
    @contextmanager
    def scene_context(self, scene_index: int):
        """Context manager for processing a single scene."""
        progress = self.start_scene(scene_index)
        try:
            yield progress
            self.complete_scene(progress, success=True)
        except Exception as e:
            self.complete_scene(progress, success=False, error=str(e))
            raise


class AsyncProgressReporter:
    """Asynchronous progress reporter using queue-based communication."""
    
    def __init__(self, progress_queue: Queue, verbosity: VerbosityLevel = VerbosityLevel.SUMMARY):
        self.progress_queue = progress_queue
        self.verbosity = verbosity
        
    def report(self, message_type: str, **kwargs):
        """Send a progress message to the queue."""
        message = {
            'type': message_type,
            'timestamp': time.time(),
            'verbosity': self.verbosity.value,
            **kwargs
        }
        self.progress_queue.put(message)
    
    def report_file_start(self, filename: str, file_number: int, total_files: int):
        """Report starting a new file."""
        self.report('file_start', 
                   filename=filename, 
                   file_number=file_number,
                   total_files=total_files)
    
    def report_step(self, step_name: str, step_number: int, total_steps: int):
        """Report current processing step."""
        self.report('step_update',
                   step_name=step_name,
                   step_number=step_number,
                   total_steps=total_steps)
    
    def report_scene_progress(self, scene_index: int, total_scenes: int, 
                            status: str, details: Optional[Dict] = None):
        """Report scene processing progress."""
        self.report('scene_progress',
                   scene_index=scene_index,
                   total_scenes=total_scenes,
                   status=status,
                   details=details or {})
    
    def report_completion(self, success: bool, stats: Dict):
        """Report file processing completion."""
        self.report('completion',
                   success=success,
                   stats=stats)


def create_progress_handler(verbosity: VerbosityLevel, 
                          total_scenes: int,
                          output_fn: Optional[Callable] = None) -> ProgressAggregator:
    """Factory function to create appropriate progress handler."""
    # Determine batch size based on total scenes and verbosity - INCREASED FOR CLUTTER REDUCTION
    if verbosity == VerbosityLevel.QUIET:
        batch_size = max(75, total_scenes // 3)  # 3 updates max
    elif verbosity == VerbosityLevel.SUMMARY:
        batch_size = max(25, total_scenes // 12)  # ~12 updates (reduced from 20)
    elif verbosity == VerbosityLevel.NORMAL:
        batch_size = max(15, total_scenes // 25)   # ~25 updates (reduced from 40)
    else:  # VERBOSE
        batch_size = max(5, total_scenes // 50)  # Batch even in verbose mode
    
    return ProgressAggregator(
        total_scenes=total_scenes,
        verbosity=verbosity,
        batch_size=batch_size,
        output_fn=output_fn
    )