#!/usr/bin/env python3
"""Unified progress management system for WhisperJAV.

This module provides centralized progress coordination to eliminate the chaos
of multiple competing progress systems and reduce message spam from 450+ to <50 messages.
"""

import sys
import time
import threading
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
import io

from whisperjav.utils.logger import logger


class VerbosityLevel(Enum):
    """Progress display verbosity levels."""
    QUIET = 1      # File progress only
    STANDARD = 2   # File + major steps
    DETAILED = 3   # File + steps + scene batches  
    DEBUG = 4      # All technical details


@dataclass
class ProgressContext:
    """Hierarchical progress context."""
    level: str  # 'file', 'step', 'task', 'subtask'
    name: str
    current: int = 0
    total: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    parent: Optional['ProgressContext'] = None
    children: List['ProgressContext'] = field(default_factory=list)
    
    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if not self.total or self.total == 0:
            return 0.0
        return min(100.0, (self.current / self.total) * 100.0)
    
    @property
    def elapsed_time(self) -> float:
        """Get elapsed time since start."""
        return time.time() - self.start_time


class ExternalProgressCapture:
    """Captures and filters external library progress output."""
    
    def __init__(self, unified_manager):
        self.unified_manager = unified_manager
        self.stdout_buffer = io.StringIO()
        self.stderr_buffer = io.StringIO()
        self.old_stdout = None
        self.old_stderr = None
        
    def __enter__(self):
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        sys.stdout = self.stdout_buffer
        sys.stderr = self.stderr_buffer
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Parse captured output for meaningful progress information
        stdout_content = self.stdout_buffer.getvalue()
        stderr_content = self.stderr_buffer.getvalue()
        
        # Extract any meaningful progress info and report it through unified manager
        self._extract_progress_info(stdout_content)
        self._extract_progress_info(stderr_content)
        
        # Restore original stdout/stderr
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        
    def _extract_progress_info(self, content: str):
        """Extract meaningful progress from library output."""
        if not content.strip():
            return
            
        lines = content.strip().split('\n')
        for line in lines:
            # Look for tqdm-like progress indicators
            if '|' in line and '%' in line and 'it/s' in line:
                # This looks like a tqdm progress bar - extract percentage if possible
                try:
                    if '%|' in line:
                        percent_part = line.split('%|')[0]
                        if percent_part and percent_part.strip()[-3:].replace(' ', '').isdigit():
                            percentage = float(percent_part.strip()[-3:])
                            self.unified_manager._report_external_progress(f"Library progress: {percentage:.0f}%")
                except (ValueError, IndexError):
                    pass
            # Look for error messages that should be preserved
            elif any(keyword in line.lower() for keyword in ['error', 'warning', 'failed', 'exception']):
                self.unified_manager._report_external_progress(f"External: {line}")


class ConsoleProgressHandler:
    """Renders hierarchical progress for console display."""
    
    def __init__(self):
        self.last_output_time = 0
        self.min_update_interval = 0.5  # Minimum seconds between updates
        
    def should_update(self) -> bool:
        """Check if enough time has passed for an update."""
        current_time = time.time()
        if current_time - self.last_output_time >= self.min_update_interval:
            self.last_output_time = current_time
            return True
        return False
    
    def render_progress(self, contexts: Dict[str, ProgressContext], verbosity: VerbosityLevel):
        """Render current progress state to console."""
        if not self.should_update():
            return
            
        if verbosity == VerbosityLevel.QUIET:
            self._render_quiet_mode(contexts)
        elif verbosity == VerbosityLevel.STANDARD:
            self._render_standard_mode(contexts)
        elif verbosity == VerbosityLevel.DETAILED:
            self._render_detailed_mode(contexts)
        else:  # DEBUG
            self._render_debug_mode(contexts)
    
    def _render_quiet_mode(self, contexts: Dict[str, ProgressContext]):
        """Minimal progress display - file progress only."""
        file_context = self._find_active_file_context(contexts)
        if file_context and file_context.total:
            progress_pct = file_context.progress_percent
            if progress_pct in [0, 25, 50, 75, 100] or int(progress_pct) % 10 == 0:
                print(f"\rProcessing scenes: ", end='', flush=True)
                #print(f"\rStep {current_step}/5: {progress_pct:3.0f}%", end='', flush=True)
    
    def _render_standard_mode(self, contexts: Dict[str, ProgressContext]):
        """Standard progress display matching industry best practices."""
        # Disable progress bar output - keep it clean and simple
        return
    
    def _render_detailed_mode(self, contexts: Dict[str, ProgressContext]):
        """Detailed progress with step and batch information."""
        # Disable progress bar output - keep it clean and simple
        return
    
    def _render_debug_mode(self, contexts: Dict[str, ProgressContext]):
        """Full debug information with all context details."""
        lines = ["=== DEBUG PROGRESS ==="]
        
        for context_id, context in contexts.items():
            indent = "  " * (len(context_id.split('_')) - 1)
            progress_info = f"{context.progress_percent:.1f}%" if context.total else "N/A"
            elapsed = f"{context.elapsed_time:.1f}s"
            lines.append(f"{indent}{context.level}: {context.name} - {progress_info} ({elapsed})")
            
            if context.details:
                for key, value in context.details.items():
                    lines.append(f"{indent}  {key}: {value}")
        
        lines.append("=" * 23)
        output = "\n".join(lines)
        print(f"\r{output}", end='', flush=True)
    
    def _create_progress_bar(self, percentage: float, width: int = 30) -> str:
        """Create ASCII progress bar."""
        filled = int(width * percentage / 100)
        return '█' * filled + '░' * (width - filled)
    
    def _find_active_file_context(self, contexts: Dict[str, ProgressContext]) -> Optional[ProgressContext]:
        """Find the active file context."""
        for context in contexts.values():
            if context.level == 'file' and context.current < (context.total or 1):
                return context
        # Return the most recent file context if none are active
        file_contexts = [ctx for ctx in contexts.values() if ctx.level == 'file']
        return file_contexts[-1] if file_contexts else None
    
    def _find_active_step_context(self, contexts: Dict[str, ProgressContext], 
                                file_context: ProgressContext) -> Optional[ProgressContext]:
        """Find the active step context for a file."""
        for context in contexts.values():
            if (context.level == 'step' and context.parent == file_context and 
                context.current < (context.total or 1)):
                return context
        return None
    
    def _find_active_task_context(self, contexts: Dict[str, ProgressContext], 
                                step_context: Optional[ProgressContext]) -> Optional[ProgressContext]:
        """Find the active task context for a step."""
        if not step_context:
            return None
        for context in contexts.values():
            if (context.level == 'task' and context.parent == step_context and 
                context.current < (context.total or 1)):
                return context
        return None


class UnifiedProgressManager:
    """Central progress coordination for all WhisperJAV operations."""
    
    def __init__(self, verbosity: VerbosityLevel = VerbosityLevel.STANDARD):
        self.verbosity = verbosity
        self.contexts: Dict[str, ProgressContext] = {}
        self.active_file_context_id = None
        self.active_step_context_id = None
        self.active_task_context_id = None
        
        self.output_handler = ConsoleProgressHandler()
        self.update_interval = 1.0  # Rate limiting for updates
        self.last_update = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # External library progress suppression
        self.external_capture = ExternalProgressCapture(self)
        
        # Message batching to reduce spam
        self._message_buffer = []
        self._last_message_flush = time.time()
        self._message_flush_interval = 2.0
    
    def start_file_processing(self, filename: str, file_num: int, total_files: int) -> str:
        """Begin processing a file - returns context ID."""
        context_id = f"file_{file_num}"
        
        with self._lock:
            context = ProgressContext(
                level='file',
                name=filename,
                current=0,
                total=5,  # Standard pipeline steps: audio, detection, transcription, processing, output
                start_time=time.time()
            )
            self.contexts[context_id] = context
            self.active_file_context_id = context_id
            
            # Log file start in non-quiet mode
            if self.verbosity != VerbosityLevel.QUIET:
                logger.info(f"\n[{file_num}/{total_files}] Starting: {filename}")
            
        self._schedule_update()
        return context_id
    
    def start_step(self, step_name: str, step_num: int, total_steps: int = 5, 
                  parent_context_id: Optional[str] = None) -> str:
        """Begin a processing step within current file."""
        parent_id = parent_context_id or self.active_file_context_id
        if not parent_id or parent_id not in self.contexts:
            return None
            
        context_id = f"{parent_id}_step_{step_num}"
        
        with self._lock:
            parent_context = self.contexts[parent_id]
            parent_context.current = step_num
            
            step_context = ProgressContext(
                level='step',
                name=step_name,
                current=0,
                total=None,  # Will be set by tasks
                parent=parent_context,
                start_time=time.time()
            )
            self.contexts[context_id] = step_context
            self.active_step_context_id = context_id
            
            # Add to parent's children
            if step_context not in parent_context.children:
                parent_context.children.append(step_context)
                
        self._schedule_update()
        return context_id
    
    def start_task(self, task_name: str, total_items: int, parent_context_id: Optional[str] = None) -> str:
        """Begin a task (like scene transcription)."""
        parent_id = parent_context_id or self.active_step_context_id
        if not parent_id or parent_id not in self.contexts:
            return None
            
        context_id = f"{parent_id}_task_{int(time.time() * 1000) % 10000}"  # Unique suffix
        
        with self._lock:
            parent_context = self.contexts[parent_id]
            
            task_context = ProgressContext(
                level='task',
                name=task_name,
                current=0,
                total=total_items,
                parent=parent_context,
                start_time=time.time()
            )
            self.contexts[context_id] = task_context
            self.active_task_context_id = context_id
            
            # Update parent step context total if not set
            if parent_context and parent_context.level == 'step' and not parent_context.total:
                parent_context.total = total_items
                
            # Add to parent's children
            if task_context not in parent_context.children:
                parent_context.children.append(task_context)
                
        self._schedule_update()
        return context_id
    
    def update_task_progress(self, context_id: str, increment: int = 1, details: Optional[Dict] = None):
        """Update task progress with smart batching and rate limiting."""
        if not context_id or context_id not in self.contexts:
            return
            
        with self._lock:
            context = self.contexts[context_id]
            context.current = min(context.current + increment, context.total or context.current + increment)
            
            if details:
                context.details.update(details)
                
            # Propagate progress up the hierarchy
            self._propagate_progress(context)
            
        # Rate-limited updates to prevent spam
        if time.time() - self.last_update > self.update_interval:
            self._schedule_update()
    
    def complete_step(self, context_id: str):
        """Mark a step as complete."""
        if not context_id or context_id not in self.contexts:
            return
            
        with self._lock:
            context = self.contexts[context_id]
            if context.level == 'step' and context.total:
                context.current = context.total
                
            # Move to next step in parent file
            if context.parent and context.parent.level == 'file':
                context.parent.current = min(context.parent.current + 1, context.parent.total or 1)
                
        self._schedule_update()
    
    def complete_file(self, context_id: str, success: bool = True, details: Optional[Dict] = None):
        """Mark a file as complete."""
        if not context_id or context_id not in self.contexts:
            return
            
        with self._lock:
            context = self.contexts[context_id]
            if context.level == 'file':
                context.current = context.total or 1
                context.details['completed'] = success
                if details:
                    context.details.update(details)
                    
        if self.verbosity != VerbosityLevel.QUIET:
            status = "✓ Complete" if success else "✗ Failed"
            logger.info(f"{status}: {context.name}")
            
        self._schedule_update()
    
    @contextmanager
    def suppress_external_progress(self):
        """Context manager to suppress external library progress."""
        if self.verbosity in [VerbosityLevel.QUIET, VerbosityLevel.STANDARD]:
            with self.external_capture:
                yield
        else:
            # In detailed/debug mode, allow some external progress
            yield
    
    def _report_external_progress(self, message: str):
        """Report progress from external libraries."""
        if self.verbosity == VerbosityLevel.DEBUG:
            logger.debug(f"External: {message}")
    
    def _schedule_update(self):
        """Schedule a progress display update with rate limiting."""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            self.output_handler.render_progress(self.contexts, self.verbosity)
    
    def _propagate_progress(self, context: ProgressContext):
        """Propagate progress up the hierarchy."""
        if context.parent:
            if context.parent.level == 'step' and context.level == 'task':
                # Update step progress based on task progress
                context.parent.current = context.current
                context.parent.total = context.total
    
    def cleanup(self):
        """Cleanup progress display."""
        if self.verbosity != VerbosityLevel.QUIET:
            print("\n")  # Move to next line after progress display