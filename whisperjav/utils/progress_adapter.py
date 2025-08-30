#!/usr/bin/env python3
"""Backward compatibility adapter for existing ProgressDisplay API.

This adapter bridges the old ProgressDisplay interface to the new UnifiedProgressManager
to maintain compatibility while migrating to the unified system.
"""

from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from whisperjav.utils.unified_progress import UnifiedProgressManager
from whisperjav.utils.logger import logger


class ProgressDisplayAdapter:
    """Adapter to bridge old ProgressDisplay calls to UnifiedProgressManager."""
    
    def __init__(self, unified_manager: UnifiedProgressManager):
        self.unified_manager = unified_manager
        self.total_files = 1
        self.current_file_context = None
        self.current_step_context = None
        self.current_task_context = None
        self.enabled = True  # Always enabled through unified manager
        
    def close(self):
        """Clean up progress display."""
        self.unified_manager.cleanup()
        
    def update_overall(self, increment: int = 1):
        """Update overall file progress."""
        # This is handled automatically by the unified manager
        # when files are completed
        pass
    
    def set_current_file(self, filename: str, file_number: int):
        """Set current file being processed."""
        self.current_file_context = self.unified_manager.start_file_processing(
            filename, file_number, self.total_files
        )
    
    def set_current_step(self, step_name: str, step_number: Optional[int] = None, total_steps: int = 5):
        """Set current processing step."""
        if step_number is None:
            step_number = 1
            
        self.current_step_context = self.unified_manager.start_step(
            step_name, step_number, total_steps, self.current_file_context
        )
    
    def start_subtask(self, task_name: str, total_items: int):
        """Start a subtask (e.g., scene transcription)."""
        if total_items > 1:  # Only create task context for meaningful subtasks
            self.current_task_context = self.unified_manager.start_task(
                task_name, total_items, self.current_step_context
            )
    
    def update_subtask(self, increment: int = 1):
        """Update subtask progress."""
        if self.current_task_context:
            self.unified_manager.update_task_progress(self.current_task_context, increment)
    
    def finish_subtask(self):
        """Finish the current subtask."""
        if self.current_step_context:
            self.unified_manager.complete_step(self.current_step_context)
        self.current_task_context = None
    
    def show_message(self, message: str, level: str = "info", duration: float = 2.0):
        """Show a message through the logger (unified with overall logging)."""
        # Route messages through the logger to maintain consistency
        # and avoid cluttering the progress display
        log_func = getattr(logger, level, logger.info)
        
        # Only show important messages in quiet/standard modes
        if self.unified_manager.verbosity.value <= 2:  # QUIET or STANDARD
            if level in ["warning", "error"]:
                log_func(message)
        else:  # DETAILED or DEBUG
            log_func(message)
    
    def show_file_complete(self, filename: str, subtitle_count: int, output_path: str):
        """Show file completion message."""
        if self.current_file_context:
            details = {
                'subtitle_count': subtitle_count,
                'output_path': output_path
            }
            self.unified_manager.complete_file(self.current_file_context, success=True, details=details)
            
        # Always log completion for user awareness
        short_name = Path(filename).name
        logger.info(f"✓ {short_name} → {subtitle_count} subtitle{'s' if subtitle_count != 1 else ''}")
    
    @contextmanager
    def pause_for_input(self):
        """Pause progress display for user input."""
        # The unified manager handles this more gracefully
        # by not interfering with console input
        yield


class DummyProgressAdapter:
    """Dummy adapter that routes everything to logger when progress is disabled."""
    
    def __init__(self):
        pass
    
    def close(self):
        pass
    
    def update_overall(self, increment: int = 1):
        pass
    
    def set_current_file(self, filename: str, file_number: int):
        logger.info(f"\nProcessing file {file_number}: {Path(filename).name}")
    
    def set_current_step(self, step_name: str, step_number: Optional[int] = None, total_steps: int = 5):
        if step_number:
            logger.info(f"Step {step_number}/{total_steps}: {step_name}")
        else:
            logger.info(step_name)
    
    def start_subtask(self, task_name: str, total_items: int):
        if total_items > 1:
            logger.info(f"{task_name} ({total_items} items)")
    
    def update_subtask(self, increment: int = 1):
        pass
    
    def finish_subtask(self):
        pass
    
    def show_message(self, message: str, level: str = "info", duration: float = 2.0):
        getattr(logger, level, logger.info)(message)
    
    def show_file_complete(self, filename: str, subtitle_count: int, output_path: str):
        logger.info(f"✓ Completed: {Path(filename).name} → {subtitle_count} subtitles")
        logger.info(f"Output saved to: {output_path}")
    
    @contextmanager
    def pause_for_input(self):
        yield


def create_progress_adapter(unified_manager: Optional[UnifiedProgressManager] = None) -> ProgressDisplayAdapter:
    """Factory function to create appropriate progress adapter."""
    if unified_manager:
        return ProgressDisplayAdapter(unified_manager)
    else:
        return DummyProgressAdapter()