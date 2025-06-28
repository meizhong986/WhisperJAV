#!/usr/bin/env python3
"""Progress display module for WhisperJAV - handles multi-level progress bars."""

from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import sys
import time
from contextlib import contextmanager

from whisperjav.utils.logger import logger


class ProgressDisplay:
    """Manages multi-level progress display for WhisperJAV processing."""
    
    def __init__(self, total_files: int, enabled: bool = True):
        self.enabled = enabled
        self.total_files = total_files
        
        if not self.enabled:
            return
            
        # Overall progress bar - usually fine as-is
        self.overall_pbar = tqdm(
            total=total_files,
            position=0,
            desc="Overall Progress",
            bar_format='{desc}: |{bar}| {n_fmt}/{total_fmt} files [{elapsed}<{remaining}]',
            file=sys.stdout,
            leave=True,
            ncols=100
        )
        
        # Status bar - no changes needed
        self.status_bar = tqdm(
            total=0,
            position=1,
            bar_format='{desc}',
            file=sys.stdout,
            leave=False
        )
        
        # Subtask progress bar - THIS IS THE KEY CHANGE
        self.subtask_pbar = tqdm(
            total=0,
            position=2,
            bar_format='{desc}: |{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]',
            file=sys.stdout,
            leave=False,
            miniters=10,
            mininterval=1.0,
            smoothing=0.3,
            ncols=100
        )
        
        # Message bar - no changes needed
        self.message_bar = tqdm(
            total=0,
            position=3,
            bar_format='{desc}',
            file=sys.stdout,
            leave=False
        )
        
        print("\n" * 4)
        
    def close(self):
        if not self.enabled:
            return
            
        self.subtask_pbar.close()
        self.status_bar.close()
        self.message_bar.close()
        self.overall_pbar.close()
        
        print("\033[4A")
        
    def update_overall(self, increment: int = 1):
        if self.enabled:
            self.overall_pbar.update(increment)
    
    def set_current_file(self, filename: str, file_number: int):
        if self.enabled:
            short_name = Path(filename).name
            if len(short_name) > 50:
                short_name = short_name[:47] + "..."
            self.status_bar.set_description(f"[{file_number}/{self.total_files}] Current: {short_name}")
        else:
            logger.info(f"\nProcessing file {file_number}/{self.total_files}: {Path(filename).name}")
    
    def set_current_step(self, step_name: str, step_number: Optional[int] = None, total_steps: int = 5):
        if self.enabled:
            if step_number:
                self.status_bar.set_description_str(
                    f"{self.status_bar.desc.split(' - ')[0]} - Step {step_number}/{total_steps}: {step_name}"
                )
            else:
                self.status_bar.set_description_str(
                    f"{self.status_bar.desc.split(' - ')[0]} - {step_name}"
                )
        else:
            logger.info(f"Step {step_number}: {step_name}" if step_number else step_name)
    


    def start_subtask(self, task_name: str, total_items: int):
        if self.enabled and total_items > 1:
            # Calculate smart miniters based on total items
            if total_items > 100:
                miniters = max(10, total_items // 50)  # 2% increments for large sets
            else:
                miniters = max(1, total_items // 20)   # 5% increments for smaller sets
                
            self.subtask_pbar.reset(total=total_items)
            self.subtask_pbar.set_description(task_name)
            # Set dynamic miniters
            self.subtask_pbar.miniters = miniters
            self.subtask_pbar.mininterval = 1.0 if total_items > 100 else 0.5
            # Disable additional info on the right (rate, time remaining)
            self.subtask_pbar.dynamic_ncols = False
            self.subtask_pbar.ncols = 100
            # Update bar format to remove timing info
            self.subtask_pbar.bar_format = '{desc}: |{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]'
            self.subtask_pbar.refresh()



    
    def update_subtask(self, increment: int = 1):
        if self.enabled:
            self.subtask_pbar.update(increment)
    
    def finish_subtask(self):
        if self.enabled:
            self.subtask_pbar.reset(total=0)
            self.subtask_pbar.set_description("")
            self.subtask_pbar.refresh()
    
    def show_message(self, message: str, level: str = "info", duration: float = 2.0):
        if self.enabled:
            colors = {
                "info": "\033[94m",    # Blue
                "warning": "\033[93m", # Yellow
                "error": "\033[91m",   # Red
                "success": "\033[92m", # Green
            }
            reset = "\033[0m"
            
            prefix = {
                "info": "[INFO]",
                "warning": "[WARN]",
                "error": "[FAIL]",
                "success": "[OK]"
            }.get(level, "[INFO]")
            
            colored_msg = f"{colors.get(level, '')}{prefix} {message}{reset}"
            self.message_bar.set_description(colored_msg)
            self.message_bar.refresh()
            
            if duration > 0:
                time.sleep(duration)
                self.message_bar.set_description("")
                self.message_bar.refresh()
        else:
            getattr(logger, level, logger.info)(message)
    
    def show_file_complete(self, filename: str, subtitle_count: int, output_path: str):
        if self.enabled:
            short_name = Path(filename).name
            if len(short_name) > 30:
                short_name = short_name[:27] + "..."
            
            msg = f"[OK] {short_name} -> {subtitle_count} subtitle{'s' if subtitle_count != 1 else ''}"
            self.show_message(msg, "success", 1.5)
        else:
            logger.info(f"Output saved to: {output_path}")
    
    @contextmanager
    def pause_for_input(self):
        if self.enabled:
            self.subtask_pbar.clear()
            self.status_bar.clear()
            self.message_bar.clear()
            self.overall_pbar.clear()
            
        yield
        
        if self.enabled:
            self.overall_pbar.refresh()
            self.status_bar.refresh()
            self.subtask_pbar.refresh()
            self.message_bar.refresh()



class DummyProgress:
    """Dummy progress class for when progress bars are disabled."""
    
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
            logger.info(f"Step {step_number}: {step_name}")
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
        logger.info(f"Completed: {Path(filename).name} → {subtitle_count} subtitles")
        logger.info(f"Output saved to: {output_path}")
    
    @contextmanager
    def pause_for_input(self):
        yield