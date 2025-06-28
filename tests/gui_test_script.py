#!/usr/bin/env python3
"""
Test script for WhisperJAV GUI
Run this to test the GUI without needing WhisperJAV installed
"""

import tkinter as tk
from tkinter import ttk
import time
import random

class MockWhisperJAV:
    """Mock WhisperJAV for testing the GUI"""
    
    def __init__(self, console_callback):
        self.console = console_callback
        
    def process(self, file_path, mode, sensitivity, language):
        """Simulate WhisperJAV processing"""
        self.console(f"Starting processing with settings:", 'info')
        self.console(f"  Mode: {mode}", 'info')
        self.console(f"  Sensitivity: {sensitivity}", 'info')
        self.console(f"  Language: {language}", 'info')
        self.console(f"  File: {file_path}", 'info')
        self.console("", 'info')
        
        # Simulate processing steps
        steps = [
            "Loading Whisper model...",
            "Extracting audio from media file...",
            "Detecting audio scenes...",
            "Found 23 scenes",
            "Transcribing scene 1/23...",
            "Transcribing scene 2/23...",
            "Transcribing scene 3/23...",
            "[Progress bar simulation would appear here]",
            "Combining transcriptions...",
            "Applying post-processing...",
            "Removing hallucinations...",
            "Adjusting timings...",
            "Saving final SRT file..."
        ]
        
        for i, step in enumerate(steps):
            time.sleep(0.5)  # Simulate processing time
            
            if "scene" in step:
                self.console(step, 'output')
            elif "Found" in step:
                self.console(step, 'success')
            else:
                self.console(step, 'info')
                
            # Simulate some progress output
            if random.random() > 0.7:
                self.console(f"  {random.choice(['✓', '→', '•'])} Processing...", 'output')
        
        self.console("", 'info')
        self.console("✅ Processing complete!", 'success')
        self.console(f"Output saved to: output/{file_path}.ja.whisperjav.srt", 'info')

def test_gui():
    """Test the GUI with mock processing"""
    print("Testing WhisperJAV GUI...")
    print("This will open a test window with simulated processing.")
    print("")
    
    # Import the GUI
    try:
        from whisperjav_gui import WhisperJAVGUI
        
        # Monkey-patch the process_files method for testing
        original_process = WhisperJAVGUI.process_files
        
        def mock_process_files(self):
            """Mock processing for testing"""
            try:
                self.log("Starting MOCK WhisperJAV processing...\n", 'info')
                self.log("NOTE: This is a test mode - not actually processing files\n", 'warning')
                
                mock = MockWhisperJAV(self.log)
                
                for i, file_path in enumerate(self.selected_files, 1):
                    if not self.processing:
                        break
                        
                    self.log(f"\n[{i}/{len(self.selected_files)}] Processing: {file_path}", 'info')
                    mock.process(file_path, self.speed_var.get(), 
                               self.granularity_var.get(), self.language_var.get())
                    
                if self.processing:
                    self.log("\n✅ All files processed (TEST MODE)!", 'success')
                else:
                    self.log("\n⚠ Processing stopped by user", 'warning')
                    
            except Exception as e:
                self.log(f"\nError: {str(e)}", 'error')
            finally:
                self.processing = False
                self.root.after(0, self.reset_ui_after_processing)
        
        # Replace with mock
        WhisperJAVGUI.process_files = mock_process_files
        
        # Create and run GUI
        root = tk.Tk()
        app = WhisperJAVGUI(root)
        
        # Add test mode indicator
        test_label = ttk.Label(root, text="TEST MODE - Not connected to WhisperJAV", 
                              foreground='red', font=('Arial', 10, 'bold'))
        test_label.place(x=10, y=10)
        
        root.mainloop()
        
    except ImportError as e:
        print(f"Error: Could not import WhisperJAV GUI: {e}")
        print("Make sure whisperjav_gui.py is in the same directory")

if __name__ == "__main__":
    test_gui()