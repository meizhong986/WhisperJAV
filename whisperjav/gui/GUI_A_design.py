#!/usr/bin/env python3
"""
WhisperJAV GUI - Improved Version with Better Layout
30% larger fonts and better space utilization
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import subprocess
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
import io
import tkinter.font as tkfont

# Fix stdout before any other imports
def fix_stdout():
    """Ensure stdout is available, create a wrapper if needed."""
    if sys.stdout is None or (hasattr(sys.stdout, 'closed') and sys.stdout.closed):
        sys.stdout = io.TextIOWrapper(
            io.BufferedWriter(io.FileIO(1, 'w')), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )
    if sys.stderr is None or (hasattr(sys.stderr, 'closed') and sys.stderr.closed):
        sys.stderr = io.TextIOWrapper(
            io.BufferedWriter(io.FileIO(2, 'w')), 
            encoding='utf-8', 
            errors='replace',
            line_buffering=True
        )

fix_stdout()

class WhisperJAVGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("WhisperJAV - Subtitle Generator")
        
        # High-DPI awareness for Windows
        if sys.platform == 'win32':
            try:
                from ctypes import windll
                windll.shcore.SetProcessDpiAwareness(1)  # System DPI aware
            except:
                pass
        
        # Set window size to 95% of 11-inch screen (1920x1080)
        self.root.geometry("1820x1000")
        self.root.minsize(1500, 800)
        
        # Variables
        self.selected_files = []
        self.output_folder = None
        self.processing = False
        self.process_thread = None
        self.log_queue = queue.Queue()
        
        # Default values
        self.speed_var = tk.StringVar(value="balanced")
        self.granularity_var = tk.StringVar(value="balanced")
        self.language_var = tk.StringVar(value="japanese")
        
        # Setup fonts for high-DPI - 30% LARGER
        self.setup_fonts()
        
        # Create UI
        self.setup_ui()
        
        # Center window on screen
        self.center_window()
        
        # Start log queue processor
        self.process_log_queue()
        
    def setup_fonts(self):
        """Configure fonts for high-DPI displays - 30% larger than before"""
        

        # Define font sizes
        self.fonts = {
            'default': tkfont.Font(family='Segoe UI', size=16),      # was 12
            'label': tkfont.Font(family='Segoe UI', size=16),        # was 12
            'heading': tkfont.Font(family='Segoe UI', size=18, weight='bold'),  # was 14
            'button': tkfont.Font(family='Segoe UI', size=16),       # was 12
            'console': tkfont.Font(family='Consolas', size=16),      # was 13
            'subtitle': tkfont.Font(family='Segoe UI', size=14),     # was 11
            'large_button': tkfont.Font(family='Segoe UI', size=18, weight='bold'),  # was 14
            'gear': tkfont.Font(family='Arial', size=32),            # was 24
            'custom_subtitle': tkfont.Font(family='Segoe UI', size=18),  # New font size for specific texts
        }

        '''
        self.fonts = {
            'default': tkfont.Font(family='Segoe UI', size=16),      # was 12
            'label': tkfont.Font(family='Segoe UI', size=16),        # was 12
            'heading': tkfont.Font(family='Segoe UI', size=18, weight='bold'),  # was 14
            'button': tkfont.Font(family='Segoe UI', size=16),       # was 12
            'console': tkfont.Font(family='Consolas', size=16),      # was 13
            'subtitle': tkfont.Font(family='Segoe UI', size=14),     # was 11
            'large_button': tkfont.Font(family='Segoe UI', size=18, weight='bold'),  # was 14
            'gear': tkfont.Font(family='Arial', size=32),            # was 24
        }
        '''
        
        # Configure default font for all widgets
        self.root.option_add('*Font', self.fonts['default'])
        
        # Style configuration for ttk widgets
        style = ttk.Style()
        style.configure('TLabel', font=self.fonts['label'])
        style.configure('TButton', font=self.fonts['button'], padding=(10, 10))  # Added padding
        style.configure('TLabelframe.Label', font=self.fonts['heading'])
        style.configure('TRadiobutton', font=self.fonts['label'])
        
        # Configure Radiobutton to have more spacing
        style.configure('TRadiobutton', padding=(5, 5))
        
    def center_window(self):
        """Center the window on screen"""
        self.root.update_idletasks()
        
        # Get screen dimensions
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Window dimensions
        window_width = 1820
        window_height = 1000
        
        # Calculate position (centered with slight offset from top)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2 - 25
        
        # Ensure window is not off-screen
        x = max(0, x)
        y = max(0, y)
        
        # Set position
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
    def setup_ui(self):
        """Create the main UI layout"""
        # Main container with less padding to use more space
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)  # Console area expands
        main_frame.columnconfigure(0, weight=1)
        
        # Top section with 4 columns
        top_frame = ttk.Frame(main_frame)
        top_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=1)
        top_frame.columnconfigure(3, weight=1)
        
        # Column 1: File Selection
        self.create_file_section(top_frame)
        
        # Column 2: Speed Control
        self.create_speed_section(top_frame)
        
        # Column 3: Granularity Control
        self.create_granularity_section(top_frame)
        
        # Column 4: Language Selection
        self.create_language_section(top_frame)
        
        # Start Processing Button
        self.create_process_button(main_frame)
        
        # Console Output
        self.create_console_section(main_frame)
        
    def create_file_section(self, parent):
        """Create file selection section"""
        frame = ttk.LabelFrame(parent, text="Choose Media and Output Destination", padding="15")
        frame.grid(row=0, column=0, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Container for buttons to control width
        button_container = ttk.Frame(frame)
        button_container.pack(fill='x', expand=True)
        
        # Select Files button - fills available width
        self.select_files_btn = ttk.Button(button_container, text="Select File(s)", 
                                          command=self.select_files)
        self.select_files_btn.pack(fill='x', pady=(0, 15))
        
        # Output Folder button - fills available width
        self.output_folder_btn = ttk.Button(button_container, text="Output Folder\n(Same as input)", 
                                           command=self.select_output_folder)
        self.output_folder_btn.pack(fill='x')
        
    def create_speed_section(self, parent):
        """Create speed control section"""
        frame = ttk.LabelFrame(parent, text="Speed Control", padding="15")
        frame.grid(row=0, column=1, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Subtitle
        ttk.Label(frame, text="Quickie vs Less Mistakes",
         font=self.fonts['custom_subtitle'], foreground='gray').pack(pady=(0, 15))
        
        # Radio buttons with increased spacing
        ttk.Radiobutton(frame, text="Faster", variable=self.speed_var, 
                       value="faster").pack(anchor=tk.W, pady=8)
        ttk.Radiobutton(frame, text="Fast", variable=self.speed_var, 
                       value="fast").pack(anchor=tk.W, pady=8)
        ttk.Radiobutton(frame, text="Balanced", variable=self.speed_var, 
                       value="balanced").pack(anchor=tk.W, pady=8)
        
    def create_granularity_section(self, parent):
        """Create granularity control section"""
        frame = ttk.LabelFrame(parent, text="Granularity Control", padding="15")
        frame.grid(row=0, column=2, padx=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Subtitle                 
        ttk.Label(frame, text="Details vs. Fewer Guesses",
         font=self.fonts['custom_subtitle'], foreground='gray').pack(pady=(0, 15))

        
        # Radio buttons with increased spacing
        ttk.Radiobutton(frame, text="Aggressive", variable=self.granularity_var, 
                       value="aggressive").pack(anchor=tk.W, pady=8)
        ttk.Radiobutton(frame, text="Balanced", variable=self.granularity_var, 
                       value="balanced").pack(anchor=tk.W, pady=8)
        ttk.Radiobutton(frame, text="Conservative", variable=self.granularity_var, 
                       value="conservative").pack(anchor=tk.W, pady=8)
        
    def create_language_section(self, parent):
        """Create language selection section"""
        frame = ttk.LabelFrame(parent, text="SRT subs Language", padding="15")
        frame.grid(row=0, column=3, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Subtitle
        ttk.Label(frame, text="Whisper direct output:", 
                 font=self.fonts['custom_subtitle'], foreground='gray').pack(pady=(0, 15))
        
        # Radio buttons with increased spacing
        ttk.Radiobutton(frame, text="Japanese", variable=self.language_var, 
                       value="japanese").pack(anchor=tk.W, pady=8)
        ttk.Radiobutton(frame, text="English", variable=self.language_var, 
                       value="english-direct").pack(anchor=tk.W, pady=8)
        
        # Advanced Settings button with gear icon
        ttk.Separator(frame, orient='horizontal').pack(fill='x', pady=(20, 15))
        
        settings_frame = ttk.Frame(frame)
        settings_frame.pack(pady=(0, 10))
        
        # Gear symbol - even larger for visibility
        gear_label = ttk.Label(settings_frame, text="⚙", font=self.fonts['gear'])
        gear_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.settings_btn = ttk.Button(settings_frame, text="Advanced\nSettings", 
                                      command=self.show_advanced_settings)
        self.settings_btn.pack(side=tk.LEFT)
        
    def create_process_button(self, parent):
        """Create the start processing button"""
        button_frame = ttk.Frame(parent)
        button_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=15)
        
        # Make button fill more width
        button_container = ttk.Frame(button_frame)
        button_container.pack(expand=True, fill='x', padx=200)  # Add horizontal padding
        
        self.process_btn = ttk.Button(button_container, text="START PROCESSING", 
                                     command=self.toggle_processing,
                                     style='Process.TButton')
        self.process_btn.pack(fill='x')
        
        # Style for the button - larger font and padding
        style = ttk.Style()
        style.configure('Process.TButton', font=self.fonts['large_button'], padding=(20, 15))
        style.configure('Stop.TButton', font=self.fonts['large_button'], padding=(20, 15))
        


    def create_console_section(self, parent):
        """Create console output section"""
        console_frame = ttk.LabelFrame(parent, text="Console output / messages", padding="10")
        console_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)

        # Console text widget with scrollbar
        console_container = ttk.Frame(console_frame)
        console_container.pack(fill='both', expand=True)

        # Console with larger font for readability
        self.console_text = tk.Text(console_container, wrap='word',
                                   bg='black', fg='white',
                                   font=self.fonts['console'],
                                   insertbackground='white')  # Make cursor visible

        # Create a style for the scrollbar
        style = ttk.Style()
        style.configure('Custom.Vertical.TScrollbar', troughcolor='gray', bordercolor='gray', arrowsize=20, width=20)

        # Scrollbar with custom style
        scrollbar = ttk.Scrollbar(console_container, command=self.console_text.yview, style='Custom.Vertical.TScrollbar')

        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        console_container.columnconfigure(0, weight=1)
        console_container.rowconfigure(0, weight=1)

        self.console_text.configure(yscrollcommand=scrollbar.set)

        # Configure text tags for colored output
        self.console_text.tag_configure('info', foreground='white')
        self.console_text.tag_configure('warning', foreground='yellow')
        self.console_text.tag_configure('error', foreground='red')
        self.console_text.tag_configure('success', foreground='#90EE90')  # Light green
        self.console_text.tag_configure('output', foreground='#ADD8E6')  # Light blue


    '''
    def create_console_section(self, parent):
        """Create console output section"""
        console_frame = ttk.LabelFrame(parent, text="Console output / messages", padding="10")
        console_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        parent.columnconfigure(0, weight=1)
        
        # Console text widget with scrollbar
        console_container = ttk.Frame(console_frame)
        console_container.pack(fill='both', expand=True)
        
        # Console with larger font for readability
        self.console_text = tk.Text(console_container, wrap='word', 
                                   bg='black', fg='white', 
                                   font=self.fonts['console'],
                                   insertbackground='white')  # Make cursor visible
        
        # Larger scrollbar for easier use
        scrollbar = ttk.Scrollbar(console_container, command=self.console_text.yview, width=20)
        
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        console_container.columnconfigure(0, weight=1)
        console_container.rowconfigure(0, weight=1)
        
        self.console_text.configure(yscrollcommand=scrollbar.set)
        
        # Configure text tags for colored output
        self.console_text.tag_configure('info', foreground='white')
        self.console_text.tag_configure('warning', foreground='yellow')
        self.console_text.tag_configure('error', foreground='red')
        self.console_text.tag_configure('success', foreground='#90EE90')  # Light green
        self.console_text.tag_configure('output', foreground='#ADD8E6')  # Light blue
    '''

    
    def select_files(self):
        """Handle file selection"""
        files = filedialog.askopenfilenames(
            title="Select media files",
            filetypes=[
                ("All Media files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg *.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus"),
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v *.mpg *.mpeg"),
                ("Audio files", "*.mp3 *.wav *.flac *.aac *.ogg *.wma *.m4a *.opus"),
                ("All files", "*.*")
            ]
        )
        
        if files:
            self.selected_files = list(files)
            count = len(self.selected_files)
            self.select_files_btn.config(text=f"Select File(s)\n({count} selected)")
            
            # Set default output folder to first file's directory
            if not self.output_folder:
                self.output_folder = str(Path(self.selected_files[0]).parent)
                self.update_output_button_text()
                
            self.log(f"Selected {count} file(s)", 'info')
            
    def select_output_folder(self):
        """Handle output folder selection"""
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_folder = folder
            self.update_output_button_text()
            self.log(f"Output folder: {folder}", 'info')
            
    def update_output_button_text(self):
        """Update output folder button text"""
        if self.output_folder:
            folder_name = Path(self.output_folder).name
            if len(folder_name) > 20:
                folder_name = folder_name[:17] + "..."
            self.output_folder_btn.config(text=f"Output Folder\n({folder_name})")
        else:
            self.output_folder_btn.config(text="Output Folder\n(Same as input)")
            
    def show_advanced_settings(self):
        """Show advanced settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Advanced Settings")
        dialog.geometry("1000x800")  # Larger dialog
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Main container
        main_frame = ttk.Frame(dialog, padding="20")
        main_frame.pack(fill='both', expand=True)
        
        # Title
        ttk.Label(main_frame, text="Current Configuration Settings", 
                 font=self.fonts['heading']).pack(pady=(0, 15))
        
        # Info label
        ttk.Label(main_frame, text="These settings will be used for processing based on your selections:", 
                 foreground='gray', font=self.fonts['subtitle']).pack(pady=(0, 15))
        
        # Create notebook for tabs
        notebook = ttk.Notebook(main_frame)
        notebook.pack(fill='both', expand=True, pady=(0, 15))
        
        # Configure notebook font
        style = ttk.Style()
        style.configure('TNotebook.Tab', font=self.fonts['label'])
        
        # Get current settings based on selections
        mode = self.speed_var.get()
        sensitivity = self.granularity_var.get()
        
        # VAD Options tab
        vad_frame = ttk.Frame(notebook, padding="20")
        notebook.add(vad_frame, text="VAD Options")
        self.create_settings_display(vad_frame, self.get_vad_settings(mode, sensitivity))
        
        # Transcribe Options tab
        transcribe_frame = ttk.Frame(notebook, padding="20")
        notebook.add(transcribe_frame, text="Transcribe Options")
        self.create_settings_display(transcribe_frame, self.get_transcribe_settings(mode, sensitivity))
        
        # Decode Options tab
        decode_frame = ttk.Frame(notebook, padding="20")
        notebook.add(decode_frame, text="Decode Options")
        self.create_settings_display(decode_frame, self.get_decode_settings(mode, sensitivity))
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", command=dialog.destroy)
        close_btn.pack(pady=(10, 0))
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
    def create_settings_display(self, parent, settings):
        """Create a display of settings in the given frame"""
        # Create a scrollable frame
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview, width=20)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Display settings with larger font and spacing
        for key, value in settings.items():
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill='x', pady=8)
            
            ttk.Label(frame, text=f"{key}:", font=self.fonts['label']).pack(side='left')
            ttk.Label(frame, text=str(value), foreground='blue', 
                     font=self.fonts['label']).pack(side='left', padx=(20, 0))
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def get_vad_settings(self, mode, sensitivity):
        """Get VAD settings based on current selections"""
        base_settings = {
            "threshold": 0.5,
            "min_speech_duration_ms": 250,
            "max_speech_duration_s": float('inf'),
            "min_silence_duration_ms": 2000,
            "speech_pad_ms": 400
        }
        
        if mode == "faster":
            base_settings["threshold"] = 0.3
        elif mode == "balanced":
            base_settings["threshold"] = 0.5
            
        if sensitivity == "aggressive":
            base_settings["min_speech_duration_ms"] = 150
        elif sensitivity == "conservative":
            base_settings["min_speech_duration_ms"] = 350
            
        return base_settings
        
    def get_transcribe_settings(self, mode, sensitivity):
        """Get transcribe settings based on current selections"""
        base_settings = {
            "temperature": 0.0,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6
        }
        
        if sensitivity == "aggressive":
            base_settings["no_speech_threshold"] = 0.4
        elif sensitivity == "conservative":
            base_settings["no_speech_threshold"] = 0.8
            
        return base_settings
        
    def get_decode_settings(self, mode, sensitivity):
        """Get decode settings based on current selections"""
        base_settings = {
            "task": "transcribe",
            "language": "ja",
            "beam_size": 5,
            "best_of": 5,
            "patience": 1.0,
            "length_penalty": 1.0,
            "suppress_tokens": "-1"
        }
        
        if mode == "faster":
            base_settings["beam_size"] = 1
            base_settings["best_of"] = 1
        elif mode == "fast":
            base_settings["beam_size"] = 3
            
        return base_settings
        
    def toggle_processing(self):
        """Toggle between start and stop processing"""
        if not self.processing:
            self.start_processing()
        else:
            self.stop_processing()
            
    def start_processing(self):
        """Start processing files"""
        # Validate inputs
        if not self.selected_files:
            messagebox.showwarning("No Files", "Please select media files to process")
            return
            
        # Set output folder if not set
        if not self.output_folder:
            self.output_folder = str(Path(self.selected_files[0]).parent)
            
        # Update UI state
        self.processing = True
        self.process_btn.config(text="STOP PROCESSING", style='Stop.TButton')
        self.disable_controls()
        
        # Clear console
        self.console_text.delete('1.0', 'end')
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.process_files)
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def stop_processing(self):
        """Stop processing"""
        self.processing = False
        self.log("\nStopping processing...", 'warning')
        
    def disable_controls(self):
        """Disable controls during processing"""
        self.select_files_btn.config(state='disabled')
        self.output_folder_btn.config(state='disabled')
        self.settings_btn.config(state='disabled')
        
        # Find and disable all radio buttons in the top frame
        for widget in self.root.winfo_children():
            self._disable_radiobuttons_recursive(widget)
                
    def _disable_radiobuttons_recursive(self, widget):
        """Recursively disable radio buttons"""
        if isinstance(widget, ttk.Radiobutton):
            widget.config(state='disabled')
        for child in widget.winfo_children():
            self._disable_radiobuttons_recursive(child)
                
    def enable_controls(self):
        """Enable controls after processing"""
        self.select_files_btn.config(state='normal')
        self.output_folder_btn.config(state='normal')
        self.settings_btn.config(state='normal')
        
        # Find and enable all radio buttons
        for widget in self.root.winfo_children():
            self._enable_radiobuttons_recursive(widget)
                
    def _enable_radiobuttons_recursive(self, widget):
        """Recursively enable radio buttons"""
        if isinstance(widget, ttk.Radiobutton):
            widget.config(state='normal')
        for child in widget.winfo_children():
            self._enable_radiobuttons_recursive(child)
            
    def process_files(self):
        """Process files in separate thread"""
        try:
            self.log("Starting WhisperJAV processing...\n", 'info')
            
            # Build command - using whisperjav.main
            base_cmd = [
                sys.executable, "-m", "whisperjav.main",
                "--mode", self.speed_var.get(),
                "--sensitivity", self.granularity_var.get(),
                "--subs-language", self.language_var.get(),
                "--output-dir", self.output_folder
            ]
            
            # Process each file
            for i, file_path in enumerate(self.selected_files, 1):
                if not self.processing:
                    break
                    
                self.log(f"\n[{i}/{len(self.selected_files)}] Processing: {Path(file_path).name}", 'info')
                
                cmd = base_cmd + [file_path]
                
                try:
                    # Run WhisperJAV
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True
                    )
                    
                    # Read output line by line
                    for line in process.stdout:
                        if not self.processing:
                            process.terminate()
                            break
                        # Use 'output' tag for process output
                        self.log_queue.put(('output', line.rstrip()))
                        
                    process.wait()
                    
                    if process.returncode == 0:
                        self.log(f"✓ Successfully processed: {Path(file_path).name}", 'success')
                    else:
                        self.log(f"✗ Failed to process: {Path(file_path).name}", 'error')
                        
                except Exception as e:
                    self.log(f"Error processing {Path(file_path).name}: {str(e)}", 'error')
                    
            if self.processing:
                self.log("\n✅ All files processed!", 'success')
            else:
                self.log("\n⚠ Processing stopped by user", 'warning')
                
        except Exception as e:
            self.log(f"\nError: {str(e)}", 'error')
        finally:
            # Reset UI
            self.processing = False
            self.root.after(0, self.reset_ui_after_processing)
            
    def reset_ui_after_processing(self):
        """Reset UI after processing completes"""
        self.process_btn.config(text="START PROCESSING", style='Process.TButton')
        self.enable_controls()
        
    def log(self, message, level='info'):
        """Add message to console with appropriate styling"""
        self.log_queue.put((level, message))
        
    def process_log_queue(self):
        """Process messages from log queue"""
        try:
            while True:
                level, message = self.log_queue.get_nowait()
                
                # Insert message with appropriate tag
                self.console_text.insert('end', message + '\n', level)
                
                # Auto-scroll to bottom
                self.console_text.see('end')
                
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.process_log_queue)

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set DPI awareness on Windows
    if sys.platform == 'win32':
        try:
            from ctypes import windll
            windll.shcore.SetProcessDpiAwareness(1)
        except:
            pass
            
    app = WhisperJAVGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()