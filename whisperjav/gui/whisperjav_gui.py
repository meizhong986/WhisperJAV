import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import os
import threading
import sys
import queue
import subprocess
import json
from pathlib import Path
import tempfile
import re

class AdvancedSettingsDialog(Toplevel):
    def __init__(self, parent, config=None):
        super().__init__(parent)
        self.title("Advanced Settings")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()
        
        # Match parent window styling
        self.configure(bg='#f0f0f0')
        
        self.config = config or {}
        self.result = None
        
        # Create notebook for different setting categories
        style = ttk.Style()
        style.configure('Settings.TNotebook', background='#f0f0f0')
        style.configure('Settings.TFrame', background='#f0f0f0')
        
        notebook = ttk.Notebook(self, style='Settings.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # VAD Settings
        vad_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(vad_frame, text="VAD Settings")
        self.create_vad_settings(vad_frame)
        
        # Transcription Settings
        trans_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(trans_frame, text="Transcription")
        self.create_transcription_settings(trans_frame)
        
        # Decoder Settings
        dec_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(dec_frame, text="Decoder")
        self.create_decoder_settings(dec_frame)
        
        # Buttons
        btn_frame = tk.Frame(self, bg='#f0f0f0')
        btn_frame.pack(pady=10)
        
        save_btn = tk.Button(btn_frame, text="Save", command=self.save,
                            bg='#4a7abc', fg='white', font=('Arial', 10),
                            padx=20, pady=5, bd=0, relief=tk.FLAT,
                            activebackground='#3a6aac')
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.destroy,
                              bg='#e0e0e0', fg='#333333', font=('Arial', 10),
                              padx=20, pady=5, bd=0, relief=tk.FLAT,
                              activebackground='#d0d0d0')
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def create_vad_settings(self, parent):
        ttk.Label(parent, text="VAD Threshold:", background='#f0f0f0').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.vad_threshold = tk.DoubleVar(value=self.config.get("vad_threshold", 0.5))
        ttk.Scale(parent, from_=0.1, to=0.9, variable=self.vad_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=5)
        self.vad_label = ttk.Label(parent, text=f"{self.vad_threshold.get():.2f}", background='#f0f0f0')
        self.vad_label.grid(row=0, column=2)
        
        def update_vad_label(val):
            self.vad_label.config(text=f"{float(val):.2f}")
        self.vad_threshold.trace('w', lambda *args: update_vad_label(self.vad_threshold.get()))
        
        ttk.Label(parent, text="Min Speech Duration (ms):", background='#f0f0f0').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.min_speech_duration = tk.IntVar(value=self.config.get("min_speech_duration", 250))
        ttk.Entry(parent, textvariable=self.min_speech_duration, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Max Speech Duration (s):", background='#f0f0f0').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.max_speech_duration = tk.IntVar(value=self.config.get("max_speech_duration", 30))
        ttk.Entry(parent, textvariable=self.max_speech_duration, width=10).grid(row=2, column=1, sticky=tk.W)
    
    def create_transcription_settings(self, parent):
        ttk.Label(parent, text="Beam Size:", background='#f0f0f0').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.beam_size = tk.IntVar(value=self.config.get("beam_size", 5))
        ttk.Entry(parent, textvariable=self.beam_size, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Patience:", background='#f0f0f0').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.patience = tk.DoubleVar(value=self.config.get("patience", 1.0))
        ttk.Entry(parent, textvariable=self.patience, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Temperature:", background='#f0f0f0').grid(row=2, column=0, sticky=tk.W, pady=5)
        self.temperature = tk.DoubleVar(value=self.config.get("temperature", 0.0))
        ttk.Entry(parent, textvariable=self.temperature, width=10).grid(row=2, column=1, sticky=tk.W)
    
    def create_decoder_settings(self, parent):
        ttk.Label(parent, text="Word Threshold:", background='#f0f0f0').grid(row=0, column=0, sticky=tk.W, pady=5)
        self.word_threshold = tk.DoubleVar(value=self.config.get("word_threshold", 0.01))
        ttk.Scale(parent, from_=0.0, to=1.0, variable=self.word_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=5)
        self.word_label = ttk.Label(parent, text=f"{self.word_threshold.get():.2f}", background='#f0f0f0')
        self.word_label.grid(row=0, column=2)
        
        def update_word_label(val):
            self.word_label.config(text=f"{float(val):.2f}")
        self.word_threshold.trace('w', lambda *args: update_word_label(self.word_threshold.get()))
        
        ttk.Label(parent, text="Logprob Threshold:", background='#f0f0f0').grid(row=1, column=0, sticky=tk.W, pady=5)
        self.logprob_threshold = tk.DoubleVar(value=self.config.get("logprob_threshold", -1.0))
        ttk.Scale(parent, from_=-5.0, to=0.0, variable=self.logprob_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=1, column=1, padx=5)
        self.logprob_label = ttk.Label(parent, text=f"{self.logprob_threshold.get():.2f}", background='#f0f0f0')
        self.logprob_label.grid(row=1, column=2)
        
        def update_logprob_label(val):
            self.logprob_label.config(text=f"{float(val):.2f}")
        self.logprob_threshold.trace('w', lambda *args: update_logprob_label(self.logprob_threshold.get()))
    
    def save(self):
        self.result = {
            "vad_threshold": self.vad_threshold.get(),
            "min_speech_duration": self.min_speech_duration.get(),
            "max_speech_duration": self.max_speech_duration.get(),
            "beam_size": self.beam_size.get(),
            "patience": self.patience.get(),
            "temperature": self.temperature.get(),
            "word_threshold": self.word_threshold.get(),
            "logprob_threshold": self.logprob_threshold.get()
        }
        self.destroy()


class RadioButtonGroup(tk.Frame):
    def __init__(self, parent, options, variable, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(bg='white')
        self.variable = variable
        self.buttons = []
        
        for i, (text, value) in enumerate(options):
            btn = tk.Frame(self, bg='white', relief=tk.SOLID, bd=1)
            btn.grid(row=i, column=0, sticky='ew', padx=10, pady=5)
            
            # Create custom radio button appearance
            radio_frame = tk.Frame(btn, bg='white', cursor='hand2')
            radio_frame.pack(fill=tk.BOTH, expand=True)
            
            # Radio circle
            canvas = tk.Canvas(radio_frame, width=20, height=20, bg='white', highlightthickness=0)
            canvas.pack(side=tk.LEFT, padx=(15, 10), pady=10)
            
            # Outer circle
            outer_circle = canvas.create_oval(2, 2, 18, 18, outline='#cccccc', width=2)
            # Inner circle (hidden by default)
            inner_circle = canvas.create_oval(6, 6, 14, 14, fill='#4a7abc', outline='', state='hidden')
            
            # Label
            label = tk.Label(radio_frame, text=text, bg='white', font=('Arial', 11), anchor='w')
            label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 15))
            
            # Store button info
            button_info = {
                'frame': btn,
                'radio_frame': radio_frame,
                'canvas': canvas,
                'outer': outer_circle,
                'inner': inner_circle,
                'label': label,
                'value': value
            }
            self.buttons.append(button_info)
            
            # Bind click events
            for widget in [radio_frame, canvas, label]:
                widget.bind('<Button-1>', lambda e, v=value: self.select(v))
            
            # Hover effects
            def on_enter(e, btn_info=button_info):
                if self.variable.get() != btn_info['value']:
                    btn_info['frame'].configure(bg='#f5f5f5')
                    btn_info['radio_frame'].configure(bg='#f5f5f5')
                    btn_info['label'].configure(bg='#f5f5f5')
                    btn_info['canvas'].configure(bg='#f5f5f5')
            
            def on_leave(e, btn_info=button_info):
                if self.variable.get() != btn_info['value']:
                    btn_info['frame'].configure(bg='white')
                    btn_info['radio_frame'].configure(bg='white')
                    btn_info['label'].configure(bg='white')
                    btn_info['canvas'].configure(bg='white')
            
            for widget in [radio_frame, canvas, label]:
                widget.bind('<Enter>', on_enter)
                widget.bind('<Leave>', on_leave)
        
        # Set initial selection
        self.select(self.variable.get())
        
        # Configure grid weights
        self.grid_columnconfigure(0, weight=1)
    
    def select(self, value):
        self.variable.set(value)
        for btn in self.buttons:
            if btn['value'] == value:
                # Selected state
                btn['frame'].configure(bg='#e8f0fe')
                btn['radio_frame'].configure(bg='#e8f0fe')
                btn['label'].configure(bg='#e8f0fe', fg='#1967d2')
                btn['canvas'].configure(bg='#e8f0fe')
                btn['canvas'].itemconfig(btn['outer'], outline='#4a7abc')
                btn['canvas'].itemconfig(btn['inner'], state='normal')
            else:
                # Unselected state
                btn['frame'].configure(bg='white')
                btn['radio_frame'].configure(bg='white')
                btn['label'].configure(bg='white', fg='black')
                btn['canvas'].configure(bg='white')
                btn['canvas'].itemconfig(btn['outer'], outline='#cccccc')
                btn['canvas'].itemconfig(btn['inner'], state='hidden')


class WhisperJAVGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhisperJAV - Japanese Adult Video Subtitle Generator")
        self.geometry("900x900")
        self.resizable(True, True)
        self.configure(bg='#f0f0f0')
        
        # Variables
        self.input_files = []
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value="balanced")
        self.sensitivity = tk.StringVar(value="balanced")
        self.subs_language = tk.StringVar(value="japanese")
        self.adaptive_classification = tk.BooleanVar()
        self.adaptive_audio_enhancement = tk.BooleanVar()
        self.smart_postprocessing = tk.BooleanVar()
        self.show_console = tk.BooleanVar(value=True)
        self.processing = False
        self.output_queue = queue.Queue()
        self.advanced_config = {}
        
        self.current_file_index = 0
        self.total_files = 0
        self.progress_line_ids = {}
        self.last_progress_text = ""
        
        
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_widgets(self):
        # Main container
        main_container = tk.Frame(self, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Top section with 4 columns
        top_section = tk.Frame(main_container, bg='#f0f0f0')
        top_section.pack(fill=tk.X, pady=(0, 15))
        
        # Configure grid columns with equal weight
        for i in range(4):
            top_section.grid_columnconfigure(i, weight=1, uniform="column")
        
        # Section 1: Choose Media and Output Destination
        media_frame = tk.Frame(top_section, bg='white', relief=tk.SOLID, bd=1)
        media_frame.grid(row=0, column=0, sticky='nsew', padx=(0, 10))
        
        media_title = tk.Label(media_frame, text="Choose Media and\nOutput Destination", 
                              bg='white', font=('Arial', 11, 'bold'), justify=tk.CENTER)
        media_title.pack(pady=(15, 10))
        
        # File selection button
        self.file_button = tk.Button(media_frame, text="Select File(s)", 
                                    command=self.select_files,
                                    bg='#f8f9fa', fg='#333333', font=('Arial', 10),
                                    padx=20, pady=8, bd=1, relief=tk.SOLID,
                                    activebackground='#e9ecef', cursor='hand2')
        self.file_button.pack(padx=20, pady=(0, 10))
        
        # File label
        self.file_label = tk.Label(media_frame, text="No files selected", 
                                  bg='white', font=('Arial', 9), fg='#666666')
        self.file_label.pack(padx=20, pady=(0, 10))
        
        # Output folder button
        self.output_button = tk.Button(media_frame, text="Output Folder", 
                                      command=self.select_output_dir,
                                      bg='#f8f9fa', fg='#333333', font=('Arial', 10),
                                      padx=20, pady=8, bd=1, relief=tk.SOLID,
                                      activebackground='#e9ecef', cursor='hand2')
        self.output_button.pack(padx=20, pady=(0, 15))
        
        # Section 2: Speed Control
        speed_frame = tk.Frame(top_section, bg='white', relief=tk.SOLID, bd=1)
        speed_frame.grid(row=0, column=1, sticky='nsew', padx=(0, 10))
        
        speed_title = tk.Label(speed_frame, text="Speed Control\nQuickie vs Less Mistakes", 
                              bg='white', font=('Arial', 11, 'bold'), justify=tk.CENTER)
        speed_title.pack(pady=(15, 15))
        
        # Speed radio buttons
        speed_options = [
            ("Faster", "faster"),
            ("Fast", "fast"),
            ("Balanced", "balanced")
        ]
        self.speed_radio = RadioButtonGroup(speed_frame, speed_options, self.mode)
        self.speed_radio.pack(fill=tk.BOTH, expand=True)
        
        # Section 3: Granularity Control
        granularity_frame = tk.Frame(top_section, bg='white', relief=tk.SOLID, bd=1)
        granularity_frame.grid(row=0, column=2, sticky='nsew', padx=(0, 10))
        
        granularity_title = tk.Label(granularity_frame, text="Granularity Control\nDetails vs. Fewer Guesses", 
                                    bg='white', font=('Arial', 11, 'bold'), justify=tk.CENTER)
        granularity_title.pack(pady=(15, 15))
        
        # Granularity radio buttons
        granularity_options = [
            ("Aggressive", "aggressive"),
            ("Balanced", "balanced"),
            ("Conservative", "conservative")
        ]
        self.granularity_radio = RadioButtonGroup(granularity_frame, granularity_options, self.sensitivity)
        self.granularity_radio.pack(fill=tk.BOTH, expand=True)
        
        # Section 4: SRT subs Language
        language_frame = tk.Frame(top_section, bg='white', relief=tk.SOLID, bd=1)
        language_frame.grid(row=0, column=3, sticky='nsew')
        
        language_title = tk.Label(language_frame, text="SRT subs Language\nDirect into:", 
                                 bg='white', font=('Arial', 11, 'bold'), justify=tk.CENTER)
        language_title.pack(pady=(15, 15))
        
        # Language radio buttons
        language_options = [
            ("Japanese", "japanese"),
            ("English", "english-direct")
        ]
        self.language_radio = RadioButtonGroup(language_frame, language_options, self.subs_language)
        self.language_radio.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Advanced settings at bottom of language section
        tk.Frame(language_frame, height=1, bg='#e0e0e0').pack(fill=tk.X, padx=20)
        
        advanced_btn = tk.Button(language_frame, text="Advanced\nSettings ⚙", 
                                command=self.open_advanced_settings,
                                bg='white', fg='#666666', font=('Arial', 10),
                                bd=0, relief=tk.FLAT, justify=tk.CENTER,
                                activebackground='#f0f0f0', cursor='hand2')
        advanced_btn.pack(pady=15)
        
        # Enhancement options below main sections
        enh_frame = tk.Frame(main_container, bg='#f0f0f0')
        enh_frame.pack(fill=tk.X, pady=(10, 15))
        
        tk.Checkbutton(enh_frame, text="Adaptive Classification", 
                      variable=self.adaptive_classification,
                      bg='#f0f0f0', font=('Arial', 10),
                      activebackground='#f0f0f0').pack(side=tk.LEFT, padx=(0, 20))
        tk.Checkbutton(enh_frame, text="Adaptive Audio Enhancement", 
                      variable=self.adaptive_audio_enhancement,
                      bg='#f0f0f0', font=('Arial', 10),
                      activebackground='#f0f0f0').pack(side=tk.LEFT, padx=(0, 20))
        tk.Checkbutton(enh_frame, text="Smart Postprocessing", 
                      variable=self.smart_postprocessing,
                      bg='#f0f0f0', font=('Arial', 10),
                      activebackground='#f0f0f0').pack(side=tk.LEFT)
        
        # Start button
        self.start_button = tk.Button(main_container, text="START PROCESSING", 
                                     command=self.start_processing,
                                     bg='#4a7abc', fg='white', font=('Arial', 12, 'bold'),
                                     padx=40, pady=12, bd=0, relief=tk.FLAT,
                                     activebackground='#3a6aac', cursor='hand2')
        self.start_button.pack(pady=(0, 15))
        
        # Progress bar frame (hidden initially)
        self.progress_frame = tk.Frame(main_container, bg='#f0f0f0')
        self.progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = tk.Label(self.progress_frame, text="", bg='#f0f0f0', font=('Arial', 9))
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='indeterminate', length=400)
        # Don't pack the progress bar initially
        
        # Console output
        console_frame = tk.Frame(main_container, bg='white', relief=tk.SOLID, bd=1)
        console_frame.pack(fill=tk.BOTH, expand=True)
        
        console_header = tk.Frame(console_frame, bg='#f8f9fa')
        console_header.pack(fill=tk.X)
        
        tk.Label(console_header, text="Console output / messages", 
                bg='#f8f9fa', font=('Arial', 11), fg='#333333').pack(pady=10)
        
        # Console text area
        self.console = scrolledtext.ScrolledText(console_frame, height=27, 
                                                bg='white', font=('Consolas', 9),
                                                wrap=tk.WORD, state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True, padx=1, pady=(0, 1))
        
        # Configure console text tags for better formatting
        self.console.tag_configure("progress", foreground="#007bff", font=('Consolas', 9, 'bold'))
        self.console.tag_configure("banner", font=('Consolas', 9, 'bold'))
        self.console.tag_configure("error", foreground="#dc3545")
        self.console.tag_configure("success", foreground="#28a745")
        self.console.tag_configure("step", foreground="#17a2b8", font=('Consolas', 9, 'italic'))
        self.console.tag_configure("current_file", foreground="#6610f2", font=('Consolas', 9, 'bold'))
        self.console.tag_configure("summary", foreground="#343a40", font=('Consolas', 9, 'bold'))
        
    def clean_console_output(self, text):
        """Clean ANSI escape sequences and format output for GUI console."""
        # Remove ANSI escape sequences
        ansi_escape = re.compile(r'''
            \x1B  # ESC
            (?:   # 7-bit C1 Fe (except CSI)
                [@-Z\\-_]
            |     # or [ for CSI, followed by parameter bytes
                \[
                [0-?]*  # Parameter bytes
                [ -/]*  # Intermediate bytes
                [@-~]   # Final byte
            )
        ''', re.VERBOSE)
        text = ansi_escape.sub('', text)
        
        # Remove carriage returns and clean up
        text = text.replace('\r', '')
        
        
        
        
        # Preserve progress bar artifacts for in-place updates
        if any(term in text for term in ["Progress:", "Overall Progress:", "Processing regions:"]):
            return text
        
        
        # Remove multiple consecutive newlines
        text = re.sub(r'\n\n+', '\n\n', text)
        
        # Clean up progress bar artifacts
        text = re.sub(r'^\s*:\s*\|\s*\|\s*\d+/\d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Overall Progress:.*\[.*\].*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Processing regions:.*$', '', text, flags=re.MULTILINE)
        
        # Format step indicators nicely
        text = re.sub(r'^\[(\d+)/(\d+)\] Current: (.+?):(.+?)$', 
                      r'[\1/\2] Processing: \3\n    \4', text, flags=re.MULTILINE)
        
        # Highlight OK messages
        text = re.sub(r'\[OK\] \[OK\]', '[✓ OK]', text)
        
        # Clean up empty lines
        lines = text.split('\n')
        cleaned_lines = [line for line in lines if line.strip()]
        
        return '\n'.join(cleaned_lines)
    
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=[("Media Files", "*.mp4 *.mp3 *.mkv *.wav *.wmv *.flac  *.m4a *.mpeg *.mpg *.ts *.avi"), ("All Files", "*.*")]
        )
        if files:
            self.input_files = list(files)
            num_files = len(self.input_files)
            file_text = f"{num_files} file{'s' if num_files > 1 else ''} selected"
            
            # Set output directory to first file's directory
            if num_files > 0:
                first_file_dir = os.path.dirname(self.input_files[0])
                self.output_dir.set(first_file_dir)
                # Update button text to show folder
                folder_name = os.path.basename(first_file_dir) or "Root"
                self.output_button.config(text=f"📁 {folder_name}")
            
            self.file_label.config(text=file_text, fg='#28a745')
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
            folder_name = os.path.basename(directory) or "Root"
            self.output_button.config(text=f"📁 {folder_name}")
    
    def open_advanced_settings(self):
        dialog = AdvancedSettingsDialog(self, self.advanced_config)
        self.wait_window(dialog)
        
        if dialog.result:
            self.advanced_config = dialog.result
            self.write_to_console("Advanced settings updated")
    
    def get_progress_type(self, line):
        """Identify the type of progress line"""
        if "Transcribing scenes" in line:
            return "scenes"
        elif "Overall Progress" in line:
            return "overall"
        return None



    def write_to_console(self, message):
        """Write cleaned message to console with appropriate formatting."""
        self.console.config(state=tk.NORMAL)
        
        # Identify progress bar type
        progress_type = self.get_progress_type(message)
        
        if progress_type:
            # Handle progress update in-place
            if progress_type in self.progress_line_ids:
                # Get the line index for this progress type
                line_index = self.progress_line_ids[progress_type]
                # Delete the old line
                self.console.delete(f"{line_index}.0", f"{line_index}.end+1c")
                # Insert the updated line at the same position
                self.console.insert(f"{line_index}.0", message + "\n", "progress")
            else:
                # First time seeing this progress type, add it
                # Get current line number
                current_line = int(self.console.index('end-1c').split('.')[0])
                self.progress_line_ids[progress_type] = current_line
                self.console.insert(tk.END, message + "\n", "progress")
            
            self.last_progress_text = message
        else:
            # Skip duplicate consecutive messages
            try:
                last_line = self.console.get("end-2l", "end-1l").strip()
                if last_line == message.strip():
                    self.console.config(state=tk.DISABLED)
                    return
            except:
                pass
            
            # Apply formatting based on content
            if "╔" in message or "║" in message or "╚" in message:
                self.console.insert(tk.END, message + "\n", "banner")
            elif "[✓ OK]" in message or "successful" in message.lower():
                self.console.insert(tk.END, message + "\n", "success")
            elif "error" in message.lower() or "failed" in message.lower():
                self.console.insert(tk.END, message + "\n", "error")
            elif "Step" in message and "/" in message:
                self.console.insert(tk.END, message + "\n", "step")
            elif "Processing:" in message:
                self.console.insert(tk.END, message + "\n", "current_file")
            elif "PROCESSING SUMMARY" in message or "=" * 10 in message:
                self.console.insert(tk.END, message + "\n", "summary")
            else:
                self.console.insert(tk.END, message + "\n")
            
            # When adding a regular line, reset progress tracking
            self.progress_line_ids = {}
        
        # Limit console to 500 lines to prevent performance issues
        line_count = int(self.console.index('end-1c').split('.')[0])
        if line_count > 500:
            # Remove oldest 100 lines
            self.console.delete('1.0', '100.0')
            # Update progress line positions
            self.update_progress_line_ids()
        
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
    
    def update_progress_line_ids(self):
        """Update progress line positions after console truncation"""
        new_ids = {}
        for progress_type, old_id in self.progress_line_ids.items():
            try:
                # Convert index to line number
                line_num = int(old_id.split('.')[0])
                if line_num > 100:
                    new_line_num = line_num - 100
                    new_ids[progress_type] = f"{new_line_num}.0"
            except:
                pass
        self.progress_line_ids = new_ids
    
    def start_processing(self):
        if not self.input_files:
            messagebox.showerror("Input Error", "Please select at least one media file.")
            return
        
        output_dir = self.output_dir.get()
        if not output_dir:
            messagebox.showerror("Output Error", "Please select an output directory.")
            return
            
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                messagebox.showerror("Output Error", "Could not create output directory.")
                return

        # Reset progress tracking
        self.current_file_index = 0
        self.total_files = len(self.input_files)
        self.progress_line_ids = {}  # Track multiple progress bars
        self.last_progress_text = ""
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED, bg='#cccccc')
        self.write_to_console("Starting processing...")
        
        # Show progress bar
        self.progress_label.config(text="Processing files...")
        self.progress_bar.pack(pady=5)
        self.progress_bar.start(10)
        
        # Create temporary config file if advanced settings were modified
        config_path = None
        if self.advanced_config:
            try:
                config_path = Path(tempfile.gettempdir()) / "whisperjav_gui_config.json"
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(self.advanced_config, f, indent=2)
                self.write_to_console(f"Using custom config: {config_path}")
            except Exception as e:
                self.write_to_console(f"Error saving config: {str(e)}")
        
        # Build command
        command = [
            sys.executable, "-m", "whisperjav.main",
            "--output-dir", output_dir,
            "--mode", self.mode.get(),
            "--sensitivity", self.sensitivity.get(),
            "--subs-language", self.subs_language.get()
        ]

        
        # Add config file if created
        if config_path:
            command.extend(["--config", str(config_path)])
        
        # Add enhancement options
        if self.adaptive_classification.get():
            command.append("--adaptive-classification")
        if self.adaptive_audio_enhancement.get():
            command.append("--adaptive-audio-enhancement")
        if self.smart_postprocessing.get():
            command.append("--smart-postprocessing")
        
        # Add input files
        command.extend(self.input_files)
        
        # Start processing in a separate thread
        threading.Thread(target=self.run_processing, args=(command,), daemon=True).start()
    
    def run_processing(self, command):
        try:
            self.write_to_console(f"Running command: {' '.join(command)}")
            
            # Set environment to ensure UTF-8 encoding
            env = os.environ.copy()
            env['PYTHONIOENCODING'] = 'utf-8'
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='strict',  # Changed from 'replace' to 'strict'
                env=env  # Pass the environment with UTF-8 encoding
            )
            
            # Capture output in real-time
            for line in process.stdout:
                self.output_queue.put(line)
                self.update_console()
            
            process.wait()
            return_code = process.returncode
            
            self.output_queue.put(f"\nProcess completed with exit code: {return_code}")
            self.update_console()
            
        except Exception as e:
            self.output_queue.put(f"Error during processing: {str(e)}")
            self.update_console()
        finally:
            self.output_queue.put(None)  # Signal processing complete
            self.after(100, self.processing_complete)
    
    def update_console(self):
        """Update console with batched output from queue"""
        batch = []
        try:
            # Process up to 20 lines per update cycle
            for _ in range(20):
                line = self.output_queue.get_nowait()
                if line is None:
                    self.processing_complete()
                    return
                
                # Preserve progress lines for special handling
                cleaned_line = self.clean_console_output(line)
                if cleaned_line:  # Only write non-empty lines
                    batch.append(cleaned_line)
        except queue.Empty:
            pass
        
        if batch:
            for line in batch:
                self.write_to_console(line)
        
        if self.processing:
            self.after(100, self.update_console)  # Maintain update frequency
    
    def processing_complete(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL, bg='#4a7abc')
        
        # Hide progress bar
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        
        self.write_to_console("Processing complete.")
    
    def on_close(self):
        if self.processing:
            if messagebox.askokcancel("Quit", "Processing is still running. Quit anyway?"):
                self.destroy()
        else:
            self.destroy()
            
            
def main():
    """Main entry point for GUI."""
    app = WhisperJAVGUI()
    app.mainloop()

if __name__ == "__main__":
    main()