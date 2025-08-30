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
import time

from whisperjav.utils.async_processor import AsyncPipelineManager, ProcessingStatus
from whisperjav.config.transcription_tuner import TranscriptionTuner
from whisperjav.config.manager import ConfigManager, quick_update_ui_preference
from whisperjav.utils.unified_progress import UnifiedProgressManager, VerbosityLevel as UnifiedVerbosityLevel
from whisperjav.utils.progress_adapter import ProgressDisplayAdapter
from whisperjav.utils.logger import logger


from whisperjav.utils.preflight_check import enforce_cuda_requirement
# Enforce CUDA requirement before initializing any GUI components
enforce_cuda_requirement()



class AdvancedSettingsDialog(Toplevel):
    def __init__(self, parent, config_manager: ConfigManager, current_sensitivity: str):
        super().__init__(parent)
        self.title("Advanced Settings")
        self.geometry("600x500")
        self.transient(parent)
        self.grab_set()
        
        self.parent = parent
        self.config_manager = config_manager
        self.current_sensitivity = current_sensitivity
        self.result = None
        
        # Configure styling to match parent
        self.configure(bg='#f0f0f0')
        
        # Load current parameter sets from configuration
        self.load_current_settings()
        
        # Create UI
        self.create_widgets()
        
        # Center dialog on parent
        self.center_on_parent()
    
    def load_current_settings(self):
        """Load current parameter sets from the configuration system."""
        try:
            # Get parameter sets for current sensitivity
            self.decoder_params = self.config_manager.get_parameter_set(
                'common_decoder_options', self.current_sensitivity
            )
            self.transcriber_params = self.config_manager.get_parameter_set(
                'common_transcriber_options', self.current_sensitivity
            )
            # Try to get VAD params, fallback to defaults if not available
            try:
                self.vad_params = self.config_manager.get_parameter_set(
                    'silero_vad_options', self.current_sensitivity
                )
            except:
                # Fallback to default VAD parameters
                self.vad_params = {
                    'threshold': 0.5,
                    'min_speech_duration_ms': 250,
                    'max_speech_duration_s': 30
                }
            
            logger.debug(f"Loaded settings for sensitivity: {self.current_sensitivity}")
            
        except Exception as e:
            logger.error(f"Failed to load current settings: {e}")
            # Fallback to defaults
            self.decoder_params = {}
            self.transcriber_params = {}
            self.vad_params = {}
    
    def create_widgets(self):
        """Create the advanced settings UI with real configuration values."""
        # Create notebook for settings categories
        style = ttk.Style()
        style.configure('Settings.TNotebook', background='#f0f0f0')
        style.configure('Settings.TFrame', background='#f0f0f0')
        
        notebook = ttk.Notebook(self, style='Settings.TNotebook')
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Decoder Settings Tab
        decoder_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(decoder_frame, text="Decoder Options")
        self.create_decoder_settings(decoder_frame)
        
        # Transcriber Settings Tab
        transcriber_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(transcriber_frame, text="Transcriber Options")
        self.create_transcriber_settings(transcriber_frame)
        
        # VAD Settings Tab  
        vad_frame = ttk.Frame(notebook, style='Settings.TFrame', padding=20)
        notebook.add(vad_frame, text="VAD Options")
        self.create_vad_settings(vad_frame)
        
        # Buttons
        self.create_button_frame()
    
    def create_decoder_settings(self, parent):
        """Create decoder settings widgets using actual config values."""
        row = 0
        
        # Beam Size
        if 'beam_size' in self.decoder_params:
            ttk.Label(parent, text="Beam Size:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.beam_size = tk.IntVar(value=self.decoder_params.get('beam_size', 5))
            ttk.Entry(parent, textvariable=self.beam_size, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(1-10)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
        
        # Temperature
        if 'temperature' in self.decoder_params:
            ttk.Label(parent, text="Temperature:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            temp_val = self.decoder_params.get('temperature', 0.0)
            if isinstance(temp_val, list):
                temp_val = temp_val[0] if temp_val else 0.0
            self.temperature = tk.DoubleVar(value=float(temp_val))
            ttk.Entry(parent, textvariable=self.temperature, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(0.0-2.0)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
        
        # Patience
        if 'patience' in self.decoder_params:
            ttk.Label(parent, text="Patience:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.patience = tk.DoubleVar(value=self.decoder_params.get('patience', 1.0))
            ttk.Entry(parent, textvariable=self.patience, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(0.1-10.0)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
    
    def create_transcriber_settings(self, parent):
        """Create transcriber settings widgets using actual config values."""
        row = 0
        
        # Compression Ratio Threshold
        if 'compression_ratio_threshold' in self.transcriber_params:
            ttk.Label(parent, text="Compression Ratio Threshold:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.compression_ratio_threshold = tk.DoubleVar(value=self.transcriber_params.get('compression_ratio_threshold', 2.4))
            ttk.Entry(parent, textvariable=self.compression_ratio_threshold, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(1.0-10.0)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
        
        # Logprob Threshold
        if 'logprob_threshold' in self.transcriber_params:
            ttk.Label(parent, text="Logprob Threshold:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.logprob_threshold = tk.DoubleVar(value=self.transcriber_params.get('logprob_threshold', -1.0))
            ttk.Entry(parent, textvariable=self.logprob_threshold, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(-10.0 to 0.0)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
        
        # No Speech Threshold
        if 'no_speech_threshold' in self.transcriber_params:
            ttk.Label(parent, text="No Speech Threshold:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.no_speech_threshold = tk.DoubleVar(value=self.transcriber_params.get('no_speech_threshold', 0.6))
            ttk.Entry(parent, textvariable=self.no_speech_threshold, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(0.0-1.0)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
    
    def create_vad_settings(self, parent):
        """Create VAD settings widgets using actual config values."""
        row = 0
        
        # VAD Threshold
        if 'threshold' in self.vad_params:
            ttk.Label(parent, text="VAD Threshold:", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.vad_threshold = tk.DoubleVar(value=self.vad_params.get('threshold', 0.5))
            ttk.Scale(parent, from_=0.0, to=1.0, variable=self.vad_threshold, 
                     orient=tk.HORIZONTAL, length=300).grid(row=row, column=1, padx=10)
            self.vad_threshold_label = ttk.Label(parent, text=f"{self.vad_threshold.get():.2f}", background='#f0f0f0')
            self.vad_threshold_label.grid(row=row, column=2)
            
            def update_vad_label(val):
                self.vad_threshold_label.config(text=f"{float(val):.2f}")
            self.vad_threshold.trace('w', lambda *args: update_vad_label(self.vad_threshold.get()))
            row += 1
        
        # Min Speech Duration
        if 'min_speech_duration_ms' in self.vad_params:
            ttk.Label(parent, text="Min Speech Duration (ms):", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.min_speech_duration_ms = tk.IntVar(value=self.vad_params.get('min_speech_duration_ms', 250))
            ttk.Entry(parent, textvariable=self.min_speech_duration_ms, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(0-5000)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
        
        # Max Speech Duration
        if 'max_speech_duration_s' in self.vad_params:
            ttk.Label(parent, text="Max Speech Duration (s):", background='#f0f0f0').grid(row=row, column=0, sticky=tk.W, pady=5)
            self.max_speech_duration_s = tk.IntVar(value=self.vad_params.get('max_speech_duration_s', 30))
            ttk.Entry(parent, textvariable=self.max_speech_duration_s, width=10).grid(row=row, column=1, sticky=tk.W, padx=10)
            ttk.Label(parent, text="(1-300)", background='#f0f0f0', foreground='gray').grid(row=row, column=2, sticky=tk.W)
            row += 1
    
    def create_button_frame(self):
        """Create save/cancel buttons."""
        btn_frame = tk.Frame(self, bg='#f0f0f0')
        btn_frame.pack(pady=20)
        
        save_btn = tk.Button(btn_frame, text="Save Changes", command=self.save_settings,
                            bg='#4a7abc', fg='white', font=('Arial', 11),
                            padx=30, pady=8, bd=0, relief=tk.FLAT,
                            activebackground='#3a6aac')
        save_btn.pack(side=tk.LEFT, padx=10)
        
        cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.cancel,
                              bg='#e0e0e0', fg='#333333', font=('Arial', 11),
                              padx=30, pady=8, bd=0, relief=tk.FLAT,
                              activebackground='#d0d0d0')
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        reset_btn = tk.Button(btn_frame, text="Reset to Defaults", command=self.reset_to_defaults,
                             bg='#ffc107', fg='#333333', font=('Arial', 11),
                             padx=30, pady=8, bd=0, relief=tk.FLAT,
                             activebackground='#ffcd39')
        reset_btn.pack(side=tk.LEFT, padx=10)
    
    def save_settings(self):
        """Save current settings back to the configuration system."""
        try:
            # Collect updated decoder parameters
            updated_decoder = self.decoder_params.copy()
            if hasattr(self, 'beam_size'):
                updated_decoder['beam_size'] = self.beam_size.get()
            if hasattr(self, 'temperature'):
                updated_decoder['temperature'] = self.temperature.get()
            if hasattr(self, 'patience'):
                updated_decoder['patience'] = self.patience.get()
            
            # Collect updated transcriber parameters  
            updated_transcriber = self.transcriber_params.copy()
            if hasattr(self, 'compression_ratio_threshold'):
                updated_transcriber['compression_ratio_threshold'] = self.compression_ratio_threshold.get()
            if hasattr(self, 'logprob_threshold'):
                updated_transcriber['logprob_threshold'] = self.logprob_threshold.get()
            if hasattr(self, 'no_speech_threshold'):
                updated_transcriber['no_speech_threshold'] = self.no_speech_threshold.get()
            
            # Collect updated VAD parameters
            updated_vad = self.vad_params.copy()
            if hasattr(self, 'vad_threshold'):
                updated_vad['threshold'] = self.vad_threshold.get()
            if hasattr(self, 'min_speech_duration_ms'):
                updated_vad['min_speech_duration_ms'] = self.min_speech_duration_ms.get()
            if hasattr(self, 'max_speech_duration_s'):
                updated_vad['max_speech_duration_s'] = self.max_speech_duration_s.get()
            
            # Update configuration through ConfigManager
            self.config_manager.update_parameter_set('common_decoder_options', self.current_sensitivity, updated_decoder)
            self.config_manager.update_parameter_set('common_transcriber_options', self.current_sensitivity, updated_transcriber)
            if updated_vad:  # Only update if we have VAD params
                self.config_manager.update_parameter_set('silero_vad_options', self.current_sensitivity, updated_vad)
            
            # Save configuration to file
            self.config_manager.save_config()
            
            logger.info(f"Saved advanced settings for sensitivity: {self.current_sensitivity}")
            self.result = True
            self.destroy()
            
        except Exception as e:
            logger.error(f"Failed to save advanced settings: {e}", exc_info=True)
            messagebox.showerror("Save Error", f"Failed to save settings:\n\n{str(e)}")
    
    def reset_to_defaults(self):
        """Reset current sensitivity profile to default values."""
        try:
            # Confirm reset action
            if messagebox.askyesno("Reset to Defaults", 
                                 f"Reset all settings for '{self.current_sensitivity}' sensitivity to defaults?\n\nThis cannot be undone."):
                
                # Get default parameter sets (use 'balanced' as baseline)
                default_decoder = self.config_manager.get_parameter_set('common_decoder_options', 'balanced')
                default_transcriber = self.config_manager.get_parameter_set('common_transcriber_options', 'balanced')
                try:
                    default_vad = self.config_manager.get_parameter_set('silero_vad_options', 'balanced')
                except:
                    default_vad = {'threshold': 0.5, 'min_speech_duration_ms': 250, 'max_speech_duration_s': 30}
                
                # Update configuration
                self.config_manager.update_parameter_set('common_decoder_options', self.current_sensitivity, default_decoder)
                self.config_manager.update_parameter_set('common_transcriber_options', self.current_sensitivity, default_transcriber)
                if default_vad:
                    self.config_manager.update_parameter_set('silero_vad_options', self.current_sensitivity, default_vad)
                self.config_manager.save_config()
                
                # Reload the dialog
                self.load_current_settings()
                self.destroy()
                
                # Reopen dialog with updated settings
                new_dialog = AdvancedSettingsDialog(self.parent, self.config_manager, self.current_sensitivity)
                self.parent.wait_window(new_dialog)
                
        except Exception as e:
            logger.error(f"Failed to reset settings: {e}", exc_info=True)
            messagebox.showerror("Reset Error", f"Failed to reset settings:\n\n{str(e)}")
    
    def cancel(self):
        """Cancel without saving."""
        self.result = False
        self.destroy()
    
    def center_on_parent(self):
        """Center dialog on parent window."""
        self.update_idletasks()
        parent_x = self.parent.winfo_x()
        parent_y = self.parent.winfo_y()
        parent_width = self.parent.winfo_width()
        parent_height = self.parent.winfo_height()
        
        dialog_width = self.winfo_width()
        dialog_height = self.winfo_height()
        
        x = parent_x + (parent_width - dialog_width) // 2
        y = parent_y + (parent_height - dialog_height) // 2
        
        self.geometry(f"{dialog_width}x{dialog_height}+{x}+{y}")


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
        
        # --- Initialize v4.3 Configuration System ---
        self.config_manager = ConfigManager()
        self.transcription_tuner = TranscriptionTuner()
        self.config_path = None
        
        # Load UI preferences from configuration
        self.ui_preferences = self.config_manager.get_ui_preferences()
        
        # --- State Variables (update with config defaults) ---
        self.input_files = []
        self.output_dir = tk.StringVar()
        self.mode = tk.StringVar(value=self.ui_preferences.get('last_mode', 'balanced'))
        self.sensitivity = tk.StringVar(value=self.ui_preferences.get('last_sensitivity', 'balanced'))
        self.subs_language = tk.StringVar(value=self.ui_preferences.get('last_language', 'japanese'))
        
        # Enhancement options (load from UI preferences)
        self.adaptive_classification = tk.BooleanVar(value=self.ui_preferences.get('adaptive_classification', False))
        self.adaptive_audio_enhancement = tk.BooleanVar(value=self.ui_preferences.get('adaptive_audio_enhancement', False))
        self.smart_postprocessing = tk.BooleanVar(value=self.ui_preferences.get('smart_postprocessing', False))
        self.show_console = tk.BooleanVar(value=self.ui_preferences.get('show_console', True))
        self.processing = False
        self.advanced_config = {}
        
        # --- Async and Progress Tracking ---
        self.async_manager = None
        self.processing_thread = None
        self.submitted_tasks_count = 0
        self.completed_tasks_count = 0
        self.current_tasks = []
        self.console_buffer = []
        
        self.progress_line_index = None
        self.last_progress_update_time = 0
        
        # Modern progress system components
        self.progress_manager = None
        self.progress_display = None
        self.progress_update_timer = None
        
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
        
        """
        self.cancel_button = tk.Button(
            main_container,  # Or wherever your start button is
            text="CANCEL PROCESSING",
            command=self.cancel_processing,
            bg='#dc3545',
            fg='white',
            font=('Arial', 12, 'bold'),
            padx=40,
            pady=12,
            bd=0,
            relief=tk.FLAT,
            activebackground='#cc2e3f',
            cursor='hand2'
        )
        # Don't pack it initially - only show during processing
        """        
        
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
        dialog = AdvancedSettingsDialog(self, self.config_manager, self.sensitivity.get())
        self.wait_window(dialog)
        
        if dialog.result:
            # Settings are automatically saved by ConfigManager in dialog
            self.write_to_console("Advanced settings updated and saved to configuration")
    
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
        
        # Skip progress bar artifacts and overwhelming outputs
        skip_patterns = ["Progress:", "Processing regions:", "||", "Transcribing scene", "VAD detected"]
        if any(pattern in message for pattern in skip_patterns):
            self.console.config(state=tk.DISABLED)
            return
        
        # Apply formatting based on content
        if "╔" in message or "║" in message or "╚" in message:
            self.console.insert(tk.END, message + "\n", "banner")
        elif "[✓" in message or "✓" in message or "successful" in message.lower():
            self.console.insert(tk.END, message + "\n", "success")
        elif "error" in message.lower() or "failed" in message.lower() or "[✗" in message or "✗" in message:
            self.console.insert(tk.END, message + "\n", "error")
        elif "Step" in message and "/" in message:
            self.console.insert(tk.END, message + "\n", "step")
        elif "Processing:" in message or "Starting:" in message:
            self.console.insert(tk.END, message + "\n", "current_file")
        elif "PROCESSING SUMMARY" in message or "PROCESSING COMPLETE" in message or "=" * 10 in message:
            self.console.insert(tk.END, message + "\n", "summary")
        else:
            self.console.insert(tk.END, message + "\n")
        
        # Limit console to 500 lines to prevent performance issues
        line_count = int(self.console.index('end-1c').split('.')[0])
        if line_count > 500:
            # Remove oldest 100 lines
            self.console.delete('1.0', '100.0')
        
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
    
    def _run_main_direct(self):
        """Run main directly when in exe mode"""
        try:
            from whisperjav.main import main as whisperjav_main
            
            self.write_to_console(f"Running in exe mode with args: {' '.join(sys.argv[1:])}")
            
            # Capture output by redirecting stdout/stderr to queue
            import io
            import contextlib
            
            class QueueWriter:
                def __init__(self, queue):
                    self.queue = queue
                
                def write(self, msg):
                    if msg and msg.strip():
                        self.queue.put(msg.strip())
                
                def flush(self):
                    pass
            
            # Redirect output to our queue
            queue_writer = QueueWriter(self.output_queue)
            
            with contextlib.redirect_stdout(queue_writer), contextlib.redirect_stderr(queue_writer):
                try:
                    # Run main
                    whisperjav_main()
                    return_code = 0
                except SystemExit as e:
                    return_code = e.code if e.code is not None else 0
            
            self.output_queue.put(f"\nProcess completed with exit code: {return_code}")
            
        except Exception as e:
            self.output_queue.put(f"Error during processing: {str(e)}")
            import traceback
            self.output_queue.put(traceback.format_exc())
        finally:
            # Restore original argv
            sys.argv = self.original_argv
            self.output_queue.put(None)  # Signal processing complete
            self.after(100, self.processing_complete)    

    def start_processing(self):
        """Start processing with async pipeline manager instead of subprocess."""
        if not self.input_files:
            messagebox.showerror("Input Error", "Please select at least one media file.")
            return
        
        output_dir = self.output_dir.get()
        if not output_dir:
            messagebox.showerror("Output Error", "Please select an output directory.")
            return
        
        # Create output directory if needed
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError:
                messagebox.showerror("Output Error", "Could not create output directory.")
                return
        
        # Reset state
        self.current_file_index = 0
        self.total_files = len(self.input_files)
        self.progress_line_ids = {}
        self.last_progress_text = ""
        self.console_buffer = []

        self.submitted_tasks_count = len(self.input_files)
        self.completed_tasks_count = 0
        self.current_tasks = [] # Populated by the progress handler        
        
        
        # Disable start button
        self.processing = True
        self.start_button.config(state=tk.DISABLED, bg='#cccccc')
        self.write_to_console("Initializing processing...")
        
        # Show progress bar
        self.progress_label.config(text="Preparing files...")

        self.progress_bar.pack(pady=5)
        self.progress_bar.start(10)
        self.write_to_console("Initializing processing...")
        
        # --- FIX: Reset state for inline progress updates ---
        self.progress_line_index = None
        self.last_progress_update_time = 0
        
        self.submitted_tasks_count = len(self.input_files)

             
        # Initialize modern progress system
        verbosity_mapping = {
            'quiet': UnifiedVerbosityLevel.QUIET,
            'summary': UnifiedVerbosityLevel.STANDARD,
            'normal': UnifiedVerbosityLevel.STANDARD,
            'verbose': UnifiedVerbosityLevel.DEBUG
        }
        
        console_verbosity = self.ui_preferences.get('console_verbosity', 'summary')
        verbosity = verbosity_mapping.get(console_verbosity, UnifiedVerbosityLevel.STANDARD)
        
        # Create unified progress system
        self.progress_manager = UnifiedProgressManager(verbosity=verbosity)
        self.progress_display = ProgressDisplayAdapter(self.progress_manager)
        
        # Create async manager with unified progress
        self.async_manager = AsyncPipelineManager(
            ui_update_callback=self.handle_async_progress,
            verbosity=verbosity
        )
        
        # Prepare configuration
        try:
            # Use correct TranscriptionTuner from v4.3 system
            config_path = Path(self.advanced_config.get('config_path')) if 'config_path' in self.advanced_config else None
            tuner = TranscriptionTuner(config_path=config_path)
            
            task = 'translate' if self.subs_language.get() == 'english-direct' else 'transcribe'
            resolved_config = tuner.resolve_params(
                pipeline_name=self.mode.get(),
                sensitivity=self.sensitivity.get(),
                task=task
            )
            
            # Add UI-specific settings
            resolved_config['output_dir'] = output_dir
            resolved_config['temp_dir'] = str(Path(tempfile.gettempdir()) / "whisperjav")
            resolved_config['keep_temp_files'] = False
            resolved_config['subs_language'] = self.subs_language.get()
            
            # Add enhancement options
            resolved_config['adaptive_classification'] = self.adaptive_classification.get()
            resolved_config['adaptive_audio_enhancement'] = self.adaptive_audio_enhancement.get()
            resolved_config['smart_postprocessing'] = self.smart_postprocessing.get()
            
        except FileNotFoundError as e:
            self.write_to_console(f"Configuration file not found: {e}")
            messagebox.showerror("Configuration Error", 
                               f"Could not load configuration file.\n\n{str(e)}\n\nPlease check your WhisperJAV installation.")
            self.processing_complete()
            return
        except ValueError as e:
            self.write_to_console(f"Configuration validation error: {e}")
            messagebox.showerror("Configuration Error", 
                               f"Invalid configuration settings.\n\n{str(e)}\n\nTry resetting to defaults in Advanced Settings.")
            self.processing_complete()
            return
        except Exception as e:
            logger.error(f"Unexpected configuration error: {e}", exc_info=True)
            self.write_to_console(f"Unexpected configuration error: {e}")
            messagebox.showerror("Error", f"An unexpected error occurred during configuration:\n\n{str(e)}")
            self.processing_complete()
            return
        
        # Discover media files
        from whisperjav.modules.media_discovery import MediaDiscovery
        discovery = MediaDiscovery()
        media_files = discovery.discover(self.input_files)
        
        if not media_files:
            self.write_to_console("No valid media files found!")
            self.processing_complete()
            return
        
        # Start async processing in a separate thread
        self.processing_thread = threading.Thread(
            target=self._process_files_async,
            args=(media_files, resolved_config),
            daemon=True
        )
        self.processing_thread.start()
        
        # Start progress update timer
        self.start_progress_updates()
    




    def _process_files_async(self, media_files, resolved_config):
        """Run async processing in background thread."""
        try:
            # This call is now NON-BLOCKING and returns a list of task IDs.
            # We store these if we need them for cancellation.
            task_ids = self.async_manager.process_files(
                media_files,
                self.mode.get(),
                resolved_config
            )
            self.submitted_task_ids = task_ids # Store the IDs
            self.submitted_tasks_count = len(task_ids)
            self.completed_tasks_count = 0

            # The thread's job is now done. It has successfully submitted the tasks.
            # It no longer waits here. The UI will be notified of completion via events.
            
        except Exception as e:
            # Use the buffer to safely communicate with the UI thread
            self.console_buffer.append(f"Error submitting tasks: {str(e)}")
            # Schedule the final UI update from the main thread
            self.after(100, self.processing_complete)


    def handle_async_progress(self, message: dict):
        """
        Handle progress updates from the async processor.

        This method runs on the main UI thread and is the bridge between the
        background processing and the GUI. It buffers messages and updates the
        console in batches to maintain UI responsiveness.
        """
        msg_type = message.get('type')
        timestamp = time.strftime('%H:%M:%S')

        # --- Handle different message types ---

        if msg_type == 'file_start':
            filename = message.get('filename', 'Unknown')
            file_num = message.get('file_number', self.current_file_index + 1)
            total = message.get('total_files', self.total_files)
            
            self.current_file_index = file_num - 1
            self.console_buffer.append(f"\n[{timestamp}] [{file_num}/{total}] Starting: {filename}")
            
            # Schedule the label update to run on the UI thread
            self.after(0, lambda: self.progress_label.config(text=f"Processing: {filename}"))
            
        elif msg_type == 'step_update': 
            self.progress_line_index = None
            step_name = message.get('step_name', '')
            self.console_buffer.append(f"  > {step_name}...")
            
        elif msg_type == 'scene_progress':
            # --- FIX (a): Logic for more frequent, time-based updates ---
            now = time.time()
            scene_idx = message.get('scene_index', 0)
            total_scenes = message.get('total_scenes', 0)
            
            # Update if it's the first scene, the last scene, every 5 scenes, OR if 10s have passed
            if (total_scenes > 0 and (
                scene_idx == 0 or 
                (scene_idx + 1) == total_scenes or 
                scene_idx % 5 == 0 or 
                (now - self.last_progress_update_time > 10))
            ):
                progress_pct = ((scene_idx + 1) / total_scenes) * 100
                progress_text = f"    - Scene transcription: {progress_pct:.0f}% ({scene_idx + 1}/{total_scenes})"
                self.console_buffer.append(progress_text)
                self.last_progress_update_time = now # Reset timer

            status = message.get('status', '')
            
            # Batch scene updates to avoid console spam.
            # Show an update every 20 scenes, on the first/last scene, or on failure.
            if scene_idx % 20 == 0 or status == 'failed' or scene_idx + 1 == total_scenes:
                if total_scenes > 0:
                    progress_pct = ((scene_idx + 1) / total_scenes) * 100
                    self.console_buffer.append(f"    - Scene transcription: {progress_pct:.0f}% ({scene_idx + 1}/{total_scenes})")
                
        elif msg_type == 'completion':
            # This message is from the pipeline when a single file finishes.
            # We can use it to show intermediate success/failure.
            success = message.get('success', False)
            stats = message.get('stats', {})
            
            if success:
                subtitles = stats.get('subtitles', 0)
                duration = stats.get('duration', 0)
                self.console_buffer.append(f"  ✓ Intermediate result: {subtitles} subtitles found in {duration:.1f}s")
            else:
                error = stats.get('error', 'Unknown error')
                self.console_buffer.append(f"  ✗ Intermediate error: {error}")
                
        elif msg_type == 'task_complete':
            # This is the final confirmation that a task is done.
            # We use this to track overall completion.
            self.completed_tasks_count += 1
            
            task_id = message.get('task_id')
            success = message.get('success', False)
            
            # Retrieve the final task object to get all details for the summary
            final_task_state = self.async_manager.processor.get_task_status(task_id)
            if final_task_state:
                self.current_tasks.append(final_task_state)

            if success:
                result = message.get('result', {})
                output_file = Path(result.get('output_files', {}).get('final_srt', ''))
                self.console_buffer.append(f"[{timestamp}] ✓ Task Complete. Output: {output_file.name}")
            else:
                error = message.get('error', 'Unknown error')
                self.console_buffer.append(f"[{timestamp}] ✗ Task Failed: {error}")
            
            # --- CRITICAL: Check if all submitted tasks are finished ---
            if self.completed_tasks_count >= self.submitted_tasks_count:
                # All jobs are done. Schedule the final cleanup and summary.
                # self.after() ensures this runs safely on the main UI thread.
                self.after(100, self.processing_complete)

        # --- Trigger a batched update to the console ---
        self.update_console_batch()

    def update_console_batch(self):
            """Update console in batches and handle inline progress."""
            if not self.console_buffer:
                return

            # Filter out any empty messages that could reset the progress line index
            processed_messages = [msg for msg in self.console_buffer if msg and msg.strip()]
            self.console_buffer.clear()
            
            if not processed_messages:
                return

            self.console.config(state=tk.NORMAL)
            
            for message in processed_messages:
                # --- FIX (b): Logic for inline progress updates ---
                if "Scene transcription:" in message:
                    if self.progress_line_index:
                        # Delete the old progress line content
                        self.console.delete(self.progress_line_index, f"{self.progress_line_index} lineend")
                        # Insert the new progress message at the same position
                        self.console.insert(self.progress_line_index, message, "progress")
                    else:
                        # This is the first progress line for this step.
                        # Ensure it starts on a new line, then store its index.
                        self.console.insert(tk.END, "\n" + message, "progress")
                        self.progress_line_index = self.console.index("end-1l linestart")
                else:
                    # This is a normal message. If a progress line was active,
                    # move to the next line before printing. Then, reset the index.
                    if self.progress_line_index:
                        self.console.insert(f"{self.progress_line_index} lineend", "\n")
                        self.progress_line_index = None
                    
                    self.console.insert(tk.END, message + "\n")

            self.console.see(tk.END)
            self.console.config(state=tk.DISABLED)

    def start_progress_updates(self):
        """Start periodic progress updates."""
        def update():
            if self.processing:
                # Force console update
                self.update_console_batch()
                
                # Update progress bar position if we know the progress
                if self.total_files > 0 and self.current_file_index > 0:
                    progress = (self.current_file_index / self.total_files) * 100
                    # Could update a determinate progress bar here if you switch from indeterminate
                    
                # Schedule next update
                self.progress_update_timer = self.after(100, update)
        
        update()


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
        """Handle processing completion."""
        self.processing = False
        self.start_button.config(state=tk.NORMAL, bg='#4a7abc')
        
        # Stop progress updates
        if self.progress_update_timer:
            self.after_cancel(self.progress_update_timer)
            self.progress_update_timer = None
        
        # Final console flush
        self.update_console_batch()
        
        # Hide progress bar
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.config(text="")
        
        # Shutdown async manager
        if self.async_manager:
            try:
                self.async_manager.shutdown()
            except Exception as e:
                logger.error(f"Error during async manager shutdown: {e}")
            self.async_manager = None
        
        # Cleanup progress manager
        if self.progress_manager:
            try:
                self.progress_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during progress manager cleanup: {e}")
            self.progress_manager = None
        
        # Show summary
        if self.current_tasks:
            successful = sum(1 for t in self.current_tasks if t.status == ProcessingStatus.COMPLETED)
            failed = sum(1 for t in self.current_tasks if t.status == ProcessingStatus.FAILED)
            
            summary = f"\n{'='*50}\nPROCESSING COMPLETE\n{'='*50}\n"
            summary += f"Total files: {len(self.current_tasks)}\n"
            summary += f"Successful: {successful}\n"
            if failed > 0:
                summary += f"Failed: {failed}\n"
                
                # List failed files
                for task in self.current_tasks:
                    if task.status == ProcessingStatus.FAILED:
                        summary += f"  - {task.media_info.get('basename', 'Unknown')}: {task.error}\n"
            
            summary += "="*50 + "\n"
            
            self.write_to_console(summary)
        
        self.write_to_console("Ready for next batch.")


    def periodic_gui_update(self):
        """
        FIX (c): New method.
        Periodically flushes the console buffer to ensure responsiveness.
        """
        if not self.processing:
            return
        self.update_console_batch()
        self.gui_update_timer = self.after(500, self.periodic_gui_update) # Reschedule for 500ms later


    def cancel_processing(self):
        """Cancel ongoing processing."""
        if not self.processing or not self.async_manager:
            return
        
        response = messagebox.askyesno(
            "Cancel Processing",
            "Are you sure you want to cancel the current processing?"
        )
        
        if response:
            self.write_to_console("\nCancelling processing...")
            
            # Cancel all running tasks
            if self.current_tasks:
                for task in self.current_tasks:
                    if task.status == ProcessingStatus.RUNNING:
                        try:
                            self.async_manager.processor.cancel_task(task.task_id)
                        except:
                            pass
            
            # This will trigger completion
            self.processing_complete()
            
        
    def on_close(self):
        # Save current UI state to configuration before closing
        try:
            ui_settings = {
                'last_mode': self.mode.get(),
                'last_sensitivity': self.sensitivity.get(),
                'last_language': self.subs_language.get(),
                'show_console': self.show_console.get(),
                'adaptive_classification': self.adaptive_classification.get(),
                'adaptive_audio_enhancement': self.adaptive_audio_enhancement.get(),
                'smart_postprocessing': self.smart_postprocessing.get()
            }
            self.config_manager.update_ui_preferences(ui_settings)
            self.config_manager.save_config()
        except Exception as e:
            logger.error(f"Failed to save UI preferences: {e}")
        
        if self.processing:
            if messagebox.askokcancel("Quit", "Processing is still running. Quit anyway?"):
                self.cleanup_and_exit()
        else:
            self.destroy()

    def cleanup_and_exit(self):
        """Clean up resources and exit."""
        if self.async_manager:
            try:
                self.async_manager.shutdown(wait=False)
            except Exception as e:
                logger.error(f"Error during async manager shutdown: {e}")
        
        if self.progress_manager:
            try:
                self.progress_manager.cleanup()
            except Exception as e:
                logger.error(f"Error during progress manager cleanup: {e}")
        
        self.destroy()
                
            
def main():
    """Main entry point for GUI."""
    app = WhisperJAVGUI()
    app.mainloop()

if __name__ == "__main__":
    main()