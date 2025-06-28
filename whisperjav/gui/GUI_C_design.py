import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox, Toplevel
import os
import threading
import sys
import queue
import subprocess
import json
from pathlib import Path

class AdvancedSettingsDialog(Toplevel):
    def __init__(self, parent, config=None):
        super().__init__(parent)
        self.title("Advanced Settings")
        self.geometry("500x400")
        self.transient(parent)
        self.grab_set()
        
        self.config = config or {}
        self.result = None
        
        # Create notebook for different setting categories
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # VAD Settings
        vad_frame = ttk.Frame(notebook, padding=10)
        notebook.add(vad_frame, text="VAD Settings")
        self.create_vad_settings(vad_frame)
        
        # Transcription Settings
        trans_frame = ttk.Frame(notebook, padding=10)
        notebook.add(trans_frame, text="Transcription")
        self.create_transcription_settings(trans_frame)
        
        # Decoder Settings
        dec_frame = ttk.Frame(notebook, padding=10)
        notebook.add(dec_frame, text="Decoder")
        self.create_decoder_settings(dec_frame)
        
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)
        
        ttk.Button(btn_frame, text="Save", command=self.save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
    
    def create_vad_settings(self, parent):
        ttk.Label(parent, text="VAD Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.vad_threshold = tk.DoubleVar(value=self.config.get("vad_threshold", 0.5))
        ttk.Scale(parent, from_=0.1, to=0.9, variable=self.vad_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=5)
        ttk.Label(parent, textvariable=tk.StringVar(value=f"{self.vad_threshold.get():.2f}")).grid(row=0, column=2)
        
        ttk.Label(parent, text="Min Speech Duration (ms):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.min_speech_duration = tk.IntVar(value=self.config.get("min_speech_duration", 250))
        ttk.Entry(parent, textvariable=self.min_speech_duration, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Max Speech Duration (s):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.max_speech_duration = tk.IntVar(value=self.config.get("max_speech_duration", 30))
        ttk.Entry(parent, textvariable=self.max_speech_duration, width=10).grid(row=2, column=1, sticky=tk.W)
    
    def create_transcription_settings(self, parent):
        ttk.Label(parent, text="Beam Size:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.beam_size = tk.IntVar(value=self.config.get("beam_size", 5))
        ttk.Entry(parent, textvariable=self.beam_size, width=10).grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Patience:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.patience = tk.DoubleVar(value=self.config.get("patience", 1.0))
        ttk.Entry(parent, textvariable=self.patience, width=10).grid(row=1, column=1, sticky=tk.W)
        
        ttk.Label(parent, text="Temperature:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.temperature = tk.DoubleVar(value=self.config.get("temperature", 0.0))
        ttk.Entry(parent, textvariable=self.temperature, width=10).grid(row=2, column=1, sticky=tk.W)
    
    def create_decoder_settings(self, parent):
        ttk.Label(parent, text="Word Threshold:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.word_threshold = tk.DoubleVar(value=self.config.get("word_threshold", 0.01))
        ttk.Scale(parent, from_=0.0, to=1.0, variable=self.word_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=0, column=1, padx=5)
        ttk.Label(parent, textvariable=tk.StringVar(value=f"{self.word_threshold.get():.2f}")).grid(row=0, column=2)
        
        ttk.Label(parent, text="Logprob Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.logprob_threshold = tk.DoubleVar(value=self.config.get("logprob_threshold", -1.0))
        ttk.Scale(parent, from_=-5.0, to=0.0, variable=self.logprob_threshold, 
                 orient=tk.HORIZONTAL, length=300).grid(row=1, column=1, padx=5)
        ttk.Label(parent, textvariable=tk.StringVar(value=f"{self.logprob_threshold.get():.2f}")).grid(row=1, column=2)
    
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

class WhisperJAVGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhisperJAV - Japanese Adult Video Subtitle Generator")
        self.geometry("800x600")
        self.resizable(True, True)
        
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
        
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Section 1: Choose Media and Output Destination
        media_frame = ttk.LabelFrame(main_frame, text="Choose Media and Output Destination", padding=10)
        media_frame.pack(fill=tk.X, pady=5)
        
        # File selection
        file_frame = ttk.Frame(media_frame)
        file_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(file_frame, text="Select File(s)").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Button(file_frame, text="Browse...", command=self.select_files).pack(side=tk.LEFT)
        
        self.file_label = ttk.Label(media_frame, text="No files selected")
        self.file_label.pack(fill=tk.X, pady=5)
        
        # Output folder
        output_frame = ttk.Frame(media_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(output_frame, text="Output Folder").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_dir, width=50).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        ttk.Button(output_frame, text="Browse...", command=self.select_output_dir).pack(side=tk.LEFT)
        
        # Section 2: Speed Control
        speed_frame = ttk.LabelFrame(main_frame, text="Speed Control  (Quickie vs Less Mistakes)", padding=10)
        speed_frame.pack(fill=tk.X, pady=5)
        
        speed_options = ttk.Frame(speed_frame)
        speed_options.pack(fill=tk.X)
        
        ttk.Radiobutton(speed_options, text="Faster", variable=self.mode, value="faster").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(speed_options, text="Fast", variable=self.mode, value="fast").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(speed_options, text="Balanced", variable=self.mode, value="balanced").pack(side=tk.LEFT, padx=10)
        
        # Section 3: Granularity Control
        granularity_frame = ttk.LabelFrame(main_frame, text="Granularity Control  (Details vs. Fewer Guesses)", padding=10)
        granularity_frame.pack(fill=tk.X, pady=5)
        
        granularity_options = ttk.Frame(granularity_frame)
        granularity_options.pack(fill=tk.X)
        
        ttk.Radiobutton(granularity_options, text="Aggressive", variable=self.sensitivity, value="aggressive").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(granularity_options, text="Balanced", variable=self.sensitivity, value="balanced").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(granularity_options, text="Conservative", variable=self.sensitivity, value="conservative").pack(side=tk.LEFT, padx=10)
        
        # Section 4: SRT subs Language
        language_frame = ttk.LabelFrame(main_frame, text="SRT subs Language", padding=10)
        language_frame.pack(fill=tk.X, pady=5)
        
        language_options = ttk.Frame(language_frame)
        language_options.pack(fill=tk.X)
        
        ttk.Radiobutton(language_options, text="Japanese", variable=self.subs_language, value="japanese").pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(language_options, text="English Direct", variable=self.subs_language, value="english-direct").pack(side=tk.LEFT, padx=10)
        
        # Advanced Settings
        advanced_frame = ttk.Frame(main_frame)
        advanced_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(advanced_frame, text="Advanced Settings...", command=self.open_advanced_settings).pack(side=tk.LEFT)
        
        # Enhancement options
        enh_frame = ttk.Frame(advanced_frame)
        enh_frame.pack(side=tk.RIGHT)
        
        ttk.Checkbutton(enh_frame, text="Adaptive Classification", 
                        variable=self.adaptive_classification).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(enh_frame, text="Adaptive Audio Enhancement", 
                        variable=self.adaptive_audio_enhancement).pack(side=tk.LEFT, padx=10)
        ttk.Checkbutton(enh_frame, text="Smart Postprocessing", 
                        variable=self.smart_postprocessing).pack(side=tk.LEFT, padx=10)
        
        # Start button
        self.start_button = ttk.Button(main_frame, text="START PROCESSING", width=50, 
                                      command=self.start_processing, style="Accent.TButton")
        self.start_button.pack(pady=10)
        
        # Console output
        self.console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding=5)
        self.console_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.console = scrolledtext.ScrolledText(self.console_frame, height=10, state=tk.DISABLED)
        self.console.pack(fill=tk.BOTH, expand=True)
        
        # Create custom style for the start button
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 10, "bold"), foreground="white", background="#4a7abc")
        
    def select_files(self):
        files = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv"), ("All Files", "*.*")]
        )
        if files:
            self.input_files = list(files)
            num_files = len(self.input_files)
            file_text = f"{num_files} file{'s' if num_files > 1 else ''} selected"
            
            # Set output directory to first file's directory
            if num_files > 0:
                first_file_dir = os.path.dirname(self.input_files[0])
                self.output_dir.set(first_file_dir)
            
            self.file_label.config(text=file_text)
    
    def select_output_dir(self):
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir.set(directory)
    
    def open_advanced_settings(self):
        dialog = AdvancedSettingsDialog(self, self.advanced_config)
        self.wait_window(dialog)
        
        if dialog.result:
            self.advanced_config = dialog.result
            self.write_to_console("Advanced settings updated")
    
    def write_to_console(self, message):
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)  # Auto-scroll to end
        self.console.config(state=tk.DISABLED)
    
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
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.write_to_console("Starting processing...")
        
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
            sys.executable, "main.py",
            "--output-dir", output_dir,
            "--mode", self.mode.get(),
            "--sensitivity", self.sensitivity.get(),
            "--subs-language", self.subs_language.get(),
            "--no-progress"
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
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace'
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
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line is None:
                    self.processing_complete()
                    return
                self.write_to_console(line.strip())
        except queue.Empty:
            pass
        
        if self.processing:
            self.after(100, self.update_console)
    
    def processing_complete(self):
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.write_to_console("Processing complete.")
    
    def on_close(self):
        if self.processing:
            if messagebox.askokcancel("Quit", "Processing is still running. Quit anyway?"):
                self.destroy()
        else:
            self.destroy()

if __name__ == "__main__":
    app = WhisperJAVGUI()
    app.mainloop()