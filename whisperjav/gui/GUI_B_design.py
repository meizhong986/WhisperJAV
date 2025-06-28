#!/usr/bin/env python3
"""
WhisperJAV GUI - A Tkinter frontend for the WhisperJAV application.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import queue
import threading
import sys
import os
from pathlib import Path
import argparse

# --- Assuming gui.py is in the root directory alongside main.py and whisperjav/ ---
# This allows importing the necessary components from the existing application
from whisperjav.utils.logger import setup_logger, logger
from whisperjav.modules.media_discovery import MediaDiscovery
from whisperjav.pipelines.faster_pipeline import FasterPipeline
from whisperjav.pipelines.fast_pipeline import FastPipeline
from whisperjav.pipelines.balanced_pipeline import BalancedPipeline
from whisperjav.config.transcription_tuner import TranscriptionTuner
from whisperjav.utils.progress_display import ProgressDisplay


class QueueIO:
    """A file-like object that writes to a queue. Used to redirect stdout/stderr."""
    def __init__(self, q):
        self.queue = q

    def write(self, text):
        self.queue.put(text)

    def flush(self):
        pass  # No-op

class WhisperJavGUI:
    """The main class for the WhisperJAV Tkinter GUI."""

    def __init__(self, root):
        self.root = root
        self.root.title("WhisperJAV")
        self.root.minsize(700, 600)

        # --- Member variables to store selections ---
        self.input_files = []
        self.output_folder = ""
        self.advanced_params_override = None

        # --- Style Configuration ---
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("TButton", padding=6, relief="flat", background="#007bff", foreground="white")
        style.map("TButton", background=[('active', '#0056b3')])
        style.configure("TLabelFrame.Label", font=('Helvetica', 12, 'bold'))

        # --- Tkinter control variables ---
        self.mode_var = tk.StringVar(value="balanced") # Default: Balanced
        self.sensitivity_var = tk.StringVar(value="balanced") # Default: Balanced
        self.language_var = tk.StringVar(value="japanese") # Default: Japanese

        # --- Setup logging queue ---
        self.log_queue = queue.Queue()

        # --- Build the UI ---
        self._create_widgets()
        self._check_log_queue()

    def _create_widgets(self):
        """Create and arrange all the widgets in the main window."""
        main_frame = ttk.Frame(self.root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # --- Top frame for options ---
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, expand=True)
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)
        options_frame.columnconfigure(2, weight=1)
        options_frame.columnconfigure(3, weight=1)


        # --- 1. Choose Media and Output Destination ---
        io_frame = ttk.LabelFrame(options_frame, text="Choose Media and Output Destination", padding="10")
        io_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        io_frame.rowconfigure(0, weight=1)
        io_frame.rowconfigure(1, weight=1)
        
        select_files_btn = ttk.Button(io_frame, text="Select File(s)", command=self._select_files)
        select_files_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        output_folder_btn = ttk.Button(io_frame, text="Output Folder", command=self._select_output_folder)
        output_folder_btn.grid(row=1, column=0, sticky="ew", padx=5, pady=5)


        # --- 2. Speed Control ---
        speed_frame = ttk.LabelFrame(options_frame, text="Speed Control", padding="10")
        speed_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        ttk.Radiobutton(speed_frame, text="Faster", variable=self.mode_var, value="faster").pack(anchor=tk.W)
        ttk.Radiobutton(speed_frame, text="Fast", variable=self.mode_var, value="fast").pack(anchor=tk.W)
        ttk.Radiobutton(speed_frame, text="Balanced", variable=self.mode_var, value="balanced").pack(anchor=tk.W)


        # --- 3. Granularity Control ---
        gran_frame = ttk.LabelFrame(options_frame, text="Granularity Control", padding="10")
        gran_frame.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")
        ttk.Radiobutton(gran_frame, text="Aggressive", variable=self.sensitivity_var, value="aggressive").pack(anchor=tk.W)
        ttk.Radiobutton(gran_frame, text="Balanced", variable=self.sensitivity_var, value="balanced").pack(anchor=tk.W)
        ttk.Radiobutton(gran_frame, text="Conservative", variable=self.sensitivity_var, value="conservative").pack(anchor=tk.W)


        # --- 4. SRT Subs Language ---
        lang_frame = ttk.LabelFrame(options_frame, text="SRT subs Language", padding="10")
        lang_frame.grid(row=0, column=3, padx=5, pady=5, sticky="nsew")
        ttk.Radiobutton(lang_frame, text="Japanese", variable=self.language_var, value="japanese").pack(anchor=tk.W)
        ttk.Radiobutton(lang_frame, text="English", variable=self.language_var, value="english-direct").pack(anchor=tk.W)
        
        adv_settings_btn = ttk.Button(lang_frame, text="⚙️ Advanced Settings", command=self._open_advanced_settings)
        adv_settings_btn.pack(anchor=tk.W, pady=(10,0))

        # --- Start Processing Button ---
        self.start_button = ttk.Button(main_frame, text="Start Processing", command=self._start_processing)
        self.start_button.pack(fill=tk.X, padx=5, pady=(10, 10))

        # --- Console Output ---
        console_frame = ttk.LabelFrame(main_frame, text="Console Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console_output = ScrolledText(console_frame, state='disabled', wrap=tk.WORD, bg='#2b2b2b', fg='#f0f0f0')
        self.console_output.pack(fill=tk.BOTH, expand=True)

    def _select_files(self):
        """Handler for 'Select File(s)' button."""
        files = filedialog.askopenfilenames(
            title="Select Media Files",
            filetypes=(("Video Files", "*.mp4 *.mkv *.avi *.mov"), ("All files", "*.*"))
        )
        if files:
            self.input_files = list(files)
            self.console_log(f"Selected {len(self.input_files)} file(s) for processing.")
        else:
            self.console_log("File selection cancelled.")

    def _select_output_folder(self):
        """Handler for 'Output Folder' button."""
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder = folder
            self.console_log(f"Output folder set to: {self.output_folder}")
        else:
            self.console_log("Output folder selection cancelled.")

    def _open_advanced_settings(self):
        """Opens the advanced settings pop-up window."""
        messagebox.showinfo(
            "Advanced Settings",
            "This will show current VAD, transcribe, and decoder options based on your selections, allowing you to override them for this run. This feature is planned for a future update.",
            parent=self.root
        )
        # Placeholder for full implementation as discussed.

    def console_log(self, message):
        """Inserts a message into the console output widget."""
        self.console_output.configure(state='normal')
        self.console_output.insert(tk.END, message + '\n')
        self.console_output.configure(state='disabled')
        self.console_output.see(tk.END) # Auto-scroll

    def _check_log_queue(self):
        """Periodically checks the queue for new log messages."""
        while not self.log_queue.empty():
            message = self.log_queue.get_nowait()
            self.console_log(message.strip())
        self.root.after(100, self._check_log_queue)

    def _toggle_controls(self, enabled):
        """Enable or disable all interactive controls."""
        state = 'normal' if enabled else 'disabled'
        for child in self.root.winfo_children():
            self._set_widget_state(child, state)
        
        # Ensure console remains scrollable but not editable
        self.console_output.configure(state='disabled')

    def _set_widget_state(self, widget, state):
        """Recursively set the state of a widget and its children."""
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass # Some widgets like frames don't have a 'state'
        
        for child in widget.winfo_children():
            self._set_widget_state(child, state)

    def _start_processing(self):
        """Validates settings and starts the processing thread."""
        if not self.input_files:
            messagebox.showerror("Input Error", "Please select one or more media files first.", parent=self.root)
            return

        self._toggle_controls(enabled=False)
        self.console_log("="*50)
        self.console_log("Starting processing...")

        # Create a thread to run the main logic, to not freeze the GUI
        processing_thread = threading.Thread(
            target=self._processing_thread_worker,
            daemon=True
        )
        processing_thread.start()

    def _processing_thread_worker(self):
        """The actual worker function that runs in a separate thread."""
        # --- Redirect stdout and stderr to our queue ---
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        queue_io = QueueIO(self.log_queue)
        sys.stdout = queue_io
        sys.stderr = queue_io
        
        try:
            # --- Build an 'args' namespace object to pass to the processing function ---
            # This simulates the command-line arguments based on GUI selections.
            args = argparse.Namespace(
                input=self.input_files,
                mode=self.mode_var.get(),
                sensitivity=self.sensitivity_var.get(),
                subs_language=self.language_var.get(),
                output_dir=self.output_folder if self.output_folder else None,
                config=None, # Advanced settings would populate this
                temp_dir=None,
                keep_temp=False,
                log_level="INFO",
                log_file=None,
                stats_file=None,
                no_progress=False, # Progress will be captured by console
                adaptive_classification=False, # Placeholder
                adaptive_audio_enhancement=False, # Placeholder
                smart_postprocessing=False, # Placeholder
                model=None, # Placeholder for advanced
                vad_threshold=None # Placeholder for advanced
            )
            
            # --- Run the main application logic adapted from main.py ---
            run_whisperjav_processing(args, self.advanced_params_override)

        except Exception as e:
            # Log any exceptions that occur within the thread
            logger.error(f"An unexpected error occurred in the processing thread: {e}", exc_info=True)
        finally:
            # --- Restore stdout/stderr and re-enable controls ---
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.log_queue.put(">>>PROCESSING_COMPLETE<<<")


def run_whisperjav_processing(args, advanced_params_override):
    """
    An adaptation of the `main()` function from `main.py` to be callable from the GUI.
    """
    # Use the GUI's logger
    global logger
    logger = setup_logger("whisperjav_gui", args.log_level)
    
    # --- Parameter Resolution using TranscriptionTuner ---
    tuner = TranscriptionTuner(config_path=args.config)
    resolved_params = tuner.get_resolved_params(
        mode=args.mode,
        sensitivity=args.sensitivity
    )

    if advanced_params_override:
         # In a full implementation, we would merge these overrides
         pass

    if not resolved_params:
        logger.error("Could not resolve transcription parameters. Check config file. Exiting.")
        return

    # --- Media Discovery ---
    # The `input` from GUI is already a list of full paths, so discovery is simpler
    discovery = MediaDiscovery()
    media_files = discovery.discover(args.input)

    if not media_files:
        logger.error(f"No valid media files found in the specified paths.")
        return

    logger.info(f"Found {len(media_files)} media file(s) to process.") # This message will appear in the GUI console

    # --- Set Output Directory Logic ---
    # If output_dir is not set, each SRT will be saved next to its video file.
    # The pipelines need to be adapted slightly to handle a null output_dir.
    output_dir_for_pipeline = args.output_dir
    if not output_dir_for_pipeline:
        logger.info("No output directory specified. Subtitles will be saved next to source files.") #

    # --- Setup and Run Pipeline ---
    progress = ProgressDisplay(len(media_files), enabled=not args.no_progress)

    pipeline_args = {
        "output_dir": output_dir_for_pipeline,
        "temp_dir": args.temp_dir,
        "keep_temp_files": args.keep_temp,
        "subs_language": args.subs_language,
        "resolved_params": resolved_params,
        "progress_display": progress,
        "adaptive_classification": args.adaptive_classification,
        "adaptive_audio_enhancement": args.adaptive_audio_enhancement,
        "smart_postprocessing": args.smart_postprocessing
    }
    
    pipeline_map = {
        "faster": FasterPipeline,
        "fast": FastPipeline,
        "balanced": BalancedPipeline
    }
    
    pipeline_class = pipeline_map.get(args.mode)
    if not pipeline_class:
        logger.error(f"Unknown mode '{args.mode}'")
        return
        
    pipeline = pipeline_class(**pipeline_args)
    
    # --- Process Files ---
    failed_files = []
    total_files = len(media_files)
    
    try:
        for i, media_info in enumerate(media_files, 1):
            file_path_str = media_info.get('path', 'Unknown File')
            file_name = Path(file_path_str).name
            progress.set_current_file(file_path_str, i)
            
            try:
                # If no output dir, set it to the file's parent for this run
                if not output_dir_for_pipeline:
                    pipeline.output_dir = str(Path(file_path_str).parent)

                metadata = pipeline.process(media_info)
                subtitle_count = metadata.get("summary", {}).get("final_subtitles_refined", 0)
                output_path = metadata.get("output_files", {}).get("final_srt", "")
                progress.show_file_complete(file_name, subtitle_count, output_path)
                progress.update_overall(1)
                
            except Exception as e:
                progress.show_message(f"Failed: {file_name} - {str(e)}", "error", 3.0)
                logger.error(f"Failed to process {file_path_str}: {e}", exc_info=True)
                failed_files.append(file_path_str)
                progress.update_overall(1)
    finally:
        progress.close()
        
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Total files: {total_files}")
    print(f"Successful: {total_files - len(failed_files)}")
    print(f"Failed: {len(failed_files)}")


if __name__ == "__main__":
    # Fix for pyinstaller executable pathing if needed
    if getattr(sys, 'frozen', False):
        application_path = os.path.dirname(sys.executable)
        os.chdir(application_path)
        
    root = tk.Tk()
    app = WhisperJavGUI(root)
    
    # Add a handler for when the processing thread finishes
    def on_processing_complete():
        msg = app.log_queue.get()
        if msg == ">>>PROCESSING_COMPLETE<<<":
            app.console_log("\nProcessing finished.")
            app._toggle_controls(enabled=True)
        else:
            app.log_queue.put(msg) # Put it back if it's not the signal
            root.after(200, on_processing_complete)

    def check_for_completion_signal():
        try:
            msg = app.log_queue.get_nowait()
            if msg == ">>>PROCESSING_COMPLETE<<<":
                app.console_log("\nProcessing finished.")
                app._toggle_controls(enabled=True)
            else:
                # This is a regular log message, put it back
                # A more robust system would have separate queues
                pass 
        except queue.Empty:
            pass
        finally:
            root.after(200, check_for_completion_signal)
    
    # Start the check
    check_for_completion_signal()
    
    root.mainloop()