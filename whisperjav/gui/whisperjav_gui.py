import os
import sys
import shlex
import threading
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

# Use site root so -m whisperjav.main resolves correctly
REPO_ROOT = Path(__file__).resolve().parents[2]

def _get_documents_dir() -> Path:
    """Best-effort resolve of the user's Documents folder across platforms.

    - On Windows, prefer SHGetKnownFolderPath(FOLDERID_Documents) and
      fall back to %USERPROFILE%/Documents or %OneDrive%/Documents.
    - On macOS/Linux, default to ~/Documents if it exists, else ~.
    """
    home = Path.home()
    # Windows-specific: try Known Folder API
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            # FOLDERID_Documents GUID {FDD39AD0-238F-46AF-ADB4-6C85480369C7}
            _FOLDERID_Documents = ctypes.c_char_p(b"{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")
            SHGetKnownFolderPath = ctypes.windll.shell32.SHGetKnownFolderPath
            SHGetKnownFolderPath.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_wchar_p)]
            SHGetKnownFolderPath.restype = ctypes.c_long

            # Convert GUID string to a GUID structure
            class GUID(ctypes.Structure):
                _fields_ = [
                    ("Data1", ctypes.c_uint32),
                    ("Data2", ctypes.c_uint16),
                    ("Data3", ctypes.c_uint16),
                    ("Data4", ctypes.c_ubyte * 8),
                ]

            def _guid_from_str(s: str) -> GUID:
                import uuid
                u = uuid.UUID(s)
                data4 = (ctypes.c_ubyte * 8)(*u.bytes[8:])
                return GUID(u.time_low, u.time_mid, u.time_hi_version, data4)

            guid = _guid_from_str("{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")
            path_ptr = ctypes.c_wchar_p()
            # Flags=0, hToken=None
            hr = SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(path_ptr))
            if hr == 0 and path_ptr.value:
                return Path(path_ptr.value)
        except Exception:
            # Fall through to environment heuristics
            pass

        # Heuristics: OneDrive/Documents preferred if present
        onedrive = os.environ.get("OneDrive")
        if onedrive:
            p = Path(onedrive) / "Documents"
            if p.exists():
                return p
        # Default to %USERPROFILE%/Documents
        p = home / "Documents"
        return p if p.exists() else home

    # Non-Windows
    docs = home / "Documents"
    return docs if docs.exists() else home


DEFAULT_OUTPUT = _get_documents_dir() / "WhisperJAV" / "output"

class WhisperJAVGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhisperJAV – Simple Runner")
        self.geometry("1000x640")
        self.proc = None
        self._make_ui()

    def _setup_notebook_style(self):
        """
        Configure file folder tab styling - physical folder metaphor design.

        Design philosophy:
        - Active tab appears to "lift" the content page (seamless connection)
        - Inactive tabs are LIGHTER and sit in background
        - Active tab breaks through tab bar border line
        - Clean, professional color palette

    Color palette:
    - Tab bar: #FFFFFF (white background)
    - Content area: #F0F2F5 (soft gray)
    - Active tab: #F0F2F5 (identical to content area)
    - Inactive tabs: #F7F8FA (lighter than active)
    - Active text: #024873 (deep blue, bold)
    - Inactive text: #4B5563 (gray-600)
    - Borders: #D1D5DB (light gray border)
    - Hover: #F2F4FA (midpoint between inactive and active)
        """
        style = ttk.Style(self)

        style.theme_use('clam')

        section_bg = '#F0F2F5'      # Content area and active tab
        tab_bar_bg = '#FFFFFF'    # Tab bar background
        inactive_bg = '#F7F8FA'     # Lighter than section – inactive tabs
        hover_bg = '#F2F4FA'        # Between inactive and active
        border_color = '#D1D5DB'

        # Tab bar container - keep outer margins modest so end tabs stay anchored
        style.configure(
            'TNotebook',
            background=tab_bar_bg,
            borderwidth=0,
            bordercolor=border_color,
            tabmargins=[20, 4, 20, 0],
            padding=[0, -2, 0, 0]
        )

        # Default tab appearance (inactive)
        style.configure(
            'TNotebook.Tab',
            padding=[80, 8, 80, 8], # padding=[40, 8, 40, 8],
            background=inactive_bg,
            foreground='#4B5563',
            font=('Segoe UI', 10),
            borderwidth=10,
            bordercolor=tab_bar_bg,
            relief='solid'
        )

        # State-based styling
        style.map(
            'TNotebook.Tab',
            background=[
                ('selected', section_bg),
                ('active', hover_bg),
                ('!selected', inactive_bg)
            ],
            foreground=[
                ('selected', '#024873'),
                ('active', '#1F2937'),
                ('!selected', '#4B5563')
            ],
            font=[
                ('selected', ('Segoe UI', 11, 'bold')),
                ('!selected', ('Segoe UI', 10))
            ],
            padding=[
                ('selected', [40, 12, 40, 12]),
                ('!selected', [40, 8, 40, 8])
            ],
            bordercolor=[
                ('selected', section_bg),
                ('active', border_color),
                ('!selected', tab_bar_bg)
            ],
            borderwidth=[
                ('selected', 10),
                ('active', 10),
                ('!selected', 10)
            ],
            relief=[
                ('selected', 'solid'),
                ('!selected', 'solid')
            ]
        )


        # Ensure content background aligns with active tab color
        style.configure('TNotebook.client', background=section_bg, borderwidth=0)

        # Apply section_bg to all widgets on the tabs for a seamless look
        style.configure('TFrame', background=section_bg)
        style.configure('TLabel', background=section_bg)
        style.configure('TRadiobutton', background=section_bg)

        # Also style the widgets for tab 2
        style.configure('TCheckbutton', background=section_bg)



    def _make_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Layout grid weights for responsive design
        frm.columnconfigure(0, weight=1)
        # Make the Log section (row 4) expand so run buttons stay visible
        frm.rowconfigure(4, weight=1)

        # Setup enhanced notebook styling for better tab discoverability
        self._setup_notebook_style()

        # Styles
        style = ttk.Style(self)
        try:
            style.configure('Info.TLabel', foreground='#666666')
        except Exception:
            pass

        # 1) Source (inputs list)
        src = ttk.LabelFrame(frm, text="Source")
        src.grid(row=0, column=0, sticky="nsew")
        src.columnconfigure(0, weight=1)
        self.inputs_listbox = tk.Listbox(src, height=4, selectmode=tk.EXTENDED)
        self.inputs_listbox.grid(row=0, column=0, columnspan=1, sticky="nsew", padx=6, pady=(6, 2))
        btns_src = ttk.Frame(src)
        btns_src.grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))
        ttk.Button(btns_src, text="Add File(s)", command=self.add_files).pack(side=tk.LEFT)
        ttk.Button(btns_src, text="Add Folder", command=self.add_folder).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns_src, text="Remove Selected", command=self.remove_selected_inputs).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns_src, text="Clear", command=self.clear_inputs).pack(side=tk.LEFT)

        # 2) Destination (output)
        dest = ttk.LabelFrame(frm, text="Destination")
        dest.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        dest.columnconfigure(1, weight=1)
        ttk.Label(dest, text="Output:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT))
        ttk.Entry(dest, textvariable=self.output_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        ttk.Button(dest, text="Browse", command=self.pick_output).grid(row=0, column=2, padx=6)
        ttk.Button(dest, text="Open", command=self.open_output).grid(row=0, column=3)

        # 3) Notebook with tabs
        self.notebook = ttk.Notebook(frm)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=(8, 0))

        # Tab 1: Transcription Mode (basic options)
        tab1 = tk.Frame(self.notebook, padx=10, pady=10, background='#F0F2F5', highlightthickness=0)
        self.notebook.add(tab1, text="Transcription Mode")

        # Configure tab1 grid
        for c in range(8):
            tab1.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        ttk.Label(tab1, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.mode_var = tk.StringVar(value="balanced")
        # Use small radio buttons for quick selection
        rb_frame = ttk.Frame(tab1)
        rb_frame.grid(row=0, column=1, sticky="w")
        for val in ("balanced", "fast", "faster"):
            ttk.Radiobutton(rb_frame, text=val, value=val, variable=self.mode_var).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(tab1, text="Sensitivity:").grid(row=0, column=2, sticky="e", padx=6)
        self.sens_var = tk.StringVar(value="balanced")
        ttk.Combobox(tab1, textvariable=self.sens_var, values=["conservative", "balanced", "aggressive"], width=14, state="readonly").grid(row=0, column=3, sticky="w")

        ttk.Label(tab1, text="Output language:").grid(row=0, column=4, sticky="e", padx=6)
        self.lang_var = tk.StringVar(value="japanese")
        ttk.Combobox(tab1, textvariable=self.lang_var, values=["japanese", "english-direct"], width=16, state="readonly").grid(row=0, column=5, sticky="w")

        # Info rows
        ttk.Label(
            tab1,
            text="Speed vs. Accuracy: 'fast' and 'faster' prioritize throughput; 'balanced' favors accuracy.",
            style='Info.TLabel', wraplength=420, justify='left'
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 6))
        ttk.Label(
            tab1,
            text="Details vs. Noise: 'conservative' reduces false positives; 'aggressive' may include noise while capturing more detail.",
            style='Info.TLabel', wraplength=420, justify='left'
        ).grid(row=1, column=3, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        # Tab 2: Transcription Adv. Options (advanced options)
        # Tab 2: Transcription Adv. Options (advanced options)
        tab2 = tk.Frame(self.notebook, padx=10, pady=10, background='#F0F2F5', highlightthickness=0)
        self.notebook.add(tab2, text="Transcription Adv. Options")

        # Force white gaps between tabs by adding padding to each tab
        self.notebook.tab(0, padding=(0, 8, 10, 8))  # Right padding of 10px
        self.notebook.tab(1, padding=(10, 8, 0, 8))  # Left padding of 10px

        # Configure tab2 grid with 4 columns
        for c in range(4):
            tab2.columnconfigure(c, weight=1)

        # Row 1 (4 elements): Adaptive Classification, Adaptive Audio Enhancements, Smart Postprocessing, Verbosity
        self.opt_adapt_cls = tk.BooleanVar(value=False)
        self.opt_adapt_enh = tk.BooleanVar(value=False)
        self.opt_smart_post = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Adaptive classification (WIP)", variable=self.opt_adapt_cls, state="disabled").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(tab2, text="Adaptive audio enhancements (WIP)", variable=self.opt_adapt_enh, state="disabled").grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(tab2, text="Smart postprocessing (WIP)", variable=self.opt_smart_post, state="disabled").grid(row=0, column=2, sticky="w", padx=6, pady=6)

        verbosity_frame = ttk.Frame(tab2)
        verbosity_frame.grid(row=0, column=3, sticky="w", padx=6, pady=6)
        ttk.Label(verbosity_frame, text="Verbosity:").pack(side=tk.LEFT)
        self.verbosity_var = tk.StringVar(value="summary")
        ttk.Combobox(verbosity_frame, textvariable=self.verbosity_var, values=["quiet", "summary", "normal", "verbose"], width=10, state="readonly").pack(side=tk.LEFT, padx=(4, 0))

        # Row 2 (4 elements): Model Override checkbox, Model Selection dropdown, Async Processing, Max Workers
        # Model override with checkbox + dropdown
        self.model_override_enabled = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Model override", variable=self.model_override_enabled, command=self._toggle_model_override).grid(row=1, column=0, sticky="w", padx=6, pady=6)

        self.model_selection_var = tk.StringVar(value="large-v3")
        self.model_selection_combo = ttk.Combobox(tab2, textvariable=self.model_selection_var, values=["large-v3", "large-v2", "turbo"], width=12, state="disabled")
        self.model_selection_combo.grid(row=1, column=1, sticky="w", padx=6, pady=6)

        self.async_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Async processing", variable=self.async_var).grid(row=1, column=2, sticky="w", padx=6, pady=6)

        workers_frame = ttk.Frame(tab2)
        workers_frame.grid(row=1, column=3, sticky="w", padx=6, pady=6)
        ttk.Label(workers_frame, text="Max workers:").pack(side=tk.LEFT)
        self.workers_var = tk.IntVar(value=1)
        ttk.Spinbox(workers_frame, from_=1, to=16, textvariable=self.workers_var, width=6).pack(side=tk.LEFT, padx=(4, 0))

        # Row 3 (3 elements): Opening Credit (spans 2 columns), Keep Temp Files, Temp Dir + Browse
        # Opening credit (single line entry with caption)
        credit_frame = ttk.Frame(tab2)
        credit_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=6)
        credit_frame.columnconfigure(0, weight=1)
        ttk.Label(credit_frame, text="Opening credit (Example: Produced by XXX):").grid(row=0, column=0, sticky="w")
        self.credit_var = tk.StringVar(value="")
        self.credit_entry = ttk.Entry(credit_frame, textvariable=self.credit_var)
        self.credit_entry.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        self.keep_temp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tab2, text="Keep temp files", variable=self.keep_temp_var).grid(row=2, column=2, sticky="w", padx=6, pady=6)

        temp_frame = ttk.Frame(tab2)
        temp_frame.grid(row=2, column=3, sticky="ew", padx=6, pady=6)
        temp_frame.columnconfigure(0, weight=1)
        ttk.Label(temp_frame, text="Temp dir:").grid(row=0, column=0, sticky="w")
        temp_entry_frame = ttk.Frame(temp_frame)
        temp_entry_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        temp_entry_frame.columnconfigure(0, weight=1)
        self.temp_var = tk.StringVar(value="")
        ttk.Entry(temp_entry_frame, textvariable=self.temp_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(temp_entry_frame, text="Browse", command=self._browse_temp_dir, width=8).grid(row=0, column=1, padx=(2, 0))

        # 4) Run controls
        run = ttk.Frame(frm)
        run.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        run.columnconfigure(1, weight=1)
        self.pbar = ttk.Progressbar(run, mode="indeterminate")
        self.pbar.grid(row=0, column=0, columnspan=4, sticky="ew", padx=6)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(run, textvariable=self.status_var).grid(row=0, column=4, sticky="e", padx=6)

        btns_run = ttk.Frame(run)
        btns_run.grid(row=1, column=0, columnspan=5, sticky="w", padx=0, pady=6)
        self.btn_start = ttk.Button(btns_run, text="Start", command=self.start)
        self.btn_cancel = ttk.Button(btns_run, text="Cancel", command=self.cancel, state=tk.DISABLED)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_cancel.pack(side=tk.LEFT, padx=8)

        # 5) Console/Log (expanded height)
        log_frame = ttk.LabelFrame(frm, text="Console")
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", height=21, state="disabled")
        self.log.grid(row=0, column=0, sticky="nsew")
        self._append_log("Ready.\n")

    def add_files(self):
        paths = filedialog.askopenfilenames(title="Select media files")
        if not paths:
            return
        for p in paths:
            if p not in self.inputs_list():
                self.inputs_listbox.insert(tk.END, p)

    def add_folder(self):
        path = filedialog.askdirectory(title="Select folder")
        if not path:
            return
        if path not in self.inputs_list():
            self.inputs_listbox.insert(tk.END, path)

    def remove_selected_inputs(self):
        sel = list(self.inputs_listbox.curselection())
        sel.reverse()  # delete from end
        for i in sel:
            self.inputs_listbox.delete(i)

    def clear_inputs(self):
        self.inputs_listbox.delete(0, tk.END)

    def inputs_list(self):
        return list(self.inputs_listbox.get(0, tk.END))

    def pick_output(self):
        path = filedialog.askdirectory(title="Select output directory", mustexist=True)
        if path:
            self.output_var.set(path)

    def open_output(self):
        path = Path(self.output_var.get() or ".")
        try:
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open output directory:\n{e}")

    def build_args(self, check_only=False):
        args = []
        if check_only:
            args += ["--check", "--check-verbose"]
            return args

        # Inputs come from listbox; pass as-is
        inputs = self.inputs_list()
        if not inputs:
            raise ValueError("Please add at least one file or folder.")
        args += inputs

        # Options
        args += ["--mode", self.mode_var.get()]
        args += ["--subs-language", self.lang_var.get()]
        args += ["--sensitivity", self.sens_var.get()]
        args += ["--output-dir", self.output_var.get()]

        if self.temp_var.get().strip():
            args += ["--temp-dir", self.temp_var.get().strip()]
        if self.keep_temp_var.get():
            args += ["--keep-temp"]
        # Note: no_progress_var removed - this option is no longer available
        if self.verbosity_var.get():
            args += ["--verbosity", self.verbosity_var.get()]

        if self.opt_adapt_cls.get():
            args += ["--adaptive-classification"]
        if self.opt_adapt_enh.get():
            args += ["--adaptive-audio-enhancement"]
        if self.opt_smart_post.get():
            args += ["--smart-postprocessing"]

        if self.async_var.get():
            args += ["--async-processing", "--max-workers", str(self.workers_var.get())]

        # Model override - only use if enabled
        if self.model_override_enabled.get():
            args += ["--model", self.model_selection_var.get()]

        # Opening credit text
        credit_text = self.credit_var.get().strip()
        if credit_text:
            args += ["--credit", credit_text]

        return args

    def _append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert(tk.END, text)
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def check_env(self):
        if self.proc:
            return
        self._run_subprocess(self.build_args(check_only=True), label="Checking environment...")

    def start(self):
        if self.proc:
            return
        try:
            args = self.build_args()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._run_subprocess(args, label="Running...")

    def cancel(self):
        if not self.proc:
            return
        self.status_var.set("Cancelling...")
        try:
            self.proc.terminate()
        except Exception:
            pass

    def _run_subprocess(self, args, label="Running..."):
        self.status_var.set(label)
        # Pretty print command (quote only paths/options with spaces)
        def q(a: str) -> str:
            return f'"{a}"' if (" " in a or "\t" in a) else a
        self._append_log("\n> whisperjav.main " + " ".join(q(a) for a in args) + "\n")

        cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args]

        # Force UTF-8 stdio in the child so logging can print ✓ and JP chars
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:replace"

        # Disable controls while running, enable Cancel and start pbar
        self._set_running_state(True)

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )

        def reader():
            try:
                for line in self.proc.stdout:
                    self._append_log(line)
            finally:
                code = self.proc.wait()
                self.after(0, self._on_proc_exit, code)

        threading.Thread(target=reader, daemon=True).start()

    def _on_proc_exit(self, code):
        self.proc = None
        if code == 0:
            self.status_var.set("Done")
            self._append_log("\nProcess completed successfully.\n")
        else:
            self.status_var.set(f"Exited ({code})")
            self._append_log(f"\nProcess exited with code {code}.\n")
        self._set_running_state(False)

    def _set_running_state(self, running: bool):
        # Start/stop progress bar and toggle buttons
        if running:
            try:
                self.pbar.start(12)
            except Exception:
                pass
            self.btn_start.configure(state=tk.DISABLED)
            self.btn_cancel.configure(state=tk.NORMAL)
        else:
            try:
                self.pbar.stop()
            except Exception:
                pass
            self.btn_start.configure(state=tk.NORMAL)
            self.btn_cancel.configure(state=tk.DISABLED)

    def _toggle_model_override(self):
        """Enable/disable the model selection dropdown based on override checkbox"""
        if self.model_override_enabled.get():
            self.model_selection_combo.configure(state="readonly")
        else:
            self.model_selection_combo.configure(state="disabled")
    
    def _browse_temp_dir(self):
        """Browse for temp directory"""
        path = filedialog.askdirectory(title="Select temporary directory")
        if path:
            self.temp_var.set(path)

    # (Presets removed by user request)

def main():
    """Entry point for the whisperjav-gui console script.
    Creates and runs the Tkinter application.
    """
    app = WhisperJAVGUI()
    app.mainloop()


if __name__ == "__main__":
    main()