import os
import sys
import shlex
import threading
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

# Use repo root so -m whisperjav.main resolves correctly
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "output"

class WhisperJAVGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("WhisperJAV – Simple Runner")
        self.geometry("1000x640")
        self.proc = None
        self._make_ui()

    def _make_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        # Layout grid weights for responsive design
        frm.columnconfigure(0, weight=1)
        # Make the Log section (row 5) expand so run buttons stay visible
        frm.rowconfigure(5, weight=1)

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

        # 3) Processing profile (common options)
        prof = ttk.LabelFrame(frm, text="Processing profile")
        prof.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        for c in range(8):
            prof.columnconfigure(c, weight=1 if c in (1, 3, 5) else 0)

        ttk.Label(prof, text="Mode:").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        self.mode_var = tk.StringVar(value="balanced")
        # Use small radio buttons for quick selection
        rb_frame = ttk.Frame(prof)
        rb_frame.grid(row=0, column=1, sticky="w")
        for val in ("balanced", "fast", "faster"):
            ttk.Radiobutton(rb_frame, text=val, value=val, variable=self.mode_var).pack(side=tk.LEFT, padx=(0, 8))

        ttk.Label(prof, text="Sensitivity:").grid(row=0, column=2, sticky="e", padx=6)
        self.sens_var = tk.StringVar(value="balanced")
        ttk.Combobox(prof, textvariable=self.sens_var, values=["conservative", "balanced", "aggressive"], width=14, state="readonly").grid(row=0, column=3, sticky="w")

        ttk.Label(prof, text="Output language:").grid(row=0, column=4, sticky="e", padx=6)
        self.lang_var = tk.StringVar(value="japanese")
        ttk.Combobox(prof, textvariable=self.lang_var, values=["japanese", "english-direct"], width=16, state="readonly").grid(row=0, column=5, sticky="w")

        # Info rows
        ttk.Label(
            prof,
            text="Speed vs. Accuracy: 'fast' and 'faster' prioritize throughput; 'balanced' favors accuracy.",
            style='Info.TLabel', wraplength=420, justify='left'
        ).grid(row=1, column=1, columnspan=2, sticky="w", padx=6, pady=(0, 6))
        ttk.Label(
            prof,
            text="Details vs. Noise: 'conservative' reduces false positives; 'aggressive' may include noise while capturing more detail.",
            style='Info.TLabel', wraplength=420, justify='left'
        ).grid(row=1, column=3, columnspan=3, sticky="w", padx=6, pady=(0, 6))

        # 4) Advanced (collapsible)
        adv_header = ttk.Frame(frm)
        adv_header.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        adv_header.columnconfigure(0, weight=1)
        self.adv_open = tk.BooleanVar(value=False)
        self._adv_btn = ttk.Button(adv_header, text="Show advanced ▸", command=self.toggle_advanced)
        self._adv_btn.grid(row=0, column=0, sticky="w")

        self.adv = ttk.LabelFrame(frm, text="Advanced options")
        # Do not grid initially; toggle_advanced will handle
        for c in range(6):
            self.adv.columnconfigure(c, weight=1 if c % 2 == 1 else 0)

        # Enhancements
        self.opt_adapt_cls = tk.BooleanVar(value=False)
        self.opt_adapt_enh = tk.BooleanVar(value=False)
        self.opt_smart_post = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.adv, text="Adaptive classification", variable=self.opt_adapt_cls).grid(row=0, column=0, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(self.adv, text="Adaptive audio enhancement", variable=self.opt_adapt_enh).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(self.adv, text="Smart postprocessing", variable=self.opt_smart_post).grid(row=0, column=2, sticky="w", padx=6, pady=6)

        # Concurrency
        self.async_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.adv, text="Async processing", variable=self.async_var).grid(row=1, column=0, sticky="w", padx=6)
        ttk.Label(self.adv, text="Max workers:").grid(row=1, column=1, sticky="e", padx=6)
        self.workers_var = tk.IntVar(value=1)
        ttk.Spinbox(self.adv, from_=1, to=16, textvariable=self.workers_var, width=6).grid(row=1, column=2, sticky="w")

        # Logging & progress
        ttk.Label(self.adv, text="Verbosity:").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        self.verbosity_var = tk.StringVar(value="summary")
        ttk.Combobox(self.adv, textvariable=self.verbosity_var, values=["quiet", "summary", "normal", "verbose"], width=12, state="readonly").grid(row=2, column=1, sticky="w", padx=0, pady=6)
        self.no_progress_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.adv, text="No progress bars", variable=self.no_progress_var).grid(row=2, column=2, sticky="w", padx=6)

        # Temp options
        self.keep_temp_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.adv, text="Keep temp files", variable=self.keep_temp_var).grid(row=3, column=0, sticky="w", padx=6)
        ttk.Label(self.adv, text="Temp dir:").grid(row=3, column=1, sticky="e", padx=6)
        self.temp_var = tk.StringVar(value="")
        ttk.Entry(self.adv, textvariable=self.temp_var, width=28).grid(row=3, column=2, sticky="w", padx=0, pady=6)

        # 5) Run controls
        run = ttk.Frame(frm)
        run.grid(row=4, column=0, sticky="ew", pady=(10, 0))
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

        # 6) Log
        log_frame = ttk.LabelFrame(frm, text="Log")
        log_frame.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, wrap="word", height=16, state="disabled")
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
        if self.no_progress_var.get():
            args += ["--no-progress"]
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

    def toggle_advanced(self):
        if self.adv_open.get():
            # Hide
            self.adv.grid_remove()
            self.adv_open.set(False)
            self._adv_btn.configure(text="Show advanced ▸")
        else:
            # Show
            self.adv.grid(row=3, column=0, sticky="ew")
            self.adv_open.set(True)
            self._adv_btn.configure(text="Hide advanced ▾")

    # (Presets removed by user request)

if __name__ == "__main__":
    app = WhisperJAVGUI()
    app.mainloop()