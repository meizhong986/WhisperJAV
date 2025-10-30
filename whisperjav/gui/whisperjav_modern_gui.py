"""
WhisperJAV GUI - Modernized Version
Uses CustomTkinter for enhanced visual design while maintaining compatibility
"""

import os
import sys
import json
import shlex
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

# Modern UI library - install with: pip install customtkinter
try:
    import customtkinter as ctk
    HAS_CTK = True
except ImportError:
    HAS_CTK = False
    print("Warning: customtkinter not installed. Using fallback mode.")
    print("Install with: pip install customtkinter")

# Always have tkinter available for variables/widgets used throughout
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Site root for module resolution
REPO_ROOT = Path(__file__).resolve().parents[2]

# Design tokens for consistent theming
DESIGN_TOKENS = {
    'colors': {
        'primary': '#0066CC',
        'primary_hover': '#0052A3',
        'success': '#16A34A',
        'error': '#DC2626',
        'warning': '#F59E0B',
        'surface': '#FFFFFF',
        'surface_elevated': '#F8FAFC',
        'surface_variant': '#F1F5F9',
        'outline': '#CBD5E1',
        'outline_variant': '#E2E8F0',
        'text_primary': '#0F172A',
        'text_secondary': '#475569',
        'text_disabled': '#94A3B8',
    },
    'spacing': {
        'xs': 4,
        'sm': 8,
        'md': 12,
        'lg': 16,
        'xl': 24,
        'xxl': 32,
    },
    'radius': {
        'sm': 4,
        'md': 8,
        'lg': 12,
    },
    'fonts': {
        'heading': ('Segoe UI', 14, 'bold'),
        'body': ('Segoe UI', 11),
        'caption': ('Segoe UI', 10),
        'mono': ('Consolas', 10),
    }
}

@dataclass
class AppSettings:
    """Application settings with defaults"""
    theme: str = 'system'
    last_output_dir: str = ''
    mode: str = 'balanced'
    sensitivity: str = 'balanced'
    language: str = 'japanese'
    verbosity: str = 'summary'
    window_geometry: Optional[str] = None
    recent_inputs: List[str] = None
    
    def __post_init__(self):
        if self.recent_inputs is None:
            self.recent_inputs = []
        if not self.last_output_dir:
            self.last_output_dir = str(self._get_default_output())
    
    @staticmethod
    def _get_default_output() -> Path:
        """Get default output directory"""
        home = Path.home()
        if sys.platform.startswith("win"):
            docs = home / "Documents"
        else:
            docs = home / "Documents" if (home / "Documents").exists() else home
        return docs / "WhisperJAV" / "output"


class SettingsManager:
    """Manages persistent application settings"""
    
    def __init__(self):
        self.config_dir = Path.home() / '.whisperjav'
        self.config_file = self.config_dir / 'config.json'
        self.settings = self.load()
    
    def load(self) -> AppSettings:
        """Load settings from disk"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    return AppSettings(**data)
            except Exception as e:
                print(f"Error loading settings: {e}")
        return AppSettings()
    
    def save(self):
        """Save settings to disk"""
        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(asdict(self.settings), f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")


class ModernWhisperGUI(ctk.CTk if HAS_CTK else tk.Tk):
    """Modern WhisperJAV GUI with enhanced UX"""
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings
        self.settings_manager = SettingsManager()
        self.settings = self.settings_manager.settings
        
        # Setup window
        self.title("WhisperJAV - Modern Transcription Tool")
        self.setup_window()
        
        # Process state
        self.proc = None
        self.start_time = None
        
        # Create UI
        self.create_ui()
        
        # Apply saved settings
        self.apply_settings()
        
        # Bind keyboard shortcuts
        self.setup_keyboard_shortcuts()
    
    def setup_window(self):
        """Configure main window"""
        # Set window size and position
        if self.settings.window_geometry:
            self.geometry(self.settings.window_geometry)
        else:
            # Center window on screen
            self.geometry("1100x720")
            self.center_window()
        
        # Set theme
        if HAS_CTK:
            ctk.set_appearance_mode(self.settings.theme)
            ctk.set_default_color_theme("blue")
        
        # Set minimum window size
        self.minsize(900, 600)
        
        # Save geometry on close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def center_window(self):
        """Center window on screen"""
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_ui(self):
        """Create the main UI"""
        # Main container with padding
        if HAS_CTK:
            main_frame = ctk.CTkFrame(self, corner_radius=0)
            main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        else:
            main_frame = ttk.Frame(self, padding=20)
            main_frame.pack(fill="both", expand=True)
        
        # Configure grid weights
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(3, weight=1)  # Console expands
        
        # 1. Source Section
        self.create_source_section(main_frame)
        
        # 2. Settings Section
        self.create_settings_section(main_frame)
        
        # 3. Control Section
        self.create_control_section(main_frame)
        
        # 4. Console Section
        self.create_console_section(main_frame)
    
    def create_source_section(self, parent):
        """Create source files section with drag & drop"""
        if HAS_CTK:
            frame = ctk.CTkFrame(parent, corner_radius=10)
            frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            
            # Title
            ctk.CTkLabel(frame, text="Source Files", 
                        font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
            
            # File list (custom implementation needed for drag-drop)
            list_frame = ctk.CTkFrame(frame, height=100)
            list_frame.pack(fill="both", expand=True, padx=15, pady=5)
            
            # For now, use traditional listbox inside CTk frame
            self.file_listbox = tk.Listbox(list_frame, selectmode=tk.EXTENDED,
                                          bg='#212121' if self.get_appearance_mode() == 'dark' else 'white',
                                          fg='white' if self.get_appearance_mode() == 'dark' else 'black')
            self.file_listbox.pack(fill="both", expand=True)
            
            # Buttons
            btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
            btn_frame.pack(fill="x", padx=15, pady=(5, 10))
            
            ctk.CTkButton(btn_frame, text="Add Files", width=100,
                         command=self.add_files).pack(side="left", padx=2)
            ctk.CTkButton(btn_frame, text="Add Folder", width=100,
                         command=self.add_folder).pack(side="left", padx=2)
            ctk.CTkButton(btn_frame, text="Remove", width=100,
                         command=self.remove_selected).pack(side="left", padx=2)
            ctk.CTkButton(btn_frame, text="Clear All", width=100,
                         command=self.clear_files).pack(side="left", padx=2)
        else:
            # Fallback to ttk
            frame = ttk.LabelFrame(parent, text="Source Files", padding=10)
            frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
            
            self.file_listbox = tk.Listbox(frame, height=5, selectmode=tk.EXTENDED)
            self.file_listbox.pack(fill="both", expand=True)
            
            btn_frame = ttk.Frame(frame)
            btn_frame.pack(fill="x", pady=(5, 0))
            
            ttk.Button(btn_frame, text="Add Files", command=self.add_files).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Add Folder", command=self.add_folder).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Remove", command=self.remove_selected).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Clear All", command=self.clear_files).pack(side="left", padx=2)
    
    def create_settings_section(self, parent):
        """Create tabbed settings section"""
        if HAS_CTK:
            frame = ctk.CTkFrame(parent, corner_radius=10)
            frame.grid(row=1, column=0, sticky="ew", pady=10)
            
            # Tabview
            tabview = ctk.CTkTabview(frame, corner_radius=8)
            tabview.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Basic Settings Tab
            basic_tab = tabview.add("Basic Settings")
            self.create_basic_settings(basic_tab)
            
            # Advanced Tab
            advanced_tab = tabview.add("Advanced")
            self.create_advanced_settings(advanced_tab)
            
            # Output Tab
            output_tab = tabview.add("Output")
            self.create_output_settings(output_tab)
        else:
            # Fallback to ttk notebook
            notebook = ttk.Notebook(parent)
            notebook.grid(row=1, column=0, sticky="ew", pady=10)
            
            basic_tab = ttk.Frame(notebook, padding=10)
            notebook.add(basic_tab, text="Basic Settings")
            self.create_basic_settings(basic_tab)
            
            advanced_tab = ttk.Frame(notebook, padding=10)
            notebook.add(advanced_tab, text="Advanced")
            self.create_advanced_settings(advanced_tab)
            
            output_tab = ttk.Frame(notebook, padding=10)
            notebook.add(output_tab, text="Output")
            self.create_output_settings(output_tab)
    
    def create_basic_settings(self, parent):
        """Create basic settings controls"""
        # Mode selection
        if HAS_CTK:
            ctk.CTkLabel(parent, text="Transcription Mode:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
            self.mode_var = tk.StringVar(value=self.settings.mode)
            mode_menu = ctk.CTkSegmentedButton(parent, values=["fast", "balanced", "accurate"],
                                              variable=self.mode_var)
            mode_menu.grid(row=0, column=1, sticky="w", padx=10, pady=5)
            
            # Sensitivity
            ctk.CTkLabel(parent, text="Sensitivity:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
            self.sensitivity_var = tk.StringVar(value=self.settings.sensitivity)
            sens_menu = ctk.CTkOptionMenu(parent, values=["conservative", "balanced", "aggressive"],
                                         variable=self.sensitivity_var)
            sens_menu.grid(row=1, column=1, sticky="w", padx=10, pady=5)
            
            # Language
            ctk.CTkLabel(parent, text="Output Language:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
            self.language_var = tk.StringVar(value=self.settings.language)
            lang_menu = ctk.CTkOptionMenu(parent, values=["japanese", "english-direct"],
                                        variable=self.language_var)
            lang_menu.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        else:
            # Fallback
            ttk.Label(parent, text="Mode:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            self.mode_var = tk.StringVar(value=self.settings.mode)
            ttk.Combobox(parent, textvariable=self.mode_var,
                        values=["fast", "balanced", "accurate"],
                        state="readonly").grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(parent, text="Sensitivity:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
            self.sensitivity_var = tk.StringVar(value=self.settings.sensitivity)
            ttk.Combobox(parent, textvariable=self.sensitivity_var,
                        values=["conservative", "balanced", "aggressive"],
                        state="readonly").grid(row=1, column=1, sticky="w", padx=5, pady=5)
            
            ttk.Label(parent, text="Language:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            self.language_var = tk.StringVar(value=self.settings.language)
            ttk.Combobox(parent, textvariable=self.language_var,
                        values=["japanese", "english-direct"],
                        state="readonly").grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    def create_advanced_settings(self, parent):
        """Create advanced settings controls"""
        if HAS_CTK:
            # Model override
            self.model_override_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(parent, text="Override Model",
                           variable=self.model_override_var,
                           command=self.toggle_model_override).grid(row=0, column=0, sticky="w", padx=10, pady=5)
            
            self.model_var = tk.StringVar(value="large-v3")
            self.model_menu = ctk.CTkOptionMenu(parent, values=["large-v3", "large-v2", "turbo"],
                                               variable=self.model_var, state="disabled")
            self.model_menu.grid(row=0, column=1, sticky="w", padx=10, pady=5)
            
            # Async processing
            self.async_var = tk.BooleanVar(value=False)
            ctk.CTkCheckBox(parent, text="Async Processing",
                           variable=self.async_var).grid(row=1, column=0, sticky="w", padx=10, pady=5)
            
            # Workers
            ctk.CTkLabel(parent, text="Max Workers:").grid(row=1, column=1, sticky="w", padx=10, pady=5)
            self.workers_var = tk.IntVar(value=1)
            worker_slider = ctk.CTkSlider(parent, from_=1, to=16, variable=self.workers_var)
            worker_slider.grid(row=1, column=2, sticky="w", padx=10, pady=5)
            self.worker_label = ctk.CTkLabel(parent, text="1")
            self.worker_label.grid(row=1, column=3, sticky="w", padx=5, pady=5)
            worker_slider.configure(command=lambda v: self.worker_label.configure(text=str(int(v))))
            
            # Verbosity
            ctk.CTkLabel(parent, text="Verbosity:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
            self.verbosity_var = tk.StringVar(value=self.settings.verbosity)
            verb_menu = ctk.CTkOptionMenu(parent, values=["quiet", "summary", "normal", "verbose"],
                                        variable=self.verbosity_var)
            verb_menu.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        else:
            # Fallback implementation
            self.model_override_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(parent, text="Override Model",
                           variable=self.model_override_var).grid(row=0, column=0, sticky="w", padx=5, pady=5)
            
            self.model_var = tk.StringVar(value="large-v3")
            self.model_menu = ttk.Combobox(parent, textvariable=self.model_var,
                                          values=["large-v3", "large-v2", "turbo"],
                                          state="disabled")
            self.model_menu.grid(row=0, column=1, sticky="w", padx=5, pady=5)
            
            self.async_var = tk.BooleanVar(value=False)
            ttk.Checkbutton(parent, text="Async Processing",
                           variable=self.async_var).grid(row=1, column=0, sticky="w", padx=5, pady=5)
            
            self.workers_var = tk.IntVar(value=1)
            ttk.Label(parent, text="Workers:").grid(row=1, column=1, sticky="w", padx=5, pady=5)
            ttk.Spinbox(parent, from_=1, to=16, textvariable=self.workers_var,
                       width=10).grid(row=1, column=2, sticky="w", padx=5, pady=5)
            
            self.verbosity_var = tk.StringVar(value=self.settings.verbosity)
            ttk.Label(parent, text="Verbosity:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
            ttk.Combobox(parent, textvariable=self.verbosity_var,
                        values=["quiet", "summary", "normal", "verbose"],
                        state="readonly").grid(row=2, column=1, sticky="w", padx=5, pady=5)
    
    def create_output_settings(self, parent):
        """Create output directory settings"""
        if HAS_CTK:
            ctk.CTkLabel(parent, text="Output Directory:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
            
            self.output_var = tk.StringVar(value=self.settings.last_output_dir)
            entry = ctk.CTkEntry(parent, textvariable=self.output_var, width=400)
            entry.grid(row=1, column=0, sticky="ew", padx=10, pady=5)
            
            btn_frame = ctk.CTkFrame(parent, fg_color="transparent")
            btn_frame.grid(row=2, column=0, sticky="w", padx=10, pady=5)
            
            ctk.CTkButton(btn_frame, text="Browse", width=100,
                         command=self.browse_output).pack(side="left", padx=2)
            ctk.CTkButton(btn_frame, text="Open Folder", width=100,
                         command=self.open_output).pack(side="left", padx=2)
            ctk.CTkButton(btn_frame, text="Reset Default", width=100,
                         command=self.reset_output).pack(side="left", padx=2)
        else:
            ttk.Label(parent, text="Output Directory:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
            
            self.output_var = tk.StringVar(value=self.settings.last_output_dir)
            ttk.Entry(parent, textvariable=self.output_var, width=50).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
            
            btn_frame = ttk.Frame(parent)
            btn_frame.grid(row=2, column=0, sticky="w", padx=5, pady=5)
            
            ttk.Button(btn_frame, text="Browse", command=self.browse_output).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Open", command=self.open_output).pack(side="left", padx=2)
            ttk.Button(btn_frame, text="Default", command=self.reset_output).pack(side="left", padx=2)
    
    def create_control_section(self, parent):
        """Create run controls with progress"""
        if HAS_CTK:
            frame = ctk.CTkFrame(parent, corner_radius=10, height=80)
            frame.grid(row=2, column=0, sticky="ew", pady=10)
            frame.grid_propagate(False)
            
            # Progress bar
            self.progress = ctk.CTkProgressBar(frame, width=500)
            self.progress.pack(pady=(15, 5))
            self.progress.set(0)
            
            # Control buttons and status
            control_frame = ctk.CTkFrame(frame, fg_color="transparent")
            control_frame.pack()
            
            self.start_btn = ctk.CTkButton(control_frame, text="Start Processing",
                                         command=self.start_processing, width=150)
            self.start_btn.pack(side="left", padx=5)
            
            self.cancel_btn = ctk.CTkButton(control_frame, text="Cancel",
                                          command=self.cancel_processing,
                                          state="disabled", width=100)
            self.cancel_btn.pack(side="left", padx=5)
            
            self.status_label = ctk.CTkLabel(control_frame, text="Ready")
            self.status_label.pack(side="left", padx=20)
        else:
            frame = ttk.LabelFrame(parent, text="Controls", padding=10)
            frame.grid(row=2, column=0, sticky="ew", pady=10)
            
            self.progress = ttk.Progressbar(frame, length=400, mode='indeterminate')
            self.progress.pack(pady=5)
            
            control_frame = ttk.Frame(frame)
            control_frame.pack()
            
            self.start_btn = ttk.Button(control_frame, text="Start", command=self.start_processing)
            self.start_btn.pack(side="left", padx=5)
            
            self.cancel_btn = ttk.Button(control_frame, text="Cancel",
                                        command=self.cancel_processing, state="disabled")
            self.cancel_btn.pack(side="left", padx=5)
            
            self.status_label = ttk.Label(control_frame, text="Ready")
            self.status_label.pack(side="left", padx=20)
    
    def create_console_section(self, parent):
        """Create console output section"""
        if HAS_CTK:
            frame = ctk.CTkFrame(parent, corner_radius=10)
            frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
            
            ctk.CTkLabel(frame, text="Console Output",
                        font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=15, pady=(10, 5))
            
            self.console = ctk.CTkTextbox(frame, width=800, height=200,
                                         font=ctk.CTkFont(family="Consolas", size=10))
            self.console.pack(fill="both", expand=True, padx=15, pady=(5, 15))
        else:
            frame = ttk.LabelFrame(parent, text="Console Output", padding=10)
            frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
            
            self.console = tk.Text(frame, wrap="word", height=12, font=('Consolas', 10))
            self.console.pack(fill="both", expand=True)
            
            scrollbar = ttk.Scrollbar(self.console)
            scrollbar.pack(side="right", fill="y")
            self.console.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=self.console.yview)
    
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.bind('<Control-o>', lambda e: self.add_files())
        self.bind('<Control-d>', lambda e: self.add_folder())
        self.bind('<Control-s>', lambda e: self.start_processing())
        self.bind('<Escape>', lambda e: self.cancel_processing() if self.proc else None)
        self.bind('<F1>', lambda e: self.show_help())
        self.bind('<Control-q>', lambda e: self.on_closing())
    
    def get_appearance_mode(self):
        """Get current appearance mode"""
        if HAS_CTK:
            return ctk.get_appearance_mode().lower()
        return 'light'
    
    def toggle_model_override(self):
        """Toggle model override state"""
        if HAS_CTK:
            state = "normal" if self.model_override_var.get() else "disabled"
            self.model_menu.configure(state=state)
        else:
            state = "readonly" if self.model_override_var.get() else "disabled"
            self.model_menu.configure(state=state)
    
    def add_files(self):
        """Add files to process"""
        files = filedialog.askopenfilenames(
            title="Select media files",
            filetypes=[("Media files", "*.mp4 *.avi *.mkv *.mov *.mp3 *.wav *.m4a"),
                      ("All files", "*.*")]
        )
        for file in files:
            if file not in self.get_input_files():
                self.file_listbox.insert(tk.END, file)
    
    def add_folder(self):
        """Add folder to process"""
        folder = filedialog.askdirectory(title="Select folder containing media files")
        if folder and folder not in self.get_input_files():
            self.file_listbox.insert(tk.END, folder)
    
    def remove_selected(self):
        """Remove selected files"""
        selection = self.file_listbox.curselection()
        for index in reversed(selection):
            self.file_listbox.delete(index)
    
    def clear_files(self):
        """Clear all files"""
        self.file_listbox.delete(0, tk.END)
    
    def get_input_files(self):
        """Get list of input files"""
        return list(self.file_listbox.get(0, tk.END))
    
    def browse_output(self):
        """Browse for output directory"""
        folder = filedialog.askdirectory(title="Select output directory")
        if folder:
            self.output_var.set(folder)
    
    def open_output(self):
        """Open output directory in file explorer"""
        output_dir = Path(self.output_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if sys.platform.startswith('win'):
            os.startfile(str(output_dir))
        elif sys.platform == 'darwin':
            subprocess.run(['open', str(output_dir)])
        else:
            subprocess.run(['xdg-open', str(output_dir)])
    
    def reset_output(self):
        """Reset output directory to default"""
        self.output_var.set(str(AppSettings._get_default_output()))
    
    def validate_inputs(self):
        """Validate inputs before processing"""
        errors = []
        
        # Check for input files
        if not self.get_input_files():
            errors.append("• No input files selected")
        
        # Validate file paths
        for path in self.get_input_files():
            if not Path(path).exists():
                errors.append(f"• File not found: {path}")
        
        # Check output directory
        output = Path(self.output_var.get())
        if not output.parent.exists():
            errors.append(f"• Output directory parent doesn't exist: {output.parent}")
        
        # Model-specific validations
        if self.model_override_var.get() and self.model_var.get() == "turbo":
            if self.language_var.get() != "english-direct":
                errors.append("• Turbo model only supports English output")
        
        if errors:
            messagebox.showerror("Validation Error",
                               "Please fix the following issues:\n\n" + "\n".join(errors))
            return False
        return True
    
    def build_command_args(self):
        """Build command line arguments"""
        args = []
        
        # Input files
        args.extend(self.get_input_files())
        
        # Basic settings
        args.extend(["--mode", self.mode_var.get()])
        args.extend(["--sensitivity", self.sensitivity_var.get()])
        args.extend(["--subs-language", self.language_var.get()])
        args.extend(["--output-dir", self.output_var.get()])
        args.extend(["--verbosity", self.verbosity_var.get()])
        
        # Advanced settings
        if self.model_override_var.get():
            args.extend(["--model", self.model_var.get()])
        
        if self.async_var.get():
            args.extend(["--async-processing", "--max-workers", str(self.workers_var.get())])
        
        return args
    
    def start_processing(self):
        """Start processing files"""
        if self.proc:
            return
        
        if not self.validate_inputs():
            return
        
        # Save settings before starting
        self.save_current_settings()
        
        # Build command
        args = self.build_command_args()
        cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main"] + args
        
        # Log command
        self.log_to_console(f"\n> Starting processing...\n> Command: {' '.join(args[:50])}...\n")
        
        # Update UI state
        self.set_processing_state(True)
        self.start_time = datetime.now()
        
        # Start subprocess
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONIOENCODING"] = "utf-8:replace"
        
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",
            env=env
        )
        
        # Start reader thread
        thread = threading.Thread(target=self.read_process_output, daemon=True)
        thread.start()
    
    def read_process_output(self):
        """Read process output in background thread"""
        try:
            for line in self.proc.stdout:
                self.after(0, self.log_to_console, line)
        finally:
            exit_code = self.proc.wait()
            self.after(0, self.on_process_complete, exit_code)
    
    def cancel_processing(self):
        """Cancel current processing"""
        if self.proc:
            self.log_to_console("\n> Cancelling...\n")
            try:
                self.proc.terminate()
            except:
                pass
    
    def on_process_complete(self, exit_code):
        """Handle process completion"""
        self.proc = None
        elapsed = datetime.now() - self.start_time if self.start_time else None
        
        if exit_code == 0:
            msg = f"\n✓ Processing completed successfully"
            if elapsed:
                msg += f" (Time: {elapsed.total_seconds():.1f}s)"
            msg += "\n"
            self.log_to_console(msg)
            if HAS_CTK:
                self.status_label.configure(text="Complete")
        else:
            self.log_to_console(f"\n✗ Process exited with code {exit_code}\n")
            if HAS_CTK:
                self.status_label.configure(text=f"Failed ({exit_code})")
        
        self.set_processing_state(False)
    
    def set_processing_state(self, processing):
        """Update UI for processing state"""
        if processing:
            if HAS_CTK:
                self.start_btn.configure(state="disabled")
                self.cancel_btn.configure(state="normal")
                self.status_label.configure(text="Processing...")
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            else:
                self.start_btn.configure(state="disabled")
                self.cancel_btn.configure(state="normal")
                self.status_label.configure(text="Processing...")
                self.progress.start()
        else:
            if HAS_CTK:
                self.start_btn.configure(state="normal")
                self.cancel_btn.configure(state="disabled")
                self.progress.stop()
                self.progress.set(0)
            else:
                self.start_btn.configure(state="normal")
                self.cancel_btn.configure(state="disabled")
                self.progress.stop()
    
    def log_to_console(self, text):
        """Log text to console"""
        if HAS_CTK:
            self.console.insert("end", text)
            self.console.see("end")
        else:
            self.console.insert(tk.END, text)
            self.console.see(tk.END)
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
WhisperJAV - Modern Transcription Tool

Keyboard Shortcuts:
• Ctrl+O: Add files
• Ctrl+D: Add folder  
• Ctrl+S: Start processing
• Escape: Cancel processing
• F1: Show this help
• Ctrl+Q: Quit

Tips:
• Drag and drop files directly onto the file list (coming soon)
• Settings are automatically saved
• Check console for detailed progress
        """
        messagebox.showinfo("Help", help_text)
    
    def save_current_settings(self):
        """Save current settings"""
        self.settings.mode = self.mode_var.get()
        self.settings.sensitivity = self.sensitivity_var.get()
        self.settings.language = self.language_var.get()
        self.settings.verbosity = self.verbosity_var.get()
        self.settings.last_output_dir = self.output_var.get()
        self.settings.recent_inputs = self.get_input_files()[:10]  # Save last 10
        self.settings_manager.save()
    
    def apply_settings(self):
        """Apply loaded settings to UI"""
        # Restore recent files
        for file in self.settings.recent_inputs[:5]:
            if Path(file).exists():
                self.file_listbox.insert(tk.END, file)
    
    def on_closing(self):
        """Handle window closing"""
        # Save window geometry
        self.settings.window_geometry = self.geometry()
        self.save_current_settings()
        
        # Cancel any running process
        if self.proc:
            self.proc.terminate()
        
        self.destroy()


def main():
    """Main entry point"""
    app = ModernWhisperGUI()
    app.mainloop()


if __name__ == "__main__":
    main()