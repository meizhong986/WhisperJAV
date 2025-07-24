import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
import sys

# ==============================================================================
# This is the TranscriptionTunerV3 class from your provided file.
# It's included here to make the application self-contained.
# A dummy logger is created to avoid dependency issues.
# ==============================================================================

class DummyLogger:
    def debug(self, msg):
        print(f"DEBUG: {msg}")
    def info(self, msg):
        print(f"INFO: {msg}")
    def warning(self, msg):
        print(f"WARNING: {msg}")
    def error(self, msg):
        print(f"ERROR: {msg}")

logger = DummyLogger()

class TranscriptionTunerV3:
    """
    Configuration resolver for WhisperJAV v3.
    Reads the modular asr_config.v3.json to produce a structured config object.
    This class is the single source of truth for resolving run configurations.
    """
    def __init__(self, config_path: Optional[Path] = None, config_data: Optional[Dict] = None):
        """
        Initializes the tuner by loading the v3 configuration file or data.
        
        Args:
            config_path: Optional path to a custom asr_config.v3.json.
            config_data: Optional dictionary containing the config.
        """
        if config_data:
            self.config = config_data
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Either config_path or config_data must be provided.")


    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Loads and validates the v3 configuration file."""
        if not config_path:
            # In the context of this app, we ensure the path is passed.
            # This is fallback logic.
            app_dir = Path(sys.argv[0]).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
            config_path = app_dir / "asr_config.v3.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"ASR configuration file not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            # A more robust loader that strips comments.
            lines = f.readlines()
            valid_json_lines = [line for line in lines if not line.strip().startswith("//")]
            config = json.loads("".join(valid_json_lines))
            
        if config.get("version") != "3.0":
            raise ValueError(f"Unsupported config version: {config.get('version')}. Expected 3.0.")
            
        logger.debug(f"Successfully loaded {config_path}")
        return config

    def resolve_params(self, pipeline_name: str, sensitivity: str, task: str) -> Dict[str, Any]:
        """
        Resolves all necessary configurations for a given run into a single,
        structured dictionary.
        """
        if pipeline_name not in self.config["pipelines"]:
            raise ValueError(f"Unknown pipeline specified: '{pipeline_name}'")
        if sensitivity not in self.config["sensitivity_profiles"]:
             raise ValueError(f"Unknown sensitivity specified: '{sensitivity}'")
        
        pipeline_cfg = self.config["pipelines"][pipeline_name]
        sensitivity_profile = self.config["sensitivity_profiles"][sensitivity]

        model_id = pipeline_cfg.get("model_overrides", {}).get(task, pipeline_cfg["workflow"]["model"])
        if model_id not in self.config["models"]:
            raise ValueError(f"Model '{model_id}' not found in configuration.")

        model_cfg = self.config["models"][model_id]
        provider = model_cfg["provider"]

        params = {
            "decoder": deepcopy(self.config["parameter_sets"]["decoder_params"][sensitivity_profile["decoder"]]),
            "vad": deepcopy(self.config["parameter_sets"]["vad_params"][sensitivity_profile["vad"]]),
            "provider": deepcopy(self.config["parameter_sets"]["provider_specific_params"][provider][sensitivity_profile["provider_settings"]])
        }

        features = {}
        for feature, setting in pipeline_cfg["workflow"].get("features", {}).items():
            if setting and setting != "none":
                config_name = setting if isinstance(setting, str) else "default"
                if feature in self.config["feature_configs"] and config_name in self.config["feature_configs"][feature]:
                    features[feature] = deepcopy(self.config["feature_configs"][feature][config_name])
        
        return {
            "pipeline_name": pipeline_name,
            "sensitivity_name": sensitivity,
            "workflow": pipeline_cfg["workflow"],
            "model": model_cfg,
            "params": params,
            "features": features,
            "task": task,
            "language": self.config.get("defaults", {}).get("language", "ja")
        }

# ==============================================================================
# Main Tkinter Application
# ==============================================================================

class App(tk.Tk):
    def __init__(self, config_path):
        super().__init__()
        self.title("ASR Configurator v3")
        self.geometry("1400x900")

        self.config_path = Path(config_path)
        self.config_data = self.load_initial_config()
        if not self.config_data:
            self.destroy()
            return
            
        self.tuner = TranscriptionTunerV3(config_data=self.config_data)

        # --- Main Layout ---
        main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- Column 1: Run Configuration ---
        run_config_frame = self.create_scrollable_frame(main_paned_window)
        main_paned_window.add(run_config_frame, weight=1)
        self.populate_run_config_frame(run_config_frame)

        # --- Column 2: Resolved Settings ---
        self.resolved_settings_frame = self.create_scrollable_frame(main_paned_window)
        main_paned_window.add(self.resolved_settings_frame, weight=2)
        
        # --- Column 3: Global Settings ---
        global_settings_frame = self.create_scrollable_frame(main_paned_window)
        main_paned_window.add(global_settings_frame, weight=2)
        self.populate_global_settings_frame(global_settings_frame)
        
        # --- Bottom Bar for Saving ---
        save_bar = ttk.Frame(self)
        save_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=(0, 10))
        ttk.Button(save_bar, text="Save", command=self.save_config).pack(side=tk.RIGHT, padx=5)
        ttk.Button(save_bar, text="Save As...", command=self.save_config_as).pack(side=tk.RIGHT)
        ttk.Button(save_bar, text="Reload from Disk", command=self.reload_config).pack(side=tk.LEFT)

        # Initial population
        self.update_resolved_display()

    def load_initial_config(self):
        """Loads the JSON config file, handling errors."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                valid_json_lines = [line for line in lines if not line.strip().startswith("//")]
                return json.loads("".join(valid_json_lines))
        except FileNotFoundError:
            messagebox.showerror("Error", f"Configuration file not found:\n{self.config_path}\n\nPlease ensure it's in the same directory as the script.")
            return None
        except json.JSONDecodeError as e:
            messagebox.showerror("Error", f"Error parsing JSON file:\n{e}")
            return None

    def create_scrollable_frame(self, parent):
        """Creates a frame with a vertical scrollbar."""
        container = ttk.Frame(parent)
        canvas = tk.Canvas(container, borderwidth=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        container.pack(fill=tk.BOTH, expand=True)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        return scrollable_frame

    def populate_run_config_frame(self, parent):
        """Fills the first column with high-level controls."""
        # --- Variables ---
        self.pipeline_var = tk.StringVar(value=list(self.config_data['pipelines'].keys())[0])
        self.sensitivity_var = tk.StringVar(value=list(self.config_data['sensitivity_profiles'].keys())[1]) # Default to balanced
        self.task_var = tk.StringVar(value='transcribe')

        # --- Widgets ---
        # Pipeline
        pipeline_lf = ttk.LabelFrame(parent, text="1. Select Pipeline", padding=10)
        pipeline_lf.pack(fill=tk.X, padx=10, pady=5)
        for name in self.config_data['pipelines'].keys():
            ttk.Radiobutton(pipeline_lf, text=name.capitalize(), variable=self.pipeline_var, value=name, command=self.update_resolved_display).pack(anchor=tk.W)

        # Sensitivity
        sensitivity_lf = ttk.LabelFrame(parent, text="2. Select Sensitivity", padding=10)
        sensitivity_lf.pack(fill=tk.X, padx=10, pady=5)
        for name in self.config_data['sensitivity_profiles'].keys():
            ttk.Radiobutton(sensitivity_lf, text=name.capitalize(), variable=self.sensitivity_var, value=name, command=self.update_resolved_display).pack(anchor=tk.W)

        # Task
        task_lf = ttk.LabelFrame(parent, text="3. Select Task", padding=10)
        task_lf.pack(fill=tk.X, padx=10, pady=5)
        ttk.Radiobutton(task_lf, text="Transcribe", variable=self.task_var, value='transcribe', command=self.update_resolved_display).pack(anchor=tk.W)
        ttk.Radiobutton(task_lf, text="Translate", variable=self.task_var, value='translate', command=self.update_resolved_display).pack(anchor=tk.W)

    def update_resolved_display(self):
        """Clears and repopulates the 'Resolved Settings' column."""
        # Clear previous widgets
        for widget in self.resolved_settings_frame.winfo_children():
            widget.destroy()
        
        # Get resolved parameters
        try:
            params = self.tuner.resolve_params(
                self.pipeline_var.get(),
                self.sensitivity_var.get(),
                self.task_var.get()
            )
        except (ValueError, KeyError) as e:
            ttk.Label(self.resolved_settings_frame, text=f"Error: {e}", foreground="red", wraplength=300).pack(anchor=tk.W, padx=10, pady=10)
            return

        # Display new parameters
        self.display_dict_in_frame(self.resolved_settings_frame, params, "Resolved Settings")

    def display_dict_in_frame(self, parent, data_dict, title):
        """Recursively displays a dictionary in a frame."""
        lf = ttk.LabelFrame(parent, text=title, padding=10)
        lf.pack(fill=tk.X, padx=10, pady=5, anchor=tk.N)

        for key, value in data_dict.items():
            if isinstance(value, dict):
                self.display_dict_in_frame(lf, value, key.replace("_", " ").capitalize())
            else:
                row_frame = ttk.Frame(lf)
                row_frame.pack(fill=tk.X, pady=2)
                ttk.Label(row_frame, text=f"{key.replace('_', ' ').capitalize()}:", font=("", 10, "bold")).pack(side=tk.LEFT, anchor=tk.W)
                
                # Convert value to string for display
                display_value = str(value)
                if isinstance(value, list):
                    display_value = ", ".join(map(str, value))

                ttk.Label(row_frame, text=display_value, wraplength=250, justify=tk.LEFT).pack(side=tk.RIGHT, anchor=tk.E, expand=True, fill=tk.X)

    def populate_global_settings_frame(self, parent):
        """Fills the third column with editable global settings."""
        self.global_vars = {} # To store StringVars for Entry widgets
        
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- Parameter Sets Tab ---
        param_sets_frame = self.create_scrollable_frame(notebook)
        notebook.add(param_sets_frame, text="Parameter Sets")
        self.create_editable_dict_view(param_sets_frame, self.config_data['parameter_sets'], ['parameter_sets'])

        # --- Models Tab ---
        models_frame = self.create_scrollable_frame(notebook)
        notebook.add(models_frame, text="Models")
        self.create_editable_dict_view(models_frame, self.config_data['models'], ['models'])

        # --- Feature Configs Tab ---
        features_frame = self.create_scrollable_frame(notebook)
        notebook.add(features_frame, text="Feature Configs")
        self.create_editable_dict_view(features_frame, self.config_data['feature_configs'], ['feature_configs'])
        
        # --- UI Preferences Tab ---
        ui_prefs_frame = self.create_scrollable_frame(notebook)
        notebook.add(ui_prefs_frame, text="UI Preferences")
        self.create_editable_dict_view(ui_prefs_frame, self.config_data['ui_preferences'], ['ui_preferences'])

    def create_editable_dict_view(self, parent, data_dict, path):
        """Recursively creates editable fields for a dictionary."""
        for key, value in data_dict.items():
            key_path = path + [key]
            if isinstance(value, dict):
                lf = ttk.LabelFrame(parent, text=key.replace("_", " ").capitalize(), padding=10)
                lf.pack(fill=tk.X, padx=5, pady=5, anchor=tk.N)
                self.create_editable_dict_view(lf, value, key_path)
            else:
                row_frame = ttk.Frame(parent)
                row_frame.pack(fill=tk.X, pady=2)
                ttk.Label(row_frame, text=f"{key.replace('_', ' ').capitalize()}:").pack(side=tk.LEFT, anchor=tk.W, padx=5)
                
                var = tk.StringVar(value=json.dumps(value) if isinstance(value, (list, bool)) else value)
                entry = ttk.Entry(row_frame, textvariable=var)
                entry.pack(side=tk.RIGHT, expand=True, fill=tk.X, padx=5)
                
                # Store the variable so we can retrieve its value later
                self.global_vars[".".join(map(str, key_path))] = var

    def save_config(self):
        """Saves the current configuration to the original file."""
        self.update_config_data_from_ui()
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2, ensure_ascii=False)
            messagebox.showinfo("Success", f"Configuration saved to:\n{self.config_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def save_config_as(self):
        """Saves the current configuration to a new file."""
        self.update_config_data_from_ui()
        new_path = filedialog.asksaveasfilename(
            initialdir=self.config_path.parent,
            initialfile="asr_config.v3.copy.json",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if new_path:
            try:
                with open(new_path, 'w', encoding='utf-8') as f:
                    json.dump(self.config_data, f, indent=2, ensure_ascii=False)
                messagebox.showinfo("Success", f"Configuration saved to:\n{new_path}")
                self.config_path = Path(new_path) # Update current path
                self.title(f"ASR Configurator v3 - {self.config_path.name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file:\n{e}")

    def update_config_data_from_ui(self):
        """Updates the main config_data dictionary from the UI's StringVars."""
        for key_path_str, var in self.global_vars.items():
            keys = key_path_str.split('.')
            current_level = self.config_data
            for key in keys[:-1]:
                current_level = current_level[key]
            
            # Try to convert back to original type (int, float, bool, list)
            try:
                # Use json.loads for robust parsing of bools, lists, etc.
                current_level[keys[-1]] = json.loads(var.get())
            except (json.JSONDecodeError, TypeError):
                # Fallback to string if it's not valid JSON (e.g., a simple string)
                current_level[keys[-1]] = var.get()
        
        # Re-initialize the tuner with the new data
        self.tuner = TranscriptionTunerV3(config_data=self.config_data)
        logger.debug("Configuration data updated from UI.")


    def reload_config(self):
        """Reloads the config from the disk, discarding UI changes."""
        if messagebox.askyesno("Confirm Reload", "Are you sure you want to reload the configuration from the disk? All unsaved changes will be lost."):
            self.config_data = self.load_initial_config()
            if self.config_data:
                self.tuner = TranscriptionTunerV3(config_data=self.config_data)
                # Clear and re-populate all frames
                for widget in self.winfo_children():
                    if isinstance(widget, ttk.PanedWindow):
                       for pane in widget.panes():
                           widget.forget(pane)
                           
                # Re-create the UI from scratch
                main_paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
                main_paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                run_config_frame = self.create_scrollable_frame(main_paned_window)
                main_paned_window.add(run_config_frame, weight=1)
                self.populate_run_config_frame(run_config_frame)
                self.resolved_settings_frame = self.create_scrollable_frame(main_paned_window)
                main_paned_window.add(self.resolved_settings_frame, weight=2)
                global_settings_frame = self.create_scrollable_frame(main_paned_window)
                main_paned_window.add(global_settings_frame, weight=2)
                self.populate_global_settings_frame(global_settings_frame)
                
                self.update_resolved_display()
                messagebox.showinfo("Success", "Configuration reloaded.")


if __name__ == "__main__":
    # The application expects 'asr_config.v3.json' to be in the same directory.
    try:
        # Determine the script's directory to find the config file
        # This works for both normal execution and when packaged with PyInstaller
        if getattr(sys, 'frozen', False):
            # The application is frozen
            application_path = Path(sys.executable).parent
        else:
            # The application is not frozen
            application_path = Path(__file__).parent
        
        config_file_path = application_path / "asr_config.v3.json"
        
        app = App(config_file_path)
        app.mainloop()

    except Exception as e:
        # A final catch-all for unexpected errors on startup.
        messagebox.showerror("Fatal Error", f"An unexpected error occurred:\n{e}")

