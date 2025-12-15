"""
WhisperJAV PyWebView API - Phase 2

Backend API for WhisperJAV web GUI.
Maintains thin wrapper pattern - delegates to CLI subprocess.
Completely decoupled from UI - can be tested standalone.
"""

import os
import sys
import time
import queue
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import webview
from webview import FileDialog


# Determine REPO_ROOT for module resolution
# Same pattern as Tkinter GUI
REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_documents_dir() -> Path:
    """
    Simple, platform-neutral Documents resolution.

    - Windows: %USERPROFILE%/Documents (via Path.home()/"Documents")
    - macOS/Linux: ~/Documents
    Fallback: user's home if Documents doesn't exist.
    """
    home = Path.home()
    docs = home / "Documents"
    return docs if docs.exists() else home


def _compute_default_output_dir() -> Path:
    """
    Compute default output per simple conventions:
    - Windows/macOS/Linux: Documents/WhisperJAV/output if Documents exists
    - Fallback: ~/WhisperJAV/output

    Always ensures the directory exists.
    """
    base = _get_documents_dir()
    if base.name.lower() != "documents" or not base.exists():
        base = Path.home()
    p = base / "WhisperJAV" / "output"
    p.mkdir(parents=True, exist_ok=True)
    return p


DEFAULT_OUTPUT = _compute_default_output_dir()


class WhisperJAVAPI:
    """
    API class exposed to JavaScript via PyWebView.

    All public methods are callable from JavaScript via:
        pywebview.api.method_name(args)

    Implements the same subprocess pattern as Tkinter GUI,
    with improved queue-based log streaming.
    """

    def __init__(self):
        """Initialize API state."""
        # Process state
        self.process: Optional[subprocess.Popen] = None
        self.status = "idle"  # idle, running, completed, cancelled, error
        self.exit_code: Optional[int] = None

        # Log streaming
        self.log_queue: queue.Queue = queue.Queue()
        self._stream_thread: Optional[threading.Thread] = None

        # Default output directory (ensure it exists and is normalized)
        self.default_output = str(_compute_default_output_dir())

    # ========================================================================
    # Process Management
    # ========================================================================

    def build_args(self, options: Dict[str, Any]) -> List[str]:
        """
        Build CLI arguments from options dictionary.

        Args:
            options: Dictionary of options from JavaScript:
                - inputs: List[str] - Input files/folders
                - mode: str - Processing mode (fidelity/fast/faster)
                - sensitivity: str - Sensitivity level (conservative/balanced/aggressive)
                - language: str - Output language (japanese/english-direct)
                - output_dir: str - Output directory path
                - temp_dir: str - Temporary directory path (optional)
                - keep_temp: bool - Keep temporary files
                - debug: bool - Enable debug logging
                - adaptive_classification: bool - Adaptive classification (WIP)
                - adaptive_audio_enhancement: bool - Adaptive audio enhancements (WIP)
                - smart_postprocessing: bool - Smart postprocessing (WIP)
                - async_processing: bool - Enable async processing
                - speech_segmenter: str - Speech segmentation backend (silero-v3.1, nemo, ten, none, or empty for default)
                - model_override: str - Model override (large-v3/large-v2/turbo)
                - credit: str - Opening credit text

        Returns:
            List of CLI arguments

        Raises:
            ValueError: If inputs are empty or invalid options
        """
        args = []

        # Check for special check_only mode
        if options.get('check_only', False):
            args += ["--check", "--check-verbose"]
            return args

        # Inputs (required)
        inputs = options.get('inputs', [])
        if not inputs:
            raise ValueError("Please add at least one file or folder.")
        args.extend(inputs)

        # Core options
        mode = options.get('mode', 'fidelity')
        args += ["--mode", mode]

        # Handle Transformers mode separately (uses --hf-* arguments)
        if mode == 'transformers':
            return self._build_transformers_args(args, options)

        # Source audio language (for transcription)
        source_language = options.get('source_language', 'japanese')
        args += ["--language", source_language]

        # Subtitle output format (native or direct-to-english)
        subs_language = options.get('subs_language', 'native')
        args += ["--subs-language", subs_language]

        sensitivity = options.get('sensitivity', 'balanced')
        args += ["--sensitivity", sensitivity]

        # Scene detection method (optional)
        scene_method = options.get('scene_detection_method', '').strip()
        if scene_method:
            args += ["--scene-detection-method", scene_method]

        output_dir = options.get('output_dir', self.default_output)
        args += ["--output-dir", output_dir]

        # Optional paths
        temp_dir = options.get('temp_dir', '').strip()
        if temp_dir:
            args += ["--temp-dir", temp_dir]

        if options.get('keep_temp', False):
            args += ["--keep-temp"]

        # Debug logging
        if options.get('debug', False):
            args += ["--debug"]

        # Advanced options (WIP features)
        if options.get('adaptive_classification', False):
            args += ["--adaptive-classification"]
        if options.get('adaptive_audio_enhancement', False):
            args += ["--adaptive-audio-enhancement"]
        if options.get('smart_postprocessing', False):
            args += ["--smart-postprocessing"]

        # Async processing
        if options.get('async_processing', False):
            args += ["--async-processing"]

        # Speech segmenter selection (replaces --no-vad)
        speech_segmenter = options.get('speech_segmenter', '').strip()
        if speech_segmenter:
            args += ["--speech-segmenter", speech_segmenter]

        # Model override
        model_override = options.get('model_override', '').strip()
        if model_override:
            args += ["--model", model_override]

        # Opening credit
        credit = options.get('credit', '').strip()
        if credit:
            args += ["--credit", credit]

        # Accept CPU mode (skip GPU warning)
        if options.get('accept_cpu_mode', False):
            args += ["--accept-cpu-mode"]

        return args

    def _build_transformers_args(self, args: List[str], options: Dict[str, Any]) -> List[str]:
        """
        Build CLI arguments for Transformers mode.

        Transformers mode uses dedicated --hf-* arguments and does not use
        sensitivity presets (those are for legacy pipelines only).

        Minimal Args by Default:
        - If hf_customized=False: Only pass essential args (language, device)
          and let the model use its tuned internal defaults.
        - If hf_customized=True: Pass all HF parameters explicitly.
        """
        # Source/subtitle language
        source_language = options.get('source_language', 'japanese')
        args += ["--language", source_language]

        subs_language = options.get('subs_language', 'native')
        args += ["--subs-language", subs_language]

        # Output directory
        output_dir = options.get('output_dir', self.default_output)
        args += ["--output-dir", output_dir]

        # Device is always passed (essential arg)
        hf_device = options.get('hf_device', 'auto')
        args += ["--hf-device", str(hf_device)]

        # Check if user explicitly customized parameters
        is_customized = options.get('hf_customized', False)

        if is_customized:
            # Full params mode: pass all HF parameters explicitly
            hf_optional_params = {
                'hf_model_id': '--hf-model-id',
                'hf_chunk_length': '--hf-chunk-length',
                'hf_stride': '--hf-stride',
                'hf_batch_size': '--hf-batch-size',
                'hf_scene': '--hf-scene',
                'hf_beam_size': '--hf-beam-size',
                'hf_temperature': '--hf-temperature',
                'hf_attn': '--hf-attn',
                'hf_timestamps': '--hf-timestamps',
                'hf_language': '--hf-language',
                'hf_dtype': '--hf-dtype',
            }

            for key, cli_arg in hf_optional_params.items():
                value = options.get(key)
                if value is not None:
                    args += [cli_arg, str(value)]
        # else: Minimal args mode - let model use its internal defaults

        # Common arguments
        temp_dir = options.get('temp_dir', '').strip()
        if temp_dir:
            args += ["--temp-dir", temp_dir]

        if options.get('keep_temp', False):
            args += ["--keep-temp"]

        if options.get('debug', False):
            args += ["--debug"]

        if options.get('accept_cpu_mode', False):
            args += ["--accept-cpu-mode"]

        return args

    def start_process(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start WhisperJAV subprocess with given options.

        Args:
            options: Dictionary of options (see build_args for format)

        Returns:
            dict: Response with success status and message
                {
                    "success": bool,
                    "message": str,
                    "command": str (optional, for debugging)
                }
        """
        # Check if already running
        if self.process is not None:
            return {
                "success": False,
                "message": "Process already running"
            }

        try:
            # Build arguments
            args = self.build_args(options)

            # Construct command
            cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args]

            # Force UTF-8 stdio in the child so logging can print ✓ and JP chars
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8:replace"

            # Log command for debugging
            def quote_arg(a: str) -> str:
                return f'"{a}"' if (" " in a or "\t" in a) else a

            cmd_display = "whisperjav.main " + " ".join(quote_arg(a) for a in args)
            self.log_queue.put(f"\n> {cmd_display}\n")

            # Start subprocess
            self.process = subprocess.Popen(
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

            # Update status
            self.status = "running"
            self.exit_code = None

            # Start log streaming thread
            self._stream_thread = threading.Thread(
                target=self._stream_output,
                daemon=True
            )
            self._stream_thread.start()

            return {
                "success": True,
                "message": "Process started successfully",
                "command": cmd_display
            }

        except ValueError as e:
            # Invalid options
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            # Unexpected error
            self.status = "error"
            return {
                "success": False,
                "message": f"Failed to start process: {e}"
            }

    def cancel_process(self) -> Dict[str, Any]:
        """
        Cancel the running subprocess.

        Returns:
            dict: Response with success status
        """
        if self.process is None:
            return {
                "success": False,
                "message": "No process running"
            }

        try:
            self.status = "cancelled"
            self.process.terminate()

            # Wait briefly for graceful shutdown
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                self.process.kill()
                self.process.wait()

            self.log_queue.put("\n[CANCELLED] Process terminated by user.\n")

            return {
                "success": True,
                "message": "Process cancelled"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to cancel process: {e}"
            }

    def get_process_status(self) -> Dict[str, Any]:
        """
        Get current process status.

        Returns:
            dict: Process status information
                {
                    "status": str,  # idle, running, completed, cancelled, error
                    "exit_code": int or None,
                    "has_logs": bool
                }
        """
        # Check if process has exited
        if self.process is not None and self.process.poll() is not None:
            self.exit_code = self.process.returncode
            self.process = None

            if self.status == "cancelled":
                pass  # Keep cancelled status
            elif self.exit_code == 0:
                self.status = "completed"
                self.log_queue.put("\n[SUCCESS] Process completed successfully.\n")
            else:
                self.status = "error"
                self.log_queue.put(f"\n[ERROR] Process exited with code {self.exit_code}.\n")

        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "has_logs": not self.log_queue.empty()
        }

    def _stream_output(self):
        """
        Background thread to read subprocess output and queue it.

        This runs in a separate thread to avoid blocking the main thread.
        """
        try:
            if self.process and self.process.stdout:
                for line in self.process.stdout:
                    self.log_queue.put(line)
        except Exception as e:
            self.log_queue.put(f"\n[ERROR] Log streaming error: {e}\n")
        finally:
            # Process has finished
            if self.process:
                self.process.wait()

    def get_logs(self) -> List[str]:
        """
        Fetch new log lines from the queue.

        JavaScript should poll this method periodically (e.g., every 100ms)
        to retrieve new log messages.

        Returns:
            List of log lines (may be empty if no new logs)
        """
        logs = []
        while not self.log_queue.empty():
            try:
                logs.append(self.log_queue.get_nowait())
            except queue.Empty:
                break
        return logs

    # ========================================================================
    # File Dialogs
    # ========================================================================

    def select_files(self) -> Dict[str, Any]:
        """
        Open native file dialog to select multiple video files.

        Returns:
            dict: Response with selected file paths
                {
                    "success": bool,
                    "paths": List[str] or None,
                    "message": str (if error)
                }
        """
        windows = webview.windows
        if not windows:
            return {
                "success": False,
                "message": "No active window"
            }

        window = windows[0]

        # File filters must be a list/tuple of filter strings. A single string will be
        # iterated character-by-character by pywebview and cause parse errors (e.g., 'M').
        # Use semicolon-separated patterns without spaces for compatibility.
        file_types = [
            'Media Files (*.mp4;*.mkv;*.avi;*.flv;*.wmv;*.webm;*.mpg;*.mpeg;*.ts;*.mp3;*.wav;*.aac;*.m4a;*.flac;*.ogg;*.opus)',
            'All Files (*.*)'
        ]

        result = window.create_file_dialog(
            FileDialog.OPEN,
            allow_multiple=True,
            file_types=file_types
        )

        if result and len(result) > 0:
            return {
                "success": True,
                "paths": result
            }
        else:
            return {
                "success": False,
                "message": "No files selected"
            }

    def select_folder(self) -> Dict[str, Any]:
        """
        Open native folder dialog to select a folder.

        Returns:
            dict: Response with selected folder path
                {
                    "success": bool,
                    "path": str or None,
                    "message": str (if error)
                }
        """
        windows = webview.windows
        if not windows:
            return {
                "success": False,
                "message": "No active window"
            }

        window = windows[0]

        result = window.create_file_dialog(
            FileDialog.FOLDER
        )

        if result and len(result) > 0:
            return {
                "success": True,
                "path": result[0]
            }
        else:
            return {
                "success": False,
                "message": "No folder selected"
            }

    def select_output_directory(self) -> Dict[str, Any]:
        """
        Open native folder dialog to select output directory.

        Returns:
            dict: Response with selected directory path
                {
                    "success": bool,
                    "path": str or None,
                    "message": str (if error)
                }
        """
        return self.select_folder()

    def open_output_folder(self, path: str) -> Dict[str, Any]:
        """
        Open output folder in file explorer.

        Args:
            path: Directory path to open

        Returns:
            dict: Response with success status
        """
        try:
            folder = Path(path)
            folder.mkdir(parents=True, exist_ok=True)

            if sys.platform.startswith("win"):
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.run(["open", str(folder)])
            else:
                subprocess.run(["xdg-open", str(folder)])

            return {
                "success": True,
                "message": "Folder opened"
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Cannot open folder: {e}"
            }

    # ========================================================================
    # Configuration & Defaults
    # ========================================================================

    def get_default_output_dir(self) -> str:
        """
        Get the default output directory path.

        Returns:
            Default output directory path
        """
        # Recompute to honor any environment changes during runtime
        self.default_output = str(_compute_default_output_dir())
        return self.default_output

    # ========================================================================
    # Component Introspection (v3.0 - Dynamic GUI support)
    # ========================================================================

    def get_available_components(self) -> Dict[str, Any]:
        """
        Get all available components for dynamic GUI population.

        Returns:
            dict with 'asr', 'vad', 'features' lists
        """
        try:
            from whisperjav.config.components import get_all_components
            return {
                "success": True,
                "components": get_all_components()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_component_schema(self, component_type: str, name: str) -> Dict[str, Any]:
        """
        Get parameter schema for a specific component.

        Args:
            component_type: 'asr', 'vad', or 'features'
            name: Component name

        Returns:
            dict with parameter schema
        """
        try:
            from whisperjav.config.components import get_component
            component = get_component(component_type, name)
            return {
                "success": True,
                "schema": component.get_schema(),
                "metadata": component.to_dict()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_legacy_pipelines(self) -> Dict[str, Any]:
        """
        Get available legacy pipeline names and info.

        Returns:
            dict with pipeline information
        """
        try:
            from whisperjav.config.legacy import LEGACY_PIPELINES, list_legacy_pipelines
            return {
                "success": True,
                "pipelines": list_legacy_pipelines(),
                "info": LEGACY_PIPELINES
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_component_defaults(self, component_type: str, name: str) -> Dict[str, Any]:
        """
        Get default parameter values for a component.

        Args:
            component_type: 'asr', 'vad', or 'features'
            name: Component name

        Returns:
            dict with default parameter values
        """
        try:
            from whisperjav.config.components import get_component
            component = get_component(component_type, name)

            # Get defaults from the balanced preset or Options defaults
            if hasattr(component, 'presets') and 'balanced' in component.presets:
                defaults = component.presets['balanced'].model_dump()
            else:
                # Fall back to Options class defaults
                defaults = component.Options().model_dump()

            return {
                "success": True,
                "defaults": defaults
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_speech_segmenter_backends(self) -> Dict[str, Any]:
        """
        Get available speech segmenter backends with availability status.

        Returns:
            dict with backends list, each containing:
                - name: backend identifier
                - display_name: human-readable name
                - available: boolean
                - install_hint: installation instructions if not available
        """
        try:
            from whisperjav.modules.speech_segmentation import SpeechSegmenterFactory
            backends = SpeechSegmenterFactory.get_available_backends()
            return {
                "success": True,
                "backends": backends
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_speech_enhancer_backends(self) -> Dict[str, Any]:
        """
        Get available speech enhancer backends with availability status.

        Returns:
            dict with backends list, each containing:
                - name: backend identifier
                - display_name: human-readable name
                - available: boolean
                - install_hint: installation instructions if not available
        """
        try:
            from whisperjav.modules.speech_enhancement import SpeechEnhancerFactory
            backends = SpeechEnhancerFactory.get_available_backends()
            return {
                "success": True,
                "backends": backends
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def validate_ensemble_config(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ensemble configuration before processing.

        Args:
            options: Ensemble configuration options

        Returns:
            dict with validation result and any errors
        """
        errors = []
        warnings = []

        try:
            from whisperjav.config.components import get_component, get_all_components

            # Check required fields
            if not options.get('inputs'):
                errors.append({
                    "field": "inputs",
                    "message": "At least one input file is required"
                })

            # Validate ASR selection
            asr_name = options.get('asr', '')
            if not asr_name:
                errors.append({
                    "field": "asr",
                    "message": "Please select an ASR engine"
                })
            else:
                try:
                    get_component('asr', asr_name)
                except Exception:
                    errors.append({
                        "field": "asr",
                        "message": f"Unknown ASR component: {asr_name}"
                    })

            # Validate VAD selection (can be 'none')
            vad_name = options.get('vad', 'none')
            if vad_name and vad_name != 'none':
                try:
                    get_component('vad', vad_name)
                except Exception:
                    errors.append({
                        "field": "vad",
                        "message": f"Unknown VAD component: {vad_name}"
                    })

            # Validate features
            features = options.get('features', [])
            for feature_name in features:
                try:
                    get_component('features', feature_name)
                except Exception:
                    errors.append({
                        "field": f"features.{feature_name}",
                        "message": f"Unknown feature: {feature_name}"
                    })

            # Validate overrides against component schemas
            overrides = options.get('overrides', {})
            for key, value in overrides.items():
                # Parse key like "asr.beam_size" or "vad.threshold"
                parts = key.split('.', 1)
                if len(parts) != 2:
                    errors.append({
                        "field": f"overrides.{key}",
                        "message": f"Invalid override key format: {key}"
                    })
                    continue

                comp_type, param_name = parts

                # Map component type
                if comp_type == 'asr':
                    comp_name = asr_name
                elif comp_type == 'vad':
                    comp_name = vad_name
                elif comp_type == 'features':
                    # For features, extract feature name from param
                    # e.g., "features.scene_detection.max_duration_s"
                    feature_parts = param_name.split('.', 1)
                    if len(feature_parts) == 2:
                        comp_name = feature_parts[0]
                        param_name = feature_parts[1]
                        comp_type = 'features'
                    else:
                        continue
                else:
                    continue

                # Validate parameter value against schema
                if comp_name and comp_name != 'none':
                    try:
                        component = get_component(comp_type, comp_name)
                        schema = component.get_schema()

                        # Find parameter in schema
                        param_schema = None
                        for p in schema:
                            if p['name'] == param_name:
                                param_schema = p
                                break

                        if param_schema and 'constraints' in param_schema:
                            constraints = param_schema['constraints']
                            if 'ge' in constraints and value < constraints['ge']:
                                errors.append({
                                    "field": f"overrides.{key}",
                                    "message": f"Value {value} is below minimum {constraints['ge']}"
                                })
                            if 'le' in constraints and value > constraints['le']:
                                errors.append({
                                    "field": f"overrides.{key}",
                                    "message": f"Value {value} exceeds maximum {constraints['le']}"
                                })
                    except Exception:
                        pass  # Skip validation if component not found

            if errors:
                return {
                    "valid": False,
                    "errors": errors,
                    "warnings": warnings
                }

            return {
                "valid": True,
                "errors": [],
                "warnings": warnings
            }

        except Exception as e:
            return {
                "valid": False,
                "errors": [{"field": "general", "message": str(e)}],
                "warnings": []
            }

    def start_ensemble_process(self, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start WhisperJAV subprocess with ensemble configuration.

        Args:
            options: Ensemble configuration options:
                - inputs: List[str] - Input files/folders
                - asr: str - ASR component name
                - vad: str - VAD component name (or 'none')
                - features: List[str] - Feature names
                - overrides: Dict[str, Any] - Parameter overrides
                - output_dir: str - Output directory
                - task: str - Task type (transcribe/translate)
                - language: str - Source language
                - temp_dir: str - Temp directory (optional)
                - keep_temp: bool - Keep temp files
                - verbosity: str - Verbosity level

        Returns:
            dict: Response with success status and message
        """
        # Check if already running
        if self.process is not None:
            return {
                "success": False,
                "message": "Process already running"
            }

        # Validate configuration first
        validation = self.validate_ensemble_config(options)
        if not validation.get('valid', False):
            error_msgs = [e['message'] for e in validation.get('errors', [])]
            return {
                "success": False,
                "message": "Invalid configuration: " + "; ".join(error_msgs)
            }

        try:
            # Build ensemble-specific arguments
            args = self._build_ensemble_args(options)

            # Construct command
            cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args]

            # Force UTF-8 stdio
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8:replace"

            # Log command
            def quote_arg(a: str) -> str:
                return f'"{a}"' if (" " in a or "\t" in a) else a

            cmd_display = "whisperjav.main " + " ".join(quote_arg(a) for a in args)
            self.log_queue.put(f"\n> {cmd_display}\n")

            # Start subprocess
            self.process = subprocess.Popen(
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

            # Update status
            self.status = "running"
            self.exit_code = None

            # Start log streaming thread
            self._stream_thread = threading.Thread(
                target=self._stream_output,
                daemon=True
            )
            self._stream_thread.start()

            return {
                "success": True,
                "message": "Ensemble process started successfully",
                "command": cmd_display
            }

        except ValueError as e:
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            self.status = "error"
            return {
                "success": False,
                "message": f"Failed to start ensemble process: {e}"
            }

    def _build_ensemble_args(self, options: Dict[str, Any]) -> List[str]:
        """
        Build CLI arguments for ensemble processing.

        Args:
            options: Ensemble configuration options

        Returns:
            List of CLI arguments
        """
        args = []

        # Inputs (required)
        inputs = options.get('inputs', [])
        if not inputs:
            raise ValueError("Please add at least one file or folder.")
        args.extend(inputs)

        # Ensemble mode uses direct component specification
        # Use --asr, --vad flags instead of --mode
        asr = options.get('asr', 'faster_whisper')
        args += ["--asr", asr]

        vad = options.get('vad', 'none')
        args += ["--vad", vad]

        # Features
        features = options.get('features', [])
        if features:
            args += ["--features", ",".join(features)]

        # Language settings
        language = options.get('language', 'ja')
        args += ["--language", language]

        task = options.get('task', 'transcribe')
        args += ["--task", task]

        # Output directory
        output_dir = options.get('output_dir', self.default_output)
        args += ["--output-dir", output_dir]

        # Parameter overrides - pass as JSON
        overrides = options.get('overrides', {})
        if overrides:
            import json
            args += ["--overrides", json.dumps(overrides)]

        # Optional paths
        temp_dir = options.get('temp_dir', '').strip()
        if temp_dir:
            args += ["--temp-dir", temp_dir]

        if options.get('keep_temp', False):
            args += ["--keep-temp"]

        # Verbosity
        verbosity = options.get('verbosity', 'summary')
        if verbosity:
            args += ["--verbosity", verbosity]

        return args

    # ========================================================================
    # Transformers Pipeline Methods
    # ========================================================================

    def _load_hf_models_registry(self) -> Dict[str, Any]:
        """
        Load HuggingFace models registry from YAML config.

        Returns:
            Dict with 'models' list and 'default_model' string.
            Falls back to hardcoded defaults if file not found or invalid.
        """
        import yaml

        # Default fallback if registry loading fails
        fallback = {
            "default_model": "kotoba-tech/kotoba-whisper-v2.2",
            "models": [
                {"id": "kotoba-tech/kotoba-whisper-v2.2", "label": "Kotoba v2.2 (Latest)", "category": "kotoba"},
                {"id": "kotoba-tech/kotoba-whisper-v2.0", "label": "Kotoba v2.0", "category": "kotoba"},
                {"id": "openai/whisper-large-v3-turbo", "label": "Whisper Large v3 Turbo", "category": "openai"},
            ]
        }

        # Locate registry file relative to this module
        registry_path = (
            Path(__file__).parent.parent /
            "config" / "v4" / "ecosystems" / "transformers" / "hf_models_registry.yaml"
        )

        if not registry_path.exists():
            # Try alternate path (if running from different location)
            alt_path = REPO_ROOT / "whisperjav" / "config" / "v4" / "ecosystems" / "transformers" / "hf_models_registry.yaml"
            if alt_path.exists():
                registry_path = alt_path
            else:
                return fallback

        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = yaml.safe_load(f)

            if not registry or "models" not in registry:
                return fallback

            return {
                "default_model": registry.get("default_model", fallback["default_model"]),
                "models": registry.get("models", fallback["models"])
            }

        except Exception:
            # YAML parse error or other issue - fall back silently
            return fallback

    def get_transformers_schema(self) -> Dict[str, Any]:
        """
        Get parameter schema for Transformers pipeline customize modal.

        Returns the schema used by the frontend to generate the customize
        modal UI for Transformers/HuggingFace pipeline parameters.

        Model options are loaded from the v4 config registry at:
        whisperjav/config/v4/ecosystems/transformers/hf_models_registry.yaml
        """
        # Load models from v4 config registry
        registry = self._load_hf_models_registry()

        # Convert registry format to dropdown options format
        # Registry: {"id": "...", "label": "..."} -> Options: {"value": "...", "label": "..."}
        model_options = [
            {"value": m["id"], "label": m["label"]}
            for m in registry["models"]
        ]
        default_model = registry["default_model"]

        return {
            "success": True,
            "schema": {
                "model": {
                    "model_id": {
                        "type": "dropdown",
                        "label": "Model",
                        "options": model_options,
                        "default": default_model
                    },
                    "device": {
                        "type": "dropdown",
                        "label": "Device",
                        "options": [
                            {"value": "auto", "label": "Auto (detect GPU)"},
                            {"value": "cuda", "label": "CUDA (GPU)"},
                            {"value": "cpu", "label": "CPU"},
                        ],
                        "default": "auto"
                    },
                    "dtype": {
                        "type": "dropdown",
                        "label": "Data Type",
                        "options": [
                            {"value": "auto", "label": "Auto"},
                            {"value": "float16", "label": "Float16 (faster)"},
                            {"value": "bfloat16", "label": "BFloat16"},
                            {"value": "float32", "label": "Float32 (slower)"},
                        ],
                        "default": "auto"
                    },
                },
                "chunking": {
                    "chunk_length_s": {
                        "type": "slider",
                        "label": "Chunk Length (s)",
                        "min": 5,
                        "max": 30,
                        "step": 1,
                        "default": 15
                    },
                    "stride_length_s": {
                        "type": "slider",
                        "label": "Stride Length (s)",
                        "min": 0,
                        "max": 10,
                        "step": 0.5,
                        "default": 0  # 0 means auto-calculate as chunk/6
                    },
                    "batch_size": {
                        "type": "slider",
                        "label": "Batch Size",
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "default": 16
                    },
                },
                "quality": {
                    "beam_size": {
                        "type": "slider",
                        "label": "Beam Size",
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "default": 5
                    },
                    "temperature": {
                        "type": "slider",
                        "label": "Temperature",
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.1,
                        "default": 0.0
                    },
                    "attn_implementation": {
                        "type": "dropdown",
                        "label": "Attention",
                        "options": [
                            {"value": "sdpa", "label": "SDPA (fastest)"},
                            {"value": "flash_attention_2", "label": "Flash Attention 2"},
                            {"value": "eager", "label": "Eager (slowest)"},
                        ],
                        "default": "sdpa"
                    },
                    "timestamps": {
                        "type": "dropdown",
                        "label": "Timestamps",
                        "options": [
                            {"value": "segment", "label": "Segment-level"},
                            {"value": "word", "label": "Word-level"},
                        ],
                        "default": "segment"
                    },
                },
                "scene": {
                    "scene": {
                        "type": "dropdown",
                        "label": "Scene Detection",
                        "options": [
                            {"value": "none", "label": "None (process whole file)"},
                            {"value": "auditok", "label": "Auditok (energy-based)"},
                            {"value": "silero", "label": "Silero (neural VAD)"},
                        ],
                        "default": "none"
                    },
                },
            }
        }

    # ========================================================================
    # Two-Pass Ensemble Methods
    # ========================================================================

    def get_pipeline_defaults(self, pipeline: str, sensitivity: str) -> Dict[str, Any]:
        """
        Get resolved parameters for a pipeline+sensitivity combination.

        Args:
            pipeline: Pipeline name ('balanced', 'fast', 'faster', 'fidelity', 'kotoba-faster-whisper', 'transformers')
            sensitivity: Sensitivity level ('conservative', 'balanced', 'aggressive')

        Returns:
            dict with resolved parameters that can be customized
        """
        # Handle Transformers separately (doesn't use legacy config resolution)
        if pipeline == 'transformers':
            return {
                "success": True,
                "pipeline": "transformers",
                "sensitivity": None,  # Sensitivity not applicable for Transformers
                "is_transformers": True,
                "params": {
                    "model_id": "kotoba-tech/kotoba-whisper-v2.2",
                    "chunk_length_s": 15,
                    "stride_length_s": None,
                    "batch_size": 8,
                    "scene": "none",
                    "beam_size": 5,
                    "temperature": 0.0,
                    "attn_implementation": "sdpa",
                    "timestamps": "segment",
                    "language": "ja",
                    "device": "auto",
                    "dtype": "auto",
                },
                "model": "kotoba-tech/kotoba-whisper-v2.2",
            }

        try:
            from whisperjav.config.legacy import resolve_legacy_pipeline

            config = resolve_legacy_pipeline(
                pipeline_name=pipeline,
                sensitivity=sensitivity,
                task='transcribe'
            )

            # Determine scene detection method (default: auditok)
            scene_detection_method = 'auditok'
            if 'features' in config and config['features'].get('scene_detection'):
                # Could be enhanced to read from config if specified
                pass

            # M11: Detect V3 config by structure, not pipeline name string
            # V3 configs (like kotoba) have 'params.asr', legacy have 'params.decoder'
            is_v3_config = 'asr' in config.get('params', {})
            is_kotoba = is_v3_config  # Currently only kotoba uses V3

            if is_v3_config:
                # Kotoba uses V3 structure with params.asr
                asr_params = config.get('params', {}).get('asr', {})
                return {
                    "success": True,
                    "pipeline": pipeline,
                    "sensitivity": sensitivity,
                    "model": config.get('model', {}).get('model_name', 'kotoba-tech/kotoba-whisper-v2.0-faster'),
                    "is_kotoba": True,
                    "params": {
                        "decoder": {
                            "task": asr_params.get('task', 'transcribe'),
                            "language": asr_params.get('language', 'ja'),
                            "beam_size": asr_params.get('beam_size', 3),
                            "best_of": asr_params.get('best_of', 3),
                            "patience": asr_params.get('patience', 2.0),
                            "suppress_blank": asr_params.get('suppress_blank', True),
                            "without_timestamps": asr_params.get('without_timestamps', False),
                        },
                        "provider": {
                            "temperature": asr_params.get('temperature', [0.0, 0.3]),
                            "compression_ratio_threshold": asr_params.get('compression_ratio_threshold', 2.4),
                            "logprob_threshold": asr_params.get('logprob_threshold', -1.5),
                            "no_speech_threshold": asr_params.get('no_speech_threshold', 0.34),
                            "condition_on_previous_text": asr_params.get('condition_on_previous_text', True),
                            "word_timestamps": asr_params.get('word_timestamps', False),
                            # Internal VAD parameters (kotoba-specific)
                            "vad_filter": asr_params.get('vad_filter', True),
                            "vad_threshold": asr_params.get('vad_threshold', 0.01),
                            "min_speech_duration_ms": asr_params.get('min_speech_duration_ms', 90),
                            "max_speech_duration_s": asr_params.get('max_speech_duration_s', 28.0),
                            "min_silence_duration_ms": asr_params.get('min_silence_duration_ms', 150),
                            "speech_pad_ms": asr_params.get('speech_pad_ms', 400),
                            "repetition_penalty": asr_params.get('repetition_penalty', 1.0),
                            "no_repeat_ngram_size": asr_params.get('no_repeat_ngram_size', 0),
                        },
                        "vad": {}  # Empty - kotoba uses internal VAD, not external
                    },
                    "scene_detection_method": scene_detection_method
                }

            # Return the params section for customization (standard pipelines)
            return {
                "success": True,
                "pipeline": pipeline,
                "sensitivity": sensitivity,
                "is_kotoba": False,
                "params": {
                    "decoder": config['params']['decoder'],
                    "provider": config['params']['provider'],
                    "vad": config['params']['vad']
                },
                "model": config['model'],
                "scene_detection_method": scene_detection_method
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def get_merge_strategies(self) -> Dict[str, Any]:
        """
        Get available merge strategies for two-pass ensemble.

        Returns:
            dict with list of merge strategies and descriptions
        """
        try:
            from whisperjav.ensemble.merge import get_available_strategies
            return {
                "success": True,
                "strategies": get_available_strategies()
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def start_ensemble_twopass(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start two-pass ensemble processing.

        Args:
            config: Two-pass ensemble configuration:
                - inputs: List[str] - Input files/folders
                - pass1: dict - Pass 1 configuration
                    - pipeline: str - Pipeline name
                    - sensitivity: str - Sensitivity level
                    - overrides: dict - Parameter overrides
                - pass2: dict - Pass 2 configuration (optional)
                    - enabled: bool - Whether to run pass 2
                    - pipeline: str - Pipeline name
                    - sensitivity: str - Sensitivity level
                    - overrides: dict - Parameter overrides
                - merge_strategy: str - Merge strategy name
                - output_dir: str - Output directory
                - subs_language: str - Subtitle language mode
                - keep_temp: bool - Keep temp files
                - verbosity: str - Verbosity level

        Returns:
            dict: Response with success status and message
        """
        # Check if already running
        if self.process is not None:
            return {
                "success": False,
                "message": "Process already running"
            }

        try:
            # Build CLI arguments for two-pass ensemble
            args = self._build_twopass_args(config)

            # Construct command
            cmd = [sys.executable, "-X", "utf8", "-m", "whisperjav.main", *args]

            # Force UTF-8 stdio
            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"
            env["PYTHONIOENCODING"] = "utf-8:replace"

            # Log command
            def quote_arg(a: str) -> str:
                return f'"{a}"' if (" " in a or "\t" in a) else a

            cmd_display = "whisperjav.main " + " ".join(quote_arg(a) for a in args)
            self.log_queue.put(f"\n> {cmd_display}\n")

            # Start subprocess
            self.process = subprocess.Popen(
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

            # Update status
            self.status = "running"
            self.exit_code = None

            # Start log streaming thread
            self._stream_thread = threading.Thread(
                target=self._stream_output,
                daemon=True
            )
            self._stream_thread.start()

            return {
                "success": True,
                "message": "Two-pass ensemble started successfully",
                "command": cmd_display
            }

        except ValueError as e:
            return {
                "success": False,
                "message": str(e)
            }
        except Exception as e:
            self.status = "error"
            return {
                "success": False,
                "message": f"Failed to start two-pass ensemble: {e}"
            }

    def _build_twopass_args(self, config: Dict[str, Any]) -> List[str]:
        """
        Build CLI arguments for two-pass ensemble processing.

        Supports Full Configuration Snapshot approach:
        - If pass is customized: send full params as JSON
        - If not customized: send pipeline+sensitivity for backend resolution

        Transformers Pass Handling:
        - isTransformers=True: Skip sensitivity, pass HF params if customized
        - isTransformers=False: Standard legacy handling with sensitivity

        Args:
            config: Two-pass ensemble configuration

        Returns:
            List of CLI arguments
        """
        import json
        args = []

        # Inputs (required)
        inputs = config.get('inputs', [])
        if not inputs:
            raise ValueError("Please add at least one file or folder.")
        args.extend(inputs)

        # Enable ensemble mode
        args.append("--ensemble")

        # Pass 1 configuration
        pass1 = config.get('pass1', {})
        args += ["--pass1-pipeline", pass1.get('pipeline', 'balanced')]

        if pass1.get('isTransformers'):
            # Transformers pass: no sensitivity, handle HF params
            if pass1.get('customized') and pass1.get('params'):
                args += ["--pass1-hf-params", json.dumps(pass1['params'])]
            # else: minimal args - backend uses model defaults
        else:
            # Legacy pass: use sensitivity
            args += ["--pass1-sensitivity", pass1.get('sensitivity', 'balanced')]
            if pass1.get('customized') and pass1.get('params'):
                args += ["--pass1-params", json.dumps(pass1['params'])]

        # Pass 1: Scene Detector
        scene1 = pass1.get('sceneDetector')
        if scene1 and scene1 != 'none':
            args += ["--pass1-scene-detector", scene1]

        # Pass 1: Speech Segmenter (legacy pipelines only)
        # Note: Transformers uses HF internal chunking; segmentation support planned for v1.8.0
        if not pass1.get('isTransformers'):
            segmenter1 = pass1.get('speechSegmenter')
            if segmenter1:  # Pass any value including "none" to disable VAD
                args += ["--pass1-speech-segmenter", segmenter1]

        # Pass 1: Speech Enhancer
        enhancer1 = pass1.get('speechEnhancer')
        if enhancer1 and enhancer1 != 'none':
            # Handle FFmpeg DSP with selected effects
            if enhancer1 == 'ffmpeg-dsp':
                dsp_effects = pass1.get('dspEffects', ['loudnorm'])
                effects_str = ','.join(dsp_effects) if dsp_effects else 'loudnorm'
                args += ["--pass1-speech-enhancer", f"ffmpeg-dsp:{effects_str}"]
            else:
                args += ["--pass1-speech-enhancer", enhancer1]

        # Pass 1: Model
        model1 = pass1.get('model')
        if model1:
            args += ["--pass1-model", model1]

        # Pass 2 configuration
        pass2 = config.get('pass2', {})
        if pass2.get('enabled', False):
            args += ["--pass2-pipeline", pass2.get('pipeline', 'fidelity')]

            if pass2.get('isTransformers'):
                # Transformers pass: no sensitivity, handle HF params
                if pass2.get('customized') and pass2.get('params'):
                    args += ["--pass2-hf-params", json.dumps(pass2['params'])]
                # else: minimal args - backend uses model defaults
            else:
                # Legacy pass: use sensitivity
                args += ["--pass2-sensitivity", pass2.get('sensitivity', 'balanced')]
                if pass2.get('customized') and pass2.get('params'):
                    args += ["--pass2-params", json.dumps(pass2['params'])]

            # Pass 2: Scene Detector
            scene2 = pass2.get('sceneDetector')
            if scene2 and scene2 != 'none':
                args += ["--pass2-scene-detector", scene2]

            # Pass 2: Speech Segmenter (legacy pipelines only)
            # Note: Transformers uses HF internal chunking; segmentation support planned for v1.8.0
            if not pass2.get('isTransformers'):
                segmenter2 = pass2.get('speechSegmenter')
                if segmenter2:  # Pass any value including "none" to disable VAD
                    args += ["--pass2-speech-segmenter", segmenter2]

            # Pass 2: Speech Enhancer
            enhancer2 = pass2.get('speechEnhancer')
            if enhancer2 and enhancer2 != 'none':
                # Handle FFmpeg DSP with selected effects
                if enhancer2 == 'ffmpeg-dsp':
                    dsp_effects = pass2.get('dspEffects', ['loudnorm'])
                    effects_str = ','.join(dsp_effects) if dsp_effects else 'loudnorm'
                    args += ["--pass2-speech-enhancer", f"ffmpeg-dsp:{effects_str}"]
                else:
                    args += ["--pass2-speech-enhancer", enhancer2]

            # Pass 2: Model
            model2 = pass2.get('model')
            if model2:
                args += ["--pass2-model", model2]

        # Merge strategy
        merge_strategy = config.get('merge_strategy', 'smart_merge')
        args += ["--merge-strategy", merge_strategy]

        # Output directory
        output_dir = config.get('output_dir', self.default_output)
        args += ["--output-dir", output_dir]

        # Subtitle language mode
        subs_language = config.get('subs_language', 'native')
        args += ["--subs-language", subs_language]

        # Source language
        source_language = config.get('source_language', 'japanese')
        args += ["--language", source_language]

        # Optional paths
        temp_dir = config.get('temp_dir', '').strip()
        if temp_dir:
            args += ["--temp-dir", temp_dir]

        if config.get('keep_temp', False):
            args += ["--keep-temp"]

        # Debug logging
        if config.get('debug', False):
            args += ["--debug"]

        return args

    # ========================================================================
    # Test Methods (Phase 1 - Keep for backward compatibility)
    # ========================================================================

    def hello_world(self) -> Dict[str, Any]:
        """
        Test method to verify Python <-> JavaScript bridge.

        Returns:
            dict: Response with success status and message
        """
        return {
            "success": True,
            "message": "Hello from Python! PyWebView bridge is working.",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_version": "Phase 2 - Backend Refactor"
        }
