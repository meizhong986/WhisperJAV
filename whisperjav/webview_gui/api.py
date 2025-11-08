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
                - mode: str - Processing mode (balanced/fast/faster)
                - sensitivity: str - Sensitivity level (conservative/balanced/aggressive)
                - language: str - Output language (japanese/english-direct)
                - output_dir: str - Output directory path
                - temp_dir: str - Temporary directory path (optional)
                - keep_temp: bool - Keep temporary files
                - verbosity: str - Verbosity level (quiet/summary/normal/verbose)
                - adaptive_classification: bool - Adaptive classification (WIP)
                - adaptive_audio_enhancement: bool - Adaptive audio enhancements (WIP)
                - smart_postprocessing: bool - Smart postprocessing (WIP)
                - async_processing: bool - Enable async processing
                - max_workers: int - Max workers for async processing
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
        mode = options.get('mode', 'balanced')
        args += ["--mode", mode]

        # Source audio language (for transcription)
        source_language = options.get('source_language', 'japanese')
        args += ["--language", source_language]

        # Subtitle output format (native or direct-to-english)
        subs_language = options.get('subs_language', 'native')
        args += ["--subs-language", subs_language]

        sensitivity = options.get('sensitivity', 'balanced')
        args += ["--sensitivity", sensitivity]

        output_dir = options.get('output_dir', self.default_output)
        args += ["--output-dir", output_dir]

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

        # Advanced options (WIP features)
        if options.get('adaptive_classification', False):
            args += ["--adaptive-classification"]
        if options.get('adaptive_audio_enhancement', False):
            args += ["--adaptive-audio-enhancement"]
        if options.get('smart_postprocessing', False):
            args += ["--smart-postprocessing"]

        # Async processing
        if options.get('async_processing', False):
            max_workers = options.get('max_workers', 1)
            args += ["--async-processing", "--max-workers", str(max_workers)]

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
