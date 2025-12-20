"""
Crash Tracer - Debug instrumentation for tracking ctranslate2 crashes.

This utility captures detailed state at critical points to help identify
the cause of STATUS_INTEGER_DIVIDE_BY_ZERO crashes in faster-whisper.

Usage:
    from whisperjav.utils.crash_tracer import CrashTracer

    tracer = CrashTracer.get_instance()
    tracer.enable()

    # In code:
    tracer.checkpoint("before_transcribe", audio_shape=audio.shape, params=params)
    result = model.transcribe(audio, **params)
    tracer.checkpoint("after_transcribe", segments=len(result))

The tracer writes to a rotating log file and also dumps state on crash signals.
"""

import os
import sys
import json
import time
import signal
import atexit
import threading
import traceback
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class CrashTracer:
    """
    Singleton tracer that captures state at checkpoints for crash analysis.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> 'CrashTracer':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self.enabled = False
        self.checkpoints = []
        self.max_checkpoints = 100
        self.start_time = time.time()
        self.trace_file = None
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # State tracking
        self.current_file = None
        self.current_scene = None
        self.transcribe_count = 0
        self.last_params = None
        self.last_audio_info = None

        # GPU state tracking
        self.gpu_available = False
        try:
            import torch
            self.gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

    def enable(self, trace_dir: Optional[Path] = None):
        """Enable crash tracing."""
        self.enabled = True

        # Set up trace file
        if trace_dir is None:
            trace_dir = Path.cwd() / "crash_traces"
        trace_dir.mkdir(exist_ok=True)

        self.trace_file = trace_dir / f"crash_trace_{self.session_id}.jsonl"

        # Register signal handlers for crash detection
        self._register_signal_handlers()

        # Register atexit handler
        atexit.register(self._on_exit)

        self._write_checkpoint({
            "event": "tracer_enabled",
            "session_id": self.session_id,
            "gpu_available": self.gpu_available,
            "python_version": sys.version,
            "platform": sys.platform,
        })

        logger.info(f"CrashTracer enabled, writing to: {self.trace_file}")

    def disable(self):
        """Disable crash tracing."""
        self.enabled = False

    def _register_signal_handlers(self):
        """Register handlers for crash-related signals."""
        if sys.platform == 'win32':
            # Windows doesn't have SIGSEGV etc. in the same way
            # But we can catch some signals
            try:
                signal.signal(signal.SIGABRT, self._signal_handler)
                signal.signal(signal.SIGFPE, self._signal_handler)
                signal.signal(signal.SIGILL, self._signal_handler)
            except (ValueError, OSError):
                pass
        else:
            # Unix-like
            for sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGFPE, signal.SIGBUS):
                try:
                    signal.signal(sig, self._signal_handler)
                except (ValueError, OSError):
                    pass

    def _signal_handler(self, signum, frame):
        """Handle crash signals."""
        self._write_checkpoint({
            "event": "CRASH_SIGNAL",
            "signal": signum,
            "signal_name": signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum),
            "stack_trace": traceback.format_stack(frame),
        })
        self._dump_final_state()
        # Re-raise to let the default handler run
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    def _on_exit(self):
        """Called on normal exit."""
        if self.enabled:
            self._write_checkpoint({
                "event": "normal_exit",
                "total_transcriptions": self.transcribe_count,
                "runtime_seconds": time.time() - self.start_time,
            })

    def checkpoint(self, name: str, **data):
        """Record a checkpoint with associated data."""
        if not self.enabled:
            return

        checkpoint = {
            "event": "checkpoint",
            "name": name,
            "timestamp": time.time() - self.start_time,
            "transcribe_count": self.transcribe_count,
            "current_file": str(self.current_file) if self.current_file else None,
            "current_scene": self.current_scene,
            **data,
        }

        # Add GPU memory info if available
        if self.gpu_available:
            try:
                import torch
                checkpoint["gpu_memory_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                checkpoint["gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            except Exception:
                pass

        self._write_checkpoint(checkpoint)

        # Keep in memory for crash dump
        self.checkpoints.append(checkpoint)
        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints.pop(0)

    def _write_checkpoint(self, data: Dict):
        """Write checkpoint to trace file."""
        if self.trace_file:
            try:
                # Ensure data is JSON serializable
                safe_data = self._make_serializable(data)
                with open(self.trace_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(safe_data, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())  # Force write to disk
            except Exception as e:
                logger.warning(f"Failed to write checkpoint: {e}")

    def _make_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable form."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        elif hasattr(obj, 'shape'):  # numpy array
            return f"array(shape={obj.shape}, dtype={obj.dtype})"
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            try:
                json.dumps(obj)
                return obj
            except (TypeError, ValueError):
                return str(obj)

    def _dump_final_state(self):
        """Dump final state for crash analysis."""
        if self.trace_file:
            crash_dump = self.trace_file.with_suffix('.crash_dump.json')
            try:
                with open(crash_dump, 'w', encoding='utf-8') as f:
                    json.dump({
                        "session_id": self.session_id,
                        "crash_time": datetime.now().isoformat(),
                        "total_transcriptions": self.transcribe_count,
                        "last_params": self._make_serializable(self.last_params),
                        "last_audio_info": self._make_serializable(self.last_audio_info),
                        "recent_checkpoints": [self._make_serializable(c) for c in self.checkpoints[-20:]],
                    }, f, indent=2)
                logger.error(f"Crash dump written to: {crash_dump}")
            except Exception as e:
                logger.error(f"Failed to write crash dump: {e}")

    # Convenience methods for common checkpoints

    def trace_transcribe_start(self, audio_info: Dict, params: Dict):
        """Trace the start of a transcription."""
        self.transcribe_count += 1
        self.last_params = params.copy()
        self.last_audio_info = audio_info.copy()

        self.checkpoint(
            "transcribe_start",
            audio_info=audio_info,
            params=params,
        )

    def trace_transcribe_generator_created(self, info):
        """Trace when the transcription generator is created."""
        self.checkpoint(
            "generator_created",
            language=getattr(info, 'language', None),
            language_probability=getattr(info, 'language_probability', None),
        )

    def trace_segment(self, index: int, start: float, end: float, text_len: int):
        """Trace a segment being yielded."""
        self.checkpoint(
            "segment_yielded",
            segment_index=index,
            start_sec=start,
            end_sec=end,
            text_length=text_len,
        )

    def trace_transcribe_complete(self, num_segments: int):
        """Trace transcription completion."""
        self.checkpoint(
            "transcribe_complete",
            num_segments=num_segments,
        )

    def trace_error(self, error: Exception, context: str = ""):
        """Trace an error."""
        self.checkpoint(
            "error",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context,
            stack_trace=traceback.format_exc(),
        )

    def set_current_file(self, file_path: Optional[Path]):
        """Set the current file being processed."""
        self.current_file = file_path

    def set_current_scene(self, scene_index: Optional[int]):
        """Set the current scene being processed."""
        self.current_scene = scene_index


# Global convenience function
def get_tracer() -> CrashTracer:
    """Get the global crash tracer instance."""
    return CrashTracer.get_instance()
