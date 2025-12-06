#!/usr/bin/env python3
"""
Real-time parameter tracing for pipeline diagnostics.

Emits JSON Lines format for streaming observability during execution.
Compatible with `tail -f`, log aggregators, and monitoring tools.
"""

import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager


class ParameterTracer:
    """
    Lightweight, thread-safe parameter tracer that emits JSON Lines.

    Usage:
        tracer = ParameterTracer("trace.jsonl")
        tracer.emit("stage_name", {"param1": "value1", ...})
        tracer.close()

    Or as context manager:
        with ParameterTracer("trace.jsonl") as tracer:
            tracer.emit("stage_name", {...})
    """

    def __init__(self, output_path: Union[str, Path],
                 buffer_size: int = 1,
                 include_timestamps: bool = True,
                 include_elapsed: bool = True):
        """
        Initialize the parameter tracer.

        Args:
            output_path: Path to JSON Lines output file
            buffer_size: Line buffer size (1 = immediate flush for real-time viewing)
            include_timestamps: Include ISO timestamp in each record
            include_elapsed: Include elapsed time since tracer start
        """
        self.output_path = Path(output_path)
        self.include_timestamps = include_timestamps
        self.include_elapsed = include_elapsed
        self.start_time = time.time()
        self._lock = threading.Lock()
        self._closed = False
        self._stage_timers: Dict[str, float] = {}

        # Ensure parent directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file with line buffering for real-time streaming
        self._file = open(self.output_path, 'w', encoding='utf-8', buffering=buffer_size)

        # Emit initialization record
        self._emit_internal("tracer_init", {
            "version": "1.0",
            "output_path": str(self.output_path),
            "start_time": datetime.now().isoformat()
        })

    def emit(self, stage: str, params: Dict[str, Any],
             level: str = "info",
             metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit a parameter snapshot for a pipeline stage.

        Args:
            stage: Name of the pipeline stage (e.g., "audio_extraction", "asr_config")
            params: Dictionary of parameters to record
            level: Log level ("debug", "info", "warn", "error")
            metrics: Optional performance metrics (duration, memory, etc.)
        """
        if self._closed:
            return

        record = {
            "stage": stage,
            "level": level,
            "params": self._sanitize_params(params)
        }

        if metrics:
            record["metrics"] = metrics

        self._emit_internal(stage, record)

    def emit_config(self, mode: str, sensitivity: str,
                    resolved_config: Dict[str, Any],
                    cli_args: Optional[Dict[str, Any]] = None) -> None:
        """
        Emit the initial configuration snapshot.

        Args:
            mode: Pipeline mode (balanced, fast, faster)
            sensitivity: Sensitivity preset (conservative, balanced, aggressive)
            resolved_config: The resolved configuration dictionary
            cli_args: Optional CLI arguments that were provided
        """
        self.emit("config_resolved", {
            "mode": mode,
            "sensitivity": sensitivity,
            "model": resolved_config.get("model", {}),
            "task": resolved_config.get("task", "transcribe"),
            "decoder_params": resolved_config.get("params", {}).get("decoder", {}),
            "vad_params": resolved_config.get("params", {}).get("vad", {}),
            "provider_params": resolved_config.get("params", {}).get("provider", {}),
            "cli_overrides": cli_args or {}
        })

    def emit_file_start(self, filename: str, file_number: int,
                        total_files: int, media_info: Dict[str, Any]) -> None:
        """Emit when starting to process a new file."""
        self.emit("file_start", {
            "filename": filename,
            "file_number": file_number,
            "total_files": total_files,
            "media_type": media_info.get("type", "unknown"),
            "duration_seconds": media_info.get("duration", 0),
            "path": str(media_info.get("path", ""))
        })

    def emit_audio_extraction(self, audio_path: str, duration: float,
                              sample_rate: int = 16000) -> None:
        """Emit audio extraction details."""
        self.emit("audio_extraction", {
            "output_path": str(audio_path),
            "duration_seconds": duration,
            "sample_rate": sample_rate
        })

    def emit_scene_detection(self, method: str, params: Dict[str, Any],
                             scenes_found: int,
                             scene_stats: Optional[Dict[str, Any]] = None) -> None:
        """Emit scene detection configuration and results."""
        self.emit("scene_detection", {
            "method": method,
            "params": params,
            "scenes_found": scenes_found,
            "stats": scene_stats or {}
        })

    def emit_asr_config(self, model: str, backend: str,
                        params: Dict[str, Any]) -> None:
        """Emit ASR configuration before transcription."""
        self.emit("asr_config", {
            "model": model,
            "backend": backend,
            "beam_size": params.get("beam_size"),
            "patience": params.get("patience"),
            "temperature": params.get("temperature"),
            "language": params.get("language"),
            "task": params.get("task"),
            "vad_filter": params.get("vad_filter", False),
            "word_timestamps": params.get("word_timestamps", True)
        })

    def emit_asr_progress(self, scene_number: int, total_scenes: int,
                          duration: float, segments_found: int) -> None:
        """Emit ASR progress for a scene."""
        self.emit("asr_progress", {
            "scene_number": scene_number,
            "total_scenes": total_scenes,
            "scene_duration_seconds": duration,
            "segments_found": segments_found
        }, level="debug")

    def emit_postprocessing(self, stats: Dict[str, Any]) -> None:
        """Emit post-processing statistics."""
        self.emit("postprocessing", {
            "total_subtitles": stats.get("total_subtitles", 0),
            "empty_removed": stats.get("empty_removed", 0),
            "hallucinations_removed": stats.get("removed_hallucinations", 0),
            "repetitions_removed": stats.get("removed_repetitions", 0),
            "duration_adjustments": stats.get("duration_adjustments", 0),
            "cps_filtered": stats.get("cps_filtered", 0)
        })

    def emit_completion(self, success: bool,
                        final_subtitles: int,
                        total_duration: float,
                        output_path: str,
                        error: Optional[str] = None) -> None:
        """Emit completion record."""
        record = {
            "success": success,
            "final_subtitles": final_subtitles,
            "total_duration_seconds": round(total_duration, 2),
            "output_path": str(output_path)
        }
        if error:
            record["error"] = error

        self.emit("completion", record, level="error" if not success else "info")

    @contextmanager
    def stage_timer(self, stage_name: str):
        """
        Context manager to time a stage and emit metrics.

        Usage:
            with tracer.stage_timer("transcription"):
                # do transcription work
        """
        start = time.time()
        self._stage_timers[stage_name] = start
        try:
            yield
        finally:
            elapsed = time.time() - start
            self.emit(f"{stage_name}_complete", {},
                     metrics={"duration_seconds": round(elapsed, 3)})

    def _emit_internal(self, stage: str, record: Dict[str, Any]) -> None:
        """Internal emit with threading safety."""
        if self._closed:
            return

        # Add metadata
        if self.include_timestamps:
            record["timestamp"] = datetime.now().isoformat()

        if self.include_elapsed:
            record["elapsed_seconds"] = round(time.time() - self.start_time, 3)

        # Thread-safe write
        with self._lock:
            try:
                self._file.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
                self._file.flush()  # Ensure immediate visibility
            except Exception:
                pass  # Silently ignore write errors to avoid disrupting pipeline

    def _sanitize_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize parameters for JSON serialization.
        Converts Path objects, handles None, etc.
        """
        sanitized = {}
        for key, value in params.items():
            if isinstance(value, Path):
                sanitized[key] = str(value)
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_params(value)
            elif isinstance(value, (list, tuple)):
                sanitized[key] = [
                    str(v) if isinstance(v, Path) else v
                    for v in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    def close(self) -> None:
        """Close the tracer and flush all pending writes."""
        if self._closed:
            return

        self._emit_internal("tracer_close", {
            "total_elapsed_seconds": round(time.time() - self.start_time, 3)
        })

        with self._lock:
            self._closed = True
            try:
                self._file.close()
            except Exception:
                pass

    def __enter__(self) -> "ParameterTracer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NullTracer:
    """
    No-op tracer for when tracing is disabled.
    All methods are no-ops to avoid conditional checks in pipeline code.
    """

    def emit(self, *args, **kwargs) -> None:
        pass

    def emit_config(self, *args, **kwargs) -> None:
        pass

    def emit_file_start(self, *args, **kwargs) -> None:
        pass

    def emit_audio_extraction(self, *args, **kwargs) -> None:
        pass

    def emit_scene_detection(self, *args, **kwargs) -> None:
        pass

    def emit_asr_config(self, *args, **kwargs) -> None:
        pass

    def emit_asr_progress(self, *args, **kwargs) -> None:
        pass

    def emit_postprocessing(self, *args, **kwargs) -> None:
        pass

    def emit_completion(self, *args, **kwargs) -> None:
        pass

    @contextmanager
    def stage_timer(self, stage_name: str):
        yield

    def close(self) -> None:
        pass

    def __enter__(self) -> "NullTracer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


def create_tracer(output_path: Optional[str] = None) -> Union[ParameterTracer, NullTracer]:
    """
    Factory function to create appropriate tracer.

    Args:
        output_path: Path for trace output, or None for NullTracer

    Returns:
        ParameterTracer if path provided, NullTracer otherwise
    """
    if output_path:
        return ParameterTracer(output_path)
    return NullTracer()
