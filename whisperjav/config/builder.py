"""
PipelineBuilder for WhisperJAV Configuration System v2.0.

Fluent API for building custom pipeline configurations.
"""

from copy import deepcopy
from typing import Any, Dict, Optional

from .errors import ConfigValidationError
from .resolver import resolve_config
from .schemas import Sensitivity


class PipelineBuilder:
    """
    Fluent builder for pipeline configurations.

    Example:
        >>> config = (
        ...     PipelineBuilder("balanced")
        ...     .with_sensitivity("aggressive")
        ...     .with_task("translate")
        ...     .with_decoder_param("beam_size", 10)
        ...     .build()
        ... )
    """

    def __init__(self, pipeline_name: str):
        """
        Initialize builder with a base pipeline.

        Args:
            pipeline_name: Base pipeline ('faster', 'fast', 'fidelity', 'balanced')
        """
        self._pipeline_name = pipeline_name
        self._sensitivity = "balanced"
        self._task = "transcribe"
        self._decoder_overrides: Dict[str, Any] = {}
        self._provider_overrides: Dict[str, Any] = {}
        self._vad_overrides: Dict[str, Any] = {}
        self._feature_overrides: Dict[str, Any] = {}
        self._kwargs: Dict[str, Any] = {}

    def with_sensitivity(self, sensitivity: str) -> "PipelineBuilder":
        """Set sensitivity level."""
        # Validate
        try:
            Sensitivity(sensitivity)
        except ValueError:
            valid = [s.value for s in Sensitivity]
            raise ConfigValidationError(
                f"Invalid sensitivity. Must be one of: {valid}",
                field="sensitivity",
                value=sensitivity
            )
        self._sensitivity = sensitivity
        return self

    def with_task(self, task: str) -> "PipelineBuilder":
        """Set task (transcribe or translate)."""
        if task not in ("transcribe", "translate"):
            raise ConfigValidationError(
                "Task must be 'transcribe' or 'translate'",
                field="task",
                value=task
            )
        self._task = task
        return self

    def with_decoder_param(self, name: str, value: Any) -> "PipelineBuilder":
        """Override a decoder parameter."""
        self._decoder_overrides[name] = value
        return self

    def with_provider_param(self, name: str, value: Any) -> "PipelineBuilder":
        """Override a provider parameter."""
        self._provider_overrides[name] = value
        return self

    def with_vad_param(self, name: str, value: Any) -> "PipelineBuilder":
        """Override a VAD parameter."""
        self._vad_overrides[name] = value
        return self

    def with_beam_size(self, size: int) -> "PipelineBuilder":
        """Set decoder beam size."""
        return self.with_decoder_param("beam_size", size)

    def with_patience(self, patience: float) -> "PipelineBuilder":
        """Set decoder patience."""
        return self.with_decoder_param("patience", patience)

    def with_temperature(self, temperature: float) -> "PipelineBuilder":
        """Set provider temperature."""
        return self.with_provider_param("temperature", temperature)

    def with_no_speech_threshold(self, threshold: float) -> "PipelineBuilder":
        """Set no speech threshold."""
        return self.with_provider_param("no_speech_threshold", threshold)

    def with_vad_threshold(self, threshold: float) -> "PipelineBuilder":
        """Set VAD threshold."""
        return self.with_vad_param("threshold", threshold)

    def with_scene_detection_method(self, method: str) -> "PipelineBuilder":
        """Set scene detection method."""
        self._kwargs["scene_detection_method"] = method
        return self

    def with_feature(self, feature_name: str, config: Dict[str, Any]) -> "PipelineBuilder":
        """Override feature configuration."""
        self._feature_overrides[feature_name] = config
        return self

    def build(self) -> Dict[str, Any]:
        """
        Build and return the final configuration.

        Returns:
            Resolved configuration dictionary.

        Raises:
            ConfigValidationError: If configuration is invalid.
        """
        # Get base config from resolver
        config = resolve_config(
            self._pipeline_name,
            self._sensitivity,
            self._task,
            **self._kwargs
        )

        # Apply overrides
        if self._decoder_overrides:
            config['params']['decoder'].update(self._decoder_overrides)

        if self._provider_overrides:
            config['params']['provider'].update(self._provider_overrides)

        if self._vad_overrides:
            config['params']['vad'].update(self._vad_overrides)

        if self._feature_overrides:
            for feature_name, feature_config in self._feature_overrides.items():
                if feature_name in config['features']:
                    config['features'][feature_name].update(feature_config)
                else:
                    config['features'][feature_name] = feature_config

        return config

    def copy(self) -> "PipelineBuilder":
        """Create a copy of this builder."""
        new_builder = PipelineBuilder(self._pipeline_name)
        new_builder._sensitivity = self._sensitivity
        new_builder._task = self._task
        new_builder._decoder_overrides = deepcopy(self._decoder_overrides)
        new_builder._provider_overrides = deepcopy(self._provider_overrides)
        new_builder._vad_overrides = deepcopy(self._vad_overrides)
        new_builder._feature_overrides = deepcopy(self._feature_overrides)
        new_builder._kwargs = deepcopy(self._kwargs)
        return new_builder


def quick_config(
    pipeline: str = "balanced",
    sensitivity: str = "balanced",
    task: str = "transcribe",
    **overrides
) -> Dict[str, Any]:
    """
    Quick configuration helper for simple use cases.

    Args:
        pipeline: Pipeline name
        sensitivity: Sensitivity level
        task: Task type
        **overrides: Parameter overrides (beam_size, patience, temperature, etc.)

    Returns:
        Resolved configuration dictionary.

    Example:
        >>> config = quick_config("balanced", "aggressive", beam_size=10)
    """
    builder = PipelineBuilder(pipeline).with_sensitivity(sensitivity).with_task(task)

    # Map common overrides
    decoder_params = {'beam_size', 'patience', 'best_of', 'language'}
    provider_params = {'temperature', 'no_speech_threshold', 'compression_ratio_threshold'}
    vad_params = {'threshold', 'min_speech_duration_ms', 'speech_pad_ms'}

    for key, value in overrides.items():
        if key in decoder_params:
            builder.with_decoder_param(key, value)
        elif key in provider_params:
            builder.with_provider_param(key, value)
        elif key in vad_params:
            builder.with_vad_param(key, value)
        elif key == 'scene_detection_method':
            builder.with_scene_detection_method(value)

    return builder.build()
