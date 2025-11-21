"""
Pipeline configuration schemas for WhisperJAV.

Defines workflow and resolved configuration structures.
"""

from typing import Any, Dict, Literal

from pydantic import Field

from .base import Backend, BaseConfig


class WorkflowConfig(BaseConfig):
    """
    Pipeline workflow definition.

    Specifies which components are used in a pipeline.
    """

    model: str = Field(
        description="Model identifier from MODELS registry."
    )
    vad: str = Field(
        description="VAD engine identifier or 'none'."
    )
    backend: Backend = Field(
        description="ASR backend to use."
    )
    features: Dict[str, str] = Field(
        default_factory=dict,
        description="Feature configurations (e.g., scene_detection, post_processing)."
    )


class ResolvedParams(BaseConfig):
    """
    Resolved parameters structure.

    Contains all parameter dictionaries for a pipeline execution.
    """

    decoder: Dict[str, Any] = Field(
        description="Decoder parameters (task, language, beam_size, etc.)."
    )
    provider: Dict[str, Any] = Field(
        description="Provider/engine parameters (temperature, etc.)."
    )
    vad: Dict[str, Any] = Field(
        default_factory=dict,
        description="VAD parameters (may be empty if packed into provider)."
    )


class ResolvedConfig(BaseConfig):
    """
    Complete resolved configuration.

    This is the output of resolve_config() and matches the current
    TranscriptionTuner.resolve_params() structure exactly for
    backward compatibility.
    """

    pipeline_name: str = Field(
        description="Name of the pipeline (faster, fast, fidelity, balanced)."
    )
    sensitivity_name: str = Field(
        description="Sensitivity level used (conservative, balanced, aggressive)."
    )
    workflow: WorkflowConfig = Field(
        description="Workflow component configuration."
    )
    model: Dict[str, Any] = Field(
        description="Model configuration (provider, model_name, compute_type, etc.)."
    )
    params: ResolvedParams = Field(
        description="All resolved parameters."
    )
    features: Dict[str, Any] = Field(
        description="Feature configurations (scene_detection, post_processing)."
    )
    task: Literal["transcribe", "translate"] = Field(
        description="Task being performed."
    )
    language: str = Field(
        default="ja",
        description="Target language code."
    )
