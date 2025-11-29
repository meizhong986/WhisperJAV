"""
Ecosystem Configuration Schema for WhisperJAV v4.

Defines the structure for ecosystem YAML configuration files.
An ecosystem represents a family of related models (e.g., Transformers, Kaldi).

Example YAML:
    schemaVersion: v1
    kind: Ecosystem

    metadata:
      name: transformers
      displayName: "HuggingFace Transformers"
      description: "Models using HuggingFace Transformers pipeline"
      tags: [huggingface, chunked, batched]

    defaults:
      model.device: auto
      model.dtype: auto
      chunk.batch_size: 16

    provider:
      module: whisperjav.modules.transformers_asr
      class: TransformersASR
      install_requires:
        - transformers>=4.36.0
        - torch>=2.0.0

    compatible_tools:
      - auditok-scene-detection
      - silero-scene-detection
"""

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, field_validator

from .base import ConfigBase, ConfigKind


class ProviderInfo(Dict[str, Any]):
    """
    Provider implementation details.

    Specifies the Python module/class that implements this ecosystem's
    ASR functionality, plus any required dependencies.
    """

    pass  # Type alias


class EcosystemConfig(ConfigBase):
    """
    Complete ecosystem configuration schema.

    An ecosystem defines:
    - defaults: Shared default values for all models in this ecosystem
    - provider: Implementation details (module, class, dependencies)
    - compatible_tools: List of auxiliary tools that work with this ecosystem
    """

    kind: Literal[ConfigKind.ECOSYSTEM] = Field(
        default=ConfigKind.ECOSYSTEM,
        description="Must be 'Ecosystem' for ecosystem configs",
    )

    defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameter values for all models in this ecosystem",
    )

    provider: Dict[str, Any] = Field(
        default_factory=dict,
        description="Implementation provider details",
    )

    compatible_tools: List[str] = Field(
        default_factory=list,
        description="List of compatible auxiliary tool names",
    )

    # Ecosystem-level settings
    supports_gpu: bool = Field(
        default=True,
        description="Whether this ecosystem supports GPU acceleration",
    )
    supports_batching: bool = Field(
        default=False,
        description="Whether this ecosystem supports batch processing",
    )
    default_language: str = Field(
        default="ja",
        description="Default language for models in this ecosystem",
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate provider has required fields."""
        required = {"module", "class"}
        if v:  # Only validate if provider is specified
            missing = required - set(v.keys())
            if missing:
                raise ValueError(f"Provider missing required fields: {missing}")
        return v

    def get_model_base_config(self) -> Dict[str, Any]:
        """
        Get the base configuration that all models inherit.

        This includes ecosystem defaults and any computed values.
        """
        return dict(self.defaults)

    def get_provider_class(self) -> type:
        """
        Dynamically import and return the provider class.

        Returns:
            The ASR class for this ecosystem

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class not found in module
        """
        if not self.provider:
            raise ValueError(f"Ecosystem '{self.metadata.name}' has no provider defined")

        import importlib

        module_path = self.provider["module"]
        class_name = self.provider["class"]

        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def check_dependencies(self) -> Dict[str, bool]:
        """
        Check if required dependencies are installed.

        Returns:
            Dict mapping package name to installation status
        """
        install_requires = self.provider.get("install_requires", [])
        results = {}

        for req in install_requires:
            # Parse requirement (e.g., "transformers>=4.36.0")
            import re

            match = re.match(r"^([a-zA-Z0-9_-]+)", req)
            if match:
                package = match.group(1)
                try:
                    __import__(package)
                    results[package] = True
                except ImportError:
                    results[package] = False

        return results
