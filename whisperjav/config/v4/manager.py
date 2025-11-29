"""
Configuration Manager for WhisperJAV v4.

The central entry point for the v4 configuration system.
Provides a unified API for accessing models, tools, ecosystems, and presets.

This manager:
- Coordinates all registries
- Provides high-level config resolution
- Handles ecosystem + model + preset composition
- Supports GUI introspection

Usage:
    from whisperjav.config.v4 import ConfigManager

    manager = ConfigManager()

    # Get resolved config for a model
    config = manager.get_model_config(
        "kotoba-whisper-v2",
        sensitivity="balanced",
        overrides={"decode.beam_size": 7}
    )

    # Get GUI schema
    gui = manager.get_gui_schema("kotoba-whisper-v2")

    # List available models
    models = manager.list_models()
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import (
    EcosystemNotFoundError,
    ModelNotFoundError,
    ToolNotFoundError,
    V4ConfigError,
)
from .loaders import ConfigMerger, deep_merge
from .registries import (
    EcosystemRegistry,
    ModelRegistry,
    PresetRegistry,
    ToolRegistry,
    get_ecosystem_registry,
    get_model_registry,
    get_preset_registry,
    get_tool_registry,
)
from .schemas.ecosystem import EcosystemConfig
from .schemas.model import ModelConfig
from .schemas.preset import PresetConfig
from .schemas.tool import ToolConfig


class ConfigManager:
    """
    Central configuration manager for WhisperJAV v4.

    Provides unified access to all configuration types with:
    - Config resolution with inheritance and overrides
    - GUI schema generation
    - Ecosystem/model/tool discovery
    - Preset application
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the ConfigManager.

        Args:
            base_path: Base path for config files. If None, uses default.
        """
        self._base_path = base_path or self._get_default_base_path()

        # Initialize registries
        self._models = get_model_registry(self._base_path)
        self._tools = get_tool_registry(self._base_path)
        self._ecosystems = get_ecosystem_registry(self._base_path)
        self._presets = get_preset_registry(self._base_path)

        self._merger = ConfigMerger()

    def _get_default_base_path(self) -> Path:
        """Get default base path for config files."""
        return Path(__file__).parent / "ecosystems"

    # =========================================================================
    # Model Operations
    # =========================================================================

    def get_model(self, name: str) -> ModelConfig:
        """
        Get a model configuration by name.

        Args:
            name: Model name

        Returns:
            ModelConfig instance

        Raises:
            ModelNotFoundError: If model not found
        """
        return self._models.get(name)

    def get_model_config(
        self,
        model_name: str,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved model configuration.

        Resolution order:
        1. Ecosystem defaults (if model has ecosystem)
        2. Model spec (base values)
        3. Model preset[sensitivity] (sensitivity overrides)
        4. Global preset[sensitivity] (if exists)
        5. User overrides

        Args:
            model_name: Model name
            sensitivity: Sensitivity preset name
            overrides: Additional user overrides

        Returns:
            Fully resolved configuration dict
        """
        model = self.get_model(model_name)

        # Start with ecosystem defaults if available
        base_config: Dict[str, Any] = {}
        if model.metadata.ecosystem:
            try:
                ecosystem = self._ecosystems.get(model.metadata.ecosystem)
                base_config = ecosystem.get_model_base_config()
            except EcosystemNotFoundError:
                pass  # No ecosystem defaults

        # Apply model spec
        config = self._merger.merge(base_config, dict(model.spec))

        # Apply model's sensitivity preset
        model_preset = model.presets.get(sensitivity, {})
        config = self._merger.merge(config, model_preset)

        # Apply global preset if exists
        global_preset = self._presets.get_or_none(sensitivity)
        if global_preset:
            config = self._merger.merge(config, global_preset.get_overrides())

        # Apply user overrides
        if overrides:
            config = self._merger.merge(config, overrides)

        return config

    def list_models(
        self,
        ecosystem: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[str]:
        """
        List available model names.

        Args:
            ecosystem: Filter by ecosystem (optional)
            include_deprecated: Include deprecated models

        Returns:
            List of model names
        """
        if ecosystem:
            models = self._models.find_by_ecosystem(ecosystem)
            if not include_deprecated:
                models = [m for m in models if not m.is_deprecated()]
            return [m.get_name() for m in models]
        else:
            return self._models.list_names(include_deprecated)

    def list_models_by_ecosystem(self) -> Dict[str, List[str]]:
        """
        Get models organized by ecosystem.

        Returns:
            Dict mapping ecosystem name to list of model names
        """
        return self._models.list_by_ecosystem()

    # =========================================================================
    # Tool Operations
    # =========================================================================

    def get_tool(self, name: str) -> ToolConfig:
        """
        Get a tool configuration by name.

        Args:
            name: Tool name

        Returns:
            ToolConfig instance

        Raises:
            ToolNotFoundError: If tool not found
        """
        return self._tools.get(name)

    def get_tool_config(
        self,
        tool_name: str,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved tool configuration.

        Args:
            tool_name: Tool name
            sensitivity: Sensitivity preset name
            overrides: Additional user overrides

        Returns:
            Fully resolved configuration dict
        """
        tool = self.get_tool(tool_name)

        # Start with tool spec
        config = dict(tool.spec)

        # Apply tool's sensitivity preset
        tool_preset = tool.presets.get(sensitivity, {})
        config = self._merger.merge(config, tool_preset)

        # Apply global preset if exists
        global_preset = self._presets.get_or_none(sensitivity)
        if global_preset:
            config = self._merger.merge(config, global_preset.get_overrides())

        # Apply user overrides
        if overrides:
            config = self._merger.merge(config, overrides)

        return config

    def list_tools(
        self,
        tool_type: Optional[str] = None,
        include_deprecated: bool = False,
    ) -> List[str]:
        """
        List available tool names.

        Args:
            tool_type: Filter by tool type (optional)
            include_deprecated: Include deprecated tools

        Returns:
            List of tool names
        """
        if tool_type:
            tools = self._tools.find_by_type(tool_type)
            if not include_deprecated:
                tools = [t for t in tools if not t.is_deprecated()]
            return [t.get_name() for t in tools]
        else:
            return self._tools.list_names(include_deprecated)

    def list_tools_by_type(self) -> Dict[str, List[str]]:
        """
        Get tools organized by type.

        Returns:
            Dict mapping tool type to list of tool names
        """
        return self._tools.list_by_type()

    # =========================================================================
    # Ecosystem Operations
    # =========================================================================

    def get_ecosystem(self, name: str) -> EcosystemConfig:
        """
        Get an ecosystem configuration by name.

        Args:
            name: Ecosystem name

        Returns:
            EcosystemConfig instance

        Raises:
            EcosystemNotFoundError: If ecosystem not found
        """
        return self._ecosystems.get(name)

    def list_ecosystems(self, include_deprecated: bool = False) -> List[str]:
        """
        List available ecosystem names.

        Returns:
            List of ecosystem names
        """
        return self._ecosystems.list_names(include_deprecated)

    def check_ecosystem_dependencies(self, ecosystem_name: str) -> Dict[str, bool]:
        """
        Check if ecosystem dependencies are installed.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            Dict mapping package name to installation status
        """
        return self._ecosystems.check_dependencies(ecosystem_name)

    # =========================================================================
    # Preset Operations
    # =========================================================================

    def get_preset(self, name: str) -> PresetConfig:
        """
        Get a preset configuration by name.

        Args:
            name: Preset name

        Returns:
            PresetConfig instance
        """
        return self._presets.get(name)

    def list_presets(self) -> List[str]:
        """
        List available preset names.

        Returns:
            List of preset names
        """
        return self._presets.list_names()

    # =========================================================================
    # GUI Operations
    # =========================================================================

    def get_gui_schema(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete GUI schema for a model.

        Returns a structure suitable for frontend rendering including:
        - Parameter definitions with types and constraints
        - GUI widget hints
        - Parameter grouping
        - Available presets

        Args:
            model_name: Model name

        Returns:
            GUI schema dict
        """
        model = self.get_model(model_name)
        return model.get_gui_schema()

    def get_tool_gui_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get complete GUI schema for a tool.

        Args:
            tool_name: Tool name

        Returns:
            GUI schema dict
        """
        tool = self.get_tool(tool_name)
        return tool.get_gui_schema()

    def get_all_gui_schemas(self) -> Dict[str, Dict[str, Any]]:
        """
        Get GUI schemas for all models.

        Returns:
            Dict mapping model name to GUI schema
        """
        schemas = {}
        for model_name in self.list_models():
            try:
                schemas[model_name] = self.get_gui_schema(model_name)
            except V4ConfigError:
                continue  # Skip problematic models
        return schemas

    # =========================================================================
    # Utility Operations
    # =========================================================================

    def refresh(self) -> None:
        """Force reload all configs from disk."""
        self._models.refresh()
        self._tools.refresh()
        self._ecosystems.refresh()
        self._presets.refresh()

    def validate_config(
        self, model_name: str, config: Dict[str, Any]
    ) -> List[str]:
        """
        Validate a config dict against model schema.

        Args:
            model_name: Model name
            config: Config dict to validate

        Returns:
            List of validation errors (empty if valid)
        """
        # Basic validation - check all keys are valid
        model = self.get_model(model_name)
        valid_keys = set(model.spec.keys())
        errors = []

        for key in config.keys():
            if key not in valid_keys:
                errors.append(f"Unknown parameter: {key}")

        return errors

    def get_compatible_tools(self, model_name: str) -> List[str]:
        """
        Get tools compatible with a model.

        Args:
            model_name: Model name

        Returns:
            List of compatible tool names
        """
        model = self.get_model(model_name)

        # First check model's explicit compatible_tools
        if model.compatible_tools:
            return model.compatible_tools

        # Fall back to ecosystem's compatible_tools
        if model.metadata.ecosystem:
            try:
                return self._ecosystems.get_compatible_tools(model.metadata.ecosystem)
            except EcosystemNotFoundError:
                pass

        # Default: all tools are compatible
        return self.list_tools()
