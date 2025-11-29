"""
GUI Introspection API for WhisperJAV v4 Configuration.

Provides a clean, frontend-friendly API for:
- Discovering available models, tools, and presets
- Getting parameter schemas with GUI hints
- Building configuration panels dynamically
- Validating user input

This API is designed to be consumed by the PyWebView GUI frontend.
All responses are JSON-serializable dictionaries.

Usage:
    from whisperjav.config.v4.gui_api import GUIAPI

    api = GUIAPI()

    # Get list of ecosystems with their models
    ecosystems = api.get_ecosystems_summary()

    # Get detailed schema for a model
    schema = api.get_model_schema("kotoba-whisper-v2")

    # Validate and apply user config
    result = api.apply_config("kotoba-whisper-v2", "balanced", user_overrides)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ModelNotFoundError, ToolNotFoundError, V4ConfigError
from .manager import ConfigManager


class GUIAPI:
    """
    Frontend-friendly API for GUI integration.

    All methods return JSON-serializable dictionaries suitable for
    passing to JavaScript via PyWebView's js.api interface.
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the GUI API.

        Args:
            base_path: Base path for config files (optional)
        """
        self._manager = ConfigManager(base_path)

    # =========================================================================
    # Discovery Methods
    # =========================================================================

    def get_ecosystems_summary(self) -> Dict[str, Any]:
        """
        Get summary of all ecosystems and their models.

        Returns:
            {
                "ecosystems": [
                    {
                        "name": "transformers",
                        "displayName": "HuggingFace Transformers",
                        "description": "...",
                        "models": ["kotoba-whisper-v2", "whisper-large-v3"],
                        "modelCount": 2
                    },
                    ...
                ],
                "totalModels": 5,
                "totalEcosystems": 2
            }
        """
        ecosystems = []
        total_models = 0

        for eco_name in self._manager.list_ecosystems():
            try:
                ecosystem = self._manager.get_ecosystem(eco_name)
                models = self._manager.list_models(ecosystem=eco_name)

                ecosystems.append({
                    "name": ecosystem.get_name(),
                    "displayName": ecosystem.get_display_name(),
                    "description": ecosystem.metadata.description,
                    "models": models,
                    "modelCount": len(models),
                    "supportsGpu": ecosystem.supports_gpu,
                    "supportsBatching": ecosystem.supports_batching,
                })
                total_models += len(models)
            except V4ConfigError:
                continue

        return {
            "ecosystems": ecosystems,
            "totalModels": total_models,
            "totalEcosystems": len(ecosystems),
        }

    def get_models_list(
        self,
        ecosystem: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get list of all models with basic info.

        Args:
            ecosystem: Filter by ecosystem (optional)

        Returns:
            {
                "models": [
                    {
                        "name": "kotoba-whisper-v2",
                        "displayName": "Kotoba Whisper v2.0",
                        "ecosystem": "transformers",
                        "tags": ["japanese", "fast"],
                        "deprecated": false
                    },
                    ...
                ]
            }
        """
        models = []
        model_names = self._manager.list_models(ecosystem=ecosystem)

        for name in model_names:
            try:
                model = self._manager.get_model(name)
                models.append({
                    "name": model.get_name(),
                    "displayName": model.get_display_name(),
                    "ecosystem": model.metadata.ecosystem,
                    "description": model.metadata.description,
                    "tags": model.metadata.tags,
                    "deprecated": model.is_deprecated(),
                })
            except V4ConfigError:
                continue

        return {"models": models}

    def get_tools_list(
        self,
        tool_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get list of all tools with basic info.

        Args:
            tool_type: Filter by tool type (optional)

        Returns:
            {
                "tools": [
                    {
                        "name": "auditok-scene-detection",
                        "displayName": "Auditok Scene Detection",
                        "toolType": "scene_detection",
                        "tags": ["energy-based"],
                    },
                    ...
                ],
                "byType": {
                    "scene_detection": ["auditok-scene-detection", ...],
                    ...
                }
            }
        """
        tools = []
        tool_names = self._manager.list_tools(tool_type=tool_type)

        for name in tool_names:
            try:
                tool = self._manager.get_tool(name)
                tools.append({
                    "name": tool.get_name(),
                    "displayName": tool.get_display_name(),
                    "toolType": tool.tool_type,
                    "description": tool.metadata.description,
                    "tags": tool.metadata.tags,
                })
            except V4ConfigError:
                continue

        by_type = self._manager.list_tools_by_type()

        return {
            "tools": tools,
            "byType": by_type,
        }

    def get_presets_list(self) -> Dict[str, Any]:
        """
        Get list of all sensitivity presets.

        Returns:
            {
                "presets": [
                    {
                        "name": "balanced",
                        "displayName": "Balanced",
                        "description": "...",
                        "recommendedFor": [...],
                        "notRecommendedFor": [...]
                    },
                    ...
                ]
            }
        """
        presets = []

        for name in self._manager.list_presets():
            try:
                preset = self._manager.get_preset(name)
                presets.append({
                    "name": preset.get_name(),
                    "displayName": preset.get_display_name(),
                    "description": preset.metadata.description,
                    "descriptionLong": preset.description_long,
                    "level": preset.level.value,
                    "recommendedFor": preset.recommended_for,
                    "notRecommendedFor": preset.not_recommended_for,
                })
            except V4ConfigError:
                continue

        return {"presets": presets}

    # =========================================================================
    # Schema Methods
    # =========================================================================

    def get_model_schema(self, model_name: str) -> Dict[str, Any]:
        """
        Get complete GUI schema for a model.

        Returns schema with:
        - Parameter definitions with types and constraints
        - GUI widget hints for each parameter
        - Parameter grouping
        - Available presets

        Args:
            model_name: Model name

        Returns:
            {
                "modelName": "kotoba-whisper-v2",
                "displayName": "Kotoba Whisper v2.0",
                "ecosystem": "transformers",
                "parameters": [
                    {
                        "key": "model.device",
                        "value": "auto",
                        "type": "str",
                        "gui": {
                            "widget": "dropdown",
                            "options": ["auto", "cuda", "cpu"],
                            "group": "model"
                        }
                    },
                    ...
                ],
                "groups": {
                    "model": ["model.id", "model.device", ...],
                    "decode": [...],
                    ...
                },
                "presets": ["conservative", "balanced", "aggressive"]
            }
        """
        try:
            return self._manager.get_gui_schema(model_name)
        except ModelNotFoundError as e:
            return {
                "error": True,
                "errorType": "ModelNotFoundError",
                "message": str(e),
                "suggestion": e.suggestion,
            }

    def get_tool_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get complete GUI schema for a tool.

        Args:
            tool_name: Tool name

        Returns:
            Tool GUI schema dict
        """
        try:
            return self._manager.get_tool_gui_schema(tool_name)
        except ToolNotFoundError as e:
            return {
                "error": True,
                "errorType": "ToolNotFoundError",
                "message": str(e),
                "suggestion": e.suggestion,
            }

    # =========================================================================
    # Configuration Methods
    # =========================================================================

    def get_resolved_config(
        self,
        model_name: str,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved configuration for a model.

        Args:
            model_name: Model name
            sensitivity: Sensitivity preset
            overrides: User overrides

        Returns:
            {
                "success": true,
                "config": {...resolved config...},
                "appliedPreset": "balanced",
                "appliedOverrides": {...}
            }
        """
        try:
            config = self._manager.get_model_config(
                model_name, sensitivity, overrides
            )
            return {
                "success": True,
                "config": config,
                "appliedPreset": sensitivity,
                "appliedOverrides": overrides or {},
            }
        except V4ConfigError as e:
            return {
                "success": False,
                "error": str(e),
                "errorType": e.__class__.__name__,
            }

    def validate_overrides(
        self,
        model_name: str,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Validate user overrides against model schema.

        Args:
            model_name: Model name
            overrides: User overrides to validate

        Returns:
            {
                "valid": true/false,
                "errors": [...list of error messages...],
                "warnings": [...list of warnings...]
            }
        """
        try:
            errors = self._manager.validate_config(model_name, overrides)
            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "warnings": [],
            }
        except V4ConfigError as e:
            return {
                "valid": False,
                "errors": [str(e)],
                "warnings": [],
            }

    # =========================================================================
    # Compatibility Methods
    # =========================================================================

    def get_compatible_tools(self, model_name: str) -> Dict[str, Any]:
        """
        Get tools compatible with a model.

        Args:
            model_name: Model name

        Returns:
            {
                "modelName": "kotoba-whisper-v2",
                "compatibleTools": ["auditok-scene-detection", ...],
                "toolsByType": {
                    "scene_detection": [...],
                    ...
                }
            }
        """
        try:
            compatible = self._manager.get_compatible_tools(model_name)
            by_type: Dict[str, List[str]] = {}

            for tool_name in compatible:
                try:
                    tool = self._manager.get_tool(tool_name)
                    tool_type = tool.tool_type or "other"
                    if tool_type not in by_type:
                        by_type[tool_type] = []
                    by_type[tool_type].append(tool_name)
                except V4ConfigError:
                    continue

            return {
                "modelName": model_name,
                "compatibleTools": compatible,
                "toolsByType": by_type,
            }
        except V4ConfigError as e:
            return {
                "error": True,
                "message": str(e),
            }

    def check_ecosystem_dependencies(
        self, ecosystem_name: str
    ) -> Dict[str, Any]:
        """
        Check if ecosystem dependencies are installed.

        Args:
            ecosystem_name: Ecosystem name

        Returns:
            {
                "ecosystem": "transformers",
                "allInstalled": true/false,
                "dependencies": {
                    "transformers": true,
                    "torch": true,
                    ...
                },
                "missing": ["package1", ...]
            }
        """
        try:
            deps = self._manager.check_ecosystem_dependencies(ecosystem_name)
            missing = [pkg for pkg, installed in deps.items() if not installed]

            return {
                "ecosystem": ecosystem_name,
                "allInstalled": len(missing) == 0,
                "dependencies": deps,
                "missing": missing,
            }
        except V4ConfigError as e:
            return {
                "error": True,
                "message": str(e),
            }


# Convenience function for creating API instance
def get_gui_api(base_path: Optional[Path] = None) -> GUIAPI:
    """Get a GUI API instance."""
    return GUIAPI(base_path)
