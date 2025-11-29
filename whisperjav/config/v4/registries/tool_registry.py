"""
Tool Registry for WhisperJAV v4 Configuration.

Provides access to all auxiliary tool configurations.
Tools are discovered from YAML files in:
    config/v4/ecosystems/tools/*.yaml

Usage:
    from whisperjav.config.v4.registries import get_tool_registry

    registry = get_tool_registry()

    # Get specific tool
    tool = registry.get("auditok-scene-detection")

    # Find tools by type
    scene_tools = registry.find_by_type("scene_detection")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from ..errors import ToolNotFoundError
from ..schemas.tool import ToolConfig
from .base_registry import BaseRegistry


class ToolRegistry(BaseRegistry[ToolConfig]):
    """
    Registry for tool configurations.

    Discovers and manages all auxiliary tool configs.
    """

    _config_class = ToolConfig
    _config_dir_name = "tools"
    _instance: Optional["ToolRegistry"] = None

    def _get_config_paths(self) -> List[Path]:
        """
        Get paths to all tool YAML files.

        Searches: ecosystems/tools/*.yaml
        """
        paths = []
        tools_dir = self._base_path / "tools"

        if tools_dir.exists():
            for yaml_file in tools_dir.glob("*.yaml"):
                paths.append(yaml_file)

        return paths

    def _raise_not_found(self, name: str) -> None:
        """Raise ToolNotFoundError with available tools."""
        raise ToolNotFoundError(
            tool_name=name,
            available_tools=self.list_names(),
        )

    def find_by_type(self, tool_type: str) -> List[ToolConfig]:
        """
        Find all tools of a specific type.

        Args:
            tool_type: Tool type (e.g., "scene_detection", "vad")

        Returns:
            List of tool configs of that type
        """
        self._ensure_loaded()
        return [
            t
            for t in self._configs.values()
            if t.tool_type == tool_type
        ]

    def get_resolved_config(
        self,
        tool_name: str,
        sensitivity: str = "balanced",
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Get fully resolved tool config with sensitivity and overrides.

        Args:
            tool_name: Tool name
            sensitivity: Sensitivity preset
            overrides: Additional user overrides

        Returns:
            Merged configuration dict
        """
        tool = self.get(tool_name)
        return tool.get_resolved_config(sensitivity, overrides)

    def get_gui_schema(self, tool_name: str) -> Dict[str, Any]:
        """
        Get GUI schema for a tool.

        Args:
            tool_name: Tool name

        Returns:
            GUI schema dict for frontend rendering
        """
        tool = self.get(tool_name)
        return tool.get_gui_schema()

    def list_by_type(self) -> Dict[str, List[str]]:
        """
        Get all tools organized by type.

        Returns:
            Dict mapping tool type to list of tool names
        """
        self._ensure_loaded()
        result: Dict[str, List[str]] = {}

        for tool in self._configs.values():
            tool_type = tool.tool_type or "unknown"
            if tool_type not in result:
                result[tool_type] = []
            result[tool_type].append(tool.get_name())

        return result


# Singleton accessor
_registry_instance: Optional[ToolRegistry] = None


def get_tool_registry(base_path: Optional[Path] = None) -> ToolRegistry:
    """
    Get the tool registry singleton.

    Args:
        base_path: Optional base path (only used on first call)

    Returns:
        ToolRegistry instance
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry(base_path)
    return _registry_instance
