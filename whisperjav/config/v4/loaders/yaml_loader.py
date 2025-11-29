"""
YAML Loader for WhisperJAV v4 Configuration.

Provides robust YAML parsing with:
- Error messages including file path and line numbers
- Schema validation using Pydantic models
- Support for extends/inheritance with cycle detection
- Safe YAML loading (no arbitrary code execution)

Usage:
    loader = YAMLLoader()
    model_config = loader.load_model("path/to/model.yaml")
    tool_config = loader.load_tool("path/to/tool.yaml")
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union

import yaml
from pydantic import BaseModel, ValidationError

from ..errors import (
    CircularDependencyError,
    IncompatibleVersionError,
    SchemaValidationError,
    YAMLParseError,
)
from ..schemas.base import ConfigBase, ConfigKind, SchemaVersion
from ..schemas.ecosystem import EcosystemConfig
from ..schemas.model import ModelConfig
from ..schemas.preset import PresetConfig
from ..schemas.tool import ToolConfig

T = TypeVar("T", bound=ConfigBase)


# Supported schema versions
SUPPORTED_VERSIONS = {SchemaVersion.V1}


class YAMLLoader:
    """
    YAML configuration loader with validation and inheritance support.

    Features:
    - Loads YAML files with clear error messages
    - Validates against Pydantic schemas
    - Supports 'extends' for config inheritance
    - Detects circular dependencies
    - Caches loaded configs for performance
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the YAML loader.

        Args:
            base_path: Base directory for resolving relative paths.
                       Defaults to the v4/ecosystems directory.
        """
        self.base_path = base_path or self._get_default_base_path()
        self._cache: Dict[Path, ConfigBase] = {}
        self._loading_stack: Set[Path] = set()  # For cycle detection

    def _get_default_base_path(self) -> Path:
        """Get the default base path for config files."""
        # config/v4/ecosystems relative to this file
        return Path(__file__).parent.parent / "ecosystems"

    def load_model(
        self,
        path: Union[str, Path],
        resolve_extends: bool = True,
    ) -> ModelConfig:
        """
        Load a model configuration file.

        Args:
            path: Path to YAML file (absolute or relative to base_path)
            resolve_extends: If True, resolve 'extends' inheritance

        Returns:
            Validated ModelConfig

        Raises:
            YAMLParseError: If YAML syntax is invalid
            SchemaValidationError: If content fails validation
            CircularDependencyError: If circular extends detected
        """
        return self._load_config(path, ModelConfig, resolve_extends)

    def load_ecosystem(
        self,
        path: Union[str, Path],
        resolve_extends: bool = True,
    ) -> EcosystemConfig:
        """
        Load an ecosystem configuration file.

        Args:
            path: Path to YAML file
            resolve_extends: If True, resolve 'extends' inheritance

        Returns:
            Validated EcosystemConfig
        """
        return self._load_config(path, EcosystemConfig, resolve_extends)

    def load_tool(
        self,
        path: Union[str, Path],
        resolve_extends: bool = True,
    ) -> ToolConfig:
        """
        Load a tool configuration file.

        Args:
            path: Path to YAML file
            resolve_extends: If True, resolve 'extends' inheritance

        Returns:
            Validated ToolConfig
        """
        return self._load_config(path, ToolConfig, resolve_extends)

    def load_preset(
        self,
        path: Union[str, Path],
        resolve_extends: bool = True,
    ) -> PresetConfig:
        """
        Load a preset configuration file.

        Args:
            path: Path to YAML file
            resolve_extends: If True, resolve 'extends' inheritance

        Returns:
            Validated PresetConfig
        """
        return self._load_config(path, PresetConfig, resolve_extends)

    def load_any(
        self,
        path: Union[str, Path],
        resolve_extends: bool = True,
    ) -> ConfigBase:
        """
        Load any configuration file, auto-detecting type from 'kind' field.

        Args:
            path: Path to YAML file
            resolve_extends: If True, resolve 'extends' inheritance

        Returns:
            Validated config of appropriate type
        """
        file_path = self._resolve_path(path)
        raw_data = self._load_raw_yaml(file_path)

        # Detect kind
        kind_str = raw_data.get("kind")
        if not kind_str:
            raise SchemaValidationError(
                "Missing required 'kind' field",
                field_path="kind",
                source_file=file_path,
            )

        # Map kind to config class
        kind_map = {
            "Model": ModelConfig,
            "Ecosystem": EcosystemConfig,
            "Tool": ToolConfig,
            "Preset": PresetConfig,
        }

        config_class = kind_map.get(kind_str)
        if not config_class:
            raise SchemaValidationError(
                f"Unknown kind: {kind_str}",
                field_path="kind",
                actual_value=kind_str,
                source_file=file_path,
            )

        return self._load_config(path, config_class, resolve_extends)

    def _load_config(
        self,
        path: Union[str, Path],
        config_class: Type[T],
        resolve_extends: bool,
    ) -> T:
        """
        Internal method to load and validate a config file.

        Args:
            path: Path to YAML file
            config_class: Pydantic model class for validation
            resolve_extends: If True, resolve inheritance

        Returns:
            Validated config instance
        """
        file_path = self._resolve_path(path)

        # Check cache
        if file_path in self._cache:
            cached = self._cache[file_path]
            if isinstance(cached, config_class):
                return cached

        # Cycle detection
        if file_path in self._loading_stack:
            chain = list(self._loading_stack) + [file_path]
            chain_names = [p.name for p in chain]
            raise CircularDependencyError(chain_names, file_path)

        self._loading_stack.add(file_path)
        try:
            raw_data = self._load_raw_yaml(file_path)

            # Validate schema version
            self._validate_version(raw_data, file_path)

            # Handle extends
            if resolve_extends and raw_data.get("extends"):
                raw_data = self._resolve_extends(raw_data, file_path, config_class)

            # Validate with Pydantic
            config = self._validate_schema(raw_data, config_class, file_path)

            # Cache and return
            self._cache[file_path] = config
            return config

        finally:
            self._loading_stack.discard(file_path)

    def _resolve_path(self, path: Union[str, Path]) -> Path:
        """Resolve path to absolute Path object."""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.base_path / path

    def _load_raw_yaml(self, file_path: Path) -> Dict[str, Any]:
        """
        Load raw YAML data from file.

        Uses safe_load to prevent arbitrary code execution.
        """
        if not file_path.exists():
            raise YAMLParseError(
                f"File not found: {file_path}",
                file_path=file_path,
            )

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if data is None:
                return {}
            if not isinstance(data, dict):
                raise YAMLParseError(
                    "YAML root must be a mapping (dict)",
                    file_path=file_path,
                )
            return data

        except yaml.YAMLError as e:
            # Extract line/column from PyYAML error
            line = column = None
            if hasattr(e, "problem_mark") and e.problem_mark:
                line = e.problem_mark.line + 1
                column = e.problem_mark.column + 1

            raise YAMLParseError(
                f"YAML syntax error: {e}",
                file_path=file_path,
                line=line,
                column=column,
                raw_error=str(e),
            )

    def _validate_version(self, data: Dict[str, Any], file_path: Path) -> None:
        """Validate schema version is supported."""
        version_str = data.get("schemaVersion", "v1")

        try:
            version = SchemaVersion(version_str)
        except ValueError:
            raise IncompatibleVersionError(
                found_version=version_str,
                supported_versions=[v.value for v in SUPPORTED_VERSIONS],
                file_path=file_path,
            )

        if version not in SUPPORTED_VERSIONS:
            raise IncompatibleVersionError(
                found_version=version_str,
                supported_versions=[v.value for v in SUPPORTED_VERSIONS],
                file_path=file_path,
            )

    def _resolve_extends(
        self,
        data: Dict[str, Any],
        file_path: Path,
        config_class: Type[T],
    ) -> Dict[str, Any]:
        """
        Resolve 'extends' inheritance.

        Loads the parent config and merges current data on top.
        """
        from .merger import deep_merge

        extends_name = data.pop("extends", None)
        if not extends_name:
            return data

        # Resolve parent path
        parent_path = file_path.parent / f"{extends_name}.yaml"
        if not parent_path.exists():
            # Try with full name
            parent_path = file_path.parent / extends_name
            if not parent_path.exists():
                raise SchemaValidationError(
                    f"Extended config not found: {extends_name}",
                    field_path="extends",
                    actual_value=extends_name,
                    source_file=file_path,
                )

        # Load parent
        parent_data = self._load_raw_yaml(parent_path)

        # Recursively resolve parent's extends
        if parent_data.get("extends"):
            parent_data = self._resolve_extends(parent_data, parent_path, config_class)

        # Merge: parent is base, current data overrides
        merged = deep_merge(parent_data, data)

        return merged

    def _validate_schema(
        self,
        data: Dict[str, Any],
        config_class: Type[T],
        file_path: Path,
    ) -> T:
        """
        Validate data against Pydantic schema.

        Converts Pydantic ValidationError to our SchemaValidationError
        with better context.
        """
        try:
            return config_class.model_validate(data)

        except ValidationError as e:
            # Get first error for main message
            errors = e.errors()
            if errors:
                first = errors[0]
                field_path = ".".join(str(p) for p in first.get("loc", []))
                message = first.get("msg", str(e))
                input_value = first.get("input")

                raise SchemaValidationError(
                    message=message,
                    field_path=field_path,
                    actual_value=input_value,
                    source_file=file_path,
                )
            else:
                raise SchemaValidationError(
                    message=str(e),
                    source_file=file_path,
                )

    def clear_cache(self) -> None:
        """Clear the config cache."""
        self._cache.clear()


# Convenience functions


def load_yaml_file(path: Union[str, Path]) -> ConfigBase:
    """
    Load any YAML config file with auto-detection.

    Args:
        path: Path to YAML file

    Returns:
        Validated config of appropriate type
    """
    loader = YAMLLoader()
    return loader.load_any(path)


def load_yaml_string(content: str, config_class: Type[T]) -> T:
    """
    Load config from YAML string.

    Args:
        content: YAML content as string
        config_class: Expected config class

    Returns:
        Validated config instance
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        raise YAMLParseError(f"YAML syntax error: {e}", raw_error=str(e))

    if not isinstance(data, dict):
        raise YAMLParseError("YAML root must be a mapping (dict)")

    try:
        return config_class.model_validate(data)
    except ValidationError as e:
        errors = e.errors()
        if errors:
            first = errors[0]
            field_path = ".".join(str(p) for p in first.get("loc", []))
            raise SchemaValidationError(
                message=first.get("msg", str(e)),
                field_path=field_path,
                actual_value=first.get("input"),
            )
        raise SchemaValidationError(message=str(e))
