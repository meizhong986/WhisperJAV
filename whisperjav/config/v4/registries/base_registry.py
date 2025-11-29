"""
Base Registry Class for WhisperJAV v4 Configuration.

Provides common functionality for all registry types:
- Lazy loading from YAML files
- Caching for performance
- Search/filter capabilities
- Thread-safe singleton pattern

All specific registries (Model, Tool, Ecosystem, Preset) inherit from this.
"""

import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, Set, Type, TypeVar

from ..errors import V4ConfigError
from ..loaders import YAMLLoader
from ..schemas.base import ConfigBase

T = TypeVar("T", bound=ConfigBase)


class BaseRegistry(ABC, Generic[T]):
    """
    Abstract base class for configuration registries.

    Provides:
    - Singleton pattern with thread safety
    - Lazy loading from YAML directory
    - Caching with optional refresh
    - Search by name, tags, ecosystem
    """

    _instance: Optional["BaseRegistry"] = None
    _lock = threading.Lock()

    # Subclasses must set these
    _config_class: Type[T]  # Pydantic config class
    _config_dir_name: str  # Directory name under ecosystems/

    def __new__(cls, *args, **kwargs):
        """Thread-safe singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the registry.

        Args:
            base_path: Base path for config files. Only used on first init.
        """
        if self._initialized:
            return

        self._base_path = base_path or self._get_default_base_path()
        self._loader = YAMLLoader(self._base_path)
        self._configs: Dict[str, T] = {}
        self._loaded = False
        self._initialized = True

    def _get_default_base_path(self) -> Path:
        """Get default base path for this registry type."""
        return Path(__file__).parent.parent / "ecosystems"

    @abstractmethod
    def _get_config_paths(self) -> List[Path]:
        """
        Get list of YAML file paths to load.

        Subclasses implement this to define where configs are found.
        """
        pass

    def _ensure_loaded(self) -> None:
        """Ensure configs are loaded (lazy loading)."""
        if not self._loaded:
            self._load_all()

    def _load_all(self) -> None:
        """Load all configs from YAML files."""
        with self._lock:
            if self._loaded:
                return

            self._configs.clear()
            paths = self._get_config_paths()

            for path in paths:
                try:
                    config = self._load_config(path)
                    self._configs[config.get_name()] = config
                except V4ConfigError as e:
                    # Log warning but continue loading others
                    import logging

                    logging.warning(f"Failed to load config {path}: {e}")

            self._loaded = True

    def _load_config(self, path: Path) -> T:
        """Load a single config file."""
        return self._loader._load_config(path, self._config_class, resolve_extends=True)

    def get(self, name: str) -> T:
        """
        Get config by name.

        Args:
            name: Unique config name

        Returns:
            Config instance

        Raises:
            Subclass-specific NotFoundError if not found
        """
        self._ensure_loaded()
        if name not in self._configs:
            self._raise_not_found(name)
        return self._configs[name]

    @abstractmethod
    def _raise_not_found(self, name: str) -> None:
        """Raise appropriate NotFoundError. Subclasses implement this."""
        pass

    def get_or_none(self, name: str) -> Optional[T]:
        """Get config by name, returning None if not found."""
        self._ensure_loaded()
        return self._configs.get(name)

    def exists(self, name: str) -> bool:
        """Check if a config exists."""
        self._ensure_loaded()
        return name in self._configs

    def list_all(self, include_deprecated: bool = False) -> List[T]:
        """
        List all configs.

        Args:
            include_deprecated: If True, include deprecated configs

        Returns:
            List of all config instances
        """
        self._ensure_loaded()
        configs = list(self._configs.values())

        if not include_deprecated:
            configs = [c for c in configs if not c.is_deprecated()]

        return configs

    def list_names(self, include_deprecated: bool = False) -> List[str]:
        """List all config names."""
        return [c.get_name() for c in self.list_all(include_deprecated)]

    def find_by_tag(self, tag: str) -> List[T]:
        """
        Find configs that have a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of matching configs
        """
        self._ensure_loaded()
        tag_lower = tag.lower()
        return [
            c
            for c in self._configs.values()
            if tag_lower in [t.lower() for t in c.metadata.tags]
        ]

    def find_by_tags(self, tags: Set[str], match_all: bool = False) -> List[T]:
        """
        Find configs matching tags.

        Args:
            tags: Set of tags to search for
            match_all: If True, config must have ALL tags. If False, ANY tag.

        Returns:
            List of matching configs
        """
        self._ensure_loaded()
        tags_lower = {t.lower() for t in tags}
        results = []

        for config in self._configs.values():
            config_tags = {t.lower() for t in config.metadata.tags}

            if match_all:
                if tags_lower.issubset(config_tags):
                    results.append(config)
            else:
                if tags_lower.intersection(config_tags):
                    results.append(config)

        return results

    def refresh(self) -> None:
        """Force reload all configs from disk."""
        with self._lock:
            self._loaded = False
            self._loader.clear_cache()
        self._ensure_loaded()

    def add_config(self, config: T) -> None:
        """
        Programmatically add a config (for testing/runtime).

        Args:
            config: Config instance to add
        """
        self._ensure_loaded()
        self._configs[config.get_name()] = config

    def remove_config(self, name: str) -> bool:
        """
        Remove a config by name.

        Args:
            name: Config name to remove

        Returns:
            True if removed, False if not found
        """
        self._ensure_loaded()
        if name in self._configs:
            del self._configs[name]
            return True
        return False

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            cls._instance = None
