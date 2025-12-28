"""
Scene Detection Backends.

This package contains backend implementations for scene detection methods.
Each backend provides a consistent interface for detecting and splitting
audio into scenes.

Available backends:
- semantic: Texture-based segmentation using MFCC features and agglomerative clustering
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .semantic_adapter import SemanticClusteringAdapter

__all__ = ["SemanticClusteringAdapter"]
