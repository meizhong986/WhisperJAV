"""
Feature Components.

Each feature (scene detection, post-processing, etc.) is defined in its own module.
"""

# Import all feature components to trigger registration
from .scene_detection import AuditokSceneDetection, SileroSceneDetection

__all__ = [
    'AuditokSceneDetection',
    'SileroSceneDetection',
]
