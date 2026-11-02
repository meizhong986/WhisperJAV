"""R-SCN: the scene detector that cuts where the sound's characteristics change (owner W431, W448-W452).

Built on the owner's reference R-SCN 1.0. Design: docs/plans/design/2026-11-02-rscn-design.md. Modules:
settings (every value, checked), analysis (measures, change strength, digital silence, quiet stretches), speech (the
speech-measure contract and its adapter), picture (black picture), partition (cuts and marks), backend (the
SceneDetector). Importing this package loads numpy and scipy only.
"""

from .backend import RSCNSceneDetector
from .settings import PREFIX, PRESETS, RSCNSettings, overrides_from_params

__all__ = ["RSCNSceneDetector", "RSCNSettings", "PRESETS", "PREFIX", "overrides_from_params"]
