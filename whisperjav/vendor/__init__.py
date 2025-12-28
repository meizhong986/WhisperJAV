"""
Vendored External Dependencies.

This package contains external modules that are copied into WhisperJAV
rather than installed via pip. This allows for:

1. Easier distribution (no extra pip install steps)
2. Version pinning (exact code is bundled)
3. Offline usage (no network required)

Modules:
    semantic_audio_clustering: Texture-based audio segmentation using
                               MFCC features and agglomerative clustering.
                               Source: https://github.com/[repo]/SemanticAudioClustering
                               Version: 7.0.0
"""
