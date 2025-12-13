"""
Speech enhancement backend implementations.

Each backend in this package implements the SpeechEnhancer protocol
defined in base.py.

Available backends:
- none: Passthrough (no enhancement)
- clearvoice: ClearerVoice speech enhancement
- bs_roformer: BS-RoFormer vocal isolation
"""

# Backends are lazy-loaded by the factory to avoid import overhead
