"""
Speech enhancement backend implementations.

Each backend in this package implements the SpeechEnhancer protocol
defined in base.py.

Available backends:
- none: Passthrough (no enhancement)
- crispasr_separator: Mel-Band-RoFormer and Mini-BS-RoFormer vocal isolation, run by CrispASR (1.10.2)
"""

# Backends are lazy-loaded by the factory to avoid import overhead
