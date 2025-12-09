"""
Speech segmentation backend implementations.

Available backends:
- silero: Silero VAD (v4.0, v3.1)
- nemo: NVIDIA NeMo VAD
- ten: TEN Framework VAD
- none: Passthrough (no segmentation)
"""

# Backends are lazily imported via factory to avoid unnecessary dependencies
