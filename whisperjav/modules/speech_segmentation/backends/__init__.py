"""
Speech segmentation backend implementations.

Available backends:
- silero: Silero VAD (v4.0, v3.1) via torch.hub
- silero-v6.2: Silero VAD v6.2 via pip (max_speech_duration_s + hysteresis)
- nemo: NVIDIA NeMo VAD
- ten: TEN Framework VAD
- none: Passthrough (no segmentation)
"""

# Backends are lazily imported via factory to avoid unnecessary dependencies
