"""
WhisperJAV Multi-Layer Visualization Utility

Standalone tool for visualizing WhisperJAV processing outputs:
- Layer 1: Waveform (16kHz mono)
- Layer 2: Scene Detection Pass 1 (coarse boundaries)
- Layer 3: Scene Detection Pass 2 (fine boundaries)
- Layer 4: VAD Segments (Silero speech regions)
- Layer 5: SRT Subtitles

Usage:
    python -m scripts.visualization.viz_cli \\
        --audio ./temp/video_extracted.wav \\
        --metadata ./temp/video_master.json \\
        --srt ./output/video.srt \\
        --output ./output/visualization.html

Prerequisites:
    pip install -r scripts/visualization/requirements.txt
"""

__version__ = "1.0.0"
