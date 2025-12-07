# WhisperJAV Multi-Track Visualization Tool

Standalone utility for visualizing WhisperJAV processing outputs. Generates interactive HTML files with a **DAW-style multi-track timeline** showing 5 horizontal tracks.

## Visualization Layout

```
┌────────────────────────────────────────────────────────────────┐
│  Audio Analysis Timeline - WhisperJAV                          │
├────────────────────────────────────────────────────────────────┤
│  1. Waveform     │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│
│  (16kHz Mono)    │                Blue amplitude envelope      │
├────────────────────────────────────────────────────────────────┤
│  2. Scene Pass 1 │ [SCENE A    ] [SCENE B  ] [SCENE C ] [D   ]│
│                  │        Olive/yellow filled blocks           │
├────────────────────────────────────────────────────────────────┤
│  3. Scene Pass 2 │ [A.1][A.2 ] [B.1   ] [C.1][C.2] [D.1]      │
│                  │        Olive/yellow filled blocks           │
├────────────────────────────────────────────────────────────────┤
│  4. VAD (Silero) │ [SPEECH] [SPEECH]   [SPEECH]    [SPEECH]   │
│                  │        Green filled blocks                  │
├────────────────────────────────────────────────────────────────┤
│  5. Subtitles    │ [00:02-00:05]  [00:13-00:18]  [00:21-00:25]│
│                  │  Hello world    Moving...      Final caption│
└────────────────────────────────────────────────────────────────┘
│◄──────────────── Range Slider for Zoom/Pan ────────────────────►│
```

## Track Details

| Track | Description | Visual Style | Data Source |
|-------|-------------|--------------|-------------|
| 1. Waveform | Audio amplitude envelope | Blue filled area | `*_extracted.wav` |
| 2. Scene Pass 1 | Coarse scene boundaries | Olive/yellow blocks with labels | `*_master.json` |
| 3. Scene Pass 2 | Fine scene boundaries | Olive/yellow blocks with labels | `*_master.json` |
| 4. VAD Segments | Speech regions (ASR preprocessing) | Green blocks with "SPEECH" labels | `*_master.json` |
| 5. Subtitles | Subtitle entries | Purple blocks with timestamp + text | `*.srt` |

## Installation

The visualization tool has its own dependencies, separate from WhisperJAV:

```bash
pip install -r scripts/visualization/requirements.txt
```

Required packages:
- `plotly>=5.0.0` - Interactive charts
- `numpy>=1.20.0` - Array processing
- `librosa>=0.9.0` - Audio processing
- `pysrt>=1.1.2` - SRT parsing

## Prerequisites: Generating Data

Run WhisperJAV with the `--keep-temp` flag to preserve intermediate files:

```bash
whisperjav video.mp4 --mode balanced --keep-temp
```

This preserves:
- `temp/{basename}_extracted.wav` - Extracted audio (16kHz mono)
- `temp/{basename}_master.json` - Processing metadata with scene detection + VAD info
- `output/{basename}.{lang}.whisperjav.srt` - Final subtitles

## Usage Examples

### Full Visualization (All Tracks)

```bash
python -m scripts.visualization.viz_cli \
    --audio ./temp/video_extracted.wav \
    --metadata ./temp/video_master.json \
    --srt ./output/video.ja.whisperjav.srt \
    --output ./output/video_viz.html
```

### Auto-Discovery Mode

Let the tool find audio and metadata automatically:

```bash
python -m scripts.visualization.viz_cli \
    --temp-dir ./temp \
    --basename video \
    --srt ./output/video.ja.whisperjav.srt \
    --output ./output/video_viz.html
```

### Metadata + SRT Only (No Waveform)

For faster generation without audio processing:

```bash
python -m scripts.visualization.viz_cli \
    --metadata ./temp/video_master.json \
    --srt ./output/video.ja.whisperjav.srt \
    --output ./output/video_viz.html
```

## Command-Line Options

| Option | Description |
|--------|-------------|
| `--audio, -a` | Path to extracted WAV file |
| `--metadata, -m` | Path to master metadata JSON |
| `--srt, -s` | Path to output SRT file (explicit path required) |
| `--temp-dir` | WhisperJAV temp directory (auto-discovers audio & metadata) |
| `--basename` | Media basename for auto-discovery |
| `--output, -o` | Output HTML file path (required) |
| `--title` | Custom visualization title |
| `--downsample-points` | Waveform detail level (default: 10000) |

## Interactive Features

The generated HTML file includes:

- **Zoom/Pan**: Use the range slider at the bottom or drag on the chart
- **Hover Tooltips**: View detailed information about each block
- **Track Toggle**: Click legend items to show/hide individual tracks
- **Scroll Zoom**: Mouse scroll to zoom in/out
- **Export**: Download chart as PNG via the toolbar

## Data Contract

### Scene Detection Metadata

The `*_master.json` file contains `scenes_detected` with:

```json
{
  "scenes_detected": [
    {
      "scene_index": 0,
      "start_time_seconds": 0.0,
      "end_time_seconds": 29.5,
      "duration_seconds": 29.5,
      "detection_pass": 1,
      "filename": "video_scene_0000.wav"
    }
  ]
}
```

- `detection_pass: 1` = Coarse (Pass 1) - shows on Track 2
- `detection_pass: 2` = Fine (Pass 2) - shows on Track 3

### VAD Segments (ASR Preprocessing)

VAD segments are captured from the ASR module's Silero VAD preprocessing:

```json
{
  "vad_segments": [
    {"start_sec": 1.2, "end_sec": 5.8},
    {"start_sec": 10.5, "end_sec": 28.9}
  ],
  "vad_method": "silero",
  "vad_params": {
    "threshold": 0.08,
    "min_speech_duration_ms": 100
  }
}
```

These segments represent speech regions detected BEFORE transcription begins.

## Troubleshooting

### Missing Plotly

```
ImportError: Plotly is required for visualization.
```

Solution: `pip install plotly`

### Empty Tracks

If you see empty tracks:
- Ensure `--keep-temp` was used when running WhisperJAV
- Check that the metadata JSON exists and is valid
- Verify file paths are correct
- For VAD: Ensure you used `--mode balanced` (VAD is part of balanced pipeline)

### Slow Performance on Long Videos

For videos longer than 1 hour, reduce waveform detail level:

```bash
python -m scripts.visualization.viz_cli \
    --audio ./temp/long_video_extracted.wav \
    --downsample-points 5000 \
    --output ./output/viz.html
```

### Block Labels Not Showing

Labels only appear inside blocks that are wide enough:
- Scene blocks: > 2 seconds
- VAD blocks: > 1.5 seconds
- Subtitle blocks: > 2 seconds

Zoom in using the range slider to see more detail.
