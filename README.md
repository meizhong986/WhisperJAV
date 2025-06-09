# WhisperJAV

Japanese Adult Video Subtitle Generator - Optimized for JAV content transcription

## Features

- 🚀 **Three Processing Modes**:
  - **Faster**: Direct transcription with Whisper Turbo
  - **Fast**: Chunked processing with standard Whisper
  - **Balanced**: Full preprocessing with WhisperWithVAD

- 🎯 **JAV-Optimized**:
  - Specialized for Japanese adult content
  - Handles background music and vocal sounds
  - Removes common hallucinations

- 🔧 **Advanced Processing**:
  - Automatic audio extraction
  - Intelligent chunking
  - Segment classification
  - Post-processing and cleanup

## Installation

```bash
pip install -r requirements.txt
python setup.py install
```

## Quick Start

```bash
# Process single file
whisperjav video.mp4

# Process with faster mode
whisperjav video.mp4 --mode faster

# Process directory
whisperjav /path/to/videos/*.mp4 --output-dir ./subtitles
```

## Requirements

- Python 3.8+
- FFmpeg
- CUDA-capable GPU (recommended)

## License

MIT License
