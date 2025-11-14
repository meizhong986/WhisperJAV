# SRT Coverage Analysis Tool

Automated diagnostic tool for analyzing SRT subtitle coverage by comparing reference subtitles against WhisperJAV test outputs.

## Purpose

This tool helps identify missing or partially captured dialogue segments in WhisperJAV transcriptions by:
1. **Comparing temporal coverage** between reference and test SRT files
2. **Tracing missing segments** through pipeline metadata to determine root causes
3. **Extracting media chunks** for manual review of missing/partial segments
4. **Generating visual reports** with Gantt charts and detailed analysis

## Features

✅ **Basic SRT Statistics** - Line counts and total durations displayed early in the process
✅ **Interval Coverage Analysis** - Pure temporal overlap calculation (≥60% coverage = pass)
✅ **Root Cause Tracing** - Identify why segments were missed (not in scene, VAD filtered, sanitization, etc.)
✅ **Media Chunk Extraction** - Auto-extract missing/partial segments with ±1s padding for review
✅ **Visual Gantt Charts** - Both static PNG and interactive HTML timelines
✅ **Comprehensive Reports** - HTML (for humans) + JSON (for machines)

## Installation

### Prerequisites

```bash
# Required
pip install matplotlib

# Optional (for interactive charts)
pip install plotly

# FFmpeg must be in PATH (for media extraction)
```

### No Installation Needed

The tool is self-contained in the `scripts/` directory:

```
scripts/
├── test_srt_coverage.py           # CLI entry point
└── srt_coverage_analysis/         # Analysis modules
    ├── __init__.py
    ├── srt_parser.py
    ├── coverage_calculator.py
    ├── metadata_tracer.py
    ├── media_extractor.py
    ├── gantt_visualizer.py
    └── report_generator.py
```

## Usage

### Basic Usage

```bash
python scripts/test_srt_coverage.py \
  --media F:/WHISPER/TEST/video.mp4 \
  --reference F:/WHISPER/TEST/reference.srt \
  --test output/video.ja.whisperjav.srt \
  --metadata .temp_dir/video_metadata.json \
  --output results/analysis_001/
```

### Command Line Options

| Option | Required | Description | Default |
|--------|----------|-------------|---------|
| `--media` | ✅ | Path to source media file (video/audio) | - |
| `--reference` | ✅ | Path to reference SRT (ground truth) | - |
| `--test` | ✅ | Path to test SRT (WhisperJAV output) | - |
| `--output` | ✅ | Output directory for results | - |
| `--metadata` | ⚪ | Path to WhisperJAV metadata JSON (enables root cause tracing) | None |
| `--coverage-threshold` | ⚪ | Minimum coverage fraction (0.0-1.0) | 0.60 (60%) |
| `--padding` | ⚪ | Padding in seconds for media extraction | 1.0s |
| `--skip-extraction` | ⚪ | Skip media chunk extraction (faster) | False |
| `--skip-interactive` | ⚪ | Skip interactive timeline (requires Plotly) | False |

### Example with Custom Threshold

```bash
python scripts/test_srt_coverage.py \
  --media video.mp4 \
  --reference reference.srt \
  --test test.srt \
  --metadata metadata.json \
  --coverage-threshold 0.70 \
  --padding 2.0 \
  --output results/
```

## Output Structure

```
results/
├── report.html                    # Comprehensive HTML report (open in browser)
├── report.json                    # Machine-readable JSON data
├── timeline.png                   # Static Gantt chart
├── timeline_interactive.html      # Interactive timeline (if Plotly available)
└── chunks/                        # Extracted media segments
    ├── seg_0001_00-01-23_missing.mp4
    ├── seg_0002_00-03-45_partial.mp4
    └── ...
```

## Understanding the Coverage Algorithm

### Coverage Calculation

For each reference segment, the tool calculates what **percentage of its time interval overlaps** with any test segments:

```
Reference Segment: [10.0s ─────────────────── 15.0s]  (5.0s duration)
Test Segment:            [11.0s ──────── 14.0s]        (3.0s overlap)

Coverage = 3.0 / 5.0 = 60% → COVERED ✓
```

### Status Classification

| Status | Coverage | Meaning | Color |
|--------|----------|---------|-------|
| **COVERED** | ≥60% | Segment adequately captured | 🟢 Green |
| **PARTIAL** | >0% <60% | Segment partially captured | 🟡 Yellow |
| **MISSING** | 0% | Segment not captured at all | 🔴 Red |

### Root Cause Categories

When metadata is provided, the tool traces missing/partial segments to determine why they were lost:

| Root Cause | Description | Action |
|------------|-------------|--------|
| `NOT_IN_SCENE` | Segment not captured by scene detection | Adjust scene detection sensitivity |
| `SCENE_NOT_TRANSCRIBED` | Scene detected but ASR failed | Check ASR logs |
| `FILTERED_OR_FAILED` | Transcribed but filtered (VAD/sanitization) | Review VAD/sanitization thresholds |

## Report Interpretation

### HTML Report Sections

1. **SRT File Statistics** - Basic stats for reference and test SRT files (line counts, total durations, timeline spans)
2. **Coverage Analysis Summary** - Overall coverage metrics
3. **Root Cause Analysis** - Distribution of failure reasons
4. **Timeline Visualization** - Visual Gantt chart showing coverage gaps
5. **Segments Needing Review** - Table of missing/partial segments with details
6. **Media Extraction Statistics** - Chunk extraction results

### JSON Report Structure

```json
{
  "metadata": {
    "reference_srt": "path/to/reference.srt",
    "test_srt": "path/to/test.srt",
    "coverage_threshold": 0.60,
    "analysis_timestamp": "2025-11-14T18:30:00Z"
  },
  "summary": {
    "total_segments": 50,
    "covered_segments": 42,
    "partial_segments": 5,
    "missing_segments": 3,
    "coverage_rate": 0.84
  },
  "segments": [
    {
      "index": 23,
      "start": 123.4,
      "end": 128.9,
      "text": "こんにちは",
      "coverage_percent": 0.0,
      "status": "MISSING",
      "root_cause": "NOT_IN_SCENE",
      "media_chunk": "chunks/seg_0023_02-03_missing.mp4"
    }
  ]
}
```

## Workflow Example

### 1. Run WhisperJAV with `--keep-temp`

```bash
whisperjav video.mp4 --mode balanced --keep-temp
```

This generates:
- `output/video.ja.whisperjav.srt` (test output)
- `.temp_dir/video_metadata.json` (pipeline metadata)

### 2. Run Coverage Analysis

```bash
python scripts/test_srt_coverage.py \
  --media video.mp4 \
  --reference ground_truth.srt \
  --test output/video.ja.whisperjav.srt \
  --metadata .temp_dir/video_metadata.json \
  --output analysis/
```

### 3. Review Results

1. Open `analysis/report.html` in browser
2. Check summary statistics and coverage rate
3. Review timeline visualization for gaps
4. Inspect media chunks in `analysis/chunks/` folder
5. Use JSON data for automated processing

### 4. Iterate and Improve

Based on root causes:
- **NOT_IN_SCENE** → Adjust scene detection parameters
- **SCENE_NOT_TRANSCRIBED** → Check ASR model/config
- **FILTERED_OR_FAILED** → Review VAD/sanitization thresholds

## Troubleshooting

### FFmpeg Not Found

```
Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.
```

**Solution**: Install FFmpeg and add to system PATH:
- Windows: Download from https://ffmpeg.org/download.html
- Linux: `sudo apt install ffmpeg`
- macOS: `brew install ffmpeg`

### Plotly Not Installed

```
Warning: Plotly not installed. Skipping interactive chart.
```

**Solution**: Install Plotly (optional):
```bash
pip install plotly
```

### File Encoding Issues

If SRT parsing fails with encoding errors:
- Tool auto-detects UTF-8, UTF-16, and Latin-1
- Manually convert if needed: `iconv -f ENCODING -t UTF-8 input.srt > output.srt`

## Advanced Usage

### Skip Media Extraction (Faster Analysis)

```bash
python scripts/test_srt_coverage.py \
  --media video.mp4 \
  --reference ref.srt \
  --test test.srt \
  --skip-extraction \
  --output results/
```

### Analysis Without Metadata (Coverage Only)

```bash
python scripts/test_srt_coverage.py \
  --media video.mp4 \
  --reference ref.srt \
  --test test.srt \
  --output results/
# Root cause tracing will be skipped
```

### Batch Processing

```bash
for test_file in test_outputs/*.srt; do
  basename=$(basename "$test_file" .srt)
  python scripts/test_srt_coverage.py \
    --media media/"$basename".mp4 \
    --reference references/"$basename".srt \
    --test "$test_file" \
    --output results/"$basename"/
done
```

## Module Testing

Each module can be tested independently:

```bash
# Test SRT parser
python -m srt_coverage_analysis.srt_parser reference.srt

# Test coverage calculator
python -m srt_coverage_analysis.coverage_calculator reference.srt test.srt

# Test metadata tracer
python -m srt_coverage_analysis.metadata_tracer metadata.json

# Test media extractor
python -m srt_coverage_analysis.media_extractor video.mp4 reference.srt chunks/

# Test Gantt visualizer
python -m srt_coverage_analysis.gantt_visualizer reference.srt test.srt charts/
```

## Version

**1.0.0** - Initial release with WhisperJAV v1.5.4

## Support

For issues or questions:
1. Check this README first
2. Review example outputs in `results/` directory
3. Examine JSON report for detailed diagnostics
4. Report bugs via GitHub Issues

---

**Happy Testing! 🎬📊**
