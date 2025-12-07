# Visualization Tool - Resume Notes

## Status: PAUSED (Ready for Testing)

**Date Paused**: 2025-12-07

---

## What Was Completed

### 1. Coarse Boundaries (Pass 1) Capture
- **Files Modified**:
  - `whisperjav/modules/scene_detection.py` - Added `coarse_boundaries` capture after auditok Pass 1
  - `whisperjav/pipelines/balanced_pipeline.py` - Saves `coarse_boundaries` to master_metadata
- **Status**: Code complete, tested, working
- **Verification**: Run `pip install -e .` then re-run WhisperJAV with `--keep-temp`

### 2. Plotly HTML Visualization
- **Files**:
  - `scripts/visualization/plotly_renderer.py` - 5-track DAW-style visualization
  - `scripts/visualization/data_loader.py` - Parses coarse_boundaries from metadata
- **Features**:
  - Track 1: Waveform
  - Track 2: Scene Pass 1 (coarse_boundaries)
  - Track 3: Scene Pass 2 (all final scenes)
  - Track 4: VAD segments
  - Track 5: Subtitles
  - Timeline at top (hh:mm:ss)
  - Vivid gray scroll bar (#4a4a6a)

### 3. DaVinci Resolve Export (NEW)
- **Files Created**:
  - `scripts/visualization/resolve/__init__.py`
  - `scripts/visualization/resolve/metadata_to_srt.py` - Converts metadata to SRT files
  - `scripts/visualization/resolve_export_cli.py` - CLI tool
- **Usage**:
  ```bash
  python -m scripts.visualization.resolve_export_cli \
      --metadata ./temp/video_master.json \
      --srt ./output/video.ja.whisperjav.srt \
      --output-dir ./resolve_import/ \
      --prefix "video" \
      --generate-script
  ```
- **Output**: 4 SRT files (scene_pass1, scene_pass2, vad_segments, subtitles) + optional Resolve import script

---

## What Remains To Do

1. **User Action Required**: Re-run WhisperJAV with `--keep-temp` to generate new metadata with `coarse_boundaries`
2. **Test Visualization**: Generate new HTML visualization with the new metadata
3. **Test Resolve Export**: Import generated SRTs into DaVinci Resolve 20
4. **Verify Pass 1 Display**: Confirm 8 coarse boundaries appear in visualization

---

## Key Files Changed (Uncommitted)

```
M whisperjav/modules/scene_detection.py      # coarse_boundaries capture
M whisperjav/pipelines/balanced_pipeline.py  # save to metadata
M scripts/visualization/data_loader.py       # parse coarse_boundaries
M scripts/visualization/plotly_renderer.py   # use coarse_boundaries for Pass 1
+ scripts/visualization/resolve/__init__.py  # NEW
+ scripts/visualization/resolve/metadata_to_srt.py  # NEW
+ scripts/visualization/resolve_export_cli.py       # NEW
```

---

## Resume Commands

```bash
# 1. Verify changes are still in place
cd C:\BIN\git\whisperJav_V1_Minami_Edition
git status

# 2. Reinstall if needed
pip install -e . --no-deps

# 3. Re-run WhisperJAV to generate new metadata
whisperjav video.mp4 --mode balanced --keep-temp

# 4. Generate Plotly visualization
python -m scripts.visualization.viz_cli \
    --metadata ./temp/video_master.json \
    --srt ./output/video.ja.whisperjav.srt \
    --output ./output/video_viz.html

# 5. Generate Resolve SRTs
python -m scripts.visualization.resolve_export_cli \
    --metadata ./temp/video_master.json \
    --srt ./output/video.ja.whisperjav.srt \
    --output-dir ./resolve_import/
```

---

## Test Verification

```python
# Verify coarse_boundaries are captured
from whisperjav.modules.scene_detection import DynamicSceneDetector
d = DynamicSceneDetector()
print(hasattr(d, 'coarse_boundaries'))  # Should be True
meta = d.get_detection_metadata()
print('coarse_boundaries' in meta)  # Should be True
```
