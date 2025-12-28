# WhisperJAV v1.7.4 Release Notes

**Release Date:** December 2024
**Status:** Stable

---

## About This Release

v1.7.4 introduces Semantic Audio Clustering, a new scene detection method that segments audio based on acoustic texture rather than silence. This release also includes bug fixes for async processing mode and improved help text.

---

## What's New

### Semantic Audio Clustering Scene Detection

A new scene detection method that uses MFCC features and agglomerative clustering to group acoustically similar segments together. Unlike energy-based methods (Auditok) that split at silence, semantic clustering analyzes audio texture to find natural scene boundaries.

| Method | How It Works | Best For |
|--------|--------------|----------|
| **Auditok** (default) | Splits at silence/energy drops | General use, dialogue-heavy content |
| **Silero** | Neural VAD detection | Noisy audio, mixed content |
| **Semantic** (new) | MFCC texture clustering | Content with continuous background music, variable acoustic environments |

#### Semantic Clustering Presets

| Preset | Segment Length | When To Use |
|--------|---------------|-------------|
| `default` | 20-420s | General use |
| `dialogue_heavy` | 15-300s | Frequent dialogue with short pauses |
| `music_heavy` | 10-180s | Background music throughout |
| `action_content` | 25-420s | Fast-paced content with variable audio |
| `conservative` | 30-420s | Prefer longer, stable segments |
| `aggressive` | 10-180s | Maximum segmentation |

### CLI Usage

```bash
# Use semantic scene detection
whisperjav video.mp4 --mode balanced --scene-detection-method semantic

# In ensemble mode
whisperjav video.mp4 --ensemble \
    --pass1-scene-detector semantic \
    --pass2-scene-detector auditok
```

### GUI Integration

Semantic is available in:
- **Transcription Mode** → Scene Detection dropdown
- **Two-Pass Ensemble** → Pass 1/Pass 2 Scene Detector dropdowns

---

## Bug Fixes

### Async Processing Mode Fixed (Issue #87)

Fixed `NameError: name 'language_code' is not defined` when using `--async-mode`. The language code mapping was missing in the async processing function.

**Before:** Async mode crashed immediately on startup
**After:** Async mode works correctly with proper language handling

### Speech Segmenter Help Text Updated

The CLI help text for `--pass1-speech-segmenter` and `--pass2-speech-segmenter` now correctly lists all available options: `silero`, `ten`, `nemo`, `whisper-vad`, `none`.

---

## Dependency Updates

### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `scikit-learn` | ≥1.3.0 | Agglomerative clustering for semantic scene detection |

Note: `scikit-learn` is a widely-used ML library with no additional system requirements.

---

## Installation

* * *

### Upgrade from v1.7.3.x (Recommended for existing users)

If you have WhisperJAV v1.7.3, v1.7.3.post1, v1.7.3.post2, v1.7.3.post3, or v1.7.3.post4 installed:

1. Open a **Command Prompt** or **PowerShell** terminal
2. Navigate to your WhisperJAV installation folder (e.g., `%LOCALAPPDATA%\WhisperJAV`)
3. Execute the following command:

```
Scripts\pip.exe install -U --no-deps git+https://github.com/meizhong986/whisperjav.git
```

4. Then install the one new dependency:

```
Scripts\pip.exe install scikit-learn>=1.3.0
```

You should see: `Successfully installed whisperjav-1.7.4`

**Why this is safe:** This upgrade only adds one new package (`scikit-learn`) which has no conflicts with your existing installation.

* * *

### New Installation / Upgrade from versions prior to v1.7.3

If you have WhisperJAV v1.7.2 or earlier, or are installing fresh:

**Download and run the standalone installer:**

`WhisperJAV-1.7.4-Windows-x86_64.exe` (attached below)

This is the recommended path as it ensures all dependencies are correctly installed without pip resolution conflicts.

* * *

### Python Expert Method

For developers or advanced users managing their own Python environment:

```bash
pip install -U git+https://github.com/meizhong986/whisperjav.git@v1.7.4
```

Note: Ensure you have PyTorch with CUDA support installed first if you want GPU acceleration.

---

## Technical Details

### Semantic Clustering Integration

The semantic clustering engine is vendored at `whisperjav/vendor/semantic_audio_clustering.py` and uses a loose-coupling adapter pattern. It:

1. Analyzes audio using MFCC features and texture analysis
2. Groups similar acoustic segments using agglomerative clustering
3. Snaps boundaries to low-energy points for clean cuts
4. Outputs segments with ASR-compatible timestamps (includes 0.2s overlap padding)

Configuration is available through the V4 YAML system at:
`whisperjav/config/v4/ecosystems/tools/semantic-scene-detection.yaml`

### Files Added

- `whisperjav/vendor/semantic_audio_clustering.py` - Core engine
- `whisperjav/vendor/__init__.py` - Vendor package init
- `whisperjav/modules/scene_detection_backends/semantic_adapter.py` - WhisperJAV adapter
- `whisperjav/modules/scene_detection_backends/__init__.py` - Backend package init
- `whisperjav/config/v4/ecosystems/tools/semantic-scene-detection.yaml` - YAML config

---

## Known Issues

- Semantic clustering adds ~10-20% processing time compared to Auditok (feature extraction overhead)
- Best suited for content with distinct acoustic scenes; may not improve results for already-clean dialogue

---

## Feedback

Report issues or suggestions at:
https://github.com/meizhong986/whisperjav/issues
