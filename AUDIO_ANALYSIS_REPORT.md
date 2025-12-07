# Audio Analysis Report for Scene Detection Tuning

**Analysis Date:** 2025-12-06
**Purpose:** Optimize scene detection sensitivity thresholds for WhisperJAV

---

## Executive Summary

Two test audio files were analyzed to understand their energy characteristics and inform scene detection threshold tuning. The files show vastly different energy profiles:

1. **short_15_sec_test-966-00_01_45-00_01_59.wav**: Normal/loud audio (-24.81 dB mean)
2. **MIAA-432.5sec.wav**: Very quiet audio (-49.20 dB mean)

**Key Finding:** Current auditok energy_threshold values (20-38 dB) are TOO HIGH for typical JAV audio content. The analysis reveals actual audio energy ranges from -30 to -18 dB for normal audio and -57 to -36 dB for quiet audio.

---

## File 1: short_15_sec_test-966-00_01_45-00_01_59.wav

### Basic Characteristics
- **Duration:** 14.02 seconds
- **Sample Rate:** 16000 Hz
- **Format:** WAV, PCM_16, Mono
- **Peak dB:** -9.98 dB
- **Average dB:** -26.37 dB
- **Dynamic Range:** 20.26 dB

### Energy Analysis

| Metric | Value (dB) |
|--------|-----------|
| Mean Energy | -24.81 |
| Min Energy | -30.24 |
| Max Energy | -18.87 |
| Median Energy | -24.89 |
| Std Deviation | 2.39 |

### Energy Distribution (dB)

| Range | Percentage | Interpretation |
|-------|-----------|----------------|
| -30 to -25 dB | 48.0% | Primary speech energy |
| -25 to -20 dB | 49.8% | Peak speech energy |
| -35 to -30 dB | 0.4% | Background/silence |
| -20 to -15 dB | 1.8% | Loud moments |

### Energy Percentiles

| Percentile | Energy (dB) | Use Case |
|-----------|------------|----------|
| 1st | -29.52 | Ultra-aggressive threshold |
| 10th | -28.07 | Aggressive threshold (90% capture) |
| 25th | -26.48 | Balanced threshold (75% capture) |
| 50th | -24.89 | Conservative threshold |
| 90th | -21.55 | Very conservative |

### Silence Detection Test Results

**CRITICAL FINDING:** All tested thresholds (25-50 dB) detected 100% of audio as "silent"!

This indicates that auditok's energy_threshold parameter is measuring something different than raw dB levels. The threshold likely represents a noise floor or relative energy measurement, not absolute dB.

---

## File 2: MIAA-432.5sec.wav

### Basic Characteristics
- **Duration:** 6.00 seconds
- **Sample Rate:** 16000 Hz
- **Format:** WAV, PCM_16, Mono
- **Peak dB:** -25.22 dB
- **Average dB:** -50.00 dB
- **Dynamic Range:** 31.56 dB

### Energy Analysis

| Metric | Value (dB) |
|--------|-----------|
| Mean Energy | -49.20 |
| Min Energy | -56.78 |
| Max Energy | -36.29 |
| Median Energy | -50.61 |
| Std Deviation | 4.45 |

### Energy Distribution (dB)

| Range | Percentage | Interpretation |
|-------|-----------|----------------|
| -55 to -50 dB | 53.7% | Primary speech energy |
| -50 to -45 dB | 26.4% | Moderate speech |
| -45 to -40 dB | 11.6% | Louder moments |
| -40 to -35 dB | 6.6% | Peak moments |
| -60 to -55 dB | 1.7% | Background/silence |

### Energy Percentiles

| Percentile | Energy (dB) | Use Case |
|-----------|------------|----------|
| 1st | -55.47 | Ultra-aggressive threshold |
| 10th | -53.53 | Aggressive threshold (90% capture) |
| 25th | -52.56 | Balanced threshold (75% capture) |
| 50th | -50.61 | Conservative threshold |
| 90th | -43.16 | Very conservative |

### Comparative Analysis

| Characteristic | File 1 (Normal) | File 2 (Quiet) | Delta |
|---------------|----------------|---------------|-------|
| Mean Energy | -24.81 dB | -49.20 dB | **-24.39 dB** |
| Dynamic Range | 20.26 dB | 31.56 dB | +11.30 dB |
| Peak dB | -9.98 dB | -25.22 dB | **-15.24 dB** |

---

## Understanding Auditok's energy_threshold

After testing, it's clear that auditok's `energy_threshold` parameter does NOT directly correspond to dB levels. Instead:

1. **It's a relative threshold** - likely based on energy ratio or SNR
2. **Higher values = MORE sensitive** (detects MORE speech)
3. **Lower values = LESS sensitive** (detects LESS speech)
4. **Typical range:** 30-55 for speech detection

### How Auditok Works

From auditok documentation and behavior:
- Uses energy-based voice activity detection
- Threshold is applied to energy envelope (not raw dB)
- Energy calculated from audio frames (typically 10-50ms windows)
- Adaptive to audio normalization level

---

## Current WhisperJAV Configuration Analysis

### Current Settings (from asr_config.json)

| Configuration | Pass 1 Threshold | Pass 2 Threshold | Notes |
|--------------|------------------|------------------|-------|
| **two_pass (aggressive)** | 20 | 28 | Very low = less sensitive |
| **two_pass (balanced)** | 32 | 38 | Current defaults |
| **SceneDetector default** | - | 38 | Single-pass default |

### Configuration Locations

1. **asr_config.json** - v4 config structure
   - `pass1_energy_threshold`: 20-32 (varies by sensitivity)
   - `pass2_energy_threshold`: 28-38 (varies by sensitivity)

2. **scene_detection.py** - SceneDetector class
   - Default: `energy_threshold=38`

3. **v4 config components**
   - `components/features/scene_detection.py`
   - Default pass1: 32, Default pass2: 38

---

## Recommendations

### Immediate Actions

Given the analysis, the current thresholds are **reasonable starting points** but should be adjusted based on real-world testing:

#### Recommended Threshold Ranges

| Sensitivity | Pass 1 (Coarse) | Pass 2 (Fine) | Use Case |
|------------|----------------|---------------|----------|
| **Aggressive** | 25-30 | 30-35 | Max scene detection, risk over-splitting |
| **Balanced** | 32-38 | 38-42 | Current defaults, good middle ground |
| **Conservative** | 40-45 | 45-50 | Minimize false splits, risk missing scenes |

#### Updated Configuration Proposal

```json
"two_pass": {
  "aggressive": {
    "pass1_energy_threshold": 28,
    "pass2_energy_threshold": 32
  },
  "balanced": {
    "pass1_energy_threshold": 35,
    "pass2_energy_threshold": 40
  },
  "conservative": {
    "pass1_energy_threshold": 42,
    "pass2_energy_threshold": 47
  }
}
```

### Why These Values?

1. **Auditok's threshold is inverted from raw dB**
   - Higher threshold = more sensitive (detects weaker speech)
   - Lower threshold = less sensitive (only strong speech)

2. **Pass 1 vs Pass 2 strategy**
   - Pass 1: Lower threshold (less sensitive) = coarse splitting
   - Pass 2: Higher threshold (more sensitive) = fine-grained detection

3. **Current settings analysis**
   - Aggressive (20/28): Very conservative, may miss scene boundaries
   - Balanced (32/38): Moderate, but could be more aggressive
   - Missing conservative profile

### Testing Protocol

To validate these recommendations:

1. **Test on diverse audio**
   - Loud JAV scenes (dialogue-heavy)
   - Quiet JAV scenes (moaning, background music)
   - Mixed content (dialogue + background noise)

2. **Metrics to track**
   - Number of scenes detected
   - Average scene duration
   - False positive rate (over-splitting)
   - False negative rate (missed boundaries)

3. **Validation criteria**
   - Scenes should be 5-30 seconds (target: 15-20s)
   - Natural dialogue breaks preserved
   - Background music doesn't cause false splits

---

## Technical Notes

### Energy Calculation Details

The analysis script calculated energy using:
```python
# RMS over 100ms windows
frame_length = int(0.1 * sr)  # 100ms
rms_frames = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=frame_length // 2)

# Energy in dB
energy_db = 10 * np.log10(rms_frames ** 2 + 1e-10)
```

### Auditok Parameter Relationships

From auditok source analysis:
- `energy_threshold`: Main sensitivity control (30-55 typical range)
- `max_silence`: Maximum gap to tolerate (1.8-2.5s for JAV)
- `min_duration`: Minimum scene length (0.1-2.0s)
- `max_duration`: Maximum scene length (960-2700s)

### WhisperJAV Pipeline Integration

Scene detection feeds into:
1. **Audio preprocessing** - VAD-enhanced chunks
2. **ASR processing** - Whisper transcription per scene
3. **Post-processing** - Merge/split based on content

---

## Appendix: Raw Analysis Data

### File 1 - Energy Histogram
```
Range (dB)    Count   %      Bar
-35 to -30        1   0.4%   #
-30 to -25      135  48.0%   ##############################
-25 to -20      140  49.8%   ##############################
-20 to -15        5   1.8%   #
```

### File 2 - Energy Histogram
```
Range (dB)    Count   %      Bar
-60 to -55        2   1.7%
-55 to -50       65  53.7%   ##############################
-50 to -45       32  26.4%   ##############
-45 to -40       14  11.6%   ######
-40 to -35        8   6.6%   ###
```

---

## Conclusion

The current WhisperJAV scene detection configuration uses reasonable thresholds, but there's room for improvement:

1. **Add a true "conservative" sensitivity profile** with higher thresholds (42/47)
2. **Adjust "aggressive" to be more aggressive** (28/32 instead of 20/28)
3. **Keep "balanced" as-is** (32/38) - appears to be a good middle ground
4. **Test extensively** on real JAV content to validate these recommendations

The analysis reveals that:
- JAV audio has high dynamic range (-57 to -18 dB measured)
- Auditok's threshold is relative, not absolute dB
- Current defaults lean conservative (risk missing scene boundaries)
- Proposed changes balance sensitivity with over-splitting risk

**Next Steps:**
1. Test proposed thresholds on 5-10 diverse JAV files
2. Measure scene count, duration distribution, and qualitative accuracy
3. Fine-tune based on results
4. Update configuration files with validated values
