# Investigation Roadmap: v1.7.1 → v1.7.3 Changes

Based on forensic analysis of HODV-22019 subtitle differences, this document provides a targeted investigation plan.

---

## Summary of Findings

**Total Reduction**: 128 subtitles (657 → 529, -19.5%)

**Breakdown**:
1. Hallucination removal: 24 subtitles (18.8%) - ✓ WORKING CORRECTLY
2. Segment 1 loss (opening 0:00-13:43): 19 subtitles (14.8%) - ⚠️ INVESTIGATE
3. Segment 10 loss (ending 123:27-137:10): 20 subtitles (15.6%) - ⚠️ INVESTIGATE
4. Duration bias + other: 65 subtitles (50.8%) - ⚠️ INVESTIGATE

---

## Investigation Priority Matrix

| Issue | Impact | Likelihood of Bug | Priority | Investigation Time |
|-------|--------|-------------------|----------|-------------------|
| Hallucination removal | 18.8% | LOW (working correctly) | P4-LOW | N/A |
| Segment 1 catastrophic loss | 14.8% + pattern concerns | HIGH | **P1-CRITICAL** | 2-4 hours |
| Very short segment bias | 13-34% (varies) | MEDIUM | **P2-HIGH** | 1-2 hours |
| Segment 10 loss | 15.6% | MEDIUM | P3-MEDIUM | 1 hour |
| Middle segment scatter | 50.8% residual | UNKNOWN | P3-MEDIUM | 3-4 hours |

---

## P1-CRITICAL: Segment 1 Catastrophic Loss (0:00-13:43)

### Symptoms
- 95.2% loss rate (20 of 21 subtitles missing)
- First subtitle in v1.7.1: 2:28 (mostly hallucinations)
- First subtitle in v1.7.3: 6:41 (also hallucination)
- First REAL speech in v1.7.1: 5:53
- First REAL speech in v1.7.3: 9:35
- **3.7 minute gap** in speech detection

### Hypothesis
Scene detection or VAD is treating the opening differently, possibly:
1. Skipping intro music/credits entirely
2. Different silence detection thresholds
3. Different minimum scene duration

### Code Areas to Investigate

#### 1. Scene Detection Module (`whisperjav/modules/scene_detection.py`)
```bash
# Compare v1.7.1 vs v1.7.3
git diff v1.7.1 v1.7.3 -- whisperjav/modules/scene_detection.py
```

**Look for**:
- Changes to silence threshold parameters
- Changes to minimum scene duration
- Changes to scene boundary detection
- New logic for skipping opening segments

**Key parameters to check**:
- `min_silence_len` - minimum silence duration to split scenes
- `silence_thresh` - dB threshold for silence detection
- `seek_step` - granularity of silence detection
- Any new "skip intro" or "skip silence" logic

#### 2. Audio Preprocessing / VAD (`whisperjav/modules/audio_preprocessing.py`)
```bash
git diff v1.7.1 v1.7.3 -- whisperjav/modules/audio_preprocessing.py
```

**Look for**:
- Changes to Silero VAD parameters
- Changes to speech probability thresholds
- Changes to minimum speech duration
- New logic for filtering opening segments

**Key parameters**:
- `threshold` - VAD confidence threshold
- `min_speech_duration_ms` - minimum speech segment length
- `min_silence_duration_ms` - minimum silence to split
- `window_size_samples` - VAD window size

#### 3. Configuration Changes (`whisperjav/config/`)
```bash
# Check if sensitivity profiles changed
git diff v1.7.1 v1.7.3 -- whisperjav/config/transcription_tuner.py
git diff v1.7.1 v1.7.3 -- whisperjav/config/v4/ecosystems/
```

**Look for**:
- Changes to "aggressive" sensitivity parameters
- New VAD or scene detection defaults
- Changes to ASR parameters that might affect segmentation

### Manual Verification Steps

1. **Extract opening audio segment**:
   ```bash
   ffmpeg -i HODV-22019.mp4 -ss 00:00:00 -t 00:15:00 -vn -acodec pcm_s16le opening_15min.wav
   ```

2. **Listen manually**:
   - Is there speech at 2:28-5:53? (where v1.7.1 has hallucinations)
   - Is there speech at 5:53-9:35? (where v1.7.1 has real speech but v1.7.3 skips)

3. **Run scene detection in debug mode** (if available):
   ```bash
   # Check if there's a debug flag to output scene boundaries
   whisperjav opening_15min.wav --mode fast --sensitivity aggressive --verbose
   ```

4. **Compare scene boundary files** (if cached):
   ```bash
   # Check if .whisperjav_cache has scene boundary data
   ls .whisperjav_cache/HODV-22019*/
   ```

### Expected Outcomes

**If v1.7.3 is CORRECT**:
- Opening 0:00-9:35 should be music/credits with no speech
- v1.7.1 hallucinations at 2:28-5:53 are false positives
- No action needed (this is improvement)

**If v1.7.3 is WRONG**:
- Real speech exists at 5:53-9:35
- Scene detection or VAD threshold is too aggressive
- Need to tune parameters or revert changes

---

## P2-HIGH: Very Short Segment Bias (<1s)

### Symptoms
- Very short segments (<1s): 33.8% missing rate
- Long segments (≥5s): 13.0% missing rate
- 2.6x difference suggests duration-based filtering
- BUT: Short utterances (うん, ああ, はい) only have 16% missing rate

### Hypothesis
Either:
1. ASR is producing fewer short segments (merging into longer ones)
2. Post-ASR filtering is removing short segments
3. VAD is merging brief utterances

### Code Areas to Investigate

#### 1. Subtitle Sanitizer (`whisperjav/modules/subtitle_sanitizer.py`)
```bash
git diff v1.7.1 v1.7.3 -- whisperjav/modules/subtitle_sanitizer.py
```

**Look for**:
- New minimum duration threshold
- Changes to repetition removal that might affect short segments
- Changes to hallucination removal that might catch short segments

**Search for patterns**:
```python
# Look for duration-based filtering
grep -n "duration" whisperjav/modules/subtitle_sanitizer.py
grep -n "min.*length" whisperjav/modules/subtitle_sanitizer.py
grep -n "<.*1" whisperjav/modules/subtitle_sanitizer.py  # less than 1 second
```

#### 2. Stable-TS ASR (`whisperjav/modules/stable_ts_asr.py`)
```bash
git diff v1.7.1 v1.7.3 -- whisperjav/modules/stable_ts_asr.py
```

**Look for**:
- Changes to `regroup` parameters
- Changes to `min_word_dur` or similar
- Changes to Japanese regrouping rules that might merge short segments

#### 3. SRT Post-processing (`whisperjav/modules/srt_postprocessing.py`)
```bash
git diff v1.7.1 v1.7.3 -- whisperjav/modules/srt_postprocessing.py
```

**Look for**:
- New filtering logic for short subtitles
- Changes to timing adjustments
- Gap filling that might merge short segments

### Diagnostic Test

Create a test file with known short utterances:

```python
# test_short_segments.py
from whisperjav.modules.subtitle_sanitizer import sanitize_subtitles

# Sample SRT with very short segments
test_srt = """1
00:00:01,000 --> 00:00:01,500
うん

2
00:00:02,000 --> 00:00:02,800
ああ、そうね

3
00:00:03,500 --> 00:00:04,000
はい

4
00:00:05,000 --> 00:00:09,000
これは長い文章です。とても長い文章なので消されないはずです。
"""

# Run through sanitizer
result = sanitize_subtitles(test_srt, ...)
print(f"Input: 4 segments")
print(f"Output: {count_segments(result)} segments")
```

### Expected Outcomes

**If filtering is the cause**:
- Find explicit duration threshold in sanitizer or post-processor
- Revert or adjust threshold

**If ASR merging is the cause**:
- Verify with stable-ts regrouping parameters
- Check if this is actually better (fewer spurious breaks)

**If VAD merging is the cause**:
- Check VAD parameters for minimum speech duration
- May be related to P1 investigation

---

## P3-MEDIUM: Segment 10 Loss (Ending Credits)

### Symptoms
- 67.2% loss rate in final segment (123:27-137:10)
- Similar to Segment 1, suggests end-of-video handling

### Investigation
Same as P1, but for ending segment:
1. Extract final 15 minutes
2. Check for outro music, credits, end cards
3. Determine if v1.7.3 is correctly skipping non-speech content

**Quick Check**:
```bash
ffmpeg -i HODV-22019.mp4 -ss 02:03:27 -t 00:15:00 -vn -acodec pcm_s16le ending_15min.wav
```

---

## P3-MEDIUM: Middle Segment Scattered Losses

### Symptoms
- Remaining ~65 subtitles (50.8%) scattered across middle segments
- No clear pattern in time, duration, or content
- May be combination of multiple factors

### Investigation Strategy

#### Step 1: Categorize the 65 missing subtitles
```python
# Create script to analyze the "other" category
# Filter out hallucinations, Segment 1, Segment 10, very short (<1s)
# See if there's a hidden pattern
```

#### Step 2: Sample 10-20 missing subtitles manually
- Listen to audio at those timestamps
- Verify if they should be transcribed
- Check for environmental noise, overlapping speech, etc.

#### Step 3: Compare ASR confidence scores (if available)
```python
# If stable-ts provides confidence scores
# Check if missing subtitles have lower confidence
# May indicate ASR filtering by confidence threshold
```

---

## Code Change Overview (What to `git diff`)

### Critical Files
1. `whisperjav/modules/scene_detection.py` - Scene boundary detection
2. `whisperjav/modules/audio_preprocessing.py` - VAD and audio cleanup
3. `whisperjav/modules/subtitle_sanitizer.py` - Hallucination and repetition removal
4. `whisperjav/modules/stable_ts_asr.py` - ASR and regrouping
5. `whisperjav/modules/srt_postprocessing.py` - Final subtitle adjustments

### Configuration Files
1. `whisperjav/config/transcription_tuner.py` - Sensitivity profiles
2. `whisperjav/config/v4/ecosystems/*/` - YAML configs (if applicable)
3. `whisperjav/config/sanitization_config.py` - Hallucination patterns

### Pipeline Files
1. `whisperjav/pipelines/fast_pipeline.py` - Pipeline using scene detection
2. `whisperjav/pipelines/balanced_pipeline.py` - Full pipeline with VAD

---

## Test Plan

### Phase 1: Quick Validation (30 minutes)
1. Run visual summary script on another test video
2. Check if pattern persists (95% loss in Segment 1, etc.)
3. If pattern is unique to HODV-22019, issue may be content-specific

### Phase 2: Code Comparison (1-2 hours)
1. `git diff v1.7.1 v1.7.3` on critical files (above)
2. Identify all parameter changes
3. Document in spreadsheet:
   - File
   - Parameter name
   - Old value
   - New value
   - Likely impact

### Phase 3: Manual Verification (2-3 hours)
1. Extract opening segment (0:00-15:00)
2. Listen manually to identify true speech regions
3. Run both v1.7.1 and v1.7.3 on isolated segment
4. Compare outputs to ground truth

### Phase 4: Targeted Testing (1-2 hours)
1. Create minimal test cases:
   - Audio file with only short utterances (<1s)
   - Audio file with intro music then speech
   - Audio file with clean continuous speech
2. Run both versions
3. Identify which scenarios trigger differences

---

## Decision Tree

```
Start
  |
  ├─> Is Segment 1 loss acceptable? (mostly hallucinations)
  |     YES: Move to P2 (short segment bias)
  |     NO:  ──> Investigate scene detection/VAD
  |                 |
  |                 ├─> Found parameter change?
  |                 |     YES: Test reverting change
  |                 |     NO:  Continue to code review
  |                 |
  |                 └─> Manual verification shows real speech missing?
  |                       YES: Revert changes or adjust thresholds
  |                       NO:  Mark as improvement, move to P2
  |
  ├─> Is short segment bias a problem? (33.8% loss)
  |     NO:  May be better segmentation, validate with users
  |     YES: ──> Investigate subtitle sanitizer/ASR regrouping
  |                 |
  |                 └─> Found duration threshold?
  |                       YES: Adjust or make configurable
  |                       NO:  Investigate ASR parameters
  |
  └─> Run comprehensive test suite
        ├─> Clean speech sample: Is loss <5%?
        |     YES: v1.7.3 is superior (better halluc removal)
        |     NO:  Real speech is being lost, needs fix
        |
        └─> Hallucination-heavy sample: Is reduction >20%?
              YES: v1.7.3 is superior (cleaner output)
              NO:  May have regressed on some content
```

---

## Success Criteria

### v1.7.3 is ACCEPTABLE if:
1. ✓ Segment 1 loss is due to correct skipping of intro music/silence
2. ✓ Short segment reduction is due to better merging (fewer spurious breaks)
3. ✓ Clean speech samples show <10% subtitle reduction
4. ✓ Hallucination-heavy samples show >30% hallucination reduction

### v1.7.3 needs FIXES if:
1. ✗ Real speech at 5:53-9:35 is being incorrectly skipped
2. ✗ Legitimate short utterances (うん, ああ) are being filtered
3. ✗ Clean speech samples show >15% subtitle reduction
4. ✗ Users report missing dialogue in production use

---

## Next Steps

1. **Immediate** (Today):
   - Run `git diff v1.7.1 v1.7.3` on critical files
   - Extract and listen to opening 15 minutes of HODV-22019
   - Document parameter changes in spreadsheet

2. **Short-term** (This Week):
   - Complete manual verification of Segment 1
   - Test short segment hypothesis with synthetic data
   - Run analysis on 2-3 additional test videos

3. **Medium-term** (Next Sprint):
   - Create comprehensive regression test suite
   - Add configuration options for discovered thresholds
   - Document expected behavior for different content types

---

**End of Roadmap**

For forensic analysis details, see: `scripts/FORENSIC_ANALYSIS_REPORT.md`
