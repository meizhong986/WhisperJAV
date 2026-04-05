# Forensic CSV Generator

## Purpose

Produces a detailed forensic CSV comparing every ground truth subtitle against a WhisperJAV pass output. Each ground-truth subtitle becomes one CSV row with full pipeline trace metadata: scene coverage, VAD coverage, raw model output, post-model filter status, sanitizer status, and a mechanically-determined loss stage.

Used during v1.8.10-hf development to analyze where each ground-truth subtitle was lost in the pipeline (scene detection → VAD → model → internal filter → sanitizer → final SRT).

## Script

`tools/forensic_csv_generator.py`

## Invocation

```bash
python tools/forensic_csv_generator.py --pass-number 1 --output <output.csv> --base-dir <base_dir>
python tools/forensic_csv_generator.py --pass-number 2 --output <output.csv> --base-dir <base_dir>
```

### Arguments

| Flag | Required | Values | Default | Purpose |
|------|----------|--------|---------|---------|
| `--pass-number` | Yes | 1 or 2 | — | Which ensemble pass to analyze |
| `--output` | Yes | CSV filename | — | Output CSV path (relative paths resolve against `--base-dir`) |
| `--base-dir` | No | Directory path | `F:\MEDIA_DLNA\SONE-853\S0104\R1` | Root directory containing test artifacts |

## Required Input Files

The script reads these files from `<base_dir>`. If a file is missing, corresponding CSV columns will be empty; the script continues.

| Path (relative to base_dir) | Purpose |
|-----------------------------|---------|
| `Ground-Truth-Netflix-Reference-Subs.srt` | Ground truth reference SRT |
| `293sec-S01E04-scene4.ja.pass{N}.srt` | Final pass output (post-sanitization) |
| `TEMP/pass{N}_worker/293sec-S01E04-scene4_stitched.srt` | Stitched SRT (pre-sanitization) |
| `TEMP/pass{N}_worker/raw_subs/293sec-S01E04-scene4_stitched.artifacts.srt` | Sanitizer removal log |
| `TEMP/pass{N}_worker/293sec-S01E04-scene4_master.json` | Pipeline metadata, scene list, quality metrics |
| `TEMP/pass{N}_worker/scenes/293sec-S01E04-scene4_semantic.json` | Semantic scene metadata (if semantic detector used) |
| `TEMP/pass{N}_worker/scene_srts/293sec-S01E04-scene4_scene_NNNN.transcribe.json` | Per-scene unfiltered model output |

Where `{N}` is the pass number, `NNNN` is 4-digit scene index.

## Hardcoded Constants

These are set in the script source (adjust if using different test media):

```python
BASENAME = "293sec-S01E04-scene4"
GT_FILENAME = "Ground-Truth-Netflix-Reference-Subs.srt"
MATCH_TOLERANCE_SEC = 3.0
```

## Output Format

UTF-8 BOM encoding (`utf-8-sig`), `QUOTE_ALL` CSV quoting, one row per ground truth subtitle.

### Columns (25 total)

| # | Column | Source / Computation |
|---|--------|---------------------|
| 1 | `GT_Sub_Number` | Ground truth SRT sequence number |
| 2 | `GT_Start_Time` | GT start (HH:MM:SS,mmm) |
| 3 | `GT_End_Time` | GT end (HH:MM:SS,mmm) |
| 4 | `GT_Text` | Ground truth subtitle text |
| 5 | `Has_Corresponding_Sub` | YES/NO: final SRT has subtitle within ±3.0s of GT start |
| 6 | `Pass_Final_Text` | Text of matched final subtitle (empty if no match) |
| 7 | `Scene_Number` | Scene index containing GT timestamp |
| 8 | `Scene_Start` | Scene start seconds (.2f) |
| 9 | `Scene_End` | Scene end seconds (.2f) |
| 10 | `Scene_Duration` | Scene duration seconds (.2f) |
| 11 | `Scene_No_Speech_Flag` | TRUE/FALSE from master JSON |
| 12 | `Semantic_Label` | From semantic scene detector (e.g. "ambient_noise", "silence") |
| 13 | `Semantic_Confidence` | Semantic classification confidence |
| 14 | `Semantic_Loudness_dB` | Scene loudness in dB |
| 15 | `VAD_Group_Number` | Index of VAD group containing timestamp |
| 16 | `Model_No_Speech_Prob` | no_speech_prob of matched segment (.4f) |
| 17 | `Model_Avg_Logprob` | avg_logprob of matched segment (.4f) |
| 18 | `Model_Compression_Ratio` | compression_ratio of matched segment (.4f) |
| 19 | `Model_Temperature` | Sampling temperature used |
| 20 | `Raw_Model_Text` | Text from transcribe.json (unfiltered output) |
| 21 | `In_Raw_SRT` | YES/NO: pre-sanitization SRT has match |
| 22 | `Raw_SRT_Text` | Matched raw SRT subtitle text |
| 23 | `Sanitizer_Removed` | YES/NO: artifacts SRT has removal near timestamp |
| 24 | `Sanitizer_Reason` | Sanitizer removal reason string |
| 25 | `Loss_Stage` | Classification label (see below) |

## Loss_Stage Labels

Determined by deterministic decision tree in `classify_loss_stage()`:

| Label | Condition |
|-------|-----------|
| `Captured` | Final subtitle exists within 3.0s of GT start |
| `Skipped_by_scenedetect` | No scene covers GT timestamp |
| `Skipped_by_VAD` | Scene exists but no VAD group covers timestamp, or scene has `no_speech_detected=True` with no model output at this location |
| `Dropped_internal_threshold` | VAD group exists, model produced text (in transcribe.json), but text does NOT appear in raw SRT (filtered by ASR module's post-model gate) |
| `Empty_transcriber_results` | VAD group exists but model returned no segments (empty in transcribe.json) |
| `Removed_by_sanitization` | Text appeared in raw SRT but was removed by sanitizer |
| `Missed_Unknown_Cause` | None of the above apply |

## Match Tolerance

`MATCH_TOLERANCE_SEC = 3.0` is a coarse temporal match based on start-time proximity only. It does NOT verify text similarity.

**Known limitation**: For dense dialogue with closely-spaced subtitles (<2s apart), the tolerance may match a GT entry to a wrong adjacent subtitle. Manual review of `GT_Text` vs `Pass_Final_Text` columns is required to identify false matches.

## Console Output

```
Generating forensic CSV for Pass <N>...
  Base directory: <path>
  Basename: <basename>
  Ground truth: N subtitles
  Pass <N> final SRT: N subtitles
  Pass <N> raw/stitched SRT: N subtitles
  Artifacts: N entries
  Scenes: N
  Semantic data: YES/NO
  VAD params: threshold=X, min_speech=Xms
  Logprob threshold: X
  No speech threshold: X

  Written: <output path> (N rows)

  Loss Stage Distribution:
    <Label>: N (X.X%)
    ...
```

## Example Usage (runs R1-R7)

```bash
cd F:\MEDIA_DLNA\SONE-853\S0104\R1\TEMP
python forensic_csv_generator.py --pass-number 1 --output S0104_R5_PASS1_FORENSIC.csv --base-dir F:\MEDIA_DLNA\SONE-853\S0104\R5
python forensic_csv_generator.py --pass-number 2 --output S0104_R5_PASS2_FORENSIC.csv --base-dir F:\MEDIA_DLNA\SONE-853\S0104\R5
```

## Adapting to Different Test Media

To use this script with a different basename / ground truth file, modify the top-level constants:

```python
BASENAME = "<your_basename>"
GT_FILENAME = "<your_ground_truth.srt>"
```

Or parameterize these via command-line flags as an enhancement.
