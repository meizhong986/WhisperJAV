# Ensemble Mode

Ensemble mode runs two different ASR pipelines on the same video and merges their output for higher accuracy. This is WhisperJAV's most powerful transcription mode.

---

## How It Works

```
Video → Pass 1 (Pipeline A) → SRT₁ ─┐
                                      ├→ Merge → Final SRT
Video → Pass 2 (Pipeline B) → SRT₂ ─┘
```

Each pipeline has different strengths. For example:

- **Whisper** (Balanced): excellent timing, good text
- **Qwen3-ASR**: excellent text quality, different timing approach

Merging combines the best of both.

---

## Setting Up Ensemble

1. Go to the **Ensemble** tab (Tab 3)
2. **Pass 1** is always active — configure pipeline, sensitivity, and options
3. **Enable Pass 2** by checking the checkbox
4. Configure Pass 2 with a different pipeline
5. Choose a **Merge Strategy**
6. Click **Start**

### Pass Configuration

Each pass has identical controls:

| Control | Description |
|---------|-------------|
| **Pipeline** | ASR backend (Balanced, Fast, Faster, Qwen3-ASR, ChronosJAV, Parakeet-CrispASR, Cohere-CrispASR, etc.) |
| **Sensitivity** | Detection threshold (Aggressive, Balanced, Conservative) |
| **Scene Detector** | How to split audio into scenes (Auditok, Silero, Semantic, None) |
| **Speech Enhancer** | Audio preprocessing (None, FFmpeg DSP, ZipEnhancer, htdemucs) |
| **Speech Segmenter** | Voice activity detection within scenes (Silero, TEN, None) |
| **Model** | Which model to use (pipeline-dependent) |

Click **Customize** on any pass for fine-grained parameter control.

---

## Merge Strategies

| Strategy | Best For |
|----------|----------|
| **Pass 1 Primary** | When Pass 1 is your trusted baseline — fills gaps from Pass 2 |
| **Smart Merge** | General use — selects the best subtitle from each pass using quality heuristics |
| **Full Merge** | Maximum coverage — combines all subtitles, resolves overlaps |
| **Longest** | Picks the longer (more detailed) subtitle when passes overlap |
| **Pass 2 Primary** | When Pass 2 is your trusted baseline |
| **Overlap 30%** | Conservative merge — requires 30% time overlap before merging |

!!! tip "Recommended Combo"
    **Balanced** (Pass 1) + **Qwen3-ASR** (Pass 2) + **Smart Merge** is a strong default for most content.

---

## BYOP: XXL Faster Whisper (v1.8.9+; Pass 1 since 1.10.0)

Select **XXL Faster Whisper** as the pipeline of Pass 1, Pass 2 or both to use [PurfView's Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) as an external program. This is "Bring Your Own Pipeline" — you supply the executable.

### Setup

1. Download Faster Whisper XXL from the link above
2. In the Ensemble tab, select **XXL Faster Whisper** for Pass 1, Pass 2 or both
3. In that pass's **Model** column, choose the model XXL runs: Large V2 (default), Large V3, Large V3 Turbo or Medium
4. Click **Browse** to point to your `faster-whisper-xxl.exe`
5. Add any extra args (default `--verbose True --standard_asia`). Choose the model in the Model column, not here: a `--model` in Extra Args is not used, and the console says so.

WhisperJAV sends XXL the input file, the output folder, the model, the language and the SRT output format (and `--task translate` when the subtitles are to be in English); you add any other XXL flags in the Extra Args field. If XXL fails for a file, the console says which pass failed and why; a failed Pass 1 marks the file failed, a failed Pass 2 leaves the file with Pass 1's subtitles, if Pass 1 made them.

### CLI

```bash
# XXL as Pass 1, Qwen3-ASR as Pass 2
whisperjav video.mp4 --ensemble --pass1-pipeline xxl --pass1-model large-v2 --pass2-pipeline qwen --xxl-exe /path/to/faster-whisper-xxl.exe

# XXL as Pass 2
whisperjav video.mp4 --ensemble --pass1-pipeline fidelity --pass2-pipeline xxl --pass2-model large-v2 --xxl-exe /path/to/faster-whisper-xxl.exe
```

`--pass1-model` / `--pass2-model` pick the model XXL runs (default `large-v2`). The extra args are the ones saved in the GUI's Extra Args field; the command line has no option for them.

---

## Serial vs Parallel Batch Mode

When processing multiple files in ensemble mode:

| Mode | Behavior |
|------|----------|
| **Parallel** (default) | All Pass 1 jobs run first, then all Pass 2, then all merges |
| **Serial** | Each file completes fully (Pass 1 → Pass 2 → Merge) before the next starts |

**Serial mode** is useful when you want to see results as they finish. Enable it with the **Serial** checkbox in the GUI or `--ensemble-serial` in CLI.

---

## Presets

Save your ensemble configuration for reuse:

1. Configure your passes, merge strategy, and parameters
2. Click **Save Preset**
3. Give it a name (e.g., "High Quality JAV", "Quick Anime")
4. Load presets later from the preset dropdown

Presets save all pass configurations, merge strategy, and custom parameters. They persist across sessions.

---

## Inline Translation

Check **"AI-translate"** after the merge strategy to automatically translate the merged output. Select your provider and model inline, or click the settings button for full configuration.
