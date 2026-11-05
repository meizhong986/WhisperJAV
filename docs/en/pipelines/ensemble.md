# Ensemble Mode

Ensemble mode runs one or two pipelines on the same video, one after the other. With Pass 2 ticked, it merges the two passes' subtitles into one file by the merge strategy you choose.

---

## How It Works

```
Video → Pass 1 (Pipeline A) → SRT₁ ─┐
                                      ├→ Merge → Final SRT
Video → Pass 2 (Pipeline B) → SRT₂ ─┘
```

Each pass runs its own pipeline on the whole video and writes its own subtitle file; the merge then builds the final subtitle file from the two.

---

## Setting Up Ensemble

1. Open the **Ensemble Mode** tab (the third tab)
2. **Pass 1** is always active — configure pipeline, sensitivity, and options
3. **Enable Pass 2** by checking the checkbox
4. Configure Pass 2 with a different pipeline
5. Choose a **Merge Strategy**
6. Click **Start**

### Pass Configuration

Each pass has identical controls:

| Control | Description |
|---------|-------------|
| **Pipeline** | The pipeline of the pass, in five groups: Whisper-Based (Balanced, Fast, Faster, Fidelity), HuggingFace (Transformers (HF)), ChronosJAV (Qwen3-ASR, Anime-Whisper), External (BYOP) (XXL Faster Whisper) and GGUF via crispasr (parakeet, cohere, reazonspeech, kotoba, whisper-turbo, whisper-large, moss). Pass 1 starts on cohere, Pass 2 on parakeet. Choosing a pipeline sets the sensitivity and speech segmenter the pass starts with |
| **Sensitivity** | Detection threshold (Aggressive, Balanced, Conservative) |
| **Scene Detector** | How to split audio into scenes (Auditok, Silero, Semantic, None) |
| **Speech Enhancer** | Audio preprocessing (None, FFmpeg DSP, ZipEnhancer, htdemucs) |
| **Speech Segmenter** | Finds the speech inside each scene: Silero v4.0, Silero v3.1, Silero v6.2, TEN VAD, WhisperSeg (JA-ASMR), FireRedVAD or None. For Balanced the menu lists instead the Silero version its built-in VAD runs (3.1, 4.0 (default), 6.2). For a GGUF pass, None sends each scene to the model whole |
| **Model** | Which model to use (pipeline-dependent) |

Click **Customize Parameters** on a pass to change its detailed settings; the button is greyed out for XXL Faster Whisper.

---

## Merge Strategies

| Strategy (GUI) | Command line | What it does |
|----------|----------|----------|
| **Pass 1 + Fill From Pass 2** (default) | `pass1_primary` | Keeps every Pass 1 subtitle; adds a Pass 2 subtitle only where it overlaps no Pass 1 subtitle |
| **Smart Merge (Best Quality)** | `smart_merge` | Pairs each Pass 1 subtitle with the Pass 2 subtitle that overlaps it most, if the overlap is at least 30% of the shorter one, and keeps one of the pair: the one with more of its own length inside the overlap; if that is about equal, the one with more text (a line of few different characters, such as ああああああああああ, counts as none); then the shorter one. Subtitles without a partner are kept from both passes |
| **Full Merge (All Lines)** | `full_merge` | Keeps every subtitle of both passes, in time order; overlapping subtitles stay |
| **Pass 2 + Fill From Pass 1** | `pass2_primary` | As Pass 1 + Fill From Pass 2, with the passes swapped |
| **Pass 1 + Fill (30% Overlap)** | `pass1_overlap` | Keeps every Pass 1 subtitle; adds a Pass 2 subtitle unless it overlaps a Pass 1 subtitle by more than 30% of that subtitle's length |
| **Pass 2 + Fill (30% Overlap)** | `pass2_overlap` | As above, with the passes swapped |
| — (command line only) | `longest` | Pairs subtitles as Smart Merge does and keeps the one with more text |

With every strategy, a subtitle whose text is the same as the one just before it is then removed.

---

## BYOP: XXL Faster Whisper (v1.8.9+; Pass 1 since 1.10.0)

Select **XXL Faster Whisper** as the pipeline of Pass 1, Pass 2 or both to use [PurfView's Faster Whisper XXL](https://github.com/Purfview/whisper-standalone-win) as an external program. The menu lists it under **External (BYOP)**: XXL is a separate program that you download yourself; WhisperJAV runs it.

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

## Serial vs Batch Mode

When processing multiple files in ensemble mode:

| Mode | Behavior |
|------|----------|
| **Batch** | All Pass 1 jobs run first, then all Pass 2, then all merges; each pass starts once for all files |
| **Serial** (default) | Each file completes fully (Pass 1 → Pass 2 → Merge) before the next starts |

**Serial mode** is the default since 1.10.1, in the GUI and on the command line: a file's subtitles are ready as soon as it is done, and a crash in one file does not lose the others. In the GUI the **Finish each file** box (ticked by default) turns it on; untick it, or use `--no-ensemble-serial` on the command line, for batch mode.

---

## Presets

A preset holds the settings of one pass: its pipeline, sensitivity, scene detector, speech enhancer, speech segmenter, model and the values of its **Customize Parameters** window. The merge strategy and the other pass are not part of it.

1. Click **Customize Parameters** on the pass and set its values
2. Click **Save Preset** at the bottom of the window and give it a name
3. Later, choose it in the **Load Preset...** menu of the same window, for either pass

Presets are kept for the next start.

---

## Inline Translation

Check **"AI-translate"** after the merge strategy to automatically translate the merged output. Select your provider and model inline, or click the settings button for full configuration.
