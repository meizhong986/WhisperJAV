# Common Workflows

Step-by-step recipes for the most common tasks.

---

## Quick Transcription (Simplest)

The fastest path from video to subtitles.

1. Launch WhisperJAV GUI
2. **Drag** a video file onto the app
3. Leave all defaults (Balanced mode, Aggressive sensitivity, Japanese)
4. Click **Start**
5. SRT file appears next to your video

**Time:** ~2-5 minutes for a 2-hour video with a modern GPU.

---

## High-Quality Transcription (Ensemble)

Uses two different ASR engines and merges their output for best accuracy.

1. Go to the **Ensemble** tab
2. **Pass 1:** Balanced pipeline (default)
3. **Enable Pass 2:** Check the Pass 2 checkbox
4. **Pass 2:** Select **Qwen3-ASR**
5. **Merge Strategy:** Smart Merge
6. Click **Start**

Both passes run, then results are intelligently combined. Takes roughly 2x the time of a single pass.

---

## ChronosJAV Pipeline (Anime/JAV Content)

Dedicated pipeline using models trained on anime and JAV dialogue.

1. Go to the **Ensemble** tab
2. **Pass 1 Pipeline:** Select **ChronosJAV**
3. **Model:** Choose from:
    - **anime-whisper** (~4GB) — best quality for anime/JAV
    - **Kotoba v2.1** (~2GB) — lighter, with punctuation
    - **Kotoba v2.0** (~2GB) — lighter, no punctuation
4. Click **Start**

!!! tip
    For maximum quality, use anime-whisper in Pass 1 and Qwen3-ASR in Pass 2 with Smart Merge.

---

## Transcribe + Translate in One Step

Get translated subtitles without a separate step.

1. Configure your transcription (any tab)
2. On the **Ensemble** tab, check **"AI-translate"**
3. Select your translation **provider** (DeepSeek, Gemini, etc.)
4. Enter your **API key** if needed
5. Click **Start**

Transcription runs first, then translation happens automatically on the result.

---

## Translate an Existing SRT File

Use Tab 4 to translate subtitles you already have.

1. Go to the **AI SRT Translate** tab (Tab 4)
2. Click **Add File(s)** and select your `.srt` file
3. Select a **Provider** and **Model**
4. Enter your **API key** and click **Test Connection**
5. Set **Target Language** (e.g., English)
6. Choose **Tone**: Standard or Adult-Explicit
7. Click **Start**

!!! note
    The Adult-Explicit tone uses specialized instructions tuned for JAV dialogue with appropriate vocabulary.

---

## Batch Processing (Multiple Files)

Process an entire folder of videos at once.

1. Click **Add Folder** and select a folder containing videos
2. All media files are added to the list
3. Configure your pipeline settings
4. Click **Start**

Files are processed sequentially. Each output SRT is saved next to its source video (or to your chosen output directory).

!!! tip "Serial Ensemble Mode"
    In Ensemble mode, enable **Serial** mode to complete each file fully (Pass 1 → Pass 2 → Merge) before starting the next. This lets you see results as they finish instead of waiting for the entire batch.

---

## CPU-Only Mode (No GPU)

WhisperJAV works without a GPU, just slower.

1. Go to the **Advanced** tab
2. Check **"Accept CPU-only mode"**
3. Use **Faster** mode for the best speed without GPU
4. Click **Start**

!!! warning
    CPU mode is 5-10x slower than GPU mode. A 2-hour video may take 30-60 minutes.

---

## WebVTT Output

Generate VTT subtitles for HTML5 video players.

1. Go to the **Advanced** tab
2. Set **Output Format** to **VTT** or **Both** (SRT + VTT)
3. Run your transcription as normal

The `.vtt` file is saved alongside the `.srt` file.
