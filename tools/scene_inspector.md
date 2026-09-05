# Scene Inspector

## Purpose

Runs WhisperJAV's scene-detection backends on a media file, the way the pipelines
do, and reports what they found: a per-scene table, aggregate statistics for
assessing the split, loudness and speech content per scene, how an existing
subtitle file lands on the scenes, three screenshots per scene plus a contact
sheet, FFmetadata chapters, and a side-by-side comparison when several backends
are run. Nothing is transcribed and no WhisperJAV configuration is changed.

Use it to judge a detector or a parameter change before spending a full
transcription run on it, to explain a subtitle gap ("was that stretch inside a
scene at all, and was there speech in it?"), or to gather scene statistics across
a library.

## Script

`tools/scene_inspector.py` (owner request 2026-09-05: P1–P7, then suggestions 1–7 and
the semantic threshold flag).

## Invocation

```bash
python tools/scene_inspector.py MOVIE.mp4
python tools/scene_inspector.py MOVIE.mp4 --backend semantic --scene-threshold 10
python tools/scene_inspector.py MOVIE.mp4 --backend auditok,semantic --speech-ratio --srt MOVIE.ja.whisperjav.srt
python tools/scene_inspector.py MOVIE.mp4 --sensitivity aggressive --no-screenshots
python tools/scene_inspector.py --list-params auditok
```

### Arguments

| Flag | Default | Purpose |
|------|---------|---------|
| `MEDIA ...` | required | One or more video or audio files |
| `--backend NAME[,NAME...]` | `auditok` | `auditok`, `silero`, `semantic`, `none`. Several names run each and add a comparison report. (`auditok` is the legacy pipelines' default; the ChronosJAV pipelines use `semantic`) |
| `--sensitivity` | none | `conservative`, `balanced` or `aggressive`: applies **only the preset's own keys** from the backend's tool YAML on top of the backend defaults (see the note below) |
| `--scene-threshold FLOAT` | 18 (YAML) | Semantic detector's clustering distance; lower tends to give more, shorter scenes. Shortcut for `--param clustering_threshold=FLOAT`; warns if the semantic backend is not selected |
| `--param KEY=VALUE` | none | Override any backend parameter; repeatable. Without it the backend runs on its own constructor defaults (P7). Unknown keys are reported and ignored |
| `--speech-ratio [SEGMENTER]` | off (`firered-vad` when given) | Run a WhisperJAV speech segmenter on every scene's audio and report speech seconds and ratio per scene; `ten`, `silero-v3.1`, `whisperseg` … also work |
| `--srt FILE` | none | Count the cues of a subtitle file per scene (a cue belongs to the scene holding its midpoint); reports scenes without cues and cues that fell into gaps |
| `--output-dir DIR` | `<media folder>/scenes_info` | Target folder (P2). With several inputs and no `--output-dir`, each file gets the folder next to it |
| `--no-screenshots` | off | Skip frame extraction and the contact sheet (P6) |
| `--image-format` | `jpg` | `jpg` or `png` |
| `--keep-scene-audio` | off | Also keep the per-scene WAV files the detector writes, in `<target>/<name>.<backend>_scene_audio/` |
| `--list-params BACKEND` | | Print the parameters a backend accepts, with their YAML defaults and named presets, then exit |
| `--quiet` | off | Warnings only on the console (the semantic engine still prints its own progress) |

## Outputs (in the target folder, one set per backend B)

| File | Content |
|------|---------|
| `<name>.B.scenes.csv` | One row per scene: index, start/end as `hh:mm:ss.mmm` and seconds, duration, gap before, detection pass, RMS and peak dBFS, speech seconds and ratio (`--speech-ratio`), cues and cues per minute (`--srt`), the three screenshot paths, backend metadata as JSON |
| `<name>.B.scenes.json` | Everything: media facts (audio duration, container duration, video geometry, fps, size), backend used with the effective parameters and the sensitivity applied, the statistics block, every scene, file names |
| `<name>.B.summary.md` | The statistics for reading: count, coverage (union of scene intervals; overlap between padded scenes reported separately), duration percentiles and histogram, scenes over the 29 s Whisper window and the 180 s aligner limit, gaps, loudness range with quietest/loudest scene, speech totals and scenes without speech, subtitle totals and scenes without cues, then the scene table |
| `<name>.B.chapters.ffmeta` | FFmetadata chapters, one per scene, titled `Scene 0001 hh:mm:ss.mmm-hh:mm:ss.mmm`. Mux: `ffmpeg -i MOVIE.mp4 -i <name>.B.chapters.ffmeta -map_metadata 1 -codec copy MOVIE.chapters.mkv` |
| `<name>.B.contact_sheet_pNN.jpg` | The begin frame of every scene, 5 per row, labelled `#index start (duration)`, 60 tiles per page |
| `screenshots/B/` | `<name>__scene0001__begin__00_12_03.450.jpg`, `...__middle__...`, `...__end__...` per scene. The end frame is taken 0.1 s before the cut. Skipped, with the reason recorded in the JSON, when the input has no video stream |
| `<name>.compare.md` / `.json` | Only with several backends: per backend the scene count, coverage, median/p90/max duration, scenes over 29 s, gaps over 1 s, detection time, and a pairwise boundary-agreement matrix (share of the row's cuts that the column also has within 1 s) |

Statistics use the **extracted audio duration**, which is what the detector saw; the
container's own duration is reported next to it because re-muxed clips can carry a
stale header (the 293 s Netflix clip in `test_media/` reports 30:55 in its header).

## Note on `--sensitivity` and the tool YAML files

The tool YAML `spec` blocks are **not** the backends' constructor defaults. For
auditok: `pass2_max_duration_s` 1800 (spec) vs 28 s (constructor), `pass2_max_silence_s`
1.8 vs 0.94, `pass1_max_silence_s` 2.5 vs 1.8; the pipelines pass a third set from the
Pydantic component preset. Applying the whole spec on the 293 s clip produced one 222 s
scene where the defaults give a 28 s maximum. `--sensitivity` therefore applies only the
keys listed under `presets.<name>` (auditok aggressive: energy thresholds 28/32, max
silence 2.0/1.2 s). This discrepancy is reported to the owner; the tool does not change
any product file.

## What was verified (2026-09-05, WJ env, RTX 3060)

- `293sec-S01E04-scene4.mkv`, `--backend auditok,semantic --speech-ratio --srt …ja.whisperjav.srt`: auditok 19 scenes / semantic 7; 57 + 21 frames; two contact sheets (one opened and checked); two chapter files, one muxed with ffmpeg and read back by ffprobe; speech 65.6 % / 66.6 % of the media; 7 auditok scenes without cues, 1 cue in a gap; `compare.md` with boundary agreement auditok→semantic 0.161, semantic→auditok 0.429.
- `--sensitivity conservative|aggressive` (auditok, audio clip): only the preset keys change (thresholds 40/45 and 28/32, silences 3.0/2.5 and 2.0/1.2); max scene stays 28 s.
- `--backend semantic --scene-threshold 10`: 6 scenes; effective `clustering_threshold: 10.0`; with `--backend auditok` the flag warns.
- Audio-only `015sec_test…wav` with `--speech-ratio ten` and `--srt`: 1 scene, screenshots skipped with `skipped_reason: "no video stream"`, speech 68.8 %.
- Semantic coverage is 100.0 % with 1.288 s of overlap reported (was 100.4 % before the union fix).
- `--list-params` for auditok/semantic; `--help` exit 0.

## Notes

- Frame extraction is one `ffmpeg -ss T -i media -frames:v 1` call per frame, about 0.2 s each here; a 400-scene film means 1,200 frames, roughly four minutes.
- `--speech-ratio` runs the segmenter once per scene on the scene WAV; FireRedVAD on the 293 s clip added a few seconds.
- The `none` backend returns the whole file as one scene; useful only as a baseline.
