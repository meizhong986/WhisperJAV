# Scene Inspector

## Purpose

Runs one of WhisperJAV's scene-detection backends on a media file, the way the
pipelines do, and reports what it found: a per-scene table, aggregate statistics
for assessing the split, and three screenshots per scene (beginning, middle, end)
so a scene can be recognised by eye. Nothing is transcribed and no WhisperJAV
configuration is changed.

Use it to judge a detector or a parameter change before spending a full
transcription run on it, to explain a subtitle gap ("was that stretch inside a
scene at all?"), or to gather scene statistics across a library.

## Script

`tools/scene_inspector.py` (owner request 2026-09-05, items P1-P7).

## Invocation

```bash
python tools/scene_inspector.py MOVIE.mp4
python tools/scene_inspector.py MOVIE.mp4 --backend semantic --param clustering_threshold=10
python tools/scene_inspector.py A.mp4 B.mkv --no-screenshots
python tools/scene_inspector.py --list-params auditok
```

### Arguments

| Flag | Default | Purpose |
|------|---------|---------|
| `MEDIA ...` | required | One or more video or audio files |
| `--backend` | `auditok` | `auditok`, `silero`, `semantic` or `none` (the legacy pipelines' default is `auditok`; the ChronosJAV pipelines use `semantic`) |
| `--param KEY=VALUE` | none | Override one backend parameter; repeatable. Without it the backend runs on its own defaults (P7) |
| `--output-dir DIR` | `<media folder>/scenes_info` | Target folder (P2). With several inputs and no `--output-dir`, each file gets the folder next to it |
| `--no-screenshots` | off | Skip frame extraction (P6) |
| `--image-format` | `jpg` | `jpg` or `png` |
| `--keep-scene-audio` | off | Also keep the per-scene WAV files the detector writes, in `<target>/<name>_scene_audio/` |
| `--list-params BACKEND` | | Print the parameters a backend accepts, with their YAML defaults and named presets, then exit |
| `--quiet` | off | Warnings only on the console |

## Outputs (in the target folder)

| File | Content |
|------|---------|
| `<name>.scenes.csv` | One row per scene: index, start/end as `hh:mm:ss.mmm` and seconds, duration, gap to the previous scene, detection pass, the three screenshot paths, backend metadata as JSON (semantic context, brute-force split marker) |
| `<name>.scenes.json` | Everything: media facts (audio duration, container duration, video geometry, fps, size), backend used and its effective parameters, the statistics block, every scene, screenshot names, coarse boundaries when the backend reports them |
| `<name>.summary.md` | The same statistics for reading: counts, coverage, duration percentiles and histogram, scenes over the 29 s Whisper window and over the 180 s aligner limit, gaps (count, total, longest, leading, trailing), scenes per minute, then the scene table |
| `screenshots/` | `<name>__scene0001__begin__00_12_03.450.jpg`, `...__middle__...`, `...__end__...` per scene. The end frame is taken 0.1 s before the cut so it still belongs to the scene. Skipped, with the reason recorded in the JSON, when the input has no video stream |

Statistics are computed against the **extracted audio duration**, which is what the
detector saw; the container's own duration is reported next to it because
re-muxed clips can carry a stale header (the 293 s Netflix clip in `test_media/`
reports 30:55 in its header).

## What was verified (2026-09-05, WJ env)

- `293sec-S01E04-scene4.mkv`, default backend: 19 scenes, 57 screenshots written, 0 failed, 10.6 s for the frames; CSV/JSON/summary present.
- Same clip, `--backend semantic --param clustering_threshold=10 --no-screenshots`: 6 scenes; the JSON's effective parameters show `clustering_threshold: 10.0`.
- `015sec_test-966-00_01_45-00_01_59.wav` (audio only): 1 scene, screenshots skipped with `skipped_reason: "no video stream"`.
- `--list-params` for all backends; `--help` exit 0.

## Notes

- Frame extraction is one `ffmpeg -ss T -i media -frames:v 1` call per frame, about 0.2 s each here; a 400-scene film means 1,200 frames, roughly four minutes.
- The `semantic` backend prints its own progress lines even with `--quiet`.
- The `none` backend returns the whole file as one scene; useful only as a baseline.
