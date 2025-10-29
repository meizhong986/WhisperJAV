## AI Coding Guide for WhisperJAV
- Stay inside existing patterns, keep edits narrow, prefer pathlib/logging helpers already in use.

### Architecture Landmarks
- `whisperjav/main.py` is the only entry that orchestrates pipelines, async processing, and optional translation handoff.
- Pipelines live in `whisperjav/pipelines/`; balanced mode funnels through `WhisperProASR`, fast/faster rely on `StableTSASR`.
- Translation lives under `whisperjav/translate/` with `cli.py` as the `whisperjav-translate` console script.
- Subtitle artifacts land in `output/`; raw scene transcripts copy to `output/raw_subs` after stitching.

### Translation Module (`whisperjav/translate`)
- CLI resolves config via `settings.resolve_config`; precedence is CLI > env vars (API key only) > `%AppData%/WhisperJAV/translate/settings.json` > defaults.
- `core.translate_subtitle` wraps PySubtrans: init options/provider/project, optionally injects instructions, then saves translated SRT.
- Provider metadata lives in `providers.py`; env var names and default models come from there—add new providers here first.
- Instructions are fetched with ETag caching (`instructions.py` → Gist → cache dir → bundled fallback in `defaults/`).
- CLI auto-writes fetched instructions to a temp file; explicit `--instructions-file` bypasses the fetch.

### Main CLI ↔ Translate Coupling
- `main.py` shells out to `whisperjav-translate` after transcription when `--translate` is set; stdout returns the translated path.
- Progress noise must remain on stderr so `process_files_sync` can capture stdout; honour `--translate-quiet` to silence progress.
- Translation subprocess inherits scene metadata via CLI flags only; any new context must be serialized into the command invocation.

### Practical Workflow
- Install editable: `pip install -e .` (ensure CUDA Torch pre-installed). Run transcription via `whisperjav <file> --mode balanced`.
- Translate standalone: `whisperjav-translate -i movie.ja.whisperjav.srt --provider deepseek --tone pornify`.
- Configure defaults interactively with `whisperjav-translate --configure`; settings go to `%AppData%\WhisperJAV\translate\settings.json`.
- To refresh cached instructions: delete the `cache/` folder next to settings or call `get_instruction_content(..., refresh=True)`.

### Implementation Conventions
- Use `Path` for filesystem work; avoid raw strings. Respect existing temp dir strategy that writes to `%TEMP%/whisperjav` by default.
- Keep logging via `logging` (translation CLI only configures root level; mains use `utils.logger`). No bare prints except deliberate stdout/stderr messaging already patterned.
- Preserve CLI argument grouping in `translate/cli.py` when adding switches so `--help` stays readable.
- When extending translation providers, ensure PySubtrans `pysubtrans_name` matches installed plugin names and update `SUPPORTED_TARGETS/SOURCES` if new languages appear.

### Gotchas & Debug Tips
- CUDA requirement fires on import in `main.py`; bypass only happens for `--check/--help/--version` arguments.
- `whisperjav-translate` expects API keys via env vars (see `providers.py`); CLI `--api-key` overrides but is not persisted.
- Instructions cache writes UTF-8 files; Windows paths derive from settings location—guard against read-only AppData.
- PySubtrans returns save path from `SaveTranslation`; handle both `str` and `Path` to avoid Windows type mismatches.