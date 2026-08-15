# AGENTS.md — Pipeline / Orchestration area dossier

> Loaded when working under `whisperjav/pipelines/` (and relevant to `ensemble/pass_worker.py`,
> `main.py`). Stores decisions/invariants — not code. Inducted 2026-06-20.
> The contracts this area MUST honor: `docs/architecture/MODULE_CONTRACTS.md` (C1, C3).
> Config behavior: `whisperjav/config/AGENTS.md`. Disciplines: `.claude/agents/_disciplines.md`.

## The contract (do not break without escalation)

`BasePipeline` (`base_pipeline.py:13-53`): implement `get_mode_name() -> str` and
`process(media_info: Dict) -> Dict`. The return dict MUST carry `output_files.final_srt`
(consumed `pass_worker.py:739-741`) + `summary.{final_subtitles_refined,total_processing_time_seconds}`
(`:801-802`). Full detail: MODULE_CONTRACTS C1.

## Two construction paths (the crucial distinction)

- **Legacy** (`balanced/fast/faster/fidelity/kotoba-faster-whisper`): in `LEGACY_PIPELINES`
  (`config/legacy.py:95-140`) → built via `resolve_legacy_pipeline()` →
  `resolve_config_v3()` (Pydantic presets).
- **Dedicated** (`transformers/qwen/crispasr`): **NOT** in `LEGACY_PIPELINES`. Built via a
  dedicated `_build_pipeline` block (`pass_worker.py:1059-1291`) placed **before** the
  `resolve_legacy_pipeline` call (which would `ValueError` on these names), plus a
  `resolved_config = None` guard in `main.py:1790-1807`.

**Registration checklist (any new pipeline):** `PIPELINE_CLASSES` (`pass_worker.py:33-42`)
+ three CLI choice lists (`main.py:165, 195, 225`) + either `LEGACY_PIPELINES` (legacy) OR a
dedicated `_build_pipeline` block + `resolved_config=None` guard (dedicated). Mirror the
`--dump-params` output for parity.

## Invariants / red-lines

- **`resolved_config = None` guards** (`main.py:1790, 1799, 1806`, + ensemble `:1752`) — dedicated
  and ensemble modes skip legacy resolution. Removing a guard silently breaks that mode.
- **CrispASR must bypass `resolve_legacy_pipeline`** (`main.py:1802-1807`) — it's an external
  subprocess provider; WhisperJAV scene/segmenter/enhancer are deliberately NOT wired to it
  (design: `docs/plans/crispasr_v190/08_*` — gitignored). Keep its integration surface minimal.
- **Nuclear-exit / cleanup pattern** (`base_pipeline.py:107-170`): in subprocess workers
  (`WHISPERJAV_SUBPROCESS_WORKER=1`) ASR `cleanup()` IS called during controlled execution to
  trigger the ctranslate2 C++ destructor safely, but `torch.cuda.empty_cache()` is SKIPPED
  (crashes on Windows during shutdown → BrokenProcessPool). Do not "unify" these branches.

## Parameter threading (the call chain to verify)

`main.py` argparse → ensemble config (`pass_worker.py:413-485`) → `_build_pipeline`
(`:1017-1291`, applies GUI overrides + sensitivity preset) → `Pipeline.__init__` → ASR.
Trace with `call-chain-verifier`; cross-check resolved values with `--dump-params`.

## Qwen/anime VAD grouping + padding (verified 2026-06-25, CORRECTED 2026-07-03)

The qwen pipeline owns its segmenter grouping/padding defaults as **scalar attrs**, not via
`segmenter_config`. At `qwen_pipeline.py:752-769` the segmenter kwargs are seeded from
`segmenter_config` then the scalars **overwrite** `max_group_duration_s`, `chunk_threshold_s`,
`start_pad_ms`, `end_pad_ms`. So:
- The four GUI "Customize Parameters" sliders ARE live: Max-Group / Frame-Gap / Start-Pad /
  End-Pad populate the scalars (`pass_worker.py:1278-1297`, `main.py:1236-1253`); the clobber
  then writes the user's value. VAD **Threshold** rides in `segmenter_config` (not clobbered).
- ⚠️ **CORRECTION (2026-07-03):** the 2026-06-25 "all four sliders live" claim was WRONG for
  **anime-whisper**. A second, earlier clobber in the constructor
  (`qwen_pipeline.py`, the `if generator_backend == "anime-whisper"` branch) used to reset
  `segmenter_chunk_threshold=0.5` / `segmenter_max_group_duration=5.0` UNCONDITIONALLY, *after*
  the constructor kwargs were assigned — so anime Frame-Gap/Max-Group (slider, CLI flag, ensemble
  default) were all silently pinned to 0.5/5.0 (the Phase-2 framer inherited it too). **Removed in
  v1.9.0.** Regression guard: `tests/test_anime_whisper.py::TestAnimeSegmenterGrouping`. When
  changing an anime-only override, grep the constructor for a matching `generator_backend ==`
  branch — that branch runs LAST and wins over any kwarg. (Cohere still has its 1.0/6.0 branch
  there; it has no standalone-CLI path to supply the value, and is out of the retune scope.)
- The **sensitivity-preset** gradient for group/chunk is DEAD for qwen (clobbered by the scalar).
  The explicit sliders are the intended control; the YAML group/chunk gradient only affects
  non-qwen whisperseg consumers. v1.9.0 also pins the default VAD **threshold** to 0.25 for
  anime/qwen3 via a `user_segmenter_overrides` injection (`pass_worker.py` ~pre-1193 / `main.py`
  ~post-1169), which likewise makes the sensitivity threshold-gradient inert for those two
  backends (explicit slider still wins; cohere excluded).
- **v1.9.0 JAV defaults** (qwen3 + anime-whisper only; cohere keeps 1.0s/6.0s + 300ms pads):
  `max_group=3.0s`, `chunk_threshold=0.3s` (group only if gap <300ms), symmetric padding
  `start=100ms`/`end=100ms`, `threshold=0.25` (owner "Option B", validated on anime-whisper
  2026-07-03; grouping value re-tuned from the earlier 4.0/0.4 — those never took effect for
  anime due to the constructor clobber above). Defaults live in `qwen_pipeline.py:119-122` +
  anime branches (`pass_worker.py:1246-1249`, `main.py:1272-1275`).
- **Asymmetric padding**: whisperseg gained `start_pad_ms`/`end_pad_ms` (`whisperseg.py`), both
  inherit `speech_pad_ms` when unset. Must be in `factory.py` whisperseg schema or
  `_sanitize_params` strips them. silero/ten/nemo/whisper-vad lack them → stripped for those.
- **GUI ms↔s**: Frame Gap is `chunk_threshold_ms` in the GUI (50ms step), converted to seconds
  at pack time (`api.py` pass1/pass2 blocks). Start/End pad are ms end-to-end (no conversion).

## Tests that pin behavior

`tests/test_config_v4.py`, `tests/test_ensemble_params.py`, `tests/test_qwen_sensitivity.py`,
`tests/test_qwen_pipeline_integration.py`, `tests/test_gui_custom_params_simulation.py`.
(Confirm assertions by reading; some names above are inferred from filenames.)
