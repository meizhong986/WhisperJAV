# AGENTS.md — Pipeline / Orchestration area dossier

> Loaded when working under `whisperjav/pipelines/` (and relevant to `ensemble/pass_worker.py`,
> `main.py`). Stores decisions/invariants — not code. Inducted 2026-06-20.
> The contracts this area MUST honor: `docs/architecture/MODULE_CONTRACTS.md` (C1, C3).
> Config behavior: `whisperjav/config/AGENTS.md`. Disciplines: `.claude/agents/_disciplines.md`.

## The contract (do not break without escalation)

`BasePipeline` (`base_pipeline.py:13-56`): implement `get_mode_name() -> str` and
`process(media_info: Dict) -> Dict`. The return dict MUST carry `output_files.final_srt`
(consumed `pass_worker.py:724-738`) + `summary.{final_subtitles_refined,total_processing_time_seconds}`
(`:786-787`). Optional: `summary.degradations` (any entry makes the file `suspect`) and, since
1.10.0, `summary.notes` (printed in the run summary; changes no state). Full detail: MODULE_CONTRACTS C1.

## Two construction paths (the crucial distinction)

- **Legacy** (`balanced/fast/faster/fidelity`): in `LEGACY_PIPELINES`
  (`config/legacy.py:95-140`) → built via `resolve_legacy_pipeline()` →
  `resolve_config_v3()` (Pydantic presets).
- **Dedicated** (`transformers/qwen`, and since 1.10.0 the Ensemble-only `parakeet-crispasr/cohere-crispasr`):
  **NOT** in `LEGACY_PIPELINES`. Built via a dedicated `_build_pipeline` block (`pass_worker.py`:
  transformers `:1104`, qwen `:1163`, CrispASR `:1437`) placed **before** the
  `resolve_legacy_pipeline` call (`:1487`, which would `ValueError` on these names), plus a
  `resolved_config = None` guard in `main.py` (`--mode transformers/qwen` `:2677, 2683`; `--ensemble` `:2637`).

**Registration checklist (any new pipeline):** `PIPELINE_CLASSES` (`pass_worker.py:38-49`)
+ the CLI choice lists (`main.py:366, 394, 450`; an Ensemble-only pipeline goes in the two pass lists
only, with no `--mode` value) + either `LEGACY_PIPELINES` (legacy) OR a dedicated `_build_pipeline`
block + `resolved_config=None` guard (dedicated) + `ENHANCE_FOR_VAD_IGNORED_BY` (`pass_worker.py:55-57`)
when the class never reads `enhance_for_vad` (`tests/test_speech_enhancer_spec.py` checks it against
the class source). Mirror the `--dump-params` output for parity.

## Invariants / red-lines

- **`resolved_config = None` guards** (`main.py:2671, 2677, 2683`, + ensemble `:2637`) — dedicated
  and ensemble modes skip legacy resolution. Removing a guard silently breaks that mode.
- **The CrispASR pipelines bypass `resolve_legacy_pipeline`** (`pass_worker.py:1437`) —
  `parakeet_crispasr_pipeline.py` and `cohere_crispasr_pipeline.py` (1.10.0) are Ensemble passes only
  (no `--mode` value). Each runs CrispASR's own program once per film, through the shared
  `whisperjav/crispasr/` package (manifest, install, runner, ojf), on cut points made from
  WhisperJAV's semantic scenes and speech segmenter (default whisperseg; "none" is refused). They use
  three pass settings (model short name, speech segmenter, Japanese language;
  `whisperjav/crispasr/pass_settings.py`); no speech enhancer, sensitivity or custom parameters in
  1.10.0. Routine messages go to `summary.notes`; `summary.degradations` (which makes a file
  `suspect`) holds only program output that had to be dropped and a damaged audio track.
  Design: `docs/plans/v1.10.0-crispasr-master-design-plan.md` (gitignored).
- **The 1.9.x "bring your own CrispASR" mode is removed** (`pipelines/crispasr_pipeline.py` and
  `crispasr_runner.py` are deleted): `--mode crispasr`, `--passN-pipeline crispasr` and the
  `--crispasr-*` flags stop the run with exit status 2 before argparse runs (`main.py:41-46`,
  `whisperjav/crispasr/legacy.py`).
- **Nuclear-exit / cleanup pattern** (`base_pipeline.py:107-170`): in subprocess workers
  (`WHISPERJAV_SUBPROCESS_WORKER=1`) ASR `cleanup()` IS called during controlled execution to
  trigger the ctranslate2 C++ destructor safely, but `torch.cuda.empty_cache()` is SKIPPED
  (crashes on Windows during shutdown → BrokenProcessPool). Do not "unify" these branches.

## Parameter threading (the call chain to verify)

`main.py` argparse → ensemble config (`pass_worker.py:415-495`) → `_build_pipeline`
(`:1038-1534`, applies GUI overrides + sensitivity preset) → `Pipeline.__init__` → ASR.
Trace with `call-chain-verifier`; cross-check resolved values with `--dump-params`.

## Qwen/anime VAD grouping + padding (verified 2026-06-25, CORRECTED 2026-07-03)

The qwen pipeline owns its segmenter grouping/padding defaults as **scalar attrs**, not via
`segmenter_config`. At `qwen_pipeline.py:752-769` the segmenter kwargs are seeded from
`segmenter_config` then the scalars **overwrite** `max_group_duration_s`, `chunk_threshold_s`,
`start_pad_ms`, `end_pad_ms`. So:
- The four GUI "Customize Parameters" sliders ARE live: Max-Group / Frame-Gap / Start-Pad /
  End-Pad populate the scalars (`pass_worker.py:1374-1393`, `main.py:1708-1725`); the clobber
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
  anime/qwen3 via a `user_segmenter_overrides` injection (`pass_worker.py:1267-1268` / `main.py:1625-1627`),
  which likewise makes the sensitivity threshold-gradient inert for those two
  backends (explicit slider still wins; cohere excluded).
- **v1.9.0 JAV defaults** (qwen3 + anime-whisper only; cohere keeps 1.0s/6.0s + 300ms pads):
  `max_group=3.0s`, `chunk_threshold=0.3s` (group only if gap <300ms), symmetric padding
  `start=100ms`/`end=100ms`, `threshold=0.25` (owner "Option B", validated on anime-whisper
  2026-07-03; grouping value re-tuned from the earlier 4.0/0.4 — those never took effect for
  anime due to the constructor clobber above). Defaults live in `qwen_pipeline.py:119-122` +
  anime branches (`pass_worker.py:1329-1336`, `main.py:1747-1756`).
- **Asymmetric padding**: whisperseg gained `start_pad_ms`/`end_pad_ms` (`whisperseg.py`), both
  inherit `speech_pad_ms` when unset. Must be in `factory.py` whisperseg schema or
  `_sanitize_params` strips them. silero/ten/nemo/whisper-vad lack them → stripped for those.
- **GUI ms↔s**: Frame Gap is `chunk_threshold_ms` in the GUI (50ms step), converted to seconds
  at pack time (`api.py` pass1/pass2 blocks). Start/End pad are ms end-to-end (no conversion).

## Tests that pin behavior

`tests/test_config_v4.py`, `tests/test_ensemble_params.py`, `tests/test_qwen_sensitivity.py`,
`tests/test_qwen_pipeline_integration.py`, `tests/test_gui_custom_params_simulation.py`.
(Confirm assertions by reading; some names above are inferred from filenames.)
