

  My Recommendations

  1. Fix F-01 and F-02 now — they're small changes (~20 lines total) that prevent latent bugs from becoming real bugs when scope expands
  2. Document the WordTimestamp ↔ Dict conversion — add a comment in the orchestrator explaining the boundary
  3. Update the ADR status to reflect Phase 7 and the actual scope boundary (which modes use the orchestrator)
  4. Don't rush the deferred items — they were deferred for good architectural reasons. The step-down retry (your Issue 4) and coupled mode migration need proper design

  Both audit documents are now in docs/architecture/ for your reference. They won't be committed (docs/ is gitignored) but they're available for future sessions.
  
  
  
  ---                                                                                                                                                                                            Assembly Mode Execution Flow (GUI Ensemble Tab)
                                                                                                                                                                                                 Here's the exact flow when a user selects assembly as the input mode for Qwen in the ensemble tab:

  What the GUI sends

  The ensemble tab's default Qwen state is inputMode: 'vad_slicing' (line 1216). If the user explicitly changes the dropdown to assembly, the config sent to the API includes:

  pass: { pipeline: "qwen", inputMode: "assembly", speechSegmenter: "ten", ... }


  API → CLI args

  api.py:2229-2230 converts this to:

  --pass1-qwen-params '{"input_mode": "assembly"}'


  And separately (api.py:2257-2258), the speech segmenter goes as:

  --pass1-speech-segmenter ten


  pass_worker.py → QwenPipeline construction

  pass_worker.py:872-873 maps the speech segmenter to qwen_segmenter:
  qwen_defaults["qwen_segmenter"] = "ten"  # from pass_config["speech_segmenter"]

  pass_worker.py:903 maps input_mode:
  "qwen_input_mode": "assembly"

  The QwenPipeline is constructed with both (line 923-929).

  The 8-Phase Pipeline Execution

  Phase 1: Audio Extraction — Same for all modes. Extract WAV at 48kHz.

  Phase 2: Scene Detection — Because input_mode == ASSEMBLY and safe_chunking == True (default), scene detection uses assembly-specific limits: min_duration=30s, max_duration=120s (lines
  495-504). Default scene detector is "semantic". Produces scenes of 30-120s.

  Phase 3: Speech Enhancement — Default "none", so this is a passthrough.

  Phase 4: Speech Segmentation (TEN VAD) — This is the key question. Because segmenter_backend == "ten" (the default), Phase 4 RUNS. It runs TEN VAD on every scene, producing
  speech_regions_per_scene: {scene_idx: SegmentationResult}. Each SegmentationResult contains speech segments with start/end times and groups.

  Phase 5: ASR via DecoupledSubtitlePipeline — Because input_mode == ASSEMBLY, the code takes the assembly branch (line 641). The orchestrator (_subtitle_pipeline.process_scenes()) executes:

  1. Step 1 (Framing): FullSceneFramer produces 1 frame per scene = [0, duration]. The framer does NOT use the VAD data — it treats each scene as one whole block.
  2. Steps 2-4 (Generate + Clean): Load Qwen3 text-only model → transcribe_text_only() on the full scene audio → unload model → AssemblyTextCleaner.clean_batch().
  3. Steps 5-7 (Align): Load ForcedAligner → align_batch() on the full scene → unload aligner.
  4. Step 9 (Sentinel + Reconstruct + Harden): Merge words → sentinel quality check → reconstruct WhisperResult → harden timestamps.

  The Phase 4 speech regions are passed to the orchestrator (orch_speech_regions, lines 658-668) and used for:
  - Sentinel collapse recovery (line 464 in orchestrator): If the alignment collapses, speech regions guide word redistribution.
  - Hardening config (line 527-533 in orchestrator, after our F-02 fix): Speech regions are available to the hardening stage.

  But crucially: Phase 4 VAD data does NOT affect the framing or generation path in assembly mode. The FullSceneFramer ignores it. The text-only generator processes the entire scene audio.
  The aligner processes the entire scene audio. VAD is only a safety net for sentinel recovery and hardening.

  Phases 6-8: Post-processing — SRT stitching, post-processing, output. Same for all modes.

  Summary: Default Assembly Flow

  Audio → Scene Detection (semantic, 30-120s scenes)
        → Speech Enhancement (none, passthrough)
        → TEN VAD (runs, produces speech regions, BUT...)
        → DecoupledSubtitlePipeline:
            FullSceneFramer (1 frame = entire scene, ignores VAD)
            → Qwen3 text-only (full scene audio)
            → AssemblyTextCleaner
            → Qwen3 ForcedAligner (full scene audio)
            → Sentinel (uses VAD regions IF collapse detected)
            → Reconstruct → Harden
        → SRT output


  ---
  Does Changing the Speech Segmenter Setting Change Anything?

  Short answer: Almost nothing for assembly mode.

  If the user changes the speech segmenter dropdown (e.g., from ten to silero-v3.1 or nemo):

  1. Phase 4 still runs with the chosen backend instead of TEN
  2. Different VAD produces different speech_regions_per_scene
  3. Those regions are passed to the orchestrator

  But in the orchestrator:
  - FullSceneFramer still ignores them — 1 frame per scene regardless
  - Text generation still processes full scene — no change
  - Alignment still processes full scene — no change
  - Sentinel recovery: Different VAD regions = different collapse redistribution IF a collapse is detected. This is a subtle difference that only manifests on quality-edge cases.
  - Hardening: The regions are now available (F-02 fix) but _apply_vad_only_timestamps() only triggers for TimestampMode.VAD_ONLY, and assembly defaults to ALIGNER_WITH_INTERPOLATION — so
  hardening doesn't use them either.

  If the user changes the segmenter to "none":
  - Phase 4 is skipped entirely (line 581: if self.segmenter_backend != "none")
  - speech_regions_per_scene is empty {}
  - orch_speech_regions is None
  - Sentinel collapse recovery falls back to proportional redistribution instead of VAD-guided
  - This is a real behavioral difference — but only when alignment collapses

  Bottom line: For assembly mode, the speech segmenter setting is essentially a dormant safety net. It affects collapse recovery strategy (VAD-guided vs. proportional) and nothing else. The
  user would not notice any difference in normal operation. The segmenter was designed for vad_slicing mode where it drives the entire temporal framing — in assembly mode, it's just along for
   the ride.

✻ Brewed for 2m 30s                                                                                                                                                                            
> based on all the learnings I have two specific questions: what is the best way --best practices way forward to handle the defferred gaps. and secondly how best to address the issues 4
  that we tabled.
                                                                                                                                                                                               ● These are deeply interrelated questions. Let me think through them architecturally before answering.                                                                                         
  Question 1: Best Practices for the Deferred Gaps                                                                                                                                             
  The deferred items fall into three distinct categories with different strategies:

  Category A: Mode Migrations (VAD_SLICING, CONTEXT_AWARE → orchestrator)

  These are the biggest gaps, and they're deferred for good reason — not because they're hard to code, but because the architectural design for step-down retry doesn't exist yet.
  VAD_SLICING's entire value proposition is its step-down (Tier 1 → Tier 2) and per-group sentinel. Migrating it to the orchestrator without step-down would be a regression.

  Best practice: Don't migrate these until the orchestrator has a proven retry/step-down mechanism. Which leads directly to Issue 4 (I'll address below). The correct dependency chain is:

  Issue 4 (step-down design) → Assembly step-down works
      → VAD_SLICING migration (uses same step-down mechanism)
      → CONTEXT_AWARE migration (simpler, may not need step-down)


  Anti-pattern to avoid: Migrating VAD_SLICING "as-is" without step-down, thinking "we'll add step-down later." That creates a regression that users will hit immediately.

  Category B: New Backends (whisper-segment, TransformersGenerator, VLLMGenerator, CTCAligner, etc.)

  These are demand-driven. Each backend is a self-contained deliverable that can be added independently when there's a user need.

  Best practice: Don't build these speculatively. The protocol+factory architecture is specifically designed so backends can be added without touching the orchestrator. When a real use case
  demands one (e.g., a user wants Whisper-Guided Qwen), build just that backend.

  Priority signal: The whisper-segment framer is the most interesting because it enables the ADR's most compelling workflow (cross-model composability). But it also requires solving a VRAM
  question — loading a second ASR model alongside Qwen. This is a design question, not a coding question.

  Category C: CLI/GUI Exposure of Composability

  The orchestrator supports workflows (SRT re-transcription, aligner-free mode) that no user can reach because there are no CLI flags or GUI options for them.

  Best practice: This is the lowest effort, highest immediate value work. A few CLI flags (--qwen-framer, --qwen-aligner) would unlock existing functionality. But the question is: do users
  actually want this? The answer should drive whether to expose it.

  Recommended Sequencing

  1. Design step-down mechanism for orchestrator (Issue 4)
  2. Implement & validate for assembly mode
  3. Migrate VAD_SLICING using same mechanism
  4. (Optional) Expose composability via CLI if users request it
  5. (Demand-driven) Build new backends when needed
  6. (Last) Migrate CONTEXT_AWARE (works fine in coupled mode)


  ---
  Question 2: How to Address Issue 4 (Assembly Step-Down)

  This is the crux. Let me explain the problem precisely, then the design options.

  The Problem

  Today in assembly mode, each scene (30-120s) is processed as one monolithic block:

  120s scene → FullSceneFramer → 1 frame → generate → align → done or fail


  If anything goes wrong (OOM, alignment collapse, empty transcription), the entire scene is lost — (None, error_diagnostics) is returned. There is no retry.

  In contrast, the old coupled/VAD_SLICING mode has step-down in _transcribe_speech_regions():
  - Tier 1: Try 30s groups
  - If failure: Tier 2: Retry with 6s groups
  - Per-group isolation: one group failing doesn't kill the scene

  The Architectural Challenge

  Step-down conflicts with the orchestrator's current batch-oriented VRAM lifecycle:

  Current flow:
    generator.load() → generate ALL scenes → generator.unload()
    → aligner.load() → align ALL scenes → aligner.unload()


  If scene 5 out of 20 fails, we've already unloaded the generator. Step-down for scene 5 means reloading it. This is the tension between batch efficiency and per-scene resilience.

  Three Design Options

  Option A: Post-Hoc Retry Pass

  After the full pipeline finishes, identify failed scenes, re-frame them with VadGroupedFramer, and run a second complete pipeline pass (load generator → generate → unload → load aligner →
  align → unload) for just the failures.

  Pass 1: All scenes with FullSceneFramer (normal batch flow)
    → Identify failed/collapsed scenes
  Pass 2: Failed scenes only with VadGroupedFramer (second batch flow)
    → Merge results


  Pros: Clean separation. Batch efficiency preserved for happy path. The retry is a complete, self-contained pipeline run — easy to reason about. No changes to the inner orchestrator flow.

  Cons: Two full VRAM load/unload cycles when failures occur (~10-15s overhead per retry pass). The generator sees different audio (frame slices vs. full scene) — text quality may differ.

  Option B: Integrated Per-Scene Retry Within process_scenes()

  The orchestrator tracks which scenes failed after step 9. Then, before returning, it does a targeted retry: re-frames just those scenes, does a second generator load/generate/unload +
  aligner load/align/unload cycle for just the failures.

  process_scenes():
    Step 1-9: Normal flow for all scenes
    → Collect failed scene indices
    If failures:
      Re-frame failed scenes with fallback framer
      generator.load() → generate failed frames → generator.unload()
      aligner.load() → align failed frames → aligner.unload()
      Step 9 again for failed scenes
      Replace failed results


  Pros: Everything stays inside process_scenes() — the caller doesn't know about retry. The VRAM cost only occurs when failures happen.

  Cons: process_scenes() becomes more complex. Needs a second framer (where does it come from? Constructor parameter?).

  Option C: Framer-Level Step-Down (TemporalFramer grows a reframe() method)

  Add a protocol method: reframe(scene_audio, previous_frames, failure_info) → FramingResult. The orchestrator calls this when a scene fails, getting finer frames back, then retries just that
   scene.

  protocol TemporalFramer:
      def frame(self, audio, sr, **kw) -> FramingResult   # initial framing
      def reframe(self, audio, sr, prev_result, **kw) -> FramingResult  # step-down


  Pros: Step-down strategy is encapsulated in the framer (separation of concerns). Different framers can have different step-down strategies. The orchestrator just asks "give me smaller
  frames" without knowing how.

  Cons: Protocol change — affects all existing framers (they'd need at least a no-op reframe()). Adds complexity to a protocol that's currently simple. The framer doesn't have enough context
  to decide what "step-down" means (it doesn't know WHY the failure happened).

  My Recommendation

  Option B is the sweet spot. Here's why:

  1. It keeps step-down invisible to callers (QwenPipeline doesn't need to change)
  2. It preserves batch efficiency for the happy path (which is 95%+ of scenes)
  3. It's the same pattern as the existing batch-fallback in generation (try batch → fall back to per-frame)
  4. The "fallback framer" can be injected at construction time — the orchestrator already takes a framer parameter; we'd add an optional fallback_framer

  The constructor change would be:

  def __init__(
      self,
      framer: TemporalFramer,
      generator: TextGenerator,
      cleaner: TextCleaner,
      aligner: Optional[TextAligner],
      hardening_config: HardeningConfig,
      fallback_framer: Optional[TemporalFramer] = None,  # Step-down
      ...
  )

  And _build_subtitle_pipeline() in QwenPipeline would wire:
  framer = TemporalFramerFactory.create("full-scene")
  fallback_framer = TemporalFramerFactory.create("vad-grouped", max_group_duration_s=29.0)

  The retry logic in process_scenes() would be:

  Normal flow → collect results
  If fallback_framer and any scene has result=None:
      Re-frame failed scenes with fallback_framer
      Second generator pass (load/generate/unload)
      Second aligner pass (load/align/unload)
      Step 9 for retried scenes
      Replace None results with retry results


  What "Failed" Means

  This needs definition. I'd suggest three triggers for step-down:
  1. Result is None (exception during processing — OOM, aligner crash)
  2. Sentinel status is COLLAPSED and recovery produced poor quality (the sentinel already detects this)
  3. Empty transcription (generator returned blank text for a scene with audio)

  Option 1 is the safe minimum. Options 2-3 require quality assessment logic.

  Relationship to VAD_SLICING Migration

  Once Option B works for assembly, migrating VAD_SLICING becomes straightforward:
  - Primary framer: VadGroupedFramer(max_group_duration_s=30.0) (Tier 1)
  - Fallback framer: VadGroupedFramer(max_group_duration_s=6.0) (Tier 2)
  - Same retry mechanism, different framers

  This is why solving Issue 4 first unblocks the entire deferred gap cluster.

  ---
  Summary

  ┌──────────────────────────────┬─────────────────────────────────────────────────────────────────────┐
  │           Question           │                               Answer                                │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ Deferred gaps strategy       │ Solve step-down first → migrate modes → add backends on demand      │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ Issue 4 approach             │ Option B: integrated retry in process_scenes() with fallback_framer │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ What triggers step-down      │ Scene result is None (safe minimum)                                 │
  ├──────────────────────────────┼─────────────────────────────────────────────────────────────────────┤
  │ Why this unblocks everything │ Same mechanism serves assembly step-down AND VAD_SLICING migration  │
  └──────────────────────────────┴─────────────────────────────────────────────────────────────────────┘

  Shall I plan the implementation?
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
   Expose Pipeline Composability via CLI and GUI

 Context

 WhisperJAV has a modular, protocol-based pipeline architecture (ADR-006) with clear component contracts. But users can't access most of it. The architecture's composability is invisible —
 hardcoded behind --input-mode assembly with fixed defaults. The user wants to surface pipeline composability so power users can tune each stage, while casual users still get working
 defaults.

 Triggered by: User observed 158-second wasted GPU time on a non-speech scene during assembly mode. No lever existed to skip it, tune scene sizes, or control any stage parameter beyond what
 the modal offered.

 ---
 Part 1: User-Centric Design Proposal

 User Personas

 ┌─────────────┬──────────────────────────────────────────────┬──────────────────────────────────────────────────────────┐
 │   Persona   │                 Description                  │                          Needs                           │
 ├─────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ Power User  │ JAV community, 8-24GB VRAM, 5-50 files/week  │ Fine control per pipeline stage, save configs as presets │
 ├─────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ Casual User │ Occasional use, GUI-only, wants "just works" │ Working defaults, single-knob tuning, clear labels       │
 ├─────────────┼──────────────────────────────────────────────┼──────────────────────────────────────────────────────────┤
 │ CLI User    │ Batch scripts, copy-paste from forums        │ Discoverable --help, preset shortcuts, clear naming      │
 └─────────────┴──────────────────────────────────────────────┴──────────────────────────────────────────────────────────┘

 Mental Model: Pipeline as Linear Stage Chain

 Users should see the pipeline as it actually works — a linear chain of stages:

 INPUT → [Scene Detection] → [Enhancement] → [Segmentation] → [Transcription] → [Cleaning] → [Alignment] → [Hardening] → OUTPUT


 The current GUI tab names (Model, Quality, Aligner, Post-Process, Scene, Context) mix stages and concerns. The redesign maps tabs to actual pipeline stages.

 Progressive Disclosure

 - Level 0: Select pipeline + input mode, press Start (today's behavior, unchanged)
 - Level 1: Ensemble grid dropdowns for component backends (already exists, well-designed)
 - Level 2: "Customize Parameters" modal with stage-mapped tabs (the redesign)
 - Level 3: Named profiles/presets for common scenarios (new)

 Input-Mode-Aware Adaptation

 The three modes have different relevant parameters:

 ┌─────────────────┬─────────────────────┬──────────────────────┬──────────────────────┐
 │      Stage      │      Assembly       │    Context-Aware     │     VAD Slicing      │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Scene Detection │ Critical (max 120s) │ Critical (12-90s)    │ Less important       │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Enhancement     │ Available           │ Available            │ Available            │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Segmentation    │ Sentinel-only       │ Step-down grouping   │ Primary chunking     │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Text Generation │ Batch text-only     │ batch_size=1 coupled │ batch_size=1 coupled │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Text Cleaning   │ Assembly cleaner    │ N/A                  │ N/A                  │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Alignment       │ Separate VRAM phase │ Coupled with ASR     │ Coupled with ASR     │
 ├─────────────────┼─────────────────────┼──────────────────────┼──────────────────────┤
 │ Step-Down       │ Available           │ Available            │ N/A (pre-sliced)     │
 └─────────────────┴─────────────────────┴──────────────────────┴──────────────────────┘

 The modal adapts: shows/hides sections, adjusts defaults and descriptions per mode.

 ---
 Part 2: GUI Interface Proposal

 Revised Tab Structure (5 tabs, pipeline-stage-mapped)

 [ Pipeline ] [ Audio Prep ] [ Transcription ] [ Alignment ] [ Output ]


 Tab 1: Pipeline — What model, what language, what context
 - ASR Model dropdown (1.7B / 0.6B)
 - Device, Data Type, Attention implementation
 - Language dropdown
 - Context hints textarea
 - Header: "Configure the ASR model and language. These are the most impactful choices."

 Tab 2: Audio Prep — Everything before ASR (3 collapsible sections)

 Section: Scene Detection
 - Backend dropdown (synced with ensemble grid)
 - Min duration slider (0.2-60s, default varies by input mode)
 - Max duration slider (30-300s, default varies by input mode)
 - Info banner for assembly: "Enforces max 120s (aligner limit 180s)"

 Section: Speech Enhancement
 - Backend dropdown (synced with ensemble grid)
 - Backend-specific params (dynamic)

 Section: Speech Segmentation
 - Backend dropdown (synced with ensemble grid)
 - Max group duration slider (5-60s, default 29s)
 - NEW: "Skip scenes with no detected speech" checkbox (default OFF)
   - Description: "Skip scenes where VAD finds zero speech. Saves processing time for music/silence sections."

 Header: "Audio is prepared in 3 stages: scene splitting, enhancement, and speech detection."

 Tab 3: Transcription — How text generation works
 - Batch size slider (1-8)
 - Max new tokens slider (1024-8192)
 - Repetition penalty slider (0.8-2.0)
 - Max tokens per audio second slider (5-40)
 - Assembly-only section: Assembly text cleaner toggle
 - Header: "Controls how the ASR model generates text from audio."

 Tab 4: Alignment — How timestamps are resolved

 Section: Forced Aligner
 - Use ForcedAligner toggle
 - Aligner model dropdown
 - Info: "3-minute segment limit"

 Section: Timestamp Resolution
 - Timestamp mode dropdown (4 options with descriptions)

 Section: Step-Down Retry (hidden for vad_slicing)
 - Enable toggle
 - Tier 1 / Tier 2 duration sliders

 Header: "Controls how generated text is mapped to precise timestamps."

 Tab 5: Output — Final formatting
 - Post-processing preset dropdown (high_moan / default / narrative)
 - Deprecated controls (collapsed, grayed out)

 Input-Mode Adaptation Behavior

 When input_mode changes:
 - Assembly: Show assembly cleaner, show step-down, scene defaults 30-120s, timestamp default "aligner_interpolation"
 - Context-Aware: Hide assembly cleaner, show step-down, scene defaults 12-90s, timestamp default "aligner_vad_fallback", batch_size locked at 1
 - VAD Slicing: Hide assembly cleaner, hide step-down, batch_size locked at 1

 Profile Dropdown (at top of modal)

 Profile: [Default ▼] [Interview] [BGM-Heavy] [Short Scenes] [High Fidelity] [Fast] [Custom]


 Selecting a profile populates all controls. Manual changes switch to "Custom (based on X)".

 ---
 Part 3: CLI Interface Proposal

 New CLI Arguments

 ┌───────────────────────────┬────────┬─────────────┬────────────────────────────────────────────────────────────────────┐
 │         Argument          │  Type  │   Default   │                              Purpose                               │
 ├───────────────────────────┼────────┼─────────────┼────────────────────────────────────────────────────────────────────┤
 │ --qwen-skip-silent-scenes │ flag   │ False       │ Skip scenes with 0% speech coverage                                │
 ├───────────────────────────┼────────┼─────────────┼────────────────────────────────────────────────────────────────────┤
 │ --qwen-scene-min-dur      │ float  │ None (auto) │ Override minimum scene duration                                    │
 ├───────────────────────────┼────────┼─────────────┼────────────────────────────────────────────────────────────────────┤
 │ --qwen-scene-max-dur      │ float  │ None (auto) │ Override maximum scene duration                                    │
 ├───────────────────────────┼────────┼─────────────┼────────────────────────────────────────────────────────────────────┤
 │ --qwen-profile            │ choice │ None        │ Named preset (interview/bgm-heavy/short-scenes/high-fidelity/fast) │
 └───────────────────────────┴────────┴─────────────┴────────────────────────────────────────────────────────────────────┘

 Argparse Regrouping (help text only, no functional change)

 Split the single qwen_group into 5 descriptive sub-groups matching the GUI tabs:
 1. Pipeline Strategy (model, device, dtype, attn, language, context)
 2. Audio Preparation (scene, enhancer, segmenter, skip-silent, min/max dur)
 3. Text Generation (batch_size, max_tokens, repetition_penalty, tokens/s, assembly_cleaner)
 4. Alignment & Timestamps (aligner, timestamp_mode, safe_chunking, stepdown)
 5. Output (postprocess_preset)

 Profile Definitions

 Stored in whisperjav/config/qwen_profiles.py:

 - interview: context_aware, semantic scenes, no enhancer, 30-90s scenes
 - bgm-heavy: assembly, semantic scenes, enhancer on, higher repetition_penalty
 - short-scenes: assembly, auditok scenes, 5-60s range, 15s group duration
 - high-fidelity: assembly, all stages enabled, semantic scenes, 30-120s
 - fast: vad_slicing, 0.6B model, auditok, no enhancer, no stepdown

 Profiles are sparse dicts. --qwen-* args override profile values.

 ---
 Part 4: Implementation Plan

 Phase 1: Backend Schema + New Pipeline Parameters

 Goal: Restructure the schema API and add new params to pipeline.

 Files:
 - whisperjav/webview_gui/api.py — Restructure get_qwen_schema(): new 5-group schema keys (pipeline, audio_prep, transcription, alignment, output). Add skip_silent_scenes, scene_min_dur,
 scene_max_dur params. Add input_mode_visibility metadata per param (tells frontend which modes show which sections).
 - whisperjav/pipelines/qwen_pipeline.py — Add constructor params: skip_silent_scenes, scene_min_dur, scene_max_dur. Phase 2: apply scene min/max when provided. Phase 4-5 boundary: skip
 scenes with zero speech when enabled.
 - whisperjav/main.py — Add 3 new CLI args.
 - whisperjav/ensemble/pass_worker.py — Add to DEFAULT_QWEN_PARAMS, prepare_qwen_params() mapping, _build_pipeline().

 Scope: ~200 lines. Dependencies: None.

 Phase 2: Frontend Tab Restructuring

 Goal: New 5-tab Customize modal that maps to pipeline stages.

 Files:
 - whisperjav/webview_gui/assets/app.js — Rewrite generateQwenTabs() and all generateQwen*Tab() methods. New methods for 5 tabs. Update applyCustomization() and resetQwenToDefaults().
 - whisperjav/webview_gui/assets/index.html — Update modal tab buttons (minimal: they're mostly dynamic).

 Scope: ~500 lines JS refactored. Dependencies: Phase 1 (schema).

 Phase 3: Input-Mode-Aware Adaptation

 Goal: Modal adapts when input mode changes.

 Files:
 - whisperjav/webview_gui/assets/app.js — Add updateModalForInputMode(mode) function. Wire to ensemble grid input-mode dropdown. Show/hide conditional sections, update defaults/descriptions.

 Scope: ~100 lines. Dependencies: Phase 2 (tab structure).

 Phase 4: Profiles (CLI + GUI)

 Goal: Named presets for common configurations.

 Files:
 - whisperjav/config/qwen_profiles.py (new) — Profile definitions.
 - whisperjav/main.py — Add --qwen-profile arg, resolution logic.
 - whisperjav/ensemble/pass_worker.py — Profile resolution in prepare_qwen_params().
 - whisperjav/webview_gui/api.py — New get_qwen_profiles() method.
 - whisperjav/webview_gui/assets/app.js — Profile dropdown at modal top.

 Scope: ~300 lines. Dependencies: Phase 1 (params), Phase 2 (tab structure for GUI part).

 Phase 5: CLI Help Regrouping

 Goal: Reorganize --help output into 5 named groups.

 Files:
 - whisperjav/main.py — Split single qwen_group into 5 argparse groups.

 Scope: ~50 lines (reorganization only). Dependencies: Phase 1 (new args exist).

 Build Order

 Phase 1 (Backend) → Phase 2 (Frontend tabs) → Phase 3 (Mode adaptation)
                  ↘ Phase 5 (CLI help)       → Phase 4 (Profiles)


 Phases 1+5 are independently shippable as CLI improvements.
 Phases 2+3 ship together as the GUI redesign.
 Phase 4 builds on top of everything.

 Deliverable

 Save the complete design document to docs/architecture/COMPOSABILITY_DESIGN.md. This is a design-only session — no code changes.
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌

 Claude has written up a plan and is ready to execute. Would you like to proceed?

 > 1. Yes, clear context and auto-accept edits (shift+tab)
   2. Yes, auto-accept edits
   3. Yes, manually approve edits
   4. Type here to tell Claude what to change

 ctrl-g to edit in Notepad · ~\.claude\plans\modular-skipping-crayon.md





























































































































































