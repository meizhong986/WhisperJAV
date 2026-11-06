"""
Tests for presets - pin the CURRENT intended sensitivity values.

HISTORY: this file originally verified Pydantic presets against the legacy
sections of asr_config.json. Those sections were removed in the v1.8.9
config cleanup (asr_config.json now only carries ui_preferences), which
left every comparison raising KeyError. As of v1.9.0 the Pydantic presets
in whisperjav/config/schemas/presets.py ARE the source of truth, so these
tests pin the intended values inline instead.

Value provenance (update the pins ONLY with a matching release-note entry):
- v1.8.10-hf2: logprob_threshold -0.75 -> -1.00, logprob_margin 0.2 -> 0.0
- v1.8.10-hf3: silero thresholds 0.18/0.35/0.05 -> 0.28/0.41/0.18;
  best_of 1 -> 2, patience 2.0 -> 1.6 (balanced pipeline mapping)
- v1.8.12:     silero max_speech/max_group tightened (components registry)
- C1 fix:      hallucination_silence_threshold None = disabled, no fallback
- CL1b:        balanced temperature [0.0]
"""


from whisperjav.config.schemas import (
    DECODER_PRESETS,
    FASTER_WHISPER_ENGINE_PRESETS,
    HALLUCINATION_THRESHOLDS,
    SILERO_VAD_PRESETS,
    STABLE_TS_ENGINE_OPTIONS,
    STABLE_TS_VAD_PRESETS,
    TRANSCRIBER_PRESETS,
    Sensitivity,
    get_decoder_preset,
    get_faster_whisper_engine_preset,
    get_silero_vad_preset,
    get_stable_ts_vad_preset,
    get_transcriber_preset,
)


class TestTranscriberPresets:
    """Pin TranscriberOptions preset values."""

    def test_balanced_values(self):
        preset = TRANSCRIBER_PRESETS[Sensitivity.BALANCED]
        assert preset.temperature == [0.0]                     # CL1b
        assert preset.compression_ratio_threshold == 2.4
        assert preset.logprob_threshold == -1.0                # v1.8.10-hf2
        assert preset.logprob_margin == 0.0                    # v1.8.10-hf2
        assert preset.no_speech_threshold == 0.65              # v1.8.10-hf3
        assert preset.condition_on_previous_text is False
        assert preset.word_timestamps is True

    def test_conservative_values(self):
        preset = TRANSCRIBER_PRESETS[Sensitivity.CONSERVATIVE]
        assert preset.temperature == [0.0]
        assert preset.compression_ratio_threshold == 2.2
        assert preset.logprob_threshold == -0.84
        assert preset.no_speech_threshold == 0.54

    def test_aggressive_values(self):
        preset = TRANSCRIBER_PRESETS[Sensitivity.AGGRESSIVE]
        assert preset.temperature == [0.0, 0.2]
        assert preset.compression_ratio_threshold == 2.6
        assert preset.logprob_threshold == -1.0
        assert preset.no_speech_threshold == 0.72


class TestDecoderPresets:
    """Pin DecoderOptions preset values."""

    def test_balanced_values(self):
        preset = DECODER_PRESETS[Sensitivity.BALANCED]
        assert preset.task == "transcribe"
        assert preset.language == "ja"
        assert preset.beam_size == 2
        assert preset.best_of == 2                             # v1.8.10-hf3
        assert preset.patience == 1.2
        assert preset.suppress_blank is True
        # None -> whisper library default applies (dropped on export)
        assert preset.suppress_tokens is None

    def test_conservative_values(self):
        preset = DECODER_PRESETS[Sensitivity.CONSERVATIVE]
        assert preset.beam_size == 2
        assert preset.best_of == 2
        assert preset.patience == 1.0

    def test_aggressive_values(self):
        preset = DECODER_PRESETS[Sensitivity.AGGRESSIVE]
        assert preset.beam_size == 3
        assert preset.best_of == 2
        assert preset.patience == 1.3


class TestSileroVADPresets:
    """Pin Silero VAD preset values (schemas copy)."""

    def test_balanced_values(self):
        preset = SILERO_VAD_PRESETS[Sensitivity.BALANCED]
        assert preset.threshold == 0.28                        # v1.8.10-hf3
        assert preset.min_speech_duration_ms == 100
        assert preset.max_speech_duration_s == 5.0             # v1.8.12
        assert preset.min_silence_duration_ms == 300
        assert preset.speech_pad_ms == 400

    def test_conservative_values(self):
        preset = SILERO_VAD_PRESETS[Sensitivity.CONSERVATIVE]
        assert preset.threshold == 0.41                        # v1.8.10-hf3
        assert preset.min_speech_duration_ms == 150
        assert preset.max_speech_duration_s == 6.0
        assert preset.speech_pad_ms == 500

    def test_aggressive_values(self):
        preset = SILERO_VAD_PRESETS[Sensitivity.AGGRESSIVE]
        assert preset.threshold == 0.18                        # v1.8.10-hf3
        assert preset.min_speech_duration_ms == 30
        assert preset.max_speech_duration_s == 4.0             # v1.8.12
        assert preset.speech_pad_ms == 300

    def test_matches_runtime_registry(self):
        """Guard: the schemas copy must not drift from the runtime registry.

        resolve_config_v3 reads presets from the components VAD registry
        (whisperjav/config/components/vad/silero.py), NOT from this schemas
        copy. Before v1.9.0 the two silently diverged (schemas kept the
        pre-v1.8.12 max_speech caps). Grouping keys (chunk_threshold_s,
        max_group_duration_s) exist only in the registry by design - see
        SegmenterGroupingOptions.
        """
        from whisperjav.config.components.base import get_vad_registry

        component = get_vad_registry()["silero"]
        shared_keys = (
            "threshold", "min_speech_duration_ms", "max_speech_duration_s",
            "min_silence_duration_ms", "speech_pad_ms",
        )
        for sens in Sensitivity:
            schemas_preset = SILERO_VAD_PRESETS[sens].model_dump()
            registry_preset = component.get_preset(sens.value).model_dump()
            for key in shared_keys:
                assert schemas_preset[key] == registry_preset[key], (
                    f"schemas/presets.py diverged from components registry: "
                    f"{sens.value}.{key}: {schemas_preset[key]} != {registry_preset[key]}"
                )


class TestStableTSVADPresets:
    """Pin Stable-TS VAD preset values."""

    def test_balanced_values(self):
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.BALANCED]
        assert preset.vad is True
        assert preset.vad_threshold == 0.25

    def test_conservative_values(self):
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.CONSERVATIVE]
        assert preset.vad_threshold == 0.35

    def test_aggressive_values(self):
        preset = STABLE_TS_VAD_PRESETS[Sensitivity.AGGRESSIVE]
        assert preset.vad_threshold == 0.1


class TestFasterWhisperEnginePresets:
    """Pin Faster-Whisper engine preset values."""

    def test_balanced_values(self):
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.BALANCED]
        assert preset.repetition_penalty == 1.5
        assert preset.no_repeat_ngram_size == 3                # H1
        assert preset.multilingual is False

    def test_conservative_values(self):
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.CONSERVATIVE]
        assert preset.repetition_penalty == 1.8
        assert preset.no_repeat_ngram_size == 3

    def test_aggressive_values(self):
        preset = FASTER_WHISPER_ENGINE_PRESETS[Sensitivity.AGGRESSIVE]
        assert preset.repetition_penalty == 1.3
        assert preset.no_repeat_ngram_size == 3
        # chunk_length=30 is NOT a bug - empirically confirmed, see
        # memory/project_v1813_session_lessons.md lesson 1
        assert preset.chunk_length == 30


class TestStableTSEngineOptions:
    """Pin Stable-TS engine option defaults."""

    def test_default_values(self):
        opts = STABLE_TS_ENGINE_OPTIONS
        assert opts.gap_padding == " ..."
        assert opts.max_instant_words == 0.5
        assert opts.nonspeech_error == 0.1
        assert opts.ignore_compatibility is True
        assert opts.regroup is True
        assert opts.q_levels == 20
        assert opts.k_size == 5


class TestHallucinationThresholds:
    """hallucination_silence_threshold: None = disabled (C1 fix)."""

    def test_values(self):
        for sens in Sensitivity:
            assert HALLUCINATION_THRESHOLDS[sens] is None, (
                f"C1 fix: hallucination_silence_threshold must be None "
                f"(disabled) for {sens.value}"
            )


class TestPresetGetters:
    """Getter functions return the same objects as the dicts."""

    def test_getters_match_dicts(self):
        for sens in Sensitivity:
            assert get_transcriber_preset(sens) == TRANSCRIBER_PRESETS[sens]
            assert get_decoder_preset(sens) == DECODER_PRESETS[sens]
            assert get_silero_vad_preset(sens) == SILERO_VAD_PRESETS[sens]
            assert get_stable_ts_vad_preset(sens) == STABLE_TS_VAD_PRESETS[sens]
            assert get_faster_whisper_engine_preset(sens) == FASTER_WHISPER_ENGINE_PRESETS[sens]


class TestPresetExport:
    """Test presets export without None values."""

    def test_transcriber_export(self):
        preset = TRANSCRIBER_PRESETS[Sensitivity.BALANCED]
        result = preset.model_dump_without_none()
        assert "initial_prompt" not in result
        assert "prepend_punctuations" not in result
        assert result["temperature"] == [0.0]                  # CL1b
        assert result["word_timestamps"] is True

    def test_decoder_export(self):
        preset = DECODER_PRESETS[Sensitivity.AGGRESSIVE]
        result = preset.model_dump_without_none()
        # suppress_tokens is None -> dropped on export, whisper default applies
        assert "suppress_tokens" not in result
        assert "length_penalty" not in result
        assert result["beam_size"] == 3

    def test_vad_export(self):
        preset = SILERO_VAD_PRESETS[Sensitivity.AGGRESSIVE]
        result = preset.model_dump_without_none()
        assert result["threshold"] == 0.18                     # v1.8.10-hf3
        assert result["min_speech_duration_ms"] == 30
        assert result["speech_pad_ms"] == 300
