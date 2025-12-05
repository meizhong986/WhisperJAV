"""
Comprehensive GUI Custom Parameters Simulation Test Suite.

This test suite simulates the GUI's two-pass ensemble custom parameters
functionality and validates the impact on each pipeline type.

Test Flow:
    GUI Config → _build_twopass_args() → CLI Args → main.py parsing →
    pass_config → pass_worker._build_pipeline() → apply_custom_params() →
    resolved_config → Pipeline initialization

Pipelines Tested:
    - balanced (faster-whisper backend)
    - fast (stable-ts backend)
    - faster (stable-ts backend)
    - fidelity (openai-whisper backend)
    - transformers (HuggingFace backend)
"""

import json
import pytest
from copy import deepcopy
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

from whisperjav.ensemble.pass_worker import (
    apply_custom_params,
    prepare_transformers_params,
    get_valid_provider_params,
    DECODER_PARAMS,
    VAD_PARAMS,
    FEATURE_PARAMS,
    MODEL_PARAMS,
    PIPELINE_BACKENDS,
)
from whisperjav.config.legacy import resolve_legacy_pipeline


# =============================================================================
# GUI Simulation Helpers
# =============================================================================

def simulate_gui_config(
    pass1_pipeline: str = "balanced",
    pass1_sensitivity: str = "balanced",
    pass1_customized: bool = False,
    pass1_params: Dict[str, Any] = None,
    pass1_is_transformers: bool = False,
    pass2_enabled: bool = True,
    pass2_pipeline: str = "fidelity",
    pass2_sensitivity: str = "balanced",
    pass2_customized: bool = False,
    pass2_params: Dict[str, Any] = None,
    pass2_is_transformers: bool = False,
    merge_strategy: str = "smart_merge",
    subs_language: str = "native",
) -> Dict[str, Any]:
    """
    Simulate a GUI configuration object as it would be sent to _build_twopass_args.

    This matches the structure used by the frontend JavaScript.
    """
    return {
        "inputs": ["/path/to/video.mp4"],
        "pass1": {
            "pipeline": pass1_pipeline,
            "sensitivity": pass1_sensitivity,
            "customized": pass1_customized,
            "params": pass1_params,
            "isTransformers": pass1_is_transformers,
        },
        "pass2": {
            "enabled": pass2_enabled,
            "pipeline": pass2_pipeline,
            "sensitivity": pass2_sensitivity,
            "customized": pass2_customized,
            "params": pass2_params,
            "isTransformers": pass2_is_transformers,
        },
        "merge_strategy": merge_strategy,
        "output_dir": "/output",
        "subs_language": subs_language,
        "source_language": "japanese",
        "temp_dir": "",
        "keep_temp": False,
        "debug": False,
    }


def build_twopass_args(config: Dict[str, Any]) -> List[str]:
    """
    Simulates api.py:_build_twopass_args() without requiring the full API class.

    This is a standalone version for testing.
    """
    args = []

    # Inputs
    inputs = config.get('inputs', [])
    if not inputs:
        raise ValueError("Please add at least one file or folder.")
    args.extend(inputs)

    # Enable ensemble mode
    args.append("--ensemble")

    # Pass 1 configuration
    pass1 = config.get('pass1', {})
    args += ["--pass1-pipeline", pass1.get('pipeline', 'balanced')]

    if pass1.get('isTransformers'):
        if pass1.get('customized') and pass1.get('params'):
            args += ["--pass1-hf-params", json.dumps(pass1['params'])]
    else:
        args += ["--pass1-sensitivity", pass1.get('sensitivity', 'balanced')]
        if pass1.get('customized') and pass1.get('params'):
            args += ["--pass1-params", json.dumps(pass1['params'])]

    # Pass 2 configuration
    pass2 = config.get('pass2', {})
    if pass2.get('enabled', False):
        args += ["--pass2-pipeline", pass2.get('pipeline', 'fidelity')]

        if pass2.get('isTransformers'):
            if pass2.get('customized') and pass2.get('params'):
                args += ["--pass2-hf-params", json.dumps(pass2['params'])]
        else:
            args += ["--pass2-sensitivity", pass2.get('sensitivity', 'balanced')]
            if pass2.get('customized') and pass2.get('params'):
                args += ["--pass2-params", json.dumps(pass2['params'])]

    # Merge strategy
    args += ["--merge-strategy", config.get('merge_strategy', 'smart_merge')]

    # Output directory
    args += ["--output-dir", config.get('output_dir', './output')]

    # Subtitle language mode
    args += ["--subs-language", config.get('subs_language', 'native')]

    # Source language
    args += ["--language", config.get('source_language', 'japanese')]

    return args


def parse_cli_args_to_pass_config(args: List[str]) -> Dict[str, Any]:
    """
    Parse CLI args back into pass_config structure (simulates main.py parsing).

    Returns the pass1_config and pass2_config dictionaries.
    """
    pass1_config = {
        'pipeline': 'balanced',
        'sensitivity': 'balanced',
        'params': None,
        'hf_params': None,
    }
    pass2_config = None

    i = 0
    while i < len(args):
        arg = args[i]

        if arg == "--pass1-pipeline":
            pass1_config['pipeline'] = args[i + 1]
            i += 2
        elif arg == "--pass1-sensitivity":
            pass1_config['sensitivity'] = args[i + 1]
            i += 2
        elif arg == "--pass1-params":
            pass1_config['params'] = json.loads(args[i + 1])
            i += 2
        elif arg == "--pass1-hf-params":
            pass1_config['hf_params'] = json.loads(args[i + 1])
            i += 2
        elif arg == "--pass2-pipeline":
            if pass2_config is None:
                pass2_config = {
                    'pipeline': 'fidelity',
                    'sensitivity': 'balanced',
                    'params': None,
                    'hf_params': None,
                }
            pass2_config['pipeline'] = args[i + 1]
            i += 2
        elif arg == "--pass2-sensitivity":
            if pass2_config is None:
                pass2_config = {
                    'pipeline': 'fidelity',
                    'sensitivity': 'balanced',
                    'params': None,
                    'hf_params': None,
                }
            pass2_config['sensitivity'] = args[i + 1]
            i += 2
        elif arg == "--pass2-params":
            if pass2_config is None:
                pass2_config = {
                    'pipeline': 'fidelity',
                    'sensitivity': 'balanced',
                    'params': None,
                    'hf_params': None,
                }
            pass2_config['params'] = json.loads(args[i + 1])
            i += 2
        elif arg == "--pass2-hf-params":
            if pass2_config is None:
                pass2_config = {
                    'pipeline': 'fidelity',
                    'sensitivity': 'balanced',
                    'params': None,
                    'hf_params': None,
                }
            pass2_config['hf_params'] = json.loads(args[i + 1])
            i += 2
        else:
            i += 1

    return {
        'pass1': pass1_config,
        'pass2': pass2_config,
    }


def simulate_full_param_flow(
    gui_config: Dict[str, Any],
    pass_number: int = 1,
) -> Dict[str, Any]:
    """
    Simulate the complete parameter flow from GUI config to resolved config.

    Returns the final resolved_config that would be passed to the pipeline.
    """
    # Step 1: GUI config → CLI args
    args = build_twopass_args(gui_config)

    # Step 2: CLI args → pass_config
    pass_configs = parse_cli_args_to_pass_config(args)

    pass_config = pass_configs['pass1'] if pass_number == 1 else pass_configs['pass2']
    if pass_config is None:
        return None

    pipeline_name = pass_config['pipeline']

    # Step 3: Transformers vs Legacy handling
    if pipeline_name == "transformers":
        # Transformers uses prepare_transformers_params
        hf_params = prepare_transformers_params(pass_config)
        return {
            'type': 'transformers',
            'hf_params': hf_params,
        }

    # Step 4: Legacy pipeline resolution
    resolved_config = resolve_legacy_pipeline(
        pipeline_name=pipeline_name,
        sensitivity=pass_config.get("sensitivity", "balanced"),
        task="transcribe",
        overrides=pass_config.get("overrides"),
    )

    # Step 5: Apply custom params
    if pass_config.get("params"):
        unknown_params = apply_custom_params(
            resolved_config=resolved_config,
            custom_params=pass_config["params"],
            pass_number=pass_number,
            pipeline_name=pipeline_name,
        )
        resolved_config['_unknown_params'] = unknown_params

    return {
        'type': 'legacy',
        'resolved_config': resolved_config,
    }


# =============================================================================
# Test: GUI to CLI Args Conversion
# =============================================================================

class TestGUIToCLIArgsConversion:
    """Test that GUI config correctly converts to CLI arguments."""

    def test_basic_balanced_fidelity_config(self):
        """Standard balanced + fidelity setup."""
        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="aggressive",
            pass2_pipeline="fidelity",
            pass2_sensitivity="conservative",
        )

        args = build_twopass_args(gui_config)

        assert "--ensemble" in args
        assert "--pass1-pipeline" in args
        assert args[args.index("--pass1-pipeline") + 1] == "balanced"
        assert "--pass1-sensitivity" in args
        assert args[args.index("--pass1-sensitivity") + 1] == "aggressive"
        assert "--pass2-pipeline" in args
        assert args[args.index("--pass2-pipeline") + 1] == "fidelity"
        assert "--pass2-sensitivity" in args
        assert args[args.index("--pass2-sensitivity") + 1] == "conservative"

    def test_custom_params_serialized_as_json(self):
        """Custom params should be serialized as JSON."""
        custom_params = {
            "beam_size": 10,
            "temperature": [0.0, 0.2],
            "model_name": "large-v3",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
        )

        args = build_twopass_args(gui_config)

        assert "--pass1-params" in args
        params_json = args[args.index("--pass1-params") + 1]
        parsed = json.loads(params_json)
        assert parsed["beam_size"] == 10
        assert parsed["temperature"] == [0.0, 0.2]
        assert parsed["model_name"] == "large-v3"

    def test_transformers_uses_hf_params(self):
        """Transformers pipeline should use --pass1-hf-params, not --pass1-params."""
        hf_params = {
            "model_id": "openai/whisper-large-v3",
            "scene": "auditok",
            "batch_size": 8,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=True,
            pass1_params=hf_params,
        )

        args = build_twopass_args(gui_config)

        assert "--pass1-hf-params" in args
        assert "--pass1-params" not in args
        # Sensitivity should not be included for transformers
        assert "--pass1-sensitivity" not in args

    def test_pass2_disabled_excludes_pass2_args(self):
        """If pass2 is disabled, no pass2 args should be included."""
        gui_config = simulate_gui_config(
            pass2_enabled=False,
        )

        args = build_twopass_args(gui_config)

        assert "--pass2-pipeline" not in args
        assert "--pass2-sensitivity" not in args


# =============================================================================
# Test: Balanced Pipeline Custom Parameters
# =============================================================================

class TestBalancedPipelineCustomParams:
    """Test custom parameters for balanced pipeline (faster-whisper backend)."""

    def test_decoder_params_applied(self):
        """Decoder params should be applied to decoder section."""
        custom_params = {
            "beam_size": 10,
            "best_of": 3,
            "patience": 1.5,
            "task": "transcribe",
            "language": "ja",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        assert result['type'] == 'legacy'
        config = result['resolved_config']
        assert config["params"]["decoder"]["beam_size"] == 10
        assert config["params"]["decoder"]["best_of"] == 3
        assert config["params"]["decoder"]["patience"] == 1.5

    def test_vad_params_applied(self):
        """VAD params should be applied to vad section."""
        custom_params = {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "speech_pad_ms": 400,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["vad"]["threshold"] == 0.3
        assert config["params"]["vad"]["min_speech_duration_ms"] == 100
        assert config["params"]["vad"]["speech_pad_ms"] == 400

    def test_faster_whisper_provider_params_applied(self):
        """Faster-whisper specific params should be applied to provider."""
        custom_params = {
            "repetition_penalty": 1.8,
            "no_repeat_ngram_size": 3,
            "hallucination_silence_threshold": 2.5,
            "temperature": [0.0, 0.2, 0.4],
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["provider"]["repetition_penalty"] == 1.8
        assert config["params"]["provider"]["no_repeat_ngram_size"] == 3
        assert config["params"]["provider"]["hallucination_silence_threshold"] == 2.5
        assert config["params"]["provider"]["temperature"] == [0.0, 0.2, 0.4]

    def test_model_params_applied(self):
        """Model params should be applied to model section."""
        custom_params = {
            "model_name": "large-v3",
            "device": "cuda:1",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["model"]["model_name"] == "large-v3"
        assert config["model"]["device"] == "cuda:1"

    def test_feature_params_discarded(self):
        """Feature params should be discarded and not reach provider."""
        custom_params = {
            "scene_detection_method": "auditok",
            "beam_size": 5,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert "scene_detection_method" not in config["params"]["provider"]
        # beam_size should still be applied
        assert config["params"]["decoder"]["beam_size"] == 5

    def test_unknown_params_rejected(self):
        """Unknown params should be rejected, not added to provider."""
        custom_params = {
            "totally_fake_param": 123,
            "beam_size": 5,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert "totally_fake_param" in config.get("_unknown_params", [])
        assert "totally_fake_param" not in config["params"]["provider"]


# =============================================================================
# Test: Fidelity Pipeline Custom Parameters
# =============================================================================

class TestFidelityPipelineCustomParams:
    """Test custom parameters for fidelity pipeline (openai-whisper backend)."""

    def test_openai_whisper_specific_params(self):
        """OpenAI Whisper specific params should be applied."""
        custom_params = {
            "fp16": False,
            "verbose": True,
            "hallucination_silence_threshold": 3.0,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="fidelity",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["provider"]["fp16"] is False
        assert config["params"]["provider"]["verbose"] is True
        assert config["params"]["provider"]["hallucination_silence_threshold"] == 3.0

    def test_faster_whisper_params_rejected_for_fidelity(self):
        """Faster-whisper specific params should be rejected for fidelity."""
        custom_params = {
            "repetition_penalty": 1.5,  # faster-whisper only
            "no_repeat_ngram_size": 2,  # faster-whisper only
            "beam_size": 5,  # valid decoder param
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="fidelity",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        # faster-whisper params should be rejected
        assert "repetition_penalty" in config.get("_unknown_params", [])
        assert "no_repeat_ngram_size" in config.get("_unknown_params", [])
        # beam_size is valid for all backends
        assert config["params"]["decoder"]["beam_size"] == 5


# =============================================================================
# Test: Fast/Faster Pipeline Custom Parameters
# =============================================================================

class TestFastPipelineCustomParams:
    """Test custom parameters for fast/faster pipelines (stable-ts backend)."""

    def test_stable_ts_specific_params(self):
        """Stable-ts specific params should be applied."""
        custom_params = {
            "regroup": False,
            "suppress_silence": True,
            "gap_padding": "...",
            "vad_threshold": 0.35,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="fast",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["provider"]["regroup"] is False
        assert config["params"]["provider"]["suppress_silence"] is True
        assert config["params"]["provider"]["gap_padding"] == "..."
        assert config["params"]["provider"]["vad_threshold"] == 0.35

    def test_faster_pipeline_same_backend(self):
        """Faster pipeline uses same stable-ts backend as fast."""
        custom_params = {
            "regroup": True,
            "suppress_word_ts": False,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="faster",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["provider"]["regroup"] is True
        assert config["params"]["provider"]["suppress_word_ts"] is False


# =============================================================================
# Test: Transformers Pipeline Custom Parameters
# =============================================================================

class TestTransformersPipelineCustomParams:
    """Test custom parameters for transformers pipeline (HuggingFace backend)."""

    def test_hf_params_mapping(self):
        """HF params should be correctly mapped."""
        hf_params = {
            "model_id": "openai/whisper-large-v3",
            "chunk_length_s": 20,
            "batch_size": 8,
            "beam_size": 10,
            "temperature": 0.2,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=True,
            pass1_params=hf_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        assert result['type'] == 'transformers'
        params = result['hf_params']
        assert params["hf_model_id"] == "openai/whisper-large-v3"
        assert params["hf_chunk_length"] == 20
        assert params["hf_batch_size"] == 8
        assert params["hf_beam_size"] == 10
        assert params["hf_temperature"] == 0.2

    def test_scene_param_mapping(self):
        """scene param should map to hf_scene."""
        hf_params = {
            "scene": "auditok",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=True,
            pass1_params=hf_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        params = result['hf_params']
        assert params["hf_scene"] == "auditok"

    def test_legacy_scene_detection_method_mapping(self):
        """Legacy scene_detection_method should map to hf_scene."""
        hf_params = {
            "scene_detection_method": "silero",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=True,
            pass1_params=hf_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        params = result['hf_params']
        assert params["hf_scene"] == "silero"

    def test_transformers_defaults_applied(self):
        """Default HF params should be applied when not customized."""
        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=False,
            pass1_params=None,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        params = result['hf_params']
        # Defaults should be applied
        assert params["hf_model_id"] == "kotoba-tech/kotoba-whisper-v2.2"
        assert params["hf_batch_size"] == 16
        assert params["hf_scene"] == "none"


# =============================================================================
# Test: Two-Pass Custom Parameters
# =============================================================================

class TestTwoPassCustomParams:
    """Test custom parameters in two-pass scenarios."""

    def test_different_params_per_pass(self):
        """Each pass can have different custom params."""
        pass1_params = {
            "beam_size": 5,
            "model_name": "turbo",
        }
        pass2_params = {
            "beam_size": 10,
            "model_name": "large-v2",
            "fp16": False,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=pass1_params,
            pass2_enabled=True,
            pass2_pipeline="fidelity",
            pass2_customized=True,
            pass2_params=pass2_params,
        )

        # Test pass 1
        result1 = simulate_full_param_flow(gui_config, pass_number=1)
        config1 = result1['resolved_config']
        assert config1["params"]["decoder"]["beam_size"] == 5
        assert config1["model"]["model_name"] == "turbo"

        # Test pass 2
        result2 = simulate_full_param_flow(gui_config, pass_number=2)
        config2 = result2['resolved_config']
        assert config2["params"]["decoder"]["beam_size"] == 10
        assert config2["model"]["model_name"] == "large-v2"
        assert config2["params"]["provider"]["fp16"] is False

    def test_mixed_transformers_and_legacy(self):
        """Pass 1 can be transformers while Pass 2 is legacy."""
        pass1_hf_params = {
            "model_id": "openai/whisper-large-v3",
            "scene": "auditok",
        }
        pass2_params = {
            "beam_size": 8,
            "model_name": "large-v2",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="transformers",
            pass1_is_transformers=True,
            pass1_customized=True,
            pass1_params=pass1_hf_params,
            pass2_enabled=True,
            pass2_pipeline="fidelity",
            pass2_is_transformers=False,
            pass2_customized=True,
            pass2_params=pass2_params,
        )

        # Test pass 1 (transformers)
        result1 = simulate_full_param_flow(gui_config, pass_number=1)
        assert result1['type'] == 'transformers'
        assert result1['hf_params']["hf_model_id"] == "openai/whisper-large-v3"
        assert result1['hf_params']["hf_scene"] == "auditok"

        # Test pass 2 (legacy)
        result2 = simulate_full_param_flow(gui_config, pass_number=2)
        assert result2['type'] == 'legacy'
        config2 = result2['resolved_config']
        assert config2["params"]["decoder"]["beam_size"] == 8
        assert config2["model"]["model_name"] == "large-v2"


# =============================================================================
# Test: Sensitivity Presets with Custom Overrides
# =============================================================================

class TestSensitivityWithCustomOverrides:
    """Test that sensitivity presets can be overridden with custom params."""

    def test_aggressive_sensitivity_with_custom_override(self):
        """Custom params should override sensitivity defaults."""
        # Aggressive sensitivity typically sets beam_size=2, best_of=1
        # Custom params should override these
        custom_params = {
            "beam_size": 10,  # Override aggressive default
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="aggressive",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        # Custom override should win
        assert config["params"]["decoder"]["beam_size"] == 10

    def test_conservative_sensitivity_with_vad_override(self):
        """VAD params can be overridden on conservative sensitivity."""
        custom_params = {
            "threshold": 0.6,  # Higher threshold
            "min_speech_duration_ms": 300,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="conservative",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["params"]["vad"]["threshold"] == 0.6
        assert config["params"]["vad"]["min_speech_duration_ms"] == 300


# =============================================================================
# Test: Comprehensive Real-World Scenarios
# =============================================================================

class TestRealWorldScenarios:
    """Test real-world parameter combinations users might configure."""

    def test_user_scenario_high_quality_transcription(self):
        """User wants maximum quality: large model, high beam size."""
        custom_params = {
            "model_name": "large-v3",
            "beam_size": 10,
            "best_of": 5,
            "temperature": [0.0],
            "no_speech_threshold": 0.3,
            "compression_ratio_threshold": 2.0,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="conservative",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["model"]["model_name"] == "large-v3"
        assert config["params"]["decoder"]["beam_size"] == 10
        assert config["params"]["decoder"]["best_of"] == 5
        assert config["params"]["provider"]["temperature"] == [0.0]
        assert config["params"]["provider"]["no_speech_threshold"] == 0.3

    def test_user_scenario_fast_draft(self):
        """User wants fast draft: turbo model, aggressive settings."""
        custom_params = {
            "model_name": "turbo",
            "beam_size": 1,
            "best_of": 1,
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_sensitivity="aggressive",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        assert config["model"]["model_name"] == "turbo"
        assert config["params"]["decoder"]["beam_size"] == 1

    def test_user_scenario_full_params_mix(self):
        """User configures params from all categories."""
        custom_params = {
            # Model
            "model_name": "large-v2",
            "device": "cuda:0",
            # Decoder
            "beam_size": 8,
            "task": "transcribe",
            "language": "ja",
            "patience": 1.5,
            # Provider (faster-whisper)
            "temperature": [0.0, 0.1, 0.2],
            "repetition_penalty": 1.6,
            "hallucination_silence_threshold": 2.5,
            # VAD
            "threshold": 0.35,
            "min_speech_duration_ms": 200,
            "speech_pad_ms": 400,
            # Feature params (should be discarded)
            "scene_detection_method": "auditok",
        }

        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']

        # Verify all categories
        assert config["model"]["model_name"] == "large-v2"
        assert config["model"]["device"] == "cuda:0"

        assert config["params"]["decoder"]["beam_size"] == 8
        assert config["params"]["decoder"]["task"] == "transcribe"
        assert config["params"]["decoder"]["language"] == "ja"
        assert config["params"]["decoder"]["patience"] == 1.5

        assert config["params"]["provider"]["temperature"] == [0.0, 0.1, 0.2]
        assert config["params"]["provider"]["repetition_penalty"] == 1.6
        assert config["params"]["provider"]["hallucination_silence_threshold"] == 2.5

        assert config["params"]["vad"]["threshold"] == 0.35
        assert config["params"]["vad"]["min_speech_duration_ms"] == 200
        assert config["params"]["vad"]["speech_pad_ms"] == 400

        # Feature param should NOT be in provider
        assert "scene_detection_method" not in config["params"]["provider"]


# =============================================================================
# Test: Error Handling and Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_custom_params(self):
        """Empty custom params dict should work (use sensitivity defaults)."""
        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=True,
            pass1_params={},  # Empty dict
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        # Should still resolve successfully with defaults
        assert result['type'] == 'legacy'
        assert 'resolved_config' in result

    def test_none_custom_params(self):
        """None custom params should work (use sensitivity defaults)."""
        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass1_customized=False,
            pass1_params=None,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        assert result['type'] == 'legacy'
        assert '_unknown_params' not in result['resolved_config']

    def test_pass2_disabled_returns_none(self):
        """Pass 2 disabled should return None for pass 2 config."""
        gui_config = simulate_gui_config(
            pass1_pipeline="balanced",
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=2)

        assert result is None

    def test_all_pipelines_can_be_resolved(self):
        """Verify all pipeline types can be resolved without errors."""
        pipelines = ["balanced", "fast", "faster", "fidelity"]

        for pipeline in pipelines:
            gui_config = simulate_gui_config(
                pass1_pipeline=pipeline,
                pass1_customized=True,
                pass1_params={"beam_size": 5},
                pass2_enabled=False,
            )

            result = simulate_full_param_flow(gui_config, pass_number=1)

            assert result is not None
            assert result['type'] == 'legacy'
            assert result['resolved_config']['params']['decoder']['beam_size'] == 5


# =============================================================================
# Test: Pipeline Backend Validation
# =============================================================================

class TestPipelineBackendValidation:
    """Test that params are validated against correct backend."""

    @pytest.mark.parametrize("pipeline,valid_param", [
        ("balanced", "repetition_penalty"),
        ("fidelity", "fp16"),
        ("fast", "regroup"),
        ("faster", "suppress_silence"),
    ])
    def test_backend_specific_param_accepted(self, pipeline, valid_param):
        """Backend-specific params should be accepted for correct pipeline."""
        custom_params = {valid_param: True}

        gui_config = simulate_gui_config(
            pass1_pipeline=pipeline,
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        unknown = config.get('_unknown_params', [])
        assert valid_param not in unknown

    @pytest.mark.parametrize("pipeline,invalid_param", [
        ("fidelity", "repetition_penalty"),  # faster-whisper only
        ("balanced", "verbose"),  # openai-whisper only
        ("fast", "fp16"),  # openai-whisper only
    ])
    def test_wrong_backend_param_rejected(self, pipeline, invalid_param):
        """Params for wrong backend should be rejected."""
        custom_params = {invalid_param: True}

        gui_config = simulate_gui_config(
            pass1_pipeline=pipeline,
            pass1_customized=True,
            pass1_params=custom_params,
            pass2_enabled=False,
        )

        result = simulate_full_param_flow(gui_config, pass_number=1)

        config = result['resolved_config']
        unknown = config.get('_unknown_params', [])
        assert invalid_param in unknown
