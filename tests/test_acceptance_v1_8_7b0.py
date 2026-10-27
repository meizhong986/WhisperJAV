"""
WhisperJAV v1.8.7b0 — Comprehensive E2E Acceptance Test Suite

Covers ALL 15 source files changed vs main:
  A  Version & Package Integrity
  B  #196 Translation chain (streaming, max_tokens, supports_streaming)
  C  #198 MPS device detection (TransformersASR)
  D  VAD padding defaults (silero_v6, ten, factory)
  E  CLI argument parsing (new v1.8.7 flags in main.py)
  F  --enhance-for-vad structural integrity (pipelines, orchestrator, exports)
  G  (removed in v1.10.0 with local_backend.py: llama-cpp local translation is gone)
  H  compute_max_output_tokens safety envelope
  I  GUI streaming wiring (api.py)
  J  Installer build consistency

Run: python -m pytest tests/test_acceptance_v1_8_7b0.py -v
"""

import os
import sys
import re
import inspect
import tempfile
import subprocess
import importlib

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


# ─────────────────────────────────────────────────────────────────────────────
# A  VERSION & PACKAGE INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestVersionIntegrity:
    """Version strings are consistent across all carrier files."""

    def test_version_is_1_8_7b0(self):
        from whisperjav import __version__
        assert __version__ == "1.8.7b0", f"Expected 1.8.7b0, got {__version__}"

    def test_version_display_is_beta(self):
        from whisperjav.__version__ import __version_display__
        assert "1.8.7" in __version_display__
        assert "beta" in __version_display__.lower() or "b0" in __version_display__

    def test_version_info_dict(self):
        from whisperjav import __version_info__
        assert __version_info__["major"] == 1
        assert __version_info__["minor"] == 8
        assert __version_info__["patch"] == 7

    def test_installer_version_file(self):
        version_file = os.path.join(REPO, "installer", "VERSION")
        content = open(version_file).read()
        assert "patch = 7" in content
        assert "prerelease = b0" in content
        assert "display_label = beta" in content

    def test_generated_wheel_exists(self):
        generated = os.path.join(REPO, "installer", "generated")
        wheels = [f for f in os.listdir(generated) if "1.8.7b0" in f and f.endswith(".whl")]
        assert len(wheels) >= 1, f"No 1.8.7b0 wheel in {generated}"

    def test_generated_construct_yaml_exists(self):
        generated = os.path.join(REPO, "installer", "generated")
        yamls = [f for f in os.listdir(generated) if "1.8.7b0" in f and f.endswith(".yaml")]
        assert len(yamls) >= 1


class TestCoreImports:
    """All changed modules import without error."""

    @pytest.mark.parametrize("module", [
        pytest.param("whisperjav.translate.cli", marks=pytest.mark.skip(reason="cli.py runs argparse at import time — tested via source inspection in TestIssue196TranslationChain")),
        "whisperjav.translate.core",
        "whisperjav.modules.transformers_asr",
        "whisperjav.modules.speech_segmentation.backends.silero_v6",
        "whisperjav.modules.speech_segmentation.backends.ten",
        "whisperjav.modules.speech_segmentation.factory",
        "whisperjav.modules.speech_enhancement",
        "whisperjav.modules.speech_enhancement.pipeline_helper",
        "whisperjav.modules.subtitle_pipeline.orchestrator",
        "whisperjav.pipelines.decoupled_pipeline",
        "whisperjav.pipelines.qwen_pipeline",
        "whisperjav.ensemble.pass_worker",
    ])
    def test_module_imports(self, module):
        mod = importlib.import_module(module)
        assert mod is not None

    def test_main_module_argparse_available(self):
        """main.py can be imported (argparse defined at module level)."""
        import whisperjav.main as m
        assert hasattr(m, "parse_arguments")
        assert hasattr(m, "main")


# ─────────────────────────────────────────────────────────────────────────────
# B  #196 TRANSLATION CHAIN
# ─────────────────────────────────────────────────────────────────────────────

class TestIssue196TranslationChain:
    """Full streaming chain: local_provider_config → opt_kwargs → CustomClient."""

    def setup_method(self):
        from PySubtrans.SettingsType import SettingsType
        from PySubtrans.Providers.Clients.CustomClient import CustomClient
        self.SettingsType = SettingsType
        self.CustomClient = CustomClient

    def _make_client(self, **overrides):
        base = {
            "pysubtrans_name": "Custom Server",
            "server_address": "http://127.0.0.1:55027",
            "endpoint": "/v1/chat/completions",
            "supports_conversation": True,
            "supports_system_messages": True,
            "max_tokens": 2392,
            "supports_streaming": True,
            "provider": "Custom Server",
            "model": "local",
            "api_key": "",
            "target_language": "English",
            "prompt": "Translate.",
            "stream_responses": True,
            "instructions": "Translate.",
            "temperature": 0.0,
            "timeout": 300,
        }
        base.update(overrides)
        return self.CustomClient(self.SettingsType(base))

    def test_enable_streaming_true_with_supports_streaming(self):
        client = self._make_client()
        assert client.enable_streaming is True

    def test_enable_streaming_false_without_supports_streaming(self):
        """Regression guard: streaming IS silently disabled without supports_streaming."""
        client = self._make_client(supports_streaming=False)
        assert client.enable_streaming is False

    def test_max_tokens_reaches_client(self):
        client = self._make_client(max_tokens=2392)
        assert client.max_tokens == 2392

    def test_supports_streaming_in_settings(self):
        client = self._make_client()
        assert client.settings.get_bool("supports_streaming", False) is True

    def test_stream_responses_in_settings(self):
        client = self._make_client()
        assert client.settings.get_bool("stream_responses", False) is True

    def test_stream_true_forced_in_cli_source(self):
        # v1.10.0: the llama-cpp branch is gone; Ollama still always streams.
        cli_path = os.path.join(REPO, "whisperjav", "translate", "cli.py")
        src = open(cli_path, encoding="utf-8").read()
        assert "stream=True," in src
        assert "Always stream for local models" in src

    def test_stream_cloud_user_controlled(self):
        cli_path = os.path.join(REPO, "whisperjav", "translate", "cli.py")
        src = open(cli_path, encoding="utf-8").read()
        assert "stream=args.stream" in src

    def test_supports_streaming_in_ollama_provider_config(self):
        # v1.10.0: was the llama-cpp provider config in cli.py; Ollama's is in providers.py.
        from whisperjav.translate.providers import PROVIDER_CONFIGS
        assert PROVIDER_CONFIGS["ollama"]["supports_streaming"] is True

    def test_supports_streaming_passthrough_in_core(self):
        core_path = os.path.join(REPO, "whisperjav", "translate", "core.py")
        src = open(core_path, encoding="utf-8").read()
        assert "supports_streaming" in src

    def test_max_tokens_passthrough_in_core(self):
        core_path = os.path.join(REPO, "whisperjav", "translate", "core.py")
        src = open(core_path, encoding="utf-8").read()
        assert "'max_tokens' in provider_config" in src

    def test_compute_max_output_tokens_exported(self):
        from whisperjav.translate.core import compute_max_output_tokens
        assert callable(compute_max_output_tokens)


# ─────────────────────────────────────────────────────────────────────────────
# C  #198 MPS DEVICE DETECTION (TransformersASR)
# ─────────────────────────────────────────────────────────────────────────────

class TestIssue198MPSDevice:
    """TransformersASR correctly handles MPS device selection."""

    def setup_method(self):
        from whisperjav.modules.transformers_asr import TransformersASR
        self.TransformersASR = TransformersASR

    def test_detect_device_handles_mps_request(self):
        """device_request='mps' branch exists in source."""
        src = inspect.getsource(self.TransformersASR._detect_device)
        assert "mps" in src

    def test_detect_device_auto_checks_mps_after_cuda(self):
        """In 'auto' path: cuda is checked first, then mps, then cpu.

        1.10.0 (design 22.3.1, owner W237 1): the auto path goes through device_detector's
        get_best_device(), which probes cuda, then mps, then cpu.
        """
        src = inspect.getsource(self.TransformersASR._detect_device)
        auto = src[:src.index('elif self.device_request == "cuda"')]
        assert "get_best_device()" in auto
        from whisperjav.utils import device_detector
        best = inspect.getsource(device_detector.get_best_device)
        assert best.index("_check_cuda_available()") < best.index("_check_mps_available()") < best.rindex('return "cpu"'), \
            "auto path must probe cuda → mps → cpu in that order"

    def test_detect_device_mps_fallback_to_cpu(self):
        """Explicit mps request falls back to cpu with warning if unavailable."""
        src = inspect.getsource(self.TransformersASR._detect_device)
        assert 'elif self.device_request == "mps"' in src
        assert "Falling back to CPU" in src

    def test_detect_device_else_passthrough(self):
        """Unknown device strings are passed through (e.g. 'cuda:1')."""
        src = inspect.getsource(self.TransformersASR._detect_device)
        assert "return self.device_request" in src

    def test_detect_dtype_mps_is_float16(self):
        """MPS gets float16 (bfloat16 is unsupported on MPS)."""
        src = inspect.getsource(self.TransformersASR._detect_dtype)
        # Find the mps block and assert float16 is returned
        assert 'device == "mps"' in src
        assert "float16" in src
        # Verify bfloat16 is NOT returned for mps
        mps_block_start = src.find('device == "mps"')
        mps_block = src[mps_block_start:mps_block_start + 80]
        assert "float16" in mps_block

    def test_detect_dtype_bfloat16_not_assigned_to_mps(self):
        """bfloat16 must never be selected for MPS."""
        src = inspect.getsource(self.TransformersASR._detect_dtype)
        # Find mps handling block
        mps_idx = src.find('device == "mps"')
        assert mps_idx != -1
        mps_return = src[mps_idx:mps_idx + 60]
        assert "bfloat16" not in mps_return

    def test_detect_device_cuda_available_uses_cuda(self):
        """When CUDA is available, auto selects cuda:0."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available — skipping live CUDA detection test")
        asr = self.TransformersASR.__new__(self.TransformersASR)
        asr.device_request = "auto"
        device = asr._detect_device()
        assert device == "cuda:0"

    def test_detect_device_explicit_cuda(self):
        """Explicit cuda request returns cuda:0 when CUDA is available."""
        import torch
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        asr = self.TransformersASR.__new__(self.TransformersASR)
        asr.device_request = "cuda"
        device = asr._detect_device()
        assert device == "cuda:0"

    def test_detect_dtype_cuda_is_bfloat16(self):
        """CUDA device gets bfloat16 under 'auto'."""
        asr = self.TransformersASR.__new__(self.TransformersASR)
        asr.dtype_request = "auto"
        dtype = asr._detect_dtype("cuda:0")
        import torch
        assert dtype == torch.bfloat16

    def test_detect_dtype_cpu_is_float32(self):
        """CPU device gets float32 under 'auto'."""
        asr = self.TransformersASR.__new__(self.TransformersASR)
        asr.dtype_request = "auto"
        dtype = asr._detect_dtype("cpu")
        import torch
        assert dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# D  VAD PADDING DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

class TestVADPaddingDefaults:
    """All three locations (backend, factory schema, docstring) agree on new defaults."""

    def test_silero_v6_speech_pad_ms_default_is_350(self):
        from whisperjav.modules.speech_segmentation.backends.silero_v6 import SileroV6SpeechSegmenter
        sig = inspect.signature(SileroV6SpeechSegmenter.__init__)
        assert sig.parameters["speech_pad_ms"].default == 350

    def test_ten_start_pad_ms_default_is_50(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        sig = inspect.signature(TenSpeechSegmenter.__init__)
        assert sig.parameters["start_pad_ms"].default == 50

    def test_ten_end_pad_ms_default_is_150(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        sig = inspect.signature(TenSpeechSegmenter.__init__)
        assert sig.parameters["end_pad_ms"].default == 150

    def test_factory_schema_silero_speech_pad_ms_is_350(self):
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        assert _PARAM_SCHEMAS["silero-v6.2"]["speech_pad_ms"][1] == 350

    def test_factory_schema_ten_start_pad_ms_is_50(self):
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        assert _PARAM_SCHEMAS["ten"]["start_pad_ms"][1] == 50

    def test_factory_schema_ten_end_pad_ms_is_150(self):
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        assert _PARAM_SCHEMAS["ten"]["end_pad_ms"][1] == 150

    def test_backend_and_factory_silero_agree(self):
        from whisperjav.modules.speech_segmentation.backends.silero_v6 import SileroV6SpeechSegmenter
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        backend_default = inspect.signature(SileroV6SpeechSegmenter.__init__).parameters["speech_pad_ms"].default
        factory_default = _PARAM_SCHEMAS["silero-v6.2"]["speech_pad_ms"][1]
        assert backend_default == factory_default, \
            f"Silero backend ({backend_default}) and factory schema ({factory_default}) disagree"

    def test_backend_and_factory_ten_start_agree(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        backend_default = inspect.signature(TenSpeechSegmenter.__init__).parameters["start_pad_ms"].default
        factory_default = _PARAM_SCHEMAS["ten"]["start_pad_ms"][1]
        assert backend_default == factory_default

    def test_backend_and_factory_ten_end_agree(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        from whisperjav.modules.speech_segmentation.factory import _PARAM_SCHEMAS
        backend_default = inspect.signature(TenSpeechSegmenter.__init__).parameters["end_pad_ms"].default
        factory_default = _PARAM_SCHEMAS["ten"]["end_pad_ms"][1]
        assert backend_default == factory_default

    def test_silero_v6_not_old_250(self):
        from whisperjav.modules.speech_segmentation.backends.silero_v6 import SileroV6SpeechSegmenter
        sig = inspect.signature(SileroV6SpeechSegmenter.__init__)
        assert sig.parameters["speech_pad_ms"].default != 250, "Old v1.8.6 value — not updated"

    def test_ten_start_not_old_0(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        sig = inspect.signature(TenSpeechSegmenter.__init__)
        assert sig.parameters["start_pad_ms"].default != 0, "Old v1.8.6 value — not updated"

    def test_ten_end_not_old_100(self):
        from whisperjav.modules.speech_segmentation.backends.ten import TenSpeechSegmenter
        sig = inspect.signature(TenSpeechSegmenter.__init__)
        assert sig.parameters["end_pad_ms"].default != 100, "Old v1.8.6 value — not updated"


# ─────────────────────────────────────────────────────────────────────────────
# E  CLI ARGUMENT PARSING — NEW v1.8.7 FLAGS
# ─────────────────────────────────────────────────────────────────────────────

class TestCLIArgumentParsing:
    """All new CLI flags added in v1.8.7 are recognised by argparse."""

    @pytest.fixture(scope="class")
    def parser(self):
        import whisperjav.main as m
        return m.parse_arguments.__wrapped__ if hasattr(m.parse_arguments, "__wrapped__") else m.parse_arguments()

    def _get_args(self, argv):
        import whisperjav.main as m
        # parse_arguments() is the function, call it with monkeypatched sys.argv
        old_argv = sys.argv
        sys.argv = ["whisperjav"] + argv
        try:
            args = m.parse_arguments()
        finally:
            sys.argv = old_argv
        return args

    def test_speech_pad_ms_global(self):
        args = self._get_args(["dummy.mp4", "--speech-pad-ms", "400"])
        assert args.speech_pad_ms == 400

    def test_speech_pad_ms_default_none(self):
        args = self._get_args(["dummy.mp4"])
        assert args.speech_pad_ms is None

    def test_vad_threshold_global(self):
        args = self._get_args(["dummy.mp4", "--vad-threshold", "0.3"])
        assert abs(args.vad_threshold - 0.3) < 1e-9

    def test_pass1_vad_threshold(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass1-vad-threshold", "0.25"])
        assert abs(args.pass1_vad_threshold - 0.25) < 1e-9

    def test_pass1_speech_pad_ms(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass1-speech-pad-ms", "200"])
        assert args.pass1_speech_pad_ms == 200

    def test_pass2_vad_threshold(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass2-vad-threshold", "0.4"])
        assert abs(args.pass2_vad_threshold - 0.4) < 1e-9

    def test_pass2_speech_pad_ms(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass2-speech-pad-ms", "300"])
        assert args.pass2_speech_pad_ms == 300

    def test_enhance_for_vad_flag(self):
        args = self._get_args(["dummy.mp4", "--enhance-for-vad"])
        assert args.enhance_for_vad is True

    def test_enhance_for_vad_default_false(self):
        args = self._get_args(["dummy.mp4"])
        assert args.enhance_for_vad is False

    def test_pass1_enhance_for_vad_flag(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass1-enhance-for-vad"])
        assert args.pass1_enhance_for_vad is True

    def test_pass2_enhance_for_vad_flag(self):
        args = self._get_args(["dummy.mp4", "--ensemble", "--pass2-enhance-for-vad"])
        assert args.pass2_enhance_for_vad is True

    def test_pass1_vad_threshold_default_none(self):
        args = self._get_args(["dummy.mp4", "--ensemble"])
        assert args.pass1_vad_threshold is None

    def test_pass2_speech_pad_ms_default_none(self):
        args = self._get_args(["dummy.mp4", "--ensemble"])
        assert args.pass2_speech_pad_ms is None

    def test_vad_threshold_clamped_in_help(self):
        """--vad-threshold described as float 0.0-1.0."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--help"],
            capture_output=True, text=True, timeout=15
        )
        assert "--vad-threshold" in result.stdout + result.stderr
        assert "--speech-pad-ms" in result.stdout + result.stderr

    def test_stream_flag_in_help(self):
        """--stream must appear in main.py --help (CLI entry point)."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--help"],
            capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, "--help should exit 0"
        assert "--stream" in result.stdout + result.stderr, \
            "--stream not found in main.py --help — flag missing from argparse"

    def test_stream_flag_accepted_by_main(self):
        """--stream must not cause 'unrecognized arguments' error."""
        result = subprocess.run(
            [sys.executable, "-m", "whisperjav.main", "--stream", "--help"],
            capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, \
            f"--stream caused exit {result.returncode}: {result.stderr[:300]}"
        assert "unrecognized" not in result.stderr.lower()

    def test_stream_default_false(self):
        """--stream defaults to False when not passed."""
        args = self._get_args(["dummy.mp4"])
        assert args.stream is False

    def test_stream_true_when_passed(self):
        """--stream sets args.stream to True."""
        args = self._get_args(["dummy.mp4", "--stream"])
        assert args.stream is True


# ─────────────────────────────────────────────────────────────────────────────
# F  --ENHANCE-FOR-VAD STRUCTURAL INTEGRITY
# ─────────────────────────────────────────────────────────────────────────────

class TestEnhanceForVADStructure:
    """Dual-track mode is wired correctly through all pipeline layers."""

    def test_decoupled_pipeline_accepts_enhance_for_vad(self):
        from whisperjav.pipelines.decoupled_pipeline import DecoupledPipeline
        sig = inspect.signature(DecoupledPipeline.__init__)
        assert "enhance_for_vad" in sig.parameters
        assert sig.parameters["enhance_for_vad"].default is False

    def test_qwen_pipeline_accepts_enhance_for_vad(self):
        from whisperjav.pipelines.qwen_pipeline import QwenPipeline
        sig = inspect.signature(QwenPipeline.__init__)
        assert "enhance_for_vad" in sig.parameters
        assert sig.parameters["enhance_for_vad"].default is False

    def test_orchestrator_process_scenes_accepts_vad_audio_paths(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
        sig = inspect.signature(DecoupledSubtitlePipeline.process_scenes)
        assert "vad_audio_paths" in sig.parameters
        assert sig.parameters["vad_audio_paths"].default is None

    def test_orchestrator_run_pass_accepts_vad_audio_paths(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
        sig = inspect.signature(DecoupledSubtitlePipeline._run_pass)
        assert "vad_audio_paths" in sig.parameters

    def test_orchestrator_stepdown_accepts_vad_audio_paths(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
        sig = inspect.signature(DecoupledSubtitlePipeline._run_stepdown_pass)
        assert "vad_audio_paths" in sig.parameters

    def test_orchestrator_step1_accepts_vad_audio_paths(self):
        from whisperjav.modules.subtitle_pipeline.orchestrator import DecoupledSubtitlePipeline
        sig = inspect.signature(DecoupledSubtitlePipeline._step1_frame_and_slice)
        assert "vad_audio_paths" in sig.parameters

    def test_resample_scenes_exported_from_speech_enhancement(self):
        from whisperjav.modules.speech_enhancement import resample_scenes
        assert callable(resample_scenes)

    def test_resample_scenes_signature(self):
        from whisperjav.modules.speech_enhancement import resample_scenes
        sig = inspect.signature(resample_scenes)
        assert "scene_paths" in sig.parameters
        assert "temp_dir" in sig.parameters

    def test_decoupled_pipeline_imports_resample_scenes(self):
        src_path = os.path.join(REPO, "whisperjav", "pipelines", "decoupled_pipeline.py")
        src = open(src_path, encoding="utf-8").read()
        assert "resample_scenes" in src
        assert "is_passthrough_backend" in src

    def test_qwen_pipeline_imports_resample_scenes(self):
        src_path = os.path.join(REPO, "whisperjav", "pipelines", "qwen_pipeline.py")
        src = open(src_path, encoding="utf-8").read()
        assert "resample_scenes" in src

    def test_pass_worker_enhance_for_vad_guard_for_non_qwen(self):
        """Pass worker warns and guards enhance_for_vad for non-qwen pipelines."""
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._build_pipeline)
        assert 'enhance_for_vad' in src
        assert 'pipeline_name not in' in src

    def test_pass_worker_propagates_enhance_for_vad_to_qwen(self):
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._build_pipeline)
        assert '"enhance_for_vad": pass_config.get("enhance_for_vad"' in src

    def test_pass_worker_cli_vad_threshold_override(self):
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._build_pipeline)
        assert 'vad_threshold' in src
        assert 'user_segmenter_overrides' in src

    def test_pass_worker_cli_speech_pad_override(self):
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._build_pipeline)
        assert 'speech_pad_ms' in src

    def test_apply_gui_overrides_handles_vad_threshold(self):
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._apply_gui_overrides)
        assert 'vad_threshold' in src
        assert 'speech_segmenter' in src

    def test_apply_gui_overrides_handles_speech_pad_ms(self):
        import whisperjav.ensemble.pass_worker as pw
        src = inspect.getsource(pw._apply_gui_overrides)
        assert 'speech_pad_ms' in src

    def test_decoupled_pipeline_dual_track_vad_scene_paths(self):
        """Phase 3 dual-track logic present in decoupled_pipeline.py."""
        src_path = os.path.join(REPO, "whisperjav", "pipelines", "decoupled_pipeline.py")
        src = open(src_path, encoding="utf-8").read()
        assert "_vad_scene_paths" in src
        assert "_orch_vad_paths" in src
        assert "original_16k_paths" in src

    def test_qwen_pipeline_dual_track_vad_scene_paths(self):
        """Phase 3 dual-track logic present in qwen_pipeline.py."""
        src_path = os.path.join(REPO, "whisperjav", "pipelines", "qwen_pipeline.py")
        src = open(src_path, encoding="utf-8").read()
        assert "_vad_scene_paths" in src
        assert "_orch_vad_paths" in src


# ─────────────────────────────────────────────────────────────────────────────
# H  compute_max_output_tokens SAFETY ENVELOPE
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMaxOutputTokens:
    """Token budget formula is correct and safe across all realistic configurations."""

    def setup_method(self):
        from whisperjav.translate.core import compute_max_output_tokens, cap_batch_size_for_context
        self.compute = compute_max_output_tokens
        self.cap = cap_batch_size_for_context

    @pytest.mark.parametrize("n_ctx,batch_size", [
        (8192, 11),   # gemma-9b 8K
        (8192, 5),    # gemma-9b 8K small batch
        (16384, 27),  # qwen 16K
        (32768, 30),  # qwen 32K
        (4096, 1),    # 4K minimal
        (131072, 30), # llama 128K context
    ])
    def test_result_within_safe_range(self, n_ctx, batch_size):
        overhead = 2500
        input_per_line = 300
        t = self.compute(batch_size, n_ctx)
        avail = n_ctx - overhead - (batch_size * input_per_line)
        assert t >= 512, f"n_ctx={n_ctx} batch={batch_size}: result {t} < 512 minimum"
        assert t <= avail or t == 512, \
            f"n_ctx={n_ctx} batch={batch_size}: result {t} exceeds available {avail}"

    def test_minimum_is_512(self):
        """Even in degenerate cases, result is never below 512."""
        result = self.compute(30, 4096)  # context too small for batch
        assert result == 512

    def test_correct_for_8k_batch11(self):
        """Known-good value: 8K ctx, batch=11 → 2392."""
        result = self.compute(11, 8192)
        assert result == 2392

    def test_correct_for_16k_batch27(self):
        """Known-good value: 16K ctx, batch=27 → 5784."""
        result = self.compute(27, 16384)
        assert result == 5784

    def test_result_increases_with_n_ctx(self):
        """Larger context window → larger output budget (same batch)."""
        t8k = self.compute(10, 8192)
        t16k = self.compute(10, 16384)
        t32k = self.compute(10, 32768)
        assert t8k <= t16k <= t32k

    def test_result_decreases_with_batch_size(self):
        """In the available-capped region (large batch), more lines = smaller budget."""
        # For 8K ctx the available cap kicks in at ~batch 10.
        # compute(5,8192)=2200 uses expected*2 cap; compute(10,8192)=2692 uses available.
        # Above the crossover, available strictly decreases with batch size.
        t10 = self.compute(10, 8192)
        t15 = self.compute(15, 8192)
        t20 = self.compute(20, 8192)
        assert t10 >= t15 >= t20, f"Expected 2692>=1192>=512, got {t10}>={t15}>={t20}"

    def test_cap_batch_prevents_degenerate_compute(self):
        """cap_batch_size prevents compute_max_output_tokens from seeing oversized batches."""
        capped = self.cap(30, 4096)
        # cap uses max(5, (4096-2500)//500) = max(5,3) = 5; then min(30,5) = 5
        assert capped == 5, f"Expected cap=5 for 4K ctx, got {capped}"
        result = self.compute(capped, 4096)
        # compute(5,4096): available=4096-2500-1500=96, expected*2=2200 → min(96,2200)=96 → max(512,96)=512
        assert result >= 512, f"Result {result} is below minimum guard of 512"

    def test_chinese_cjk_output_safe(self):
        """Chinese output (≤ English tokens/line) is within formula envelope."""
        # For Chinese target, output_per_line could be ~80 tokens (vs 120 for English)
        # Our formula uses 120 (English) — conservative, always >= Chinese real cost
        # Verify the formula doesn't over-allocate (result must fit in available tokens)
        overhead = 2500
        input_per_line = 300
        for batch_size, n_ctx in [(10, 8192), (17, 8192), (27, 16384)]:
            t = self.compute(batch_size, n_ctx)
            avail = n_ctx - overhead - (batch_size * input_per_line)
            assert t <= avail or t == 512, \
                f"Chinese output case exceeds context: batch={batch_size} n_ctx={n_ctx}"


# ─────────────────────────────────────────────────────────────────────────────
# I  GUI STREAMING WIRING (api.py)
# ─────────────────────────────────────────────────────────────────────────────

class TestGUIStreamingWiring:
    """Both GUI paths wire --stream with default True."""

    def setup_method(self):
        api_path = os.path.join(REPO, "whisperjav", "webview_gui", "api.py")
        self.src = open(api_path, encoding="utf-8").read()

    def test_integrated_pipeline_translate_stream_default_true(self):
        assert "config.get('translate_stream', True)" in self.src

    def test_integrated_pipeline_appends_stream_flag(self):
        assert 'args += ["--stream"]' in self.src

    def test_standalone_translate_stream_default_true(self):
        assert "options.get('stream', True)" in self.src

    def test_standalone_translate_appends_stream_flag(self):
        assert 'args.append("--stream")' in self.src

    def test_issue_196_comment_present_in_api(self):
        """The PR comment explaining streaming rationale is in api.py."""
        assert "#196" in self.src or "read timeout" in self.src.lower()


# ─────────────────────────────────────────────────────────────────────────────
# J  INSTALLER BUILD CONSISTENCY
# ─────────────────────────────────────────────────────────────────────────────

class TestInstallerConsistency:
    """Installer generated files are consistent with version 1.8.7b0."""

    def setup_method(self):
        self.generated = os.path.join(REPO, "installer", "generated")

    def test_construct_yaml_version_string(self):
        yamls = [f for f in os.listdir(self.generated) if "1.8.7b0" in f and f.endswith(".yaml")]
        assert yamls, "No 1.8.7b0 yaml found"
        content = open(os.path.join(self.generated, yamls[0])).read()
        assert "1.8.7b0" in content

    def test_wheel_version_in_filename(self):
        wheels = [f for f in os.listdir(self.generated) if f.endswith(".whl")]
        names = [w for w in wheels if "1.8.7b0" in w]
        assert names, f"No 1.8.7b0 wheel in {wheels}"

    def test_no_old_version_in_core_module(self):
        from whisperjav import __version__
        assert "1.8.6" not in __version__, "Still on 1.8.6 — version not bumped"

    def test_installer_validation_script_passes(self):
        """Run the generated validation script."""
        scripts = [f for f in os.listdir(self.generated)
                   if f.startswith("validate_installer") and "1.8.7b0" in f and f.endswith(".py")]
        assert scripts, "No validation script found"
        result = subprocess.run(
            [sys.executable, scripts[0]],
            capture_output=True, text=True, timeout=30,
            cwd=self.generated,
        )
        assert result.returncode == 0, \
            f"Installer validation failed:\n{result.stdout}\n{result.stderr}"


# ─────────────────────────────────────────────────────────────────────────────
# K  VAD THRESHOLD ROUTING IN main.py
# ─────────────────────────────────────────────────────────────────────────────

class TestVADThresholdRouting:
    """main.py routes --vad-threshold and --speech-pad-ms to the right config keys."""

    def setup_method(self):
        self.src = open(os.path.join(REPO, "whisperjav", "main.py"), encoding="utf-8").read()

    def test_vad_threshold_routes_to_vad_params(self):
        assert 'resolved_config["params"]["vad"]["threshold"]' in self.src

    def test_vad_threshold_routes_to_speech_segmenter_params(self):
        assert 'resolved_config["params"]["speech_segmenter"]["threshold"]' in self.src

    def test_speech_pad_ms_routes_to_vad_params(self):
        assert 'resolved_config["params"]["vad"]["speech_pad_ms"]' in self.src

    def test_speech_pad_ms_routes_to_speech_segmenter_params(self):
        assert 'resolved_config["params"]["speech_segmenter"]["speech_pad_ms"]' in self.src

    def test_pass1_vad_threshold_flows_to_pass_config(self):
        assert "pass1_vad_threshold" in self.src

    def test_pass1_speech_pad_ms_flows_to_pass_config(self):
        assert "pass1_speech_pad_ms" in self.src

    def test_pass2_vad_threshold_flows_to_pass_config(self):
        assert "pass2_vad_threshold" in self.src

    def test_pass2_speech_pad_ms_flows_to_pass_config(self):
        assert "pass2_speech_pad_ms" in self.src

    def test_enhance_for_vad_flows_to_pipeline_kwargs(self):
        assert "enhance_for_vad" in self.src
        assert "_pipeline_kwargs" in self.src

    def test_enhance_for_vad_flows_to_pass_config(self):
        assert "pass1_enhance_for_vad" in self.src
        assert "pass2_enhance_for_vad" in self.src
