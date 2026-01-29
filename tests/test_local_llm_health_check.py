#!/usr/bin/env python3
"""
End-to-end tests for local LLM server health check and diagnostics.

These tests verify that the health check system:
1. Correctly identifies GPU vs CPU execution
2. Measures and reports inference speed
3. Generates appropriate warnings for slow configurations
4. Reports diagnostics clearly to the user

The health check is CRITICAL - a health check that passes but doesn't
tell you if your system can handle the workload is useless.

Run with: pytest tests/test_local_llm_health_check.py -v
"""

import io
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from whisperjav.translate.local_backend import (
    ServerDiagnostics,
    _parse_server_stderr,
)


class TestServerDiagnostics:
    """Tests for ServerDiagnostics dataclass behavior."""

    def test_gpu_accelerated_when_layers_on_gpu(self):
        """GPU acceleration should be detected when layers are offloaded."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=True
        )
        assert diag.is_gpu_accelerated is True

    def test_not_gpu_accelerated_when_zero_layers(self):
        """Should correctly identify CPU-only execution."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=0,
            total_layers=35,
            using_cuda=False
        )
        assert diag.is_gpu_accelerated is False

    def test_not_gpu_accelerated_when_cuda_false(self):
        """Even with layer count, no CUDA means no GPU."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=False  # No CUDA
        )
        assert diag.is_gpu_accelerated is False

    def test_offload_percent_full_gpu(self):
        """100% offload when all layers on GPU."""
        diag = ServerDiagnostics(gpu_layers_loaded=35, total_layers=35)
        assert diag.gpu_offload_percent == 100.0

    def test_offload_percent_partial(self):
        """Partial offload percentage calculated correctly."""
        diag = ServerDiagnostics(gpu_layers_loaded=20, total_layers=40)
        assert diag.gpu_offload_percent == 50.0

    def test_offload_percent_zero_layers(self):
        """Zero division handled gracefully."""
        diag = ServerDiagnostics(gpu_layers_loaded=0, total_layers=0)
        assert diag.gpu_offload_percent == 0.0

    def test_batch_time_estimate_fast_gpu(self):
        """Fast GPU should estimate short batch time."""
        diag = ServerDiagnostics(inference_speed_tps=50.0)  # 50 tokens/sec
        # 1000 tokens / 50 tps = 20 seconds
        assert diag.estimated_batch_time_seconds == pytest.approx(20.0)

    def test_batch_time_estimate_slow_cpu(self):
        """Slow CPU should estimate long batch time."""
        diag = ServerDiagnostics(inference_speed_tps=2.0)  # 2 tokens/sec
        # 1000 tokens / 2 tps = 500 seconds
        assert diag.estimated_batch_time_seconds == pytest.approx(500.0)

    def test_batch_time_infinite_when_zero_speed(self):
        """Zero speed should return infinity (avoid division by zero)."""
        diag = ServerDiagnostics(inference_speed_tps=0.0)
        assert diag.estimated_batch_time_seconds == float('inf')


class TestServerDiagnosticsStatusSummary:
    """Tests for human-readable status summary generation."""

    def test_status_summary_full_gpu(self):
        """Status should show GPU layers and speed."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=True,
            vram_used_gb=5.2,
            inference_speed_tps=28.5
        )
        summary = diag.get_status_summary()

        assert "GPU" in summary
        assert "35 layers" in summary
        assert "CUDA" in summary
        assert "5.2GB VRAM" in summary
        assert "28.5 tokens/sec" in summary

    def test_status_summary_cpu_only(self):
        """CPU-only status should be clear and alarming."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=0,
            total_layers=35,
            using_cuda=False,
            inference_speed_tps=2.3
        )
        summary = diag.get_status_summary()

        assert "CPU ONLY" in summary
        assert "no GPU" in summary.lower() or "CPU ONLY" in summary
        assert "2.3 tokens/sec" in summary

    def test_status_summary_partial_gpu(self):
        """Partial GPU offload should show percentage."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=20,
            total_layers=40,
            using_cuda=True,
            inference_speed_tps=15.0
        )
        summary = diag.get_status_summary()

        assert "Partial" in summary or "20/40" in summary
        assert "50%" in summary


class TestServerDiagnosticsWarnings:
    """Tests for warning generation based on diagnostics."""

    def test_no_warnings_for_healthy_gpu_config(self):
        """Fast GPU config should generate no warnings."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=True,
            inference_speed_tps=30.0  # Fast
        )
        warnings = diag.get_warnings()
        assert len(warnings) == 0

    def test_warning_for_cpu_only(self):
        """CPU-only should warn about slow translation."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=0,
            total_layers=35,
            using_cuda=False,
            inference_speed_tps=3.0
        )
        warnings = diag.get_warnings()

        assert len(warnings) >= 1
        # Should mention CPU and slow
        combined = " ".join(warnings).lower()
        assert "cpu" in combined
        assert "slow" in combined

    def test_warning_for_very_slow_inference(self):
        """Very slow inference should warn about timeout."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=True,
            inference_speed_tps=2.0  # Very slow even with GPU
        )
        warnings = diag.get_warnings()

        assert len(warnings) >= 1
        combined = " ".join(warnings).lower()
        assert "slow" in combined or "timeout" in combined

    def test_warning_for_partial_gpu_offload(self):
        """Partial GPU offload should warn about mixed mode."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=10,
            total_layers=35,
            using_cuda=True,
            inference_speed_tps=8.0
        )
        warnings = diag.get_warnings()

        assert len(warnings) >= 1
        combined = " ".join(warnings).lower()
        # Should mention partial offload
        assert "partial" in combined or "%" in combined or "offload" in combined


class TestStderrParsing:
    """Tests for parsing llama-cpp-python server stderr output."""

    def _write_temp_stderr(self, content: str) -> str:
        """Helper to write content to temp file and return path."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(content)
            return f.name

    def test_parse_gpu_layers_offloaded(self):
        """Should extract GPU layer count from offload message."""
        stderr = """
llama_model_loader: loaded meta data with 24 key-value pairs
llm_load_tensors: offloading 35 repeating layers to GPU
llm_load_tensors: VRAM used: 5.23 GiB
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.gpu_layers_loaded == 35
            assert diag.using_cuda is True
        finally:
            os.unlink(path)

    def test_parse_vram_usage(self):
        """Should extract VRAM usage."""
        stderr = """
llm_load_tensors: VRAM used: 6.78 GiB
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.vram_used_gb == pytest.approx(6.78)
        finally:
            os.unlink(path)

    def test_parse_cuda_init(self):
        """Should detect CUDA initialization."""
        stderr = """
ggml_cuda_init: found 1 CUDA devices
Device 0: NVIDIA GeForce RTX 3080
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.using_cuda is True
        finally:
            os.unlink(path)

    def test_parse_cpu_only_mode(self):
        """Should detect CPU-only operation."""
        stderr = """
llama_model_loader: loaded meta data
llm_load_tensors: using CPU backend
no CUDA devices found
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.gpu_layers_loaded == 0
            assert diag.using_cuda is False
        finally:
            os.unlink(path)

    def test_parse_total_layers(self):
        """Should extract total layer count from model info."""
        stderr = """
llama_model_loader: - model has 35 layers
llm_load_tensors: offloading 35 repeating layers to GPU
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.total_layers == 35
        finally:
            os.unlink(path)

    def test_parse_empty_file(self):
        """Should handle empty stderr gracefully."""
        path = self._write_temp_stderr("")
        try:
            diag = _parse_server_stderr(path)
            # Should return defaults, not crash
            assert diag.gpu_layers_loaded == 0
            assert diag.using_cuda is False
        finally:
            os.unlink(path)

    def test_parse_nonexistent_file(self):
        """Should handle missing file gracefully."""
        diag = _parse_server_stderr("/nonexistent/path/to/file.log")
        # Should return defaults, not crash
        assert diag.gpu_layers_loaded == 0

    def test_parse_partial_gpu_offload(self):
        """Should detect partial GPU offload (layers/total format)."""
        stderr = """
llm_load_tensors: offloaded 20/35 layers to GPU
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.gpu_layers_loaded == 20
            assert diag.total_layers == 35
        finally:
            os.unlink(path)


class TestHealthCheckIntegration:
    """Integration tests for the full health check flow."""

    def test_health_check_returns_diagnostics_on_success(self):
        """Health check should return diagnostics when server is ready."""
        # This test verifies the return type contract
        from whisperjav.translate.local_backend import _wait_for_server
        import time

        # Mock urllib to simulate successful server response
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = b'{"usage": {"completion_tokens": 15}}'
        mock_response.__enter__ = MagicMock(return_value=mock_response)
        mock_response.__exit__ = MagicMock(return_value=False)

        # Track calls to simulate realistic timing
        call_count = [0]
        original_time = time.time

        def mock_urlopen(*args, **kwargs):
            call_count[0] += 1
            # First call is HTTP ready check, second is inference
            # Add small delay to simulate real request time
            time.sleep(0.1)  # 100ms delay
            return mock_response

        with patch('urllib.request.urlopen', side_effect=mock_urlopen):
            success, error, diagnostics = _wait_for_server(12345, max_wait=5)

            # Should succeed and return diagnostics
            assert success is True
            assert error is None
            assert diagnostics is not None
            assert isinstance(diagnostics, ServerDiagnostics)
            # With 15 tokens in 0.1s, speed should be ~150 t/s
            # But due to timing variations, just check it's positive
            assert diagnostics.inference_speed_tps > 0

    def test_health_check_returns_none_diagnostics_on_failure(self):
        """Health check should return None diagnostics on failure."""
        from whisperjav.translate.local_backend import _wait_for_server

        with patch('urllib.request.urlopen') as mock_urlopen:
            # Simulate connection refused
            import urllib.error
            mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

            success, error, diagnostics = _wait_for_server(12345, max_wait=1)

            assert success is False
            assert error is not None
            assert diagnostics is None


class TestConsoleOutputE2E:
    """E2E tests verifying console output to users."""

    def test_diagnostics_printed_to_console(self, capsys):
        """Verify diagnostics are printed in a visible format."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=35,
            total_layers=35,
            using_cuda=True,
            vram_used_gb=5.2,
            inference_speed_tps=28.5
        )

        # Simulate the console output from start_local_server
        print("")
        print("=" * 60)
        print("  LOCAL LLM SERVER STATUS")
        print("=" * 60)
        print(f"  {diag.get_status_summary()}")
        print("=" * 60)
        print("")

        captured = capsys.readouterr()

        # Verify key information is visible
        assert "LOCAL LLM SERVER STATUS" in captured.out
        assert "GPU" in captured.out
        assert "28.5 tokens/sec" in captured.out

    def test_warnings_printed_prominently(self, capsys):
        """Verify warnings are printed with emphasis."""
        diag = ServerDiagnostics(
            gpu_layers_loaded=0,
            total_layers=35,
            using_cuda=False,
            inference_speed_tps=2.3
        )

        warnings = diag.get_warnings()

        print("")
        print("=" * 60)
        print("  LOCAL LLM SERVER STATUS")
        print("=" * 60)
        print(f"  {diag.get_status_summary()}")

        if warnings:
            print("")
            print("  " + "!" * 56)
            print("  WARNINGS:")
            for warning in warnings:
                for line in warning.split('. '):
                    print(f"    - {line.strip()}")
            print("  " + "!" * 56)

        print("=" * 60)
        print("")

        captured = capsys.readouterr()

        # Verify warnings are prominently displayed
        assert "WARNINGS" in captured.out
        assert "CPU" in captured.out
        assert "slow" in captured.out.lower()
        # Verify visual emphasis
        assert "!" in captured.out


class TestRealWorldStderrSamples:
    """Tests using real-world stderr samples from llama-cpp-python."""

    def _write_temp_stderr(self, content: str) -> str:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            f.write(content)
            return f.name

    def test_rtx_3080_full_offload(self):
        """Real stderr from RTX 3080 with full GPU offload."""
        stderr = """
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model params     = 8.03 B
llm_load_tensors: ggml ctx size =    0.27 MiB
llm_load_tensors: offloading 33 repeating layers to GPU
llm_load_tensors: offloading non-repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
llm_load_tensors:        CPU buffer size =    88.81 MiB
llm_load_tensors:      CUDA0 buffer size =  4685.67 MiB
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3080, compute capability 8.6, VMM: yes
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.gpu_layers_loaded == 33
            assert diag.total_layers == 33
            assert diag.using_cuda is True
        finally:
            os.unlink(path)

    def test_cpu_fallback_no_cuda(self):
        """Real stderr when CUDA is not available."""
        stderr = """
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model params     = 8.03 B
llm_load_tensors: using CPU backend
llm_load_tensors:        CPU buffer size =  4774.00 MiB
"""
        path = self._write_temp_stderr(stderr)
        try:
            diag = _parse_server_stderr(path)
            assert diag.gpu_layers_loaded == 0
            assert diag.using_cuda is False
        finally:
            os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
