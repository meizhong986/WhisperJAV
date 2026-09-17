"""
E2E validation for issue #196 fixes:
  - Fix 1: stream=True forced for local LLM provider
  - Fix 1b: supports_streaming=True required for CustomClient.enable_streaming
  - Fix 2: Metal/MPS backend correctly detected (not labeled as CUDA)
  - Fix 3: GUI --stream wired in both code paths, default True
  - Fix 4: max_tokens computed correctly and fits within context window
  - Fix 5: max_tokens flows into CustomClient request body
  - Fix 6: Cloud provider stream path unchanged (user-controlled)
"""

import os
import sys
import re
import ast
import inspect
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Full streaming chain — local_provider_config → opt_kwargs → CustomClient
# ─────────────────────────────────────────────────────────────────────────────

from PySubtrans.SettingsType import SettingsType
from PySubtrans.Providers.Clients.CustomClient import CustomClient

# Exact local_provider_config as built by cli.py (post-fix)
local_provider_config = {
    "pysubtrans_name": "Custom Server",
    "server_address": "http://127.0.0.1:55027",
    "endpoint": "/v1/chat/completions",
    "supports_conversation": True,
    "supports_system_messages": True,
    "max_tokens": 2392,
    "supports_streaming": True,  # THE FIX
}

# Exact opt_kwargs construction from core.py translate_subtitle()
opt_kwargs = {
    "provider": local_provider_config["pysubtrans_name"],
    "model": "local",
    "api_key": "",
    "target_language": "English",
    "prompt": "Translate these subtitles from japanese into english.",
    "preprocess_subtitles": True,
    "scene_threshold": 20.0,
    "max_batch_size": 10,
    "postprocess_translation": True,
    "stream_responses": True,   # stream=True forced for local in cli.py
}
# core.py pass-throughs
for key in ("server_address", "endpoint", "supports_conversation",
            "supports_system_messages", "max_tokens",
            "max_completion_tokens", "supports_streaming"):
    if key in local_provider_config:
        opt_kwargs[key] = local_provider_config[key]

# Simulate client_settings as PySubtrans produces them
client_settings = SettingsType(
    dict(opt_kwargs) | {"instructions": "Translate.", "temperature": 0.0, "timeout": 300}
)
client = CustomClient(client_settings)

check("1a enable_streaming=True through full chain",
      client.enable_streaming is True)
check("1b max_tokens=2392 reaches CustomClient",
      client.max_tokens == 2392,
      f"got {client.max_tokens}")
check("1c supports_streaming in client settings",
      client.settings.get_bool("supports_streaming", False) is True)
check("1d stream_responses in client settings",
      client.settings.get_bool("stream_responses", False) is True)

# Without supports_streaming — proves the bug would still exist without fix 1b
settings_no_support = SettingsType(
    dict(opt_kwargs, supports_streaming=False) |
    {"instructions": "Translate.", "temperature": 0.0, "timeout": 300}
)
client_broken = CustomClient(settings_no_support)
check("1e Without supports_streaming streaming is silently disabled (regression guard)",
      client_broken.enable_streaming is False,
      "confirms the bug that fix 1b addresses")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Metal/MPS detection in _parse_server_stderr
# ─────────────────────────────────────────────────────────────────────────────

from whisperjav.translate.local_backend import _parse_server_stderr, ServerDiagnostics


def write_stderr(content: str) -> str:
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.close()
    return f.name


METAL_STDERR = """
llama_model_loader: loaded meta data
llama_model_loader: - model has 33 layers
llm_load_tensors: offloading 33 repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
ggml_metal_init: allocating
ggml_metal_init: found device: Apple M1 Pro
llm_load_tensors: VRAM used: 5.20 GiB
"""

CUDA_FULL_STDERR = """
llama_model_loader: - model has 33 layers
llm_load_tensors: offloading 33 repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
ggml_cuda_init: found 1 CUDA devices
ggml_cuda_init: CUDA0: NVIDIA GeForce RTX 3060, 12288 MiB
llm_load_tensors: VRAM used: 5.20 GiB
"""

CUDA_PARTIAL_STDERR = """
llama_model_loader: - model has 33 layers
llm_load_tensors: offloading 20 repeating layers to GPU
llm_load_tensors: offloaded 20/33 layers to GPU
ggml_cuda_init: found 1 CUDA devices
llm_load_tensors: VRAM used: 3.10 GiB
"""

CPU_STDERR = """
llama_model_loader: - model has 33 layers
llm_load_tensors: using CPU backend
"""

# Edge case: GPU offloading found but no backend init message (fallback to CUDA)
AMBIGUOUS_STDERR = """
llama_model_loader: - model has 33 layers
llm_load_tensors: offloading 33 repeating layers to GPU
llm_load_tensors: offloaded 33/33 layers to GPU
"""

stderr_cases = [
    ("Metal full offload",    METAL_STDERR,     True,  False, 33, True),
    ("CUDA full offload",     CUDA_FULL_STDERR,  False, True,  33, True),
    ("CUDA partial offload",  CUDA_PARTIAL_STDERR, False, True, 20, True),
    ("CPU only",              CPU_STDERR,        False, False,  0, False),
    ("Ambiguous GPU (fallback=CUDA)", AMBIGUOUS_STDERR, False, True, 33, True),
]

for label, content, exp_metal, exp_cuda, exp_layers, exp_gpu in stderr_cases:
    p = write_stderr(content)
    d = _parse_server_stderr(p)
    os.unlink(p)
    ok = (
        d.using_metal == exp_metal
        and d.using_cuda == exp_cuda
        and d.gpu_layers_loaded == exp_layers
        and d.is_gpu_accelerated == exp_gpu
    )
    check(f"2  {label}",
          ok,
          f"metal={d.using_metal} cuda={d.using_cuda} layers={d.gpu_layers_loaded}")

# Status summary backend labels
dm = ServerDiagnostics(gpu_layers_loaded=33, total_layers=33,
                       using_metal=True, inference_speed_tps=16.4)
dc = ServerDiagnostics(gpu_layers_loaded=33, total_layers=33,
                       using_cuda=True, inference_speed_tps=50.0)
dp = ServerDiagnostics(gpu_layers_loaded=20, total_layers=33,
                       using_cuda=True, inference_speed_tps=30.0)
d0 = ServerDiagnostics()

check("2  Metal status says 'Metal/MPS'", "Metal/MPS" in dm.get_status_summary())
check("2  CUDA status says 'CUDA'",       "CUDA"      in dc.get_status_summary())
check("2  Partial CUDA says 'Partial'",   "Partial"   in dp.get_status_summary())
check("2  CPU status says 'CPU ONLY'",    "CPU ONLY"  in d0.get_status_summary())
check("2  Metal status does NOT say 'CUDA'",
      "CUDA" not in dm.get_status_summary(),
      dm.get_status_summary())


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: compute_max_output_tokens — values correct and fit in context window
# ─────────────────────────────────────────────────────────────────────────────

from whisperjav.translate.core import compute_max_output_tokens

overhead = 2500
input_per_line = 300

for n_ctx, batch_size in [(8192, 11), (16384, 27), (32768, 30), (8192, 5)]:
    t = compute_max_output_tokens(batch_size, n_ctx)
    avail = n_ctx - overhead - (batch_size * input_per_line)
    # Must be positive, fit in available space, and at least 512
    ok = 512 <= t <= avail
    check(f"3  max_tokens(n_ctx={n_ctx}, batch={batch_size})={t} in [512,{avail}]",
          ok,
          f"FAIL: {t} not in [512, {avail}]" if not ok else "")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: CLI source — local forced True, cloud user-controlled
# Read source as text to avoid argparse side-effects on import.
# ─────────────────────────────────────────────────────────────────────────────

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(REPO, "whisperjav", "translate", "cli.py"), encoding="utf-8") as f:
    cli_src = f.read()
with open(os.path.join(REPO, "whisperjav", "translate", "core.py"), encoding="utf-8") as f:
    core_src = f.read()

check("4a Local provider stream forced True in source",
      "stream=True," in cli_src and "Always stream for local LLM" in cli_src)
check("4b Cloud provider stream user-controlled in source",
      "stream=args.stream" in cli_src)
check("4c supports_streaming=True in local_provider_config",
      "'supports_streaming': True" in cli_src)
check("4d supports_streaming passed through in core.py",
      "supports_streaming" in core_src)
check("4e max_tokens passed through in core.py",
      "'max_tokens' in provider_config" in core_src)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: GUI --stream wiring — both code paths, default True
# ─────────────────────────────────────────────────────────────────────────────

with open(os.path.join(REPO, "whisperjav", "webview_gui", "api.py"), encoding="utf-8") as f:
    api_src = f.read()

check("5a GUI integrated pipeline has translate_stream default True",
      "config.get('translate_stream', True)" in api_src)
check("5b GUI standalone translate has stream default True",
      "options.get('stream', True)" in api_src)
check("5c GUI integrated pipeline appends --stream",
      'args += ["--stream"]' in api_src)
check("5d GUI standalone translate appends --stream",
      'args.append("--stream")' in api_src)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────

print()
print("=" * 72)
print("  E2E VALIDATION — Issue #196 Fixes")
print("=" * 72)
passed = failed = 0
for name, ok, detail in results:
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}")
    if detail and not ok:
        print(f"         {detail}")
    if ok:
        passed += 1
    else:
        failed += 1
print("=" * 72)
verdict = "ALL GOOD" if failed == 0 else f"{failed} FAILURE(S) — FIX REQUIRED"
print(f"  {passed} passed, {failed} failed  —  {verdict}")
print()

sys.exit(0 if failed == 0 else 1)
