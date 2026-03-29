"""
Ollama Translation Diagnostic Test Suite
=========================================

Standalone tests for diagnosing Ollama translation failures.
Requires Ollama running locally with at least one model pulled.

Usage:
    # Run all diagnostics against a specific model
    OLLAMA_TEST_MODEL="hf.co/mradermacher/Ninja-v1-NSFW-128k-GGUF:Q6_K" pytest tests/test_ollama_translation_diagnostic.py -s -v

    # Run against all locally installed models
    OLLAMA_TEST_MODE=all-local pytest tests/test_ollama_translation_diagnostic.py -s -v

    # Run against all curated models (skips models not pulled)
    OLLAMA_TEST_MODE=curated pytest tests/test_ollama_translation_diagnostic.py -s -v

    # Quick smoke test with first available model
    pytest tests/test_ollama_translation_diagnostic.py -s -v

    # Windows (set env var first):
    set OLLAMA_TEST_MODEL=hf.co/mradermacher/Ninja-v1-NSFW-128k-GGUF:Q6_K
    pytest tests/test_ollama_translation_diagnostic.py -s -v

The tests capture and print raw model responses so you can see exactly
what the model returns vs what PySubtrans expects.
"""

import json
import os
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

import pytest

# ---------------------------------------------------------------------------
# Windows console encoding fix — force UTF-8 for Japanese text output
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Ollama HTTP helpers (zero dependency — mirrors OllamaManager pattern)
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def _ollama_get(path: str, timeout: int = 5) -> dict:
    url = f"{OLLAMA_BASE_URL}{path}"
    req = urllib.request.Request(url, method="GET")
    resp = urllib.request.urlopen(req, timeout=timeout)
    body = resp.read().decode("utf-8", errors="replace")
    resp.close()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"_raw": body}


def _ollama_post(path: str, data: dict, timeout: int = 10) -> dict:
    url = f"{OLLAMA_BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=timeout)
    resp_body = resp.read().decode("utf-8", errors="replace")
    resp.close()
    try:
        return json.loads(resp_body)
    except json.JSONDecodeError:
        return {"_raw": resp_body}


def _ollama_chat(model: str, messages: list[dict], temperature: float = 0.3,
                 max_tokens: int = 2048, num_ctx: int = None,
                 timeout: int = 120) -> dict:
    """Send a chat request via Ollama's native /api/chat (non-streaming).

    Returns the full Ollama response dict including eval_count, eval_duration, etc.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    if num_ctx is not None:
        payload["options"]["num_ctx"] = num_ctx
    if max_tokens:
        payload["options"]["num_predict"] = max_tokens

    url = f"{OLLAMA_BASE_URL}/api/chat"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=timeout)
    resp_body = resp.read().decode("utf-8", errors="replace")
    resp.close()
    return json.loads(resp_body)


def _ollama_chat_openai(model: str, messages: list[dict], temperature: float = 0.3,
                        max_tokens: int = 2048, timeout: int = 120) -> dict:
    """Send a chat request via Ollama's OpenAI-compatible /v1/chat/completions.

    This is the EXACT path WhisperJAV uses via PySubtrans CustomClient.
    Note: this endpoint does NOT accept num_ctx — that's the gap.
    """
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    url = f"{OLLAMA_BASE_URL}/v1/chat/completions"
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    resp = urllib.request.urlopen(req, timeout=timeout)
    resp_body = resp.read().decode("utf-8", errors="replace")
    resp.close()
    return json.loads(resp_body)


# ---------------------------------------------------------------------------
# Test data — minimal JAV subtitle batch
# ---------------------------------------------------------------------------

INSTRUCTION_TEXT = Path(__file__).parent.parent / "whisperjav" / "translate" / "defaults" / "pornify.txt"

MINI_BATCH_3_LINES = textwrap.dedent("""\
    Translate these subtitles from japanese into english.

    #1
    Original>
    お願い、もっと触って
    Translation>

    #2
    Original>
    気持ちいい、止めないで
    Translation>

    #3
    Original>
    すごい、こんなの初めて
    Translation>
""")

REALISTIC_BATCH_6_LINES = textwrap.dedent("""\
    Translate these subtitles from japanese into english.

    #1
    Original>
    はじめまして、田中と申します
    Translation>

    #2
    Original>
    今日はお時間いただきありがとうございます
    Translation>

    #3
    Original>
    少し緊張していますけど
    Translation>

    #4
    Original>
    お願い、もっと触って
    Translation>

    #5
    Original>
    気持ちいい、止めないで
    Translation>

    #6
    Original>
    すごい、こんなの初めて
    Translation>
""")


def _load_instructions() -> str:
    if INSTRUCTION_TEXT.exists():
        text = INSTRUCTION_TEXT.read_text(encoding="utf-8")
        # Extract just the ### instructions section
        parts = text.split("### instructions")
        if len(parts) > 1:
            body = parts[1].split("### retry_instructions")[0].strip()
            return body
    return ""


def _build_messages(batch_prompt: str, system_instructions: str = None) -> list[dict]:
    msgs = []
    if system_instructions:
        msgs.append({"role": "system", "content": system_instructions})
    msgs.append({"role": "user", "content": batch_prompt})
    return msgs


# ---------------------------------------------------------------------------
# PySubtrans parser import (use the real thing)
# ---------------------------------------------------------------------------

def _parse_with_pysubtrans(response_text: str) -> tuple[bool, int, str]:
    """Try parsing a response with PySubtrans's actual TranslationParser.

    Returns (success, match_count, detail_message).
    """
    try:
        from PySubtrans.TranslationParser import TranslationParser, default_pattern, fallback_patterns
        import regex

        patterns = [regex.compile(p, regex.MULTILINE) for p in [default_pattern] + fallback_patterns]

        text = f"{response_text}\n\n"
        for i, pattern in enumerate(patterns):
            matches = list(pattern.finditer(text))
            if matches:
                bodies = []
                for m in matches:
                    num = m.group("number")
                    body = m.group("body") if "body" in m.groupdict() else ""
                    bodies.append(f"  #{num}: {(body or '').strip()[:80]}")
                detail = f"Pattern {i} matched {len(matches)} lines:\n" + "\n".join(bodies)
                return True, len(matches), detail

        return False, 0, "No patterns matched"

    except ImportError:
        # Fallback: basic regex check
        import re
        matches = re.findall(r"#(\d+)\s+.*?Translation[>:]\s*\n(.+?)(?=\n#\d|\Z)", response_text, re.DOTALL)
        if matches:
            return True, len(matches), f"Basic regex matched {len(matches)} lines"
        return False, 0, "No matches (basic regex fallback)"


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

def _ollama_available() -> bool:
    try:
        _ollama_get("/", timeout=2)
        return True
    except Exception:
        return False


def _get_local_models() -> list[str]:
    try:
        resp = _ollama_get("/api/tags", timeout=5)
        return [m["name"] for m in resp.get("models", [])]
    except Exception:
        return []


def _load_curated_models() -> list[dict]:
    config_path = Path(__file__).parent.parent / "whisperjav" / "config" / "ollama_models.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return []


def _get_test_models() -> list[str]:
    """Resolve which models to test based on environment variables.

    OLLAMA_TEST_MODEL=<model>      — test a specific model
    OLLAMA_TEST_MODE=all-local     — test all locally installed models
    OLLAMA_TEST_MODE=curated       — test all curated models (skips unpulled)
    (default)                      — first available local model
    """
    specific = os.environ.get("OLLAMA_TEST_MODEL")
    if specific:
        return [specific]

    mode = os.environ.get("OLLAMA_TEST_MODE", "").lower()
    local_models = _get_local_models()

    if mode == "all-local":
        return local_models

    if mode == "curated":
        curated = _load_curated_models()
        curated_names = [m["model"] for m in curated]
        available = [m for m in curated_names if any(m in lm or lm in m for lm in local_models)]
        return available

    # Default: first available model
    if local_models:
        return [local_models[0]]
    return []


@pytest.fixture(scope="session")
def ollama_server():
    if not _ollama_available():
        pytest.skip("Ollama server not running")
    return OLLAMA_BASE_URL


# ---------------------------------------------------------------------------
# TEST 1: Model Metadata Inspection
# ---------------------------------------------------------------------------

class TestModelMetadata:
    """Query /api/show for each model and report template, context, parameters."""

    def test_model_info(self, ollama_server):
        """Inspect model metadata: chat template, context window, parameters."""
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        for model in models:
            print(f"\n{'='*70}")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            try:
                info = _ollama_post("/api/show", {"name": model}, timeout=10)
            except Exception as e:
                print(f"  ERROR: Cannot query model info: {e}")
                print(f"  Model may not be pulled. Run: ollama pull {model}")
                continue

            # Template
            template = info.get("template", "")
            print(f"\n  CHAT TEMPLATE:")
            if template:
                # Show first 500 chars
                print(f"    {template[:500]}")
                if len(template) > 500:
                    print(f"    ... ({len(template)} chars total)")
            else:
                print(f"    *** NO TEMPLATE — model may not support chat format ***")
                print(f"    This means /v1/chat/completions may produce garbled output.")

            # Parameters
            params = info.get("parameters", "")
            print(f"\n  PARAMETERS:")
            if params:
                for line in params.strip().splitlines():
                    print(f"    {line.strip()}")
            else:
                print(f"    (none embedded)")

            # Model info — context length
            model_info = info.get("model_info", {})
            ctx_keys = ["context_length", "num_ctx", "llama.context_length",
                        "general.context_length"]
            found_ctx = None
            for key in ctx_keys:
                val = model_info.get(key)
                if val:
                    found_ctx = (key, val)
                    break

            print(f"\n  CONTEXT WINDOW:")
            if found_ctx:
                print(f"    {found_ctx[0]} = {found_ctx[1]}")
            else:
                print(f"    *** NOT DETECTED — OllamaManager will fall back to 8192 ***")
                print(f"    Available model_info keys: {list(model_info.keys())[:20]}")

            # System message support
            print(f"\n  SYSTEM MESSAGE SUPPORT:")
            if template and ("system" in template.lower() or "<<SYS>>" in template):
                print(f"    Likely supported (template contains 'system' role)")
            elif template:
                print(f"    Unclear — template does not obviously reference system role")
            else:
                print(f"    Unknown — no template available")

            # Model family / architecture
            arch = model_info.get("general.architecture", "unknown")
            family = model_info.get("general.name", "unknown")
            print(f"\n  ARCHITECTURE: {arch}")
            print(f"  FAMILY: {family}")

            # Quantization
            quant = model_info.get("general.file_type", "unknown")
            print(f"  QUANTIZATION: {quant}")

            print()


# ---------------------------------------------------------------------------
# TEST 2: Raw Response Capture — The #1 Diagnostic
# ---------------------------------------------------------------------------

class TestRawResponse:
    """Send a real translation batch and capture the raw response."""

    def test_mini_batch_native_api(self, ollama_server):
        """Send 3-line batch via /api/chat (native Ollama) and show raw response."""
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        instructions = _load_instructions()

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: 3-line batch via /api/chat (native)")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            messages = _build_messages(MINI_BATCH_3_LINES, instructions)

            t0 = time.time()
            try:
                resp = _ollama_chat(model, messages, temperature=0.3,
                                    max_tokens=1024, num_ctx=8192, timeout=180)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.time() - t0

            content = resp.get("message", {}).get("content", "")
            eval_count = resp.get("eval_count", 0)
            eval_duration = resp.get("eval_duration", 0)
            prompt_eval_count = resp.get("prompt_eval_count", 0)
            tps = (eval_count / (eval_duration / 1e9)) if eval_duration else 0

            print(f"\n  TIMING: {elapsed:.1f}s | {eval_count} tokens | {tps:.1f} tok/s")
            print(f"  PROMPT TOKENS: {prompt_eval_count}")
            print(f"\n  RAW RESPONSE ({len(content)} chars):")
            print(f"  {'-'*60}")
            print(textwrap.indent(content, "  "))
            print(f"  {'-'*60}")

            # Parse with PySubtrans
            success, count, detail = _parse_with_pysubtrans(content)
            print(f"\n  PYSUBTRANS PARSE: {'PASS' if success else 'FAIL'}")
            print(f"  {detail}")

            if not success:
                print(f"\n  *** FORMAT COMPLIANCE FAILURE ***")
                print(f"  The model's response does not match any of PySubtrans's 7 regex patterns.")
                print(f"  Expected format:")
                print(f"    #1")
                print(f"    Original>")
                print(f"    お願い、もっと触って")
                print(f"    Translation>")
                print(f"    Please, touch me more")

            print()

    def test_mini_batch_instructions_in_user_msg(self, ollama_server):
        """Send 3-line batch with instructions embedded in user message (no system role).

        This simulates the fix for models with bare templates ({{ .Prompt }}).
        When supports_system_messages=False, PySubtrans wraps instructions
        into the user message so they survive any template.
        """
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        instructions = _load_instructions()

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: 3-line batch with instructions IN user message (fix simulation)")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            # Embed instructions in user message — no system message at all
            separator = "--------"
            wrapped = f"{separator}\nSYSTEM\n{separator}\n{instructions}\n{separator}"
            combined = f"{wrapped}\n\n{MINI_BATCH_3_LINES}"
            messages = [{"role": "user", "content": combined}]

            t0 = time.time()
            try:
                resp = _ollama_chat(model, messages, temperature=0.3,
                                    max_tokens=1024, num_ctx=8192, timeout=180)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.time() - t0

            content = resp.get("message", {}).get("content", "")
            eval_count = resp.get("eval_count", 0)
            eval_duration = resp.get("eval_duration", 0)
            prompt_eval_count = resp.get("prompt_eval_count", 0)
            tps = (eval_count / (eval_duration / 1e9)) if eval_duration else 0

            print(f"\n  TIMING: {elapsed:.1f}s | {eval_count} tokens | {tps:.1f} tok/s")
            print(f"  PROMPT TOKENS: {prompt_eval_count}")
            print(f"\n  RAW RESPONSE ({len(content)} chars):")
            print(f"  {'-'*60}")
            print(textwrap.indent(content, "  "))
            print(f"  {'-'*60}")

            success, count, detail = _parse_with_pysubtrans(content)
            print(f"\n  PYSUBTRANS PARSE: {'PASS' if success else 'FAIL'}")
            print(f"  {detail}")

            if success and count == 3:
                print(f"\n  *** FIX VERIFIED: all 3 lines matched ***")
            elif success:
                print(f"\n  Partial: {count}/3 lines matched")

            print()

    def test_mini_batch_openai_api(self, ollama_server):
        """Send 3-line batch via /v1/chat/completions (PySubtrans path) and show raw response.

        This is the EXACT code path WhisperJAV uses. Note that num_ctx
        CANNOT be passed through this endpoint — Ollama uses its default.
        """
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        instructions = _load_instructions()

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: 3-line batch via /v1/chat/completions (WhisperJAV path)")
            print(f"MODEL: {model}")
            print(f"NOTE: num_ctx NOT sent — Ollama uses model default")
            print(f"{'='*70}")

            messages = _build_messages(MINI_BATCH_3_LINES, instructions)

            t0 = time.time()
            try:
                resp = _ollama_chat_openai(model, messages, temperature=0.3,
                                           max_tokens=1024, timeout=180)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.time() - t0

            choices = resp.get("choices", [])
            content = choices[0]["message"]["content"] if choices else ""
            usage = resp.get("usage", {})

            print(f"\n  TIMING: {elapsed:.1f}s")
            print(f"  USAGE: prompt={usage.get('prompt_tokens', '?')}, "
                  f"completion={usage.get('completion_tokens', '?')}, "
                  f"total={usage.get('total_tokens', '?')}")
            print(f"\n  RAW RESPONSE ({len(content)} chars):")
            print(f"  {'-'*60}")
            print(textwrap.indent(content, "  "))
            print(f"  {'-'*60}")

            # Parse with PySubtrans
            success, count, detail = _parse_with_pysubtrans(content)
            print(f"\n  PYSUBTRANS PARSE: {'PASS' if success else 'FAIL'}")
            print(f"  {detail}")

            print()


# ---------------------------------------------------------------------------
# TEST 3: num_ctx Gap — Compare Native vs OpenAI endpoint
# ---------------------------------------------------------------------------

class TestNumCtxGap:
    """Demonstrate the num_ctx passthrough gap between native and OpenAI endpoints."""

    def test_num_ctx_native_vs_openai(self, ollama_server):
        """Compare prompt_eval_count between native (with num_ctx) and OpenAI (without).

        If the numbers differ significantly, Ollama is using different context
        windows for the same model depending on the endpoint — confirming the gap.
        """
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        instructions = _load_instructions()
        messages = _build_messages(MINI_BATCH_3_LINES, instructions)

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: num_ctx gap comparison")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            # Native with explicit num_ctx=8192
            try:
                native_resp = _ollama_chat(model, messages, temperature=0.1,
                                           max_tokens=512, num_ctx=8192, timeout=180)
                native_prompt_tokens = native_resp.get("prompt_eval_count", "?")
                native_content = native_resp.get("message", {}).get("content", "")
            except Exception as e:
                print(f"  Native API error: {e}")
                continue

            # OpenAI compat WITHOUT num_ctx (WhisperJAV's actual path)
            try:
                openai_resp = _ollama_chat_openai(model, messages, temperature=0.1,
                                                  max_tokens=512, timeout=180)
                openai_prompt_tokens = openai_resp.get("usage", {}).get("prompt_tokens", "?")
                openai_content = openai_resp.get("choices", [{}])[0].get("message", {}).get("content", "")
            except Exception as e:
                print(f"  OpenAI API error: {e}")
                continue

            print(f"\n  Native /api/chat (num_ctx=8192):")
            print(f"    Prompt tokens: {native_prompt_tokens}")
            print(f"    Response length: {len(native_content)} chars")

            print(f"\n  OpenAI /v1/chat/completions (no num_ctx):")
            print(f"    Prompt tokens: {openai_prompt_tokens}")
            print(f"    Response length: {len(openai_content)} chars")

            if native_prompt_tokens != "?" and openai_prompt_tokens != "?":
                if native_prompt_tokens != openai_prompt_tokens:
                    print(f"\n  *** GAP DETECTED: prompt token counts differ ***")
                    print(f"  Ollama is using different context windows for the same prompt.")
                else:
                    print(f"\n  Prompt token counts match — no obvious context gap.")

            # Parse both
            native_ok, _, _ = _parse_with_pysubtrans(native_content)
            openai_ok, _, _ = _parse_with_pysubtrans(openai_content)
            print(f"\n  Format compliance — native: {'PASS' if native_ok else 'FAIL'}, "
                  f"openai: {'PASS' if openai_ok else 'FAIL'}")

            if native_ok and not openai_ok:
                print(f"  *** Native works but OpenAI path fails — the endpoint is the problem ***")
            elif not native_ok and not openai_ok:
                print(f"  Both fail — likely a model capability issue, not an endpoint issue.")

            print()


# ---------------------------------------------------------------------------
# TEST 4: Realistic Batch (6 lines, closer to production)
# ---------------------------------------------------------------------------

class TestRealisticBatch:
    """Test with a 6-line batch closer to production batch_size=11."""

    def test_6_line_batch(self, ollama_server):
        """Send a 6-line batch and validate format compliance."""
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        instructions = _load_instructions()

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: 6-line realistic batch")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            messages = _build_messages(REALISTIC_BATCH_6_LINES, instructions)

            t0 = time.time()
            try:
                resp = _ollama_chat(model, messages, temperature=0.3,
                                    max_tokens=2048, num_ctx=8192, timeout=300)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue
            elapsed = time.time() - t0

            content = resp.get("message", {}).get("content", "")
            eval_count = resp.get("eval_count", 0)

            print(f"\n  TIMING: {elapsed:.1f}s | {eval_count} tokens")
            print(f"\n  RAW RESPONSE ({len(content)} chars):")
            print(f"  {'-'*60}")
            print(textwrap.indent(content, "  "))
            print(f"  {'-'*60}")

            success, count, detail = _parse_with_pysubtrans(content)
            print(f"\n  PYSUBTRANS PARSE: {'PASS' if success else 'FAIL'}")
            print(f"  Expected 6 lines, got {count}")
            print(f"  {detail}")

            if success and count < 6:
                print(f"\n  *** PARTIAL: only {count}/6 lines parsed — model dropped lines ***")

            print()


# ---------------------------------------------------------------------------
# TEST 5: All Curated Models Summary Report
# ---------------------------------------------------------------------------

class TestCuratedModelReport:
    """Generate a compatibility report for all curated models."""

    def test_curated_compatibility_report(self, ollama_server):
        """Check metadata + quick format test for every curated model.

        Produces a summary table showing which models are likely to work.
        """
        curated = _load_curated_models()
        if not curated:
            pytest.skip("No curated models config found")

        local_models = _get_local_models()
        instructions = _load_instructions()

        print(f"\n{'='*70}")
        print(f"CURATED MODEL COMPATIBILITY REPORT")
        print(f"{'='*70}")
        print(f"Local models: {len(local_models)}")
        print(f"Curated models: {len(curated)}")

        results = []

        for entry in curated:
            model = entry["model"]
            label = entry.get("label", model)
            rank = entry.get("rank", "?")

            # Check if pulled
            is_local = any(model in lm or lm in model for lm in local_models)

            if not is_local:
                results.append({
                    "rank": rank, "label": label, "pulled": False,
                    "template": "?", "ctx": "?", "format_ok": "?"
                })
                continue

            # Get metadata
            try:
                info = _ollama_post("/api/show", {"name": model}, timeout=10)
            except Exception:
                results.append({
                    "rank": rank, "label": label, "pulled": True,
                    "template": "ERR", "ctx": "ERR", "format_ok": "ERR"
                })
                continue

            template = info.get("template", "")
            has_template = "YES" if template else "NO"

            model_info = info.get("model_info", {})
            ctx = None
            for key in ["context_length", "llama.context_length", "general.context_length"]:
                val = model_info.get(key)
                if val:
                    ctx = val
                    break

            # Quick format test (3-line batch)
            format_ok = "?"
            try:
                messages = _build_messages(MINI_BATCH_3_LINES, instructions)
                resp = _ollama_chat(model, messages, temperature=0.1,
                                    max_tokens=1024, num_ctx=8192, timeout=180)
                content = resp.get("message", {}).get("content", "")
                ok, count, _ = _parse_with_pysubtrans(content)
                format_ok = f"PASS({count})" if ok else "FAIL"
            except Exception as e:
                format_ok = f"ERR:{str(e)[:30]}"

            results.append({
                "rank": rank, "label": label, "pulled": True,
                "template": has_template, "ctx": ctx or "fallback(8192)",
                "format_ok": format_ok
            })

        # Print summary table
        print(f"\n  {'Rank':<5} {'Label':<40} {'Pulled':<8} {'Template':<10} {'Context':<15} {'Format'}")
        print(f"  {'-'*5} {'-'*40} {'-'*8} {'-'*10} {'-'*15} {'-'*10}")
        for r in results:
            pulled = "YES" if r["pulled"] else "NO"
            print(f"  {r['rank']:<5} {r['label']:<40} {pulled:<8} {r['template']:<10} "
                  f"{str(r['ctx']):<15} {r['format_ok']}")

        print()


# ---------------------------------------------------------------------------
# TEST 6: WhisperJAV Integration — OllamaManager Config vs Reality
# ---------------------------------------------------------------------------

class TestOllamaManagerConfig:
    """Verify OllamaManager's config resolution against actual model metadata."""

    def test_config_vs_reality(self, ollama_server):
        """Compare OllamaManager.ensure_ready() output against /api/show reality."""
        models = _get_test_models()
        if not models:
            pytest.skip("No models to test")

        try:
            from whisperjav.translate.ollama_manager import OllamaManager, OLLAMA_MODEL_CONFIGS
        except ImportError:
            pytest.skip("whisperjav not importable")

        mgr = OllamaManager()

        for model in models:
            print(f"\n{'='*70}")
            print(f"TEST: OllamaManager config vs Ollama reality")
            print(f"MODEL: {model}")
            print(f"{'='*70}")

            # What OllamaManager resolves
            in_config = model in OLLAMA_MODEL_CONFIGS
            print(f"\n  In OLLAMA_MODEL_CONFIGS: {in_config}")
            if in_config:
                cfg = OLLAMA_MODEL_CONFIGS[model]
                print(f"  Config: num_ctx={cfg['num_ctx']}, batch_size={cfg['batch_size']}, "
                      f"temp={cfg['temperature']}")

            # What ensure_ready returns
            try:
                readiness = mgr.ensure_ready(model=model, auto_start=False, interactive=False)
                print(f"\n  ensure_ready() returned:")
                print(f"    num_ctx={readiness['num_ctx']}")
                print(f"    batch_size={readiness['batch_size']}")
                print(f"    temperature={readiness['temperature']}")
            except Exception as e:
                print(f"\n  ensure_ready() failed: {e}")
                readiness = None

            # What Ollama actually reports
            actual_ctx = mgr.get_context_length(model)
            print(f"\n  Ollama /api/show context_length: {actual_ctx}")

            if readiness and actual_ctx != readiness["num_ctx"]:
                print(f"\n  *** MISMATCH: OllamaManager uses {readiness['num_ctx']} "
                      f"but model reports {actual_ctx} ***")

            # The critical gap: does num_ctx reach the API call?
            print(f"\n  CRITICAL: PySubtrans CustomClient._generate_request_body() does NOT")
            print(f"  include num_ctx in the request to /v1/chat/completions.")
            print(f"  Ollama will use the model's default context, not {readiness['num_ctx'] if readiness else '?'}.")

            print()
