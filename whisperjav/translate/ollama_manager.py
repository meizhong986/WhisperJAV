"""
Ollama Manager — detection, lifecycle, and model management for Ollama integration.

Central class for all Ollama interaction. Zero new pip dependencies (uses urllib.request).

Usage:
    from whisperjav.translate.ollama_manager import OllamaManager

    mgr = OllamaManager()
    readiness = mgr.ensure_ready(model="gemma3:12b", auto_start=True)
    # readiness = {model, num_ctx, batch_size, temperature, server_started, base_url}
"""

import atexit
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

# ============================================================================
# Exceptions
# ============================================================================

class OllamaError(Exception):
    """Base exception for Ollama-related errors."""
    pass


class OllamaNotInstalledError(OllamaError):
    """Ollama binary not found on the system."""
    pass


class OllamaNotRunningError(OllamaError):
    """Ollama is installed but the server is not running."""
    pass


class ModelNotAvailableError(OllamaError):
    """Requested model is not pulled locally."""
    pass


# ============================================================================
# Model configuration
# ============================================================================

@dataclass
class ModelRecommendation:
    """Hardware-aware model recommendation."""
    name: str           # e.g. "gemma3:12b"
    download_size: str  # e.g. "7.3 GB"
    quality: str        # "basic" | "good" | "very_good" | "excellent"
    note: str           # Human-readable explanation
    num_ctx: int        # Recommended context window
    batch_size: int     # Recommended batch size for this context
    temperature: float  # Recommended temperature


OLLAMA_MODEL_CONFIGS = {
    "qwen2.5:3b": {
        "num_ctx": 4096,
        "batch_size": 8,
        "temperature": 0.3,
        "download_size": "2.0 GB",
        "quality": "basic",
        "min_vram_gb": 0,
    },
    "gemma3:4b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.3,
        "download_size": "2.5 GB",
        "quality": "good",
        "min_vram_gb": 4,
    },
    "qwen2.5:7b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.5,
        "download_size": "4.7 GB",
        "quality": "good",
        "min_vram_gb": 8,
    },
    "gemma3:12b": {
        "num_ctx": 8192,
        "batch_size": 11,
        "temperature": 0.5,
        "download_size": "7.3 GB",
        "quality": "very_good",
        "min_vram_gb": 12,
    },
    "qwen2.5:14b": {
        "num_ctx": 16384,
        "batch_size": 20,
        "temperature": 0.5,
        "download_size": "9.0 GB",
        "quality": "excellent",
        "min_vram_gb": 16,
    },
}


# ============================================================================
# OllamaManager
# ============================================================================

class OllamaManager:
    """Central manager for Ollama server detection, lifecycle, and model management."""

    def __init__(self, base_url: str = None):
        """Initialize OllamaManager.

        Resolution order for base URL:
            1. Explicit base_url parameter
            2. OLLAMA_HOST environment variable
            3. http://localhost:11434 (default)
        """
        if base_url:
            self.base_url = base_url.rstrip('/')
        elif os.environ.get('OLLAMA_HOST'):
            self.base_url = os.environ['OLLAMA_HOST'].rstrip('/')
        else:
            self.base_url = 'http://localhost:11434'

        self._server_process = None

    # ── Server Detection & Lifecycle ──────────────────────────────────

    def detect_server(self) -> bool:
        """Check if Ollama server is running.

        Returns True if GET / responds with "Ollama is running".
        """
        try:
            self._http_get('/', timeout=2)
            # Ollama returns plain text "Ollama is running" at root
            return True
        except Exception:
            return False

    def detect_installation(self) -> Optional[str]:
        """Check if Ollama binary is installed.

        Returns path to the ollama binary, or None if not found.
        """
        return shutil.which("ollama")

    def start_server(self, timeout: int = 15) -> bool:
        """Start Ollama server via 'ollama serve' subprocess.

        Only starts if server is not already running. Registers atexit
        handler for cleanup. On Windows, uses CREATE_NEW_PROCESS_GROUP
        for proper subprocess management.

        Returns True if server started (or was already running).
        """
        if self.detect_server():
            return True

        ollama_path = self.detect_installation()
        if not ollama_path:
            return False

        # Start ollama serve as a subprocess
        kwargs = {
            'stdout': subprocess.DEVNULL,
            'stderr': subprocess.DEVNULL,
        }
        if platform.system() == 'Windows':
            kwargs['creationflags'] = subprocess.CREATE_NEW_PROCESS_GROUP

        try:
            self._server_process = subprocess.Popen(
                [ollama_path, 'serve'],
                **kwargs
            )
        except OSError as e:
            print(f"[OLLAMA] Failed to start server: {e}", file=sys.stderr)
            return False

        # Register cleanup
        atexit.register(self.stop_server)

        # Poll until server is ready
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.5)
            if self.detect_server():
                return True
            # Check if process died
            if self._server_process.poll() is not None:
                print("[OLLAMA] Server process exited unexpectedly", file=sys.stderr)
                self._server_process = None
                return False

        print(f"[OLLAMA] Server did not become ready within {timeout}s", file=sys.stderr)
        return False

    def stop_server(self):
        """Stop the server subprocess we started (if any).

        Does NOT stop a system-level Ollama service or tray app.
        """
        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except Exception:
                try:
                    self._server_process.kill()
                except Exception:
                    pass
            self._server_process = None

    # ── Model Management ──────────────────────────────────────────────

    def list_models(self) -> list:
        """List locally available models.

        Returns list of model dicts from GET /api/tags.
        """
        resp = self._http_get('/api/tags', timeout=5)
        return resp.get('models', [])

    def check_model(self, name: str) -> bool:
        """Check if a specific model is available locally.

        Returns True if model exists, False otherwise.
        """
        try:
            self._http_post('/api/show', {'name': name}, timeout=5)
            return True
        except Exception:
            return False

    def get_model_info(self, name: str) -> dict:
        """Get full model metadata via POST /api/show.

        Returns model info dict, or empty dict on failure.
        """
        try:
            return self._http_post('/api/show', {'name': name}, timeout=5)
        except Exception:
            return {}

    def get_context_length(self, name: str) -> int:
        """Extract context length from model metadata.

        Checks model_info dict and parameters string for context_length
        or num_ctx. Falls back to 8192 if not found (old Ollama versions).
        """
        info = self.get_model_info(name)
        if not info:
            return 8192

        # Check model_info dict (newer Ollama versions)
        model_info = info.get('model_info', {})
        if isinstance(model_info, dict):
            # Various key names Ollama uses across versions
            for key in ('context_length', 'num_ctx',
                        'llama.context_length',
                        'general.context_length'):
                val = model_info.get(key)
                if val and isinstance(val, int | float):
                    return int(val)

        # Check parameters string (some Ollama versions embed it here)
        params_str = info.get('parameters', '')
        if isinstance(params_str, str):
            for line in params_str.splitlines():
                line = line.strip()
                if line.startswith('num_ctx'):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            return int(parts[-1])
                        except ValueError:
                            pass

        return 8192  # Safe fallback

    def supports_system_messages(self, name: str) -> bool:
        """Check if a model's chat template handles system messages.

        Models pulled from hf.co/ as raw GGUFs often have a bare
        ``{{ .Prompt }}`` template that drops the system role entirely.
        When that happens, instructions sent as a system message are
        silently discarded and the LLM never sees them.

        Returns True if the template contains a reference to .System
        (meaning system messages are rendered), False otherwise.
        """
        info = self.get_model_info(name)
        if not info:
            return True  # Optimistic fallback if we can't check

        template = info.get('template', '')
        if not template:
            return False  # No template at all — cannot handle chat roles

        # Ollama Go templates use {{ .System }} to render system messages.
        # If the template doesn't reference it, system messages are dropped.
        return '.System' in template

    def pull_model(self, name: str, progress_callback: Callable = None) -> bool:
        """Pull (download) a model. Resumable via Ollama's built-in support.

        Args:
            name: Model name (e.g., "gemma3:12b")
            progress_callback: Optional callback(status, completed, total)

        Returns True on success.
        """
        import urllib.error
        import urllib.request

        url = f"{self.base_url}/api/pull"
        data = json.dumps({'name': name, 'stream': True}).encode('utf-8')
        req = urllib.request.Request(url, data=data, method='POST')
        req.add_header('Content-Type', 'application/json')

        try:
            resp = urllib.request.urlopen(req, timeout=600)
        except (urllib.error.URLError, OSError) as e:
            print(f"[OLLAMA] Failed to pull model: {e}", file=sys.stderr)
            return False

        # Read streaming NDJSON response
        try:
            for line in resp:
                line = line.decode('utf-8', errors='replace').strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue

                status = msg.get('status', '')
                completed = msg.get('completed', 0)
                total = msg.get('total', 0)

                if progress_callback:
                    progress_callback(status, completed, total)
                elif total > 0:
                    pct = (completed / total) * 100 if total else 0
                    print(f"\r[OLLAMA] {status}: {pct:.0f}%", end='', file=sys.stderr)

                if msg.get('error'):
                    print(f"\n[OLLAMA] Pull error: {msg['error']}", file=sys.stderr)
                    return False
        finally:
            resp.close()

        if not progress_callback:
            print(file=sys.stderr)  # Newline after progress
        return True

    def unload_model(self, name: str) -> bool:
        """Unload a model from VRAM/RAM immediately.

        Sends a generate request with keep_alive=0 which tells Ollama to
        evict the model from memory right away instead of waiting for the
        default idle timeout (typically 5 minutes).

        Args:
            name: Model name (e.g., "gemma3:4b")

        Returns True if the request succeeded, False otherwise.
        """
        try:
            self._http_post('/api/generate', {
                'model': name,
                'keep_alive': 0,
            }, timeout=10)
            return True
        except Exception as e:
            print(f"[OLLAMA] Failed to unload model: {e}", file=sys.stderr)
            return False

    # ── Hardware-Aware Recommendation ─────────────────────────────────

    def recommend_model(self, vram_gb: float = None) -> ModelRecommendation:
        """Recommend a model based on available VRAM.

        If vram_gb is not provided, attempts auto-detection via
        torch.cuda.mem_get_info() or nvidia-smi.
        """
        if vram_gb is None:
            vram_gb = self._detect_vram_gb() or 0

        # Walk configs from largest to smallest
        if vram_gb >= 16:
            pick = "qwen2.5:14b"
        elif vram_gb >= 12:
            pick = "gemma3:12b"
        elif vram_gb >= 8:
            pick = "qwen2.5:7b"
        elif vram_gb >= 4:
            pick = "gemma3:4b"
        else:
            pick = "qwen2.5:3b"

        cfg = OLLAMA_MODEL_CONFIGS[pick]
        note = f"Recommended for {'CPU/low VRAM' if vram_gb < 4 else f'{vram_gb:.0f} GB VRAM'}"

        return ModelRecommendation(
            name=pick,
            download_size=cfg['download_size'],
            quality=cfg['quality'],
            note=note,
            num_ctx=cfg['num_ctx'],
            batch_size=cfg['batch_size'],
            temperature=cfg['temperature'],
        )

    # ── Orchestration ─────────────────────────────────────────────────

    def ensure_ready(
        self,
        model: str = None,
        auto_start: bool = True,
        auto_pull: bool = False,
        interactive: bool = True,
    ) -> dict:
        """Full orchestration: detect server, start if needed, check model, pull if needed.

        Returns:
            dict with keys: model, num_ctx, batch_size, temperature, server_started, base_url

        Raises:
            OllamaNotInstalledError: Ollama not found on system
            OllamaNotRunningError: Server not running and couldn't start
            ModelNotAvailableError: Model not available and couldn't pull
        """
        server_started = False

        # Step 1: Server detection
        if not self.detect_server():
            if not self.detect_installation():
                install_msg = self._platform_install_guide()
                raise OllamaNotInstalledError(
                    f"Ollama is not installed.\n\n{install_msg}"
                )

            if auto_start:
                print("[OLLAMA] Server not running, starting...", file=sys.stderr)
                if self.start_server():
                    server_started = True
                    print("[OLLAMA] Server started successfully", file=sys.stderr)
                else:
                    raise OllamaNotRunningError(
                        "Failed to start Ollama server. Try running 'ollama serve' manually."
                    )
            elif interactive:
                answer = input("Ollama server is not running. Start it? [Y/n] ").strip().lower()
                if answer in ('', 'y', 'yes'):
                    if self.start_server():
                        server_started = True
                        print("[OLLAMA] Server started successfully", file=sys.stderr)
                    else:
                        raise OllamaNotRunningError(
                            "Failed to start Ollama server. Try running 'ollama serve' manually."
                        )
                else:
                    raise OllamaNotRunningError("Ollama server is not running.")
            else:
                raise OllamaNotRunningError(
                    "Ollama server is not running. Start it with: ollama serve"
                )

        # Step 2: Model selection
        if model is None:
            rec = self.recommend_model()
            model = rec.name
            print(f"[OLLAMA] Auto-selected model: {model} ({rec.quality}, {rec.download_size})",
                  file=sys.stderr)

        # Step 3: Normalize model name (add :latest if no tag)
        if ':' not in model:
            # Check if it matches a known config key with tag
            for cfg_key in OLLAMA_MODEL_CONFIGS:
                if cfg_key.startswith(model + ':'):
                    model = cfg_key
                    break
            else:
                model = model + ':latest'

        # Step 4: Check model availability
        if not self.check_model(model):
            cfg = OLLAMA_MODEL_CONFIGS.get(model, {})
            size_str = cfg.get('download_size', 'unknown size')

            if auto_pull:
                print(f"[OLLAMA] Model '{model}' not found locally, downloading ({size_str})...",
                      file=sys.stderr)
                if not self.pull_model(model):
                    raise ModelNotAvailableError(f"Failed to download model: {model}")
            elif interactive:
                answer = input(
                    f"Model '{model}' not found locally ({size_str}). Download? [Y/n] "
                ).strip().lower()
                if answer in ('', 'y', 'yes'):
                    if not self.pull_model(model):
                        raise ModelNotAvailableError(f"Failed to download model: {model}")
                else:
                    raise ModelNotAvailableError(
                        f"Model '{model}' not available. Pull it with: ollama pull {model}"
                    )
            else:
                raise ModelNotAvailableError(
                    f"Model '{model}' not available locally.\n"
                    f"Pull it with: ollama pull {model}"
                )

        # Step 5: Get context length and build config
        actual_ctx = self.get_context_length(model)

        # Use curated config if available, otherwise use dynamic values
        cfg = OLLAMA_MODEL_CONFIGS.get(model, {})
        num_ctx = cfg.get('num_ctx', min(actual_ctx, 8192))
        batch_size = cfg.get('batch_size', 11)
        temperature = cfg.get('temperature', 0.5)

        # Step 6: Check if model template handles system messages
        has_system = self.supports_system_messages(model)
        if not has_system:
            print(f"[OLLAMA] Model template does not handle system messages — "
                  f"instructions will be embedded in user message", file=sys.stderr)

        # Step 7: Return readiness info
        return {
            'model': model,
            'num_ctx': num_ctx,
            'batch_size': batch_size,
            'temperature': temperature,
            'server_started': server_started,
            'base_url': self.base_url,
            'supports_system_messages': has_system,
        }

    # ── Internal Helpers ──────────────────────────────────────────────

    def _detect_vram_gb(self) -> Optional[float]:
        """Detect available GPU VRAM in GB.

        Tries torch.cuda.mem_get_info() first, then nvidia-smi subprocess.
        Returns None if no GPU detected.
        """
        # Try torch first (often already imported)
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info(0)
                return total / (1024 ** 3)
        except Exception:
            pass

        # Try nvidia-smi subprocess
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # First GPU's total memory in MiB
                mem_mib = float(result.stdout.strip().splitlines()[0])
                return mem_mib / 1024
        except Exception:
            pass

        return None

    def _http_get(self, path: str, timeout: int = 5) -> dict:
        """HTTP GET request to Ollama API. Returns parsed JSON or empty dict."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"
        req = urllib.request.Request(url, method='GET')
        resp = urllib.request.urlopen(req, timeout=timeout)
        body = resp.read().decode('utf-8', errors='replace')
        resp.close()
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return {}

    def _http_post(self, path: str, data: dict, timeout: int = 10) -> dict:
        """HTTP POST request to Ollama API. Returns parsed JSON."""
        import urllib.error
        import urllib.request

        url = f"{self.base_url}{path}"
        body = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=body, method='POST')
        req.add_header('Content-Type', 'application/json')
        resp = urllib.request.urlopen(req, timeout=timeout)
        resp_body = resp.read().decode('utf-8', errors='replace')
        resp.close()
        try:
            return json.loads(resp_body)
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _platform_install_guide() -> str:
        """Return platform-specific Ollama installation instructions."""
        system = platform.system()
        if system == 'Windows':
            return (
                "Install Ollama for Windows:\n"
                "  1. Download from https://ollama.com/download/windows\n"
                "  2. Run the installer\n"
                "  3. Ollama will start automatically"
            )
        elif system == 'Darwin':
            return (
                "Install Ollama for macOS:\n"
                "  1. Download from https://ollama.com/download/mac\n"
                "  2. Or via Homebrew: brew install ollama\n"
                "  3. Start with: ollama serve"
            )
        else:
            return (
                "Install Ollama for Linux:\n"
                "  curl -fsSL https://ollama.com/install.sh | sh\n"
                "  Then start with: ollama serve"
            )
