"""#356: with no --translate-model, the Ollama auto-pick went by VRAM only and chose a model that was never pulled;
the translation then failed while a usable model was on disk. Now an already-pulled known model is preferred.
The Ollama server is replaced by stand-in answers (no server, no download)."""
import pytest

from whisperjav.translate import ollama_manager as om


class _Mgr(om.OllamaManager):
    def __init__(self, pulled, vram):
        self.base_url = "http://stand-in"
        self._pulled = set(pulled)
        self._vram = vram
        self.pulls = []

    def detect_server(self):
        return True

    def _detect_vram_gb(self):
        return self._vram

    def list_models(self):
        return [{"name": n} for n in self._pulled]

    def check_model(self, name):
        return name in self._pulled

    def pull_model(self, name, progress_callback=None):
        self.pulls.append(name)
        return False

    def get_context_length(self, name):
        return 8192

    def supports_system_messages(self, name):
        return True

    def is_thinking_model(self, name):
        return False

    def _ensure_optimized_variant(self, model):
        return model


def test_the_pick_falls_back_to_a_pulled_model_that_fits():
    m = _Mgr(pulled={"qwen2.5:7b", "gemma3:4b", "llama3:70b"}, vram=12)   # 12 GB would pick gemma3:12b
    r = m.ensure_ready(model=None, auto_pull=False, interactive=False)
    assert r["model"] == "qwen2.5:7b" and m.pulls == []


def test_a_larger_pulled_model_is_not_taken_beyond_the_vram_pick():
    m = _Mgr(pulled={"qwen2.5:14b", "gemma3:4b"}, vram=8)                  # 8 GB picks qwen2.5:7b
    assert m.ensure_ready(model=None, auto_pull=False, interactive=False)["model"] == "gemma3:4b"


def test_nothing_pulled_keeps_the_old_behaviour():
    m = _Mgr(pulled=set(), vram=12)
    with pytest.raises(om.ModelNotAvailableError, match="gemma3:12b"):
        m.ensure_ready(model=None, auto_pull=False, interactive=False)


def test_an_explicit_model_is_never_swapped():
    m = _Mgr(pulled={"qwen2.5:7b"}, vram=12)
    with pytest.raises(om.ModelNotAvailableError, match="gemma3:12b"):
        m.ensure_ready(model="gemma3:12b", auto_pull=False, interactive=False)


def test_recommended_model_already_pulled_is_used():
    m = _Mgr(pulled={"gemma3:12b", "qwen2.5:7b"}, vram=12)
    assert m.ensure_ready(model=None, auto_pull=False, interactive=False)["model"] == "gemma3:12b"


def test_with_auto_pull_the_vram_pick_is_still_downloaded():
    """Review finding: the GUI always passes --yes; there the better model was downloaded before and still is."""
    m = _Mgr(pulled={"qwen2.5:3b"}, vram=16)
    with pytest.raises(om.ModelNotAvailableError):      # the stand-in download fails
        m.ensure_ready(model=None, auto_pull=True, interactive=False)
    assert m.pulls == ["qwen2.5:14b"]


def test_the_interactive_prompt_is_still_offered():
    m = _Mgr(pulled={"qwen2.5:7b"}, vram=12)
    import builtins
    asked = []
    orig = builtins.input
    builtins.input = lambda prompt="": asked.append(prompt) or "n"
    try:
        with pytest.raises(om.ModelNotAvailableError, match="gemma3:12b"):
            m.ensure_ready(model=None, auto_pull=False, interactive=True)
    finally:
        builtins.input = orig
    assert asked and "gemma3:12b" in asked[0]
