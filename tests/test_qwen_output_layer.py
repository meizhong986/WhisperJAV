"""R-QFT (#494; owner W491 2): a Qwen3-ASR model whose file stores no output layer gets it from its input embeddings;
one that cannot take the copy stops the run with a plain message.

The menu check reads the model files' headers from this computer's cache only (no model is loaded); a model not in
the cache is skipped. The GPU load and transcription check of the fine-tunes is a separate script with its output
(``D:\\WJ110_measure\\b_2026-11-03\\``).
"""
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from whisperjav.modules import qwen_asr
from whisperjav.modules.qwen_asr import INPUT_EMBEDDINGS, OUTPUT_LAYER, QwenASR, _checkpoint_tensor_names
from whisperjav.modules.qwen_errors import PREFIX as UNUSABLE_PREFIX, QwenModelUnusable

REPO = Path(__file__).resolve().parent.parent


def _fake_asr(model_id="someone/Qwen3-ASR-fine-tune", vocab=12, hidden=4, head_rows=None, tied=True):
    """A QwenASR whose ``model`` has the attribute chain of the fork's Qwen3ASRModel, with tiny layers."""
    asr = QwenASR.__new__(QwenASR)
    asr.model_id = model_id
    embed = torch.nn.Embedding(vocab, hidden)
    head = torch.nn.Linear(hidden, head_rows or vocab, bias=False)
    thinker = SimpleNamespace(lm_head=head, get_input_embeddings=lambda: embed,
                              config=SimpleNamespace(text_config=SimpleNamespace(tie_word_embeddings=tied)))
    asr.model = SimpleNamespace(model=SimpleNamespace(thinker=thinker))
    asr._standalone_aligner = None
    return asr, head, embed


def _names(monkeypatch, names):
    monkeypatch.setattr(qwen_asr, "_checkpoint_tensor_names", lambda model_id: names)


# ── the copy ──────────────────────────────────────────────────────────────────

def test_a_model_without_an_output_layer_gets_its_input_embeddings(monkeypatch):
    asr, head, embed = _fake_asr()
    _names(monkeypatch, {INPUT_EMBEDDINGS, "thinker.model.norm.weight"})
    asr._ensure_output_layer()
    assert torch.equal(head.weight, embed.weight)


def test_a_model_that_stores_its_output_layer_is_left_as_it_is(monkeypatch):
    asr, head, embed = _fake_asr()
    before = head.weight.detach().clone()
    _names(monkeypatch, {INPUT_EMBEDDINGS, OUTPUT_LAYER})
    asr._ensure_output_layer()
    assert torch.equal(head.weight, before) and not torch.equal(head.weight, embed.weight)


def test_when_the_files_cannot_be_read_the_settings_decide_as_in_1_9_5(monkeypatch):
    """A model in .bin files, or not in the cache: copied when its settings tie the two and they differ (review 1.4)."""
    _names(monkeypatch, None)
    asr, head, embed = _fake_asr()
    asr._ensure_output_layer()
    assert torch.equal(head.weight, embed.weight)
    asr, head, embed = _fake_asr(tied=False)
    before = head.weight.detach().clone()
    asr._ensure_output_layer()
    assert torch.equal(head.weight, before) and not torch.equal(head.weight, embed.weight)


def test_a_model_whose_output_layer_equals_its_embeddings_is_left_alone(monkeypatch):
    """The standard Qwen3-ASR models store an output layer byte for byte equal to their input embeddings."""
    _names(monkeypatch, None)
    asr, head, embed = _fake_asr()
    with torch.no_grad():
        head.weight.copy_(embed.weight)
    before = head.weight.data_ptr()
    asr._ensure_output_layer()
    assert torch.equal(head.weight, embed.weight) and head.weight.data_ptr() == before


# ── the stop (owner W491 2) ───────────────────────────────────────────────────

@pytest.mark.parametrize("names, kwargs, reason", [
    ({"thinker.model.norm.weight"}, {}, "its file has no input embeddings to make one from"),
    ({INPUT_EMBEDDINGS}, {"tied": False}, "its settings say the output layer is not the same as the input embeddings"),
    ({INPUT_EMBEDDINGS}, {"head_rows": 5}, "do not fit the output layer"),
])
def test_a_model_that_cannot_take_the_copy_stops_with_a_plain_message(monkeypatch, names, kwargs, reason):
    asr, _, _ = _fake_asr(**kwargs)
    _names(monkeypatch, names)
    with pytest.raises(QwenModelUnusable) as stop:
        asr._ensure_output_layer()
    text = str(stop.value)
    assert text.startswith("stopped before transcription: the ASR model someone/Qwen3-ASR-fine-tune cannot be used.")
    assert reason in text and "Choose another Qwen3-ASR model, for example Qwen/Qwen3-ASR-1.7B." in text
    assert asr.model is None, "the model is freed before the stop (review 1.5)"


def test_a_model_whose_layers_cannot_be_reached_is_named_as_a_fault_of_whisperjav(monkeypatch):
    asr, _, _ = _fake_asr()
    asr.model = SimpleNamespace(model=SimpleNamespace())
    _names(monkeypatch, {INPUT_EMBEDDINGS})
    with pytest.raises(QwenModelUnusable) as stop:
        asr._ensure_output_layer()
    text = str(stop.value)
    assert text.startswith(UNUSABLE_PREFIX + "someone/Qwen3-ASR-fine-tune could not be checked:")
    assert "This is a fault in WhisperJAV, not in the model; please report it." in text and asr.model is None


def test_both_load_paths_make_the_copy():
    src = (REPO / "whisperjav" / "modules" / "qwen_asr.py").read_text(encoding="utf-8")
    for start, end in (("def load_model(self)", "def transcribe("), ("def load_model_text_only(self)",
                                                                     "def load_aligner_only(self)")):
        body = src[src.index(start):src.index(end)]
        assert body.index("from_pretrained(") < body.index("self._ensure_output_layer()") < body.index(
            "self._apply_generation_config()")


# ── reading the model file's tensor names ─────────────────────────────────────

def test_the_names_come_from_one_safetensors_file(tmp_path):
    save_file({INPUT_EMBEDDINGS: torch.zeros(2, 2), "thinker.model.norm.weight": torch.zeros(2)},
              str(tmp_path / "model.safetensors"))
    assert _checkpoint_tensor_names(str(tmp_path)) == {INPUT_EMBEDDINGS, "thinker.model.norm.weight"}


def test_the_names_come_from_the_index_of_a_split_model(tmp_path):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {OUTPUT_LAYER: "model-00002-of-00002.safetensors",
                        INPUT_EMBEDDINGS: "model-00001-of-00002.safetensors"}}), encoding="utf-8")
    assert _checkpoint_tensor_names(str(tmp_path)) == {OUTPUT_LAYER, INPUT_EMBEDDINGS}


def test_a_folder_without_safetensors_gives_no_names(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"")
    assert _checkpoint_tensor_names(str(tmp_path)) is None


def test_a_model_id_not_in_the_cache_gives_no_names():
    assert _checkpoint_tensor_names("nobody-at-all/no-such-model-xyz") is None


# ── every Qwen3-ASR model in the menus ────────────────────────────────────────

def _menu_models():
    """The Qwen3-ASR model IDs offered in the GUI's two model menus (api.py Customize, app.js Ensemble)."""
    found = set()
    for rel in ("whisperjav/webview_gui/api.py", "whisperjav/webview_gui/assets/app.js"):
        text = (REPO / rel).read_text(encoding="utf-8")
        found.update(re.findall(r"""value["']?\s*:\s*["']([\w.-]+/Qwen3-ASR-[\w.-]+)["']""", text))
    return sorted(found)


def test_the_menus_offer_the_four_known_models():
    assert _menu_models() == ["Qwen/Qwen3-ASR-0.6B", "Qwen/Qwen3-ASR-1.7B",
                              "jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame", "neosophie/Qwen3-ASR-1.7B-JA"]


@pytest.mark.parametrize("model_id", _menu_models())
def test_each_menu_model_has_or_can_take_its_output_layer(model_id):
    names = _checkpoint_tensor_names(model_id)
    if names is None:
        pytest.skip(f"{model_id} is not in this computer's cache")
    if OUTPUT_LAYER in names:
        return
    from huggingface_hub import try_to_load_from_cache
    config = json.loads(Path(try_to_load_from_cache(model_id, "config.json")).read_text(encoding="utf-8"))
    assert INPUT_EMBEDDINGS in names
    assert config["thinker_config"]["text_config"].get("tie_word_embeddings", True) is True


# ── the run stops (single pass) and the pass stops (Ensemble) ─────────────────

def test_a_single_pass_run_stops_at_the_first_file():
    src = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
    body = src[src.index("def process_files_sync("):src.index("def process_files_async(")]
    stop = body[body.index("if isinstance(e, QwenModelUnusable):"):]
    assert "for rest in media_files[i:]:" in stop and stop.index("skipped_outcome(") < stop.index("break")
    assert "exc_info=not isinstance(e, (AudioIntegrityStop, QwenModelUnusable))" in body


def test_an_ensemble_pass_fails_every_file_with_the_plain_message_and_loads_once(tmp_path, monkeypatch):
    from whisperjav.ensemble import pass_worker

    message = "stopped before transcription: the ASR model x/Qwen3-ASR-y cannot be used. ..."
    calls = []

    class Pipeline:
        def process(self, media_info):
            calls.append(media_info["basename"])
            raise QwenModelUnusable(message)

        def cleanup(self):
            pass

    class Exited(Exception):
        pass

    def exit_with(result_file, result, tracer, exit_code):
        raise Exited(result)

    monkeypatch.setattr(pass_worker, "_build_pipeline", lambda **kwargs: Pipeline())
    monkeypatch.setattr(pass_worker, "_write_dropbox_and_exit", exit_with)
    monkeypatch.setattr(pass_worker, "setup_logger", lambda *a, **k: None)
    monkeypatch.setattr("whisperjav.utils.model_loader.patch_hf_hub_downloads", lambda: None)
    media = [{"basename": n, "path": str(tmp_path / f"{n}.mp4")} for n in ("a", "b", "c")]
    payload = pass_worker.WorkerPayload(
        pass_number=1, media_files=media, pass_config={"pipeline": "qwen"}, output_dir=str(tmp_path),
        temp_dir=str(tmp_path), keep_temp_files=False, subs_language="native", extra_kwargs={}, language_code="ja")
    with pytest.raises(Exited) as done:
        pass_worker.run_pass_worker(payload, str(tmp_path / "result.pkl"))
    results = done.value.args[0]["results"]
    assert calls == ["a"]
    assert [(r["basename"], r["status"], r["error"]) for r in results] == [
        ("a", "failed", message), ("b", "failed", message), ("c", "failed", message)]


def test_the_ensemble_summary_names_the_reason_for_either_pass():
    src = (REPO / "whisperjav" / "main.py").read_text(encoding="utf-8")
    assert 'if not _why and (_p1_error.startswith("stopped before transcription")' in src
    assert "or _p2_error.startswith(QWEN_UNUSABLE_PREFIX)):" in src
    # Its own prefix: the damaged-audio stop of pass 2 stays out of the summary as before (review 1.8).
    assert UNUSABLE_PREFIX.startswith("stopped before transcription: ") and UNUSABLE_PREFIX != "stopped before transcription: "


def test_a_single_pass_run_stops_and_the_later_files_are_not_started(tmp_path, monkeypatch):
    """The real file loop of a --mode qwen run (main.process_files_sync) with a ChronosJAV pipeline whose model
    cannot be used: the first file fails with the plain message, the two after it are not started (review 1.1)."""
    import types
    from whisperjav import main as wj_main

    message = UNUSABLE_PREFIX + "x/Qwen3-ASR-y cannot be used. ..."
    calls = []

    class Pipeline:
        def __init__(self, **kwargs):
            self.output_dir = kwargs.get("output_dir")

        def process(self, media_info):
            calls.append(media_info["basename"])
            raise QwenModelUnusable(message)

        def cleanup(self):
            pass

        def __getattr__(self, name):          # any other attribute the run sets or reads
            return None

    fake = types.ModuleType("whisperjav.pipelines.qwen_pipeline")
    fake.QwenPipeline = Pipeline
    monkeypatch.setitem(__import__("sys").modules, "whisperjav.pipelines.qwen_pipeline", fake)
    media = []
    for n in ("a", "b", "c"):
        (tmp_path / f"{n}.mp4").write_bytes(b"")
        media.append({"path": str(tmp_path / f"{n}.mp4"), "basename": n, "duration": 60.0})
    monkeypatch.setattr("sys.argv", ["whisperjav", *[m["path"] for m in media], "--mode", "qwen",
                                     "--output-dir", str(tmp_path / "out"), "--temp-dir", str(tmp_path / "temp"),
                                     "--no-progress"])
    args = wj_main.parse_arguments()
    outcomes = wj_main.process_files_sync(media, args, None, [])
    assert calls == ["a"]
    assert [(Path(o.path).stem, o.state) for o in outcomes] == [("a", "failed"), ("b", "skipped"), ("c", "skipped")]
    assert outcomes[0].detail == message
    assert outcomes[1].detail == "not started: the run stopped because its ASR model cannot be used"
