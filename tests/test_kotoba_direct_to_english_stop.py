"""Kotoba-Whisper bilingual with Direct to English stops before any audio is read (1.10.3, owner W750 A3; wording
W764 2, texts sheet item 1). The model gives the English text without times (kotoba-tech/kotoba-whisper#5)."""
import argparse
import json

import pytest

from whisperjav.main import (KOTOBA_DIRECT_TO_ENGLISH_TEXT, transformers_models,
                             validate_direct_to_english_models)


def single(**kw):
    base = dict(ensemble=False, mode="transformers", pipeline=None, subs_language="direct-to-english",
                hf_model_id="kotoba-tech/kotoba-whisper-bilingual-v1.0")
    base.update(kw)
    return argparse.Namespace(**base)


def ensemble(**kw):
    base = dict(ensemble=True, mode="balanced", pipeline=None, subs_language="direct-to-english",
                pass1_pipeline="transformers", pass1_model=None, pass1_hf_params=None,
                pass2_pipeline=None, pass2_model=None, pass2_hf_params=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_the_default_transformers_model_with_direct_to_english_stops():
    with pytest.raises(ValueError) as stop:
        validate_direct_to_english_models(single())
    assert str(stop.value) == KOTOBA_DIRECT_TO_ENGLISH_TEXT
    assert KOTOBA_DIRECT_TO_ENGLISH_TEXT.startswith("Direct to English does not work with Kotoba-Whisper bilingual")


@pytest.mark.parametrize("args", [
    single(subs_language="native"),                                   # Japanese subtitles: the way that works
    single(hf_model_id="openai/whisper-large-v3"),                    # another model
    single(mode="qwen"),                                              # another pipeline
    single(pipeline="decoupled"),                                     # --pipeline decoupled runs instead of --mode
    ensemble(pass1_pipeline="qwen"),
    ensemble(pass1_model="openai/whisper-large-v3"),
])
def test_other_runs_go_on(args):
    validate_direct_to_english_models(args)


def test_an_ensemble_pass_takes_its_model_as_the_pass_worker_does():
    """--passN-model, else --passN-hf-params' model_id, else the default (Kotoba bilingual)."""
    assert transformers_models(ensemble()) == ["kotoba-tech/kotoba-whisper-bilingual-v1.0"]
    params = json.dumps({"model_id": "openai/whisper-large-v3"})
    assert transformers_models(ensemble(pass1_hf_params=params)) == ["openai/whisper-large-v3"]
    assert transformers_models(ensemble(pass1_hf_params=params, pass1_model="kotoba-tech/kotoba-whisper-bilingual-v1.0")) \
        == ["kotoba-tech/kotoba-whisper-bilingual-v1.0"]
    with pytest.raises(ValueError):
        validate_direct_to_english_models(ensemble(pass2_pipeline="transformers", pass1_pipeline="qwen"))
    validate_direct_to_english_models(ensemble(pass1_hf_params=params))
