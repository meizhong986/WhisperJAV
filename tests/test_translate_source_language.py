"""Regression tests: AI SRT Translate must honor the user-selected source language.

`--source` must reach the merged config instead of falling back to `japanese`.
They import only the light `settings` and `instructions` modules, so no heavy extras are needed.
"""

import argparse

from whisperjav.translate.instructions import adapt_instructions_to_source_language
from whisperjav.translate.settings import DEFAULT_SETTINGS, resolve_config


def _make_args(**overrides):
    """Build an argparse.Namespace mirroring the fields cli.py feeds resolve_config.

    argparse gives `--source` a default of 'japanese' (never None), so the
    real Namespace always carries a source value.
    """
    base = {
        "source": "japanese",
        "target": "english",
        "provider": None,
        "model": None,
        "tone": None,
        "scene_threshold": None,
        "max_batch_size": None,
        "temperature": None,
        "top_p": None,
        "movie_title": None,
        "movie_plot": None,
        "actress": None,
        "ollama_url": None,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_source_korean_is_honored():
    cfg = resolve_config(_make_args(source="korean"), settings=None)
    assert cfg.get("source") == "korean"


def test_source_chinese_is_honored():
    cfg = resolve_config(_make_args(source="chinese"), settings=None)
    assert cfg.get("source") == "chinese"


def test_source_defaults_to_japanese():
    cfg = resolve_config(_make_args(source="japanese"), settings=None)
    assert cfg.get("source") == "japanese"


def test_merged_config_always_exposes_source_key():
    """cli.py reads merged.get('source', ...); the key should always be present."""
    cfg = resolve_config(_make_args(source="korean"), settings=None)
    assert "source" in cfg


def test_default_settings_includes_source():
    assert DEFAULT_SETTINGS.get("source") == "japanese"


def test_source_not_overridden_by_settings_file():
    """A settings file without 'source' must not clobber the CLI selection."""
    settings = {"provider": "deepseek", "target_language": "english"}
    cfg = resolve_config(_make_args(source="korean"), settings=settings)
    assert cfg.get("source") == "korean"


# ---------------------------------------------------------------------------
# Instruction-template language adaptation
# ---------------------------------------------------------------------------

_TEMPLATE = (
    "You are a professional translator specializing in Japanese translations. "
    "Your task is to translate the Japanese movie subtitles into the target language. "
    "Often in Japanese Adult Videos the tone becomes explicit.\n"
    "#200\nOriginal>\n変わりゆく時代において、\nTranslation>\n"
)


def test_adapt_japanese_is_noop():
    assert adapt_instructions_to_source_language(_TEMPLATE, "japanese") == _TEMPLATE


def test_adapt_korean_replaces_word_japanese():
    out = adapt_instructions_to_source_language(_TEMPLATE, "korean")
    assert "Japanese" not in out
    assert "specializing in Korean translations" in out
    assert "translate the Korean movie subtitles" in out
    assert "Korean Adult Videos" in out


def test_adapt_preserves_embedded_japanese_script():
    """Japanese example text must survive. Only the word 'Japanese' changes."""
    out = adapt_instructions_to_source_language(_TEMPLATE, "korean")
    assert "変わりゆく時代において" in out


def test_adapt_empty_or_missing_is_safe():
    assert adapt_instructions_to_source_language("", "korean") == ""
    assert adapt_instructions_to_source_language(_TEMPLATE, "") == _TEMPLATE
