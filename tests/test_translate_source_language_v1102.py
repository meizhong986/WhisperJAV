"""1.10.2: the translation source language reaches the translator.

Before 1.10.2, resolve_config() never mapped --source, so `merged.get('source', 'japanese')` in translate/cli.py was
always "japanese" and every request said "Translate these subtitles from Japanese", also for the GUI's choice (the GUI
sends --source). Found by the contributor of PR #508.
"""
import argparse

import pytest

from whisperjav.translate.settings import resolve_config


def _args(**kwargs):
    base = dict(source="japanese", target="english", provider="deepseek", model=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


@pytest.mark.parametrize("source", ["japanese", "korean", "chinese", "english"])
def test_cli_source_reaches_the_merged_settings(source):
    merged = resolve_config(_args(source=source), settings={})
    assert merged.get("source", "japanese") == source


def test_cli_source_wins_over_a_settings_file():
    merged = resolve_config(_args(source="korean"), settings={"source": "chinese"})
    assert merged["source"] == "korean"


def test_target_language_mapping_unchanged():
    merged = resolve_config(_args(target="chinese"), settings={})
    assert merged["target_language"] == "chinese"
