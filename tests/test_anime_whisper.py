"""
Unit tests for anime-whisper TextGenerator and TextCleaner integration.

Tests:
    - Protocol compliance (TextGenerator, TextCleaner)
    - Cleaner rules (missing 。, repetition removal, empty text, batch)
    - Factory registration (generator + cleaner discoverable and instantiable)
    - QwenPipeline backward compatibility (generator_backend defaults to "qwen3")
    - pass_worker.py integration (generator_backend in defaults + mapping)
"""

import inspect
import pytest


# ─── Cleaner Tests ────────────────────────────────────────────────────────────

class TestAnimeWhisperCleaner:
    """Tests for AnimeWhisperCleaner text cleaning rules."""

    @pytest.fixture
    def cleaner(self):
        from whisperjav.modules.subtitle_pipeline.cleaners.anime_whisper import (
            AnimeWhisperCleaner,
        )
        return AnimeWhisperCleaner()

    # -- Sentence-final 。 insertion --

    def test_adds_missing_period(self, cleaner):
        """Text without sentence-final punctuation gets 。 appended."""
        assert cleaner.clean("こんにちは") == "こんにちは。"

    def test_adds_period_to_kanji(self, cleaner):
        """Kanji-ending text gets 。."""
        assert cleaner.clean("今日は天気") == "今日は天気。"

    def test_preserves_existing_period(self, cleaner):
        """Text ending with 。 is not double-punctuated."""
        assert cleaner.clean("こんにちは。") == "こんにちは。"

    def test_preserves_exclamation(self, cleaner):
        """Half-width ! is preserved, no 。 added."""
        assert cleaner.clean("本当に!") == "本当に!"

    def test_preserves_question(self, cleaner):
        """Half-width ? is preserved, no 。 added."""
        assert cleaner.clean("何ですか?") == "何ですか?"

    def test_preserves_ellipsis(self, cleaner):
        """Ellipsis … is preserved, no 。 added."""
        assert cleaner.clean("あの…") == "あの…"

    def test_preserves_fullwidth_exclamation(self, cleaner):
        """Full-width ！ is preserved."""
        assert cleaner.clean("すごい！") == "すごい！"

    def test_preserves_fullwidth_question(self, cleaner):
        """Full-width ？ is preserved."""
        assert cleaner.clean("本当？") == "本当？"

    def test_preserves_music_note(self, cleaner):
        """♪ is sentence-final punctuation, no 。 added."""
        assert cleaner.clean("ラララ♪") == "ラララ♪"

    def test_preserves_comma(self, cleaner):
        """、is sentence-final punctuation (model artifact), no 。 added."""
        assert cleaner.clean("でも、") == "でも、"

    def test_preserves_tilde(self, cleaner):
        """～ is sentence-final punctuation, no 。 added."""
        assert cleaner.clean("やだ～") == "やだ～"

    # -- Repetition removal --

    def test_removes_triple_repetition(self, cleaner):
        """3+ consecutive repeats of same phrase are collapsed to 1."""
        # "ああああああ" = 6 × "あ" which matches pattern (.{2,20}?) with 3+ repeats
        # But single char "あ" is below 2-char minimum, so regex needs 2+ char phrases
        assert cleaner.clean("ですですですです") == "です。"  # 4× "です" → 1× "です" + 。

    def test_removes_long_phrase_repetition(self, cleaner):
        """Long repeated phrases are collapsed."""
        phrase = "そうですね"
        repeated = phrase * 4  # 4 consecutive repeats
        result = cleaner.clean(repeated)
        assert result == "そうですね。"  # Collapsed to single + 。

    def test_preserves_double_repetition(self, cleaner):
        """2 consecutive repeats (below threshold) are NOT collapsed."""
        assert cleaner.clean("ですです") == "ですです。"  # 2× "です" kept + 。

    def test_preserves_natural_text(self, cleaner):
        """Normal text without repetition is unchanged (plus 。)."""
        assert cleaner.clean("今日はいい天気ですね") == "今日はいい天気ですね。"

    # -- Empty / whitespace handling --

    def test_empty_returns_empty(self, cleaner):
        """Empty string input returns empty string."""
        assert cleaner.clean("") == ""

    def test_whitespace_returns_empty(self, cleaner):
        """Whitespace-only input returns empty string."""
        assert cleaner.clean("   ") == ""

    def test_none_returns_empty(self, cleaner):
        """None input returns empty string."""
        assert cleaner.clean(None) == ""

    # -- Batch processing --

    def test_batch_basic(self, cleaner):
        """clean_batch processes list correctly."""
        results = cleaner.clean_batch(["こんにちは", "本当に!", ""])
        assert results == ["こんにちは。", "本当に!", ""]

    def test_batch_empty_list(self, cleaner):
        """Empty batch returns empty list."""
        assert cleaner.clean_batch([]) == []

    def test_batch_all_empty(self, cleaner):
        """Batch of empty strings returns empty strings."""
        assert cleaner.clean_batch(["", "  ", ""]) == ["", "", ""]


# ─── Protocol Compliance ──────────────────────────────────────────────────────

class TestProtocolCompliance:
    """Verify anime-whisper classes satisfy TextGenerator / TextCleaner protocols."""

    def test_generator_is_text_generator(self):
        """AnimeWhisperGenerator satisfies TextGenerator protocol."""
        from whisperjav.modules.subtitle_pipeline.protocols import TextGenerator
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        assert isinstance(gen, TextGenerator)

    def test_cleaner_is_text_cleaner(self):
        """AnimeWhisperCleaner satisfies TextCleaner protocol."""
        from whisperjav.modules.subtitle_pipeline.protocols import TextCleaner
        from whisperjav.modules.subtitle_pipeline.cleaners.anime_whisper import (
            AnimeWhisperCleaner,
        )
        cleaner = AnimeWhisperCleaner()
        assert isinstance(cleaner, TextCleaner)

    def test_generator_has_required_methods(self):
        """AnimeWhisperGenerator has all protocol methods."""
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        assert hasattr(gen, "load")
        assert hasattr(gen, "unload")
        assert hasattr(gen, "generate")
        assert hasattr(gen, "generate_batch")
        assert hasattr(gen, "cleanup")
        assert callable(gen.load)
        assert callable(gen.unload)
        assert callable(gen.generate)
        assert callable(gen.generate_batch)
        assert callable(gen.cleanup)

    def test_cleaner_has_required_methods(self):
        """AnimeWhisperCleaner has all protocol methods."""
        from whisperjav.modules.subtitle_pipeline.cleaners.anime_whisper import (
            AnimeWhisperCleaner,
        )
        cleaner = AnimeWhisperCleaner()
        assert hasattr(cleaner, "clean")
        assert hasattr(cleaner, "clean_batch")
        assert callable(cleaner.clean)
        assert callable(cleaner.clean_batch)


# ─── Factory Registration ─────────────────────────────────────────────────────

class TestFactoryRegistration:
    """Verify anime-whisper is registered in both factories."""

    def test_generator_factory_lists_anime_whisper(self):
        """TextGeneratorFactory.available() includes 'anime-whisper'."""
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )
        assert "anime-whisper" in TextGeneratorFactory.available()

    def test_cleaner_factory_lists_anime_whisper(self):
        """TextCleanerFactory.available() includes 'anime-whisper'."""
        from whisperjav.modules.subtitle_pipeline.cleaners.factory import (
            TextCleanerFactory,
        )
        assert "anime-whisper" in TextCleanerFactory.available()

    def test_generator_factory_creates_instance(self):
        """TextGeneratorFactory.create('anime-whisper') returns valid instance."""
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )
        from whisperjav.modules.subtitle_pipeline.protocols import TextGenerator
        gen = TextGeneratorFactory.create("anime-whisper")
        assert isinstance(gen, TextGenerator)

    def test_cleaner_factory_creates_instance(self):
        """TextCleanerFactory.create('anime-whisper') returns valid instance."""
        from whisperjav.modules.subtitle_pipeline.cleaners.factory import (
            TextCleanerFactory,
        )
        from whisperjav.modules.subtitle_pipeline.protocols import TextCleaner
        cleaner = TextCleanerFactory.create("anime-whisper")
        assert isinstance(cleaner, TextCleaner)

    def test_generator_factory_custom_params(self):
        """Factory forwards constructor kwargs to AnimeWhisperGenerator."""
        from whisperjav.modules.subtitle_pipeline.generators.factory import (
            TextGeneratorFactory,
        )
        gen = TextGeneratorFactory.create(
            "anime-whisper",
            model_id="custom/model",
            max_new_tokens=128,
            no_repeat_ngram_size=7,
        )
        assert gen._config["model_id"] == "custom/model"
        assert gen._config["max_new_tokens"] == 128
        assert gen._config["no_repeat_ngram_size"] == 7


# ─── Generator Configuration ─────────────────────────────────────────────────

class TestGeneratorConfig:
    """Verify AnimeWhisperGenerator default configuration."""

    def test_default_model_id(self):
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        assert gen._config["model_id"] == "litagin/anime-whisper"

    def test_default_no_repeat_ngram(self):
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        assert gen._config["no_repeat_ngram_size"] == 5

    def test_default_max_new_tokens(self):
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        # 444 = max_target_positions (448) minus 4 special tokens.
        # Must be explicit because model's generation_config defaults to 4096.
        assert gen._config["max_new_tokens"] == 444

    def test_not_loaded_initially(self):
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        assert not gen.is_loaded

    def test_generate_without_load_raises(self):
        """Calling generate before load raises RuntimeError."""
        from pathlib import Path
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        with pytest.raises(RuntimeError, match="called before load"):
            gen.generate(Path("test.wav"))

    def test_generate_batch_without_load_raises(self):
        """Calling generate_batch before load raises RuntimeError."""
        from pathlib import Path
        from whisperjav.modules.subtitle_pipeline.generators.anime_whisper import (
            AnimeWhisperGenerator,
        )
        gen = AnimeWhisperGenerator()
        with pytest.raises(RuntimeError, match="called before load"):
            gen.generate_batch([Path("test.wav")])


# ─── Backward Compatibility ──────────────────────────────────────────────────

class TestBackwardCompatibility:
    """Verify QwenPipeline and pass_worker defaults are unchanged."""

    def test_qwen_pipeline_default_generator_is_qwen3(self):
        """QwenPipeline.__init__ defaults generator_backend to 'qwen3'."""
        from whisperjav.pipelines.qwen_pipeline import QwenPipeline
        sig = inspect.signature(QwenPipeline.__init__)
        default = sig.parameters["generator_backend"].default
        assert default == "qwen3"

    def test_pass_worker_default_generator_is_qwen3(self):
        """DEFAULT_QWEN_PARAMS has qwen_generator_backend = 'qwen3'."""
        from whisperjav.ensemble.pass_worker import DEFAULT_QWEN_PARAMS
        assert DEFAULT_QWEN_PARAMS["qwen_generator_backend"] == "qwen3"

    def test_pass_worker_mapping_includes_generator_backend(self):
        """prepare_qwen_params maps 'generator_backend' → 'qwen_generator_backend'."""
        from whisperjav.ensemble.pass_worker import prepare_qwen_params
        # Pass generator_backend via qwen_params
        result = prepare_qwen_params({
            "qwen_params": {"generator_backend": "anime-whisper"}
        })
        assert result["qwen_generator_backend"] == "anime-whisper"

    def test_pass_worker_mapping_default(self):
        """prepare_qwen_params with no overrides returns default qwen3."""
        from whisperjav.ensemble.pass_worker import prepare_qwen_params
        result = prepare_qwen_params({})
        assert result["qwen_generator_backend"] == "qwen3"

    def test_cli_arg_defined(self):
        """--qwen-generator argument is defined in main.py parse_arguments."""
        import whisperjav.main as main_module
        source = inspect.getsource(main_module.parse_arguments)
        assert "--qwen-generator" in source
        assert "anime-whisper" in source
